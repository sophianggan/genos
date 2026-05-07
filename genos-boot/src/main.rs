#![no_main]
#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use uefi::prelude::*;

use genos_agent::repl::Repl;
use genos_hal::{disk, screen};
use genos_kernel::config::ModelConfig;
use genos_kernel::gemma4::Gemma4Model;
use genos_kernel::gguf::GGUFFile;
use genos_kernel::inference::Transformer;
use genos_kernel::sampler::Sampler;
use genos_kernel::sysconfig::SystemConfig;
use genos_kernel::tokenizer::Tokenizer;

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();

    // Disable the UEFI watchdog timer — default is 5 minutes, which is
    // too short for loading a 2.8 GB model under QEMU TCG emulation.
    let _ = uefi::boot::set_watchdog_timer(0, 0x10000, None);

    screen::clear();
    screen::println("================================================");
    screen::println("  genos v0.1.0 — bare-metal LLM operating system");
    screen::println("================================================");
    screen::println("");

    // Load system config (optional)
    let sys_config = match disk::read_file("\\system\\config.toml") {
        Ok(data) => {
            screen::println("[boot] Loaded system config from \\system\\config.toml");
            SystemConfig::parse(&data)
        }
        Err(_) => {
            screen::println("[boot] No config.toml found, using defaults.");
            SystemConfig::default_config()
        }
    };

    // Read sampling params from config
    let temperature = sys_config.get_f32("sampling.temperature").unwrap_or(1.0);
    let top_p = sys_config.get_f32("sampling.top_p").unwrap_or(0.9);
    let top_k = sys_config.get_usize("sampling.top_k").unwrap_or(64);
    let rep_penalty = sys_config.get_f32("sampling.rep_penalty").unwrap_or(1.0);
    let min_p = sys_config.get_f32("sampling.min_p").unwrap_or(0.0);

    // --- Try Gemma 4 E2B GGUF first ---
    screen::println("[boot] Loading Gemma 4 GGUF from \\models\\gemma4-e2b.gguf ...");
    screen::println("[boot] (This takes several minutes under QEMU emulation)");
    match disk::read_file_paged_with_progress("\\models\\gemma4-e2b.gguf", |done, total| {
        let pct = (done as u64 * 100 / total as u64) as usize;
        let mb_done = done / (1024 * 1024);
        let mb_total = total / (1024 * 1024);
        // Print progress every ~32 MB (every 8 chunks of 4 MB)
        if mb_done % 32 == 0 || done == total {
            screen::print(&format!("\r[boot] Loading: {} / {} MB ({}%)  ", mb_done, mb_total, pct));
        }
    }) {
        Ok(gguf_data) => {
            screen::println("");
            screen::print(&format!("[boot] Loaded {} MB into memory.\n", gguf_data.len() / (1024 * 1024)));

            match GGUFFile::parse(gguf_data.as_slice()) {
            Ok(gguf) => {
                screen::println("[boot] GGUF parsed successfully.");
                screen::print(&format!("[boot] Tensors: {}, Metadata entries: {}\n",
                    gguf.tensors.len(), gguf.metadata.len()));

                // Load Gemma 4 tokenizer (HuggingFace BPE format)
                let tokenizer = match disk::read_file("\\models\\tokenizer.json") {
                    Ok(vocab) => {
                        screen::println("[boot] Loading Gemma 4 tokenizer (262k vocab)...");
                        let merges = disk::read_file("\\models\\merges.txt")
                            .unwrap_or_else(|_| alloc::vec::Vec::new());
                        Tokenizer::load_hf(&vocab, &merges)
                    }
                    Err(_) => {
                        screen::println("[boot] ERROR: Gemma 4 requires tokenizer.json");
                        screen::println("[boot] Falling back to Stories15M...");
                        boot_stories15m(temperature, top_p);
                        return Status::SUCCESS;
                    }
                };

                screen::println("[boot] Building Gemma 4 model from GGUF...");
                match Gemma4Model::from_gguf(&gguf) {
                    Ok(model) => {
                        let vocab_size = model.config.vocab_size;
                        let max_seq = model.config.max_position_embeddings;
                        screen::print(&format!(
                            "[boot] Gemma 4 E2B: {} layers, {} vocab, {}k context\n",
                            model.config.num_layers,
                            vocab_size,
                            max_seq / 1024,
                        ));
                        screen::println("[boot] Ready. (Gemma 4 E2B)");
                        screen::println("");

                        let sampler = Sampler::new_full(
                            vocab_size, temperature, top_p,
                            top_k, min_p, rep_penalty, 42,
                        );

                        // Run the full agent REPL with Gemma 4
                        let mut repl = Repl::new(
                            Box::new(model),
                            tokenizer,
                            sampler,
                            max_seq,
                        );
                        repl.run();
                        return Status::SUCCESS;
                    }
                    Err(_e) => {
                        screen::println("[boot] ERROR building Gemma 4 model.");
                        screen::println("[boot] Falling back to Stories15M...");
                    }
                }
            }
            Err(_e) => {
                screen::println("[boot] ERROR parsing GGUF.");
                screen::println("[boot] Falling back to Stories15M...");
            }
        }
        }
        Err(e) => {
            screen::print("[boot] Gemma 4 load failed: ");
            screen::println(e.as_str());
            screen::println("[boot] Falling back to Stories15M...");
        }
    }

    // --- Fallback: Stories15M ---
    boot_stories15m(temperature, top_p);

    Status::SUCCESS
}

/// Boot with the Stories15M model (Phase A/B/C path).
fn boot_stories15m(temperature: f32, top_p: f32) {
    screen::println("[boot] Loading model from \\models\\stories15m.bin ...");
    let model_data = match disk::read_file("\\models\\stories15m.bin") {
        Ok(data) => {
            screen::println("[boot] Model file loaded.");
            data
        }
        Err(e) => {
            screen::println("[boot] ERROR: Could not load model file.");
            screen::println("[boot] Make sure stories15m.bin is in \\models\\ on the EFI partition.");
            screen::print("[boot] Error: ");
            screen::println(e.as_str());
            screen::println("");
            screen::println("[boot] Entering shell in stub mode (no model).");
            enter_stub_repl();
            return;
        }
    };

    screen::println("[boot] Loading tokenizer from \\models\\tokenizer.bin ...");
    let tok_data = match disk::read_file("\\models\\tokenizer.bin") {
        Ok(data) => data,
        Err(e) => {
            screen::println("[boot] ERROR: Could not load tokenizer.");
            screen::print("[boot] Error: ");
            screen::println(e.as_str());
            return;
        }
    };

    screen::println("[boot] Parsing model weights...");
    let config = match ModelConfig::from_bytes(&model_data) {
        Some(c) => c,
        None => {
            screen::println("[boot] ERROR: Invalid model file format.");
            return;
        }
    };

    screen::print(&format!(
        "[boot] Model: dim={} layers={} heads={} vocab={} seq_len={}\n",
        config.dim, config.n_layers, config.n_heads, config.vocab_size, config.seq_len
    ));

    screen::println("[boot] Building transformer (this takes ~30s in emulation)...");
    let transformer = Transformer::new(config, &model_data[28..]);

    screen::println("[boot] Parsing tokenizer...");
    let tokenizer = Tokenizer::load(&tok_data, config.vocab_size);
    screen::println("[boot] Tokenizer loaded.");

    let sampler = Sampler::new(config.vocab_size, temperature, top_p, 42);

    screen::println("[boot] Ready. (Stories15M)");
    screen::println("");

    let mut repl = Repl::new(Box::new(transformer), tokenizer, sampler, config.seq_len);
    repl.run();
}

fn enter_stub_repl() {
    use genos_hal::keyboard;

    screen::println("Type 'exit' to shut down.");
    screen::println("");
    loop {
        screen::print("genos> ");
        let input = keyboard::read_line();
        let trimmed = input.trim();
        if trimmed == "exit" || trimmed == "quit" {
            screen::println("Shutting down...");
            break;
        }
        screen::println("[stub] No model loaded. Cannot generate.");
    }
}
