#![no_main]
#![no_std]

extern crate alloc;

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
    if let Ok(gguf_data) = disk::read_file("\\models\\gemma4-e2b.gguf") {
        screen::println("[boot] Found Gemma 4 E2B GGUF model.");
        screen::print(&format!("[boot] GGUF file size: {} MB\n", gguf_data.len() / (1024 * 1024)));

        match GGUFFile::parse(&gguf_data) {
            Ok(gguf) => {
                screen::println("[boot] GGUF parsed successfully.");
                screen::print(&format!("[boot] Tensors: {}, Metadata entries: {}\n",
                    gguf.tensors.len(), gguf.metadata.len()));

                // Load Gemma 4 tokenizer (HuggingFace BPE format)
                let tokenizer = match disk::read_file("\\models\\tokenizer.json") {
                    Ok(vocab) => {
                        screen::println("[boot] Loading Gemma 4 tokenizer (262k vocab)...");
                        // merges.txt is optional — Gemma 4's tokenizer.json
                        // embeds merges in the HuggingFace format.
                        let merges = disk::read_file("\\models\\merges.txt")
                            .unwrap_or_else(|_| alloc::vec::Vec::new());
                        Tokenizer::load_hf(&vocab, &merges)
                    }
                    Err(_) => {
                        screen::println("[boot] ERROR: Gemma 4 requires tokenizer.json");
                        screen::println("[boot] Falling back to Stories15M...");
                        boot_stories15m(&sys_config, temperature, top_p);
                        return Status::SUCCESS;
                    }
                };

                screen::println("[boot] Building Gemma 4 model from GGUF...");
                match Gemma4Model::from_gguf(&gguf) {
                    Ok(mut model) => {
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

                        // Gemma 4 uses its own generation loop via LLMRuntime
                        let sampler = Sampler::new_full(
                            vocab_size, temperature, top_p,
                            top_k, min_p, rep_penalty, 42,
                        );

                        // Simple generation loop for Gemma 4
                        gemma4_repl(&mut model, &tokenizer, sampler, max_seq);
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

    // --- Fallback: Stories15M ---
    boot_stories15m(&sys_config, temperature, top_p);

    Status::SUCCESS
}

/// Boot with the Stories15M model (Phase A/B/C path).
fn boot_stories15m(_sys_config: &SystemConfig, temperature: f32, top_p: f32) {
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

    screen::print("[boot] Model: dim=");
    print_usize(config.dim);
    screen::print(" layers=");
    print_usize(config.n_layers);
    screen::print(" heads=");
    print_usize(config.n_heads);
    screen::print(" vocab=");
    print_usize(config.vocab_size);
    screen::print(" seq_len=");
    print_usize(config.seq_len);
    screen::println("");

    screen::println("[boot] Building transformer (this takes ~30s in emulation)...");
    let transformer = Transformer::new(config, &model_data[28..]);

    screen::println("[boot] Parsing tokenizer...");
    let tokenizer = Tokenizer::load(&tok_data, config.vocab_size);
    screen::println("[boot] Tokenizer loaded.");

    let sampler = Sampler::new(config.vocab_size, temperature, top_p, 42);

    screen::println("[boot] Ready.");
    screen::println("");

    let mut repl = Repl::new(transformer, tokenizer, sampler, config.seq_len);
    repl.run();
}

/// Minimal REPL for Gemma 4 using LLMRuntime trait.
fn gemma4_repl(
    model: &mut Gemma4Model,
    tokenizer: &Tokenizer,
    mut sampler: Sampler,
    max_seq_len: usize,
) {
    use alloc::string::String;
    use alloc::vec::Vec;
    use genos_hal::keyboard;

    screen::println("genos v0.1.0 — bare-metal LLM operating system (Gemma 4 E2B)");
    screen::println("Type a prompt and press Enter. Type 'exit' to shut down.");
    screen::println("");

    loop {
        screen::print("genos> ");
        let input = keyboard::read_line();
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "quit" {
            screen::println("Shutting down genos...");
            break;
        }

        // Tokenize input
        let prompt = format!("User: {}\nAssistant:", trimmed);
        let tokens = tokenizer.encode(&prompt, true, false);
        if tokens.is_empty() {
            screen::println("[error] Could not tokenize input.");
            continue;
        }

        let num_prompt = tokens.len();
        let mut token: u32 = tokens[0];
        let mut pos = 0usize;
        let mut output = String::new();

        if num_prompt > 1 {
            screen::print(&format!("[processing {} tokens] ", num_prompt));
        }

        while pos < max_seq_len.min(num_prompt + 512) {
            let logits = model.forward_pass(token, pos);

            let next_token = if pos < num_prompt - 1 {
                if pos > 0 && pos % 10 == 0 {
                    screen::print(".");
                }
                tokens[pos + 1]
            } else {
                let mut logits_buf: Vec<f32> = logits.to_vec();
                sampler.sample(&mut logits_buf)
            };

            pos += 1;

            if pos >= num_prompt {
                if pos == num_prompt {
                    screen::println("");
                }
                let piece = tokenizer.decode(token, next_token);
                screen::print(piece);
                output.push_str(piece);
            }

            // EOS tokens: 1 (BOS), 2 (EOS) for llama-style, or model-specific
            if next_token == 1 || next_token == 2 {
                break;
            }

            token = next_token;
        }

        screen::println("");
        screen::println("");
    }
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

fn print_usize(n: usize) {
    let s = format!("{}", n);
    screen::print(&s);
}
