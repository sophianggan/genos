#![no_main]
#![no_std]

extern crate alloc;

use uefi::prelude::*;

use genos_agent::repl::Repl;
use genos_hal::{disk, screen};
use genos_kernel::config::ModelConfig;
use genos_kernel::inference::Transformer;
use genos_kernel::sampler::Sampler;
use genos_kernel::tokenizer::Tokenizer;

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();

    screen::clear();
    screen::println("================================================");
    screen::println("  genos v0.1.0 — bare-metal LLM operating system");
    screen::println("================================================");
    screen::println("");

    // Load model weights
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
            return Status::SUCCESS;
        }
    };

    // Load tokenizer
    screen::println("[boot] Loading tokenizer from \\models\\tokenizer.bin ...");
    let tok_data = match disk::read_file("\\models\\tokenizer.bin") {
        Ok(data) => data,
        Err(e) => {
            screen::println("[boot] ERROR: Could not load tokenizer.");
            screen::print("[boot] Error: ");
            screen::println(e.as_str());
            return Status::SUCCESS;
        }
    };

    // Parse model
    screen::println("[boot] Parsing model weights...");
    let config = match ModelConfig::from_bytes(&model_data) {
        Some(c) => c,
        None => {
            screen::println("[boot] ERROR: Invalid model file format.");
            return Status::SUCCESS;
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

    let transformer = Transformer::new(config, &model_data[28..]);

    // Parse tokenizer
    screen::println("[boot] Parsing tokenizer...");
    let tokenizer = Tokenizer::load(&tok_data, config.vocab_size);
    screen::println("[boot] Tokenizer loaded.");

    // Create sampler
    let sampler = Sampler::new(config.vocab_size, 1.0, 0.9, 42);

    screen::println("[boot] Ready.");
    screen::println("");

    // Enter REPL
    let mut repl = Repl::new(transformer, tokenizer, sampler, config.seq_len);
    repl.run();

    Status::SUCCESS
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
    use alloc::format;
    let s = format!("{}", n);
    screen::print(&s);
}
