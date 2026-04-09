use alloc::vec::Vec;
use genos_hal::{keyboard, screen};
use genos_kernel::inference::Transformer;
use genos_kernel::sampler::Sampler;
use genos_kernel::tokenizer::Tokenizer;

/// The main REPL (Read-Eval-Print Loop) for genos.
/// Takes user input, tokenizes it, runs the transformer, and streams output.
pub struct Repl {
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
    max_seq_len: usize,
}

impl Repl {
    pub fn new(
        transformer: Transformer,
        tokenizer: Tokenizer,
        sampler: Sampler,
        max_seq_len: usize,
    ) -> Self {
        Repl {
            transformer,
            tokenizer,
            sampler,
            max_seq_len,
        }
    }

    /// Run the REPL loop. Blocks forever until user types "exit".
    pub fn run(&mut self) {
        screen::println("Type a prompt and press Enter. Type 'exit' to shut down.");
        screen::println("This is a story-completion model (Stories15M).");
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

            if trimmed == "help" {
                self.print_help();
                continue;
            }

            // Reset transformer state for a fresh generation
            self.transformer.reset();

            // Tokenize the input
            let tokens = self.tokenizer.encode(trimmed, true, false);

            if tokens.is_empty() {
                screen::println("[error] Could not tokenize input.");
                continue;
            }

            // Run generation
            self.generate(&tokens);

            screen::println("");
            screen::println("");
        }
    }

    /// Generate tokens given a prompt token sequence.
    fn generate(&mut self, prompt_tokens: &[u32]) {
        let num_prompt_tokens = prompt_tokens.len();
        let mut token: u32 = prompt_tokens[0];
        let mut pos = 0usize;
        let mut next_token: u32;

        while pos < self.max_seq_len {
            // Forward pass
            let logits = self.transformer.forward(token, pos);

            if pos < num_prompt_tokens - 1 {
                // Still processing prompt tokens — advance to next prompt token
                next_token = prompt_tokens[pos + 1];
            } else {
                // Generating: sample from logits
                let mut logits_buf: Vec<f32> = logits.to_vec();
                next_token = self.sampler.sample(&mut logits_buf);
            }

            pos += 1;

            // Print as soon as we've consumed the full prompt.
            // pos > num_prompt_tokens - 1 means we've processed all prompt tokens
            // and next_token is a freshly sampled (generated) token.
            if pos >= num_prompt_tokens {
                let piece = self.tokenizer.decode(token, next_token);
                screen::print(piece);
            }

            // Stop on EOS token (token 2) or if we hit a special token
            if next_token == 2u32 {
                break;
            }

            token = next_token;
        }
    }

    fn print_help(&self) {
        screen::println("=== genos help ===");
        screen::println("  Type any text to generate a story continuation.");
        screen::println("  Commands:");
        screen::println("    help  - Show this help message");
        screen::println("    exit  - Shut down genos");
        screen::println("    quit  - Shut down genos");
        screen::println("");
    }
}
