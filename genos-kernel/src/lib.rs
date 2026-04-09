#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod config;
pub mod inference;
pub mod sampler;
pub mod tokenizer;
pub mod weights;

/// Common interface for LLM inference backends.
/// Implemented by Stories15M (Phase A) and Gemma 4 E2B (Phase D),
/// letting genos-agent stay unchanged across model upgrades.
pub trait LLMRuntime {
    /// Run one forward pass for the given token at the given sequence position.
    /// Returns logits over the full vocabulary.
    fn forward(&mut self, token: u32, pos: usize) -> &[f32];
    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
    /// Maximum supported sequence length.
    fn max_seq_len(&self) -> usize;
}
