#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std as alloc;

pub mod config;
pub mod gemma4;
pub mod gguf;
pub mod inference;
pub mod json;
pub mod kv_cache;
pub mod sampler;
pub mod simd;
pub mod sysconfig;
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
