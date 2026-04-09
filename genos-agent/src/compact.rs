//! Pre-compact verbatim archiver.
//!
//! When context_tokens_used > 80% of context_budget:
//! 1. Archive oldest ~500 tokens verbatim to palace overflow file
//! 2. Scan for L1-worthy facts (key = value patterns)
//! 3. Drop those tokens from active context
//!
//! MemPalace lesson: NEVER lose a word. Every byte goes to palace
//! before being dropped from context. No summarization at this stage —
//! that's Phase C's consolidation pass.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use genos_hal::disk;
use genos_tools::memory;
use genos_kernel::json::{json_object, JsonValue};

/// Tracks conversation history for pre-compact archiving.
pub struct ContextHistory {
    /// All turn text accumulated so far (prompt + response).
    turns: Vec<String>,
    /// Total approximate token count across all turns.
    total_tokens: usize,
    /// Number of overflow archives written this session.
    overflow_count: usize,
    /// Session ID for file naming.
    session_id: String,
}

impl ContextHistory {
    pub fn new(session_id: &str) -> Self {
        ContextHistory {
            turns: Vec::new(),
            total_tokens: 0,
            overflow_count: 0,
            session_id: String::from(session_id),
        }
    }

    /// Record a turn's text and approximate token count.
    pub fn record_turn(&mut self, text: &str, approx_tokens: usize) {
        self.turns.push(String::from(text));
        self.total_tokens += approx_tokens;
    }

    /// Get total approximate tokens in history.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Run pre-compact: archive oldest turns if over budget.
    /// Returns the number of tokens freed.
    pub fn pre_compact(&mut self, budget: usize) -> usize {
        // Only fire at 80% threshold
        if self.total_tokens * 5 <= budget * 4 {
            return 0;
        }

        // Target: archive enough to get back to ~60% usage
        let target = budget * 3 / 5;
        let tokens_to_archive = self.total_tokens.saturating_sub(target);
        let mut archived_text = String::new();
        let mut archived_tokens: usize = 0;

        while archived_tokens < tokens_to_archive && !self.turns.is_empty() {
            let turn = self.turns.remove(0);
            let turn_tokens = estimate_tokens(&turn);
            archived_text.push_str(&turn);
            archived_text.push('\n');
            archived_tokens += turn_tokens;
        }

        if archived_tokens == 0 {
            return 0;
        }

        // Write verbatim to palace overflow file
        self.overflow_count += 1;
        let path = format!(
            "\\palace\\sessions\\{}_overflow_{}.txt",
            self.session_id, self.overflow_count
        );
        let _ = disk::write_file(&path, archived_text.as_bytes());

        // Scan for L1-worthy facts (simple heuristic: lines with " = ")
        scan_for_facts(&archived_text);

        self.total_tokens -= archived_tokens;
        archived_tokens
    }
}

/// Estimate token count from text (rough: ~4 chars per token for English).
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

/// Scan archived text for potential L1 facts and store them.
fn scan_for_facts(text: &str) {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(eq_pos) = trimmed.find(" = ") {
            let key = trimmed[..eq_pos].trim();
            let value = trimmed[eq_pos + 3..].trim();
            // Only store if key looks like a fact identifier (alphanumeric + underscores)
            if !key.is_empty()
                && key.len() < 64
                && key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            {
                let args = json_object(&[
                    ("key", JsonValue::Str(String::from(key))),
                    ("value", JsonValue::Str(String::from(value))),
                ]);
                let _ = memory::tool_facts_set(&args, "auto_compact", "auto");
            }
        }
    }
}
