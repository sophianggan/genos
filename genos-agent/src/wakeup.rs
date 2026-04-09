//! OS wake-up protocol — builds the runtime system prompt from palace state.
//!
//! Inspired by MemPalace's wake-up: load 170 tokens of world knowledge
//! before the first turn. This is NOT a static file — it is built at
//! runtime from palace state every boot.
//!
//! Wake-up sequence:
//! 1. Read identity.txt (L0, ~50 tokens)
//! 2. Read facts.kv (L1, ~120 tokens)
//! 3. Generate tool manifest from registry
//! 4. Assemble: [L0] + [L1] + [tool manifest]

use alloc::string::String;
use alloc::format;
use genos_tools::palace;
use genos_tools::protocol::ToolRegistry;
use genos_tools::sys::format_status_header;

/// The assembled system prompt, ready for injection into every turn.
pub struct WakeUpState {
    /// The full system prompt (identity + facts + tool manifest).
    pub system_prompt: String,
    /// Current session ID.
    pub session_id: String,
    /// Turn counter (starts at 0, incremented after each turn).
    pub turn: usize,
    /// Context budget in tokens (from config, default 256 for Stories15M).
    pub context_budget: usize,
    /// Approximate tokens used so far.
    pub context_used: usize,
}

impl WakeUpState {
    /// Build the status header for the current turn.
    pub fn status_header(&self) -> String {
        format_status_header(
            &self.session_id,
            self.turn,
            self.context_used,
            self.context_budget,
        )
    }

    /// Build the full prompt prefix: system_prompt + status header.
    pub fn prompt_prefix(&self) -> String {
        let header = self.status_header();
        format!("{}\n{}\n", self.system_prompt, header)
    }

    /// Advance to next turn and update token usage estimate.
    pub fn advance_turn(&mut self, tokens_this_turn: usize) {
        self.turn += 1;
        self.context_used += tokens_this_turn;
    }

    /// Check if context budget is above the pre-compact threshold (80%).
    pub fn needs_pre_compact(&self) -> bool {
        self.context_used * 5 > self.context_budget * 4 // > 80%
    }
}

/// Load the wake-up state: read palace, build system prompt, return WakeUpState.
///
/// Call this once at boot after `palace::ensure_structure()`.
pub fn load(session_id: &str) -> WakeUpState {
    // Step 1: Read identity (L0)
    let identity = palace::read_identity();

    // Step 2: Read facts (L1) — format as key-value lines
    let facts_vec = palace::read_facts();
    let facts = if facts_vec.is_empty() {
        String::from("No facts loaded yet.")
    } else {
        let mut s = String::new();
        for (k, v) in &facts_vec {
            s.push_str(k);
            s.push_str(" = ");
            s.push_str(v);
            s.push('\n');
        }
        s
    };

    // Step 3: Generate tool manifest
    let registry = ToolRegistry::default_registry();
    let manifest = registry.manifest();

    // Step 4: Assemble system prompt
    let system_prompt = format!(
        "{}\n\nKnown facts:\n{}\n\n{}\n\
         When you want to use a tool, emit the JSON on its own line.\n\
         After receiving the result, continue your response.\n\
         Always respond helpfully and concisely.",
        identity, facts, manifest
    );

    WakeUpState {
        system_prompt,
        session_id: String::from(session_id),
        turn: 0,
        context_budget: 256, // Stories15M max_seq_len; overridden by config if present
        context_used: 0,
    }
}

/// Load wake-up state with a custom context budget (from config.toml).
pub fn load_with_budget(session_id: &str, context_budget: usize) -> WakeUpState {
    let mut state = load(session_id);
    state.context_budget = context_budget;
    state
}
