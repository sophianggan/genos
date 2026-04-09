use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use genos_hal::{keyboard, screen, timer};
use genos_kernel::inference::Transformer;
use genos_kernel::json;
use genos_kernel::sampler::Sampler;
use genos_kernel::tokenizer::Tokenizer;
use genos_tools::journal;
use genos_tools::palace;
use genos_tools::protocol::ToolCall;
use genos_tools::graph::EntityGraph;
use genos_tools::search::SearchIndex;
use crate::compact::ContextHistory;
use crate::consolidate::Consolidator;
use crate::policy::PolicyEngine;
use crate::tools;
use crate::wakeup;

/// The main agent loop for genos.
///
/// Implements the full 11-step per-turn sequence from MASTER.md:
/// 1. Read user input
/// 2. Prepend wake-up header
/// 3. Build full prompt
/// 4. Call LLM kernel, stream tokens
/// 5. Scan output for tool calls
/// 6. Execute tools, append results
/// 7. Continue until EOS
/// 8. Classify output hall type
/// 9. Write verbatim to palace
/// 10. Run pre_compact if needed
/// 11. Update turn counter
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

    /// Boot and run the agent loop.
    ///
    /// Boot sequence (before first turn):
    /// 1. palace::ensure_structure() — create dirs/files
    /// 2. wakeup::load() — build system prompt from palace
    /// 3. Initialize session, log start
    pub fn run(&mut self) {
        // --- Boot sequence ---

        // Step 1: Ensure palace structure exists
        let time_suffix = timer::get_wall_time()
            .map(|wt| wt.session_id_suffix())
            .unwrap_or_else(|| String::from("unknown"));
        let session_id = palace::ensure_structure(&time_suffix);

        // Step 2: Load wake-up state (system prompt from palace)
        let mut state = wakeup::load(&session_id);
        screen::print_status(&format!(
            "Session {} | Phase B | Context budget: {} tokens",
            session_id, state.context_budget
        ));

        // Step 3: Initialize context history and policy
        let mut history = ContextHistory::new(&session_id);
        let mut policy = PolicyEngine::new();
        let mut search_index = SearchIndex::new();
        let mut entity_graph = EntityGraph::new();
        entity_graph.load();
        let mut consolidator = Consolidator::new();
        let mut input_history = keyboard::InputHistory::new(10);

        screen::println("");
        screen::println("genos v0.1.0 — bare-metal LLM operating system");
        screen::println("Type a prompt and press Enter. Type 'exit' to shut down.");
        screen::println("");

        // --- Main agent loop ---
        loop {
            screen::print("genos> ");
            let input = keyboard::read_line_with_history(&mut input_history);
            let trimmed = input.trim();

            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "exit" || trimmed == "quit" {
                screen::println("Shutting down genos...");
                // Log session end
                let end_entry = format!(
                    "session_end turn={} context_used={}",
                    state.turn, state.context_used
                );
                palace::append_session_journal(&session_id, &end_entry);
                break;
            }

            if trimmed == "help" {
                self.print_help();
                continue;
            }

            // === Per-turn 11-step sequence ===

            // Step 1: User input captured above

            // Step 2: Build wake-up header
            let header = state.status_header();

            // Step 3: Build full prompt
            // [system_prompt] [wake_up_header] [user_input]
            let full_prompt = format!(
                "{}\n{}\nUser: {}\nAssistant:",
                state.system_prompt, header, trimmed
            );

            // Reset transformer for fresh generation
            self.transformer.reset();

            // Tokenize
            let tokens = self.tokenizer.encode(&full_prompt, true, false);
            if tokens.is_empty() {
                screen::println("[error] Could not tokenize input.");
                continue;
            }

            // Step 4: Generate — run LLM, stream tokens
            let output_text = self.generate_and_collect(&tokens);

            screen::println("");

            // Step 5: Scan output for tool calls
            let tool_calls = json::extract_tool_calls(&output_text);

            // Step 6: Execute tools if any
            let timestamp = timer::get_wall_time()
                .map(|wt| wt.iso8601())
                .unwrap_or_else(|| String::from("unknown"));

            let mut tool_results_text = String::new();
            for tc_json in &tool_calls {
                if let Some(tc) = ToolCall::from_json(tc_json, &session_id) {
                    // Display tool call
                    let summary = tc.args.to_json_string();
                    screen::print_tool_call(&tc.tool, &summary);

                    // Reset per-turn policy counts at first tool call
                    policy.reset_turn();

                    // Execute
                    let result = tools::execute(&tc, &mut policy, &timestamp, state.turn, &mut search_index, &mut entity_graph);

                    // Display result
                    let result_summary = if result.ok {
                        result.result.to_json_string()
                    } else {
                        result
                            .error
                            .as_ref()
                            .map(|e| e.message.clone())
                            .unwrap_or_else(|| String::from("unknown error"))
                    };
                    screen::print_tool_result(&tc.tool, result.ok, &result_summary);

                    // Append result JSON to context
                    let result_json = result.to_json().to_json_string();
                    tool_results_text.push_str(&result_json);
                    tool_results_text.push('\n');
                }
            }

            // Step 7: If tool calls were made, we'd re-run LLM with results
            // (Stories15M can't actually iteratively use tool results,
            // but the infrastructure is ready for Gemma 4 in Phase D)

            // Step 8: Classify output hall type
            let hall = journal::classify_hall(trimmed, &output_text);

            // Step 9: Write verbatim to palace
            let turn_record = journal::TurnRecord {
                session_id: session_id.clone(),
                turn: state.turn,
                timestamp: timestamp.clone(),
                hall: hall.clone(),
                input: String::from(trimmed),
                output: output_text.clone(),
                tool_calls: tool_calls.clone(),
            };
            turn_record.store();

            // Step 10: Run pre_compact if context budget > 80%
            let turn_text = format!("User: {}\nAssistant: {}", trimmed, output_text);
            let approx_tokens = (turn_text.len() + 3) / 4;
            history.record_turn(&turn_text, approx_tokens);

            if state.needs_pre_compact() {
                let freed = history.pre_compact(state.context_budget);
                if freed > 0 {
                    screen::println(&format!("[compact] Archived {} tokens to palace", freed));
                    state.context_used = state.context_used.saturating_sub(freed);
                }
            }

            // Step 11: Update turn counter
            state.advance_turn(approx_tokens);

            // Step 12: Run consolidation if due
            if consolidator.should_run(state.turn) {
                let ts = timer::get_wall_time()
                    .map(|wt| wt.iso8601())
                    .unwrap_or_else(|| String::from("unknown"));
                let (extracted, updated, archived) =
                    consolidator.run(&session_id, state.turn, &ts, &mut search_index);
                if extracted > 0 || archived > 0 {
                    screen::println(&format!(
                        "[consolidate] {} facts extracted, {} updated, {} archived",
                        extracted, updated, archived
                    ));
                }
            }

            // Advance cooperative clock
            timer::advance_ms(100);

            // Step 13: TUI status bar
            let uptime = timer::now_ms() / 1000;
            let mem = timer::get_memory_info();
            screen::render_status_bar(
                "stories15m",
                0.0, // tokens/sec computed in Phase D
                (mem.total_kb.saturating_sub(mem.free_kb) / 1024) as usize,
                (mem.total_kb / 1024) as usize,
                state.turn,
                uptime,
            );

            screen::println("");
        }
    }

    /// Generate tokens and collect the output text.
    fn generate_and_collect(&mut self, prompt_tokens: &[u32]) -> String {
        let num_prompt_tokens = prompt_tokens.len();
        let mut token: u32 = prompt_tokens[0];
        let mut pos = 0usize;
        let mut output = String::new();

        while pos < self.max_seq_len {
            let logits = self.transformer.forward(token, pos);

            let next_token = if pos < num_prompt_tokens - 1 {
                prompt_tokens[pos + 1]
            } else {
                let mut logits_buf: Vec<f32> = logits.to_vec();
                self.sampler.sample(&mut logits_buf)
            };

            pos += 1;

            if pos >= num_prompt_tokens {
                let piece = self.tokenizer.decode(token, next_token);
                screen::print(piece);
                output.push_str(piece);
            }

            if next_token == 2u32 {
                break;
            }

            token = next_token;
        }

        output
    }

    fn print_help(&self) {
        screen::println("=== genos help ===");
        screen::println("  Type any text and genos will respond.");
        screen::println("  The model can emit tool calls as JSON.");
        screen::println("  Commands:");
        screen::println("    help  - Show this help message");
        screen::println("    exit  - Shut down genos");
        screen::println("    quit  - Shut down genos");
        screen::println("");
        screen::println("  Available tools:");
        screen::println("    fs.read, fs.write, fs.list, fs.delete");
        screen::println("    net.fetch");
        screen::println("    memory.facts_get, memory.facts_set");
        screen::println("    sys.clock, sys.introspect");
        screen::println("");
    }
}
