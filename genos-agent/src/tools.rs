//! Tool executor — routes parsed tool calls to handler functions.
//!
//! Central dispatch: takes a ToolCall, checks policy, calls the right
//! handler in genos-tools, returns ToolResult.

use genos_hal::timer;
use genos_kernel::json::JsonValue;
use genos_tools::protocol::{ToolCall, ToolResult};
use genos_tools::search::SearchIndex;
use crate::policy::{PolicyCheck, PolicyEngine};

/// Execute a tool call, checking policy first.
pub fn execute(
    call: &ToolCall,
    policy: &mut PolicyEngine,
    timestamp: &str,
    turn: usize,
    index: &mut SearchIndex,
) -> ToolResult {
    // Check policy
    match policy.check(&call.tool) {
        PolicyCheck::Allowed => {}
        PolicyCheck::Disabled => {
            return ToolResult::failure(
                &call.call_id,
                "tool_disabled",
                "tool is disabled by policy",
                false,
            );
        }
        PolicyCheck::NeedsApproval => {
            return ToolResult::failure(
                &call.call_id,
                "approval_required",
                "tool requires user approval",
                false,
            );
        }
        PolicyCheck::RateLimited => {
            return ToolResult::failure(
                &call.call_id,
                "rate_limited",
                "tool call limit exceeded for this turn",
                true,
            );
        }
    }

    // Record call for rate limiting
    policy.record_call(&call.tool);

    // Measure execution time
    let start = timer::now_ms();

    // Dispatch to handler
    let mut result = dispatch(
        &call.tool, &call.args, &call.call_id, timestamp,
        &call.session_id, turn, index,
    );

    // Set elapsed time
    let elapsed = timer::now_ms() - start;
    result.elapsed_ms = elapsed;

    result
}

/// Dispatch a tool call to the appropriate handler.
fn dispatch(
    tool: &str,
    args: &JsonValue,
    call_id: &str,
    timestamp: &str,
    session_id: &str,
    turn: usize,
    index: &mut SearchIndex,
) -> ToolResult {
    match tool {
        "fs.read" => genos_tools::fs::tool_read(args, call_id),
        "fs.write" => genos_tools::fs::tool_write(args, call_id),
        "fs.list" => genos_tools::fs::tool_list(args, call_id),
        "fs.delete" => genos_tools::fs::tool_delete(args, call_id),
        "net.fetch" => genos_tools::net::tool_fetch(args, call_id),
        "memory.facts_get" => genos_tools::memory::tool_facts_get(args, call_id),
        "memory.facts_set" => genos_tools::memory::tool_facts_set(args, call_id, timestamp),
        "memory.log_turn" => {
            let sid = args
                .get("session_id")
                .and_then(|v| v.as_str())
                .unwrap_or(session_id);
            genos_tools::memory::tool_log_turn(args, call_id, sid)
        }
        "memory.store" => genos_tools::memory::tool_store(args, call_id, session_id, turn, index),
        "memory.search" => genos_tools::memory::tool_search(args, call_id, turn, index),
        "memory.consolidate" => genos_tools::memory::tool_consolidate(args, call_id, session_id, timestamp),
        "memory.forget" => genos_tools::memory::tool_forget(args, call_id),
        "sys.clock" => genos_tools::sys::tool_clock(args, call_id),
        "sys.introspect" => genos_tools::sys::tool_introspect(args, call_id),
        _ => ToolResult::failure(
            call_id,
            "unknown_tool",
            "tool not found in registry",
            false,
        ),
    }
}
