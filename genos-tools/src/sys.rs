//! System tools: sys.clock and sys.introspect.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::format;
use genos_hal::timer;
use genos_kernel::json::JsonValue;
use crate::protocol::ToolResult;

/// sys.clock() -> current wall-clock time as ISO 8601
pub fn tool_clock(_args: &JsonValue, call_id: &str) -> ToolResult {
    match timer::get_wall_time() {
        Some(wt) => {
            let mut map = BTreeMap::new();
            map.insert(String::from("iso8601"), JsonValue::Str(wt.iso8601()));
            map.insert(String::from("uptime_ms"), JsonValue::Number(timer::now_ms() as f64));
            ToolResult::success(call_id, JsonValue::Object(map), 0)
        }
        None => ToolResult::failure(call_id, "clock_error", "UEFI GetTime() unavailable", true),
    }
}

/// sys.introspect() -> system status (RAM, context budget, uptime)
pub fn tool_introspect(_args: &JsonValue, call_id: &str) -> ToolResult {
    let mut map = BTreeMap::new();

    map.insert(String::from("uptime_ms"), JsonValue::Number(timer::now_ms() as f64));
    map.insert(String::from("model"), JsonValue::Str(String::from("stories15m")));
    map.insert(String::from("vocab_size"), JsonValue::Number(32000.0));
    map.insert(String::from("max_seq_len"), JsonValue::Number(256.0));
    map.insert(String::from("phase"), JsonValue::Str(String::from("B")));
    map.insert(String::from("version"), JsonValue::Str(String::from("0.1.0")));

    let mem = timer::get_memory_info();
    map.insert(String::from("ram_free_kb"), JsonValue::Number(mem.free_kb as f64));
    map.insert(String::from("ram_total_kb"), JsonValue::Number(mem.total_kb as f64));

    ToolResult::success(call_id, JsonValue::Object(map), 0)
}

/// Format a status header for the system prompt.
/// "[Session: {id} | {ISO8601} | Turn: {N} | Context: {used}/{budget}t | RAM: {free}MB]"
pub fn format_status_header(
    session_id: &str,
    turn: usize,
    context_used: usize,
    context_budget: usize,
) -> String {
    let time_str = timer::get_wall_time()
        .map(|wt| wt.iso8601())
        .unwrap_or_else(|| String::from("unknown"));

    let mem = timer::get_memory_info();
    let ram_mb = mem.free_kb / 1024;

    format!(
        "[Session: {} | {} | Turn: {} | Context: {}/{}t | RAM: {}MB]",
        session_id, time_str, turn, context_used, context_budget, ram_mb
    )
}
