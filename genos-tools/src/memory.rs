//! Memory tools: facts_get, facts_set (with contradiction detection), log_turn.
//! These are the Phase B memory tools — palace is write-only, search comes in Phase C.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use genos_kernel::json::JsonValue;
use crate::palace;
use crate::protocol::ToolResult;

/// memory.facts_get() -> all L1 facts as JSON array of {key, value}
pub fn tool_facts_get(_args: &JsonValue, call_id: &str) -> ToolResult {
    let facts = palace::read_facts();
    let arr: Vec<JsonValue> = facts
        .iter()
        .map(|(k, v)| {
            genos_kernel::json::json_object(&[
                ("key", JsonValue::Str(k.clone())),
                ("value", JsonValue::Str(v.clone())),
            ])
        })
        .collect();
    ToolResult::success(call_id, JsonValue::Array(arr), 0)
}

/// memory.facts_set(key, value) -> with contradiction detection.
/// If the key already exists with a different value:
///   - Old entry gets [superseded] annotation
///   - Both versions are kept with timestamps
///   - Returns a warning in the result
pub fn tool_facts_set(args: &JsonValue, call_id: &str, timestamp: &str) -> ToolResult {
    let key = match args.get("key").and_then(|v| v.as_str()) {
        Some(k) => k,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'key'", false),
    };
    let value = match args.get("value").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'value'", false),
    };

    let existing_facts = palace::read_facts();
    let mut contradiction = false;
    let mut old_value = String::new();

    // Check for contradiction
    for (k, v) in &existing_facts {
        if k == key && v != value {
            contradiction = true;
            old_value = v.clone();
            break;
        }
    }

    if contradiction {
        // Rewrite facts.kv with the old entry marked as superseded
        let raw = palace::read_facts_raw();
        let mut new_content = String::new();

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(key) {
                if let Some(eq_pos) = trimmed.find(" = ") {
                    let line_key = trimmed[..eq_pos].trim();
                    if line_key == key && !trimmed.contains("[superseded") {
                        // Mark old entry as superseded
                        new_content.push_str(&format!(
                            "{} [superseded: {}, was: {}]\n",
                            trimmed, timestamp, old_value
                        ));
                        continue;
                    }
                }
            }
            new_content.push_str(line);
            new_content.push('\n');
        }
        // Append new value
        new_content.push_str(&format!("{} = {} [valid_from: {}]\n", key, value, timestamp));
        palace::write_facts_raw(&new_content);

        // Also archive the superseded entry
        let superseded_entry = format!(
            "{} = {} [valid_until: {}, superseded_by: {}]\n",
            key, old_value, timestamp, value
        );
        let archive_path = "\\palace\\wings\\general\\halls\\facts\\superseded.kv";
        let mut archive = match genos_hal::disk::read_file(archive_path) {
            Ok(data) => String::from_utf8_lossy(&data).into_owned(),
            Err(_) => String::new(),
        };
        archive.push_str(&superseded_entry);
        let _ = genos_hal::disk::write_file(archive_path, archive.as_bytes());

        let warning = format!(
            "Fact updated: {} = \"{}\" (was: \"{}\")",
            key, value, old_value
        );
        ToolResult::success(
            call_id,
            genos_kernel::json::json_object(&[
                ("updated", JsonValue::Bool(true)),
                ("contradiction", JsonValue::Bool(true)),
                ("warning", JsonValue::Str(warning)),
            ]),
            0,
        )
    } else {
        // Check if key already exists with same value (no-op)
        let already_exists = existing_facts.iter().any(|(k, v)| k == key && v == value);
        if already_exists {
            return ToolResult::success(
                call_id,
                genos_kernel::json::json_object(&[
                    ("updated", JsonValue::Bool(false)),
                    ("reason", JsonValue::Str(String::from("value unchanged"))),
                ]),
                0,
            );
        }

        // New key — append to facts.kv
        let mut raw = palace::read_facts_raw();
        raw.push_str(&format!("{} = {} [valid_from: {}]\n", key, value, timestamp));
        palace::write_facts_raw(&raw);

        ToolResult::success(
            call_id,
            genos_kernel::json::json_object(&[
                ("updated", JsonValue::Bool(true)),
                ("contradiction", JsonValue::Bool(false)),
            ]),
            0,
        )
    }
}

/// memory.log_turn(data) -> append raw verbatim turn data to session journal
pub fn tool_log_turn(args: &JsonValue, call_id: &str, session_id: &str) -> ToolResult {
    let data = match args.get("data") {
        Some(d) => d.to_json_string(),
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'data'", false),
    };
    palace::append_session_journal(session_id, &data);
    ToolResult::success(call_id, JsonValue::Bool(true), 0)
}
