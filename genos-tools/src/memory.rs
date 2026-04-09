//! Memory tools: facts_get, facts_set (with contradiction detection), log_turn,
//! store, search, forget, consolidate.
//! Phase B: palace write-only. Phase C: full four-layer memory stack operational.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use genos_kernel::json::JsonValue;
use crate::palace;
use crate::protocol::ToolResult;
use crate::search::SearchIndex;

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

/// memory.store(content, wing, hall, tags?) -> entry_id
/// Writes verbatim content to palace hall file and updates the search index.
pub fn tool_store(
    args: &JsonValue,
    call_id: &str,
    session_id: &str,
    turn: usize,
    index: &mut SearchIndex,
) -> ToolResult {
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'content'", false),
    };
    let wing = args.get("wing").and_then(|v| v.as_str()).unwrap_or("general");
    let hall_str = args.get("hall").and_then(|v| v.as_str()).unwrap_or("discoveries");
    let hall = palace::Hall::from_str(hall_str).unwrap_or(palace::Hall::Discoveries);

    // Write to palace
    palace::store_in_hall(session_id, turn, hall, content);

    // Build entry path for indexing
    let path = format!(
        "\\palace\\wings\\{}\\halls\\{}\\{}_{:04}.txt",
        wing, hall.as_str(), session_id, turn
    );

    // Update search index
    index.index_entry(&path, wing, hall.as_str(), content, turn);

    let entry_id = format!("{}:{}", session_id, turn);
    ToolResult::success(
        call_id,
        genos_kernel::json::json_object(&[
            ("entry_id", JsonValue::Str(entry_id)),
            ("path", JsonValue::Str(path)),
            ("indexed", JsonValue::Bool(true)),
        ]),
        0,
    )
}

/// memory.search(query, wing?, hall?, top_k?) -> Vec<SearchResult>
/// Delegates to the TF-IDF search index over palace halls.
pub fn tool_search(
    args: &JsonValue,
    call_id: &str,
    current_turn: usize,
    index: &SearchIndex,
) -> ToolResult {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'query'", false),
    };
    let wing = args.get("wing").and_then(|v| v.as_str());
    let hall = args.get("hall").and_then(|v| v.as_str());
    let top_k = args.get("top_k")
        .and_then(|v| v.as_f64())
        .map(|n| n as usize)
        .unwrap_or(5);

    let results = index.search(query, wing, hall, top_k, current_turn);

    let arr: Vec<JsonValue> = results
        .iter()
        .map(|r| {
            genos_kernel::json::json_object(&[
                ("path", JsonValue::Str(r.path.clone())),
                ("wing", JsonValue::Str(r.wing.clone())),
                ("hall", JsonValue::Str(r.hall.clone())),
                ("text", JsonValue::Str(r.text.clone())),
                ("score", JsonValue::Number(r.score)),
                ("turn", JsonValue::Number(r.turn as f64)),
            ])
        })
        .collect();

    ToolResult::success(call_id, JsonValue::Array(arr), 0)
}

/// memory.forget(entry_id) -> move entry to archive, remove from index.
/// entry_id format: "{session_id}:{turn}"
pub fn tool_forget(args: &JsonValue, call_id: &str) -> ToolResult {
    let entry_id = match args.get("entry_id").and_then(|v| v.as_str()) {
        Some(id) => id,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'entry_id'", false),
    };

    // Parse entry_id -> session_id:turn
    let parts: Vec<&str> = entry_id.split(':').collect();
    if parts.len() != 2 {
        return ToolResult::failure(call_id, "invalid_args", "entry_id must be 'session:turn'", false);
    }
    let session_id = parts[0];
    let turn: usize = match parts[1].parse() {
        Ok(t) => t,
        Err(_) => return ToolResult::failure(call_id, "invalid_args", "turn must be a number", false),
    };

    // Try each hall to find the file
    let mut found = false;
    for hall in palace::Hall::all() {
        let src_path = format!(
            "\\palace\\wings\\general\\halls\\{}\\{}_{:04}.txt",
            hall.as_str(), session_id, turn
        );
        if let Ok(data) = genos_hal::disk::read_file(&src_path) {
            // Move to archive
            let archive_path = format!(
                "\\palace\\archive\\{}_{:04}_{}.txt",
                session_id, turn, hall.as_str()
            );
            let _ = genos_hal::disk::write_file(&archive_path, &data);
            // Delete original (write empty)
            let _ = genos_hal::disk::write_file(&src_path, b"[archived]");
            found = true;
            break;
        }
    }

    if !found {
        return ToolResult::failure(call_id, "not_found", "entry not found in palace", false);
    }

    ToolResult::success(
        call_id,
        genos_kernel::json::json_object(&[
            ("archived", JsonValue::Bool(true)),
            ("entry_id", JsonValue::Str(String::from(entry_id))),
        ]),
        0,
    )
}

/// memory.consolidate() -> force a consolidation pass.
/// Scans last N journal entries, extracts facts, updates L1.
/// Returns summary of what was consolidated.
pub fn tool_consolidate(
    args: &JsonValue,
    call_id: &str,
    session_id: &str,
    timestamp: &str,
) -> ToolResult {
    let n = args.get("n")
        .and_then(|v| v.as_f64())
        .map(|v| v as usize)
        .unwrap_or(20);

    // Read session journal
    let journal_path = format!("\\palace\\sessions\\{}.jsonl", session_id);
    let data = match genos_hal::disk::read_file(&journal_path) {
        Ok(d) => d,
        Err(_) => return ToolResult::failure(call_id, "no_journal", "no session journal found", false),
    };
    let text = core::str::from_utf8(&data).unwrap_or("");

    // Collect last N lines
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = if lines.len() > n { lines.len() - n } else { 0 };
    let recent = &lines[start..];

    // Extract key=value patterns from journal entries
    let mut extracted: Vec<(String, String)> = Vec::new();
    for line in recent {
        if let Ok(val) = genos_kernel::json::parse(line) {
            // Look for factual content in output fields
            if let Some(output) = val.get("output").and_then(|v| v.as_str()) {
                // Simple pattern: lines containing "=" or ":" that look like facts
                for fact_line in output.lines() {
                    let trimmed = fact_line.trim();
                    if let Some(eq_pos) = trimmed.find(" = ") {
                        let k = String::from(trimmed[..eq_pos].trim());
                        let v = String::from(trimmed[eq_pos + 3..].trim());
                        if !k.is_empty() && !v.is_empty() && k.len() < 64 {
                            extracted.push((k, v));
                        }
                    }
                }
            }
        }
    }

    // Apply extracted facts via facts_set (reuses contradiction detection)
    let mut updated = 0usize;
    for (key, value) in &extracted {
        let fact_args = genos_kernel::json::json_object(&[
            ("key", JsonValue::Str(key.clone())),
            ("value", JsonValue::Str(value.clone())),
        ]);
        let result = tool_facts_set(&fact_args, "consolidate", timestamp);
        if result.ok {
            updated += 1;
        }
    }

    ToolResult::success(
        call_id,
        genos_kernel::json::json_object(&[
            ("entries_scanned", JsonValue::Number(recent.len() as f64)),
            ("facts_extracted", JsonValue::Number(extracted.len() as f64)),
            ("facts_updated", JsonValue::Number(updated as f64)),
        ]),
        0,
    )
}
