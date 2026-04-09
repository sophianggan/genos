//! Hall-routed session journal.
//! Every turn is classified into a hall type and stored verbatim.

use alloc::string::String;
use alloc::format;
use alloc::vec::Vec;
use genos_kernel::json::{JsonValue, json_object};
use crate::palace::{self, Hall};

/// Classify a turn's content into a hall type.
/// Uses simple keyword heuristics (Stories15M can't self-classify).
/// Phase D with Gemma 4 will use the model itself for classification.
pub fn classify_hall(input: &str, output: &str) -> Hall {
    let combined = {
        let mut s = String::with_capacity(input.len() + output.len() + 1);
        for c in input.chars() {
            s.push(if c.is_ascii_uppercase() { (c as u8 + 32) as char } else { c });
        }
        s.push(' ');
        for c in output.chars() {
            s.push(if c.is_ascii_uppercase() { (c as u8 + 32) as char } else { c });
        }
        s
    };

    // Check for preference indicators
    if combined.contains("prefer") || combined.contains("like") || combined.contains("want")
        || combined.contains("favorite") || combined.contains("always use")
    {
        return Hall::Preferences;
    }

    // Check for advice/recommendation
    if combined.contains("recommend") || combined.contains("should") || combined.contains("suggest")
        || combined.contains("try") || combined.contains("better to")
    {
        return Hall::Advice;
    }

    // Check for discovery/insight
    if combined.contains("found") || combined.contains("learned") || combined.contains("realized")
        || combined.contains("discovered") || combined.contains("turns out")
    {
        return Hall::Discoveries;
    }

    // Check for decisions/facts
    if combined.contains("decided") || combined.contains("set") || combined.contains("changed")
        || combined.contains("updated") || combined.contains("my name")
    {
        return Hall::Facts;
    }

    // Default: event (a thing that happened)
    Hall::Events
}

/// A complete turn record for journaling.
pub struct TurnRecord {
    pub session_id: String,
    pub turn: usize,
    pub timestamp: String,
    pub hall: Hall,
    pub input: String,
    pub output: String,
    pub tool_calls: Vec<JsonValue>,
}

impl TurnRecord {
    /// Serialize to JSONL entry.
    pub fn to_jsonl(&self) -> String {
        let tool_calls_json = JsonValue::Array(self.tool_calls.clone());

        let obj = json_object(&[
            ("session_id", JsonValue::Str(self.session_id.clone())),
            ("turn", JsonValue::Number(self.turn as f64)),
            ("ts", JsonValue::Str(self.timestamp.clone())),
            ("hall", JsonValue::Str(String::from(self.hall.as_str()))),
            ("wing", JsonValue::Str(String::from("general"))),
            ("input", JsonValue::Str(self.input.clone())),
            ("output", JsonValue::Str(self.output.clone())),
            ("tool_calls", tool_calls_json),
        ]);
        obj.to_json_string()
    }

    /// Store this turn: write to session journal AND to the appropriate hall.
    pub fn store(&self) {
        // 1. Append to flat session journal
        let jsonl = self.to_jsonl();
        palace::append_session_journal(&self.session_id, &jsonl);

        // 2. Write verbatim to hall-typed file
        let content = format!(
            "[Turn {} | {} | {}]\nUser: {}\nAssistant: {}\n",
            self.turn, self.timestamp, self.hall.as_str(),
            self.input, self.output
        );
        palace::store_in_hall(&self.session_id, self.turn, self.hall, &content);
    }
}


