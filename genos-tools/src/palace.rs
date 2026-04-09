//! Palace memory scaffold — creates the MemPalace directory hierarchy.
//!
//! Palace structure: Wings → Rooms → Halls → Drawers
//! Phase B establishes the general wing with 5 hall types.

use alloc::string::String;
use alloc::format;
use genos_hal::disk;

/// Hall types for classifying memory entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hall {
    Facts,
    Events,
    Discoveries,
    Preferences,
    Advice,
}

impl Hall {
    pub fn as_str(&self) -> &'static str {
        match self {
            Hall::Facts => "facts",
            Hall::Events => "events",
            Hall::Discoveries => "discoveries",
            Hall::Preferences => "preferences",
            Hall::Advice => "advice",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "facts" => Some(Hall::Facts),
            "events" => Some(Hall::Events),
            "discoveries" => Some(Hall::Discoveries),
            "preferences" => Some(Hall::Preferences),
            "advice" => Some(Hall::Advice),
            _ => None,
        }
    }

    pub fn all() -> &'static [Hall] {
        &[Hall::Facts, Hall::Events, Hall::Discoveries, Hall::Preferences, Hall::Advice]
    }
}

/// Default identity template written on first boot.
const DEFAULT_IDENTITY: &str = "\
You are genos, a bare-metal LLM operating system.
You run directly on hardware via UEFI — no Linux, no Windows, no BSD.
You make every decision through structured tool calls.
You have persistent memory stored in a palace structure.
You are helpful, precise, and self-aware of your context window.
";

/// Default facts.kv template.
const DEFAULT_FACTS: &str = "\
os_name = genos
os_version = 0.2.0
model = stories15m
phase = B
";

/// Ensure the full palace directory structure exists.
/// Creates directories and default files on first boot.
/// Returns the session ID for this boot.
pub fn ensure_structure(time_suffix: &str) -> String {
    let session_id = format!("session_{}", time_suffix);

    // Create palace root files
    ensure_file("\\palace\\identity.txt", DEFAULT_IDENTITY);
    ensure_file("\\palace\\facts.kv", DEFAULT_FACTS);

    // Create hall directories by writing a .keep file in each
    for hall in Hall::all() {
        let keep_path = format!("\\palace\\wings\\general\\halls\\{}\\.keep", hall.as_str());
        ensure_file(&keep_path, "");
    }

    // Create sessions directory
    let sessions_keep = format!("\\palace\\sessions\\.keep");
    ensure_file(&sessions_keep, "");

    session_id
}

/// Write a file only if it doesn't already exist (first boot).
fn ensure_file(path: &str, default_content: &str) {
    if disk::read_file(path).is_err() {
        let _ = disk::write_file(path, default_content.as_bytes());
    }
}

/// Read identity.txt (L0 memory).
pub fn read_identity() -> String {
    match disk::read_file("\\palace\\identity.txt") {
        Ok(data) => String::from_utf8_lossy(&data).into_owned(),
        Err(_) => String::from(DEFAULT_IDENTITY),
    }
}

/// Read facts.kv (L1 memory) as key-value pairs.
pub fn read_facts() -> alloc::vec::Vec<(String, String)> {
    let data = match disk::read_file("\\palace\\facts.kv") {
        Ok(d) => d,
        Err(_) => return alloc::vec::Vec::new(),
    };
    let text = core::str::from_utf8(&data).unwrap_or("");
    parse_facts_kv(text)
}

/// Write facts.kv from key-value pairs.
pub fn write_facts(facts: &[(String, String)]) {
    let mut content = String::new();
    for (k, v) in facts {
        content.push_str(k);
        content.push_str(" = ");
        content.push_str(v);
        content.push('\n');
    }
    let _ = disk::write_file("\\palace\\facts.kv", content.as_bytes());
}

/// Read facts.kv as raw text.
pub fn read_facts_raw() -> String {
    match disk::read_file("\\palace\\facts.kv") {
        Ok(data) => String::from_utf8_lossy(&data).into_owned(),
        Err(_) => String::new(),
    }
}

/// Write raw text to facts.kv.
pub fn write_facts_raw(content: &str) {
    let _ = disk::write_file("\\palace\\facts.kv", content.as_bytes());
}

/// Parse a facts.kv file into key-value pairs.
fn parse_facts_kv(text: &str) -> alloc::vec::Vec<(String, String)> {
    let mut result = alloc::vec::Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        if let Some(eq_pos) = line.find(" = ") {
            let key = String::from(line[..eq_pos].trim());
            let val = String::from(line[eq_pos + 3..].trim());
            result.push((key, val));
        } else if let Some(eq_pos) = line.find('=') {
            let key = String::from(line[..eq_pos].trim());
            let val = String::from(line[eq_pos + 1..].trim());
            result.push((key, val));
        }
    }
    result
}

/// Store a verbatim entry in the appropriate hall.
pub fn store_in_hall(
    session_id: &str,
    turn: usize,
    hall: Hall,
    content: &str,
) {
    let path = format!(
        "\\palace\\wings\\general\\halls\\{}\\{}_{:04}.txt",
        hall.as_str(),
        session_id,
        turn
    );
    let _ = disk::write_file(&path, content.as_bytes());
}

/// Append a JSONL entry to the session journal.
pub fn append_session_journal(session_id: &str, entry: &str) {
    let path = format!("\\palace\\sessions\\{}.jsonl", session_id);
    // Read existing content and append
    let mut content = match disk::read_file(&path) {
        Ok(data) => String::from_utf8_lossy(&data).into_owned(),
        Err(_) => String::new(),
    };
    content.push_str(entry);
    content.push('\n');
    let _ = disk::write_file(&path, content.as_bytes());
}
