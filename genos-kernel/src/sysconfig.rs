//! Minimal TOML key=value parser for system configuration.
//! Supports: string values, integer values, float values, boolean values.
//! One key=value per line, # comments, [section] headers (stored as prefix).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::format;

/// System configuration loaded from \system\config.toml.
pub struct SystemConfig {
    pub values: BTreeMap<String, String>,
}

impl SystemConfig {
    /// Parse TOML-like config from raw bytes.
    pub fn parse(data: &[u8]) -> Self {
        let text = core::str::from_utf8(data).unwrap_or("");
        let mut values = BTreeMap::new();
        let mut section = String::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = String::from(&line[1..line.len() - 1]);
                continue;
            }
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim();
                let val = line[eq_pos + 1..].trim();
                // Strip quotes from string values
                let val = if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
                    &val[1..val.len() - 1]
                } else {
                    val
                };
                let full_key = if section.is_empty() {
                    String::from(key)
                } else {
                    format!("{}.{}", section, key)
                };
                values.insert(full_key, String::from(val));
            }
        }

        SystemConfig { values }
    }

    /// Create default config when no file exists.
    pub fn default_config() -> Self {
        let mut values = BTreeMap::new();
        values.insert(String::from("sampling.temperature"), String::from("1.0"));
        values.insert(String::from("sampling.top_p"), String::from("0.9"));
        values.insert(String::from("sampling.top_k"), String::from("40"));
        values.insert(String::from("context.budget"), String::from("256"));
        values.insert(String::from("context.pre_compact_threshold"), String::from("0.8"));
        values.insert(String::from("memory.consolidation_interval"), String::from("20"));
        SystemConfig { values }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get_str(key).and_then(|s| parse_f32(s))
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.get_str(key).and_then(|s| parse_usize(s))
    }

    /// Serialize to TOML string for writing back.
    pub fn to_toml(&self) -> String {
        let mut out = String::from("# genos system configuration\n\n");
        let mut last_section = String::new();
        for (key, val) in &self.values {
            if let Some(dot) = key.find('.') {
                let section = &key[..dot];
                let subkey = &key[dot + 1..];
                if section != last_section.as_str() {
                    if !last_section.is_empty() {
                        out.push('\n');
                    }
                    out.push('[');
                    out.push_str(section);
                    out.push_str("]\n");
                    last_section = String::from(section);
                }
                out.push_str(subkey);
            } else {
                out.push_str(key);
            }
            out.push_str(" = ");
            // Try to detect if it's numeric or boolean, otherwise quote
            if val == "true" || val == "false" || parse_f32(val).is_some() {
                out.push_str(val);
            } else {
                out.push('"');
                out.push_str(val);
                out.push('"');
            }
            out.push('\n');
        }
        out
    }
}

fn parse_f32(s: &str) -> Option<f32> {
    let bytes = s.as_bytes();
    if bytes.is_empty() { return None; }
    let mut i = 0;
    let neg = if bytes[0] == b'-' { i += 1; true } else { false };
    let mut int_part: f32 = 0.0;
    let mut has_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        int_part = int_part * 10.0 + (bytes[i] - b'0') as f32;
        i += 1;
        has_digit = true;
    }
    let mut frac = 0.0f32;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let mut factor = 0.1f32;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            frac += (bytes[i] - b'0') as f32 * factor;
            factor *= 0.1;
            i += 1;
            has_digit = true;
        }
    }
    if !has_digit || i != bytes.len() { return None; }
    let result = int_part + frac;
    Some(if neg { -result } else { result })
}

fn parse_usize(s: &str) -> Option<usize> {
    let mut result: usize = 0;
    for &b in s.as_bytes() {
        if !b.is_ascii_digit() { return None; }
        result = result.checked_mul(10)?.checked_add((b - b'0') as usize)?;
    }
    if s.is_empty() { None } else { Some(result) }
}
