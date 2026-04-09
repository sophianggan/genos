//! Hand-rolled JSON parser for no_std environments.
//! Supports: objects, arrays, strings, numbers, booleans, null.
//! ~300 LOC, zero external dependencies.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?.get(key)
    }

    /// Serialize to JSON string.
    pub fn to_json_string(&self) -> String {
        match self {
            JsonValue::Null => String::from("null"),
            JsonValue::Bool(b) => if *b { String::from("true") } else { String::from("false") },
            JsonValue::Number(n) => {
                if *n == (*n as i64) as f64 && n.is_finite() {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            JsonValue::Str(s) => {
                let mut out = String::from("\"");
                for ch in s.chars() {
                    match ch {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c => out.push(c),
                    }
                }
                out.push('"');
                out
            }
            JsonValue::Array(a) => {
                let mut out = String::from("[");
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    out.push_str(&v.to_json_string());
                }
                out.push(']');
                out
            }
            JsonValue::Object(m) => {
                let mut out = String::from("{");
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    out.push_str(&JsonValue::Str(k.clone()).to_json_string());
                    out.push(':');
                    out.push_str(&v.to_json_string());
                }
                out.push('}');
                out
            }
        }
    }
}

/// JSON parse error.
#[derive(Debug)]
pub struct JsonError {
    pub msg: String,
    pub pos: usize,
}

/// Parse a JSON string into a JsonValue.
pub fn parse(input: &str) -> Result<JsonValue, JsonError> {
    let bytes = input.as_bytes();
    let mut pos = 0;
    skip_whitespace(bytes, &mut pos);
    let val = parse_value(bytes, &mut pos)?;
    skip_whitespace(bytes, &mut pos);
    Ok(val)
}

fn parse_value(b: &[u8], pos: &mut usize) -> Result<JsonValue, JsonError> {
    skip_whitespace(b, pos);
    if *pos >= b.len() {
        return Err(JsonError { msg: String::from("unexpected end"), pos: *pos });
    }
    match b[*pos] {
        b'"' => parse_string(b, pos).map(JsonValue::Str),
        b'{' => parse_object(b, pos),
        b'[' => parse_array(b, pos),
        b't' | b'f' => parse_bool(b, pos),
        b'n' => parse_null(b, pos),
        b'-' | b'0'..=b'9' => parse_number(b, pos),
        c => Err(JsonError { msg: format!("unexpected char '{}'", c as char), pos: *pos }),
    }
}

fn parse_string(b: &[u8], pos: &mut usize) -> Result<String, JsonError> {
    if b[*pos] != b'"' {
        return Err(JsonError { msg: String::from("expected '\"'"), pos: *pos });
    }
    *pos += 1;
    let mut s = String::new();
    while *pos < b.len() {
        match b[*pos] {
            b'"' => {
                *pos += 1;
                return Ok(s);
            }
            b'\\' => {
                *pos += 1;
                if *pos >= b.len() {
                    return Err(JsonError { msg: String::from("unterminated escape"), pos: *pos });
                }
                match b[*pos] {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b'r' => s.push('\r'),
                    b't' => s.push('\t'),
                    b'u' => {
                        // Parse 4 hex digits
                        *pos += 1;
                        if *pos + 4 > b.len() {
                            return Err(JsonError { msg: String::from("short unicode escape"), pos: *pos });
                        }
                        let hex = core::str::from_utf8(&b[*pos..*pos + 4])
                            .map_err(|_| JsonError { msg: String::from("bad unicode"), pos: *pos })?;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|_| JsonError { msg: String::from("bad hex"), pos: *pos })?;
                        if let Some(c) = char::from_u32(code) {
                            s.push(c);
                        }
                        *pos += 3; // will be incremented below
                    }
                    c => s.push(c as char),
                }
            }
            c => s.push(c as char),
        }
        *pos += 1;
    }
    Err(JsonError { msg: String::from("unterminated string"), pos: *pos })
}

fn parse_number(b: &[u8], pos: &mut usize) -> Result<JsonValue, JsonError> {
    let start = *pos;
    if *pos < b.len() && b[*pos] == b'-' {
        *pos += 1;
    }
    while *pos < b.len() && b[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos < b.len() && b[*pos] == b'.' {
        *pos += 1;
        while *pos < b.len() && b[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    if *pos < b.len() && (b[*pos] == b'e' || b[*pos] == b'E') {
        *pos += 1;
        if *pos < b.len() && (b[*pos] == b'+' || b[*pos] == b'-') {
            *pos += 1;
        }
        while *pos < b.len() && b[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    let num_str = core::str::from_utf8(&b[start..*pos])
        .map_err(|_| JsonError { msg: String::from("bad number"), pos: start })?;
    let n = parse_f64(num_str)
        .ok_or_else(|| JsonError { msg: String::from("bad number"), pos: start })?;
    Ok(JsonValue::Number(n))
}

/// Simple f64 parser for no_std (no f64::from_str available without std).
fn parse_f64(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut i = 0;
    let negative = if bytes[0] == b'-' { i += 1; true } else { false };

    let mut int_part: f64 = 0.0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        int_part = int_part * 10.0 + (bytes[i] - b'0') as f64;
        i += 1;
    }

    let mut frac_part: f64 = 0.0;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let mut factor = 0.1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            frac_part += (bytes[i] - b'0') as f64 * factor;
            factor *= 0.1;
            i += 1;
        }
    }

    let mut result = int_part + frac_part;

    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        let exp_neg = if i < bytes.len() && bytes[i] == b'-' { i += 1; true }
                      else if i < bytes.len() && bytes[i] == b'+' { i += 1; false }
                      else { false };
        let mut exp: i32 = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            exp = exp * 10 + (bytes[i] - b'0') as i32;
            i += 1;
        }
        if exp_neg { exp = -exp; }
        // Manual pow10
        let mut factor = 1.0f64;
        let abs_exp = if exp < 0 { -exp } else { exp } as u32;
        for _ in 0..abs_exp {
            factor *= 10.0;
        }
        if exp < 0 { result /= factor; } else { result *= factor; }
    }

    if negative { result = -result; }
    Some(result)
}

fn parse_bool(b: &[u8], pos: &mut usize) -> Result<JsonValue, JsonError> {
    if b[*pos..].starts_with(b"true") {
        *pos += 4;
        Ok(JsonValue::Bool(true))
    } else if b[*pos..].starts_with(b"false") {
        *pos += 5;
        Ok(JsonValue::Bool(false))
    } else {
        Err(JsonError { msg: String::from("expected bool"), pos: *pos })
    }
}

fn parse_null(b: &[u8], pos: &mut usize) -> Result<JsonValue, JsonError> {
    if b[*pos..].starts_with(b"null") {
        *pos += 4;
        Ok(JsonValue::Null)
    } else {
        Err(JsonError { msg: String::from("expected null"), pos: *pos })
    }
}

fn parse_object(b: &[u8], pos: &mut usize) -> Result<JsonValue, JsonError> {
    *pos += 1; // skip '{'
    skip_whitespace(b, pos);
    let mut map = BTreeMap::new();
    if *pos < b.len() && b[*pos] == b'}' {
        *pos += 1;
        return Ok(JsonValue::Object(map));
    }
    loop {
        skip_whitespace(b, pos);
        let key = parse_string(b, pos)?;
        skip_whitespace(b, pos);
        if *pos >= b.len() || b[*pos] != b':' {
            return Err(JsonError { msg: String::from("expected ':'"), pos: *pos });
        }
        *pos += 1;
        let val = parse_value(b, pos)?;
        map.insert(key, val);
        skip_whitespace(b, pos);
        if *pos >= b.len() {
            return Err(JsonError { msg: String::from("unterminated object"), pos: *pos });
        }
        if b[*pos] == b'}' {
            *pos += 1;
            return Ok(JsonValue::Object(map));
        }
        if b[*pos] != b',' {
            return Err(JsonError { msg: String::from("expected ',' or '}'"), pos: *pos });
        }
        *pos += 1;
    }
}

fn parse_array(b: &[u8], pos: &mut usize) -> Result<JsonValue, JsonError> {
    *pos += 1; // skip '['
    skip_whitespace(b, pos);
    let mut arr = Vec::new();
    if *pos < b.len() && b[*pos] == b']' {
        *pos += 1;
        return Ok(JsonValue::Array(arr));
    }
    loop {
        let val = parse_value(b, pos)?;
        arr.push(val);
        skip_whitespace(b, pos);
        if *pos >= b.len() {
            return Err(JsonError { msg: String::from("unterminated array"), pos: *pos });
        }
        if b[*pos] == b']' {
            *pos += 1;
            return Ok(JsonValue::Array(arr));
        }
        if b[*pos] != b',' {
            return Err(JsonError { msg: String::from("expected ',' or ']'"), pos: *pos });
        }
        *pos += 1;
    }
}

fn skip_whitespace(b: &[u8], pos: &mut usize) {
    while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

/// Helper: build a JSON object from key-value pairs.
pub fn json_object(pairs: &[(&str, JsonValue)]) -> JsonValue {
    let mut map = BTreeMap::new();
    for (k, v) in pairs {
        map.insert(String::from(*k), v.clone());
    }
    JsonValue::Object(map)
}

/// Scan for JSON tool calls in model output.
/// Finds `{"tool": "...", "args": {...}}` patterns.
pub fn extract_tool_calls(output: &str) -> Vec<JsonValue> {
    let mut calls = Vec::new();
    let bytes = output.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // Try parsing from this position
            let rest = &output[i..];
            if let Ok(val) = parse(rest) {
                if let Some(obj) = val.as_object() {
                    if obj.contains_key("tool") {
                        calls.push(val);
                    }
                }
            }
            // Skip past this '{' to find next potential call
            // Find matching '}' by scanning
            let mut depth = 0;
            let mut j = i;
            while j < bytes.len() {
                match bytes[j] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            i = j + 1;
                            break;
                        }
                    }
                    b'"' => {
                        // Skip string content
                        j += 1;
                        while j < bytes.len() && bytes[j] != b'"' {
                            if bytes[j] == b'\\' { j += 1; }
                            j += 1;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if depth != 0 { i += 1; }
        } else {
            i += 1;
        }
    }
    calls
}
