//! Network HAL — HTTP/1.1 client over UEFI TCP4.
//!
//! Implements URL parsing, HTTP request/response handling, and a `fetch()` API.
//! The raw TCP4 socket layer is stubbed pending uefi crate TCP4 support;
//! all higher-level HTTP logic (parsing, serialization) is fully functional.
//!
//! For QEMU testing: `make qemu-net` starts with `-netdev user` and e1000.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

/// Network errors.
pub enum NetError {
    NoInterface,
    DnsError,
    ConnectFailed,
    Timeout,
    ParseError,
    HttpError(u16),
    IoError,
}

impl NetError {
    pub fn as_str(&self) -> &str {
        match self {
            NetError::NoInterface => "no network interface",
            NetError::DnsError => "DNS resolution failed",
            NetError::ConnectFailed => "TCP connection failed",
            NetError::Timeout => "request timed out",
            NetError::ParseError => "HTTP parse error",
            NetError::HttpError(_) => "HTTP error",
            NetError::IoError => "I/O error",
        }
    }
}

/// Parsed URL components.
#[allow(dead_code)]
struct ParsedUrl<'a> {
    host: &'a str,
    port: u16,
    path: &'a str,
}

/// Parse a simple HTTP URL into host, port, path.
#[allow(dead_code)]
fn parse_url(url: &str) -> Result<ParsedUrl<'_>, NetError> {
    let rest = url.strip_prefix("http://").ok_or(NetError::ParseError)?;

    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };

    let (host, port) = match host_port.find(':') {
        Some(i) => {
            let p = parse_u16(&host_port[i + 1..]).ok_or(NetError::ParseError)?;
            (&host_port[..i], p)
        }
        None => (host_port, 80u16),
    };

    if host.is_empty() {
        return Err(NetError::ParseError);
    }

    Ok(ParsedUrl { host, port, path })
}

#[allow(dead_code)]
fn parse_u16(s: &str) -> Option<u16> {
    let mut n: u16 = 0;
    for b in s.bytes() {
        if b < b'0' || b > b'9' {
            return None;
        }
        n = n.checked_mul(10)?.checked_add((b - b'0') as u16)?;
    }
    Some(n)
}

/// Build an HTTP/1.1 GET request.
#[allow(dead_code)]
fn build_get_request(host: &str, path: &str) -> Vec<u8> {
    format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: genos/0.1\r\n\r\n",
        path, host
    )
    .into_bytes()
}

/// Parse HTTP response: extract status code and body.
#[allow(dead_code)]
fn parse_response(data: &[u8]) -> Result<(u16, Vec<u8>), NetError> {
    let header_end = find_header_end(data).ok_or(NetError::ParseError)?;
    let headers = core::str::from_utf8(&data[..header_end]).map_err(|_| NetError::ParseError)?;

    let status_line = headers.lines().next().ok_or(NetError::ParseError)?;
    let status_code = parse_status_code(status_line)?;

    let body_start = header_end + 4;
    let body = if body_start < data.len() {
        data[body_start..].to_vec()
    } else {
        Vec::new()
    };

    Ok((status_code, body))
}

#[allow(dead_code)]
fn find_header_end(data: &[u8]) -> Option<usize> {
    for i in 0..data.len().saturating_sub(3) {
        if &data[i..i + 4] == b"\r\n\r\n" {
            return Some(i);
        }
    }
    None
}

#[allow(dead_code)]
fn parse_status_code(line: &str) -> Result<u16, NetError> {
    let mut parts = line.split(' ');
    let _version = parts.next().ok_or(NetError::ParseError)?;
    let code_str = parts.next().ok_or(NetError::ParseError)?;
    parse_u16(code_str).ok_or(NetError::ParseError)
}

/// Fetch a URL via HTTP GET. Returns the response body bytes.
///
/// Primary network API for genos tools. Currently returns `NoInterface`
/// because the uefi 0.37 crate doesn't expose TCP4 bindings.
/// All HTTP logic (URL parsing, request building, response parsing)
/// is implemented and ready for TCP4 wiring.
pub fn fetch(url: &str) -> Result<Vec<u8>, NetError> {
    let parsed = parse_url(url)?;
    // Build request (validates URL structure even if we can't send it yet)
    let _request = build_get_request(parsed.host, parsed.path);
    // TCP4 socket layer is not yet available in the uefi crate.
    Err(NetError::NoInterface)
}

/// Convenience: fetch a URL and return the body as a UTF-8 string.
pub fn fetch_text(url: &str) -> Result<String, NetError> {
    let bytes = fetch(url)?;
    String::from_utf8(bytes).map_err(|_| NetError::ParseError)
}
