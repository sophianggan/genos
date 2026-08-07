//! Network HAL — HTTP/1.1 client over UEFI TCP4.
//!
//! Implements URL parsing, HTTP request/response handling, and a `fetch()` API.
//! The raw TCP4 socket layer is stubbed pending uefi crate TCP4 support;
//! all higher-level HTTP logic (parsing, serialization) is fully functional.
//!
//! For QEMU testing: `make qemu-net` starts with `-netdev user` and e1000.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Network errors.
pub enum NetError {
    NoInterface,
    DnsError,
    ConnectFailed,
    Timeout,
    ParseError,
    HttpError(u16),
    InvalidHeader,
    ResponseTooLarge,
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
            NetError::InvalidHeader => "invalid HTTP header",
            NetError::ResponseTooLarge => "HTTP response too large",
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
#[cfg(not(feature = "hosted"))]
pub fn fetch(url: &str) -> Result<Vec<u8>, NetError> {
    let parsed = parse_url(url)?;
    // Build request (validates URL structure even if we can't send it yet)
    let _request = build_get_request(parsed.host, parsed.path);
    // TCP4 socket layer is not yet available in the uefi crate.
    Err(NetError::NoInterface)
}

/// Hosted fetch: uses std::net TCP to perform real HTTP GET requests.
#[cfg(feature = "hosted")]
pub fn fetch(url: &str) -> Result<Vec<u8>, NetError> {
    fetch_following_redirects(url, 0)
}

#[cfg(feature = "hosted")]
fn fetch_following_redirects(url: &str, depth: u8) -> Result<Vec<u8>, NetError> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    if depth > 5 {
        return Err(NetError::ParseError); // too many redirects
    }

    let is_https = url.starts_with("https://");
    let is_http = url.starts_with("http://");
    if !is_http && !is_https {
        return Err(NetError::ParseError);
    }

    let parsed = parse_url_either(url)?;
    let request = build_get_request(&parsed.host, &parsed.path);
    let addr = format!("{}:{}", parsed.host, parsed.port);

    // Read the full raw response into a buffer
    let response: Vec<u8> = if is_https {
        use native_tls::TlsConnector;
        let connector = TlsConnector::new().map_err(|_| NetError::ConnectFailed)?;
        let tcp = TcpStream::connect(&addr).map_err(|_| NetError::ConnectFailed)?;
        tcp.set_read_timeout(Some(Duration::from_secs(15)))
            .map_err(|_| NetError::IoError)?;
        let mut tls = connector
            .connect(&parsed.host, tcp)
            .map_err(|_| NetError::ConnectFailed)?;
        tls.write_all(&request).map_err(|_| NetError::IoError)?;
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            match tls.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(_) => break,
            }
        }
        buf
    } else {
        let mut stream = TcpStream::connect(&addr).map_err(|_| NetError::ConnectFailed)?;
        stream
            .set_read_timeout(Some(Duration::from_secs(15)))
            .map_err(|_| NetError::IoError)?;
        stream.write_all(&request).map_err(|_| NetError::IoError)?;
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(_) => break,
            }
        }
        buf
    };

    let (status, body) = parse_response(&response)?;

    // Follow 3xx redirects
    if status >= 300 && status < 400 {
        if let Some(location) = extract_location_header(&response) {
            return fetch_following_redirects(&location, depth + 1);
        }
    }

    if status >= 400 {
        return Err(NetError::HttpError(status));
    }
    Ok(body)
}

/// Parse a URL that may be http:// or https://
fn parse_url_either(url: &str) -> Result<ParsedUrlOwned, NetError> {
    let (rest, default_port) = if let Some(r) = url.strip_prefix("https://") {
        (r, 443u16)
    } else if let Some(r) = url.strip_prefix("http://") {
        (r, 80u16)
    } else {
        return Err(NetError::ParseError);
    };

    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };

    let (host, port) = match host_port.find(':') {
        Some(i) => {
            let p = parse_u16(&host_port[i + 1..]).ok_or(NetError::ParseError)?;
            (host_port[..i].to_string(), p)
        }
        None => (host_port.to_string(), default_port),
    };

    if host.is_empty() {
        return Err(NetError::ParseError);
    }

    Ok(ParsedUrlOwned {
        host,
        port,
        path: path.to_string(),
    })
}

struct ParsedUrlOwned {
    host: String,
    #[cfg_attr(not(feature = "hosted"), allow(dead_code))]
    port: u16,
    path: String,
}

/// An owned HTTP header for transport-neutral callers such as MCP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpHeader {
    pub name: String,
    pub value: String,
}

impl HttpHeader {
    pub fn new(name: &str, value: &str) -> Self {
        Self {
            name: String::from(name),
            value: String::from(value),
        }
    }
}

/// Full HTTP response used when status and headers are protocol-significant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<HttpHeader>,
    pub body: Vec<u8>,
}

/// Perform an HTTP request with caller-provided headers and a bounded response.
///
/// Hosted builds support HTTP and HTTPS. Bare-metal builds validate framing but
/// return `NoInterface` until the UEFI TCP driver is wired.
#[cfg(not(feature = "hosted"))]
pub fn request(
    method: &str,
    url: &str,
    headers: &[HttpHeader],
    body: &[u8],
    _timeout_ms: u64,
    _max_response_bytes: usize,
) -> Result<HttpResponse, NetError> {
    let parsed = parse_url_either(url)?;
    let _request = build_http_request(method, &parsed.host, &parsed.path, headers, body)?;
    Err(NetError::NoInterface)
}

#[cfg(feature = "hosted")]
pub fn request(
    method: &str,
    url: &str,
    headers: &[HttpHeader],
    body: &[u8],
    timeout_ms: u64,
    max_response_bytes: usize,
) -> Result<HttpResponse, NetError> {
    use std::io::Write;
    use std::net::TcpStream;
    use std::time::Duration;

    let parsed = parse_url_either(url)?;
    let request = build_http_request(method, &parsed.host, &parsed.path, headers, body)?;
    let address = format!("{}:{}", parsed.host, parsed.port);
    let timeout = Duration::from_millis(timeout_ms.max(1));
    let max_wire_bytes = max_response_bytes.saturating_add(64 * 1024);
    let mut response = Vec::new();

    if url.starts_with("https://") {
        use native_tls::TlsConnector;
        let connector = TlsConnector::new().map_err(|_| NetError::ConnectFailed)?;
        let tcp = TcpStream::connect(&address).map_err(|_| NetError::ConnectFailed)?;
        tcp.set_read_timeout(Some(timeout))
            .map_err(|_| NetError::IoError)?;
        tcp.set_write_timeout(Some(timeout))
            .map_err(|_| NetError::IoError)?;
        let mut stream = connector
            .connect(&parsed.host, tcp)
            .map_err(|_| NetError::ConnectFailed)?;
        stream.write_all(&request).map_err(|_| NetError::IoError)?;
        read_bounded(&mut stream, &mut response, max_wire_bytes)?;
    } else {
        let mut stream = TcpStream::connect(&address).map_err(|_| NetError::ConnectFailed)?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|_| NetError::IoError)?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(|_| NetError::IoError)?;
        stream.write_all(&request).map_err(|_| NetError::IoError)?;
        read_bounded(&mut stream, &mut response, max_wire_bytes)?;
    }

    parse_full_response(&response, max_response_bytes)
}

fn build_http_request(
    method: &str,
    host: &str,
    path: &str,
    headers: &[HttpHeader],
    body: &[u8],
) -> Result<Vec<u8>, NetError> {
    if method.is_empty()
        || !method
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b'-')
    {
        return Err(NetError::ParseError);
    }
    let mut request = format!(
        "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: genos/0.1\r\n",
        method, path, host
    )
    .into_bytes();
    let mut has_content_length = false;
    for header in headers {
        if !valid_header_name(&header.name)
            || header
                .value
                .bytes()
                .any(|byte| byte == b'\r' || byte == b'\n')
        {
            return Err(NetError::InvalidHeader);
        }
        if header.name.eq_ignore_ascii_case("Content-Length") {
            has_content_length = true;
        }
        request.extend_from_slice(header.name.as_bytes());
        request.extend_from_slice(b": ");
        request.extend_from_slice(header.value.as_bytes());
        request.extend_from_slice(b"\r\n");
    }
    if !has_content_length {
        request.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
    }
    request.extend_from_slice(b"\r\n");
    request.extend_from_slice(body);
    Ok(request)
}

fn valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

#[cfg(feature = "hosted")]
fn read_bounded<R: std::io::Read>(
    reader: &mut R,
    output: &mut Vec<u8>,
    max_wire_bytes: usize,
) -> Result<(), NetError> {
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => {
                if output.len().saturating_add(count) > max_wire_bytes {
                    return Err(NetError::ResponseTooLarge);
                }
                output.extend_from_slice(&chunk[..count]);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(NetError::Timeout)
            }
            Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {
                return Err(NetError::Timeout)
            }
            Err(_) => return Err(NetError::IoError),
        }
    }
    Ok(())
}

#[cfg(feature = "hosted")]
fn parse_full_response(data: &[u8], max_body_bytes: usize) -> Result<HttpResponse, NetError> {
    let header_end = find_header_end(data).ok_or(NetError::ParseError)?;
    let header_text =
        core::str::from_utf8(&data[..header_end]).map_err(|_| NetError::ParseError)?;
    let mut lines = header_text.lines();
    let status = parse_status_code(lines.next().ok_or(NetError::ParseError)?)?;
    let mut headers = Vec::new();
    let mut chunked = false;
    for line in lines {
        let (name, value) = line.split_once(':').ok_or(NetError::ParseError)?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("Transfer-Encoding") && value.eq_ignore_ascii_case("chunked") {
            chunked = true;
        }
        headers.push(HttpHeader::new(name.trim(), value));
    }
    let body_start = header_end + 4;
    let wire_body = data.get(body_start..).unwrap_or(&[]);
    let body = if chunked {
        decode_chunked(wire_body, max_body_bytes)?
    } else {
        if wire_body.len() > max_body_bytes {
            return Err(NetError::ResponseTooLarge);
        }
        wire_body.to_vec()
    };
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

#[cfg(feature = "hosted")]
fn decode_chunked(data: &[u8], max_body_bytes: usize) -> Result<Vec<u8>, NetError> {
    let mut position = 0usize;
    let mut output = Vec::new();
    loop {
        let line_end = find_crlf(data, position).ok_or(NetError::ParseError)?;
        let size_text =
            core::str::from_utf8(&data[position..line_end]).map_err(|_| NetError::ParseError)?;
        let size_text = size_text.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16).map_err(|_| NetError::ParseError)?;
        position = line_end + 2;
        if size == 0 {
            break;
        }
        if output.len().saturating_add(size) > max_body_bytes {
            return Err(NetError::ResponseTooLarge);
        }
        let end = position.checked_add(size).ok_or(NetError::ParseError)?;
        let chunk = data.get(position..end).ok_or(NetError::ParseError)?;
        output.extend_from_slice(chunk);
        if data.get(end..end + 2) != Some(b"\r\n") {
            return Err(NetError::ParseError);
        }
        position = end + 2;
    }
    Ok(output)
}

#[cfg(feature = "hosted")]
fn find_crlf(data: &[u8], start: usize) -> Option<usize> {
    for index in start..data.len().saturating_sub(1) {
        if data.get(index..index + 2) == Some(b"\r\n") {
            return Some(index);
        }
    }
    None
}

/// Extract the Location header from a raw HTTP response.
#[cfg(feature = "hosted")]
fn extract_location_header(response: &[u8]) -> Option<String> {
    let header_end = find_header_end(response)?;
    let headers = core::str::from_utf8(&response[..header_end]).ok()?;
    for line in headers.lines() {
        let lower = line.to_lowercase();
        if lower.starts_with("location:") {
            let loc = line[9..].trim();
            return Some(loc.to_string());
        }
    }
    None
}

/// Convenience: fetch a URL and return the body as a UTF-8 string.
pub fn fetch_text(url: &str) -> Result<String, NetError> {
    let bytes = fetch(url)?;
    String::from_utf8(bytes).map_err(|_| NetError::ParseError)
}
