//! Network HAL — bounded HTTP/1.1 over hosted sockets or UEFI HTTP.
//!
//! Implements URL parsing, request/response handling, and a `fetch()` API. The
//! firmware path uses the UEFI HTTP service binding directly so callers can
//! supply protocol-significant headers such as MCP authorization and routing.
//!
//! For QEMU testing: `make qemu-net` starts with `-netdev user` and e1000.

use alloc::format;
use alloc::string::{String, ToString};
#[cfg(not(feature = "hosted"))]
use alloc::vec;
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
#[cfg(not(feature = "hosted"))]
pub fn fetch(url: &str) -> Result<Vec<u8>, NetError> {
    let response = request("GET", url, &[], &[], 15_000, 4 * 1024 * 1024)?;
    if response.status >= 400 {
        return Err(NetError::HttpError(response.status));
    }
    Ok(response.body)
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
/// Hosted builds use TCP and the platform TLS stack. Bare-metal builds use the
/// firmware's HTTP service binding, including its HTTPS implementation and
/// trust configuration when the firmware provides those capabilities.
#[cfg(not(feature = "hosted"))]
pub fn request(
    method: &str,
    url: &str,
    headers: &[HttpHeader],
    body: &[u8],
    timeout_ms: u64,
    max_response_bytes: usize,
) -> Result<HttpResponse, NetError> {
    let parsed = parse_url_either(url)?;
    let _request = build_http_request(method, &parsed.host, &parsed.path, headers, body)?;
    firmware_request(
        method,
        url,
        &parsed.host,
        headers,
        body,
        timeout_ms,
        max_response_bytes,
    )
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
                .any(|byte| matches!(byte, b'\0' | b'\r' | b'\n'))
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
        && name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

#[cfg(not(feature = "hosted"))]
fn firmware_request(
    method: &str,
    url: &str,
    host: &str,
    headers: &[HttpHeader],
    body: &[u8],
    timeout_ms: u64,
    max_response_bytes: usize,
) -> Result<HttpResponse, NetError> {
    use uefi::boot;
    use uefi::proto::network::http::HttpBinding;

    let handles = boot::find_handles::<HttpBinding>().map_err(|_| NetError::NoInterface)?;
    let mut last_error = NetError::NoInterface;
    for handle in handles {
        let result = FirmwareHttp::new(handle).and_then(|mut http| {
            http.configure(timeout_ms)?;
            http.send(method, url, host, headers, body)?;
            http.receive(max_response_bytes)
        });
        match result {
            Ok(response) => return Ok(response),
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

#[cfg(not(feature = "hosted"))]
struct FirmwareHttp {
    child_handle: uefi::Handle,
    binding: uefi::boot::ScopedProtocol<uefi::proto::network::http::HttpBinding>,
    protocol: Option<uefi::boot::ScopedProtocol<uefi::proto::network::http::Http>>,
}

#[cfg(not(feature = "hosted"))]
impl FirmwareHttp {
    fn new(nic_handle: uefi::Handle) -> Result<Self, NetError> {
        use uefi::boot::{self, OpenProtocolAttributes, OpenProtocolParams};
        use uefi::proto::network::http::{Http, HttpBinding};

        // SAFETY: both protocols are opened with the current image as agent,
        // retained in ScopedProtocol values, and closed before child teardown.
        let mut binding = unsafe {
            boot::open_protocol::<HttpBinding>(
                OpenProtocolParams {
                    handle: nic_handle,
                    agent: boot::image_handle(),
                    controller: None,
                },
                OpenProtocolAttributes::GetProtocol,
            )
        }
        .map_err(|_| NetError::NoInterface)?;
        let child_handle = binding
            .create_child()
            .map_err(|_| NetError::ConnectFailed)?;
        // SAFETY: child_handle was just created by this binding and remains
        // alive until FirmwareHttp::drop destroys it.
        let protocol = unsafe {
            boot::open_protocol::<Http>(
                OpenProtocolParams {
                    handle: child_handle,
                    agent: boot::image_handle(),
                    controller: None,
                },
                OpenProtocolAttributes::GetProtocol,
            )
        };
        match protocol {
            Ok(protocol) => Ok(Self {
                child_handle,
                binding,
                protocol: Some(protocol),
            }),
            Err(_) => {
                let _ = binding.destroy_child(child_handle);
                Err(NetError::ConnectFailed)
            }
        }
    }

    fn configure(&mut self, timeout_ms: u64) -> Result<(), NetError> {
        use uefi_raw::protocol::network::http::{
            HttpAccessPoint, HttpConfigData, HttpV4AccessPoint, HttpVersion,
        };

        let ipv4 = HttpV4AccessPoint {
            use_default_addr: true.into(),
            ..Default::default()
        };
        let config = HttpConfigData {
            http_version: HttpVersion::HTTP_VERSION_11,
            time_out_millisec: timeout_ms.max(1).min(u32::MAX as u64) as u32,
            local_addr_is_ipv6: false.into(),
            access_point: HttpAccessPoint { ipv4_node: &ipv4 },
        };
        self.protocol
            .as_mut()
            .ok_or(NetError::NoInterface)?
            .configure(&config)
            .map_err(|_| NetError::ConnectFailed)
    }

    fn send(
        &mut self,
        method: &str,
        url: &str,
        host: &str,
        headers: &[HttpHeader],
        body: &[u8],
    ) -> Result<(), NetError> {
        use core::ffi::c_void;
        use uefi::Status;
        use uefi_raw::protocol::network::http::{
            HttpHeader as RawHeader, HttpMessage, HttpRequestData, HttpToken,
        };

        let method = firmware_method(method)?;
        let url = uefi::CString16::try_from(url).map_err(|_| NetError::ParseError)?;
        let mut header_storage: Vec<Vec<u8>> = Vec::new();
        let mut push_header = |name: &str, value: &str| {
            let mut name_bytes = name.as_bytes().to_vec();
            name_bytes.push(0);
            header_storage.push(name_bytes);
            let mut value_bytes = value.as_bytes().to_vec();
            value_bytes.push(0);
            header_storage.push(value_bytes);
        };
        if !headers
            .iter()
            .any(|header| header.name.eq_ignore_ascii_case("Host"))
        {
            push_header("Host", host);
        }
        for header in headers {
            push_header(&header.name, &header.value);
        }
        if !headers
            .iter()
            .any(|header| header.name.eq_ignore_ascii_case("Content-Length"))
        {
            push_header("Content-Length", &body.len().to_string());
        }
        drop(push_header);

        let mut raw_headers = Vec::with_capacity(header_storage.len() / 2);
        for pair in header_storage.chunks_exact(2) {
            raw_headers.push(RawHeader {
                field_name: pair[0].as_ptr(),
                field_value: pair[1].as_ptr(),
            });
        }
        let mut request_data = HttpRequestData {
            method,
            url: url.as_ptr().cast::<u16>(),
        };
        let mut body = body.to_vec();
        let mut message = HttpMessage::default();
        message.data.request = &mut request_data;
        message.header_count = raw_headers.len();
        message.header = raw_headers.as_mut_ptr();
        message.body_length = body.len();
        message.body = if body.is_empty() {
            core::ptr::null_mut()
        } else {
            body.as_mut_ptr().cast::<c_void>()
        };
        let mut token = HttpToken {
            status: Status::NOT_READY,
            message: &mut message,
            ..Default::default()
        };
        let protocol = self.protocol.as_mut().ok_or(NetError::NoInterface)?;
        protocol
            .request(&mut token)
            .map_err(|_| NetError::ConnectFailed)?;
        while token.status == Status::NOT_READY {
            protocol.poll().map_err(|_| NetError::IoError)?;
        }
        if token.status == Status::TIMEOUT {
            return Err(NetError::Timeout);
        }
        if token.status != Status::SUCCESS {
            return Err(NetError::ConnectFailed);
        }
        Ok(())
    }

    fn receive(&mut self, max_response_bytes: usize) -> Result<HttpResponse, NetError> {
        use core::ffi::{c_char, c_void, CStr};
        use uefi::Status;
        use uefi_raw::protocol::network::http::{
            HttpMessage, HttpResponseData, HttpStatusCode, HttpToken,
        };

        let first_capacity = max_response_bytes.saturating_add(1).min(16 * 1024).max(1);
        let mut first_body = vec![0; first_capacity];
        let mut response_data = HttpResponseData {
            status_code: HttpStatusCode::STATUS_UNSUPPORTED,
        };
        let mut message = HttpMessage::default();
        message.data.response = &mut response_data;
        message.body_length = first_body.len();
        message.body = first_body.as_mut_ptr().cast::<c_void>();
        let mut token = HttpToken {
            status: Status::NOT_READY,
            message: &mut message,
            ..Default::default()
        };
        let protocol = self.protocol.as_mut().ok_or(NetError::NoInterface)?;
        protocol
            .response(&mut token)
            .map_err(|_| NetError::IoError)?;
        while token.status == Status::NOT_READY {
            protocol.poll().map_err(|_| NetError::IoError)?;
        }
        if token.status == Status::TIMEOUT {
            return Err(NetError::Timeout);
        }
        if token.status != Status::SUCCESS && token.status != Status::HTTP_ERROR {
            return Err(NetError::IoError);
        }

        let status = status_code(response_data.status_code).ok_or(NetError::ParseError)?;
        let mut headers = Vec::new();
        for index in 0..message.header_count {
            // SAFETY: UEFI owns the response header array for the duration of
            // this call and reports its exact element count.
            let raw = unsafe { &*message.header.add(index) };
            // SAFETY: UEFI HTTP header fields are NUL-terminated ASCII strings.
            let name = unsafe { CStr::from_ptr(raw.field_name.cast::<c_char>()) }
                .to_str()
                .map_err(|_| NetError::ParseError)?;
            let value = unsafe { CStr::from_ptr(raw.field_value.cast::<c_char>()) }
                .to_str()
                .map_err(|_| NetError::ParseError)?;
            headers.push(HttpHeader::new(name, value));
        }
        if message.body_length > first_body.len() {
            return Err(NetError::IoError);
        }
        first_body.truncate(message.body_length);
        if first_body.len() > max_response_bytes {
            return Err(NetError::ResponseTooLarge);
        }
        let mut body = first_body;

        loop {
            let capacity = max_response_bytes
                .saturating_sub(body.len())
                .saturating_add(1)
                .min(16 * 1024)
                .max(1);
            let mut chunk = vec![0; capacity];
            let mut more = HttpMessage {
                body_length: chunk.len(),
                body: chunk.as_mut_ptr().cast::<c_void>(),
                ..Default::default()
            };
            let mut more_token = HttpToken {
                status: Status::NOT_READY,
                message: &mut more,
                ..Default::default()
            };
            protocol
                .response(&mut more_token)
                .map_err(|_| NetError::IoError)?;
            while more_token.status == Status::NOT_READY {
                protocol.poll().map_err(|_| NetError::IoError)?;
            }
            if more_token.status == Status::TIMEOUT {
                return Err(NetError::Timeout);
            }
            if more_token.status != Status::SUCCESS {
                return Err(NetError::IoError);
            }
            if more.body_length == 0 {
                break;
            }
            if more.body_length > chunk.len()
                || body.len().saturating_add(more.body_length) > max_response_bytes
            {
                return Err(NetError::ResponseTooLarge);
            }
            body.extend_from_slice(&chunk[..more.body_length]);
        }

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

#[cfg(not(feature = "hosted"))]
impl Drop for FirmwareHttp {
    fn drop(&mut self) {
        self.protocol = None;
        let _ = self.binding.destroy_child(self.child_handle);
    }
}

#[cfg(not(feature = "hosted"))]
fn firmware_method(
    method: &str,
) -> Result<uefi_raw::protocol::network::http::HttpMethod, NetError> {
    use uefi_raw::protocol::network::http::HttpMethod;

    match method {
        "GET" => Ok(HttpMethod::GET),
        "POST" => Ok(HttpMethod::POST),
        "PATCH" => Ok(HttpMethod::PATCH),
        "OPTIONS" => Ok(HttpMethod::OPTIONS),
        "CONNECT" => Ok(HttpMethod::CONNECT),
        "HEAD" => Ok(HttpMethod::HEAD),
        "PUT" => Ok(HttpMethod::PUT),
        "DELETE" => Ok(HttpMethod::DELETE),
        "TRACE" => Ok(HttpMethod::TRACE),
        _ => Err(NetError::ParseError),
    }
}

#[cfg(not(feature = "hosted"))]
fn status_code(status: uefi_raw::protocol::network::http::HttpStatusCode) -> Option<u16> {
    use uefi_raw::protocol::network::http::HttpStatusCode as S;

    Some(match status {
        S::STATUS_100_CONTINUE => 100,
        S::STATUS_101_SWITCHING_PROTOCOLS => 101,
        S::STATUS_200_OK => 200,
        S::STATUS_201_CREATED => 201,
        S::STATUS_202_ACCEPTED => 202,
        S::STATUS_203_NON_AUTHORITATIVE_INFORMATION => 203,
        S::STATUS_204_NO_CONTENT => 204,
        S::STATUS_205_RESET_CONTENT => 205,
        S::STATUS_206_PARTIAL_CONTENT => 206,
        S::STATUS_300_MULTIPLE_CHOICES => 300,
        S::STATUS_301_MOVED_PERMANENTLY => 301,
        S::STATUS_302_FOUND => 302,
        S::STATUS_303_SEE_OTHER => 303,
        S::STATUS_304_NOT_MODIFIED => 304,
        S::STATUS_305_USE_PROXY => 305,
        S::STATUS_307_TEMPORARY_REDIRECT => 307,
        S::STATUS_308_PERMANENT_REDIRECT => 308,
        S::STATUS_400_BAD_REQUEST => 400,
        S::STATUS_401_UNAUTHORIZED => 401,
        S::STATUS_402_PAYMENT_REQUIRED => 402,
        S::STATUS_403_FORBIDDEN => 403,
        S::STATUS_404_NOT_FOUND => 404,
        S::STATUS_405_METHOD_NOT_ALLOWED => 405,
        S::STATUS_406_NOT_ACCEPTABLE => 406,
        S::STATUS_407_PROXY_AUTHENTICATION_REQUIRED => 407,
        S::STATUS_408_REQUEST_TIME_OUT => 408,
        S::STATUS_409_CONFLICT => 409,
        S::STATUS_410_GONE => 410,
        S::STATUS_411_LENGTH_REQUIRED => 411,
        S::STATUS_412_PRECONDITION_FAILED => 412,
        S::STATUS_413_REQUEST_ENTITY_TOO_LARGE => 413,
        S::STATUS_414_REQUEST_URI_TOO_LARGE => 414,
        S::STATUS_415_UNSUPPORTED_MEDIA_TYPE => 415,
        S::STATUS_416_REQUESTED_RANGE_NOT_SATISFIED => 416,
        S::STATUS_417_EXPECTATION_FAILED => 417,
        S::STATUS_500_INTERNAL_SERVER_ERROR => 500,
        S::STATUS_501_NOT_IMPLEMENTED => 501,
        S::STATUS_502_BAD_GATEWAY => 502,
        S::STATUS_503_SERVICE_UNAVAILABLE => 503,
        S::STATUS_504_GATEWAY_TIME_OUT => 504,
        S::STATUS_505_VERSION_NOT_SUPPORTED => 505,
        _ => return None,
    })
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
