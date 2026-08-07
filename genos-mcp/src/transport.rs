//! Transport boundary and MCP Streamable HTTP framing.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use genos_kernel::json;

use crate::config::ServerConfig;
use crate::wire::{McpError, RpcRequest, RpcResponse, LATEST_PROTOCOL_VERSION};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub name: String,
    pub value: String,
    pub sensitive: bool,
}

impl Header {
    pub fn public(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            sensitive: false,
        }
    }

    pub fn sensitive(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            sensitive: true,
        }
    }

    pub fn redacted_value(&self) -> &str {
        if self.sensitive {
            "[redacted]"
        } else {
            &self.value
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<Header>,
    pub body: Vec<u8>,
    pub timeout_ms: u64,
    pub max_response_bytes: usize,
}

impl HttpRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .map(|header| header.value.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<Header>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .map(|header| header.value.as_str())
    }
}

pub trait HttpTransport {
    fn send(&mut self, request: &HttpRequest) -> Result<HttpResponse, TransportError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportErrorKind {
    InvalidEndpoint,
    InsecureEndpoint,
    Dns,
    Connect,
    Tls,
    Timeout,
    Cancelled,
    ResponseTooLarge,
    UnsupportedContentType,
    HttpStatus,
    InvalidResponse,
    Io,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportError {
    pub kind: TransportErrorKind,
    pub message: String,
    pub retryable: bool,
    pub status: Option<u16>,
}

impl TransportError {
    pub fn new(kind: TransportErrorKind, message: &str, retryable: bool) -> Self {
        Self {
            kind,
            message: message.to_string(),
            retryable,
            status: None,
        }
    }

    pub fn http_status(status: u16) -> Self {
        Self {
            kind: TransportErrorKind::HttpStatus,
            message: String::from("MCP endpoint returned an HTTP error"),
            retryable: status == 408 || status == 429 || status >= 500,
            status: Some(status),
        }
    }
}

pub struct McpHttpClient<T> {
    transport: T,
}

impl<T: HttpTransport> McpHttpClient<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    pub fn exchange(
        &mut self,
        server: &ServerConfig,
        rpc: &RpcRequest,
        auth_headers: &[Header],
    ) -> Result<RpcResponse, TransportError> {
        let request = build_http_request(server, rpc, auth_headers)?;
        let response = self.transport.send(&request)?;
        decode_http_response(&response, request.max_response_bytes)
    }

    pub fn into_inner(self) -> T {
        self.transport
    }
}

pub fn build_http_request(
    server: &ServerConfig,
    rpc: &RpcRequest,
    auth_headers: &[Header],
) -> Result<HttpRequest, TransportError> {
    validate_endpoint(&server.endpoint, !auth_headers.is_empty())?;
    for header in auth_headers {
        validate_header(header)?;
    }

    let mut headers = Vec::new();
    headers.push(Header::public("Content-Type", "application/json"));
    headers.push(Header::public(
        "Accept",
        "application/json, text/event-stream",
    ));
    headers.push(Header::public(
        "MCP-Protocol-Version",
        rpc.protocol_version().unwrap_or(LATEST_PROTOCOL_VERSION),
    ));
    headers.push(Header::public("Mcp-Method", &rpc.method));
    if let Some(name) = rpc.params.get("name").and_then(|value| value.as_str()) {
        headers.push(Header::public("Mcp-Name", name));
    }
    headers.extend_from_slice(auth_headers);

    Ok(HttpRequest {
        url: server.endpoint.clone(),
        headers,
        body: rpc.to_json_string().into_bytes(),
        timeout_ms: server.timeout_ms,
        max_response_bytes: server.max_response_bytes,
    })
}

pub fn decode_http_response(
    response: &HttpResponse,
    max_response_bytes: usize,
) -> Result<RpcResponse, TransportError> {
    if response.body.len() > max_response_bytes {
        return Err(TransportError::new(
            TransportErrorKind::ResponseTooLarge,
            "MCP response exceeded configured limit",
            false,
        ));
    }
    if !(200..300).contains(&response.status) {
        return Err(TransportError::http_status(response.status));
    }
    if let Some(content_type) = response.header("Content-Type") {
        let content_type = content_type.split(';').next().unwrap_or("").trim();
        if content_type != "application/json" {
            return Err(TransportError::new(
                TransportErrorKind::UnsupportedContentType,
                "expected application/json MCP response",
                false,
            ));
        }
    }
    let body = core::str::from_utf8(&response.body).map_err(|_| {
        TransportError::new(
            TransportErrorKind::InvalidResponse,
            "MCP response was not UTF-8",
            false,
        )
    })?;
    let value = json::parse(body).map_err(|_| {
        TransportError::new(
            TransportErrorKind::InvalidResponse,
            "MCP response was not valid JSON",
            false,
        )
    })?;
    RpcResponse::from_json(&value).map_err(protocol_error)
}

fn protocol_error(error: McpError) -> TransportError {
    TransportError::new(TransportErrorKind::InvalidResponse, &error.message, false)
}

fn validate_endpoint(url: &str, authenticated: bool) -> Result<(), TransportError> {
    if url.starts_with("https://") {
        return Ok(());
    }
    if let Some(rest) = url.strip_prefix("http://") {
        let authority = rest.split('/').next().unwrap_or("");
        let host = authority
            .strip_prefix('[')
            .and_then(|value| value.split(']').next())
            .unwrap_or_else(|| authority.split(':').next().unwrap_or(""));
        if host == "localhost" || host == "127.0.0.1" || host == "::1" {
            return Ok(());
        }
        let message = if authenticated {
            "credentials may only be sent to HTTPS or loopback endpoints"
        } else {
            "remote MCP endpoints must use HTTPS"
        };
        return Err(TransportError::new(
            TransportErrorKind::InsecureEndpoint,
            message,
            false,
        ));
    }
    Err(TransportError::new(
        TransportErrorKind::InvalidEndpoint,
        "MCP endpoint must use https:// or loopback http://",
        false,
    ))
}

fn validate_header(header: &Header) -> Result<(), TransportError> {
    let valid_name = !header.name.is_empty()
        && header
            .name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
    let valid_value = !header
        .value
        .bytes()
        .any(|byte| byte == b'\r' || byte == b'\n');
    if !valid_name || !valid_value {
        return Err(TransportError::new(
            TransportErrorKind::InvalidEndpoint,
            "invalid authentication header",
            false,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;

    use super::*;
    use crate::config::{AuthConfig, TransportKind, TrustLevel};
    use crate::wire::{method, ClientIdentity, RequestId, RequestMetadata};
    use genos_kernel::json::JsonValue;

    fn server(endpoint: &str) -> ServerConfig {
        ServerConfig {
            id: String::from("test"),
            transport: TransportKind::StreamableHttp,
            endpoint: endpoint.to_string(),
            enabled: true,
            auth: AuthConfig::None,
            trust: TrustLevel::Untrusted,
            allow_tools: true,
            allow_resources: true,
            allow_prompts: true,
            require_approval: true,
            timeout_ms: 1000,
            max_response_bytes: 4096,
            allowed_tools: Vec::new(),
        }
    }

    #[test]
    fn modern_http_request_has_routing_headers() {
        let metadata = RequestMetadata::modern(ClientIdentity::genos());
        let mut params = BTreeMap::new();
        params.insert(String::from("name"), JsonValue::Str(String::from("search")));
        let rpc = RpcRequest::new(
            RequestId::Number(7),
            method::TOOLS_CALL,
            JsonValue::Object(params),
            &metadata,
        );
        let request = build_http_request(&server("https://example.com/mcp"), &rpc, &[]).unwrap();
        assert_eq!(request.header("Mcp-Method"), Some("tools/call"));
        assert_eq!(request.header("Mcp-Name"), Some("search"));
        assert_eq!(
            request.header("MCP-Protocol-Version"),
            Some(LATEST_PROTOCOL_VERSION)
        );
    }

    #[test]
    fn refuses_remote_cleartext_and_header_injection() {
        let metadata = RequestMetadata::modern(ClientIdentity::genos());
        let rpc = RpcRequest::discover(RequestId::Number(1), &metadata);
        let error = build_http_request(&server("http://example.com/mcp"), &rpc, &[]).unwrap_err();
        assert_eq!(error.kind, TransportErrorKind::InsecureEndpoint);

        let injected = Header::sensitive("Authorization", "Bearer ok\r\nX-Evil: yes");
        assert!(build_http_request(&server("https://example.com/mcp"), &rpc, &[injected]).is_err());
    }
}
