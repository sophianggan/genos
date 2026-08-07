//! Adapter from the MCP transport contract to the GenOS network HAL.

use alloc::string::String;
use alloc::vec::Vec;
use genos_hal::net;
use genos_mcp::transport::{
    Header, HttpRequest, HttpResponse, HttpTransport, TransportError, TransportErrorKind,
};

pub struct HalMcpTransport;

impl HttpTransport for HalMcpTransport {
    fn send(&mut self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        let headers: Vec<net::HttpHeader> = request
            .headers
            .iter()
            .map(|header| net::HttpHeader::new(&header.name, &header.value))
            .collect();
        let response = net::request(
            "POST",
            &request.url,
            &headers,
            &request.body,
            request.timeout_ms,
            request.max_response_bytes,
        )
        .map_err(map_error)?;
        Ok(HttpResponse {
            status: response.status,
            headers: response
                .headers
                .into_iter()
                .map(|header| Header::public(&header.name, &header.value))
                .collect(),
            body: response.body,
        })
    }
}

fn map_error(error: net::NetError) -> TransportError {
    let kind = match error {
        net::NetError::DnsError => TransportErrorKind::Dns,
        net::NetError::ConnectFailed | net::NetError::NoInterface => TransportErrorKind::Connect,
        net::NetError::Timeout => TransportErrorKind::Timeout,
        net::NetError::ResponseTooLarge => TransportErrorKind::ResponseTooLarge,
        net::NetError::ParseError | net::NetError::InvalidHeader => {
            TransportErrorKind::InvalidResponse
        }
        net::NetError::HttpError(_) => TransportErrorKind::HttpStatus,
        net::NetError::IoError => TransportErrorKind::Io,
    };
    let retryable = matches!(
        kind,
        TransportErrorKind::Dns
            | TransportErrorKind::Connect
            | TransportErrorKind::Timeout
            | TransportErrorKind::Io
    );
    TransportError::new(kind, error.as_str(), retryable)
}

/// Produce a safe request summary without serializing sensitive values.
pub fn request_summary(request: &HttpRequest) -> String {
    let mut output = String::from("POST ");
    output.push_str(&request.url);
    output.push_str(" headers=[");
    for (index, header) in request.headers.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&header.name);
        output.push('=');
        output.push_str(header.redacted_value());
    }
    output.push(']');
    output
}
