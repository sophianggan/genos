use genos_kernel::json;
use genos_mcp::{RpcRequest, RpcResponse};
use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::str::FromStr;

const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

fn main() {
    let config = match BridgeConfig::from_args(env::args().skip(1).collect()) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("error: {message}");
            print_usage();
            std::process::exit(2);
        }
    };
    let listener = TcpListener::bind(config.listen).unwrap_or_else(|error| {
        eprintln!("error: cannot bind {}: {error}", config.listen);
        std::process::exit(1);
    });
    let mut child = StdioPeer::spawn(&config.command).unwrap_or_else(|error| {
        eprintln!("error: cannot start MCP server: {error}");
        std::process::exit(1);
    });
    eprintln!(
        "genos-mcp-bridge listening on http://{}/mcp -> {:?}",
        config.listen, config.command
    );

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(error) = handle_connection(&mut stream, &mut child) {
                    let _ = write_error(&mut stream, 502, "bridge_error", &error);
                }
            }
            Err(error) => eprintln!("connection error: {error}"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct BridgeConfig {
    listen: SocketAddr,
    command: Vec<String>,
}

impl BridgeConfig {
    fn from_args(args: Vec<String>) -> Result<Self, String> {
        let mut listen = SocketAddr::from_str("127.0.0.1:8787").unwrap();
        let mut index = 0usize;
        while index < args.len() {
            match args[index].as_str() {
                "--listen" => {
                    index += 1;
                    listen = args
                        .get(index)
                        .ok_or_else(|| String::from("--listen requires an address"))?
                        .parse()
                        .map_err(|_| String::from("invalid --listen address"))?;
                    index += 1;
                }
                "--" => {
                    let command = args[index + 1..].to_vec();
                    return Self::validated(listen, command);
                }
                argument => return Err(format!("unknown bridge argument: {argument}")),
            }
        }
        Err(String::from("missing MCP server command after --"))
    }

    fn validated(listen: SocketAddr, command: Vec<String>) -> Result<Self, String> {
        if command.is_empty() {
            return Err(String::from("MCP server command cannot be empty"));
        }
        if !is_loopback(listen.ip()) {
            return Err(String::from(
                "the built-in bridge may only bind loopback; use a TLS reverse proxy for remote access",
            ));
        }
        Ok(Self { listen, command })
    }
}

fn is_loopback(address: IpAddr) -> bool {
    address.is_loopback()
}

struct StdioPeer {
    _child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl StdioPeer {
    fn spawn(command: &[String]) -> Result<Self, String> {
        let mut child = Command::new(&command[0])
            .args(&command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| error.to_string())?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| String::from("child stdin unavailable"))?;
        let output = child
            .stdout
            .take()
            .map(BufReader::new)
            .ok_or_else(|| String::from("child stdout unavailable"))?;
        Ok(Self {
            _child: child,
            input,
            output,
        })
    }

    fn exchange(&mut self, request: &RpcRequest) -> Result<String, String> {
        let request_id = request.id.clone();
        self.input
            .write_all(request.to_json_string().as_bytes())
            .and_then(|_| self.input.write_all(b"\n"))
            .and_then(|_| self.input.flush())
            .map_err(|error| error.to_string())?;

        let mut total = 0usize;
        for _ in 0..1_000 {
            let mut line = String::new();
            let count = self
                .output
                .read_line(&mut line)
                .map_err(|error| error.to_string())?;
            if count == 0 {
                return Err(String::from("MCP server closed stdout"));
            }
            total = total.saturating_add(count);
            if total > MAX_RESPONSE_BYTES {
                return Err(String::from("MCP stdio response exceeded 4 MiB"));
            }
            let value = match json::parse(line.trim()) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let response = match RpcResponse::from_json(&value) {
                Ok(response) => response,
                // Notifications and server-to-client requests are not the
                // response to this HTTP request. Modern MRTR carries input
                // requests inside the eventual result instead.
                Err(_) => continue,
            };
            if response.id == request_id {
                return Ok(response.to_json().to_json_string());
            }
        }
        Err(String::from("no matching MCP response after 1000 messages"))
    }
}

fn handle_connection(stream: &mut TcpStream, peer: &mut StdioPeer) -> Result<(), String> {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(30)))
        .map_err(|error| error.to_string())?;
    let request = read_http_request(stream)?;
    if request.method == "GET" && request.path == "/healthz" {
        return write_http(stream, 200, "text/plain", b"ok\n");
    }
    if request.method != "POST" || request.path != "/mcp" {
        return write_error(stream, 404, "not_found", "expected POST /mcp");
    }
    validate_origin(&request.headers)?;
    let body = std::str::from_utf8(&request.body)
        .map_err(|_| String::from("request body must be UTF-8"))?;
    let value = json::parse(body).map_err(|_| String::from("request body must be JSON"))?;
    let rpc = RpcRequest::from_json(&value).map_err(|error| error.message)?;
    validate_routing_headers(&request.headers, &rpc)?;
    let response = peer.exchange(&rpc)?;
    write_http(stream, 200, "application/json", response.as_bytes())
}

struct IncomingRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut TcpStream) -> Result<IncomingRequest, String> {
    let mut data = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let count = stream.read(&mut chunk).map_err(|error| error.to_string())?;
        if count == 0 {
            return Err(String::from("connection closed before request headers"));
        }
        data.extend_from_slice(&chunk[..count]);
        if data.len() > 64 * 1024 {
            return Err(String::from("request headers exceeded 64 KiB"));
        }
        if let Some(index) = find_bytes(&data, b"\r\n\r\n") {
            break index;
        }
    };
    let header_text = std::str::from_utf8(&data[..header_end])
        .map_err(|_| String::from("request headers must be UTF-8"))?;
    let mut lines = header_text.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| String::from("request line is missing"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| String::from("HTTP method is missing"))?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| String::from("HTTP path is missing"))?
        .to_string();
    if request_parts.next() != Some("HTTP/1.1") || request_parts.next().is_some() {
        return Err(String::from("expected HTTP/1.1 request"));
    }
    let mut headers = BTreeMap::new();
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| String::from("malformed HTTP header"))?;
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
    }
    let content_length = headers
        .get("content-length")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| String::from("invalid Content-Length"))?
        .unwrap_or(0);
    if content_length > MAX_REQUEST_BYTES {
        return Err(String::from("request body exceeded 4 MiB"));
    }
    let body_start = header_end + 4;
    while data.len().saturating_sub(body_start) < content_length {
        let count = stream.read(&mut chunk).map_err(|error| error.to_string())?;
        if count == 0 {
            return Err(String::from("connection closed before request body"));
        }
        data.extend_from_slice(&chunk[..count]);
        if data.len().saturating_sub(body_start) > MAX_REQUEST_BYTES {
            return Err(String::from("request body exceeded 4 MiB"));
        }
    }
    Ok(IncomingRequest {
        method,
        path,
        headers,
        body: data[body_start..body_start + content_length].to_vec(),
    })
}

fn validate_origin(headers: &BTreeMap<String, String>) -> Result<(), String> {
    if let Some(origin) = headers.get("origin") {
        if origin != "http://127.0.0.1" && origin != "http://localhost" {
            return Err(String::from("Origin is not allowed"));
        }
    }
    Ok(())
}

fn validate_routing_headers(
    headers: &BTreeMap<String, String>,
    request: &RpcRequest,
) -> Result<(), String> {
    let version = headers
        .get("mcp-protocol-version")
        .ok_or_else(|| String::from("MCP-Protocol-Version header is required"))?;
    if Some(version.as_str()) != request.protocol_version() {
        return Err(String::from("protocol version header/body mismatch"));
    }
    let method = headers
        .get("mcp-method")
        .ok_or_else(|| String::from("Mcp-Method header is required"))?;
    if method != &request.method {
        return Err(String::from("method header/body mismatch"));
    }
    let body_name = request.params.get("name").and_then(|value| value.as_str());
    let header_name = headers.get("mcp-name").map(String::as_str);
    if body_name != header_name {
        return Err(String::from("name header/body mismatch"));
    }
    Ok(())
}

fn write_http(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        502 => "Bad Gateway",
        _ => "Error",
    };
    let headers = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n",
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(body))
        .and_then(|_| stream.flush())
        .map_err(|error| error.to_string())
}

fn write_error(
    stream: &mut TcpStream,
    status: u16,
    code: &str,
    message: &str,
) -> Result<(), String> {
    let body = format!(
        "{{\"error\":{{\"code\":\"{}\",\"message\":\"{}\"}}}}",
        json_escape(code),
        json_escape(message)
    );
    write_http(stream, status, "application/json", body.as_bytes())
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn print_usage() {
    eprintln!("usage: genos-mcp-bridge [--listen 127.0.0.1:8787] -- COMMAND [ARGS...]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_safe_loopback_configuration() {
        let config = BridgeConfig::from_args(vec![
            String::from("--listen"),
            String::from("127.0.0.1:9000"),
            String::from("--"),
            String::from("npx"),
            String::from("server"),
        ])
        .unwrap();
        assert_eq!(config.listen.port(), 9000);
        assert_eq!(config.command[0], "npx");
    }

    #[test]
    fn refuses_non_loopback_bind() {
        assert!(BridgeConfig::from_args(vec![
            String::from("--listen"),
            String::from("0.0.0.0:8787"),
            String::from("--"),
            String::from("server"),
        ])
        .is_err());
    }

    #[test]
    fn routing_headers_must_match_body() {
        let value = json::parse(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search","_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"genos","version":"1"},"io.modelcontextprotocol/clientCapabilities":{}}}}"#,
        )
        .unwrap();
        let request = RpcRequest::from_json(&value).unwrap();
        let mut headers = BTreeMap::new();
        headers.insert(
            String::from("mcp-protocol-version"),
            String::from("2026-07-28"),
        );
        headers.insert(String::from("mcp-method"), String::from("tools/call"));
        headers.insert(String::from("mcp-name"), String::from("other"));
        assert!(validate_routing_headers(&headers, &request).is_err());
    }
}
