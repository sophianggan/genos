//! MCP host runtime integrated with the GenOS agent.

use alloc::format;
use alloc::string::{String, ToString};
use genos_hal::disk;
use genos_kernel::json::JsonValue;
use genos_mcp::client::ClientStatus;
use genos_mcp::execution::{ListKind, McpHostError};
use genos_mcp::security::{CredentialError, CredentialResolver, Secret};
use genos_mcp::{
    resolve_auth, ClientManager, GuardedResult, McpConfig, McpHttpClient, PolicyDecision,
    PolicyFirewall, Provenance,
};
use genos_tools::mcp_transport::HalMcpTransport;
use genos_tools::protocol::ToolResult;

pub struct McpRuntime {
    session_id: String,
    manager: ClientManager,
    http: McpHttpClient<HalMcpTransport>,
    firewall: PolicyFirewall,
    config_error: Option<String>,
}

impl McpRuntime {
    pub fn load(session_id: &str) -> Self {
        let (config, config_error) = match disk::read_file("\\system\\mcp.toml") {
            Ok(bytes) => match McpConfig::parse(&bytes) {
                Ok(config) => (config, None),
                Err(error) => (
                    McpConfig::empty(),
                    Some(format!("line {}: {}", error.line, error.message)),
                ),
            },
            Err(_) => (McpConfig::empty(), None),
        };
        Self {
            session_id: session_id.to_string(),
            manager: ClientManager::from_config(&config),
            http: McpHttpClient::new(HalMcpTransport),
            firewall: PolicyFirewall::new(),
            config_error,
        }
    }

    pub fn server_count(&self) -> usize {
        self.manager.len()
    }

    pub fn reset_turn(&mut self) {
        self.firewall.reset_turn();
    }

    pub fn servers(&self, call_id: &str) -> ToolResult {
        let servers = self
            .manager
            .iter()
            .map(|(id, client)| {
                object(&[
                    ("id", JsonValue::Str(id.to_string())),
                    ("endpoint", JsonValue::Str(client.server.endpoint.clone())),
                    (
                        "status",
                        JsonValue::Str(String::from(status_name(client.status))),
                    ),
                    (
                        "authenticated",
                        JsonValue::Bool(client.server.is_authenticated()),
                    ),
                ])
            })
            .collect();
        let mut values = alloc::collections::BTreeMap::new();
        values.insert(String::from("servers"), JsonValue::Array(servers));
        if let Some(error) = &self.config_error {
            values.insert(String::from("config_error"), JsonValue::Str(error.clone()));
        }
        ToolResult::success(call_id, JsonValue::Object(values), 0)
    }

    pub fn connect(&mut self, args: &JsonValue, call_id: &str) -> ToolResult {
        let server_id = match required_string(args, "server") {
            Ok(value) => value,
            Err(message) => return ToolResult::failure(call_id, "invalid_args", message, false),
        };
        match self.connect_server(server_id) {
            Ok(discovered) => ToolResult::success(
                call_id,
                object(&[
                    ("server", JsonValue::Str(server_id.to_string())),
                    ("connected", JsonValue::Bool(true)),
                    ("catalog_items", JsonValue::Number(discovered as f64)),
                ]),
                0,
            ),
            Err((code, message, retryable)) => {
                ToolResult::failure(call_id, code, &message, retryable)
            }
        }
    }

    pub fn tools(&self, args: &JsonValue, call_id: &str) -> ToolResult {
        let server_filter = args.get("server").and_then(JsonValue::as_str);
        let tools = self
            .manager
            .catalog()
            .tools()
            .into_iter()
            .filter(|entry| {
                server_filter
                    .map(|server| server == entry.server_id)
                    .unwrap_or(true)
            })
            .map(|entry| {
                object(&[
                    ("name", JsonValue::Str(entry.exposed_name)),
                    ("server", JsonValue::Str(entry.server_id)),
                    ("schema", entry.tool.input_schema),
                    (
                        "description",
                        entry
                            .tool
                            .description
                            .map(JsonValue::Str)
                            .unwrap_or(JsonValue::Null),
                    ),
                ])
            })
            .collect();
        ToolResult::success(call_id, JsonValue::Array(tools), 0)
    }

    pub fn call(&mut self, args: &JsonValue, call_id: &str, now_ms: u64) -> ToolResult {
        let server_id = match required_string(args, "server") {
            Ok(value) => value,
            Err(message) => return ToolResult::failure(call_id, "invalid_args", message, false),
        };
        let tool_name = match required_string(args, "name") {
            Ok(value) => value,
            Err(message) => return ToolResult::failure(call_id, "invalid_args", message, false),
        };
        let arguments = args.get("arguments").cloned().unwrap_or_else(empty_object);
        let exposed_name = format!("mcp.{}.{}", server_id, tool_name);
        let (server, tool) = match self.manager.catalog().resolve_tool(&exposed_name) {
            Some((_, tool)) => {
                let server = match self.manager.client(server_id) {
                    Some(client) => client.server.clone(),
                    None => {
                        return ToolResult::failure(
                            call_id,
                            "unknown_server",
                            "unknown MCP server",
                            false,
                        )
                    }
                };
                (server, tool.clone())
            }
            None => {
                return ToolResult::failure(
                    call_id,
                    "unknown_tool",
                    "connect the server and verify the namespaced tool first",
                    false,
                )
            }
        };
        match self
            .firewall
            .check_tool(&server, &tool, &arguments, None, now_ms)
        {
            PolicyDecision::Allow => {}
            PolicyDecision::ApprovalRequired { reason } => {
                return ToolResult::failure(call_id, "approval_required", &reason, false)
            }
            PolicyDecision::Deny { reason } => {
                return ToolResult::failure(call_id, "policy_denied", &reason, false)
            }
            PolicyDecision::RateLimited => {
                return ToolResult::failure(
                    call_id,
                    "rate_limited",
                    "MCP call budget exceeded",
                    true,
                )
            }
            PolicyDecision::CircuitOpen => {
                return ToolResult::failure(
                    call_id,
                    "circuit_open",
                    "MCP server circuit breaker is open",
                    true,
                )
            }
        }
        let (_, request) = match self.manager.plan_tool_call(&exposed_name, arguments) {
            Ok(value) => value,
            Err(error) => return host_error(call_id, error),
        };
        let result = self.exchange(&server, &request);
        self.firewall
            .record_result(server_id, result.is_ok(), now_ms);
        match result {
            Ok(response) => match response.result {
                Some(value) => {
                    let guarded = GuardedResult::filter(
                        &value,
                        Provenance {
                            server_id: server_id.to_string(),
                            primitive: tool_name.to_string(),
                            trust: server.trust,
                        },
                        server.max_response_bytes,
                    );
                    ToolResult::success(call_id, guarded.value, 0)
                }
                None => remote_failure(call_id, response.error),
            },
            Err((code, message, retryable)) => {
                ToolResult::failure(call_id, code, &message, retryable)
            }
        }
    }

    pub fn read_resource(&mut self, args: &JsonValue, call_id: &str) -> ToolResult {
        let server_id = match required_string(args, "server") {
            Ok(value) => value,
            Err(message) => return ToolResult::failure(call_id, "invalid_args", message, false),
        };
        let uri = match required_string(args, "uri") {
            Ok(value) => value,
            Err(message) => return ToolResult::failure(call_id, "invalid_args", message, false),
        };
        let server = match self.manager.client(server_id) {
            Some(client) => client.server.clone(),
            None => {
                return ToolResult::failure(call_id, "unknown_server", "unknown MCP server", false)
            }
        };
        let request = match self.manager.resource_request(server_id, uri) {
            Ok(request) => request,
            Err(error) => return host_error(call_id, error),
        };
        self.exchange_result(&server, &request, call_id)
    }

    pub fn get_prompt(&mut self, args: &JsonValue, call_id: &str) -> ToolResult {
        let server_id = match required_string(args, "server") {
            Ok(value) => value,
            Err(message) => return ToolResult::failure(call_id, "invalid_args", message, false),
        };
        let name = match required_string(args, "name") {
            Ok(value) => value,
            Err(message) => return ToolResult::failure(call_id, "invalid_args", message, false),
        };
        let arguments = args.get("arguments").cloned().unwrap_or_else(empty_object);
        let server = match self.manager.client(server_id) {
            Some(client) => client.server.clone(),
            None => {
                return ToolResult::failure(call_id, "unknown_server", "unknown MCP server", false)
            }
        };
        let request = match self.manager.prompt_request(server_id, name, arguments) {
            Ok(request) => request,
            Err(error) => return host_error(call_id, error),
        };
        self.exchange_result(&server, &request, call_id)
    }

    fn connect_server(&mut self, server_id: &str) -> Result<usize, ErrorTuple> {
        let server = self
            .manager
            .client(server_id)
            .ok_or(("unknown_server", String::from("unknown MCP server"), false))?
            .server
            .clone();
        let discover = self
            .manager
            .client_mut(server_id)
            .map_err(|error| ("client_error", error.message, false))?
            .discover_request()
            .map_err(|error| ("client_error", error.message, false))?;
        let response = self.exchange(&server, &discover)?;
        self.manager
            .client_mut(server_id)
            .map_err(|error| ("client_error", error.message, false))?
            .accept_discovery(&response)
            .map_err(|error| ("discovery_error", error.message, false))?;
        if self.manager.client(server_id).map(|client| client.status)
            == Some(ClientStatus::LegacyAdapterRequired)
        {
            return Err((
                "legacy_adapter_required",
                String::from("server requires a legacy MCP bridge adapter"),
                false,
            ));
        }

        let requests = self
            .manager
            .client_mut(server_id)
            .map_err(|error| ("client_error", error.message, false))?
            .list_requests()
            .map_err(|error| ("client_error", error.message, false))?;
        for request in requests {
            let kind = match request.method.as_str() {
                genos_mcp::wire::method::TOOLS_LIST => ListKind::Tools,
                genos_mcp::wire::method::RESOURCES_LIST => ListKind::Resources,
                genos_mcp::wire::method::PROMPTS_LIST => ListKind::Prompts,
                _ => continue,
            };
            let response = self.exchange(&server, &request)?;
            let mut cursor = self
                .manager
                .ingest_list_page(server_id, kind, &response, 0, true)
                .map_err(|_| {
                    (
                        "catalog_error",
                        String::from("invalid MCP list response"),
                        false,
                    )
                })?;
            let mut pages = 1usize;
            while let Some(next) = cursor {
                if pages >= 100 {
                    return Err((
                        "pagination_limit",
                        String::from("MCP catalog exceeded 100 pages"),
                        false,
                    ));
                }
                let request = self
                    .manager
                    .next_page_request(server_id, kind, &next)
                    .map_err(|_| {
                        (
                            "catalog_error",
                            String::from("could not build MCP page request"),
                            false,
                        )
                    })?;
                let response = self.exchange(&server, &request)?;
                cursor = self
                    .manager
                    .ingest_list_page(server_id, kind, &response, 0, false)
                    .map_err(|_| {
                        (
                            "catalog_error",
                            String::from("invalid MCP list page"),
                            false,
                        )
                    })?;
                pages += 1;
            }
        }
        let catalog = self.manager.catalog().server(server_id);
        Ok(catalog
            .map(|catalog| catalog.tools.len() + catalog.resources.len() + catalog.prompts.len())
            .unwrap_or(0))
    }

    fn exchange(
        &mut self,
        server: &genos_mcp::ServerConfig,
        request: &genos_mcp::RpcRequest,
    ) -> Result<genos_mcp::RpcResponse, ErrorTuple> {
        let mut resolver = GenosCredentialResolver;
        let auth = resolve_auth(&server.auth, &mut resolver)
            .map_err(|error| ("credential_error", error.message, false))?;
        let response = self
            .http
            .exchange(server, request, auth.headers())
            .map_err(|error| ("transport_error", error.message, error.retryable));
        genos_tools::journal::log_mcp_event(
            &self.session_id,
            &server.id,
            &request.method,
            if response.is_ok() {
                "completed"
            } else {
                "failed"
            },
        );
        response
    }

    fn exchange_result(
        &mut self,
        server: &genos_mcp::ServerConfig,
        request: &genos_mcp::RpcRequest,
        call_id: &str,
    ) -> ToolResult {
        match self.exchange(server, request) {
            Ok(response) => match response.result {
                Some(result) => {
                    let guarded = GuardedResult::filter(
                        &result,
                        Provenance {
                            server_id: server.id.clone(),
                            primitive: request.method.clone(),
                            trust: server.trust.clone(),
                        },
                        server.max_response_bytes,
                    );
                    ToolResult::success(call_id, guarded.value, 0)
                }
                None => remote_failure(call_id, response.error),
            },
            Err((code, message, retryable)) => {
                ToolResult::failure(call_id, code, &message, retryable)
            }
        }
    }
}

type ErrorTuple = (&'static str, String, bool);

struct GenosCredentialResolver;

impl CredentialResolver for GenosCredentialResolver {
    fn resolve(&mut self, reference: &genos_mcp::CredentialRef) -> Result<Secret, CredentialError> {
        match reference {
            genos_mcp::CredentialRef::File(path) => disk::read_file(path)
                .map(|mut bytes| {
                    while matches!(bytes.last(), Some(b'\n' | b'\r')) {
                        bytes.pop();
                    }
                    Secret::new(bytes)
                })
                .map_err(|_| CredentialError::new("credential file could not be read")),
            genos_mcp::CredentialRef::Environment(_) => Err(CredentialError::new(
                "environment credentials require the hosted bridge",
            )),
            genos_mcp::CredentialRef::OAuthProfile(_) => Err(CredentialError::new(
                "OAuth profiles require the hosted authorization bridge",
            )),
            genos_mcp::CredentialRef::FirmwareVariable(_) => Err(CredentialError::new(
                "firmware credential provider is not installed",
            )),
        }
    }
}

fn status_name(status: ClientStatus) -> &'static str {
    match status {
        ClientStatus::Configured => "configured",
        ClientStatus::Discovering => "discovering",
        ClientStatus::Ready => "ready",
        ClientStatus::LegacyAdapterRequired => "legacy_adapter_required",
        ClientStatus::Degraded => "degraded",
        ClientStatus::Disabled => "disabled",
    }
}

fn required_string<'a>(args: &'a JsonValue, key: &str) -> Result<&'a str, &'static str> {
    args.get(key)
        .and_then(JsonValue::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("required string argument is missing")
}

fn host_error(call_id: &str, error: McpHostError) -> ToolResult {
    let (code, message) = match error {
        McpHostError::Client(error) => ("client_error", error.message),
        McpHostError::Remote(error) => ("remote_error", error.message),
        McpHostError::UnknownTool(message) => ("unknown_tool", message),
        McpHostError::CapabilityDenied(message) => ("policy_denied", message),
        McpHostError::InvalidArguments(message) => ("invalid_args", message),
        McpHostError::InvalidResult(message) => ("invalid_result", message),
    };
    ToolResult::failure(call_id, code, &message, false)
}

fn remote_failure(call_id: &str, error: Option<genos_mcp::McpError>) -> ToolResult {
    let message = error
        .map(|error| error.message)
        .unwrap_or_else(|| String::from("MCP response had no result"));
    ToolResult::failure(call_id, "remote_error", &message, false)
}

fn object(values: &[(&str, JsonValue)]) -> JsonValue {
    let mut object = alloc::collections::BTreeMap::new();
    for (key, value) in values {
        object.insert((*key).to_string(), value.clone());
    }
    JsonValue::Object(object)
}

fn empty_object() -> JsonValue {
    JsonValue::Object(alloc::collections::BTreeMap::new())
}
