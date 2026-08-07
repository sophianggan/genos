//! MCP host-side client isolation, discovery, and request planning.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use genos_kernel::json::JsonValue;

use crate::config::{McpConfig, ServerConfig};
use crate::primitives::Catalog;
use crate::wire::{
    method, ClientIdentity, RequestId, RequestMetadata, RpcRequest, RpcResponse,
    LATEST_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSIONS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientStatus {
    Configured,
    Discovering,
    Ready,
    LegacyAdapterRequired,
    Degraded,
    Disabled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerIdentity {
    pub name: String,
    pub version: String,
    pub title: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Discovery {
    pub supported_versions: Vec<String>,
    pub selected_version: String,
    pub capabilities: JsonValue,
    pub identity: Option<ServerIdentity>,
    pub instructions: Option<String>,
    pub ttl_ms: Option<u64>,
}

impl Discovery {
    pub fn from_result(result: &JsonValue) -> Result<Self, ClientError> {
        let supported_versions = result
            .get("supportedVersions")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| ClientError::invalid_discovery("supportedVersions is required"))?
            .iter()
            .map(|value| {
                value.as_str().map(ToString::to_string).ok_or_else(|| {
                    ClientError::invalid_discovery("supportedVersions must contain strings")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let selected_version = select_protocol(&supported_versions).ok_or_else(|| {
            ClientError::new(
                ClientErrorKind::UnsupportedProtocol,
                "server does not support a known MCP protocol version",
            )
        })?;
        let capabilities = result
            .get("capabilities")
            .cloned()
            .unwrap_or_else(empty_object);
        if capabilities.as_object().is_none() {
            return Err(ClientError::invalid_discovery(
                "server capabilities must be an object",
            ));
        }
        let identity = result
            .get("_meta")
            .and_then(|meta| meta.get("io.modelcontextprotocol/serverInfo"))
            .map(parse_server_identity)
            .transpose()?;
        let instructions = optional_string(result, "instructions")?;
        let ttl_ms = result.get("ttlMs").and_then(JsonValue::as_f64).map(|ttl| {
            if ttl.is_sign_negative() {
                0
            } else {
                ttl as u64
            }
        });
        Ok(Self {
            supported_versions,
            selected_version,
            capabilities,
            identity,
            instructions,
            ttl_ms,
        })
    }

    pub fn supports(&self, capability: &str) -> bool {
        self.capabilities.get(capability).is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClientState {
    pub server: ServerConfig,
    pub status: ClientStatus,
    pub discovery: Option<Discovery>,
    request_sequence: u64,
}

impl ClientState {
    pub fn new(server: ServerConfig) -> Self {
        let status = if server.enabled {
            ClientStatus::Configured
        } else {
            ClientStatus::Disabled
        };
        Self {
            server,
            status,
            discovery: None,
            request_sequence: 0,
        }
    }

    pub fn metadata(&self) -> RequestMetadata {
        let mut metadata = RequestMetadata::modern(ClientIdentity::genos());
        if let Some(discovery) = &self.discovery {
            metadata.protocol_version = discovery.selected_version.clone();
        }
        metadata
    }

    pub fn next_id(&mut self) -> RequestId {
        self.request_sequence = self.request_sequence.saturating_add(1);
        RequestId::String(format!("{}-{}", self.server.id, self.request_sequence))
    }

    pub fn discover_request(&mut self) -> Result<RpcRequest, ClientError> {
        if self.status == ClientStatus::Disabled {
            return Err(ClientError::new(
                ClientErrorKind::Disabled,
                "MCP server is disabled",
            ));
        }
        self.status = ClientStatus::Discovering;
        let id = self.next_id();
        Ok(RpcRequest::discover(id, &self.metadata()))
    }

    pub fn accept_discovery(&mut self, response: &RpcResponse) -> Result<(), ClientError> {
        let result = response.result.as_ref().ok_or_else(|| {
            let message = response
                .error
                .as_ref()
                .map(|error| error.message.as_str())
                .unwrap_or("discovery returned no result");
            ClientError::invalid_discovery(message)
        })?;
        let discovery = Discovery::from_result(result)?;
        self.status = if discovery.selected_version == LATEST_PROTOCOL_VERSION {
            ClientStatus::Ready
        } else {
            ClientStatus::LegacyAdapterRequired
        };
        self.discovery = Some(discovery);
        Ok(())
    }

    pub fn list_requests(&mut self) -> Result<Vec<RpcRequest>, ClientError> {
        if self.status != ClientStatus::Ready {
            return Err(ClientError::new(
                ClientErrorKind::NotReady,
                "MCP client must complete modern discovery first",
            ));
        }
        let supports = |capability: &str| {
            self.discovery
                .as_ref()
                .map(|discovery| discovery.supports(capability))
                .unwrap_or(false)
        };
        let list_tools = self.server.allow_tools && supports("tools");
        let list_resources = self.server.allow_resources && supports("resources");
        let list_prompts = self.server.allow_prompts && supports("prompts");
        let metadata = self.metadata();
        let mut requests = Vec::new();
        if list_tools {
            let id = self.next_id();
            requests.push(RpcRequest::new(
                id,
                method::TOOLS_LIST,
                empty_object(),
                &metadata,
            ));
        }
        if list_resources {
            let id = self.next_id();
            requests.push(RpcRequest::new(
                id,
                method::RESOURCES_LIST,
                empty_object(),
                &metadata,
            ));
        }
        if list_prompts {
            let id = self.next_id();
            requests.push(RpcRequest::new(
                id,
                method::PROMPTS_LIST,
                empty_object(),
                &metadata,
            ));
        }
        Ok(requests)
    }

    pub fn request(&mut self, method: &str, params: JsonValue) -> Result<RpcRequest, ClientError> {
        if self.status != ClientStatus::Ready {
            return Err(ClientError::new(
                ClientErrorKind::NotReady,
                "MCP client is not ready",
            ));
        }
        let id = self.next_id();
        Ok(RpcRequest::new(id, method, params, &self.metadata()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClientManager {
    clients: BTreeMap<String, ClientState>,
    catalog: Catalog,
}

impl ClientManager {
    pub fn from_config(config: &McpConfig) -> Self {
        let mut clients = BTreeMap::new();
        for server in &config.servers {
            clients.insert(server.id.clone(), ClientState::new(server.clone()));
        }
        Self {
            clients,
            catalog: Catalog::new(),
        }
    }

    pub fn client(&self, server_id: &str) -> Option<&ClientState> {
        self.clients.get(server_id)
    }

    pub fn client_mut(&mut self, server_id: &str) -> Result<&mut ClientState, ClientError> {
        self.clients
            .get_mut(server_id)
            .ok_or_else(|| ClientError::new(ClientErrorKind::UnknownServer, "unknown MCP server"))
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn catalog_mut(&mut self) -> &mut Catalog {
        &mut self.catalog
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ClientState)> {
        self.clients
            .iter()
            .map(|(id, client)| (id.as_str(), client))
    }

    pub fn len(&self) -> usize {
        self.clients.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientErrorKind {
    UnknownServer,
    Disabled,
    NotReady,
    InvalidDiscovery,
    UnsupportedProtocol,
    Remote,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientError {
    pub kind: ClientErrorKind,
    pub message: String,
}

impl ClientError {
    pub fn new(kind: ClientErrorKind, message: &str) -> Self {
        Self {
            kind,
            message: message.to_string(),
        }
    }

    fn invalid_discovery(message: &str) -> Self {
        Self::new(ClientErrorKind::InvalidDiscovery, message)
    }
}

fn select_protocol(supported: &[String]) -> Option<String> {
    if supported
        .iter()
        .any(|version| version == LATEST_PROTOCOL_VERSION)
    {
        return Some(LATEST_PROTOCOL_VERSION.to_string());
    }
    for legacy in LEGACY_PROTOCOL_VERSIONS {
        if supported.iter().any(|version| version == legacy) {
            return Some((*legacy).to_string());
        }
    }
    None
}

fn parse_server_identity(value: &JsonValue) -> Result<ServerIdentity, ClientError> {
    let name = value
        .get("name")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| ClientError::invalid_discovery("serverInfo.name is required"))?;
    let version = value
        .get("version")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| ClientError::invalid_discovery("serverInfo.version is required"))?;
    Ok(ServerIdentity {
        name: name.to_string(),
        version: version.to_string(),
        title: optional_string(value, "title")?,
    })
}

fn optional_string(value: &JsonValue, field: &str) -> Result<Option<String>, ClientError> {
    match value.get(field) {
        None => Ok(None),
        Some(JsonValue::Str(value)) => Ok(Some(value.clone())),
        Some(_) => Err(ClientError::invalid_discovery(
            "optional discovery string has wrong type",
        )),
    }
}

fn empty_object() -> JsonValue {
    JsonValue::Object(BTreeMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::McpConfig;
    use genos_kernel::json::parse;

    fn manager() -> ClientManager {
        let config = McpConfig::parse(
            br#"
            [[servers]]
            id = "one"
            endpoint = "https://one.example/mcp"
            [[servers]]
            id = "two"
            endpoint = "https://two.example/mcp"
            enabled = false
            "#,
        )
        .unwrap();
        ClientManager::from_config(&config)
    }

    #[test]
    fn creates_exactly_one_client_per_server() {
        let manager = manager();
        assert_eq!(manager.len(), 2);
        assert_eq!(
            manager.client("one").unwrap().status,
            ClientStatus::Configured
        );
        assert_eq!(
            manager.client("two").unwrap().status,
            ClientStatus::Disabled
        );
    }

    #[test]
    fn discovery_selects_modern_protocol_and_capabilities() {
        let mut manager = manager();
        let client = manager.client_mut("one").unwrap();
        let request = client.discover_request().unwrap();
        let result = parse(
            r#"{
              "resultType":"complete",
              "supportedVersions":["2026-07-28","2025-11-25"],
              "capabilities":{"tools":{},"resources":{}},
              "_meta":{"io.modelcontextprotocol/serverInfo":{"name":"demo","version":"1"}}
            }"#,
        )
        .unwrap();
        client
            .accept_discovery(&RpcResponse::success(request.id, result))
            .unwrap();
        assert_eq!(client.status, ClientStatus::Ready);
        let requests = client.list_requests().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].method, method::TOOLS_LIST);
        assert_eq!(requests[1].method, method::RESOURCES_LIST);
    }

    #[test]
    fn legacy_server_is_kept_behind_adapter_boundary() {
        let mut manager = manager();
        let client = manager.client_mut("one").unwrap();
        let request = client.discover_request().unwrap();
        let result = parse(
            r#"{"resultType":"complete","supportedVersions":["2025-11-25"],"capabilities":{}}"#,
        )
        .unwrap();
        client
            .accept_discovery(&RpcResponse::success(request.id, result))
            .unwrap();
        assert_eq!(client.status, ClientStatus::LegacyAdapterRequired);
        assert!(client.list_requests().is_err());
    }
}
