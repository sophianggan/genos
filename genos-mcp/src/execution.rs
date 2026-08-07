//! Host execution planning, paginated catalog ingestion, and result states.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use genos_kernel::json::JsonValue;

use crate::client::{ClientError, ClientManager};
use crate::primitives::{CacheScope, Page, Prompt, Resource, ServerCatalog, Tool};
use crate::wire::{method, McpError, RpcRequest, RpcResponse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListKind {
    Tools,
    Resources,
    Prompts,
}

impl ListKind {
    pub fn method(self) -> &'static str {
        match self {
            Self::Tools => method::TOOLS_LIST,
            Self::Resources => method::RESOURCES_LIST,
            Self::Prompts => method::PROMPTS_LIST,
        }
    }

    fn is_allowed(self, manager: &ClientManager, server_id: &str) -> bool {
        manager
            .client(server_id)
            .map(|client| match self {
                Self::Tools => client.server.allow_tools,
                Self::Resources => client.server.allow_resources,
                Self::Prompts => client.server.allow_prompts,
            })
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolInvocation {
    pub server_id: String,
    pub tool_name: String,
    pub arguments: JsonValue,
}

impl ToolInvocation {
    pub fn exposed_name(&self) -> String {
        crate::primitives::namespaced(&self.server_id, &self.tool_name)
    }
}

impl ClientManager {
    /// Resolve a namespaced model-selected tool into one isolated server call.
    pub fn plan_tool_call(
        &mut self,
        exposed_name: &str,
        arguments: JsonValue,
    ) -> Result<(ToolInvocation, RpcRequest), McpHostError> {
        if arguments.as_object().is_none() {
            return Err(McpHostError::InvalidArguments(String::from(
                "tool arguments must be an object",
            )));
        }
        let (server_id, tool_name) = {
            let (server_id, tool) = self
                .catalog()
                .resolve_tool(exposed_name)
                .ok_or_else(|| McpHostError::UnknownTool(exposed_name.to_string()))?;
            (server_id.to_string(), tool.name.clone())
        };
        let client = self.client_mut(&server_id)?;
        if !client.server.allow_tools {
            return Err(McpHostError::CapabilityDenied(String::from("tools")));
        }
        if !client.server.allowed_tools.is_empty()
            && !client
                .server
                .allowed_tools
                .iter()
                .any(|allowed| allowed == &tool_name)
        {
            return Err(McpHostError::CapabilityDenied(tool_name));
        }
        let params = object(&[
            ("name", JsonValue::Str(tool_name.clone())),
            ("arguments", arguments.clone()),
        ]);
        let request = client.request(method::TOOLS_CALL, params)?;
        Ok((
            ToolInvocation {
                server_id,
                tool_name,
                arguments,
            },
            request,
        ))
    }

    pub fn resource_request(
        &mut self,
        server_id: &str,
        uri: &str,
    ) -> Result<RpcRequest, McpHostError> {
        let client = self.client_mut(server_id)?;
        if !client.server.allow_resources {
            return Err(McpHostError::CapabilityDenied(String::from("resources")));
        }
        Ok(client.request(
            method::RESOURCES_READ,
            object(&[("uri", JsonValue::Str(uri.to_string()))]),
        )?)
    }

    pub fn prompt_request(
        &mut self,
        server_id: &str,
        name: &str,
        arguments: JsonValue,
    ) -> Result<RpcRequest, McpHostError> {
        let client = self.client_mut(server_id)?;
        if !client.server.allow_prompts {
            return Err(McpHostError::CapabilityDenied(String::from("prompts")));
        }
        Ok(client.request(
            method::PROMPTS_GET,
            object(&[
                ("name", JsonValue::Str(name.to_string())),
                ("arguments", arguments),
            ]),
        )?)
    }

    pub fn next_page_request(
        &mut self,
        server_id: &str,
        kind: ListKind,
        cursor: &str,
    ) -> Result<RpcRequest, McpHostError> {
        if !kind.is_allowed(self, server_id) {
            return Err(McpHostError::CapabilityDenied(kind.method().to_string()));
        }
        Ok(self.client_mut(server_id)?.request(
            kind.method(),
            object(&[("cursor", JsonValue::Str(cursor.to_string()))]),
        )?)
    }

    /// Merge one list page. Returns the opaque cursor for the next page.
    pub fn ingest_list_page(
        &mut self,
        server_id: &str,
        kind: ListKind,
        response: &RpcResponse,
        fetched_at_ms: u64,
        first_page: bool,
    ) -> Result<Option<String>, McpHostError> {
        let result = response_result(response)?;
        let mut catalog = if first_page {
            ServerCatalog::new(server_id)
        } else {
            self.catalog()
                .server(server_id)
                .cloned()
                .unwrap_or_else(|| ServerCatalog::new(server_id))
        };
        catalog.fetched_at_ms = fetched_at_ms;
        let cursor = match kind {
            ListKind::Tools => {
                let page = Page::<Tool>::tools(result)
                    .map_err(|error| McpHostError::InvalidResult(error.message))?;
                if first_page {
                    catalog.tools = page.items;
                } else {
                    catalog.tools.extend(page.items);
                }
                apply_cache(&mut catalog, page.ttl_ms, page.cache_scope);
                page.next_cursor
            }
            ListKind::Resources => {
                let page = Page::<Resource>::resources(result)
                    .map_err(|error| McpHostError::InvalidResult(error.message))?;
                if first_page {
                    catalog.resources = page.items;
                } else {
                    catalog.resources.extend(page.items);
                }
                apply_cache(&mut catalog, page.ttl_ms, page.cache_scope);
                page.next_cursor
            }
            ListKind::Prompts => {
                let page = Page::<Prompt>::prompts(result)
                    .map_err(|error| McpHostError::InvalidResult(error.message))?;
                if first_page {
                    catalog.prompts = page.items;
                } else {
                    catalog.prompts.extend(page.items);
                }
                apply_cache(&mut catalog, page.ttl_ms, page.cache_scope);
                page.next_cursor
            }
        };
        self.catalog_mut().replace(catalog);
        Ok(cursor)
    }
}

fn apply_cache(catalog: &mut ServerCatalog, ttl_ms: Option<u64>, scope: CacheScope) {
    catalog.ttl_ms = match (catalog.ttl_ms, ttl_ms) {
        (Some(current), Some(incoming)) => Some(current.min(incoming)),
        (None, incoming) => incoming,
        (current, None) => current,
    };
    catalog.cache_scope = most_restrictive(catalog.cache_scope, scope);
}

fn most_restrictive(left: CacheScope, right: CacheScope) -> CacheScope {
    use CacheScope::{None, Private, Public};
    match (left, right) {
        (None, _) | (_, None) => None,
        (Private, _) | (_, Private) => Private,
        (Public, Public) => Public,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Text {
        text: String,
        annotations: Option<JsonValue>,
    },
    Image {
        data: String,
        mime_type: String,
        annotations: Option<JsonValue>,
    },
    Audio {
        data: String,
        mime_type: String,
        annotations: Option<JsonValue>,
    },
    ResourceLink {
        uri: String,
        name: String,
        mime_type: Option<String>,
        annotations: Option<JsonValue>,
    },
    EmbeddedResource(JsonValue),
    Unknown(JsonValue),
}

impl Content {
    pub fn from_json(value: &JsonValue) -> Result<Self, McpHostError> {
        let kind = value
            .get("type")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| McpHostError::InvalidResult(String::from("content type is missing")))?;
        let annotations = value.get("annotations").cloned();
        match kind {
            "text" => Ok(Self::Text {
                text: required_string(value, "text")?,
                annotations,
            }),
            "image" => Ok(Self::Image {
                data: required_string(value, "data")?,
                mime_type: required_string(value, "mimeType")?,
                annotations,
            }),
            "audio" => Ok(Self::Audio {
                data: required_string(value, "data")?,
                mime_type: required_string(value, "mimeType")?,
                annotations,
            }),
            "resource_link" => Ok(Self::ResourceLink {
                uri: required_string(value, "uri")?,
                name: required_string(value, "name")?,
                mime_type: optional_string(value, "mimeType")?,
                annotations,
            }),
            "resource" => Ok(Self::EmbeddedResource(value.clone())),
            _ => Ok(Self::Unknown(value.clone())),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CallOutcome {
    Complete {
        content: Vec<Content>,
        structured_content: Option<JsonValue>,
        is_error: bool,
    },
    InputRequired {
        input_requests: JsonValue,
        request_state: String,
    },
    Extension(JsonValue),
}

impl CallOutcome {
    pub fn from_response(response: &RpcResponse) -> Result<Self, McpHostError> {
        let result = response_result(response)?;
        let result_type = result
            .get("resultType")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| McpHostError::InvalidResult(String::from("resultType is required")))?;
        match result_type {
            "complete" => {
                let content = match result.get("content") {
                    None => Vec::new(),
                    Some(JsonValue::Array(values)) => values
                        .iter()
                        .map(Content::from_json)
                        .collect::<Result<Vec<_>, _>>()?,
                    Some(_) => {
                        return Err(McpHostError::InvalidResult(String::from(
                            "tool content must be an array",
                        )))
                    }
                };
                Ok(Self::Complete {
                    content,
                    structured_content: result.get("structuredContent").cloned(),
                    is_error: result
                        .get("isError")
                        .and_then(JsonValue::as_bool)
                        .unwrap_or(false),
                })
            }
            "input_required" => Ok(Self::InputRequired {
                input_requests: result.get("inputRequests").cloned().ok_or_else(|| {
                    McpHostError::InvalidResult(String::from(
                        "input_required result needs inputRequests",
                    ))
                })?,
                request_state: required_string(result, "requestState")?,
            }),
            _ => Ok(Self::Extension(result.clone())),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Continuation {
    pub server_id: String,
    pub method: String,
    pub original_params: JsonValue,
    pub request_state: String,
}

impl Continuation {
    pub fn from_outcome(
        server_id: &str,
        request: &RpcRequest,
        outcome: &CallOutcome,
    ) -> Option<Self> {
        match outcome {
            CallOutcome::InputRequired { request_state, .. } => Some(Self {
                server_id: server_id.to_string(),
                method: request.method.clone(),
                original_params: request.params.clone(),
                request_state: request_state.clone(),
            }),
            _ => None,
        }
    }

    pub fn resume(
        self,
        manager: &mut ClientManager,
        input_responses: JsonValue,
    ) -> Result<RpcRequest, McpHostError> {
        if input_responses.as_object().is_none() {
            return Err(McpHostError::InvalidArguments(String::from(
                "inputResponses must be an object",
            )));
        }
        let mut params = self
            .original_params
            .as_object()
            .cloned()
            .unwrap_or_default();
        // Metadata is regenerated by ClientState::request.
        params.remove("_meta");
        params.insert(
            String::from("requestState"),
            JsonValue::Str(self.request_state),
        );
        params.insert(String::from("inputResponses"), input_responses);
        Ok(manager
            .client_mut(&self.server_id)?
            .request(&self.method, JsonValue::Object(params))?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum McpHostError {
    Client(ClientError),
    Remote(McpError),
    UnknownTool(String),
    CapabilityDenied(String),
    InvalidArguments(String),
    InvalidResult(String),
}

impl From<ClientError> for McpHostError {
    fn from(error: ClientError) -> Self {
        Self::Client(error)
    }
}

fn response_result(response: &RpcResponse) -> Result<&JsonValue, McpHostError> {
    if let Some(error) = &response.error {
        return Err(McpHostError::Remote(error.clone()));
    }
    response
        .result
        .as_ref()
        .ok_or_else(|| McpHostError::InvalidResult(String::from("response result is missing")))
}

fn required_string(value: &JsonValue, key: &str) -> Result<String, McpHostError> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| {
            McpHostError::InvalidResult(String::from("required result string is missing"))
        })
}

fn optional_string(value: &JsonValue, key: &str) -> Result<Option<String>, McpHostError> {
    match value.get(key) {
        None => Ok(None),
        Some(JsonValue::Str(value)) => Ok(Some(value.clone())),
        Some(_) => Err(McpHostError::InvalidResult(String::from(
            "optional result string has wrong type",
        ))),
    }
}

fn object(values: &[(&str, JsonValue)]) -> JsonValue {
    let mut object = BTreeMap::new();
    for (key, value) in values {
        object.insert((*key).to_string(), value.clone());
    }
    JsonValue::Object(object)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::client::{ClientStatus, Discovery};
    use crate::config::McpConfig;
    use genos_kernel::json::parse;

    fn ready_manager() -> ClientManager {
        let config = McpConfig::parse(
            br#"
            [[servers]]
            id = "demo"
            endpoint = "https://demo.example/mcp"
            "#,
        )
        .unwrap();
        let mut manager = ClientManager::from_config(&config);
        let client = manager.client_mut("demo").unwrap();
        client.status = ClientStatus::Ready;
        client.discovery = Some(Discovery {
            supported_versions: vec![String::from("2026-07-28")],
            selected_version: String::from("2026-07-28"),
            capabilities: parse(r#"{"tools":{}}"#).unwrap(),
            identity: None,
            instructions: None,
            ttl_ms: None,
        });
        manager
    }

    #[test]
    fn ingests_pages_then_resolves_namespaced_call() {
        let mut manager = ready_manager();
        let first = parse(
            r#"{"resultType":"complete","tools":[{"name":"search","inputSchema":{"type":"object"}}],"nextCursor":"p2"}"#,
        )
        .unwrap();
        let cursor = manager
            .ingest_list_page(
                "demo",
                ListKind::Tools,
                &RpcResponse::success(crate::RequestId::Number(1), first),
                100,
                true,
            )
            .unwrap();
        assert_eq!(cursor.as_deref(), Some("p2"));
        let arguments = parse(r#"{"q":"genos"}"#).unwrap();
        let (invocation, request) = manager
            .plan_tool_call("mcp.demo.search", arguments)
            .unwrap();
        assert_eq!(invocation.tool_name, "search");
        assert_eq!(request.method, method::TOOLS_CALL);
    }

    #[test]
    fn parses_input_required_and_resumes_with_explicit_state() {
        let response = parse(
            r#"{
              "resultType":"input_required",
              "inputRequests":{"confirm":{"type":"elicitation","message":"Continue?"}},
              "requestState":"opaque-state"
            }"#,
        )
        .unwrap();
        let outcome = CallOutcome::from_response(&RpcResponse::success(
            crate::RequestId::Number(1),
            response,
        ))
        .unwrap();
        let mut manager = ready_manager();
        let request = manager
            .client_mut("demo")
            .unwrap()
            .request(
                method::TOOLS_CALL,
                object(&[("name", JsonValue::Str(String::from("dangerous")))]),
            )
            .unwrap();
        let continuation = Continuation::from_outcome("demo", &request, &outcome).unwrap();
        let resumed = continuation
            .resume(&mut manager, parse(r#"{"confirm":true}"#).unwrap())
            .unwrap();
        assert_eq!(
            resumed.params.get("requestState").unwrap().as_str(),
            Some("opaque-state")
        );
        assert!(resumed.params.get("inputResponses").is_some());
    }

    #[test]
    fn preserves_unknown_extension_result() {
        let result = parse(r#"{"resultType":"io.example/task","task":{"id":"42"}}"#).unwrap();
        let outcome =
            CallOutcome::from_response(&RpcResponse::success(crate::RequestId::Number(1), result))
                .unwrap();
        assert!(matches!(outcome, CallOutcome::Extension(_)));
    }
}
