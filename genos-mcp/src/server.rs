//! Transport-independent MCP server router for exposing GenOS capabilities.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use genos_kernel::json::JsonValue;

use crate::primitives::{Prompt, Resource, Tool};
use crate::wire::{method, McpError, RpcRequest, RpcResponse, LATEST_PROTOCOL_VERSION};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub title: Option<String>,
}

impl ServerInfo {
    fn to_json(&self) -> JsonValue {
        let mut values = BTreeMap::new();
        values.insert(String::from("name"), JsonValue::Str(self.name.clone()));
        values.insert(
            String::from("version"),
            JsonValue::Str(self.version.clone()),
        );
        if let Some(title) = &self.title {
            values.insert(String::from("title"), JsonValue::Str(title.clone()));
        }
        JsonValue::Object(values)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServerCapabilities {
    pub tools: bool,
    pub resources: bool,
    pub prompts: bool,
}

impl ServerCapabilities {
    fn to_json(self) -> JsonValue {
        let mut values = BTreeMap::new();
        if self.tools {
            values.insert(String::from("tools"), empty_object());
        }
        if self.resources {
            values.insert(String::from("resources"), empty_object());
        }
        if self.prompts {
            values.insert(String::from("prompts"), empty_object());
        }
        JsonValue::Object(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingMetadata {
    pub protocol_version: String,
    pub method: String,
    pub name: Option<String>,
}

impl RoutingMetadata {
    pub fn from_request(request: &RpcRequest) -> Self {
        Self {
            protocol_version: request
                .protocol_version()
                .unwrap_or(LATEST_PROTOCOL_VERSION)
                .to_string(),
            method: request.method.clone(),
            name: request
                .params
                .get("name")
                .and_then(JsonValue::as_str)
                .map(ToString::to_string),
        }
    }
}

/// Application callbacks behind the protocol router. Implementations perform
/// their own policy checks; the router only enforces protocol structure.
pub trait McpService {
    fn info(&self) -> ServerInfo;
    fn instructions(&self) -> Option<String> {
        None
    }
    fn capabilities(&self) -> ServerCapabilities;
    fn tools(&self) -> Vec<Tool> {
        Vec::new()
    }
    fn resources(&self) -> Vec<Resource> {
        Vec::new()
    }
    fn prompts(&self) -> Vec<Prompt> {
        Vec::new()
    }
    fn call_tool(&mut self, name: &str, arguments: &JsonValue) -> Result<JsonValue, McpError>;
    fn read_resource(&mut self, uri: &str) -> Result<JsonValue, McpError>;
    fn get_prompt(&mut self, name: &str, arguments: &JsonValue) -> Result<JsonValue, McpError>;
}

pub struct McpServer<S> {
    service: S,
    catalog_ttl_ms: u64,
}

impl<S: McpService> McpServer<S> {
    pub fn new(service: S) -> Self {
        Self {
            service,
            catalog_ttl_ms: 60_000,
        }
    }

    pub fn with_catalog_ttl(mut self, ttl_ms: u64) -> Self {
        self.catalog_ttl_ms = ttl_ms;
        self
    }

    pub fn handle(&mut self, request: RpcRequest, routing: &RoutingMetadata) -> RpcResponse {
        if let Err(error) = validate_routing(&request, routing) {
            return RpcResponse::failure(request.id, error);
        }
        if request.protocol_version() != Some(LATEST_PROTOCOL_VERSION) {
            return RpcResponse::failure(request.id, unsupported_version());
        }
        let result = self.route(&request);
        match result {
            Ok(result) => RpcResponse::success(request.id, result),
            Err(error) => RpcResponse::failure(request.id, error),
        }
    }

    pub fn service(&self) -> &S {
        &self.service
    }

    pub fn service_mut(&mut self) -> &mut S {
        &mut self.service
    }

    fn route(&mut self, request: &RpcRequest) -> Result<JsonValue, McpError> {
        match request.method.as_str() {
            method::SERVER_DISCOVER => Ok(self.discovery()),
            method::TOOLS_LIST => {
                if !self.service.capabilities().tools {
                    return Err(McpError::method_not_found(method::TOOLS_LIST));
                }
                let mut tools = self.service.tools();
                tools.sort_by(|left, right| left.name.cmp(&right.name));
                Ok(list_result(
                    "tools",
                    tools.iter().map(Tool::to_json).collect(),
                    self.catalog_ttl_ms,
                ))
            }
            method::RESOURCES_LIST => {
                if !self.service.capabilities().resources {
                    return Err(McpError::method_not_found(method::RESOURCES_LIST));
                }
                let mut resources = self.service.resources();
                resources.sort_by(|left, right| left.uri.cmp(&right.uri));
                Ok(list_result(
                    "resources",
                    resources.iter().map(Resource::to_json).collect(),
                    self.catalog_ttl_ms,
                ))
            }
            method::PROMPTS_LIST => {
                if !self.service.capabilities().prompts {
                    return Err(McpError::method_not_found(method::PROMPTS_LIST));
                }
                let mut prompts = self.service.prompts();
                prompts.sort_by(|left, right| left.name.cmp(&right.name));
                Ok(list_result(
                    "prompts",
                    prompts.iter().map(Prompt::to_json).collect(),
                    self.catalog_ttl_ms,
                ))
            }
            method::TOOLS_CALL => {
                let name = required_string(&request.params, "name")?;
                let arguments = request
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(empty_object);
                if arguments.as_object().is_none() {
                    return Err(McpError::invalid_params("arguments must be an object"));
                }
                self.service.call_tool(name, &arguments)
            }
            method::RESOURCES_READ => {
                let uri = required_string(&request.params, "uri")?;
                self.service.read_resource(uri)
            }
            method::PROMPTS_GET => {
                let name = required_string(&request.params, "name")?;
                let arguments = request
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(empty_object);
                self.service.get_prompt(name, &arguments)
            }
            method::PING => Ok(complete_object()),
            unknown => Err(McpError::method_not_found(unknown)),
        }
    }

    fn discovery(&self) -> JsonValue {
        let mut result = match complete_object() {
            JsonValue::Object(values) => values,
            _ => BTreeMap::new(),
        };
        result.insert(
            String::from("supportedVersions"),
            JsonValue::Array(alloc::vec![JsonValue::Str(String::from(
                LATEST_PROTOCOL_VERSION,
            ))]),
        );
        result.insert(
            String::from("capabilities"),
            self.service.capabilities().to_json(),
        );
        if let Some(instructions) = self.service.instructions() {
            result.insert(String::from("instructions"), JsonValue::Str(instructions));
        }
        result.insert(
            String::from("ttlMs"),
            JsonValue::Number(self.catalog_ttl_ms as f64),
        );
        result.insert(
            String::from("cacheScope"),
            JsonValue::Str(String::from("private")),
        );
        let mut metadata = BTreeMap::new();
        metadata.insert(
            String::from("io.modelcontextprotocol/serverInfo"),
            self.service.info().to_json(),
        );
        result.insert(String::from("_meta"), JsonValue::Object(metadata));
        JsonValue::Object(result)
    }
}

fn validate_routing(request: &RpcRequest, routing: &RoutingMetadata) -> Result<(), McpError> {
    if routing.protocol_version != request.protocol_version().unwrap_or("") {
        return Err(McpError::invalid_request(
            "MCP-Protocol-Version header does not match request metadata",
        ));
    }
    if routing.method != request.method {
        return Err(McpError::invalid_request(
            "Mcp-Method header does not match request method",
        ));
    }
    let body_name = request.params.get("name").and_then(JsonValue::as_str);
    if routing.name.as_deref() != body_name {
        return Err(McpError::invalid_request(
            "Mcp-Name header does not match request name",
        ));
    }
    Ok(())
}

fn unsupported_version() -> McpError {
    let mut error = McpError::new(
        McpError::UNSUPPORTED_PROTOCOL_VERSION,
        "unsupported MCP protocol version",
    );
    error.data = Some(object(&[(
        "supported",
        JsonValue::Array(alloc::vec![JsonValue::Str(String::from(
            LATEST_PROTOCOL_VERSION,
        ))]),
    )]));
    error
}

fn required_string<'a>(value: &'a JsonValue, field: &str) -> Result<&'a str, McpError> {
    value
        .get(field)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| McpError::invalid_params("required string parameter is missing"))
}

fn list_result(field: &str, items: Vec<JsonValue>, ttl_ms: u64) -> JsonValue {
    object(&[
        ("resultType", JsonValue::Str(String::from("complete"))),
        (field, JsonValue::Array(items)),
        ("ttlMs", JsonValue::Number(ttl_ms as f64)),
        ("cacheScope", JsonValue::Str(String::from("private"))),
    ])
}

fn complete_object() -> JsonValue {
    object(&[("resultType", JsonValue::Str(String::from("complete")))])
}

fn empty_object() -> JsonValue {
    JsonValue::Object(BTreeMap::new())
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
    use super::*;
    use crate::wire::{ClientIdentity, RequestId, RequestMetadata};

    struct DemoService;

    impl McpService for DemoService {
        fn info(&self) -> ServerInfo {
            ServerInfo {
                name: String::from("genos-demo"),
                version: String::from("1"),
                title: Some(String::from("GenOS Demo")),
            }
        }

        fn capabilities(&self) -> ServerCapabilities {
            ServerCapabilities {
                tools: true,
                resources: false,
                prompts: false,
            }
        }

        fn tools(&self) -> Vec<Tool> {
            alloc::vec![Tool {
                name: String::from("echo"),
                title: None,
                description: Some(String::from("Echo arguments")),
                input_schema: object(&[("type", JsonValue::Str(String::from("object")))]),
                output_schema: None,
                annotations: None,
            }]
        }

        fn call_tool(&mut self, _name: &str, arguments: &JsonValue) -> Result<JsonValue, McpError> {
            Ok(object(&[
                ("resultType", JsonValue::Str(String::from("complete"))),
                ("structuredContent", arguments.clone()),
            ]))
        }

        fn read_resource(&mut self, _uri: &str) -> Result<JsonValue, McpError> {
            Err(McpError::method_not_found(method::RESOURCES_READ))
        }

        fn get_prompt(
            &mut self,
            _name: &str,
            _arguments: &JsonValue,
        ) -> Result<JsonValue, McpError> {
            Err(McpError::method_not_found(method::PROMPTS_GET))
        }
    }

    fn request(method_name: &str, params: JsonValue) -> RpcRequest {
        RpcRequest::new(
            RequestId::Number(1),
            method_name,
            params,
            &RequestMetadata::modern(ClientIdentity::genos()),
        )
    }

    #[test]
    fn discovers_and_lists_deterministically() {
        let mut server = McpServer::new(DemoService);
        let discover = request(method::SERVER_DISCOVER, empty_object());
        let response = server.handle(discover.clone(), &RoutingMetadata::from_request(&discover));
        let result = response.result.unwrap();
        assert_eq!(
            result.get("supportedVersions").unwrap().as_array().unwrap()[0].as_str(),
            Some(LATEST_PROTOCOL_VERSION)
        );

        let list = request(method::TOOLS_LIST, empty_object());
        let response = server.handle(list.clone(), &RoutingMetadata::from_request(&list));
        assert_eq!(
            response
                .result
                .unwrap()
                .get("tools")
                .unwrap()
                .as_array()
                .unwrap()[0]
                .get("name")
                .unwrap()
                .as_str(),
            Some("echo")
        );
    }

    #[test]
    fn rejects_header_body_mismatch() {
        let mut server = McpServer::new(DemoService);
        let call = request(
            method::TOOLS_CALL,
            object(&[("name", JsonValue::Str(String::from("echo")))]),
        );
        let mut routing = RoutingMetadata::from_request(&call);
        routing.name = Some(String::from("other"));
        let response = server.handle(call, &routing);
        assert_eq!(response.error.unwrap().code, McpError::INVALID_REQUEST);
    }
}
