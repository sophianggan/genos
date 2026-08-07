//! Transport-independent MCP JSON-RPC messages.
//!
//! MCP 2026-07-28 makes requests self-contained: version, client identity, and
//! capabilities travel in `_meta` on every request. This module owns that wire
//! contract while transports own framing, authentication, and I/O.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use genos_kernel::json::JsonValue;

pub const LATEST_PROTOCOL_VERSION: &str = "2026-07-28";
pub const LEGACY_PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26"];

pub mod method {
    pub const SERVER_DISCOVER: &str = "server/discover";
    pub const TOOLS_LIST: &str = "tools/list";
    pub const TOOLS_CALL: &str = "tools/call";
    pub const RESOURCES_LIST: &str = "resources/list";
    pub const RESOURCES_READ: &str = "resources/read";
    pub const PROMPTS_LIST: &str = "prompts/list";
    pub const PROMPTS_GET: &str = "prompts/get";
    pub const COMPLETION_COMPLETE: &str = "completion/complete";
    pub const PING: &str = "ping";
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestId {
    String(String),
    Number(i64),
}

impl RequestId {
    pub fn to_json(&self) -> JsonValue {
        match self {
            Self::String(value) => JsonValue::Str(value.clone()),
            Self::Number(value) => JsonValue::Number(*value as f64),
        }
    }

    pub fn from_json(value: &JsonValue) -> Option<Self> {
        match value {
            JsonValue::Str(value) => Some(Self::String(value.clone())),
            JsonValue::Number(value) if value.is_finite() && *value == (*value as i64) as f64 => {
                Some(Self::Number(*value as i64))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientIdentity {
    pub name: String,
    pub version: String,
}

impl ClientIdentity {
    pub fn genos() -> Self {
        Self {
            name: String::from("genos"),
            version: String::from(env!("CARGO_PKG_VERSION")),
        }
    }

    fn to_json(&self) -> JsonValue {
        object(&[
            ("name", JsonValue::Str(self.name.clone())),
            ("version", JsonValue::Str(self.version.clone())),
        ])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestMetadata {
    pub protocol_version: String,
    pub client: ClientIdentity,
    pub capabilities: JsonValue,
    pub extra: BTreeMap<String, JsonValue>,
}

impl RequestMetadata {
    pub fn modern(client: ClientIdentity) -> Self {
        Self {
            protocol_version: String::from(LATEST_PROTOCOL_VERSION),
            client,
            capabilities: object(&[]),
            extra: BTreeMap::new(),
        }
    }

    pub fn with_capabilities(mut self, capabilities: JsonValue) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn to_json(&self) -> JsonValue {
        let mut values = self.extra.clone();
        values.insert(
            String::from("io.modelcontextprotocol/protocolVersion"),
            JsonValue::Str(self.protocol_version.clone()),
        );
        values.insert(
            String::from("io.modelcontextprotocol/clientInfo"),
            self.client.to_json(),
        );
        values.insert(
            String::from("io.modelcontextprotocol/clientCapabilities"),
            self.capabilities.clone(),
        );
        JsonValue::Object(values)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RpcRequest {
    pub id: RequestId,
    pub method: String,
    pub params: JsonValue,
}

impl RpcRequest {
    pub fn new(id: RequestId, method: &str, params: JsonValue, metadata: &RequestMetadata) -> Self {
        let params = attach_metadata(params, metadata.to_json());
        Self {
            id,
            method: method.to_string(),
            params,
        }
    }

    pub fn discover(id: RequestId, metadata: &RequestMetadata) -> Self {
        Self::new(id, method::SERVER_DISCOVER, object(&[]), metadata)
    }

    pub fn to_json(&self) -> JsonValue {
        object(&[
            ("jsonrpc", JsonValue::Str(String::from("2.0"))),
            ("id", self.id.to_json()),
            ("method", JsonValue::Str(self.method.clone())),
            ("params", self.params.clone()),
        ])
    }

    pub fn to_json_string(&self) -> String {
        self.to_json().to_json_string()
    }

    pub fn from_json(value: &JsonValue) -> Result<Self, McpError> {
        if value.get("jsonrpc").and_then(JsonValue::as_str) != Some("2.0") {
            return Err(McpError::invalid_request("jsonrpc must be '2.0'"));
        }
        let id = value
            .get("id")
            .and_then(RequestId::from_json)
            .ok_or_else(|| McpError::invalid_request("request id must be a string or integer"))?;
        let method = value
            .get("method")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| McpError::invalid_request("request method is missing"))?;
        let params = value.get("params").cloned().unwrap_or_else(|| object(&[]));
        if params.as_object().is_none() {
            return Err(McpError::invalid_params("params must be an object"));
        }
        Ok(Self {
            id,
            method: method.to_string(),
            params,
        })
    }

    pub fn protocol_version(&self) -> Option<&str> {
        self.params
            .get("_meta")?
            .get("io.modelcontextprotocol/protocolVersion")?
            .as_str()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RpcResponse {
    pub id: RequestId,
    pub result: Option<JsonValue>,
    pub error: Option<McpError>,
}

impl RpcResponse {
    pub fn success(id: RequestId, result: JsonValue) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn failure(id: RequestId, error: McpError) -> Self {
        Self {
            id,
            result: None,
            error: Some(error),
        }
    }

    pub fn from_json(value: &JsonValue) -> Result<Self, McpError> {
        if value.get("jsonrpc").and_then(JsonValue::as_str) != Some("2.0") {
            return Err(McpError::invalid_request("jsonrpc must be '2.0'"));
        }
        let id = value
            .get("id")
            .and_then(RequestId::from_json)
            .ok_or_else(|| McpError::invalid_request("response id is missing"))?;
        let result = value.get("result").cloned();
        let error = value.get("error").map(McpError::from_json).transpose()?;
        if result.is_some() == error.is_some() {
            return Err(McpError::invalid_request(
                "response must contain exactly one of result or error",
            ));
        }
        Ok(Self { id, result, error })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut values = BTreeMap::new();
        values.insert(String::from("jsonrpc"), JsonValue::Str(String::from("2.0")));
        values.insert(String::from("id"), self.id.to_json());
        if let Some(result) = &self.result {
            values.insert(String::from("result"), result.clone());
        }
        if let Some(error) = &self.error {
            values.insert(String::from("error"), error.to_json());
        }
        JsonValue::Object(values)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    pub data: Option<JsonValue>,
}

impl McpError {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    pub const UNSUPPORTED_PROTOCOL_VERSION: i32 = -32022;

    pub fn new(code: i32, message: &str) -> Self {
        Self {
            code,
            message: message.to_string(),
            data: None,
        }
    }

    pub fn invalid_request(message: &str) -> Self {
        Self::new(Self::INVALID_REQUEST, message)
    }

    pub fn invalid_params(message: &str) -> Self {
        Self::new(Self::INVALID_PARAMS, message)
    }

    pub fn method_not_found(method: &str) -> Self {
        Self::new(Self::METHOD_NOT_FOUND, method)
    }

    pub fn from_json(value: &JsonValue) -> Result<Self, McpError> {
        let code = value
            .get("code")
            .and_then(JsonValue::as_f64)
            .ok_or_else(|| McpError::invalid_request("error code is missing"))?
            as i32;
        let message = value
            .get("message")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| McpError::invalid_request("error message is missing"))?;
        Ok(Self {
            code,
            message: message.to_string(),
            data: value.get("data").cloned(),
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut values = BTreeMap::new();
        values.insert(String::from("code"), JsonValue::Number(self.code as f64));
        values.insert(
            String::from("message"),
            JsonValue::Str(self.message.clone()),
        );
        if let Some(data) = &self.data {
            values.insert(String::from("data"), data.clone());
        }
        JsonValue::Object(values)
    }
}

fn attach_metadata(params: JsonValue, metadata: JsonValue) -> JsonValue {
    let mut values = match params {
        JsonValue::Object(values) => values,
        _ => BTreeMap::new(),
    };
    values.insert(String::from("_meta"), metadata);
    JsonValue::Object(values)
}

pub(crate) fn object(values: &[(&str, JsonValue)]) -> JsonValue {
    let mut object = BTreeMap::new();
    for (key, value) in values {
        object.insert((*key).to_string(), value.clone());
    }
    JsonValue::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use genos_kernel::json::parse;

    #[test]
    fn modern_request_is_self_contained() {
        let metadata = RequestMetadata::modern(ClientIdentity::genos());
        let request = RpcRequest::discover(RequestId::Number(1), &metadata);
        assert_eq!(request.protocol_version(), Some(LATEST_PROTOCOL_VERSION));
        assert_eq!(request.method, method::SERVER_DISCOVER);
        assert!(request.to_json_string().contains("clientCapabilities"));
    }

    #[test]
    fn parses_error_response() {
        let json =
            parse(r#"{"jsonrpc":"2.0","id":"a","error":{"code":-32022,"message":"unsupported"}}"#)
                .unwrap();
        let response = RpcResponse::from_json(&json).unwrap();
        assert_eq!(
            response.error.unwrap().code,
            McpError::UNSUPPORTED_PROTOCOL_VERSION
        );
    }
}
