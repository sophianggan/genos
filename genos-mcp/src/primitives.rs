//! MCP primitive descriptors and deterministic, namespaced catalogs.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use genos_kernel::json::JsonValue;

#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input_schema: JsonValue,
    pub output_schema: Option<JsonValue>,
    pub annotations: Option<JsonValue>,
}

impl Tool {
    pub fn from_json(value: &JsonValue) -> Result<Self, PrimitiveError> {
        let name = required_string(value, "name")?;
        let input_schema = value
            .get("inputSchema")
            .cloned()
            .ok_or_else(|| PrimitiveError::new("tool inputSchema is required"))?;
        if input_schema.as_object().is_none() {
            return Err(PrimitiveError::new("tool inputSchema must be an object"));
        }
        Ok(Self {
            name,
            title: optional_string(value, "title")?,
            description: optional_string(value, "description")?,
            input_schema,
            output_schema: value.get("outputSchema").cloned(),
            annotations: value.get("annotations").cloned(),
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut values = BTreeMap::new();
        values.insert(String::from("name"), JsonValue::Str(self.name.clone()));
        if let Some(title) = &self.title {
            values.insert(String::from("title"), JsonValue::Str(title.clone()));
        }
        if let Some(description) = &self.description {
            values.insert(
                String::from("description"),
                JsonValue::Str(description.clone()),
            );
        }
        values.insert(String::from("inputSchema"), self.input_schema.clone());
        if let Some(output_schema) = &self.output_schema {
            values.insert(String::from("outputSchema"), output_schema.clone());
        }
        if let Some(annotations) = &self.annotations {
            values.insert(String::from("annotations"), annotations.clone());
        }
        JsonValue::Object(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resource {
    pub uri: String,
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

impl Resource {
    pub fn from_json(value: &JsonValue) -> Result<Self, PrimitiveError> {
        Ok(Self {
            uri: required_string(value, "uri")?,
            name: required_string(value, "name")?,
            title: optional_string(value, "title")?,
            description: optional_string(value, "description")?,
            mime_type: optional_string(value, "mimeType")?,
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut values = BTreeMap::new();
        values.insert(String::from("uri"), JsonValue::Str(self.uri.clone()));
        values.insert(String::from("name"), JsonValue::Str(self.name.clone()));
        if let Some(title) = &self.title {
            values.insert(String::from("title"), JsonValue::Str(title.clone()));
        }
        if let Some(description) = &self.description {
            values.insert(
                String::from("description"),
                JsonValue::Str(description.clone()),
            );
        }
        if let Some(mime_type) = &self.mime_type {
            values.insert(String::from("mimeType"), JsonValue::Str(mime_type.clone()));
        }
        JsonValue::Object(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptArgument {
    pub name: String,
    pub description: Option<String>,
    pub required: bool,
}

impl PromptArgument {
    fn from_json(value: &JsonValue) -> Result<Self, PrimitiveError> {
        Ok(Self {
            name: required_string(value, "name")?,
            description: optional_string(value, "description")?,
            required: value
                .get("required")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub arguments: Vec<PromptArgument>,
}

impl Prompt {
    pub fn from_json(value: &JsonValue) -> Result<Self, PrimitiveError> {
        let arguments = match value.get("arguments") {
            None => Vec::new(),
            Some(JsonValue::Array(arguments)) => arguments
                .iter()
                .map(PromptArgument::from_json)
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(PrimitiveError::new("prompt arguments must be an array")),
        };
        Ok(Self {
            name: required_string(value, "name")?,
            title: optional_string(value, "title")?,
            description: optional_string(value, "description")?,
            arguments,
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut values = BTreeMap::new();
        values.insert(String::from("name"), JsonValue::Str(self.name.clone()));
        if let Some(title) = &self.title {
            values.insert(String::from("title"), JsonValue::Str(title.clone()));
        }
        if let Some(description) = &self.description {
            values.insert(
                String::from("description"),
                JsonValue::Str(description.clone()),
            );
        }
        if !self.arguments.is_empty() {
            let arguments = self
                .arguments
                .iter()
                .map(|argument| {
                    let mut value = BTreeMap::new();
                    value.insert(String::from("name"), JsonValue::Str(argument.name.clone()));
                    if let Some(description) = &argument.description {
                        value.insert(
                            String::from("description"),
                            JsonValue::Str(description.clone()),
                        );
                    }
                    value.insert(String::from("required"), JsonValue::Bool(argument.required));
                    JsonValue::Object(value)
                })
                .collect();
            values.insert(String::from("arguments"), JsonValue::Array(arguments));
        }
        JsonValue::Object(values)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheScope {
    None,
    Private,
    Public,
}

impl CacheScope {
    fn from_json(value: Option<&JsonValue>) -> Result<Self, PrimitiveError> {
        match value.and_then(JsonValue::as_str) {
            None | Some("none") => Ok(Self::None),
            Some("private") => Ok(Self::Private),
            Some("public") => Ok(Self::Public),
            Some(_) => Err(PrimitiveError::new("invalid MCP cacheScope")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
    pub ttl_ms: Option<u64>,
    pub cache_scope: CacheScope,
}

impl Page<Tool> {
    pub fn tools(result: &JsonValue) -> Result<Self, PrimitiveError> {
        parse_page(result, "tools", Tool::from_json)
    }
}

impl Page<Resource> {
    pub fn resources(result: &JsonValue) -> Result<Self, PrimitiveError> {
        parse_page(result, "resources", Resource::from_json)
    }
}

impl Page<Prompt> {
    pub fn prompts(result: &JsonValue) -> Result<Self, PrimitiveError> {
        parse_page(result, "prompts", Prompt::from_json)
    }
}

fn parse_page<T, F>(
    result: &JsonValue,
    field: &str,
    parse_item: F,
) -> Result<Page<T>, PrimitiveError>
where
    F: Fn(&JsonValue) -> Result<T, PrimitiveError>,
{
    let values = result
        .get(field)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| PrimitiveError::new("list result is missing its item array"))?;
    let items = values
        .iter()
        .map(parse_item)
        .collect::<Result<Vec<_>, _>>()?;
    let ttl_ms = result.get("ttlMs").and_then(JsonValue::as_f64).map(|ttl| {
        if ttl.is_sign_negative() {
            0
        } else {
            ttl as u64
        }
    });
    Ok(Page {
        items,
        next_cursor: optional_string(result, "nextCursor")?,
        ttl_ms,
        cache_scope: CacheScope::from_json(result.get("cacheScope"))?,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServerCatalog {
    pub server_id: String,
    pub tools: Vec<Tool>,
    pub resources: Vec<Resource>,
    pub prompts: Vec<Prompt>,
    pub fetched_at_ms: u64,
    pub ttl_ms: Option<u64>,
    pub cache_scope: CacheScope,
}

impl ServerCatalog {
    pub fn new(server_id: &str) -> Self {
        Self {
            server_id: server_id.to_string(),
            tools: Vec::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
            fetched_at_ms: 0,
            ttl_ms: None,
            cache_scope: CacheScope::None,
        }
    }

    pub fn sort(&mut self) {
        self.tools.sort_by(|left, right| left.name.cmp(&right.name));
        self.resources
            .sort_by(|left, right| left.uri.cmp(&right.uri));
        self.prompts
            .sort_by(|left, right| left.name.cmp(&right.name));
    }

    pub fn is_fresh(&self, now_ms: u64) -> bool {
        match self.ttl_ms {
            Some(ttl) => now_ms.saturating_sub(self.fetched_at_ms) < ttl,
            None => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CatalogTool {
    pub server_id: String,
    pub exposed_name: String,
    pub tool: Tool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Catalog {
    servers: BTreeMap<String, ServerCatalog>,
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            servers: BTreeMap::new(),
        }
    }

    pub fn replace(&mut self, mut catalog: ServerCatalog) {
        catalog.sort();
        self.servers.insert(catalog.server_id.clone(), catalog);
    }

    pub fn remove(&mut self, server_id: &str) -> Option<ServerCatalog> {
        self.servers.remove(server_id)
    }

    pub fn server(&self, server_id: &str) -> Option<&ServerCatalog> {
        self.servers.get(server_id)
    }

    pub fn tools(&self) -> Vec<CatalogTool> {
        let mut tools = Vec::new();
        for (server_id, catalog) in &self.servers {
            for tool in &catalog.tools {
                tools.push(CatalogTool {
                    server_id: server_id.clone(),
                    exposed_name: namespaced(server_id, &tool.name),
                    tool: tool.clone(),
                });
            }
        }
        tools
    }

    pub fn resolve_tool(&self, exposed_name: &str) -> Option<(&str, &Tool)> {
        let (server_id, primitive_name) = split_namespaced(exposed_name)?;
        let catalog = self.servers.get(server_id)?;
        let tool = catalog
            .tools
            .iter()
            .find(|tool| tool.name == primitive_name)?;
        Some((catalog.server_id.as_str(), tool))
    }

    pub fn server_count(&self) -> usize {
        self.servers.len()
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

pub fn namespaced(server_id: &str, primitive_name: &str) -> String {
    let mut value = String::from("mcp.");
    value.push_str(server_id);
    value.push('.');
    value.push_str(primitive_name);
    value
}

pub fn split_namespaced(value: &str) -> Option<(&str, &str)> {
    let rest = value.strip_prefix("mcp.")?;
    let (server, primitive) = rest.split_once('.')?;
    if server.is_empty() || primitive.is_empty() {
        return None;
    }
    Some((server, primitive))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimitiveError {
    pub message: String,
}

impl PrimitiveError {
    fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

fn required_string(value: &JsonValue, field: &str) -> Result<String, PrimitiveError> {
    value
        .get(field)
        .and_then(JsonValue::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| PrimitiveError::new("required string field is missing"))
}

fn optional_string(value: &JsonValue, field: &str) -> Result<Option<String>, PrimitiveError> {
    match value.get(field) {
        None => Ok(None),
        Some(JsonValue::Str(value)) => Ok(Some(value.clone())),
        Some(_) => Err(PrimitiveError::new("optional string field has wrong type")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genos_kernel::json::parse;

    #[test]
    fn parses_and_orders_tool_catalogs() {
        let result = parse(
            r#"{
                "resultType":"complete",
                "tools":[
                    {"name":"zeta","inputSchema":{"type":"object"}},
                    {"name":"alpha","description":"first","inputSchema":{"type":"object"}}
                ],
                "ttlMs":60000,
                "cacheScope":"private"
            }"#,
        )
        .unwrap();
        let page = Page::<Tool>::tools(&result).unwrap();
        let mut server = ServerCatalog::new("demo");
        server.tools = page.items;
        server.ttl_ms = page.ttl_ms;
        server.cache_scope = page.cache_scope;
        let mut catalog = Catalog::new();
        catalog.replace(server);
        let tools = catalog.tools();
        assert_eq!(tools[0].exposed_name, "mcp.demo.alpha");
        assert_eq!(tools[1].exposed_name, "mcp.demo.zeta");
        assert_eq!(
            catalog.resolve_tool("mcp.demo.alpha").unwrap().1.name,
            "alpha"
        );
    }

    #[test]
    fn rejects_malformed_descriptors() {
        let missing_schema = parse(r#"{"name":"unsafe"}"#).unwrap();
        assert!(Tool::from_json(&missing_schema).is_err());
        assert!(split_namespaced("plain.tool").is_none());
    }
}
