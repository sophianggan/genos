//! Provider-neutral MCP endpoint configuration.
//!
//! The parser intentionally supports a small, auditable TOML subset. Secrets
//! are never accepted inline: authentication always names a credential source.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransportKind {
    StreamableHttp,
    StdioBridge,
    Custom(String),
}

impl TransportKind {
    fn parse(value: &str) -> Self {
        match value {
            "streamable_http" => Self::StreamableHttp,
            "stdio_bridge" => Self::StdioBridge,
            other => Self::Custom(other.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrustLevel {
    Untrusted,
    ReadOnly,
    Trusted,
}

impl TrustLevel {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "untrusted" => Some(Self::Untrusted),
            "read_only" => Some(Self::ReadOnly),
            "trusted" => Some(Self::Trusted),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialRef {
    Environment(String),
    File(String),
    OAuthProfile(String),
    FirmwareVariable(String),
}

impl CredentialRef {
    pub fn parse(value: &str) -> Result<Self, &'static str> {
        let (scheme, locator) = value
            .split_once(':')
            .ok_or("credential_ref must use scheme:value")?;
        if locator.trim().is_empty() {
            return Err("credential_ref locator cannot be empty");
        }
        match scheme {
            "env" => Ok(Self::Environment(locator.to_string())),
            "file" => Ok(Self::File(locator.to_string())),
            "oauth" => Ok(Self::OAuthProfile(locator.to_string())),
            "firmware" => Ok(Self::FirmwareVariable(locator.to_string())),
            _ => Err("unsupported credential_ref scheme"),
        }
    }

    /// Safe representation for status screens and audit logs.
    pub fn redacted(&self) -> String {
        match self {
            Self::Environment(_) => String::from("env:[redacted]"),
            Self::File(_) => String::from("file:[redacted]"),
            Self::OAuthProfile(_) => String::from("oauth:[redacted]"),
            Self::FirmwareVariable(_) => String::from("firmware:[redacted]"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthConfig {
    None,
    Bearer {
        credential: CredentialRef,
    },
    ApiKey {
        header: String,
        credential: CredentialRef,
    },
    OAuth {
        profile: CredentialRef,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpDefaults {
    pub timeout_ms: u64,
    pub max_response_bytes: usize,
    pub require_approval: bool,
}

impl Default for McpDefaults {
    fn default() -> Self {
        Self {
            timeout_ms: 15_000,
            max_response_bytes: 1_048_576,
            require_approval: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerConfig {
    pub id: String,
    pub transport: TransportKind,
    pub endpoint: String,
    pub enabled: bool,
    pub auth: AuthConfig,
    pub trust: TrustLevel,
    pub allow_tools: bool,
    pub allow_resources: bool,
    pub allow_prompts: bool,
    pub require_approval: bool,
    pub timeout_ms: u64,
    pub max_response_bytes: usize,
    pub allowed_tools: Vec<String>,
}

impl ServerConfig {
    pub fn namespaced(&self, primitive: &str) -> String {
        let mut name = String::from("mcp.");
        name.push_str(&self.id);
        name.push('.');
        name.push_str(primitive);
        name
    }

    pub fn is_authenticated(&self) -> bool {
        !matches!(self.auth, AuthConfig::None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpConfig {
    pub defaults: McpDefaults,
    pub servers: Vec<ServerConfig>,
}

impl McpConfig {
    pub fn empty() -> Self {
        Self {
            defaults: McpDefaults::default(),
            servers: Vec::new(),
        }
    }

    pub fn parse(data: &[u8]) -> Result<Self, ConfigError> {
        let text = core::str::from_utf8(data)
            .map_err(|_| ConfigError::new(0, "configuration must be UTF-8"))?;
        let mut config = Self::empty();
        let mut section = Section::Root;
        let mut pending: Option<PendingServer> = None;

        for (index, raw_line) in text.lines().enumerate() {
            let line_number = index + 1;
            let line = strip_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }
            if line == "[defaults]" {
                finish_pending(&mut config, &mut pending, line_number)?;
                section = Section::Defaults;
                continue;
            }
            if line == "[[servers]]" {
                finish_pending(&mut config, &mut pending, line_number)?;
                pending = Some(PendingServer::from_defaults(&config.defaults));
                section = Section::Server;
                continue;
            }

            let (key, raw_value) = line
                .split_once('=')
                .ok_or_else(|| ConfigError::new(line_number, "expected key = value"))?;
            let key = key.trim();
            let value = raw_value.trim();
            match section {
                Section::Defaults => parse_default(&mut config.defaults, key, value, line_number)?,
                Section::Server => parse_server(
                    pending
                        .as_mut()
                        .ok_or_else(|| ConfigError::new(line_number, "missing [[servers]]"))?,
                    key,
                    value,
                    line_number,
                )?,
                Section::Root => {
                    return Err(ConfigError::new(
                        line_number,
                        "keys must be inside [defaults] or [[servers]]",
                    ))
                }
            }
        }
        finish_pending(&mut config, &mut pending, text.lines().count() + 1)?;
        ensure_unique_ids(&config.servers)?;
        Ok(config)
    }

    pub fn enabled_servers(&self) -> impl Iterator<Item = &ServerConfig> {
        self.servers.iter().filter(|server| server.enabled)
    }

    pub fn find(&self, id: &str) -> Option<&ServerConfig> {
        self.servers.iter().find(|server| server.id == id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
    pub line: usize,
    pub message: String,
}

impl ConfigError {
    fn new(line: usize, message: &str) -> Self {
        Self {
            line,
            message: message.to_string(),
        }
    }
}

enum Section {
    Root,
    Defaults,
    Server,
}

struct PendingServer {
    id: Option<String>,
    transport: TransportKind,
    endpoint: Option<String>,
    enabled: bool,
    auth_kind: String,
    credential_ref: Option<String>,
    header_name: String,
    trust: TrustLevel,
    allow_tools: bool,
    allow_resources: bool,
    allow_prompts: bool,
    require_approval: bool,
    timeout_ms: u64,
    max_response_bytes: usize,
    allowed_tools: Vec<String>,
}

impl PendingServer {
    fn from_defaults(defaults: &McpDefaults) -> Self {
        Self {
            id: None,
            transport: TransportKind::StreamableHttp,
            endpoint: None,
            enabled: true,
            auth_kind: String::from("none"),
            credential_ref: None,
            header_name: String::from("X-API-Key"),
            trust: TrustLevel::Untrusted,
            allow_tools: true,
            allow_resources: true,
            allow_prompts: true,
            require_approval: defaults.require_approval,
            timeout_ms: defaults.timeout_ms,
            max_response_bytes: defaults.max_response_bytes,
            allowed_tools: Vec::new(),
        }
    }

    fn finish(self, line: usize) -> Result<ServerConfig, ConfigError> {
        let id = self
            .id
            .ok_or_else(|| ConfigError::new(line, "server id is required"))?;
        if !valid_id(&id) {
            return Err(ConfigError::new(
                line,
                "server id may contain only lowercase letters, digits, '-' and '_'",
            ));
        }
        let endpoint = self
            .endpoint
            .ok_or_else(|| ConfigError::new(line, "server endpoint is required"))?;
        let credential = match self.credential_ref {
            Some(value) => Some(
                CredentialRef::parse(&value).map_err(|message| ConfigError::new(line, message))?,
            ),
            None => None,
        };
        let auth = match self.auth_kind.as_str() {
            "none" => {
                if credential.is_some() {
                    return Err(ConfigError::new(
                        line,
                        "credential_ref is not valid when auth = 'none'",
                    ));
                }
                AuthConfig::None
            }
            "bearer" => AuthConfig::Bearer {
                credential: credential
                    .ok_or_else(|| ConfigError::new(line, "bearer auth requires credential_ref"))?,
            },
            "api_key" => AuthConfig::ApiKey {
                header: self.header_name,
                credential: credential.ok_or_else(|| {
                    ConfigError::new(line, "api_key auth requires credential_ref")
                })?,
            },
            "oauth" => {
                let profile = credential
                    .ok_or_else(|| ConfigError::new(line, "oauth auth requires credential_ref"))?;
                if !matches!(profile, CredentialRef::OAuthProfile(_)) {
                    return Err(ConfigError::new(
                        line,
                        "oauth auth requires an oauth: credential_ref",
                    ));
                }
                AuthConfig::OAuth { profile }
            }
            _ => return Err(ConfigError::new(line, "unsupported auth mode")),
        };
        if self.max_response_bytes == 0 {
            return Err(ConfigError::new(
                line,
                "max_response_bytes must be positive",
            ));
        }
        Ok(ServerConfig {
            id,
            transport: self.transport,
            endpoint,
            enabled: self.enabled,
            auth,
            trust: self.trust,
            allow_tools: self.allow_tools,
            allow_resources: self.allow_resources,
            allow_prompts: self.allow_prompts,
            require_approval: self.require_approval,
            timeout_ms: self.timeout_ms,
            max_response_bytes: self.max_response_bytes,
            allowed_tools: self.allowed_tools,
        })
    }
}

fn finish_pending(
    config: &mut McpConfig,
    pending: &mut Option<PendingServer>,
    line: usize,
) -> Result<(), ConfigError> {
    if let Some(server) = pending.take() {
        config.servers.push(server.finish(line)?);
    }
    Ok(())
}

fn parse_default(
    defaults: &mut McpDefaults,
    key: &str,
    value: &str,
    line: usize,
) -> Result<(), ConfigError> {
    match key {
        "timeout_ms" => defaults.timeout_ms = parse_u64(value, line)?,
        "max_response_bytes" => defaults.max_response_bytes = parse_usize(value, line)?,
        "require_approval" => defaults.require_approval = parse_bool(value, line)?,
        _ => return Err(ConfigError::new(line, "unknown defaults key")),
    }
    Ok(())
}

fn parse_server(
    server: &mut PendingServer,
    key: &str,
    value: &str,
    line: usize,
) -> Result<(), ConfigError> {
    match key {
        "id" => server.id = Some(parse_string(value, line)?),
        "transport" => server.transport = TransportKind::parse(&parse_string(value, line)?),
        "endpoint" => server.endpoint = Some(parse_string(value, line)?),
        "enabled" => server.enabled = parse_bool(value, line)?,
        "auth" => server.auth_kind = parse_string(value, line)?,
        "credential_ref" => server.credential_ref = Some(parse_string(value, line)?),
        "header_name" => server.header_name = parse_string(value, line)?,
        "trust" => {
            let trust = parse_string(value, line)?;
            server.trust = TrustLevel::parse(&trust)
                .ok_or_else(|| ConfigError::new(line, "invalid trust level"))?;
        }
        "allow_tools" => server.allow_tools = parse_bool(value, line)?,
        "allow_resources" => server.allow_resources = parse_bool(value, line)?,
        "allow_prompts" => server.allow_prompts = parse_bool(value, line)?,
        "require_approval" => server.require_approval = parse_bool(value, line)?,
        "timeout_ms" => server.timeout_ms = parse_u64(value, line)?,
        "max_response_bytes" => server.max_response_bytes = parse_usize(value, line)?,
        "allowed_tools" => server.allowed_tools = parse_string_array(value, line)?,
        _ => return Err(ConfigError::new(line, "unknown server key")),
    }
    Ok(())
}

fn parse_string(value: &str, line: usize) -> Result<String, ConfigError> {
    let value = value.trim();
    if value.len() < 2 || !value.starts_with('"') || !value.ends_with('"') {
        return Err(ConfigError::new(line, "expected quoted string"));
    }
    Ok(value[1..value.len() - 1].to_string())
}

fn parse_string_array(value: &str, line: usize) -> Result<Vec<String>, ConfigError> {
    let value = value.trim();
    if !value.starts_with('[') || !value.ends_with(']') {
        return Err(ConfigError::new(line, "expected string array"));
    }
    let inner = value[1..value.len() - 1].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|part| parse_string(part.trim(), line))
        .collect()
}

fn parse_bool(value: &str, line: usize) -> Result<bool, ConfigError> {
    match value.trim() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ConfigError::new(line, "expected boolean")),
    }
}

fn parse_u64(value: &str, line: usize) -> Result<u64, ConfigError> {
    let mut result = 0u64;
    let value = value.trim();
    if value.is_empty() {
        return Err(ConfigError::new(line, "expected positive integer"));
    }
    for byte in value.bytes() {
        if !byte.is_ascii_digit() {
            return Err(ConfigError::new(line, "expected positive integer"));
        }
        result = result
            .checked_mul(10)
            .and_then(|number| number.checked_add((byte - b'0') as u64))
            .ok_or_else(|| ConfigError::new(line, "integer overflow"))?;
    }
    Ok(result)
}

fn parse_usize(value: &str, line: usize) -> Result<usize, ConfigError> {
    usize::try_from(parse_u64(value, line)?).map_err(|_| ConfigError::new(line, "integer overflow"))
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
}

fn ensure_unique_ids(servers: &[ServerConfig]) -> Result<(), ConfigError> {
    for (index, server) in servers.iter().enumerate() {
        if servers[..index]
            .iter()
            .any(|candidate| candidate.id == server.id)
        {
            return Err(ConfigError::new(0, "server ids must be unique"));
        }
    }
    Ok(())
}

fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    for (index, byte) in line.bytes().enumerate() {
        match byte {
            b'"' => quoted = !quoted,
            b'#' if !quoted => return &line[..index],
            _ => {}
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiple_provider_neutral_servers() {
        let config = McpConfig::parse(
            br#"
            [defaults]
            timeout_ms = 9000
            require_approval = true

            [[servers]]
            id = "search"
            endpoint = "https://mcp.example.com/mcp"
            auth = "bearer"
            credential_ref = "env:SEARCH_TOKEN"
            allowed_tools = ["search", "fetch"]

            [[servers]]
            id = "local"
            transport = "stdio_bridge"
            endpoint = "http://127.0.0.1:8787/mcp"
            trust = "read_only"
            "#,
        )
        .unwrap();
        assert_eq!(config.servers.len(), 2);
        assert_eq!(config.servers[0].timeout_ms, 9000);
        assert_eq!(config.servers[0].namespaced("search"), "mcp.search.search");
        assert!(matches!(config.servers[0].auth, AuthConfig::Bearer { .. }));
        assert_eq!(config.servers[1].transport, TransportKind::StdioBridge);
    }

    #[test]
    fn rejects_inline_credentials_and_duplicate_ids() {
        assert!(CredentialRef::parse("secret-token-without-source").is_err());
        let duplicate = br#"
            [[servers]]
            id = "same"
            endpoint = "https://one.example/mcp"
            [[servers]]
            id = "same"
            endpoint = "https://two.example/mcp"
        "#;
        assert!(McpConfig::parse(duplicate).is_err());
    }
}
