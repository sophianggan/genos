//! Host-enforced security boundary for untrusted MCP servers.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use genos_kernel::json::JsonValue;

use crate::config::{AuthConfig, CredentialRef, ServerConfig, TrustLevel};
use crate::primitives::Tool;
use crate::transport::Header;

/// Secret bytes with redacted formatting and best-effort memory clearing.
pub struct Secret {
    bytes: Vec<u8>,
}

impl Secret {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn expose(&self) -> &[u8] {
        &self.bytes
    }
}

impl core::fmt::Debug for Secret {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("Secret([redacted])")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.bytes.fill(0);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialError {
    pub message: String,
}

impl CredentialError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

pub trait CredentialResolver {
    fn resolve(&mut self, reference: &CredentialRef) -> Result<Secret, CredentialError>;
}

/// Request-local authentication headers. Sensitive strings are cleared on
/// drop; they must never be cloned into logs or long-lived state.
pub struct ResolvedAuth {
    headers: Vec<Header>,
}

impl ResolvedAuth {
    pub fn none() -> Self {
        Self {
            headers: Vec::new(),
        }
    }

    pub fn headers(&self) -> &[Header] {
        &self.headers
    }
}

impl core::fmt::Debug for ResolvedAuth {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResolvedAuth")
            .field("headers", &"[redacted]")
            .finish()
    }
}

impl Drop for ResolvedAuth {
    fn drop(&mut self) {
        for header in &mut self.headers {
            if header.sensitive {
                // Replacing the value prevents accidental later access. The
                // Secret wrapper clears the source allocation separately.
                header.value.clear();
            }
        }
    }
}

pub fn resolve_auth(
    auth: &AuthConfig,
    resolver: &mut dyn CredentialResolver,
) -> Result<ResolvedAuth, CredentialError> {
    let (name, prefix, reference) = match auth {
        AuthConfig::None => return Ok(ResolvedAuth::none()),
        AuthConfig::Bearer { credential } => ("Authorization", "Bearer ", credential),
        AuthConfig::ApiKey { header, credential } => (header.as_str(), "", credential),
        AuthConfig::OAuth { profile } => ("Authorization", "Bearer ", profile),
    };
    let secret = resolver.resolve(reference)?;
    let value = core::str::from_utf8(secret.expose())
        .map_err(|_| CredentialError::new("credential must be UTF-8"))?;
    if value.is_empty() {
        return Err(CredentialError::new("credential is empty"));
    }
    if value.bytes().any(|byte| byte == b'\r' || byte == b'\n') {
        return Err(CredentialError::new("credential contains a line break"));
    }
    let mut header_value = String::from(prefix);
    header_value.push_str(value);
    Ok(ResolvedAuth {
        headers: alloc::vec![Header::sensitive(name, &header_value)],
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApprovalGrant {
    pub server_id: String,
    pub tool_name: String,
    pub exact_arguments: JsonValue,
    pub expires_at_ms: u64,
}

impl ApprovalGrant {
    pub fn matches(
        &self,
        server_id: &str,
        tool_name: &str,
        arguments: &JsonValue,
        now_ms: u64,
    ) -> bool {
        now_ms <= self.expires_at_ms
            && self.server_id == server_id
            && self.tool_name == tool_name
            && self.exact_arguments == *arguments
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyDecision {
    Allow,
    ApprovalRequired { reason: String },
    Deny { reason: String },
    RateLimited,
    CircuitOpen,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ServerBudget {
    calls_this_turn: usize,
    consecutive_failures: usize,
    circuit_open_until_ms: u64,
}

impl ServerBudget {
    fn new() -> Self {
        Self {
            calls_this_turn: 0,
            consecutive_failures: 0,
            circuit_open_until_ms: 0,
        }
    }
}

pub struct PolicyFirewall {
    max_calls_per_turn: usize,
    max_argument_bytes: usize,
    failure_threshold: usize,
    circuit_cooldown_ms: u64,
    budgets: BTreeMap<String, ServerBudget>,
}

impl PolicyFirewall {
    pub fn new() -> Self {
        Self {
            max_calls_per_turn: 10,
            max_argument_bytes: 64 * 1024,
            failure_threshold: 3,
            circuit_cooldown_ms: 30_000,
            budgets: BTreeMap::new(),
        }
    }

    pub fn reset_turn(&mut self) {
        for budget in self.budgets.values_mut() {
            budget.calls_this_turn = 0;
        }
    }

    pub fn check_tool(
        &mut self,
        server: &ServerConfig,
        tool: &Tool,
        arguments: &JsonValue,
        approval: Option<&ApprovalGrant>,
        now_ms: u64,
    ) -> PolicyDecision {
        if !server.enabled || !server.allow_tools {
            return PolicyDecision::Deny {
                reason: String::from("server or tool capability is disabled"),
            };
        }
        if arguments.as_object().is_none() {
            return PolicyDecision::Deny {
                reason: String::from("tool arguments must be an object"),
            };
        }
        if arguments.to_json_string().len() > self.max_argument_bytes {
            return PolicyDecision::Deny {
                reason: String::from("tool arguments exceed the configured limit"),
            };
        }
        if !server.allowed_tools.is_empty()
            && !server
                .allowed_tools
                .iter()
                .any(|allowed| allowed == &tool.name)
        {
            return PolicyDecision::Deny {
                reason: String::from("tool is not in the endpoint allowlist"),
            };
        }

        let budget = self
            .budgets
            .entry(server.id.clone())
            .or_insert_with(ServerBudget::new);
        if budget.circuit_open_until_ms > now_ms {
            return PolicyDecision::CircuitOpen;
        }
        if budget.calls_this_turn >= self.max_calls_per_turn {
            return PolicyDecision::RateLimited;
        }

        let read_only_hint = annotation_bool(tool, "readOnlyHint");
        let destructive_hint = annotation_bool(tool, "destructiveHint");
        if server.trust == TrustLevel::ReadOnly && read_only_hint != Some(true) {
            return PolicyDecision::Deny {
                reason: String::from(
                    "read-only endpoint policy requires an explicitly read-only tool",
                ),
            };
        }

        let exact_approval = approval
            .map(|grant| grant.matches(&server.id, &tool.name, arguments, now_ms))
            .unwrap_or(false);
        let approval_required = server.require_approval
            || server.trust == TrustLevel::Untrusted
            || destructive_hint == Some(true);
        if approval_required && !exact_approval {
            return PolicyDecision::ApprovalRequired {
                reason: if destructive_hint == Some(true) {
                    String::from("tool is annotated as destructive")
                } else {
                    String::from("endpoint policy requires exact-call approval")
                },
            };
        }

        budget.calls_this_turn = budget.calls_this_turn.saturating_add(1);
        PolicyDecision::Allow
    }

    pub fn record_result(&mut self, server_id: &str, success: bool, now_ms: u64) {
        let budget = self
            .budgets
            .entry(server_id.to_string())
            .or_insert_with(ServerBudget::new);
        if success {
            budget.consecutive_failures = 0;
            budget.circuit_open_until_ms = 0;
        } else {
            budget.consecutive_failures = budget.consecutive_failures.saturating_add(1);
            if budget.consecutive_failures >= self.failure_threshold {
                budget.circuit_open_until_ms = now_ms.saturating_add(self.circuit_cooldown_ms);
                budget.consecutive_failures = 0;
            }
        }
    }
}

impl Default for PolicyFirewall {
    fn default() -> Self {
        Self::new()
    }
}

fn annotation_bool(tool: &Tool, key: &str) -> Option<bool> {
    tool.annotations
        .as_ref()
        .and_then(|annotations| annotations.get(key))
        .and_then(JsonValue::as_bool)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub server_id: String,
    pub primitive: String,
    pub trust: TrustLevel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GuardedResult {
    pub value: JsonValue,
    pub provenance: Provenance,
    /// External content is data, never an instruction source.
    pub untrusted_content: bool,
    pub truncated: bool,
}

impl GuardedResult {
    pub fn filter(value: &JsonValue, provenance: Provenance, max_bytes: usize) -> Self {
        let mut redacted = redact_json(value);
        let mut truncated = false;
        if redacted.to_json_string().len() > max_bytes {
            redacted = object(&[
                ("truncated", JsonValue::Bool(true)),
                (
                    "message",
                    JsonValue::Str(String::from("MCP result exceeded the host output limit")),
                ),
            ]);
            truncated = true;
        }
        Self {
            value: redacted,
            untrusted_content: provenance.trust != TrustLevel::Trusted,
            provenance,
            truncated,
        }
    }
}

fn redact_json(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(values) => {
            let mut output = BTreeMap::new();
            for (key, value) in values {
                let normalized = key.to_ascii_lowercase().replace('-', "_");
                let sensitive = matches!(
                    normalized.as_str(),
                    "authorization"
                        | "password"
                        | "token"
                        | "access_token"
                        | "refresh_token"
                        | "api_key"
                        | "secret"
                        | "cookie"
                );
                output.insert(
                    key.clone(),
                    if sensitive {
                        JsonValue::Str(String::from("[redacted]"))
                    } else {
                        redact_json(value)
                    },
                );
            }
            JsonValue::Object(output)
        }
        JsonValue::Array(values) => JsonValue::Array(values.iter().map(redact_json).collect()),
        other => other.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditEvent {
    pub timestamp_ms: u64,
    pub server_id: String,
    pub operation: String,
    pub decision: String,
    pub request_id: Option<String>,
}

pub struct AuditLog {
    capacity: usize,
    events: Vec<AuditEvent>,
}

impl AuditLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            events: Vec::new(),
        }
    }

    pub fn record(&mut self, event: AuditEvent) {
        if self.events.len() == self.capacity {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    pub fn events(&self) -> &[AuditEvent] {
        &self.events
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
    use super::*;
    use crate::config::{McpConfig, ServerConfig};
    use genos_kernel::json::parse;

    struct TestResolver;

    impl CredentialResolver for TestResolver {
        fn resolve(&mut self, _reference: &CredentialRef) -> Result<Secret, CredentialError> {
            Ok(Secret::new(b"test-token".to_vec()))
        }
    }

    fn server() -> ServerConfig {
        McpConfig::parse(
            br#"
            [[servers]]
            id = "demo"
            endpoint = "https://example.com/mcp"
            auth = "bearer"
            credential_ref = "env:MCP_TOKEN"
            "#,
        )
        .unwrap()
        .servers
        .remove(0)
    }

    fn tool(annotations: Option<JsonValue>) -> Tool {
        Tool {
            name: String::from("write"),
            title: None,
            description: None,
            input_schema: parse(r#"{"type":"object"}"#).unwrap(),
            output_schema: None,
            annotations,
        }
    }

    #[test]
    fn resolves_auth_without_exposing_debug_value() {
        let server = server();
        let mut resolver = TestResolver;
        let auth = resolve_auth(&server.auth, &mut resolver).unwrap();
        assert_eq!(auth.headers()[0].value, "Bearer test-token");
        assert_eq!(
            alloc::format!("{:?}", auth),
            "ResolvedAuth { headers: \"[redacted]\" }"
        );
    }

    #[test]
    fn approval_is_bound_to_server_tool_arguments_and_expiry() {
        let server = server();
        let arguments = parse(r#"{"path":"/data/a"}"#).unwrap();
        let grant = ApprovalGrant {
            server_id: server.id.clone(),
            tool_name: String::from("write"),
            exact_arguments: arguments.clone(),
            expires_at_ms: 100,
        };
        let mut firewall = PolicyFirewall::new();
        assert_eq!(
            firewall.check_tool(&server, &tool(None), &arguments, Some(&grant), 99),
            PolicyDecision::Allow
        );
        let changed = parse(r#"{"path":"/data/b"}"#).unwrap();
        assert!(matches!(
            firewall.check_tool(&server, &tool(None), &changed, Some(&grant), 99),
            PolicyDecision::ApprovalRequired { .. }
        ));
    }

    #[test]
    fn output_filter_redacts_secrets_and_labels_provenance() {
        let value =
            parse(r#"{"answer":"ok","access_token":"do-not-log","nested":{"password":"nope"}}"#)
                .unwrap();
        let guarded = GuardedResult::filter(
            &value,
            Provenance {
                server_id: String::from("demo"),
                primitive: String::from("read"),
                trust: TrustLevel::Untrusted,
            },
            4096,
        );
        assert_eq!(
            guarded.value.get("access_token").unwrap().as_str(),
            Some("[redacted]")
        );
        assert!(guarded.untrusted_content);
    }
}
