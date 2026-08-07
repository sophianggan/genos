//! Policy gating layer for tool calls.
//!
//! Per-tool flags: enabled, require_approval, max_calls_per_turn.
//! Any tool call rejected by policy returns a typed error.
//! Config loaded from \system\config.toml [policy] section.

use alloc::collections::BTreeMap;
use alloc::string::String;

/// Per-tool policy configuration.
pub struct ToolPolicy {
    pub enabled: bool,
    pub require_approval: bool,
    pub max_calls_per_turn: usize,
}

impl ToolPolicy {
    fn default_allowed() -> Self {
        ToolPolicy {
            enabled: true,
            require_approval: false,
            max_calls_per_turn: 10,
        }
    }

    fn default_protected() -> Self {
        ToolPolicy {
            enabled: true,
            require_approval: true,
            max_calls_per_turn: 5,
        }
    }
}

/// Policy engine — checks tool calls against policy before execution.
pub struct PolicyEngine {
    policies: BTreeMap<String, ToolPolicy>,
    /// Per-turn call counts (reset each turn).
    call_counts: BTreeMap<String, usize>,
}

/// Result of a policy check.
pub enum PolicyCheck {
    /// Tool call is allowed to proceed.
    Allowed,
    /// Tool is disabled by policy.
    Disabled,
    /// Tool requires user approval before execution.
    NeedsApproval,
    /// Tool has exceeded max calls per turn.
    RateLimited,
}

impl PolicyEngine {
    /// Create with default policies for all Phase B tools.
    pub fn new() -> Self {
        let mut policies = BTreeMap::new();

        // Read tools — always allowed
        policies.insert(String::from("fs.read"), ToolPolicy::default_allowed());
        policies.insert(String::from("fs.list"), ToolPolicy::default_allowed());
        policies.insert(String::from("memory.facts_get"), ToolPolicy::default_allowed());
        policies.insert(String::from("memory.log_turn"), ToolPolicy::default_allowed());
        policies.insert(String::from("sys.clock"), ToolPolicy::default_allowed());
        policies.insert(String::from("sys.introspect"), ToolPolicy::default_allowed());

        // Write tools — allowed but tracked
        policies.insert(String::from("fs.write"), ToolPolicy::default_allowed());
        policies.insert(String::from("memory.facts_set"), ToolPolicy::default_allowed());

        // Net — allowed (gracefully degrades)
        policies.insert(String::from("net.fetch"), ToolPolicy::default_allowed());

        // MCP management is host-checked again by the MCP policy firewall.
        policies.insert(String::from("mcp.servers"), ToolPolicy::default_allowed());
        policies.insert(String::from("mcp.connect"), ToolPolicy::default_allowed());
        policies.insert(String::from("mcp.tools"), ToolPolicy::default_allowed());
        policies.insert(String::from("mcp.call"), ToolPolicy::default_allowed());
        policies.insert(String::from("mcp.resource"), ToolPolicy::default_allowed());
        policies.insert(String::from("mcp.prompt"), ToolPolicy::default_allowed());

        // Destructive tools — require approval
        policies.insert(String::from("fs.delete"), ToolPolicy::default_protected());

        PolicyEngine {
            policies,
            call_counts: BTreeMap::new(),
        }
    }

    /// Check if a tool call is allowed by policy.
    pub fn check(&self, tool_name: &str) -> PolicyCheck {
        let policy = match self.policies.get(tool_name) {
            Some(p) => p,
            None => return PolicyCheck::Disabled, // Unknown tools are disabled
        };

        if !policy.enabled {
            return PolicyCheck::Disabled;
        }

        if policy.require_approval {
            return PolicyCheck::NeedsApproval;
        }

        // Check rate limit
        let count = self.call_counts.get(tool_name).copied().unwrap_or(0);
        if count >= policy.max_calls_per_turn {
            return PolicyCheck::RateLimited;
        }

        PolicyCheck::Allowed
    }

    /// Record that a tool was called (for rate limiting).
    pub fn record_call(&mut self, tool_name: &str) {
        let count = self.call_counts.entry(String::from(tool_name)).or_insert(0);
        *count += 1;
    }

    /// Reset per-turn call counts (call at start of each turn).
    pub fn reset_turn(&mut self) {
        self.call_counts.clear();
    }

    /// Set policy for a specific tool.
    pub fn set_policy(&mut self, tool_name: &str, policy: ToolPolicy) {
        self.policies.insert(String::from(tool_name), policy);
    }

    /// Disable a tool entirely.
    pub fn disable_tool(&mut self, tool_name: &str) {
        if let Some(p) = self.policies.get_mut(tool_name) {
            p.enabled = false;
        }
    }

    /// Enable a tool.
    pub fn enable_tool(&mut self, tool_name: &str) {
        if let Some(p) = self.policies.get_mut(tool_name) {
            p.enabled = true;
        }
    }
}
