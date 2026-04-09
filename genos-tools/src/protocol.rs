//! Tool-call protocol types and tool registry for genos.
//!
//! Model emits: {"tool": "fs.read", "args": {"path": "\\data\\x.txt"}, "call_id": "c1"}
//! System returns: {"call_id": "c1", "ok": true, "result": ..., "elapsed_ms": 12}

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use genos_kernel::json::JsonValue;

/// A parsed tool call from model output.
pub struct ToolCall {
    pub tool: String,
    pub args: JsonValue,
    pub call_id: String,
    pub session_id: String,
}

impl ToolCall {
    /// Parse a ToolCall from a JsonValue (the parsed JSON object).
    pub fn from_json(val: &JsonValue, session_id: &str) -> Option<Self> {
        let obj = val.as_object()?;
        let tool = obj.get("tool")?.as_str()?.into();
        let args = obj.get("args").cloned().unwrap_or(JsonValue::Object(BTreeMap::new()));
        let call_id = obj.get("call_id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| String::from("auto"));
        Some(ToolCall {
            tool,
            args,
            call_id,
            session_id: String::from(session_id),
        })
    }
}

/// Result of executing a tool.
pub struct ToolResult {
    pub call_id: String,
    pub ok: bool,
    pub result: JsonValue,
    pub error: Option<ToolError>,
    pub elapsed_ms: u64,
}

impl ToolResult {
    pub fn success(call_id: &str, result: JsonValue, elapsed_ms: u64) -> Self {
        ToolResult {
            call_id: String::from(call_id),
            ok: true,
            result,
            error: None,
            elapsed_ms,
        }
    }

    pub fn failure(call_id: &str, code: &str, message: &str, retryable: bool) -> Self {
        ToolResult {
            call_id: String::from(call_id),
            ok: false,
            result: JsonValue::Null,
            error: Some(ToolError {
                code: String::from(code),
                message: String::from(message),
                retryable,
            }),
            elapsed_ms: 0,
        }
    }

    /// Serialize to JSON for injection into prompt.
    pub fn to_json(&self) -> JsonValue {
        let mut map = BTreeMap::new();
        map.insert(String::from("call_id"), JsonValue::Str(self.call_id.clone()));
        map.insert(String::from("ok"), JsonValue::Bool(self.ok));
        if self.ok {
            map.insert(String::from("result"), self.result.clone());
        }
        if let Some(ref err) = self.error {
            let mut em = BTreeMap::new();
            em.insert(String::from("code"), JsonValue::Str(err.code.clone()));
            em.insert(String::from("message"), JsonValue::Str(err.message.clone()));
            em.insert(String::from("retryable"), JsonValue::Bool(err.retryable));
            map.insert(String::from("error"), JsonValue::Object(em));
        }
        map.insert(String::from("elapsed_ms"), JsonValue::Number(self.elapsed_ms as f64));
        JsonValue::Object(map)
    }
}

/// Tool error details.
pub struct ToolError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

/// Description of a tool for the manifest.
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub params: &'static [(&'static str, &'static str)], // (name, description)
}

/// Tool registry — maps tool names to their specs.
pub struct ToolRegistry {
    tools: Vec<ToolSpec>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry { tools: Vec::new() }
    }

    pub fn register(&mut self, spec: ToolSpec) {
        self.tools.push(spec);
    }

    pub fn find(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.iter().find(|t| t.name == name)
    }

    pub fn list(&self) -> &[ToolSpec] {
        &self.tools
    }

    /// Generate tool manifest for system prompt.
    pub fn manifest(&self) -> String {
        let mut out = String::from("Available tools:\n");
        for tool in &self.tools {
            out.push_str("- ");
            out.push_str(tool.name);
            out.push_str(": ");
            out.push_str(tool.description);
            if !tool.params.is_empty() {
                out.push_str(" (");
                for (i, (name, desc)) in tool.params.iter().enumerate() {
                    if i > 0 { out.push_str(", "); }
                    out.push_str(name);
                    out.push_str(": ");
                    out.push_str(desc);
                }
                out.push(')');
            }
            out.push('\n');
        }
        out.push_str("\nTo call a tool, emit JSON: {\"tool\": \"name\", \"args\": {...}, \"call_id\": \"c1\"}\n");
        out
    }

    /// Build the default registry with all Phase B tools.
    pub fn default_registry() -> Self {
        let mut reg = ToolRegistry::new();
        reg.register(ToolSpec {
            name: "fs.read",
            description: "Read file contents",
            params: &[("path", "file path")],
        });
        reg.register(ToolSpec {
            name: "fs.write",
            description: "Write content to file",
            params: &[("path", "file path"), ("content", "data to write")],
        });
        reg.register(ToolSpec {
            name: "fs.list",
            description: "List directory contents",
            params: &[("path", "directory path")],
        });
        reg.register(ToolSpec {
            name: "fs.delete",
            description: "Delete a file",
            params: &[("path", "file path")],
        });
        reg.register(ToolSpec {
            name: "net.fetch",
            description: "Fetch URL content (HTTP only)",
            params: &[("url", "URL to fetch")],
        });
        reg.register(ToolSpec {
            name: "memory.facts_get",
            description: "Get all L1 facts from palace",
            params: &[],
        });
        reg.register(ToolSpec {
            name: "memory.facts_set",
            description: "Set a fact in palace (with contradiction detection)",
            params: &[("key", "fact key"), ("value", "fact value")],
        });
        reg.register(ToolSpec {
            name: "memory.log_turn",
            description: "Log a turn to the session journal",
            params: &[("data", "turn data as JSON")],
        });
        reg.register(ToolSpec {
            name: "memory.store",
            description: "Store content in palace hall and index it",
            params: &[("content", "text to store"), ("wing", "wing name (default: general)"), ("hall", "hall name (default: discoveries)"), ("tags", "optional tags")],
        });
        reg.register(ToolSpec {
            name: "memory.search",
            description: "Search palace memory by query",
            params: &[("query", "search query"), ("wing", "filter by wing"), ("hall", "filter by hall"), ("top_k", "max results (default: 5)")],
        });
        reg.register(ToolSpec {
            name: "memory.consolidate",
            description: "Force consolidation: extract facts from recent journal entries",
            params: &[("n", "number of recent entries to scan (default: 20)")],
        });
        reg.register(ToolSpec {
            name: "memory.forget",
            description: "Archive a memory entry, removing it from active palace",
            params: &[("entry_id", "entry ID (format: session:turn)")],
        });
        reg.register(ToolSpec {
            name: "memory.graph_add",
            description: "Add a triple to the entity graph",
            params: &[("entity_a", "subject entity"), ("relation", "relationship"), ("entity_b", "object entity")],
        });
        reg.register(ToolSpec {
            name: "memory.graph_query",
            description: "Query entity relationships (optionally at a point in time)",
            params: &[("entity", "entity to query"), ("as_of", "optional ISO 8601 timestamp")],
        });
        reg.register(ToolSpec {
            name: "memory.graph_invalidate",
            description: "Invalidate a triple (set valid_until to now)",
            params: &[("entity_a", "subject"), ("relation", "relationship"), ("entity_b", "object")],
        });
        reg.register(ToolSpec {
            name: "memory.graph_timeline",
            description: "Get chronological history of an entity",
            params: &[("entity", "entity to get timeline for")],
        });
        reg.register(ToolSpec {
            name: "palace.create_wing",
            description: "Create a named wing (project or person) in the palace",
            params: &[("name", "wing name"), ("wing_type", "project or person")],
        });
        reg.register(ToolSpec {
            name: "palace.list_wings",
            description: "List all wings in the palace",
            params: &[],
        });
        reg.register(ToolSpec {
            name: "palace.find_tunnels",
            description: "Find cross-wing tunnels (shared halls between wings)",
            params: &[],
        });
        reg.register(ToolSpec {
            name: "sys.clock",
            description: "Get current wall-clock time",
            params: &[],
        });
        reg.register(ToolSpec {
            name: "sys.introspect",
            description: "Get system status (RAM, context, uptime)",
            params: &[],
        });
        reg
    }
}
