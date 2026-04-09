//! Temporal entity graph — triple store with validity windows.
//!
//! Triple format: (entity_a, relation, entity_b, valid_from, valid_until)
//! Storage: in-memory BTreeMap indexed by (entity_a, relation).
//! Invalidation sets valid_until, not deletion (append-only semantics).
//! Persisted as JSONL at `\palace\graph.jsonl`.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use genos_kernel::json::JsonValue;
use crate::protocol::ToolResult;

/// A single triple with temporal validity.
#[derive(Clone)]
pub struct Triple {
    pub entity_a: String,
    pub relation: String,
    pub entity_b: String,
    pub valid_from: String,
    pub valid_until: Option<String>,
}

impl Triple {
    pub fn is_valid_now(&self) -> bool {
        self.valid_until.is_none()
    }

    /// Serialize to JSONL line.
    pub fn to_json_line(&self) -> String {
        let until = match &self.valid_until {
            Some(u) => format!("\"{}\"", u),
            None => String::from("null"),
        };
        format!(
            "{{\"a\":\"{}\",\"r\":\"{}\",\"b\":\"{}\",\"from\":\"{}\",\"until\":{}}}",
            self.entity_a, self.relation, self.entity_b, self.valid_from, until
        )
    }

    /// Parse from JsonValue.
    pub fn from_json(val: &JsonValue) -> Option<Self> {
        let a = val.get("a").and_then(|v| v.as_str())?;
        let r = val.get("r").and_then(|v| v.as_str())?;
        let b = val.get("b").and_then(|v| v.as_str())?;
        let from = val.get("from").and_then(|v| v.as_str())?;
        let until = val.get("until").and_then(|v| v.as_str()).map(String::from);
        Some(Triple {
            entity_a: String::from(a),
            relation: String::from(r),
            entity_b: String::from(b),
            valid_from: String::from(from),
            valid_until: until,
        })
    }
}

/// Temporal entity graph with BTreeMap index over (entity_a, relation).
pub struct EntityGraph {
    /// All triples, indexed by (entity_a, relation) for fast lookup.
    index: BTreeMap<(String, String), Vec<Triple>>,
    /// Total triple count (including invalidated).
    count: usize,
}

const GRAPH_PATH: &str = "\\palace\\graph.jsonl";

impl EntityGraph {
    pub fn new() -> Self {
        EntityGraph {
            index: BTreeMap::new(),
            count: 0,
        }
    }

    /// Load graph from disk (JSONL).
    pub fn load(&mut self) {
        let data = match genos_hal::disk::read_file(GRAPH_PATH) {
            Ok(d) => d,
            Err(_) => return,
        };
        let text = core::str::from_utf8(&data).unwrap_or("");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(val) = genos_kernel::json::parse(line) {
                if let Some(triple) = Triple::from_json(&val) {
                    let key = (triple.entity_a.clone(), triple.relation.clone());
                    self.index.entry(key).or_insert_with(Vec::new).push(triple);
                    self.count += 1;
                }
            }
        }
    }

    /// Save entire graph to disk (JSONL).
    fn save(&self) {
        let mut content = String::new();
        for triples in self.index.values() {
            for t in triples {
                content.push_str(&t.to_json_line());
                content.push('\n');
            }
        }
        let _ = genos_hal::disk::write_file(GRAPH_PATH, content.as_bytes());
    }

    /// Add a triple with current timestamp.
    pub fn add(&mut self, entity_a: &str, relation: &str, entity_b: &str, timestamp: &str) {
        let triple = Triple {
            entity_a: String::from(entity_a),
            relation: String::from(relation),
            entity_b: String::from(entity_b),
            valid_from: String::from(timestamp),
            valid_until: None,
        };
        let key = (String::from(entity_a), String::from(relation));
        self.index.entry(key).or_insert_with(Vec::new).push(triple);
        self.count += 1;
        self.save();
    }

    /// Query all currently-valid triples for an entity.
    pub fn query(&self, entity: &str) -> Vec<&Triple> {
        let mut results = Vec::new();
        // Check as entity_a
        for ((a, _), triples) in &self.index {
            if a == entity {
                for t in triples {
                    if t.is_valid_now() {
                        results.push(t);
                    }
                }
            }
        }
        // Also check as entity_b (reverse lookup)
        for triples in self.index.values() {
            for t in triples {
                if t.entity_b == entity && t.is_valid_now() {
                    results.push(t);
                }
            }
        }
        results
    }

    /// Query triples valid at a specific timestamp (as_of).
    /// Simple string comparison — timestamps must be ISO 8601.
    pub fn query_as_of(&self, entity: &str, as_of: &str) -> Vec<&Triple> {
        let mut results = Vec::new();
        for ((a, _), triples) in &self.index {
            if a == entity {
                for t in triples {
                    let after_from = t.valid_from.as_str() <= as_of;
                    let before_until = match &t.valid_until {
                        Some(u) => as_of < u.as_str(),
                        None => true,
                    };
                    if after_from && before_until {
                        results.push(t);
                    }
                }
            }
        }
        results
    }

    /// Invalidate a triple by setting valid_until.
    pub fn invalidate(
        &mut self,
        entity_a: &str,
        relation: &str,
        entity_b: &str,
        timestamp: &str,
    ) -> bool {
        let key = (String::from(entity_a), String::from(relation));
        if let Some(triples) = self.index.get_mut(&key) {
            for t in triples.iter_mut() {
                if t.entity_b == entity_b && t.is_valid_now() {
                    t.valid_until = Some(String::from(timestamp));
                    self.save();
                    return true;
                }
            }
        }
        false
    }

    /// Get full timeline for an entity (all triples, including invalidated).
    pub fn timeline(&self, entity: &str) -> Vec<&Triple> {
        let mut results = Vec::new();
        for ((a, _), triples) in &self.index {
            if a == entity {
                for t in triples {
                    results.push(t);
                }
            }
        }
        // Sort by valid_from
        results.sort_by(|a, b| a.valid_from.cmp(&b.valid_from));
        results
    }

    /// Total triples (including invalidated).
    pub fn len(&self) -> usize {
        self.count
    }
}

// ── Tool handlers ──────────────────────────────────────────────────

/// memory.graph_add(entity_a, relation, entity_b)
pub fn tool_graph_add(
    args: &JsonValue,
    call_id: &str,
    timestamp: &str,
    graph: &mut EntityGraph,
) -> ToolResult {
    let a = match args.get("entity_a").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'entity_a'", false),
    };
    let r = match args.get("relation").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'relation'", false),
    };
    let b = match args.get("entity_b").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'entity_b'", false),
    };

    graph.add(a, r, b, timestamp);

    ToolResult::success(
        call_id,
        genos_kernel::json::json_object(&[
            ("added", JsonValue::Bool(true)),
            ("triple", JsonValue::Str(format!("({}, {}, {})", a, r, b))),
        ]),
        0,
    )
}

/// memory.graph_query(entity, as_of?)
pub fn tool_graph_query(
    args: &JsonValue,
    call_id: &str,
    graph: &EntityGraph,
) -> ToolResult {
    let entity = match args.get("entity").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'entity'", false),
    };

    let results = if let Some(as_of) = args.get("as_of").and_then(|v| v.as_str()) {
        graph.query_as_of(entity, as_of)
    } else {
        graph.query(entity)
    };

    let arr: Vec<JsonValue> = results
        .iter()
        .map(|t| {
            genos_kernel::json::json_object(&[
                ("entity_a", JsonValue::Str(t.entity_a.clone())),
                ("relation", JsonValue::Str(t.relation.clone())),
                ("entity_b", JsonValue::Str(t.entity_b.clone())),
                ("valid_from", JsonValue::Str(t.valid_from.clone())),
            ])
        })
        .collect();

    ToolResult::success(call_id, JsonValue::Array(arr), 0)
}

/// memory.graph_invalidate(entity_a, relation, entity_b)
pub fn tool_graph_invalidate(
    args: &JsonValue,
    call_id: &str,
    timestamp: &str,
    graph: &mut EntityGraph,
) -> ToolResult {
    let a = match args.get("entity_a").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'entity_a'", false),
    };
    let r = match args.get("relation").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'relation'", false),
    };
    let b = match args.get("entity_b").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'entity_b'", false),
    };

    let found = graph.invalidate(a, r, b, timestamp);

    if found {
        ToolResult::success(
            call_id,
            genos_kernel::json::json_object(&[("invalidated", JsonValue::Bool(true))]),
            0,
        )
    } else {
        ToolResult::failure(call_id, "not_found", "no matching active triple found", false)
    }
}

/// memory.graph_timeline(entity) -> chronological story
pub fn tool_graph_timeline(
    args: &JsonValue,
    call_id: &str,
    graph: &EntityGraph,
) -> ToolResult {
    let entity = match args.get("entity").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'entity'", false),
    };

    let timeline = graph.timeline(entity);

    let arr: Vec<JsonValue> = timeline
        .iter()
        .map(|t| {
            let mut pairs: Vec<(&str, JsonValue)> = alloc::vec![
                ("entity_a", JsonValue::Str(t.entity_a.clone())),
                ("relation", JsonValue::Str(t.relation.clone())),
                ("entity_b", JsonValue::Str(t.entity_b.clone())),
                ("valid_from", JsonValue::Str(t.valid_from.clone())),
            ];
            if let Some(ref u) = t.valid_until {
                pairs.push(("valid_until", JsonValue::Str(u.clone())));
            }
            genos_kernel::json::json_object(&pairs)
        })
        .collect();

    ToolResult::success(call_id, JsonValue::Array(arr), 0)
}
