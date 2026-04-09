//! Filesystem tools: read, write, list, delete.
//! All operations go through genos-hal disk HAL (UEFI FAT32).

use alloc::string::String;
use alloc::vec::Vec;
use genos_hal::disk;
use genos_kernel::json::JsonValue;
use crate::protocol::ToolResult;

/// Raw fs helpers (used internally by other tools, not just the tool protocol).
pub struct FsTool;

impl FsTool {
    pub fn read(path: &str) -> Result<String, &'static str> {
        let data = disk::read_file(path).map_err(|_| "failed to read file")?;
        String::from_utf8(data).map_err(|_| "file is not valid UTF-8")
    }

    pub fn read_bytes(path: &str) -> Result<Vec<u8>, &'static str> {
        disk::read_file(path).map_err(|_| "failed to read file")
    }

    pub fn write(path: &str, content: &str) -> Result<(), &'static str> {
        disk::write_file(path, content.as_bytes()).map_err(|_| "failed to write file")
    }

    pub fn write_bytes(path: &str, data: &[u8]) -> Result<(), &'static str> {
        disk::write_file(path, data).map_err(|_| "failed to write file")
    }
}

// --- Tool protocol handlers ---

/// fs.read(path) -> file contents as string
pub fn tool_read(args: &JsonValue, call_id: &str) -> ToolResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'path'", false),
    };
    match disk::read_file(path) {
        Ok(data) => {
            let text = String::from_utf8_lossy(&data).into_owned();
            ToolResult::success(call_id, JsonValue::Str(text), 0)
        }
        Err(e) => ToolResult::failure(call_id, "read_error", e.as_str(), true),
    }
}

/// fs.write(path, content) -> success bool
pub fn tool_write(args: &JsonValue, call_id: &str) -> ToolResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'path'", false),
    };
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'content'", false),
    };
    match disk::write_file(path, content.as_bytes()) {
        Ok(()) => ToolResult::success(call_id, JsonValue::Bool(true), 0),
        Err(e) => ToolResult::failure(call_id, "write_error", e.as_str(), true),
    }
}

/// fs.list(path) -> not yet implemented (needs UEFI dir enum)
pub fn tool_list(_args: &JsonValue, call_id: &str) -> ToolResult {
    ToolResult::failure(call_id, "not_implemented", "fs.list requires directory enumeration", false)
}

/// fs.delete(path) -> blocked until policy engine
pub fn tool_delete(_args: &JsonValue, call_id: &str) -> ToolResult {
    ToolResult::failure(call_id, "policy_required", "fs.delete requires policy approval", false)
}
