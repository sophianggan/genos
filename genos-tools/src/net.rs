//! net.fetch tool handler.

use alloc::string::String;
use genos_hal::net;
use genos_kernel::json::JsonValue;
use crate::protocol::ToolResult;

/// net.fetch(url) -> response body as string
pub fn tool_fetch(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'url'", false),
    };

    // Reject non-HTTP/HTTPS URLs
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return ToolResult::failure(
            call_id,
            "invalid_args",
            "only http:// and https:// URLs are supported",
            false,
        );
    }

    if url.is_empty() || url == "http://" || url == "https://" {
        return ToolResult::failure(call_id, "invalid_args", "empty URL", false);
    }

    match net::fetch(url) {
        Ok(data) => {
            let text = String::from_utf8_lossy(&data).into_owned();
            ToolResult::success(call_id, JsonValue::Str(text), 0)
        }
        Err(e) => ToolResult::failure(call_id, "net_error", e.as_str(), true),
    }
}
