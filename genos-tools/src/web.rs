//! Web intelligence tools: get_page, read_llms, extract_links, cache.
//!
//! Protocol: always check /llms.txt first → /llm.txt → fallback raw HTML.
//! HTML tag stripping (~200 LOC, no external crate).
//! Semantic chunk extraction into typed segments.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use alloc::format;
use genos_kernel::json::{json_object, JsonValue};
use crate::protocol::ToolResult;

/// Type of extracted content chunk.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Title,
    Heading { level: u8 },
    Paragraph,
    Code,
    ListItem,
    Table,
}

impl ChunkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkType::Title => "title",
            ChunkType::Heading { .. } => "heading",
            ChunkType::Paragraph => "paragraph",
            ChunkType::Code => "code",
            ChunkType::ListItem => "list_item",
            ChunkType::Table => "table",
        }
    }
}

/// A semantic chunk extracted from a web page.
pub struct Chunk {
    pub typ: ChunkType,
    pub text: String,
    pub tokens_estimate: usize,
}

impl Chunk {
    pub fn to_json(&self) -> JsonValue {
        json_object(&[
            ("type", JsonValue::Str(String::from(self.typ.as_str()))),
            ("text", JsonValue::Str(self.text.clone())),
            ("tokens", JsonValue::Number(self.tokens_estimate as f64)),
        ])
    }
}

/// Result of fetching and parsing a web page.
pub struct PageContent {
    pub url: String,
    pub domain: String,
    pub chunks: Vec<Chunk>,
    pub total_tokens: usize,
    pub has_more: bool,
    pub source_llms_txt: bool,
}

impl PageContent {
    pub fn to_json(&self) -> JsonValue {
        let chunks_json: Vec<JsonValue> = self.chunks.iter().map(|c| c.to_json()).collect();
        json_object(&[
            ("url", JsonValue::Str(self.url.clone())),
            ("domain", JsonValue::Str(self.domain.clone())),
            ("chunks", JsonValue::Array(chunks_json)),
            ("total_tokens", JsonValue::Number(self.total_tokens as f64)),
            ("has_more", JsonValue::Bool(self.has_more)),
            ("source_llms_txt", JsonValue::Bool(self.source_llms_txt)),
        ])
    }
}

/// Extract domain from a URL.
fn extract_domain(url: &str) -> String {
    let rest = url.strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    match rest.find('/') {
        Some(i) => String::from(&rest[..i]),
        None => String::from(rest),
    }
}

/// Strip HTML tags from raw HTML and extract semantic chunks.
pub fn strip_html(html: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current_text = String::new();
    let mut in_tag = false;
    let mut tag_name = String::new();
    let mut current_type = ChunkType::Paragraph;
    let mut in_script = false;
    let mut in_style = false;
    let mut collecting_tag = false;

    let bytes = html.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let ch = bytes[i] as char;

        if in_tag {
            if ch == '>' {
                in_tag = false;
                collecting_tag = false;
                let tag_lower = tag_name_lower(&tag_name);

                // Handle block-level tags
                match tag_lower.as_str() {
                    "script" => in_script = true,
                    "/script" => in_script = false,
                    "style" => in_style = true,
                    "/style" => in_style = false,
                    "title" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Title;
                    }
                    "/title" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Paragraph;
                    }
                    "h1" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Heading { level: 1 };
                    }
                    "h2" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Heading { level: 2 };
                    }
                    "h3" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Heading { level: 3 };
                    }
                    "h4" | "h5" | "h6" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Heading { level: 4 };
                    }
                    "/h1" | "/h2" | "/h3" | "/h4" | "/h5" | "/h6" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Paragraph;
                    }
                    "pre" | "code" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Code;
                    }
                    "/pre" | "/code" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Paragraph;
                    }
                    "li" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::ListItem;
                    }
                    "/li" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Paragraph;
                    }
                    "table" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Table;
                    }
                    "/table" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Paragraph;
                    }
                    "br" | "br/" => current_text.push('\n'),
                    "p" | "/p" | "div" | "/div" | "section" | "/section" | "article"
                    | "/article" | "header" | "/header" | "footer" | "/footer" | "nav"
                    | "/nav" | "main" | "/main" => {
                        flush_chunk(&mut chunks, &mut current_text, &current_type);
                        current_type = ChunkType::Paragraph;
                    }
                    _ => {}
                }
                tag_name.clear();
            } else if collecting_tag {
                if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
                    collecting_tag = false;
                } else {
                    tag_name.push(ch);
                }
            }
        } else if ch == '<' {
            in_tag = true;
            collecting_tag = true;
            tag_name.clear();
        } else if ch == '&' {
            // Decode HTML entities
            let entity = decode_entity(bytes, &mut i);
            if !in_script && !in_style {
                current_text.push_str(&entity);
            }
        } else if !in_script && !in_style {
            // Normalize whitespace
            if ch == '\n' || ch == '\r' || ch == '\t' {
                if !current_text.ends_with(' ') {
                    current_text.push(' ');
                }
            } else {
                current_text.push(ch);
            }
        }

        i += 1;
    }

    // Flush remaining text
    flush_chunk(&mut chunks, &mut current_text, &current_type);

    chunks
}

/// Flush accumulated text into a chunk if non-empty.
fn flush_chunk(chunks: &mut Vec<Chunk>, text: &mut String, typ: &ChunkType) {
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        let tokens_estimate = (trimmed.len() + 3) / 4;
        chunks.push(Chunk {
            typ: typ.clone(),
            text: String::from(trimmed),
            tokens_estimate,
        });
    }
    text.clear();
}

/// Lowercase a tag name for matching.
fn tag_name_lower(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_uppercase() {
            s.push((ch as u8 + 32) as char);
        } else {
            s.push(ch);
        }
    }
    s
}

/// Decode basic HTML entities.
fn decode_entity(bytes: &[u8], pos: &mut usize) -> String {
    let start = *pos + 1; // skip &
    let mut end = start;
    while end < bytes.len() && end - start < 10 {
        if bytes[end] == b';' {
            let entity = core::str::from_utf8(&bytes[start..end]).unwrap_or("");
            *pos = end; // will be incremented by caller
            return match entity {
                "amp" => String::from("&"),
                "lt" => String::from("<"),
                "gt" => String::from(">"),
                "quot" => String::from("\""),
                "apos" => String::from("'"),
                "nbsp" => String::from(" "),
                _ if entity.starts_with('#') => {
                    // Numeric entity
                    let num_str = &entity[1..];
                    let code = if num_str.starts_with('x') || num_str.starts_with('X') {
                        u32::from_str_radix(&num_str[1..], 16).ok()
                    } else {
                        parse_u32(num_str)
                    };
                    match code.and_then(char::from_u32) {
                        Some(c) => {
                            let mut s = String::new();
                            s.push(c);
                            s
                        }
                        None => String::from("?"),
                    }
                }
                _ => format!("&{};", entity),
            };
        }
        end += 1;
    }
    String::from("&")
}

fn parse_u32(s: &str) -> Option<u32> {
    let mut n: u32 = 0;
    for b in s.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add((b - b'0') as u32)?;
    }
    Some(n)
}

/// Fetch a page with llms.txt-first protocol.
/// 1. Try <host>/llms.txt
/// 2. Try <host>/llm.txt
/// 3. Fallback: fetch URL, strip HTML
fn fetch_page(url: &str, max_tokens: usize) -> PageContent {
    let domain = extract_domain(url);
    let scheme = if url.starts_with("https://") { "https" } else { "http" };

    // Step 1: Try llms.txt
    let llms_url = format!("{}://{}/llms.txt", scheme, domain);
    if let Ok(data) = genos_hal::net::fetch(&llms_url) {
        let text = String::from_utf8_lossy(&data).into_owned();
        if !text.is_empty() {
            let tokens_estimate = (text.len() + 3) / 4;
            let (truncated, has_more) = truncate_to_tokens(&text, max_tokens);
            return PageContent {
                url: String::from(url),
                domain,
                chunks: vec![Chunk {
                    typ: ChunkType::Paragraph,
                    text: truncated,
                    tokens_estimate,
                }],
                total_tokens: tokens_estimate,
                has_more,
                source_llms_txt: true,
            };
        }
    }

    // Step 2: Try llm.txt
    let llm_url = format!("{}://{}/llm.txt", scheme, domain);
    if let Ok(data) = genos_hal::net::fetch(&llm_url) {
        let text = String::from_utf8_lossy(&data).into_owned();
        if !text.is_empty() {
            let tokens_estimate = (text.len() + 3) / 4;
            let (truncated, has_more) = truncate_to_tokens(&text, max_tokens);
            return PageContent {
                url: String::from(url),
                domain,
                chunks: vec![Chunk {
                    typ: ChunkType::Paragraph,
                    text: truncated,
                    tokens_estimate,
                }],
                total_tokens: tokens_estimate,
                has_more,
                source_llms_txt: true,
            };
        }
    }

    // Step 3: Fetch actual URL and strip HTML
    match genos_hal::net::fetch(url) {
        Ok(data) => {
            let html = String::from_utf8_lossy(&data).into_owned();
            let mut chunks = strip_html(&html);
            let total_tokens: usize = chunks.iter().map(|c| c.tokens_estimate).sum();

            // Truncate chunks to fit max_tokens
            let mut has_more = false;
            let mut acc = 0usize;
            let mut keep = chunks.len();
            for (i, chunk) in chunks.iter().enumerate() {
                acc += chunk.tokens_estimate;
                if acc > max_tokens {
                    keep = i;
                    has_more = true;
                    break;
                }
            }
            chunks.truncate(keep);

            PageContent {
                url: String::from(url),
                domain,
                chunks,
                total_tokens,
                has_more,
                source_llms_txt: false,
            }
        }
        Err(_) => PageContent {
            url: String::from(url),
            domain,
            chunks: Vec::new(),
            total_tokens: 0,
            has_more: false,
            source_llms_txt: false,
        },
    }
}

/// Truncate text to approximately max_tokens (4 chars/token).
fn truncate_to_tokens(text: &str, max_tokens: usize) -> (String, bool) {
    let max_chars = max_tokens * 4;
    if text.len() <= max_chars {
        (String::from(text), false)
    } else {
        // Find a good break point (word boundary)
        let mut end = max_chars;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        while end > 0 && !text.as_bytes()[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        if end == 0 {
            end = max_chars.min(text.len());
        }
        (String::from(&text[..end]), true)
    }
}

// --- Tool Protocol Handlers ---

/// web.get_page(url, max_tokens?) -> PageContent as JSON
pub fn tool_get_page(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'url'", false),
    };

    // HTTP and HTTPS supported
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return ToolResult::failure(
            call_id,
            "invalid_args",
            "only http:// and https:// URLs are supported",
            false,
        );
    }

    let max_tokens = args.get("max_tokens")
        .and_then(|v| v.as_f64())
        .map(|n| n as usize)
        .unwrap_or(1024);

    let page = fetch_page(url, max_tokens);
    ToolResult::success(call_id, page.to_json(), 0)
}

/// web.read_llms(url) -> raw llms.txt content if found
pub fn tool_read_llms(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'url'", false),
    };

    let domain = extract_domain(url);

    // Try llms.txt on both https and http
    let scheme = if url.starts_with("https://") { "https" } else { "http" };
    for suffix in &["llms.txt", "llm.txt"] {
        let check_url = format!("{}://{}/{}", scheme, domain, suffix);
        if let Ok(data) = genos_hal::net::fetch(&check_url) {
            let text = String::from_utf8_lossy(&data).into_owned();
            if !text.is_empty() {
                return ToolResult::success(call_id, JsonValue::Str(text), 0);
            }
        }
    }

    ToolResult::success(call_id, JsonValue::Null, 0)
}

/// web.extract_links(url) -> list of links found in page HTML
pub fn tool_extract_links(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'url'", false),
    };

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return ToolResult::failure(call_id, "invalid_args", "only http:// and https:// URLs are supported", false);
    }

    let domain = extract_domain(url);

    match genos_hal::net::fetch(url) {
        Ok(data) => {
            let html = String::from_utf8_lossy(&data).into_owned();
            let links = extract_links_from_html(&html, &domain);
            let links_json: Vec<JsonValue> = links.into_iter().map(|(href, text, link_type)| {
                json_object(&[
                    ("url", JsonValue::Str(href)),
                    ("text", JsonValue::Str(text)),
                    ("type", JsonValue::Str(link_type)),
                ])
            }).collect();
            ToolResult::success(call_id, JsonValue::Array(links_json), 0)
        }
        Err(e) => ToolResult::failure(call_id, "net_error", e.as_str(), true),
    }
}

/// Extract links from HTML by finding <a href="..."> tags.
fn extract_links_from_html(html: &str, domain: &str) -> Vec<(String, String, String)> {
    let mut links = Vec::new();
    let bytes = html.as_bytes();
    let mut i = 0;

    while i < bytes.len().saturating_sub(8) {
        // Look for <a with attributes containing href
        if bytes[i] == b'<' && (bytes[i + 1] == b'a' || bytes[i + 1] == b'A')
            && (bytes[i + 2] == b' ' || bytes[i + 2] == b'\t')
        {
            // Find href="..."
            let tag_end = find_byte(bytes, b'>', i);
            if let Some(end) = tag_end {
                let tag_content = core::str::from_utf8(&bytes[i..end + 1]).unwrap_or("");
                if let Some(href) = extract_attr(tag_content, "href") {
                    // Extract link text (between <a ...> and </a>)
                    let text_start = end + 1;
                    let text_end = find_closing_tag(bytes, text_start, "a");
                    let link_text = if let Some(te) = text_end {
                        let raw = core::str::from_utf8(&bytes[text_start..te]).unwrap_or("");
                        strip_tags_simple(raw)
                    } else {
                        String::new()
                    };

                    let link_type = if href.contains(domain) || href.starts_with('/') || href.starts_with('#') {
                        String::from("internal")
                    } else if href.ends_with(".pdf") || href.ends_with(".doc") || href.ends_with(".docx") {
                        String::from("document")
                    } else {
                        String::from("external")
                    };

                    links.push((String::from(href), link_text, link_type));
                }
                i = end + 1;
                continue;
            }
        }
        i += 1;
    }

    links
}

fn find_byte(bytes: &[u8], target: u8, start: usize) -> Option<usize> {
    for i in start..bytes.len() {
        if bytes[i] == target {
            return Some(i);
        }
    }
    None
}

fn find_closing_tag(bytes: &[u8], start: usize, tag: &str) -> Option<usize> {
    let close = format!("</{}", tag);
    let close_bytes = close.as_bytes();
    for i in start..bytes.len().saturating_sub(close_bytes.len()) {
        if bytes[i] == b'<' && bytes[i + 1] == b'/' {
            let tag_start = &bytes[i..];
            if tag_start.len() >= close_bytes.len() {
                let candidate = core::str::from_utf8(&tag_start[..close_bytes.len()]).unwrap_or("");
                let candidate_lower = {
                    let mut s = String::new();
                    for c in candidate.chars() {
                        s.push(if c.is_ascii_uppercase() { (c as u8 + 32) as char } else { c });
                    }
                    s
                };
                if candidate_lower == close {
                    return Some(i);
                }
            }
        }
    }
    None
}

/// Extract an attribute value from a tag string.
fn extract_attr(tag: &str, attr_name: &str) -> Option<String> {
    let search = format!("{}=\"", attr_name);
    let search_sq = format!("{}='", attr_name);

    let start = tag.find(&search).or_else(|| tag.find(&search_sq))?;
    let quote_char = if tag[start..].contains(&search) { '"' } else { '\'' };
    let val_start = start + attr_name.len() + 2; // skip attr="
    let remaining = &tag[val_start..];
    let end = remaining.find(quote_char)?;
    Some(String::from(&remaining[..end]))
}

/// Remove all HTML tags from a string (simple version for link text).
fn strip_tags_simple(s: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    result.trim().into()
}

/// Simple web page cache — stores fetched pages in \palace\cache\
pub struct WebCache;

impl WebCache {
    /// Read a cached page.
    pub fn read(url: &str) -> Option<String> {
        let path = Self::cache_path(url);
        match genos_hal::disk::read_file(&path) {
            Ok(data) => Some(String::from_utf8_lossy(&data).into_owned()),
            Err(_) => None,
        }
    }

    /// Write a page to cache.
    pub fn write(url: &str, content: &str) {
        let path = Self::cache_path(url);
        let _ = genos_hal::disk::write_file(&path, content.as_bytes());
    }

    /// Invalidate (delete) a cached page.
    pub fn invalidate(url: &str) {
        let path = Self::cache_path(url);
        // Write empty to invalidate (no delete API in Phase B)
        let _ = genos_hal::disk::write_file(&path, b"");
    }

    /// Generate cache file path from URL.
    fn cache_path(url: &str) -> String {
        // Hash URL to a simple filename
        let mut hash: u32 = 0;
        for b in url.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(b as u32);
        }
        format!("\\palace\\cache\\page_{:08x}.txt", hash)
    }
}

/// web.cache_read(url) -> cached page content
pub fn tool_cache_read(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'url'", false),
    };
    match WebCache::read(url) {
        Some(content) => ToolResult::success(call_id, JsonValue::Str(content), 0),
        None => ToolResult::success(call_id, JsonValue::Null, 0),
    }
}

/// web.cache_invalidate(url) -> true
pub fn tool_cache_invalidate(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return ToolResult::failure(call_id, "invalid_args", "missing 'url'", false),
    };
    WebCache::invalidate(url);
    ToolResult::success(call_id, JsonValue::Bool(true), 0)
}

// ─── web.pdf ─────────────────────────────────────────────────────

/// web.pdf(url) -> {url, text, pages_estimated, truncated}
///
/// Fetches a PDF from a URL, decompresses its content streams with flate2,
/// and extracts readable text from PDF text operators (Tj / TJ).
pub fn tool_web_pdf(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) if !u.is_empty() => u,
        _ => return ToolResult::failure(call_id, "invalid_args", "url is required", false),
    };

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return ToolResult::failure(call_id, "invalid_args", "only http:// and https:// URLs are supported", false);
    }

    let bytes = match genos_hal::net::fetch(url) {
        Ok(b) => b,
        Err(e) => return ToolResult::failure(call_id, "fetch_failed", e.as_str(), true),
    };

    // Quick sanity check: PDFs start with "%PDF"
    if !bytes.starts_with(b"%PDF") {
        return ToolResult::failure(call_id, "not_a_pdf", "URL did not return a PDF document", false);
    }

    let (text, pages_estimated) = extract_pdf_text(&bytes);

    const MAX_CHARS: usize = 30_000;
    let truncated = text.len() > MAX_CHARS;
    let text_out = if truncated {
        // Truncate at a word boundary
        let mut end = MAX_CHARS;
        while end > 0 && !text.is_char_boundary(end) { end -= 1; }
        while end > 0 && !text.as_bytes()[end - 1].is_ascii_whitespace() { end -= 1; }
        if end == 0 { end = MAX_CHARS.min(text.len()); }
        String::from(&text[..end])
    } else {
        text
    };

    let result = json_object(&[
        ("url",             JsonValue::Str(String::from(url))),
        ("text",            JsonValue::Str(text_out)),
        ("pages_estimated", JsonValue::Number(pages_estimated as f64)),
        ("truncated",       JsonValue::Bool(truncated)),
    ]);
    ToolResult::success(call_id, result, 0)
}

/// Extract text from PDF bytes.
/// Returns (text, estimated_page_count).
fn extract_pdf_text(data: &[u8]) -> (String, usize) {
    let mut text = String::new();
    let mut page_count = 0usize;

    // Count "/Page " occurrences as a rough page estimate
    let page_marker = b"/Type /Page\n";
    let mut i = 0;
    while i + page_marker.len() <= data.len() {
        if data[i..].starts_with(b"/Type /Page") {
            page_count += 1;
        }
        i += 1;
    }
    if page_count == 0 { page_count = 1; }

    // Locate every "stream … endstream" block
    let stream_marker  = b"stream";
    let endstream_marker = b"endstream";
    let mut pos = 0;

    while pos < data.len() {
        // Find next "stream" keyword
        let stream_pos = match find_subsequence(data, stream_marker, pos) {
            Some(p) => p,
            None => break,
        };

        // The stream data starts after "stream\r\n" or "stream\n"
        let after_keyword = stream_pos + stream_marker.len();
        let data_start = if data.get(after_keyword) == Some(&b'\r') && data.get(after_keyword + 1) == Some(&b'\n') {
            after_keyword + 2
        } else if data.get(after_keyword) == Some(&b'\n') {
            after_keyword + 1
        } else {
            after_keyword
        };

        // Find the matching "endstream"
        let end_pos = match find_subsequence(data, endstream_marker, data_start) {
            Some(p) => p,
            None => break,
        };

        // Look at the object header (bytes before "stream\n") for /FlateDecode
        let header_start = header_search_start(data, stream_pos);
        let header = &data[header_start..stream_pos];
        let is_flat = contains_subsequence(header, b"/FlateDecode")
            || contains_subsequence(header, b"/Fl ");

        let stream_bytes = &data[data_start..end_pos];

        if is_flat {
            if let Some(decompressed) = decompress_zlib(stream_bytes) {
                extract_text_from_content_stream(&decompressed, &mut text);
            }
        } else {
            // Try plain (uncompressed) content stream
            if let Ok(s) = core::str::from_utf8(stream_bytes) {
                extract_text_from_content_stream_str(s, &mut text);
            }
        }

        pos = end_pos + endstream_marker.len();
    }

    (text, page_count)
}

/// Find the start of the object header preceding a stream (up to 1 KB back).
fn header_search_start(data: &[u8], stream_pos: usize) -> usize {
    if stream_pos > 1024 { stream_pos - 1024 } else { 0 }
}

/// Find first occurrence of `needle` in `haystack` starting at `start`.
fn find_subsequence(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    if needle.len() > haystack.len() { return None; }
    let limit = haystack.len() - needle.len();
    for i in start..=limit {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

/// Check if `haystack` contains `needle`.
fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    find_subsequence(haystack, needle, 0).is_some()
}

/// Decompress zlib/deflate bytes using the flate2 crate (hosted builds only).
#[cfg(feature = "hosted")]
fn decompress_zlib(data: &[u8]) -> Option<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    match decoder.read_to_end(&mut out) {
        Ok(_) => Some(out),
        Err(_) => {
            // Some PDFs use raw deflate without the zlib wrapper
            use flate2::read::DeflateDecoder;
            let mut decoder2 = DeflateDecoder::new(data);
            let mut out2 = Vec::new();
            match decoder2.read_to_end(&mut out2) {
                Ok(_) if !out2.is_empty() => Some(out2),
                _ => None,
            }
        }
    }
}

#[cfg(not(feature = "hosted"))]
fn decompress_zlib(_data: &[u8]) -> Option<Vec<u8>> {
    None
}

/// Extract visible text from a decompressed PDF content stream (bytes).
fn extract_text_from_content_stream(data: &[u8], out: &mut String) {
    if let Ok(s) = core::str::from_utf8(data) {
        extract_text_from_content_stream_str(s, out);
    } else {
        // Try latin-1 fallback: filter to printable ASCII
        let ascii: String = data.iter()
            .map(|&b| if b >= 0x20 && b < 0x7f { b as char } else { ' ' })
            .collect();
        extract_text_from_content_stream_str(&ascii, out);
    }
}

/// Extract visible text from a PDF content stream string.
///
/// Handles the two common PDF text operators:
///   (text) Tj      — show single string
///   [(text) ...] TJ — show array of strings
fn extract_text_from_content_stream_str(s: &str, out: &mut String) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    // Collect all parenthesised strings just before a Tj/TJ operator.
    // We track raw strings and flush them when we see Tj or TJ.
    let mut pending: Vec<String> = Vec::new();

    while i < len {
        match bytes[i] {
            b'(' => {
                // Read PDF literal string, handling escape sequences and nesting
                let (pstr, consumed) = read_pdf_string(bytes, i);
                pending.push(pstr);
                i += consumed;
            }
            b'T' if i + 1 < len && (bytes[i + 1] == b'j' || bytes[i + 1] == b'J') => {
                // Flush pending strings as a text run
                for s in pending.drain(..) {
                    if !s.trim().is_empty() {
                        out.push_str(s.trim());
                        out.push(' ');
                    }
                }
                i += 2;
            }
            b'B' if i + 1 < len && bytes[i + 1] == b'T' => {
                // Begin Text block — just advance
                i += 2;
            }
            b'E' if i + 1 < len && bytes[i + 1] == b'T' => {
                // End Text block — insert newline to separate blocks
                if !out.ends_with('\n') && !out.is_empty() {
                    out.push('\n');
                }
                pending.clear();
                i += 2;
            }
            _ => {
                // Any non-string, non-operator character resets pending
                // unless it's whitespace/brackets (part of TJ array syntax)
                if bytes[i] != b'[' && bytes[i] != b']'
                    && bytes[i] != b' ' && bytes[i] != b'\n'
                    && bytes[i] != b'\r' && bytes[i] != b'\t'
                {
                    // Non-whitespace, non-array character not consumed above:
                    // if previous token was not a string, discard pending
                    if !pending.is_empty() {
                        let last = pending.last().map(|s| s.as_str()).unwrap_or("");
                        // Keep pending if this looks like it could precede Tj
                        let _ = last;
                    }
                }
                i += 1;
            }
        }
    }
}

/// Read a PDF literal string starting at `start` (which must be `(`).
/// Returns (decoded_string, bytes_consumed_including_parens).
fn read_pdf_string(bytes: &[u8], start: usize) -> (String, usize) {
    let mut result = String::new();
    let mut i = start + 1; // skip opening '('
    let mut depth = 1usize;

    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => {
                match bytes[i + 1] {
                    b'n'  => { result.push('\n'); i += 2; }
                    b'r'  => { result.push('\r'); i += 2; }
                    b't'  => { result.push('\t'); i += 2; }
                    b'('  => { result.push('(');  i += 2; }
                    b')'  => { result.push(')');  i += 2; }
                    b'\\' => { result.push('\\'); i += 2; }
                    b'0'..=b'7' => {
                        // Octal escape up to 3 digits
                        let mut oct = 0u32;
                        let mut j = 0;
                        while j < 3 && i + 1 + j < bytes.len()
                            && bytes[i + 1 + j] >= b'0' && bytes[i + 1 + j] <= b'7'
                        {
                            oct = oct * 8 + (bytes[i + 1 + j] - b'0') as u32;
                            j += 1;
                        }
                        if let Some(c) = char::from_u32(oct) { result.push(c); }
                        i += 1 + j;
                    }
                    _ => { i += 2; }
                }
            }
            b'(' => { depth += 1; result.push('('); i += 1; }
            b')' => {
                depth -= 1;
                if depth > 0 { result.push(')'); }
                i += 1;
            }
            b => {
                let ch = b as char;
                // Only include printable ASCII and common whitespace
                if ch.is_ascii_graphic() || ch == ' ' {
                    result.push(ch);
                }
                i += 1;
            }
        }
    }

    let consumed = i - start;
    (result, consumed)
}
