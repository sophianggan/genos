//! Web intelligence tools: get_page, read_llms, extract_links, cache.
//!
//! Protocol: always check /llms.txt first → /llm.txt → fallback raw HTML.
//! HTML tag stripping (~200 LOC, no external crate).
//! Semantic chunk extraction into typed segments.

use alloc::string::{String, ToString};
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

// ─── web.feed ────────────────────────────────────────────────────

/// web.feed(url, max_items?) -> {feed_title, feed_url, items, item_count}
///
/// Fetches an RSS or Atom feed and returns a structured list of entries.
/// No XML parser crate needed — uses lightweight tag scanning.
pub fn tool_web_feed(args: &JsonValue, call_id: &str) -> ToolResult {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(u) if !u.is_empty() => u,
        _ => return ToolResult::failure(call_id, "invalid_args", "url is required", false),
    };
    let max_items = args.get("max_items").and_then(|v| v.as_f64()).unwrap_or(20.0) as usize;
    let max_items = max_items.min(50).max(1);

    let xml = match genos_hal::net::fetch_text(url) {
        Ok(s) => s,
        Err(e) => return ToolResult::failure(call_id, "fetch_failed", e.as_str(), true),
    };

    let (feed_title, items) = if xml.contains("<entry") {
        parse_atom(&xml, max_items)
    } else {
        parse_rss(&xml, max_items)
    };

    let item_count = items.len();
    let items_json: Vec<JsonValue> = items
        .into_iter()
        .map(|(title, item_url, summary, published_at)| {
            json_object(&[
                ("title",        JsonValue::Str(title)),
                ("url",          JsonValue::Str(item_url)),
                ("summary",      JsonValue::Str(summary)),
                ("published_at", JsonValue::Str(published_at)),
            ])
        })
        .collect();

    let result = json_object(&[
        ("feed_title",  JsonValue::Str(feed_title)),
        ("feed_url",    JsonValue::Str(String::from(url))),
        ("items",       JsonValue::Array(items_json)),
        ("item_count",  JsonValue::Number(item_count as f64)),
    ]);
    ToolResult::success(call_id, result, 0)
}

/// Parse RSS 2.0 XML — scans for `<item>` blocks.
fn parse_rss(xml: &str, max: usize) -> (String, Vec<(String, String, String, String)>) {
    let feed_title = feed_extract_tag(xml, "title").unwrap_or_default().to_string();
    let mut items = Vec::new();
    let mut pos = 0usize;
    while items.len() < max && pos < xml.len() {
        let item_open = match xml[pos..].find("<item") {
            Some(p) => pos + p,
            None => break,
        };
        let item_close = match xml[item_open..].find("</item>") {
            Some(p) => item_open + p + 7,
            None => break,
        };
        let block = &xml[item_open..item_close];
        let title       = feed_extract_tag(block, "title").unwrap_or("").trim().to_string();
        let link        = feed_extract_tag(block, "link").unwrap_or("").trim().to_string();
        let description = feed_extract_tag(block, "description").unwrap_or("").trim().to_string();
        let pub_date    = feed_extract_tag(block, "pubDate").unwrap_or("").trim().to_string();
        // Strip CDATA wrappers if present
        let title       = strip_cdata(&title);
        let description = strip_cdata(&description);
        items.push((title, link, description, pub_date));
        pos = item_close;
    }
    (feed_title, items)
}

/// Parse Atom XML — scans for `<entry>` blocks.
fn parse_atom(xml: &str, max: usize) -> (String, Vec<(String, String, String, String)>) {
    let feed_title = feed_extract_tag(xml, "title").unwrap_or_default().to_string();
    let mut items = Vec::new();
    let mut pos = 0usize;
    while items.len() < max && pos < xml.len() {
        let entry_open = match xml[pos..].find("<entry") {
            Some(p) => pos + p,
            None => break,
        };
        let entry_close = match xml[entry_open..].find("</entry>") {
            Some(p) => entry_open + p + 8,
            None => break,
        };
        let block = &xml[entry_open..entry_close];
        let title     = feed_extract_tag(block, "title").unwrap_or("").trim().to_string();
        let summary   = feed_extract_tag(block, "summary").unwrap_or("").trim().to_string();
        let published = feed_extract_tag(block, "published")
            .or_else(|| feed_extract_tag(block, "updated"))
            .unwrap_or("").trim().to_string();
        // Atom link is an empty element: <link href="..."/>
        let link = feed_extract_attr(block, "link", "href").unwrap_or_default();
        let title   = strip_cdata(&title);
        let summary = strip_cdata(&summary);
        items.push((title, link, summary, published));
        pos = entry_close;
    }
    (feed_title, items)
}

/// Extract inner text of the first `<tag>...</tag>` in `xml`.
fn feed_extract_tag<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open  = format!("<{}>",  tag);
    let close = format!("</{}>", tag);
    let start = xml.find(open.as_str())? + open.len();
    let end   = xml[start..].find(close.as_str())? + start;
    Some(&xml[start..end])
}

/// Extract an attribute value from the first occurrence of `<element attr="...">`.
fn feed_extract_attr(xml: &str, element: &str, attr: &str) -> Option<String> {
    let elem_start = xml.find(&format!("<{}", element))?;
    let tag_end = xml[elem_start..].find('>')?  + elem_start;
    let tag = &xml[elem_start..tag_end + 1];
    let key = format!("{}=\"", attr);
    let val_start = tag.find(key.as_str())? + key.len();
    let val_end = tag[val_start..].find('"')? + val_start;
    Some(tag[val_start..val_end].to_string())
}

/// Remove `<![CDATA[...]]>` wrappers if present.
fn strip_cdata(s: &str) -> String {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix("<![CDATA[").and_then(|t| t.strip_suffix("]]>")) {
        inner.trim().to_string()
    } else {
        s.to_string()
    }
}
