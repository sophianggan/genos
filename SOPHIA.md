# SOPHIA.md — Contributor Task Guide

Hi Sophie! This file is your complete guide for contributing to genos.
Read this fully before writing any code. It explains what the project is,
what already exists, what needs building, and exactly how to do it.

---

## What is genos?

genos is a bare-metal operating system written in Rust where the OS itself
**is** a large language model. It boots directly from a USB/SSD via UEFI
(no Linux, no Windows underneath), loads a quantized LLM, and the LLM makes
every decision through structured tool calls.

The LLM receives tool results as text, reasons about them, and emits new
tool calls — think of it as a reasoning loop with persistent memory and
internet access, running directly on hardware.

---

## Repo layout

```
genos/                          ← git root, everything lives here
  genos-boot/                   ← UEFI entry point — DO NOT TOUCH
  genos-hal/                    ← Hardware abstraction (screen, disk, net)
  genos-kernel/                 ← LLM runtime (inference, tokenizer) — DO NOT TOUCH
  genos-tools/                  ← Tool implementations ← YOUR WORK GOES HERE
    src/
      web.rs                    ← web.get_page, web.extract_links, web.read_llms
      net.rs                    ← net.fetch (raw HTTP/HTTPS fetch)
      memory.rs                 ← memory.store, memory.search, etc.
      fs.rs                     ← fs.read, fs.write, fs.list, fs.delete
      graph.rs                  ← memory.graph_* (entity graph)
      palace.rs                 ← palace.* (session-scoped storage)
      sys.rs                    ← sys.clock, sys.introspect
      journal.rs                ← turn journaling
      search.rs                 ← TF-IDF search index
      protocol.rs               ← ToolCall / ToolResult types
  genos-agent/                  ← REPL loop and tool dispatch
    src/
      tools.rs                  ← dispatch table — ADD NEW TOOLS HERE
      repl.rs                   ← main inference loop — DO NOT TOUCH
  tests/                        ← Integration test suite ← ADD TESTS HERE
    src/main.rs
  Cargo.toml                    ← workspace manifest
  Makefile                      ← build/qemu/test commands
  MASTER.md                     ← full architecture spec
  SOPHIA.md                     ← this file
```

**Simple rule:** you only ever edit files in `genos-tools/` and `tests/`.
The one exception is `genos-agent/src/tools.rs` where you register each
new tool in the dispatch table (two lines per tool).

---

## How to set up

```bash
# Clone the repo
git clone https://github.com/n33levo/genos
cd genos

# Install Rust nightly (the toolchain file handles this automatically)
rustup update

# Run the existing tests to make sure everything works before you start
cd tests
cargo test -- --test-threads=1
# Expected: "test result: ok. 86 passed"

cd ..   # back to genos/ root
```

All 86 tests must pass before you write anything. If they do not pass,
something is wrong with the environment — check with Neel before proceeding.

---

## How the tool system works

The LLM emits JSON like this:
```json
{"tool": "web.get_page", "args": {"url": "https://example.com"}, "call_id": "c1"}
```

`genos-agent/src/tools.rs` has a `dispatch()` function that matches on the
tool name and calls the right function:
```rust
"web.get_page" => genos_tools::web::tool_get_page(args, call_id),
```

The function lives in `genos-tools/src/web.rs` and returns a `ToolResult`.

The `ToolResult` gets serialized back to JSON and injected into the LLM's
context so it can read the result and decide what to do next.

---

## Pattern for every new tool

### Step 1 — Write the function in genos-tools/src/

Look at an existing simple tool for the pattern. Here is the exact structure:

```rust
// In genos-tools/src/web.rs (or a new file for a new category)

/// web.search(query, n?) -> list of {url, title, snippet}
pub fn tool_web_search(args: &JsonValue, call_id: &str) -> ToolResult {
    // 1. Extract and validate arguments
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) if !q.is_empty() => q,
        _ => return ToolResult::failure(call_id, "invalid_args", "query is required", false),
    };
    let n = args.get("n").and_then(|v| v.as_number()).unwrap_or(5.0) as usize;
    let n = n.min(10).max(1);

    // 2. Do the actual work
    let results = match do_search(query, n) {
        Ok(r) => r,
        Err(e) => return ToolResult::failure(call_id, "fetch_failed", &e, true),
    };

    // 3. Return success with a JSON result
    let items: Vec<JsonValue> = results.into_iter().map(|(url, title, snippet)| {
        json_object(&[
            ("url",     JsonValue::Str(url)),
            ("title",   JsonValue::Str(title)),
            ("snippet", JsonValue::Str(snippet)),
        ])
    }).collect();

    ToolResult::success(call_id, JsonValue::Array(items), 0)
}
```

Key rules:
- Always validate args at the top and return `ToolResult::failure` with code `"invalid_args"` for bad input
- Return `ToolResult::failure` with `retryable: true` for network/IO errors (the agent can retry)
- Return `ToolResult::failure` with `retryable: false` for logic errors
- Keep the function focused — one tool, one thing

### Step 2 — Export it from the module

If you add a new file (e.g., `src/search_web.rs`), add this to `genos-tools/src/lib.rs`:
```rust
pub mod search_web;
```

If you add to an existing file, it's already exported.

### Step 3 — Register it in the dispatch table

Open `genos-agent/src/tools.rs`, find the `dispatch()` function, and add:
```rust
"web.search" => genos_tools::web::tool_web_search(args, call_id),
```
Put it near the other `web.*` entries. That's it — two lines.

### Step 4 — Write tests

Open `tests/src/main.rs`. Find the `mod test_network` or `mod test_web` section
and add your tests there. Follow the exact pattern of existing tests.

Every tool needs at minimum:
1. A test that calls it with valid args and asserts the result is `ok: true`
   and the content is sensible (check for real expected strings, not just non-empty)
2. A test that calls it with invalid/missing args and asserts `ok: false` with
   `error.code == "invalid_args"`

Run tests with:
```bash
cd tests
cargo test -- --test-threads=1
```

Tests MUST pass before you open a PR.

---

## The three tools that need building

### Tool 1: `web.search` — Search the web

**Why it is needed:** The LLM has no way to discover URLs on its own right now.
Without search, it can only read pages it already knows the URL of, which makes
autonomous research impossible.

**How to implement it — DuckDuckGo HTML scraping:**

DuckDuckGo has a "lite" HTML endpoint that requires no API key and works with
a simple HTTP GET. The URL is:
```
https://html.duckduckgo.com/html/?q=YOUR+QUERY+HERE
```

This returns an HTML page. You fetch it with `genos_hal::net::fetch_text()`,
then parse out the result links and snippets from the HTML.

The HTML structure you're looking for (simplified):
```html
<a class="result__a" href="https://example.com">Page title</a>
<a class="result__snippet">Short description of the page</a>
```

You don't need a full HTML parser — just scan for these class patterns.

**What the tool should return:**
```json
[
  {"url": "https://example.com", "title": "Page title", "snippet": "Short description"},
  {"url": "https://other.com",   "title": "Other",      "snippet": "Another result"}
]
```

**Function signature:**
```rust
// In genos-tools/src/web.rs
pub fn tool_web_search(args: &JsonValue, call_id: &str) -> ToolResult
// args: { "query": "rust async programming", "n": 5 }
// n defaults to 5, max 10
```

**Tool name in dispatch table:** `"web.search"`

**Tests to write:**
```
web_search_returns_results        — search "rust programming", check ok=true, results non-empty,
                                    each result has "url" and "title" fields
web_search_rejects_empty_query    — empty query string → ok=false, code="invalid_args"
web_search_rejects_missing_query  — no "query" key at all → ok=false, code="invalid_args"
```

---

### Tool 2: `web.pdf` — Read a PDF from a URL

**Why it is needed:** arXiv (academic papers), government documents, technical
manuals, and most formal publications are PDFs. Without this tool, the LLM
cannot read any paper from arxiv.org or any PDF link on the web.

**How to implement it:**

1. Fetch the raw bytes with `genos_hal::net::fetch()` (the byte-level version,
   not `fetch_text`)
2. Parse the PDF manually — no external crate needed for basic PDFs

**PDF parsing (simplified, covers ~85% of real-world PDFs):**

Modern PDFs store text in "content streams" compressed with zlib (FlateDecode).
The structure you need to find:

```
Step 1: Find all objects that contain "stream" and "endstream"
Step 2: For each stream, check if it has "/Filter /FlateDecode" in its header
Step 3: Decompress the stream bytes using the `flate2` crate
Step 4: In the decompressed bytes, find text between "BT" and "ET" markers
Step 5: Within those blocks, extract strings between "(" and ")" that follow "Tj" or "TJ"
Step 6: Concatenate all extracted strings, clean up whitespace
```

You will need to add `flate2` to `genos-tools/Cargo.toml`:
```toml
[target.'cfg(feature = "hosted")'.dependencies]
flate2 = "1"
```
(Only for hosted/std builds — not available in no_std UEFI.)

**What the tool should return:**
```json
{
  "url": "https://arxiv.org/pdf/2404.01234",
  "text": "Abstract: We propose...\n1 Introduction\nRecent advances...",
  "pages_estimated": 12,
  "truncated": false
}
```

Limit extracted text to ~8000 tokens (approximately 30,000 characters) to fit
in the LLM context window. Set `"truncated": true` if you cut it off.

**Function signature:**
```rust
// In genos-tools/src/web.rs (or new file genos-tools/src/pdf.rs)
pub fn tool_web_pdf(args: &JsonValue, call_id: &str) -> ToolResult
// args: { "url": "https://arxiv.org/pdf/2404.01234" }
```

**Tool name in dispatch table:** `"web.pdf"`

**Tests to write:**
```
web_pdf_fetches_arxiv_paper       — fetch a real arxiv PDF URL, check ok=true,
                                    result has "text" field, text is non-empty,
                                    text contains at least one real word like "the" or "abstract"
web_pdf_rejects_non_pdf_url       — a non-PDF URL (or invalid URL) → ok=false
web_pdf_rejects_missing_url       — no "url" key → ok=false, code="invalid_args"
```

For the arxiv test, use a stable URL like:
`https://arxiv.org/pdf/1706.03762` (the "Attention is All You Need" paper —
it will always exist and always be a PDF)

---

### Tool 3: `web.feed` — Read an RSS or Atom feed

**Why it is needed:** RSS/Atom feeds let the LLM monitor a stream of content —
arXiv's daily new papers, news sites, blogs. Without this, the agent can't do
"check what's new in machine learning today" without manually visiting pages.

**How to implement it:**

1. Fetch the URL with `genos_hal::net::fetch_text()`
2. Parse the XML by scanning for the relevant tags

RSS and Atom are both XML. You do NOT need a full XML parser.
Scan for these patterns:

**RSS format:**
```xml
<item>
  <title>Paper title here</title>
  <link>https://arxiv.org/abs/2404.01234</link>
  <description>Summary text here</description>
  <pubDate>Thu, 04 Apr 2024 00:00:00 GMT</pubDate>
</item>
```

**Atom format:**
```xml
<entry>
  <title>Paper title here</title>
  <link href="https://arxiv.org/abs/2404.01234"/>
  <summary>Summary text here</summary>
  <published>2024-04-04T00:00:00Z</published>
</entry>
```

For both: extract the inner text of `<title>`, `<link>` (or `href` attribute),
`<description>`/`<summary>`, and `<pubDate>`/`<published>` for each item/entry.

A simple XML tag extractor:
```rust
fn extract_tag<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open  = format!("<{}>",  tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)?  + open.len();
    let end   = xml[start..].find(&close)? + start;
    Some(&xml[start..end])
}
```

**What the tool should return:**
```json
{
  "feed_title": "arXiv cs.AI",
  "feed_url": "https://export.arxiv.org/rss/cs.AI",
  "items": [
    {
      "title":       "Attention Sinks in Large Language Models",
      "url":         "https://arxiv.org/abs/2404.01234",
      "summary":     "We study the phenomenon of attention sinks...",
      "published_at": "2024-04-04T00:00:00Z"
    }
  ],
  "item_count": 25
}
```

**Function signature:**
```rust
// In genos-tools/src/web.rs (or new file genos-tools/src/feed.rs)
pub fn tool_web_feed(args: &JsonValue, call_id: &str) -> ToolResult
// args: { "url": "https://export.arxiv.org/rss/cs.AI", "max_items": 20 }
// max_items defaults to 20, max 50
```

**Tool name in dispatch table:** `"web.feed"`

**Good test feed URLs (always available, no auth required):**
- `https://export.arxiv.org/rss/cs.AI` — arXiv AI papers feed (RSS)
- `https://hnrss.org/frontpage` — Hacker News front page (RSS)

**Tests to write:**
```
web_feed_reads_arxiv_rss          — fetch the arXiv AI RSS feed, check ok=true,
                                    result has "items" array, at least 1 item,
                                    items have "title" and "url" fields
web_feed_reads_hackernews         — fetch HN frontpage feed, same checks
web_feed_rejects_missing_url      — no "url" key → ok=false, code="invalid_args"
web_feed_rejects_empty_url        — empty "url" → ok=false, code="invalid_args"
```

---

## How fetch works (background info)

All network access goes through `genos-hal`. Do NOT use `reqwest` or any other
HTTP crate — go through the HAL:

```rust
use genos_hal::net;

// For text responses (HTML, XML, plain text):
let text: String = net::fetch_text(url).map_err(|e| format!("{:?}", e))?;

// For binary responses (PDFs, images):
let bytes: Vec<u8> = net::fetch(url).map_err(|e| format!("{:?}", e))?;
```

`fetch_text` and `fetch` both handle HTTP and HTTPS, follow redirects
(up to 5 hops), and have a 15-second timeout. You do not need to do anything
special for HTTPS — it just works.

---

## Important constraints

- **No external crates** for network access, JSON, or string parsing — the
  HAL and kernel already provide these
- `flate2` is allowed for PDF decompression (add to `genos-tools/Cargo.toml`
  under the hosted feature gate as shown in the PDF section above)
- All new code must be in `genos-tools/src/` and use `no_std`-compatible
  approaches where possible (avoid `std::collections::HashMap` — use
  `alloc::collections::BTreeMap` instead; avoid `println!` — result data
  goes into ToolResult, not stdout)
- Function signatures must match: `pub fn tool_xyz(args: &JsonValue, call_id: &str) -> ToolResult`

---

## Checklist before opening a PR

- [ ] `cd tests && cargo test -- --test-threads=1` → all tests green, zero failures
- [ ] No compiler warnings: `cargo build --tests 2>&1 | grep warning` finds nothing new
- [ ] Each new tool is registered in `genos-agent/src/tools.rs`
- [ ] Each new tool has at least one "happy path" test and one "invalid args" test
- [ ] The happy-path tests use real network endpoints and assert meaningful content
      (not just "non-empty" — check for an actual expected word or field)

---

## Quick reference: useful types

```rust
use genos_kernel::json::{json_object, JsonValue};
use crate::protocol::ToolResult;

// Build a JSON object:
json_object(&[
    ("key1", JsonValue::Str(String::from("value"))),
    ("key2", JsonValue::Number(42.0)),
    ("key3", JsonValue::Bool(true)),
    ("key4", JsonValue::Array(vec![...])),
])

// Return success:
ToolResult::success(call_id, JsonValue::Array(items), 0)

// Return failure:
ToolResult::failure(call_id, "invalid_args", "url is required", false)
ToolResult::failure(call_id, "fetch_failed", "connection timed out", true)
//                                                                    ^^^^ retryable?
```

---

## Questions?

Ask Neel. Or read `MASTER.md` for the full architecture spec.
The existing tools in `genos-tools/src/web.rs` are the best reference —
especially `tool_get_page` which does fetch + parse + return all in one function.
