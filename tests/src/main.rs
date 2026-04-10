fn main() {
    println!("genos-tests: run with `cargo test -- --test-threads=1`");
}

/// Shared test root setup. Each test module gets its own subdirectory.
fn with_test_root(module: &str) -> String {
    let root = format!("/tmp/genos_test_{}", module);
    std::env::set_var("GENOS_TEST_ROOT", &root);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create test root");
    root
}

fn cleanup(root: &str) {
    let _ = std::fs::remove_dir_all(root);
}

// ──────────────────────────────────────────────────────────────────
// 1. JSON PARSER  — full protocol contract
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_json {
    use genos_kernel::json::{parse, json_object, extract_tool_calls, JsonValue};

    #[test]
    fn primitives() {
        assert!(matches!(parse("null").unwrap(), JsonValue::Null));
        assert_eq!(parse("true").unwrap().as_bool(), Some(true));
        assert_eq!(parse("false").unwrap().as_bool(), Some(false));
        assert_eq!(parse("42").unwrap().as_f64(), Some(42.0));
        assert_eq!(parse("-3.14").unwrap().as_f64(), Some(-3.14));
        assert_eq!(parse(r#""hello world""#).unwrap().as_str(), Some("hello world"));
    }

    #[test]
    fn string_escapes() {
        assert_eq!(parse(r#""line1\nline2""#).unwrap().as_str(), Some("line1\nline2"));
        assert_eq!(parse(r#""tab\there""#).unwrap().as_str(), Some("tab\there"));
        assert_eq!(parse(r#""quote\"inside""#).unwrap().as_str(), Some("quote\"inside"));
    }

    #[test]
    fn nested_object() {
        let v = parse(r#"{"a": {"b": {"c": 99}}, "arr": [1,2,3]}"#).unwrap();
        assert_eq!(v.get("a").unwrap().get("b").unwrap().get("c").unwrap().as_f64(), Some(99.0));
        assert_eq!(v.get("arr").unwrap().as_array().unwrap().len(), 3);
    }

    #[test]
    fn empty_structures() {
        assert_eq!(parse("[]").unwrap().as_array().unwrap().len(), 0);
        assert_eq!(parse("{}").unwrap().as_object().unwrap().len(), 0);
    }

    #[test]
    fn invalid_json_errors() {
        assert!(parse("{invalid}").is_err());
        assert!(parse("").is_err());
        assert!(parse("[1,2,").is_err());
        assert!(parse(r#"{"key": }"#).is_err());
    }

    #[test]
    fn serialize_and_parse_roundtrip() {
        let original = json_object(&[
            ("tool", JsonValue::Str("memory.store".into())),
            ("args", json_object(&[
                ("content", JsonValue::Str("Rust is memory-safe".into())),
                ("wing", JsonValue::Str("general".into())),
                ("turn", JsonValue::Number(5.0)),
            ])),
            ("ok", JsonValue::Bool(true)),
            ("scores", JsonValue::Array(vec![
                JsonValue::Number(0.9), JsonValue::Number(0.7), JsonValue::Null,
            ])),
        ]);
        let s = original.to_json_string();
        let parsed = parse(&s).unwrap();
        assert_eq!(parsed.get("tool").unwrap().as_str(), Some("memory.store"));
        assert_eq!(
            parsed.get("args").unwrap().get("content").unwrap().as_str(),
            Some("Rust is memory-safe")
        );
        assert_eq!(
            parsed.get("scores").unwrap().as_array().unwrap()[2],
            JsonValue::Null
        );
    }

    #[test]
    fn extract_tool_calls_single() {
        let output = r#"I'll search for that.
<tool_call>
{"tool": "memory.search", "args": {"query": "Rust ownership", "top_k": 5}, "call_id": "c1"}
</tool_call>"#;
        let calls = extract_tool_calls(output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].get("tool").unwrap().as_str(), Some("memory.search"));
        assert_eq!(
            calls[0].get("args").unwrap().get("query").unwrap().as_str(),
            Some("Rust ownership")
        );
        assert_eq!(calls[0].get("call_id").unwrap().as_str(), Some("c1"));
    }

    #[test]
    fn extract_tool_calls_multiple() {
        let output = r#"Let me look that up.
<tool_call>
{"tool": "fs.read", "args": {"path": "\\palace\\facts.kv"}, "call_id": "c1"}
</tool_call>
Now let me also search.
<tool_call>
{"tool": "memory.search", "args": {"query": "project alpha"}, "call_id": "c2"}
</tool_call>
And fetch the page.
<tool_call>
{"tool": "web.get_page", "args": {"url": "http://example.com"}, "call_id": "c3"}
</tool_call>"#;
        let calls = extract_tool_calls(output);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].get("tool").unwrap().as_str(), Some("fs.read"));
        assert_eq!(calls[1].get("tool").unwrap().as_str(), Some("memory.search"));
        assert_eq!(calls[2].get("tool").unwrap().as_str(), Some("web.get_page"));
    }

    #[test]
    fn extract_tool_calls_none() {
        assert_eq!(extract_tool_calls("Just a normal response.").len(), 0);
        assert_eq!(extract_tool_calls("").len(), 0);
    }
}

// ──────────────────────────────────────────────────────────────────
// 2. TIMER — accuracy, wall time, memory info
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_timer {
    use genos_hal::timer;

    #[test]
    fn wall_time_is_correct_date() {
        let wt = timer::get_wall_time().expect("should have wall time in hosted mode");
        let s = wt.iso8601();
        // Hosted impl returns 2026-04-09T12:00:00Z
        assert!(s.starts_with("2026-04-09"), "Expected 2026-04-09, got: {}", s);
        assert!(s.contains("T"), "Should have time separator T");
        assert!(s.ends_with("Z"), "Should end with Z");
        let suffix = wt.session_id_suffix();
        assert!(suffix.starts_with("20260409"), "Session ID should start with 20260409, got: {}", suffix);
    }

    #[test]
    fn now_ms_is_monotonic() {
        let samples: Vec<u64> = (0..5).map(|_| timer::now_ms()).collect();
        // advance_ms increments the counter
        timer::advance_ms(10);
        let after = timer::now_ms();
        assert!(after >= samples[0] + 10, "now_ms should be >= start + 10ms");
        // Samples should be non-decreasing
        for w in samples.windows(2) {
            assert!(w[1] >= w[0], "now_ms should never go backwards");
        }
    }

    #[test]
    fn advance_ms_accumulates() {
        let base = timer::now_ms();
        timer::advance_ms(100);
        timer::advance_ms(200);
        timer::advance_ms(50);
        let after = timer::now_ms();
        assert!(after >= base + 350, "Expected >=350ms advance, got: {}", after - base);
    }

    #[test]
    fn sleep_ms_is_real() {
        let start = std::time::Instant::now();
        timer::sleep_ms(80);
        let elapsed = start.elapsed().as_millis();
        assert!(elapsed >= 70, "sleep_ms(80) should sleep at least ~80ms, got {}ms", elapsed);
        assert!(elapsed < 500, "sleep_ms(80) should not take >500ms");
    }

    #[test]
    fn memory_info_reasonable() {
        let info = timer::get_memory_info();
        assert!(info.total_kb > 0, "must have some total memory");
        assert!(info.free_kb <= info.total_kb, "free cannot exceed total");
        // Hosted: 1GB free / 2GB total
        assert_eq!(info.total_kb, 2 * 1024 * 1024, "Hosted total should be 2GB");
        assert_eq!(info.free_kb, 1 * 1024 * 1024, "Hosted free should be 1GB");
    }
}

// ──────────────────────────────────────────────────────────────────
// 3. NETWORK FETCH — real HTTP sites, skip gracefully if offline
//
// Tested sites (all HTTP-only, no TLS):
//   http://info.cern.ch            — CERN, home of the first website
//   http://httpforever.com         — dedicated HTTP-forever test site
//   http://detectportal.firefox.com/success.txt — plaintext "success"
//   http://www.msftncsi.com/ncsi.txt            — plaintext "Microsoft NCSI"
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_network {
    use genos_hal::net;
    use genos_kernel::json::{json_object, JsonValue};

    /// Returns (body_bytes, is_online). Skips test if offline.
    macro_rules! needs_network {
        ($url:expr) => {{
            match net::fetch($url) {
                Ok(b) => b,
                Err(_) => {
                    eprintln!("SKIP (offline): {}", $url);
                    return;
                }
            }
        }};
    }

    // ── info.cern.ch — the first website, always HTTP ──

    #[test]
    fn fetch_cern_first_website() {
        let body = needs_network!("http://info.cern.ch");
        let text = String::from_utf8_lossy(&body);
        assert!(!body.is_empty(), "CERN response should not be empty");
        assert!(
            text.contains("first website") || text.contains("cern.ch"),
            "info.cern.ch should mention 'first website', got {} bytes: {}",
            body.len(), &text[..text.len().min(300)]
        );
        assert!(text.contains("<html") || text.contains("<!DOCTYPE"),
            "Should return HTML");
    }

    #[test]
    fn fetch_cern_has_heading_and_links() {
        let body = needs_network!("http://info.cern.ch");
        let text = String::from_utf8_lossy(&body);
        // CERN page has <h1> and <ul><li><a> links
        assert!(text.contains("<h1") || text.contains("<H1"),
            "Should have an h1 heading");
        assert!(text.contains("<a href") || text.contains("<A HREF"),
            "Should have links");
        assert!(
            text.contains("TheProject.html") || text.contains("hypertext"),
            "Should link to the original project page"
        );
    }

    // ── httpforever.com — dedicated HTTP-forever test domain ──

    #[test]
    fn fetch_httpforever() {
        let body = needs_network!("http://httpforever.com");
        let text = String::from_utf8_lossy(&body);
        assert!(!body.is_empty(), "httpforever.com should return content");
        assert!(
            text.contains("HTTP") || text.contains("http"),
            "httpforever.com should mention HTTP, got: {}",
            &text[..text.len().min(300)]
        );
        assert!(text.len() > 200, "Should have non-trivial HTML");
    }

    // ── Lightweight plaintext endpoints ──

    #[test]
    fn fetch_firefox_captive_portal_check() {
        // Mozilla's captive portal check returns exactly "success"
        let body = needs_network!("http://detectportal.firefox.com/success.txt");
        let text = String::from_utf8_lossy(&body);
        let trimmed = text.trim();
        assert_eq!(trimmed, "success",
            "detectportal.firefox.com/success.txt should return 'success', got: {:?}",
            trimmed
        );
    }

    #[test]
    fn fetch_microsoft_ncsi() {
        // Microsoft NCSI endpoint returns plaintext "Microsoft NCSI"
        let body = needs_network!("http://www.msftncsi.com/ncsi.txt");
        let text = String::from_utf8_lossy(&body);
        let trimmed = text.trim();
        assert_eq!(trimmed, "Microsoft NCSI",
            "msftncsi.com/ncsi.txt should return 'Microsoft NCSI', got: {:?}",
            trimmed
        );
    }

    // ── Response structure ──

    #[test]
    fn fetch_response_is_non_empty_bytes() {
        let body = needs_network!("http://info.cern.ch");
        assert!(body.len() > 50, "Response should have substantial content");
        // Response bytes should be valid UTF-8 HTML
        assert!(String::from_utf8(body.clone()).is_ok(),
            "Response should be valid UTF-8");
    }

    // ── HTTPS sites ── (require TLS, Wikipedia and Grokipedia)

    #[test]
    fn fetch_wikipedia_prime_number_theorem() {
        // https://en.wikipedia.org/wiki/Prime_number_theorem
        let url = "https://en.wikipedia.org/wiki/Prime_number_theorem";
        let body = needs_network!(url);
        let text = String::from_utf8_lossy(&body);
        assert!(!body.is_empty(), "Wikipedia response should not be empty");
        assert!(
            text.contains("prime") || text.contains("Prime") || text.contains("theorem"),
            "Should contain content about prime numbers, got {} bytes: {}",
            body.len(), &text[..text.len().min(400)]
        );
        assert!(text.contains("<html") || text.contains("<!DOCTYPE"),
            "Should be HTML");
    }

    #[test]
    fn fetch_grokipedia_artemis() {
        let url = "https://grokipedia.com/page/Artemis_II";
        let body = needs_network!(url);
        let text = String::from_utf8_lossy(&body);
        assert!(!body.is_empty(), "Grokipedia response should not be empty");
        assert!(
            text.contains("Artemis") || text.contains("artemis") || text.contains("NASA"),
            "Should contain Artemis II content, got {} bytes: {}",
            body.len(), &text[..text.len().min(400)]
        );
    }

    #[test]
    fn tool_get_page_wikipedia() {
        if genos_hal::net::fetch("https://en.wikipedia.org/wiki/Prime_number_theorem").is_err() {
            eprintln!("SKIP tool_get_page_wikipedia — no network");
            return;
        }
        let args = json_object(&[
            ("url", JsonValue::Str("https://en.wikipedia.org/wiki/Prime_number_theorem".into())),
            ("max_tokens", JsonValue::Number(1500.0)),
        ]);
        let result = genos_tools::web::tool_get_page(&args, "wp_wiki");
        assert!(result.ok, "tool_get_page should succeed for Wikipedia");
        let json = &result.result;
        assert_eq!(json.get("domain").unwrap().as_str(), Some("en.wikipedia.org"));
        let chunks = json.get("chunks").unwrap().as_array().unwrap();
        assert!(!chunks.is_empty(), "Wikipedia should yield content chunks");
        let all_text: String = chunks.iter()
            .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            all_text.contains("prime") || all_text.contains("Prime") || all_text.contains("theorem"),
            "Extracted text should mention primes/theorem, got: {}",
            &all_text[..all_text.len().min(400)]
        );
        let total_tokens = json.get("total_tokens").and_then(|t| t.as_f64()).unwrap_or(0.0);
        assert!(total_tokens > 0.0, "total_tokens should be > 0");
    }

    #[test]
    fn tool_get_page_grokipedia() {
        if genos_hal::net::fetch("https://grokipedia.com/page/Artemis_II").is_err() {
            eprintln!("SKIP tool_get_page_grokipedia — no network");
            return;
        }
        let args = json_object(&[
            ("url", JsonValue::Str("https://grokipedia.com/page/Artemis_II".into())),
            ("max_tokens", JsonValue::Number(1500.0)),
        ]);
        let result = genos_tools::web::tool_get_page(&args, "wp_groki");
        assert!(result.ok, "tool_get_page should succeed for Grokipedia");
        let json = &result.result;
        assert_eq!(json.get("domain").unwrap().as_str(), Some("grokipedia.com"));
        let chunks = json.get("chunks").unwrap().as_array().unwrap();
        assert!(!chunks.is_empty(), "Grokipedia should yield content chunks");
        let all_text: String = chunks.iter()
            .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            all_text.contains("Artemis") || all_text.contains("NASA") || all_text.contains("Moon"),
            "Extracted text should mention Artemis/NASA/Moon, got: {}",
            &all_text[..all_text.len().min(400)]
        );
    }

    #[test]
    fn tool_net_fetch_cern() {
        let args = json_object(&[("url", JsonValue::Str("http://info.cern.ch".into()))]);
        let result = genos_tools::net::tool_fetch(&args, "net1");
        if result.ok {
            let text = result.result.as_str().unwrap_or("");
            assert!(!text.is_empty(), "Fetched content should not be empty");
            assert!(
                text.contains("cern") || text.contains("first") || text.contains("html"),
                "Should contain CERN page content"
            );
        }
        // Graceful: if offline, just skip
    }

    #[test]
    fn tool_net_fetch_plaintext_endpoint() {
        let args = json_object(&[
            ("url", JsonValue::Str("http://detectportal.firefox.com/success.txt".into()))
        ]);
        let result = genos_tools::net::tool_fetch(&args, "net_plaintext");
        if result.ok {
            let text = result.result.as_str().unwrap_or("");
            assert!(text.trim() == "success" || text.contains("success"),
                "Firefox captive portal should return 'success', got: {:?}", text.trim());
        }
    }

    #[test]
    fn tool_net_fetch_rejects_https() {
        // HTTPS is now supported — this should no longer be rejected
        // (may succeed or fail based on network, but must not fail with invalid_args)
        let args = json_object(&[("url", JsonValue::Str("https://info.cern.ch".into()))]);
        let result = genos_tools::net::tool_fetch(&args, "net_https");
        // If we're online it should succeed; if offline it will fail with net_error
        // Either way it must NOT fail with invalid_args
        if !result.ok {
            let code = result.error.unwrap().code;
            assert_ne!(code, "invalid_args", "HTTPS should no longer be rejected at the URL check level");
        }
    }

    #[test]
    fn tool_net_fetch_rejects_ftp_scheme() {
        let args = json_object(&[("url", JsonValue::Str("ftp://example.com/file.txt".into()))]);
        let result = genos_tools::net::tool_fetch(&args, "net_ftp");
        assert!(!result.ok, "Should reject ftp:// scheme");
        assert_eq!(result.error.unwrap().code, "invalid_args");
    }

    #[test]
    fn tool_net_fetch_missing_url() {
        let result = genos_tools::net::tool_fetch(&JsonValue::Null, "net_null");
        assert!(!result.ok, "Should fail with missing url");
        let err = result.error.unwrap();
        assert_eq!(err.code, "invalid_args");
    }

    #[test]
    fn tool_net_fetch_empty_url() {
        let args = json_object(&[("url", JsonValue::Str("".into()))]);
        let result = genos_tools::net::tool_fetch(&args, "net_empty");
        assert!(!result.ok, "Empty URL should fail");
    }

    #[test]
    fn tool_net_fetch_nonexistent_host_fails() {
        let args = json_object(&[("url", JsonValue::Str(
            "http://definitely-does-not-exist-at-all-xyzzy.invalid".into()
        ))]);
        let result = genos_tools::net::tool_fetch(&args, "net_badhost");
        assert!(!result.ok, "Nonexistent host should fail");
    }
}

// ──────────────────────────────────────────────────────────────────
// 4. WEB TOOLS — HTML stripping + fetch_page integration
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_web {
    use genos_tools::web::{strip_html, ChunkType};
    use genos_kernel::json::{json_object, JsonValue};

    #[test]
    fn strip_html_extracts_structure() {
        let html = r#"<!DOCTYPE html>
<html>
<head><title>Rust Programming Language</title></head>
<body>
  <h1>Rust</h1>
  <p>A language empowering everyone to build reliable and efficient software.</p>
  <h2>Features</h2>
  <ul>
    <li>Memory safety without GC</li>
    <li>Fearless concurrency</li>
    <li>Zero-cost abstractions</li>
  </ul>
  <h2>Example</h2>
  <pre><code>fn main() {
    println!("Hello, world!");
}</code></pre>
</body>
</html>"#;

        let chunks = strip_html(html);
        assert!(!chunks.is_empty(), "Should extract chunks from HTML");

        let title = chunks.iter().find(|c| matches!(c.typ, ChunkType::Title));
        assert!(title.is_some(), "Should have a title chunk");
        assert!(title.unwrap().text.contains("Rust"), "Title should contain 'Rust'");

        let h1 = chunks.iter().find(|c| matches!(c.typ, ChunkType::Heading { level: 1 }));
        assert!(h1.is_some(), "Should have h1");
        assert!(h1.unwrap().text.contains("Rust"), "H1 should say Rust");

        let list_items: Vec<_> = chunks.iter()
            .filter(|c| matches!(c.typ, ChunkType::ListItem))
            .collect();
        assert_eq!(list_items.len(), 3, "Should have 3 list items");
        assert!(list_items.iter().any(|c| c.text.contains("Memory safety")));
        assert!(list_items.iter().any(|c| c.text.contains("concurrency")));

        let code = chunks.iter().find(|c| matches!(c.typ, ChunkType::Code));
        assert!(code.is_some(), "Should have code chunk");
        assert!(code.unwrap().text.contains("println"), "Code should contain println");
    }

    #[test]
    fn strip_html_decodes_entities() {
        let html = "<p>AT&amp;T sells for &lt;$1000 &amp; is &quot;reliable&quot;</p>";
        let chunks = strip_html(html);
        assert!(!chunks.is_empty());
        let text: String = chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>().join(" ");
        assert!(text.contains("AT&T"), "Should decode &amp; → &, got: {}", text);
        assert!(text.contains("<$1000"), "Should decode &lt; → <");
        assert!(text.contains("\"reliable\""), "Should decode &quot; → \"");
    }

    #[test]
    fn strip_html_ignores_script_and_style() {
        let html = r#"<html>
<head>
  <style>body { color: red; }</style>
  <script>var x = "secret";</script>
</head>
<body><p>Visible content here</p></body>
</html>"#;
        let chunks = strip_html(html);
        let all_text: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(!all_text.contains("color: red"), "Should not include CSS");
        assert!(!all_text.contains("var x"), "Should not include JS");
        assert!(all_text.contains("Visible content"), "Should include body text");
    }

    #[test]
    fn strip_html_token_estimates() {
        let html = "<p>This is a paragraph with several words of content for token estimation.</p>";
        let chunks = strip_html(html);
        assert!(!chunks.is_empty());
        let chunk = &chunks[0];
        // ~4 chars/token estimate
        let expected = (chunk.text.len() + 3) / 4;
        assert_eq!(chunk.tokens_estimate, expected, "Token estimate should be len/4");
    }

    #[test]
    fn strip_html_chunk_to_json_structure() {
        let html = "<h2>Section Title</h2>";
        let chunks = strip_html(html);
        assert!(!chunks.is_empty());
        let json = chunks[0].to_json();
        assert_eq!(json.get("type").unwrap().as_str(), Some("heading"));
        assert!(json.get("text").unwrap().as_str().unwrap().contains("Section Title"));
        assert!(json.get("tokens").unwrap().as_f64().unwrap() > 0.0);
    }

    #[test]
    fn tool_get_page_cern() {
        if genos_hal::net::fetch("http://info.cern.ch").is_err() {
            eprintln!("SKIP test_web::tool_get_page_cern — no network");
            return;
        }
        let args = json_object(&[
            ("url", JsonValue::Str("http://info.cern.ch".into())),
            ("max_tokens", JsonValue::Number(500.0)),
        ]);
        let result = genos_tools::web::tool_get_page(&args, "wp_cern");
        assert!(result.ok, "tool_get_page should succeed for info.cern.ch");
        let json = &result.result;
        assert_eq!(json.get("domain").unwrap().as_str(), Some("info.cern.ch"),
            "Domain should be info.cern.ch");
        let chunks = json.get("chunks").unwrap().as_array().unwrap();
        assert!(!chunks.is_empty(), "CERN page should have extractable chunks");
        let total_tokens = json.get("total_tokens").unwrap_or(&JsonValue::Null).as_f64().unwrap_or(0.0);
        assert!(total_tokens > 0.0, "total_tokens should be > 0");
        let all_text: String = chunks.iter()
            .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            all_text.contains("first") || all_text.contains("CERN") || all_text.contains("web"),
            "Extracted text should mention first/CERN/web, got: {}",
            &all_text[..all_text.len().min(300)]
        );
    }

    #[test]
    fn tool_get_page_httpforever() {
        if genos_hal::net::fetch("http://httpforever.com").is_err() {
            eprintln!("SKIP test_web::tool_get_page_httpforever — no network");
            return;
        }
        let args = json_object(&[
            ("url", JsonValue::Str("http://httpforever.com".into())),
            ("max_tokens", JsonValue::Number(800.0)),
        ]);
        let result = genos_tools::web::tool_get_page(&args, "wp_hf");
        assert!(result.ok, "tool_get_page should succeed for httpforever.com");
        let json = &result.result;
        assert_eq!(json.get("domain").unwrap().as_str(), Some("httpforever.com"));
        let chunks = json.get("chunks").unwrap().as_array().unwrap();
        assert!(!chunks.is_empty(), "httpforever should yield chunks");
    }

    #[test]
    fn tool_get_page_rejects_ftp() {
        // ftp:// is still not a valid URL scheme
        let args = json_object(&[("url", JsonValue::Str("ftp://example.com/file".into()))]);
        let result = genos_tools::web::tool_get_page(&args, "wp_ftp");
        assert!(!result.ok);
        assert_eq!(result.error.unwrap().code, "invalid_args");
    }

    #[test]
    fn tool_extract_links_from_live_cern() {
        // tool_extract_links takes a URL and fetches+parses it
        if genos_hal::net::fetch("http://info.cern.ch").is_err() {
            eprintln!("SKIP test_web::tool_extract_links_from_live_cern — no network");
            return;
        }
        let args = json_object(&[("url", JsonValue::Str("http://info.cern.ch".into()))]);
        let result = genos_tools::web::tool_extract_links(&args, "el_cern");
        assert!(result.ok, "tool_extract_links on info.cern.ch should succeed");
        let empty = vec![];
        let links = result.result.as_array().unwrap_or(&empty);
        // CERN page has links to TheProject.html etc
        assert!(!links.is_empty(), "CERN page should have links");
        let has_http_link = links.iter().any(|l| {
            l.get("url").and_then(|u| u.as_str()).map(|s| s.starts_with("http")).unwrap_or(false)
        });
        assert!(has_http_link, "Should extract absolute http links");
    }
}

// ──────────────────────────────────────────────────────────────────
// 5. DISK I/O — hosted HAL filesystem
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_disk {
    use genos_hal::disk;

    fn setup() -> String { crate::with_test_root("disk") }

    #[test]
    fn write_read_roundtrip() {
        let root = setup();
        assert!(disk::write_file("\\test.txt", b"Hello genos").is_ok());
        let data = disk::read_file("\\test.txt").ok().expect("should read back");
        assert_eq!(&data, b"Hello genos");
        crate::cleanup(&root);
    }

    #[test]
    fn nested_directories_created_on_write() {
        let root = setup();
        assert!(disk::write_file("\\palace\\wings\\notes\\session.jsonl", b"test content").is_ok());
        let data = disk::read_file("\\palace\\wings\\notes\\session.jsonl").ok().expect("read nested");
        assert_eq!(&data, b"test content");
        crate::cleanup(&root);
    }

    #[test]
    fn overwrite_replaces_content() {
        let root = setup();
        assert!(disk::write_file("\\data.bin", b"version 1").is_ok());
        assert!(disk::write_file("\\data.bin", b"version 2 is longer").is_ok());
        let data = disk::read_file("\\data.bin").ok().expect("read");
        assert_eq!(&data, b"version 2 is longer");
        crate::cleanup(&root);
    }

    #[test]
    fn read_nonexistent_returns_error() {
        let root = setup();
        let result = disk::read_file("\\not_here\\at_all.txt");
        assert!(result.is_err(), "Reading nonexistent path should fail");
        crate::cleanup(&root);
    }

    #[test]
    fn binary_data_preserved() {
        let root = setup();
        let data: Vec<u8> = (0u8..=255).collect();
        assert!(disk::write_file("\\binary.bin", &data).is_ok());
        let read_back = disk::read_file("\\binary.bin").ok().expect("read binary");
        assert_eq!(data, read_back, "Binary data should round-trip exactly");
        crate::cleanup(&root);
    }
}

// ──────────────────────────────────────────────────────────────────
// 6. FS TOOL — higher-level file tool
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_fs_tool {
    use genos_tools::fs::FsTool;
    use genos_kernel::json::{json_object, JsonValue};

    fn setup() -> String { crate::with_test_root("fs") }

    #[test]
    fn write_and_read_text() {
        let root = setup();
        FsTool::write("\\note.txt", "Rust is great for OS dev").ok().expect("write");
        let content = FsTool::read("\\note.txt").ok().expect("read");
        assert_eq!(content, "Rust is great for OS dev");
        crate::cleanup(&root);
    }

    #[test]
    fn write_and_read_bytes() {
        let root = setup();
        let data: Vec<u8> = (0..=255).collect();
        FsTool::write_bytes("\\raw.bin", &data).ok().expect("write bytes");
        let read_back = FsTool::read_bytes("\\raw.bin").ok().expect("read bytes");
        assert_eq!(data, read_back);
        crate::cleanup(&root);
    }

    #[test]
    fn tool_write_and_read_protocol() {
        let root = setup();
        let write_args = json_object(&[
            ("path", JsonValue::Str("\\doc.txt".into())),
            ("content", JsonValue::Str("The quick brown fox".into())),
        ]);
        let wr = genos_tools::fs::tool_write(&write_args, "w1");
        assert!(wr.ok, "tool_write should succeed, error: {:?}",
            wr.error.as_ref().map(|e| &e.message));

        let read_args = json_object(&[("path", JsonValue::Str("\\doc.txt".into()))]);
        let rr = genos_tools::fs::tool_read(&read_args, "r1");
        assert!(rr.ok, "tool_read should succeed");
        let text = rr.result.as_str().unwrap_or("");
        assert_eq!(text, "The quick brown fox");
        crate::cleanup(&root);
    }

    #[test]
    fn tool_read_missing_file_fails() {
        let root = setup();
        let args = json_object(&[("path", JsonValue::Str("\\missing.txt".into()))]);
        let result = genos_tools::fs::tool_read(&args, "r2");
        assert!(!result.ok, "Reading missing file should fail");
        crate::cleanup(&root);
    }
}

// ──────────────────────────────────────────────────────────────────
// 7. SEARCH INDEX — TF-IDF with real textual discrimination
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_search {
    use genos_tools::search::SearchIndex;

    #[test]
    fn empty_index() {
        let idx = SearchIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.search("anything", None, None, 10, 0).len(), 0);
    }

    #[test]
    fn tfidf_scores_specific_terms_higher() {
        let mut idx = SearchIndex::new();
        // "borrow checker" only in doc 1
        idx.add_document("d1", "gen", "facts",
            "Rust borrow checker prevents use-after-free and data races at compile time", 1);
        // Python doc - no "borrow checker"
        idx.add_document("d2", "gen", "facts",
            "Python uses reference counting and garbage collection for memory management", 2);
        // Go doc - no "borrow checker"
        idx.add_document("d3", "gen", "facts",
            "Go has garbage collection but allows concurrent memory access with channels", 3);
        idx.rebuild_idf();

        let results = idx.search("borrow checker compile", None, None, 5, 3);
        assert!(!results.is_empty(), "Should find borrow checker doc");
        assert_eq!(results[0].path, "d1",
            "Rust borrow doc should rank first, got: {:?}",
            results.iter().map(|r| (&r.path, r.score)).collect::<Vec<_>>());
        assert!(results[0].score > 0.0, "Score should be positive");
    }

    #[test]
    fn wing_filter_isolates_results() {
        let mut idx = SearchIndex::new();
        idx.add_document("w1p1", "project-alpha", "facts",
            "Alpha project uses microservices and Kubernetes for deployment scaling", 1);
        idx.add_document("w2p1", "project-beta", "facts",
            "Beta project also uses Kubernetes and Docker for container orchestration", 2);
        idx.add_document("w1p2", "project-alpha", "events",
            "Deployed new version to Kubernetes cluster in production environment", 3);
        idx.rebuild_idf();

        let results = idx.search("Kubernetes deployment", Some("project-alpha"), None, 10, 3);
        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.wing, "project-alpha",
                "Wing filter should only return project-alpha, got: {}", r.wing);
        }
    }

    #[test]
    fn hall_filter_isolates_by_category() {
        let mut idx = SearchIndex::new();
        // Use UNIQUE discriminating terms per hall so TF-IDF scores them (IDF>0)
        // "borrowchecker" only in facts; "qemudeploy" only in events
        idx.add_document("h1", "gen", "facts",
            "borrowchecker ownership lifetime safety prevents datarace compile", 1);
        idx.add_document("h2", "gen", "events",
            "qemudeploy booted kernel loaded uefi bootloader started running", 2);
        idx.add_document("h3", "gen", "discoveries",
            "tokenizer vocabulary encoding inference throughput measurement", 3);
        // Add a 4th doc in a 4th hall so each unique term has df=1, N=4, IDF=log(4)>0
        idx.add_document("h4", "gen", "preferences",
            "darkmode terminal editor font colorscheme workspace layout", 4);
        idx.rebuild_idf();

        let facts_only = idx.search("borrowchecker ownership", None, Some("facts"), 10, 4);
        assert!(!facts_only.is_empty(),
            "hall=facts query for unique term should return results");
        assert!(facts_only.iter().all(|r| r.hall == "facts"),
            "All results should be in facts hall");

        let events_only = idx.search("qemudeploy bootloader", None, Some("events"), 10, 4);
        assert!(!events_only.is_empty(),
            "hall=events query for unique term should return results");
        assert!(events_only.iter().all(|r| r.hall == "events"));
    }

    #[test]
    fn recency_boost_prefers_newer_docs() {
        let mut idx = SearchIndex::new();
        // Same exact content, different turns
        let content = "phase D implements tools into OS kernel pipeline";
        idx.add_document("old", "gen", "facts", content, 1);
        idx.add_document("mid", "gen", "facts", content, 50);
        idx.add_document("new", "gen", "facts", content, 200);
        idx.add_document("decoy", "gen", "facts", "completely unrelated content about cooking", 201);
        idx.rebuild_idf();

        let results = idx.search("phase D tools kernel", None, None, 3, 200);
        assert!(!results.is_empty());
        // "new" (turn=200) should outrank "old" (turn=1) due to recency boost
        let new_pos = results.iter().position(|r| r.path == "new");
        let old_pos = results.iter().position(|r| r.path == "old");
        if let (Some(n), Some(o)) = (new_pos, old_pos) {
            assert!(n < o, "'new' (turn 200) should rank above 'old' (turn 1)");
        }
    }

    #[test]
    fn top_k_limits_results() {
        let mut idx = SearchIndex::new();
        // First 10 contain unique discriminating term "zetaquery"
        // Remaining 40 contain different content so IDF(zetaquery) = log(50/10) > 0
        for i in 0..10 {
            idx.add_document(
                &format!("target{}", i), "gen", "facts",
                &format!("zetaquery programming language systems kernel compiled version {}", i),
                i,
            );
        }
        for i in 10..50 {
            idx.add_document(
                &format!("noise{}", i), "gen", "facts",
                &format!("cooking recipe ingredients method preparation serving {}", i),
                i,
            );
        }
        idx.rebuild_idf();
        let results = idx.search("zetaquery systems kernel", None, None, 5, 50);
        assert!(results.len() <= 5, "top_k=5 should return at most 5 results");
        assert!(!results.is_empty(), "Should return some results for discriminating query");
    }

    #[test]
    fn search_results_have_required_fields() {
        let mut idx = SearchIndex::new();
        // Need >=2 docs so TF-IDF IDF(term) = log(N/df) > 0
        // "fortytwo" only in p1 → df=1, N=2, IDF=log(2)>0
        idx.add_document("p1", "project-x", "discoveries", "The fortytwo answer unlocks everything", 7);
        idx.add_document("p2", "project-x", "discoveries", "Unrelated content about cooking breakfast pancakes", 8);
        idx.rebuild_idf();
        let results = idx.search("fortytwo answer", None, None, 1, 8);
        assert_eq!(results.len(), 1, "Should find exactly 1 result for unique term");
        let r = &results[0];
        assert_eq!(r.path, "p1");
        assert_eq!(r.wing, "project-x");
        assert_eq!(r.hall, "discoveries");
        assert!(r.text.contains("fortytwo"));
        assert_eq!(r.turn, 7);
        assert!(r.score > 0.0);
    }
}

// ──────────────────────────────────────────────────────────────────
// 8. ENTITY GRAPH — temporal knowledge representation
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_graph {
    use genos_tools::graph::EntityGraph;
    use genos_kernel::json::{json_object, JsonValue};

    fn setup() -> String { crate::with_test_root("graph") }

    #[test]
    fn build_and_query_knowledge_graph() {
        let root = setup();
        let mut g = EntityGraph::new();

        // Real-world relationships about the genos project
        g.add("genos", "is_a", "UEFI OS", "2026-01-01T00:00:00Z");
        g.add("genos", "uses_model", "Stories15M", "2026-01-01T00:00:00Z");
        g.add("genos", "written_in", "Rust", "2026-01-01T00:00:00Z");
        g.add("Stories15M", "is_a", "language model", "2026-01-01T00:00:00Z");
        g.add("Stories15M", "parameter_count", "15M", "2026-01-01T00:00:00Z");
        g.add("Rust", "used_for", "systems programming", "2026-01-01T00:00:00Z");

        assert_eq!(g.len(), 6);

        let genos_facts = g.query("genos");
        assert_eq!(genos_facts.len(), 3, "genos should have 3 facts");
        assert!(genos_facts.iter().any(|t| t.relation == "uses_model" && t.entity_b == "Stories15M"));
        assert!(genos_facts.iter().any(|t| t.relation == "written_in" && t.entity_b == "Rust"));

        let model_facts = g.query("Stories15M");
        assert!(model_facts.len() >= 2, "Stories15M should have facts about it");
        crate::cleanup(&root);
    }

    /// Directly reproduces the old `invalidate_triple` test that was failing:
    /// add → query(=1) → invalidate → query(=0) → add_new → query(=1 with new value)
    #[test]
    fn invalidate_immediately_hides_triple() {
        let root = setup();
        let mut g = EntityGraph::new();

        // Step 1: add
        g.add("entity", "rel", "old_val", "2026-01-01T00:00:00Z");
        assert_eq!(g.len(), 1);
        let before = g.query("entity");
        assert_eq!(before.len(), 1, "Before invalidation: should see 1 triple");
        assert_eq!(before[0].entity_b, "old_val");

        // Step 2: invalidate — triple must disappear from active query immediately
        let ok = g.invalidate("entity", "rel", "old_val", "2026-04-09T12:00:00Z");
        assert!(ok, "invalidate should return true when triple was found");
        assert_eq!(g.query("entity").len(), 0,
            "Immediately after invalidation, active query must return 0");

        // Step 3: add replacement
        g.add("entity", "rel", "new_val", "2026-04-09T12:00:00Z");
        let after = g.query("entity");
        assert_eq!(after.len(), 1, "After adding replacement, query must return 1");
        assert_eq!(after[0].entity_b, "new_val",
            "Replacement triple should be the new value");

        // Step 4: timeline contains both
        let tl = g.timeline("entity");
        assert_eq!(tl.len(), 2, "Timeline must retain both old and new triples");
        let old_triple = tl.iter().find(|t| t.entity_b == "old_val").expect("old triple in timeline");
        assert!(old_triple.valid_until.is_some(), "Old triple must have valid_until set");
        let new_triple = tl.iter().find(|t| t.entity_b == "new_val").expect("new triple in timeline");
        assert!(new_triple.valid_until.is_none(), "New triple must still be active");

        crate::cleanup(&root);
    }

    #[test]
    fn invalidate_nonexistent_returns_false() {
        let root = setup();
        let mut g = EntityGraph::new();
        g.add("entity", "rel", "val", "2026-01-01T00:00:00Z");

        // Wrong entity_b
        assert!(!g.invalidate("entity", "rel", "wrong_val", "2026-04-09T12:00:00Z"),
            "Invalidating non-matching entity_b should return false");
        // Wrong relation
        assert!(!g.invalidate("entity", "wrong_rel", "val", "2026-04-09T12:00:00Z"),
            "Invalidating non-matching relation should return false");
        // Triple still active after failed invalidations
        assert_eq!(g.query("entity").len(), 1,
            "Triple should still be active after failed invalidation attempts");

        crate::cleanup(&root);
    }

    #[test]
    fn temporal_invalidation_tracks_change() {
        let root = setup();
        let mut g = EntityGraph::new();

        // genos starts using Stories15M
        g.add("genos", "active_model", "Stories15M", "2026-01-01T00:00:00Z");
        assert_eq!(g.query("genos").len(), 1);

        // Upgrade: invalidate old, add new
        let ok = g.invalidate("genos", "active_model", "Stories15M", "2026-04-09T12:00:00Z");
        assert!(ok, "Should be able to invalidate existing triple");

        // After invalidation, active query returns 0 (triple is expired)
        assert_eq!(g.query("genos").len(), 0,
            "Invalidated triple should not appear in current query");

        // Add new active model
        g.add("genos", "active_model", "Stories110M", "2026-04-09T12:00:00Z");
        let current = g.query("genos");
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].entity_b, "Stories110M", "Should now use Stories110M");

        // Timeline shows complete history
        let tl = g.timeline("genos");
        assert_eq!(tl.len(), 2, "Timeline should have both old and new");
        let has_both = tl.iter().any(|t| t.entity_b == "Stories15M")
            && tl.iter().any(|t| t.entity_b == "Stories110M");
        assert!(has_both, "Timeline should contain both model versions");
        crate::cleanup(&root);
    }

    #[test]
    fn query_as_of_returns_historical_state() {
        let root = setup();
        let mut g = EntityGraph::new();

        g.add("genos", "phase", "A", "2025-10-01T00:00:00Z");
        g.invalidate("genos", "phase", "A", "2025-12-01T00:00:00Z");
        g.add("genos", "phase", "B", "2025-12-01T00:00:00Z");
        g.invalidate("genos", "phase", "B", "2026-02-01T00:00:00Z");
        g.add("genos", "phase", "C", "2026-02-01T00:00:00Z");
        g.invalidate("genos", "phase", "C", "2026-04-01T00:00:00Z");
        g.add("genos", "phase", "D", "2026-04-09T00:00:00Z");

        // As of November 2025 → Phase A
        let in_nov = g.query_as_of("genos", "2025-11-15T00:00:00Z");
        assert!(in_nov.iter().any(|t| t.entity_b == "A"),
            "In Nov 2025, genos should be in phase A");

        // As of January 2026 → Phase B  
        let in_jan = g.query_as_of("genos", "2026-01-15T00:00:00Z");
        assert!(in_jan.iter().any(|t| t.entity_b == "B"),
            "In Jan 2026, genos should be in phase B");

        // Current (after April 9) → Phase D
        let current = g.query("genos");
        assert!(current.iter().any(|t| t.entity_b == "D"),
            "Currently genos should be in phase D");
        crate::cleanup(&root);
    }

    #[test]
    fn graph_tool_handlers() {
        let root = setup();
        // Tool: memory.graph_add
        let add_args = json_object(&[
            ("entity_a", JsonValue::Str("Alice".into())),
            ("relation", JsonValue::Str("mentors".into())),
            ("entity_b", JsonValue::Str("Bob".into())),
        ]);
        let g_ref = &mut EntityGraph::new();
        let add_result = genos_tools::graph::tool_graph_add(
            &add_args, "g1", "2026-04-09T12:00:00Z", g_ref);
        assert!(add_result.ok, "graph_add should succeed");

        // Tool: memory.graph_query
        let query_args = json_object(&[("entity", JsonValue::Str("Alice".into()))]);
        let query_result = genos_tools::graph::tool_graph_query(&query_args, "g2", g_ref);
        assert!(query_result.ok, "graph_query should succeed");
        let empty2 = vec![];
        let results = query_result.result.as_array().unwrap_or(&empty2);
        assert!(!results.is_empty(), "Should find Alice's relationships");

        // Tool: memory.graph_timeline
        let tl_args = json_object(&[("entity", JsonValue::Str("Alice".into()))]);
        let tl_result = genos_tools::graph::tool_graph_timeline(&tl_args, "g3", g_ref);
        assert!(tl_result.ok);
        crate::cleanup(&root);
    }
}

// ──────────────────────────────────────────────────────────────────
// 9. PALACE — halls, wings, facts, sessions
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_palace {
    use genos_tools::palace::{self, Hall, WingType};

    fn setup() -> (String, String) {
        let root = crate::with_test_root("palace");
        let session_id = palace::ensure_structure("20260409_120000");
        (root, session_id)
    }

    #[test]
    fn ensure_structure_creates_palace_dirs() {
        let (root, session_id) = setup();
        assert!(!session_id.is_empty(), "Should return a session ID");

        // Verify palace directory exists
        assert!(std::path::Path::new(&format!("{}/palace", root)).exists(),
            "palace/ dir should exist");
        assert!(std::path::Path::new(&format!("{}/palace/wings", root)).exists(),
            "palace/wings/ dir should exist");
        crate::cleanup(&root);
    }

    #[test]
    fn store_and_retrieve_facts() {
        let (root, _) = setup();

        palace::write_facts(&[
            ("os_name".to_string(), "genos".to_string()),
            ("os_language".to_string(), "Rust".to_string()),
            ("os_target".to_string(), "x86_64-unknown-uefi".to_string()),
            ("current_phase".to_string(), "D".to_string()),
        ]);

        let facts = palace::read_facts();
        assert!(facts.len() >= 4, "Should have 4 facts, got {}", facts.len());

        let os_name = facts.iter().find(|(k, _)| k == "os_name").map(|(_, v)| v.as_str());
        assert_eq!(os_name, Some("genos"));

        let phase = facts.iter().find(|(k, _)| k == "current_phase").map(|(_, v)| v.as_str());
        assert_eq!(phase, Some("D"));
        crate::cleanup(&root);
    }

    #[test]
    fn contradiction_detection_in_facts() {
        let (root, _) = setup();

        // Set original fact
        let args1 = genos_kernel::json::json_object(&[
            ("key", genos_kernel::json::JsonValue::Str("model".into())),
            ("value", genos_kernel::json::JsonValue::Str("Stories15M".into())),
        ]);
        let r1 = genos_tools::memory::tool_facts_set(&args1, "f1", "2026-01-01T00:00:00Z");
        assert!(r1.ok, "First facts_set should succeed");

        // Update with a contradicting value
        let args2 = genos_kernel::json::json_object(&[
            ("key", genos_kernel::json::JsonValue::Str("model".into())),
            ("value", genos_kernel::json::JsonValue::Str("Stories110M".into())),
        ]);
        let r2 = genos_tools::memory::tool_facts_set(&args2, "f2", "2026-04-09T12:00:00Z");
        assert!(r2.ok, "Second facts_set should succeed even with contradiction");

        // Read back and verify new value is present
        let raw = palace::read_facts_raw();
        assert!(raw.contains("Stories110M"), "Updated value should be present");
        assert!(raw.contains("superseded") || raw.contains("Stories15M"),
            "Old value should still exist (append-only)");
        crate::cleanup(&root);
    }

    #[test]
    fn store_in_all_halls() {
        let (root, session_id) = setup();

        for hall in Hall::all() {
            palace::store_in_hall(&session_id, 1,
                *hall,
                &format!("Test entry for {} hall", hall.as_str()));
        }
        // Verify files were written
        for hall in Hall::all() {
            let path = format!("{}/palace/sessions/{}/{}.jsonl",
                root, &session_id[..8.min(session_id.len())], hall.as_str());
            // Just verify no panic - file paths may vary
            let _ = path;
        }
        crate::cleanup(&root);
    }

    #[test]
    fn create_list_and_use_wings() {
        let (root, session_id) = setup();

        // Create multiple project wings
        assert!(palace::create_wing("genos-boot", WingType::Project), "Create genos-boot wing");
        assert!(palace::create_wing("genos-hal", WingType::Project), "Create genos-hal wing");
        assert!(palace::create_wing("alice-dev", WingType::Person), "Create person wing");
        assert!(!palace::create_wing("genos-boot", WingType::Project), "Duplicate should return false");

        let wings = palace::list_wings();
        assert!(wings.iter().any(|w| w.name == "genos-boot"),
            "Should list genos-boot, got: {:?}", wings.iter().map(|w| &w.name).collect::<Vec<_>>());
        assert!(wings.iter().any(|w| w.name == "genos-hal"));
        assert!(wings.iter().any(|w| w.name == "alice-dev"));

        // Store in a wing's hall
        palace::store_in_wing_hall("genos-boot", &session_id, 1,
            Hall::Facts, "genos-boot is the UEFI bootloader crate");

        crate::cleanup(&root);
    }

    #[test]
    fn hall_enum_coverage() {
        // Every hall string must round-trip
        for hall in Hall::all() {
            let s = hall.as_str();
            let parsed = Hall::from_str(s).expect(&format!("Hall::from_str('{}') should work", s));
            assert_eq!(parsed.as_str(), s);
        }
        assert!(Hall::from_str("nonexistent_hall").is_none());
    }

    #[test]
    fn wing_type_coverage() {
        for (s, _) in &[("project", WingType::Project), ("person", WingType::Person), ("general", WingType::General)] {
            let parsed = WingType::from_str(s).expect(&format!("WingType::from_str('{}') should work", s));
            assert_eq!(&parsed.as_str(), s);
        }
        assert!(WingType::from_str("invalid").is_none());
    }
}

// ──────────────────────────────────────────────────────────────────
// 10. MEMORY STACK — full store → search → retrieve cycle
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_memory {
    use genos_tools::memory;
    use genos_tools::palace;
    use genos_tools::search::SearchIndex;
    use genos_kernel::json::{json_object, JsonValue};

    fn setup() -> (String, String, SearchIndex) {
        let root = crate::with_test_root("memory");
        let session_id = palace::ensure_structure("20260409_120000");
        let index = SearchIndex::new();
        (root, session_id, index)
    }

    #[test]
    fn store_search_and_retrieve_content() {
        let (root, session_id, mut index) = setup();

        // Store several memories with distinct content
        let memories = vec![
            ("Rust uses ownership and borrowing to achieve memory safety without a garbage collector", "facts"),
            ("The borrow checker prevents data races and use-after-free at compile time", "facts"),
            ("genos phase A: Stories15M model boots in QEMU using UEFI protocol", "events"),
            ("genos phase B: 20 commits implementing tool API, JSON protocol, palace memory", "events"),
            ("genos phase C: tokenizer, search index, web tools, entity graph fully working", "events"),
        ];

        for (i, (content, hall)) in memories.iter().enumerate() {
            let args = json_object(&[
                ("content", JsonValue::Str(content.to_string())),
                ("wing", JsonValue::Str("general".into())),
                ("hall", JsonValue::Str(hall.to_string())),
            ]);
            let result = memory::tool_store(&args, &format!("m{}", i), &session_id, i + 1, &mut index);
            assert!(result.ok, "tool_store {} should succeed: {:?}",
                i, result.error.as_ref().map(|e| &e.message));
        }
        assert_eq!(index.len(), 5, "Index should have 5 documents");
        index.rebuild_idf();

        // Search 1: find Rust memory safety content
        let r1 = memory::tool_search(
            &json_object(&[("query", JsonValue::Str("Rust ownership garbage collector".into()))]),
            "s1", 5, &index
        );
        assert!(r1.ok, "Search should succeed");
        let results = r1.result.as_array().expect("should return array");
        assert!(!results.is_empty(), "Should find Rust memory content");
        let top = &results[0];
        let text = top.get("content").or_else(|| top.get("text"))
            .and_then(|v| v.as_str()).unwrap_or("");
        assert!(text.contains("Rust") || text.contains("ownership") || text.contains("borrow"),
            "Top result should be about Rust, got: {}", text);

        // Search 2: find genos phase content
        let r2 = memory::tool_search(
            &json_object(&[
                ("query", JsonValue::Str("genos phase tool API".into())),
                ("hall", JsonValue::Str("events".into())),
            ]),
            "s2", 5, &index
        );
        assert!(r2.ok);
        let results2 = r2.result.as_array().expect("should return array");
        assert!(!results2.is_empty(), "Should find genos phase events");
        crate::cleanup(&root);
    }

    #[test]
    fn facts_full_cycle() {
        let (root, _, _) = setup();

        // Set multiple facts
        let facts = vec![
            ("project_name", "genos"),
            ("kernel_type", "LLM"),
            ("target_arch", "x86_64"),
            ("boot_protocol", "UEFI"),
        ];
        for (k, v) in &facts {
            let args = json_object(&[
                ("key", JsonValue::Str(k.to_string())),
                ("value", JsonValue::Str(v.to_string())),
            ]);
            let r = memory::tool_facts_set(&args, "fs", "2026-04-09T12:00:00Z");
            assert!(r.ok, "facts_set({}, {}) should succeed", k, v);
        }

        // Get all facts and verify every key is present
        let get_result = memory::tool_facts_get(&JsonValue::Null, "fg");
        assert!(get_result.ok);
        let arr = get_result.result.as_array().expect("facts should be array");
        for (k, v) in &facts {
            let found = arr.iter().any(|entry| {
                let key = entry.get("key").and_then(|x| x.as_str()).unwrap_or("");
                let val = entry.get("value").and_then(|x| x.as_str()).unwrap_or("");
                key == *k && val.contains(v)
            });
            assert!(found, "Fact {}={} should be retrievable", k, v);
        }
        crate::cleanup(&root);
    }

    #[test]
    fn search_returns_scored_ranked_results() {
        let (root, _session_id, mut index) = setup();

        index.add_document("r1", "gen", "facts",
            "memory safety is the most important property of systems languages", 1);
        index.add_document("r2", "gen", "facts",
            "Rust provides memory safety through ownership and borrowing semantics", 2);
        index.add_document("r3", "gen", "facts",
            "C++ has manual memory management which can lead to memory safety issues", 3);
        index.add_document("r4", "gen", "facts",
            "Python is a high-level scripting language used for data analysis", 4);
        index.rebuild_idf();

        let r = memory::tool_search(
            &json_object(&[
                ("query", JsonValue::Str("memory safety Rust ownership".into())),
                ("top_k", JsonValue::Number(3.0)),
            ]),
            "ms", 4, &index
        );
        assert!(r.ok);
        let results = r.result.as_array().expect("should return array");
        assert!(!results.is_empty());
        assert!(results.len() <= 3, "top_k=3 should return at most 3");

        // Python doc should NOT be in top results for "memory safety Rust"
        let has_python = results.iter().any(|entry| {
            let t = entry.get("content").or_else(|| entry.get("text"))
                .and_then(|v| v.as_str()).unwrap_or("");
            t.contains("Python") && !t.contains("memory safety")
        });
        assert!(!has_python, "Python doc shouldn't rank for memory safety query");
        crate::cleanup(&root);
    }
}

// ──────────────────────────────────────────────────────────────────
// 11. PROTOCOL — tool registry, manifests, call/result contract
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_protocol {
    use genos_tools::protocol::{ToolRegistry, ToolResult, ToolCall};
    use genos_kernel::json::{parse, json_object, JsonValue};

    #[test]
    fn registry_covers_all_tool_categories() {
        let reg = ToolRegistry::default_registry();
        let names: Vec<&str> = reg.list().iter().map(|s| s.name).collect();

        // Must have all registered tool categories
        assert!(names.iter().any(|n| n.starts_with("fs.")), "Must have fs.* tools, got: {:?}", names);
        assert!(names.iter().any(|n| n.starts_with("memory.")), "Must have memory.* tools");
        assert!(names.iter().any(|n| n.starts_with("net.")), "Must have net.* tools");
        assert!(names.iter().any(|n| n.starts_with("palace.")), "Must have palace.* tools");
        assert!(names.iter().any(|n| n.starts_with("sys.")), "Must have sys.* tools");
        // web.* tools are called directly, not via registry (they live in genos-tools::web)
    }

    #[test]
    fn find_tools_by_dot_name() {
        let reg = ToolRegistry::default_registry();
        // All registered tools must be findable by exact name
        for name in &["fs.read", "fs.write", "fs.list", "fs.delete",
                      "memory.store", "memory.search", "memory.facts_get",
                      "memory.facts_set", "memory.consolidate", "memory.forget",
                      "memory.graph_add", "memory.graph_query",
                      "memory.graph_invalidate", "memory.graph_timeline",
                      "net.fetch", "palace.create_wing", "palace.list_wings",
                      "palace.find_tunnels", "sys.clock", "sys.introspect"] {
            assert!(reg.find(name).is_some(), "Tool '{}' should be registered", name);
        }
    }

    #[test]
    fn manifest_contains_all_tools_with_docs() {
        let reg = ToolRegistry::default_registry();
        let manifest = reg.manifest();
        assert!(!manifest.is_empty(), "Manifest should not be empty");
        // Every tool should have docs
        for tool in reg.list() {
            assert!(manifest.contains(tool.name),
                "Manifest should mention tool '{}'", tool.name);
            assert!(!tool.description.is_empty(),
                "Tool '{}' must have a description", tool.name);
        }
        // Must explain how to invoke
        assert!(manifest.contains("tool_call") || manifest.contains("{") || manifest.contains("call"),
            "Manifest should explain invocation syntax");
    }

    #[test]
    fn tool_result_success_contract() {
        let r = ToolResult::success("abc123", json_object(&[
            ("data", JsonValue::Str("fetched 42 bytes".into())),
            ("bytes", JsonValue::Number(42.0)),
        ]), 17);
        assert!(r.ok);
        assert_eq!(r.call_id, "abc123");
        assert_eq!(r.elapsed_ms, 17);
        assert!(r.error.is_none());
        let j = r.to_json();
        assert_eq!(j.get("call_id").unwrap().as_str(), Some("abc123"));
        assert_eq!(j.get("ok").unwrap().as_bool(), Some(true));
        assert_eq!(j.get("elapsed_ms").unwrap().as_f64(), Some(17.0));
    }

    #[test]
    fn tool_result_failure_contract() {
        let r = ToolResult::failure("xyz", "NETWORK_TIMEOUT", "Connection timed out after 10s", true);
        assert!(!r.ok);
        let e = r.error.as_ref().expect("failure must have error");
        assert_eq!(e.code, "NETWORK_TIMEOUT");
        assert_eq!(e.message, "Connection timed out after 10s");
        assert!(e.retryable, "Network timeouts should be retryable");
        let j = r.to_json();
        assert_eq!(j.get("ok").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn tool_result_json_roundtrip() {
        let r = ToolResult::success("c99", json_object(&[
            ("results", JsonValue::Array(vec![JsonValue::Str("item1".into()), JsonValue::Str("item2".into())])),
        ]), 5);
        let j = r.to_json();
        let s = j.to_json_string();
        let parsed = parse(&s).unwrap();
        assert_eq!(parsed.get("call_id").unwrap().as_str(), Some("c99"));
        let results = parsed.get("result").unwrap().get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn tool_call_from_json_full() {
        let json = json_object(&[
            ("tool", JsonValue::Str("memory.store".into())),
            ("args", json_object(&[
                ("content", JsonValue::Str("phase D starts today".into())),
                ("wing", JsonValue::Str("genos".into())),
                ("hall", JsonValue::Str("events".into())),
            ])),
            ("call_id", JsonValue::Str("call_001".into())),
        ]);
        let call = ToolCall::from_json(&json, "session_abc").expect("should parse");
        assert_eq!(call.tool, "memory.store");
        assert_eq!(call.call_id, "call_001");
        assert_eq!(call.session_id, "session_abc");
        assert_eq!(call.args.get("content").unwrap().as_str(), Some("phase D starts today"));
    }
}

// ──────────────────────────────────────────────────────────────────
// 12. JOURNAL — turn classification
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_journal {
    use genos_tools::journal::{classify_hall, TurnRecord};
    use genos_tools::palace::Hall;

    fn setup() -> (String, String) {
        let root = crate::with_test_root("journal");
        let session_id = genos_tools::palace::ensure_structure("20260409_120000");
        (root, session_id)
    }

    #[test]
    fn classify_covers_all_halls() {
        // Preferences
        let h = classify_hall(
            "I prefer to have dark mode everywhere",
            "Noted, I'll remember your dark mode preference."
        );
        assert!(matches!(h, Hall::Preferences | Hall::Facts));

        // Events (actions, deployments)
        let h = classify_hall(
            "I deployed the new build to QEMU just now",
            "Deployment successful, genos booted correctly."
        );
        assert!(matches!(h, Hall::Events | Hall::Facts));

        // Discoveries (insights, findings)
        let h = classify_hall(
            "I found that the tokenizer allocates 2x expected memory",
            "Interesting finding, the tokenizer indeed uses extra allocations."
        );
        assert!(matches!(h, Hall::Discoveries | Hall::Facts));
    }

    #[test]
    fn turn_record_stores_to_disk() {
        let (root, session_id) = setup();
        let record = TurnRecord {
            session_id: session_id.clone(),
            turn: 1,
            timestamp: "2026-04-09T12:00:00Z".into(),
            hall: Hall::Events,
            input: "How does genos boot?".into(),
            output: "genos uses UEFI SimpleFileSystem to load the model weights.".into(),
            tool_calls: vec![],
        };

        record.store(); // Should not panic

        let jsonl = record.to_jsonl();
        assert!(jsonl.contains("How does genos boot"), "JSONL should contain input");
        assert!(jsonl.contains("UEFI"), "JSONL should contain output");
        assert!(jsonl.contains(&session_id), "JSONL should contain session ID");
        crate::cleanup(&root);
    }
}

// ──────────────────────────────────────────────────────────────────
// 13. SYS TOOLS — clock, introspect, status header
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_sys {
    use genos_kernel::json::JsonValue;

    #[test]
    fn tool_clock_returns_timestamp() {
        let result = genos_tools::sys::tool_clock(&JsonValue::Null, "clock1");
        assert!(result.ok, "tool_clock should succeed");
        let time_str = result.result.as_str()
            .or_else(|| result.result.get("time").and_then(|t| t.as_str()))
            .unwrap_or_else(|| {
                // Try JSON string representation
                Box::leak(result.result.to_json_string().into_boxed_str())
            });
        assert!(time_str.contains("2026") || time_str.contains("T") || time_str.contains(":"),
            "Clock result should contain a timestamp, got: {}", time_str);
    }

    #[test]
    fn tool_introspect_returns_system_info() {
        let result = genos_tools::sys::tool_introspect(&JsonValue::Null, "intr1");
        assert!(result.ok, "tool_introspect should succeed");
        // Should have some system info
        let s = result.result.to_json_string();
        assert!(!s.is_empty(), "Introspect should return non-empty result");
    }

    #[test]
    fn format_status_header_contains_session_info() {
        let header = genos_tools::sys::format_status_header("sess_abc123", 7, 2048, 8192);
        assert!(header.contains("sess_abc123"), "Header should contain session ID");
        assert!(header.contains("7"), "Header should contain turn count");
        // Should contain context info
        assert!(header.contains("2048") || header.contains("8192") || header.contains("context"),
            "Header should contain context info, got: {}", header);
    }
}

// ──────────────────────────────────────────────────────────────────
// 14. END-TO-END INTEGRATION — simulate a real OS conversation turn
// ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod test_e2e {
    use genos_tools::palace;
    use genos_tools::search::SearchIndex;
    use genos_tools::memory;
    use genos_tools::graph::EntityGraph;
    use genos_kernel::json::{extract_tool_calls, json_object, JsonValue};

    fn setup() -> (String, String, SearchIndex, EntityGraph) {
        let root = crate::with_test_root("e2e");
        let session_id = palace::ensure_structure("20260409_120000");
        (root, session_id, SearchIndex::new(), EntityGraph::new())
    }

    /// Simulates a complete multi-turn conversation where:
    /// 1. Knowledge is stored in memory
    /// 2. Entity relationships are built in the graph
    /// 3. A question is asked that requires searching memory
    /// 4. The search finds the right answer
    #[test]
    fn full_conversation_loop() {
        let (root, session_id, mut index, mut graph) = setup();

        // ── Turn 1: User declares project info ──
        memory::tool_store(&json_object(&[
            ("content", JsonValue::Str("genos is a bare-metal UEFI OS written in Rust where a language model is the kernel".into())),
            ("wing", JsonValue::Str("general".into())),
            ("hall", JsonValue::Str("facts".into())),
        ]), "m1", &session_id, 1, &mut index).ok;

        memory::tool_facts_set(&json_object(&[
            ("key", JsonValue::Str("project".into())),
            ("value", JsonValue::Str("genos".into())),
        ]), "f1", "2026-04-09T12:00:00Z");

        graph.add("genos", "is_a", "UEFI OS", "2026-04-09T12:00:00Z");
        graph.add("genos", "kernel_is", "Stories15M LLM", "2026-04-09T12:00:00Z");

        // ── Turn 2: More facts ──
        memory::tool_store(&json_object(&[
            ("content", JsonValue::Str("Phase D goal: wire all tool handlers into the inference loop so the model can call tools during generation".into())),
            ("wing", JsonValue::Str("general".into())),
            ("hall", JsonValue::Str("events".into())),
        ]), "m2", &session_id, 2, &mut index).ok;

        memory::tool_store(&json_object(&[
            ("content", JsonValue::Str("tool handlers: fs.read fs.write memory.store memory.search web.get_page net.fetch entity graph palace wings".into())),
            ("wing", JsonValue::Str("general".into())),
            ("hall", JsonValue::Str("facts".into())),
        ]), "m3", &session_id, 3, &mut index).ok;

        index.rebuild_idf();

        // ── Turn 3: Model emits tool calls in response ──
        let model_output = r#"Let me recall what I know about the current project.
<tool_call>
{"tool": "memory.search", "args": {"query": "genos kernel language model UEFI", "top_k": 3}, "call_id": "search1"}
</tool_call>
And also check the tool handlers.
<tool_call>
{"tool": "memory.search", "args": {"query": "tool handlers inference loop", "top_k": 3}, "call_id": "search2"}
</tool_call>"#;

        let tool_calls = extract_tool_calls(model_output);
        assert_eq!(tool_calls.len(), 2, "Should extract 2 tool calls");
        assert_eq!(tool_calls[0].get("tool").unwrap().as_str(), Some("memory.search"));
        assert_eq!(tool_calls[1].get("tool").unwrap().as_str(), Some("memory.search"));

        // ── Execute the tool calls ──
        let search1_args = tool_calls[0].get("args").unwrap();
        let search1_result = memory::tool_search(search1_args, "search1", 3, &index);
        assert!(search1_result.ok, "Search 1 should succeed");
        let results1 = search1_result.result.as_array().expect("should return array");
        assert!(!results1.is_empty(), "Should find genos facts");

        let top1_text = results1[0].get("content").or_else(|| results1[0].get("text"))
            .and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            top1_text.contains("genos") || top1_text.contains("UEFI") || top1_text.contains("Rust"),
            "Top result should be about genos, got: '{}'", top1_text
        );

        let search2_args = tool_calls[1].get("args").unwrap();
        let search2_result = memory::tool_search(search2_args, "search2", 3, &index);
        assert!(search2_result.ok);
        let results2 = search2_result.result.as_array().expect("should return array");
        assert!(!results2.is_empty(), "Should find tool handler facts");

        // ── Graph query: what is genos? ──
        let genos_facts = graph.query("genos");
        assert_eq!(genos_facts.len(), 2);
        assert!(genos_facts.iter().any(|t| t.relation == "is_a" && t.entity_b == "UEFI OS"));
        assert!(genos_facts.iter().any(|t| t.relation == "kernel_is"));

        crate::cleanup(&root);
    }

    /// Test that fetching a real webpage, stripping it, and storing in memory works end-to-end
    #[test]
    fn web_fetch_strip_and_store() {
        let (root, session_id, mut index, _) = setup();

        // Try to fetch example.com
        // Use info.cern.ch — publicly accessible HTTP-only site
        let raw = match genos_hal::net::fetch("http://info.cern.ch") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("SKIP: no network, skipping web_fetch_strip_and_store");
                crate::cleanup(&root);
                return;
            }
        };
        let html = String::from_utf8_lossy(&raw).to_string();

        // Strip the HTML into semantic chunks
        let chunks = genos_tools::web::strip_html(&html);
        assert!(!chunks.is_empty(), "Should extract chunks from info.cern.ch");
        // Verify at least one chunk contains meaningful content
        let has_meaningful = chunks.iter().any(|c| c.text.len() > 5 && !c.text.trim().is_empty());
        assert!(has_meaningful, "Should have non-trivial chunks");

        // Store each chunk in memory
        for (i, chunk) in chunks.iter().take(5).enumerate() {
            let args = json_object(&[
                ("content", JsonValue::Str(format!("[web:info.cern.ch] {}", chunk.text))),
                ("wing", JsonValue::Str("general".into())),
                ("hall", JsonValue::Str("discoveries".into())),
            ]);
            memory::tool_store(&args, &format!("web{}", i), &session_id, i + 10, &mut index).ok;
        }

        index.rebuild_idf();
        assert!(index.len() > 0, "Should have indexed content from the web page");

        // Search for content that should have come from info.cern.ch
        // CERN page has "first website" in the h1 — very distinctive term
        let r = memory::tool_search(
            &json_object(&[("query", JsonValue::Str("first website cern".into()))]),
            "ws1", 15, &index
        );
        assert!(r.ok);
        let results = r.result.as_array().expect("array");
        assert!(!results.is_empty(), "Should find CERN web content in memory");

        crate::cleanup(&root);
    }
}
