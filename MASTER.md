# genos — Bare-Metal LLM Operating System
## Complete Master Specification

**The LLM *is* the OS.** Boot directly from USB via UEFI (no Linux, Windows, or BSD ever), load a quantized instruction model, and expose a TUI shell where the LLM makes every decision through structured tool calls, persistent memory, and agentic reasoning.

**Repo**: https://github.com/n33levo/genos  
**Status**: Phase A+B+C+D complete, E planned  
**Stack**: Rust nightly · `x86_64-unknown-uefi` · `#![no_std]` + `core` + `alloc` · `uefi = "0.37"` · `libm = "0.2"`

---

## Table of Contents

1. [What You're Building](#what-youre-building)
2. [Rust Tech Stack](#rust-tech-stack)
3. [Architecture & Components](#architecture--components)
4. [Complete Phase Roadmap](#complete-phase-roadmap)
5. [Detailed Subsystems](#detailed-subsystems)
6. [Master Prompt for Copilot/Claude](#master-prompt-for-copilotclaude)
7. [Key Decisions & Rationale](#key-decisions--rationale)
8. [Risks & Mitigations](#risks--mitigations)

---

## What You're Building

**genos** is not a firmware demo. It's a self-contained **bare-metal LLM living environment**:

- Boots directly from UEFI USB/SSD (no operating system underneath)
- Uses a TurboQuant-compressed Gemma-class model (Gemma 4 E2B: 2.3B effective / 5.1B total params) as the **core decision engine**
- Provides a tool API for:
  - Web browsing with `llms.txt` / `llm.txt` awareness
  - Persistent memory with entity graph tracking and consolidation
  - File I/O and journaling
  - Auto-research and experiment execution
  - Self-directed tool building at runtime
- Exposes a **TUI shell** where users can:
  - Chat with an instruction-following model
  - Manage long-term memory and research projects
  - Monitor system resources (tokens/sec, RAM, uptime, context usage)
  - Execute background tasks cooperatively

Think of it as "a personal, self-directed AI assistant that lives entirely on your hardware with no cloud dependency."

---

## Rust Tech Stack

### Language & Targets

- **Rust nightly** (stable where possible) with:
  - `#![no_std]` for UEFI bare-metal target
  - `core` and `alloc` crates only
  - Global heap allocator via `linked_list_allocator` or custom buddy allocator
- **Primary target**: `x86_64-unknown-uefi`
- **Future targets**: ARM UEFI, ARM SBC bootloaders (architecture deferred to Phase D+)

### Core Dependencies

| Crate | Purpose | Constraint |
|-------|---------|-----------|
| `uefi = "0.37"` | UEFI firmware bindings (screen, disk, network, timers) | Must support `#![no_std]` |
| `libm = "0.2"` | Math functions (exp, sqrt, sin, cos, log) for no_std | No hardware FPU assumption |
| `linked_list_allocator` | Global heap allocator | Minimal, no_std compatible |
| `tokenizers` or `rust-tokenizers` | BPE/Unigram tokenization | Phase C: on-device tokenizer for Gemma 4 (262k vocab) |

### Optional Future Additions

| Crate | Purpose | Phase |
|-------|---------|-------|
| `rustls` or `bearssl-rs` | HTTPS support | Phase D (45KB footprint for BearSSL acceptable) |
| `sqlite` (vendored, no_std adapter) | Temporal entity graph storage | Phase C (memory subsystem) |
| `lua54` (no_std Rust port) | Code execution sandbox | Phase E (tool builder) |

### Build & Dev Tooling

- **QEMU 10.x+** with **OVMF** firmware (x86_64) for testing bounds
- **Makefile** with targets: `build`, `esp` (FAT32 image), `qemu`, `qemu-debug`, `test`, `flash`
- **GDB/LLDB** for remote kernel debugging
- **Cargo** with `.cargo/config.toml` enforcing target and build flags
- **Unit tests** compiled for `std` (hosted mode via `#[cfg(feature = "hosted")]`)

---

## Architecture & Components

### Workspace Structure (5 Crates)

```
genos/
├── genos-boot/          UEFI entry point, panic handler, boot sequence
├── genos-hal/           Hardware abstraction (screen, keyboard, disk, net, timers)
├── genos-kernel/        LLM runtime (model loading, inference, tokenizer, sampler)
├── genos-tools/         Tool implementations (fs, web, memory, code, research, sys)
├── genos-agent/         REPL shell, scheduler, policy engine, task coordination
├── Cargo.toml           Workspace root
├── Makefile             Build automation
└── resources/           System artifacts (prompts, configs)
```

### 1. genos-boot

**Responsibilities**:
- UEFI entry point (`#[entry]`, `#![no_std]`, `#![no_main]`)
- Panic handler (prints to UEFI text output)
- Global heap allocator initialization
- Boot sequence: load model + tokenizer, hand off to agent REPL

**Key functions**:
- `fn main() -> Status` — entry point, orchestrates boot
- Model/tokenizer loading from FAT32 disk
- Fallback to stub REPL if model load fails

### 2. genos-hal (Hardware Abstraction)

**Modules**:

```rust
// genos-hal/src/
pub mod screen;      // print(), println(), clear(), scrollback
pub mod keyboard;    // read_key(), read_line(), backspace, cursor
pub mod disk;        // read_file(), write_file(), list_dir()
pub mod net;         // fetch(url), http_get(), http_post(), dns_resolve()
pub mod timer;       // now_ms(), sleep_ms(), monotonic_ns()
pub mod alloc;       // global allocator init, memory map queries
```

**Design principle**: all platform-specific UEFI calls are confined here, enabling board/architecture porting.

### 3. genos-kernel (LLM Runtime)

**Modules**:

```rust
// genos-kernel/src/
pub mod config;        // ModelConfig struct, parsing headers
pub mod weights;       // Weights loading and offset computation
pub mod inference;     // Forward pass (RMSNorm, attention, FFN, softmax)
pub mod tokenizer;     // BPE encode/decode (llama2.c format → Gemma 4 vocab.json)
pub mod sampler;       // Top-p, top-k, temperature, repetition penalty
pub mod gguf;          // GGUF v3 parser (Phase D)
pub mod kv_cache;      // KV-cache management, QJL quantization (Phase D)
pub mod simd;          // AVX2 matmul kernels (Phase D)
```

**Core trait**:
```rust
pub trait LLMRuntime {
    fn forward(&mut self, token: u32, pos: usize) -> &[f32];   // Logits
    fn context_len(&self) -> usize;
    fn vocab_size(&self) -> usize;
}
```

### 4. genos-tools (Tools Implementation)

**Tool modules**:

```rust
// genos-tools/src/
pub mod fs;            // read(), write(), list(), delete()
pub mod net;           // fetch(), search() — wraps hal::net
pub mod web;           // get_page(), read_llms(), diff(), cache ops
pub mod memory;        // Four-layer stack: L0 identity, L1 facts, L2 semantic, L3 archive
pub mod research;      // research.start(), status(), summarize()
pub mod code;          // eval(lang, src) — Lua + expression evaluator sandbox
pub mod sys;           // introspect(), clock(), gc(), context_budget()
pub mod json;          // Tool-call JSON parsing and builder
```

**Tool call protocol**:
```json
// Model's output (tool invocation)
{"tool": "web.read", "args": {"url": "https://example.com"}, "call_id": "c1"}

// System response
{
  "call_id": "c1",
  "ok": true,
  "result": {"chunks": [...], "source_meta": {...}},
  "elapsed_ms": 412
}
```

### 5. genos-agent (OS Shell & Scheduler)

**Components**:

```rust
// genos-agent/src/
pub mod repl;          // Main loop: user input → prompt → LLM → tool dispatch
pub mod scheduler;     // Cooperative task queue, round-robin
pub mod policy;        // Safety checks (destructive ops, rate limits, context ceiling)
pub mod journal;       // Conversation logging to journal.jsonl
pub mod tools;         // Tool registry and dispatcher
```

**Main REPL loop**:
```
1. User input
2. Query memory (memory.search based on input)
3. Assemble prompt: [system + identity + relevant_memories + input]
4. Call LLM kernel
5. Parse JSON tool calls from output
6. Execute tools, append results to prompt
7. Continue until model emits "final" marker or tool calls EOS token
8. Stream final response to TUI
9. Log turn to journal.jsonl
10. On every Nth turn: trigger memory consolidation pass
```

---

## Complete Phase Roadmap

### ✅ Phase A — Bare-Metal LLM Hello World (COMPLETE)

**Status**: Stories15M forward pass boots and runs in QEMU end-to-end.

**What shipped**:
- UEFI entry point, panic handler
- Global heap allocator (`linked_list_allocator`)
- Screen HAL (UEFI SimpleTextOutput wrapping with `CString16`)
- Keyboard HAL (UEFI SimpleTextInput, backspace handling, Duration stall fix)
- Disk HAL (UEFI SimpleFileSystem, FAT32 path handling)
- Stories15M Llama 2 forward pass (288 dim, 6 layers, RMSNorm → QKV → RoPE → attention → SwiGLU FFN → classifier)
- llama2.c binary format loader (28-byte header, flat f32 weights)
- BPE tokenizer (llama2.c tokenizer.bin format with bounds checking)
- Top-p sampler (xorshift64 RNG, softmax, top-p filtering)
- REPL loop (user types prompt, model streams response tokens)
- QEMU boot verified with OVMF pflash, release build (~20x faster than debug under TCG)
- 5 commits: pflash fix, release build, tokenizer bounds check, boot progress message, generation off-by-one fix

**ESP layout (current)**:
```
/EFI/BOOT/BOOTX64.EFI     ← genos binary (467KB, release)
/models/stories15m.bin     ← Stories15M weights (llama2.c format)
/models/tokenizer.bin      ← BPE tokenizer (llama2.c format)
/resources/prompt.txt      ← system prompt stub
```

**Verification**: `make qemu` boots clean, user types "hi", model responds "hippo is a big fish. He can sw..." ✓

---

### ✅ Phase B — OS Kernel Layer (COMPLETE)

**Goal**: Tool API, JSON tool-call protocol, palace memory scaffold, OS wake-up protocol, HTTP client. *Depends on Phase A.*

**Status**: All 11 deliverables implemented across 20 commits. ~3,900 LOC across 32 files. Zero warnings on clean build.

**Core insight from research**: MemPalace (96.6% LongMemEval R@5) proves that *storing everything verbatim + structured retrieval* beats LLM summarization. Wing+room filtering alone adds 34% recall. The memory structure must be established in Phase B or Phase C builds on sand. TurboQuant proves that verbatim data compressed with QJL achieves near-zero accuracy loss — the right approach is never drop data, always compress+preserve. Both findings change Phase B significantly.

**Deliverables**:

1. **Palace directory scaffold** — `genos-tools/src/palace.rs`, runs on first boot
   - genos's memory system is organized as a **Palace** (from MemPalace): Wings → Rooms → Halls → Drawers
   - Creates `\palace\` hierarchy if it doesn't exist:
     ```
     \palace\
       identity.txt           ← L0: who is genos, who is the user (~50 tokens)
       facts.kv               ← L1: critical facts, regenerated by consolidation (~120 tokens)
       wings\
         general\             ← default wing for all non-project content
           halls\
             facts\           ← decisions made, choices locked in
             events\          ← sessions, milestones, what happened
             discoveries\     ← breakthroughs, new insights
             preferences\     ← habits, likes, user behavior
             advice\          ← recommendations and solutions
       sessions\              ← per-session verbatim journals
     ```
   - `palace::init()` runs once on first boot, writes identity template if empty
   - `palace::ensure_structure()` runs every boot, creates any missing dirs/files
   - Returns the current session ID: `session_{YYYYMMDD}_{HHmmss}`

2. **OS wake-up protocol** — `genos-agent/src/wakeup.rs`
   - Inspired by MemPalace's wake-up: load 170 tokens of world knowledge before first turn
   - On every boot, `wakeup::load() -> String`:
     1. Read `\palace\identity.txt` (L0, ~50 tokens)
     2. Read `\palace\facts.kv` (L1, ~120 tokens)
     3. Generate runtime tool manifest from tool registry
     4. Assemble final system prompt: `[L0] + [L1] + [tool manifest]`
   - This is NOT a static file read from `\system\prompt.txt`. It is **built at runtime from palace state**. The model wakes up knowing its world.
   - First boot: identity.txt is a default template ("You are genos..."); user updates via `memory.facts_set("user_name", "Alice")` etc.
   - Format of the wake-up header prepended to every single prompt turn:
     ```
     [Session: {id} | {ISO8601} | Turn: {N} | Context: {used}/{budget}t | RAM: {free}MB]
     ```
   - This header is non-negotiable. Without it the model is temporally blind and has no OS self-awareness.

3. **Timer HAL + wall clock** — `genos-hal/src/timer.rs`
   - `get_wall_time() -> EfiTime` via `EFI_RUNTIME_SERVICES.GetTime()` (called at boot, cached)
   - `now_ms() -> u64` monotonic via UEFI `EFI_TIMER_ARCH_PROTOCOL`
   - `monotonic_ns() -> u64` via x86 TSC (`rdtsc` intrinsic) for sub-millisecond benchmarking
   - `sleep_ms(u64)` via UEFI `Stall()`
   - Wall clock is used by wake-up protocol and all memory entry timestamps

4. **HTTP HAL** — `genos-hal/src/net.rs`
   - `fetch(url: &str) -> Result<Vec<u8>, NetError>` wrapping `EFI_HTTP_PROTOCOL`
   - GET and POST support, header parsing (Content-Length, Content-Type, chunked encoding)
   - DNS hostname resolution via `EFI_DNS4_PROTOCOL`
   - HTTP/1.1 `Connection: keep-alive` for warm connections (~50ms vs ~400ms per subsequent request on same domain)
   - Timeout: 5s default, configurable per call
   - No TLS (HTTP only; BearSSL added in Phase D)

5. **Tool-call protocol** — `genos-tools/src/json.rs`
   - Hand-rolled JSON parser (~300 LOC, no external crate, no_std)
   - `ToolCall { tool: String, args: JsonValue, call_id: String, session_id: String }`
   - `ToolResult { call_id: String, ok: bool, result: JsonValue, error: Option<ToolError>, elapsed_ms: u64 }`
   - `ToolError { code: String, message: String, retryable: bool }`
   - `session_id` on every call/response enables cross-session audit log from day 1
   - Tool manifest auto-generated from registry — system prompt always reflects current tools

6. **Core tool executor** — `genos-agent/src/tools.rs`, Phase B tools:
   - `fs.read(path) -> Vec<u8>`
   - `fs.write(path, content) -> bool`
   - `fs.list(dir) -> Vec<FileEntry>`
   - `fs.delete(path) -> bool` (policy check required in Phase E; for Phase B, just log)
   - `net.fetch(url) -> Vec<u8>` (raw bytes)
   - `memory.facts_get() -> Vec<(String, String)>` — returns L1 facts.kv
   - `memory.facts_set(key, value)` — includes contradiction detection (see item 9 below)
   - `memory.log_turn(turn_data)` — raw verbatim journal append

7. **Hall-routed session journal** — wired from Phase B, expanded in Phase C
   - Every turn is stored in the palace with a hall tag. Agent auto-classifies:
     - A command executed → `events`
     - A decision made → `facts`
     - Something the model learned → `discoveries`
     - A preference expressed by user → `preferences`
     - A recommendation given → `advice`
   - Classification prompt (on every turn, ~20 tokens of overhead):
     `"Tag this exchange: facts|events|discoveries|preferences|advice"`
   - Written to: `\palace\wings\general\halls\{hall}\session_{id}_{turn}.txt` (verbatim)
   - Also written to flat journal: `\palace\sessions\{session_id}.jsonl`
   - Format: `{session_id, turn, ts, hall, wing, input, output, tool_calls: [...]}`
   - **Raw verbatim only** — no summarization at write time (MemPalace lesson: summarization loses 12-34% recall vs verbatim)

8. **Pre-compact verbatim archiver** — `genos-agent/src/repl.rs`, `fn pre_compact()`
   - Track `context_tokens_used` after every turn
   - When `context_tokens_used > 80% of context_budget`:
     1. Archive oldest 500 tokens verbatim to `\palace\sessions\{session_id}\overflow_{N}.txt`
     2. Scan that text for any L1-worthy facts (`key: value` patterns), call `memory.facts_set()` for each
     3. Drop those tokens from the active context
   - **Never lose a word**. Every byte goes to palace before being dropped from context.
   - This is the MemPalace "PreCompact hook" — fires before any context compression, not after.
   - Critical: do NOT summarize at this stage. Verbatim archive first. Summarization is Phase C's consolidation pass.

9. **Contradiction detection in facts.kv** — built into `memory.facts_set()`
   - Before writing: check if key already exists in `facts.kv`
   - If same key, different value:
     - Timestamp both entries:
       ```
       current_project = "old_value" [valid_until: 2026-04-09T14:22Z, superseded_by: entry_042]
       current_project = "new_value" [valid_from: 2026-04-09T14:22Z]
       ```
     - Show user inline: `⚠ Fact updated: current_project = "new_value" (was: "old_value")`
     - Old entry archived to `\palace\wings\general\halls\facts\superseded.kv`, not deleted
   - This is Phase B's primitive contradiction detection — Phase C wires this into the entity graph

10. **Screen improvements** — `genos-hal/src/screen.rs`
    - Scrollback buffer: last 100 lines in ring buffer in memory
    - ANSI-like attribute support: bold, inverse, color (4 colors via UEFI text attributes)
    - Cursor positioning: enable status bar at bottom row

11. **Agent loop refactor** — `genos-agent/src/repl.rs`
    - Boot sequence (before first turn):
      1. `palace::ensure_structure()` — create dirs/files
      2. `wakeup::load()` — build L0+L1 system prompt
      3. Assign `session_id`, log session start to palace
    - Per-turn sequence:
      1. Read user input
      2. Prepend wake-up header: `[Session: ... | ISO8601 | Turn: N | Context: X/Yt]`
      3. Build full prompt: `[system_prompt] [wake_up_header] [last_K_turns] [user_input]`
      4. Call LLM kernel, stream tokens
      5. Scan output for JSON `{"tool": ..., "args": ..., "call_id": ...}`
      6. Execute tool, append `{"tool_result": {...}}` to prompt
      7. Continue until EOS or model stops emitting tool calls
      8. Classify output hall type (facts/events/discoveries/preferences/advice)
      9. Write verbatim to palace: `palace::store(session_id, turn, hall, input, output)`
      10. Run `pre_compact()` if context budget > 80%
      11. Update turn counter and session journal

**Verification**:
- Unit tests for JSON parser, facts.kv contradiction detection on host (`cargo test -p genos-kernel --features hosted`)
- `fs.write`/`fs.read` round-trips on FAT32 in QEMU
- `net.fetch` retrieves a short HTTP resource (test locally with `python3 -m http.server`)
- Wake-up loads L0+L1 correctly on cold boot (palace dirs created, identity loaded)
- Pre-compact fires at 80% context: verify verbatim file appears in `\palace\sessions\`
- Contradiction detection: set same key twice, verify both versions timestamped in facts.kv
- Session journal grows correctly across multiple REPL turns

**ESP layout after Phase B**:
```
\EFI\BOOT\BOOTX64.EFI
\models\stories15m.bin
\models\tokenizer.bin
\palace\
  identity.txt              ← L0: ~50 tokens, edited by user or auto-generated
  facts.kv                  ← L1: ~120 tokens, key-value facts
  wings\general\halls\
    facts\                  ← decisions, locked choices
    events\                 ← session logs, what happened
    discoveries\            ← insights, learned things
    preferences\            ← user habits and preferences
    advice\                 ← model recommendations
  sessions\                 ← per-session verbatim .jsonl files
\system\
  config.toml               ← sampling params, context budget, consolidation interval
```

**What Phase B deliberately does NOT do** (moved to Phase C):
- Semantic/TF-IDF search — palace is write-only in Phase B, search comes in Phase C
- Consolidation pass — Phase C adds the background summarizer
- Project/person wings — Phase B uses only `wings\general\`
- Cross-session retrieval — journal exists in Phase B, search index built in Phase C
- Entity graph (temporal triples) — Phase C

---

### ✅ Phase C — Web Intelligence + Memory (COMPLETE)

All 11 deliverables implemented across 10 commits. ~1,500 LOC across 8 new/modified files.
Key additions: full BPE tokenizer, TF-IDF search index, web intelligence (llms.txt-first),
extended memory tools (store/search/consolidate/forget), temporal entity graph,
consolidation pass, project wings + tunnels, cross-session continuity, input history, TUI status bar.

**Goal**: Semantic search over the palace, web browsing, cross-session retrieval, consolidation pass, richer TUI. *Depends on Phase B. The palace directory structure and hall taxonomy already exist — Phase C indexes and searches it.*

**Deliverables**:

1. **`web.get_page(url)` tool** — `genos-tools/src/web.rs`
   - **Protocol**: always check `<host>/llms.txt` first → `<host>/llm.txt` → fallback raw HTML
   - HTML tag stripping (~200 LOC regex-free parser, no external crate)
   - Semantic chunk extraction: `[{type: "title"|"heading"|"paragraph"|"code"|"table", text, depth?}]`
   - Context budget respect: if `total_chunks_tokens > budget`, return first N + `has_more: true` + pagination cursor
   - Source metadata alongside every page: `{domain, age_days, has_llms_txt: bool}`

2. **Palace search index (L2)** — `genos-tools/src/palace.rs`
   - Build TF-IDF index over palace halls: `\palace\wings\*\halls\*\*.txt`
   - Index shard per hall type — queries hit only the relevant hall, not the full archive
   - `palace.search(query, wing?, hall?, top_k?) -> [(entry_id, text, score, timestamp)]`
   - Wing+hall filtering: MemPalace benchmark shows 34% recall improvement (94.8% vs 60.9%) vs global search
   - Index rebuilt by consolidation pass; fast incremental append at Phase B write time
   - Return results in recency-weighted order: `score = tfidf_score * (1.0 / (1.0 + days_old))`

3. **Four-layer memory stack** — fully operational in Phase C (L0+L1 bootstrapped in Phase B)

   | Layer | Storage | Size | When loaded |
   |-------|---------|------|-------------|
   | **L0** | `\palace\identity.txt` | ~50 tokens | Always (already loaded by Phase B wake-up) |
   | **L1** | `\palace\facts.kv` | ~120 tokens | Always (already loaded by Phase B wake-up) |
   | **L2** | TF-IDF index over palace halls | On demand | Auto-search on every user input |
   | **L3** | Raw verbatim drawers in `\palace\sessions\` + overflow files | On demand | Explicit search only |

   L2 fires automatically: before assembling the prompt, run `palace.search(user_input, top_k=3)`. If score > threshold, inject those N entries into prompt. Otherwise skip.

4. **Consolidation pass** — `genos-tools/src/palace.rs`, background task
   - Triggers every 20 turns (configurable in `\system\config.toml`)
   - Steps:
     1. Read last N session journal entries (`\palace\sessions\{session_id}.jsonl`)
     2. Run model with extraction prompt: `"Extract decisions, preferences, and facts as key=value pairs"`
     3. For each extracted fact: call `memory.facts_set(key, value)` (contradiction check happens here — already wired from Phase B)
     4. Re-rank all L2 index entries by `access_count + recency_weight`
     5. Prune entries with `last_access > 90 days` → move to `\palace\archive\`
     6. Rebuild any stale index shards
   - This pass converts Phase B's raw verbatim dump into queryable organized memory over time

5. **Project wings + tunnels** — `genos-tools/src/palace.rs`
   - Phase B only has `wings\general\`. Phase C introduces named wings:
     - `wings\{project_name}\` — one per project (model creates via `palace.create_wing("language_genius")`)
     - `wings\{person_name}\` — one per person the user works with
   - Tunnels: when the same room name appears in two different wings (e.g., "auth-migration" under both a project wing and a person wing), a tunnel auto-links them. Cross-wing search follows tunnels.
   - `palace.list_wings() -> Vec<WingMeta>`
   - `palace.create_wing(name, wing_type: "project"|"person") -> WingId`
   - `palace.find_tunnels(room_name) -> Vec<(wing_a, wing_b, room_name)>`

6. **Temporal entity graph** — `genos-tools/src/memory/graph.rs`
   - Triple store: `(entity_a, relation, entity_b, valid_from, valid_until)`
   - Storage: flat binary B-tree over `(entity_a, relation)` — no SQLite, no external crate
   - Facts have validity windows. Invalidation sets `valid_until`, not deletion.
   - `memory.graph_add(entity_a, relation, entity_b)` — inserts with current timestamp
   - `memory.graph_query(entity, as_of?) -> Vec<(entity_b, relation)>`
   - `memory.graph_invalidate(entity_a, relation, entity_b)` — sets `valid_until = now`
   - `memory.graph_timeline(entity) -> Vec<Triple>` — chronological story

7. **Extended memory tools** (adds to Phase B's `facts_get/set`):
   - `memory.store(content, wing, hall, tags?) -> entry_id` — write verbatim to palace + update index
   - `memory.search(query, wing?, hall?, top_k?) -> Vec<SearchResult>` — delegates to palace search
   - `memory.consolidate()` — force consolidation pass (normally background)
   - `memory.forget(entry_id)` — move entry to archive, remove from index

8. **Conversation continuity** — cross-session recall
   - On boot: after wake-up (L0+L1), search palace for last session's context: `palace.search("last session summary", top_k=2)`
   - Inject last session summary into prompt context if found
   - User experience: model knows what was discussed last time without being told

9. **TUI status bar** — `genos-hal/src/screen.rs`
   - Bottom row: `[{model}] | {tokens/sec} tok/s | RAM {used}/{total}GB | turns: {N} | uptime: {HH:MM:SS}`
   - Updated after every response

10. **Input line editing** — `genos-hal/src/keyboard.rs`
    - Left/right cursor movement (Phase A already has backspace)
    - Up-arrow: cycle through last 10 inputs (stored in agent memory ring buffer, not palace)

11. **Full BPE tokenizer** — `genos-kernel/src/tokenizer.rs`
    - Load from `\models\vocab.json` + `\models\merges.txt` (Hugging Face BPE format)
    - BPE encode: text → token IDs
    - Decode: token IDs → text, with leading-space stripping
    - Required for Gemma 4 E2B's 262k vocab in Phase D

**Verification**:
- `web.get_page` returns readable chunks with correct llms.txt priority
- `palace.search("topic")` returns relevant verbatim entries from Phase B-written journals
- Wing creation works, tunnel linking fires when same room appears in two wings
- Continuity: reboot QEMU, model correctly references something from last session
- Consolidation pass runs after 20 turns, updates facts.kv correctly
- Memory round-trips: `memory.store` → `memory.search` returns it

**ESP layout after Phase C**:
```
\palace\
  identity.txt
  facts.kv
  wings\
    general\halls\{facts,events,discoveries,preferences,advice}\
    {project_name}\halls\...
    {person_name}\halls\...
  sessions\               ← verbatim session journals
  archive\                ← entries >90 days old
\models\vocab.json        ← Full BPE vocabulary (for Phase D)
\models\merges.txt        ← BPE merge operations
\system\config.toml       ← consolidation interval, context budget, etc.
```

---

### ✅ Phase D — Gemma 4 E2B + Quantization (COMPLETE)

**Goal**: Replace Stories15M with a real instruction-following model. *Depends on Phase C; GGUF loader can start parallel with C.*

**Status**: All 8 deliverables implemented. ~1,770 LOC across 4 new kernel files + 2 modified files. 23 kernel unit tests + 86 integration tests = 109 passing, zero warnings.

**What shipped**:
- GGUF v3 parser (`gguf.rs`, ~400 LOC): header, metadata, tensor info, aligned data section, all GGML quant types
- Quantized math engine (`simd.rs`, ~530 LOC): F16/BF16 conversion, Q4_0/Q8_0/Q4_K dequant + vec_dot, blocked matmul dispatcher, embedding lookup, GELU (tanh variant), RMSNorm, softmax, logit softcapping, RoPE with configurable theta+rotary_dim, PolarQuant (polar coordinate decomposition with runtime trig LUTs)
- QJL KV cache (`kv_cache.rs`, ~290 LOC): 3-bit key compression (~9× savings), 2-bit value compression (~13× savings), per-layer storage with sliding window support, memory estimation
- Gemma 4 E2B forward pass (`gemma4.rs`, ~550 LOC): config from GGUF metadata or hardcoded defaults, PLE injection, hybrid sliding-window (512) + full attention (every 5th layer), p-RoPE (theta=10k standard / theta=1M with partial_rotary_factor=0.25), 8:1 GQA, GeGLU FFN (double-wide), logit softcapping (30.0), implements LLMRuntime trait
- Sampling upgrades (`sampler.rs`): top-k, min-p, repetition penalty, full pipeline: temperature → softmax → top-k → min-p → top-p → sample

**Architecture corrections vs original spec** (verified against actual HuggingFace config.json):
- sliding_window = 512 (not 4096)
- Full attention layers every 5th (indices 4,9,14,19,24,29,34), not every 6th
- head_dim = 256 (sliding) / 512 (full), not uniform
- num_key_value_heads = 1 (8:1 GQA), not 8

**Target model**: **Gemma 4 E2B** (`google/gemma-4-E2B-it`)
- 2.3B effective / 5.1B total parameters
- 35 layers, 262k vocabulary, 128k context
- Hybrid sliding-window + global attention (local every layer, global every 6th)
- Per-Layer Embeddings (PLE) injection per transformer block
- GQA (8 KV heads) — enables smaller KV cache
- Native function calling (built into base instruction tuning)
- ~3.2GB at int4 quantization, ~1.5GB at int3

**Deliverables**:

1. **GGUF v3 loader** — `genos-kernel/src/gguf.rs`
   - Parse magic bytes `GGUF`, version, metadata key-value
   - Tensor metadata: name, shape, dtype (f32, f16, int4, int3)
   - Memory-map tensor data for zero-copy loading (UEFI FAT32 → kernel memory)
   - Validate architecture: build Gemma 4 layer config from metadata

2. **Gemma 4 forward pass** — extend `genos-kernel/src/inference.rs`
   - Per-Layer Embeddings: inject embeddings at each transformer block (not just start)
   - Sliding-window attention for local layers: window size 4096
   - Full-attention layers (every 6th): `Q @ K.T → softmax → V`
   - GQA interpolation: expand KV heads if model was trained with KV sharing
   - GeGLU FFN variant (replaces SwiGLU): `(X @ W_a) * (X @ W_b)` (no gating)

3. **PolarQuant weight quantization** — `genos-kernel/src/simd.rs`
   - TurboQuant component 1: **PolarQuant** quantizes model *weights* using polar coordinate decomposition
   - Weights stored as polar form (magnitude + angle), eliminating per-block memory overhead vs Q4_K_M
   - Dequantize-on-the-fly during matmul: decompose polar → cartesian → `f32` inside AVX2 registers
   - Target: Q4_K_M parity (~3.2GB Gemma 4 E2B) but with zero scaling-factor overhead
   - Block size 32 compatible with GGUF format; fall back to standard Q4_K_M if polar path not available
   - AVX2 matmul kernel with in-register dequantization; pure-Rust fallback for non-AVX2

4. **QJL KV-cache** — `genos-kernel/src/kv_cache.rs`
   - TurboQuant component 2: **QJL** (Johnson-Lindenstrauss 1-bit sign compression) quantizes the *KV cache*, not the weights
   - Keys: 3-bit via JL sign projection; Values: 2-bit (values are smoother, tolerate more compression)
   - Reduces KV footprint from ~4GB (unquantized, 128k context) to ~700MB with zero accuracy loss
   - No training required — projection matrix is a fixed random sign matrix (seeded, deterministic)
   - Separate from PolarQuant: PolarQuant = weights, QJL = KV cache. Together = TurboQuant.
   - Custom slab allocator for KV blocks allocated per sequence position
   - Also apply QJL to palace L2 semantic search vectors (Phase C) for 6× memory reduction on embedding index

5. **Sampling upgrades** — extend `genos-kernel/src/sampler.rs`
   - Top-k (keep only top K tokens by probability)
   - Repetition penalty (reduce likelihood of repeated n-grams)
   - Min-p (minimum probability threshold)
   - Read config from `\system\config.toml`: temperature, top_k, top_p, rep_penalty

6. **`sys.introspect()` and context awareness**
   - `sys.clock() -> {iso8601, unix_ts, uefi_time}`
   - `sys.introspect() -> {ram_free_mb, ram_used_mb, tokens_per_sec, context_tokens_used, context_tokens_budget, uptime_secs, active_tasks, disk_free_mb}`
   - Model can check `context_tokens_used` and decide when to trigger `sys.gc()`

7. **Context window management** — `genos-agent/src/repl.rs`
   - Track `context_tokens_used` every turn
   - When `context_tokens_used > 0.75 * context_budget`, trigger `sys.gc()`
   - `sys.gc()`: model compresses oldest 1000 tokens into key facts, stores to memory, frees context
   - Model continues without losing continuity

8. **`bench` tool** — `genos-tools/src/sys.rs`
   - Run N forward passes of varying sequence lengths
   - Report: tokens/sec, peak RAM, KV-cache memory usage, layer latency breakdown
   - Helps users understand performance on their hardware

**Verification**:
- Gemma 4 E2B boots and generates coherent multi-turn conversation
- Tool calls (JSON) are correctly parsed and executed
- Follows instructions: "List the authors of the website you just read" → correctly recalls `web.read` results
- Context window management: run 50+ turns without crash
- Target: 5-15 tokens/sec on modern x86 CPU with AVX2
- KV-cache uses <1GB for 8k context window

**ESP layout**:
```
/models/gemma4-e2b.gguf   ← Gemma 4 E2B weights (GGUF v3, int4, ~3.2GB)
/models/gemma4-e2b-vocab.json   ← Gemma 4 tokenizer vocab (262k)
/models/gemma4-e2b-merges.txt   ← Gemma 4 BPE merges
/system/config.toml       ← Sampling: temp, top_k, top_p, rep_penalty
```

---

### Phase E — Agentic Environment

**Goal**: Autonomous research, tool creation, multi-panel TUI, background tasks. *Depends on Phase D.*

**Deliverables**:

1. **Auto-research agent** — `genos-tools/src/research.rs`
   - `research.start(topic, depth, output_dir) -> job_id`
   - Internal loop:
     - `web.search(topic)` → get 5 URLs
     - For each: `web.get_page()` or `web.read_llms()` to ingest
     - Extract entities + facts → `memory.graph_add(...)`
     - Store findings → `memory.store(..., namespace="research")`
     - Detect contradictions: `memory.graph_query(entity)` checks for conflicting info
     - If depth > 1: extract links, recurse
   - On completion: `research.summarize(job_id)` compresses findings into structured summary
   - Writes outputs to `\research\{topic}\`

2. **Cooperative scheduler** — `genos-agent/src/scheduler.rs`
   - Task queue (named tasks: REPL foreground, research jobs, memory consolidation)
   - Round-robin: each task runs until it yields (either outputs N tokens or calls a tool)
   - No preemption — tasks are cooperative
   - Priority: foreground REPL > active research > background consolidation
   - `task.yield()` indicates task is ready to continue or is blocked on I/O

3. **Tool builder** — user-defined tools at runtime
   - Model emits `tool.define` with JSON spec:
     ```json
     {
       "name": "calc.compound_interest",
       "doc": "Computes interest",
       "args": [{"name": "principal", "type": "f64"}, ...],
       "impl": {"type": "lua", "code": "..."}
     }
     ```
   - Policy engine shows user: "Create tool `calc.compound_interest`? [y/n]"
   - On approval: write to `\tools\user_tools.json`, register at runtime
   - Model can now call it immediately

4. **Code execution sandbox** — `genos-tools/src/code.rs`
   - `code.eval(lang, src, timeout_ms) -> {stdout, stderr, exit_code, elapsed_ms}`
   - Lua 5.4: vendored no_std Rust port (~250KB)
     - No access to `fs`, `net`, or `sys` namespaces (safe sandbox)
     - Can use standard Lua libs: math, string, table, io to stdout
   - Restricted expression evaluator for pure math: `(3.14159 * r^2)` parser
   - Timeout enforces CPU cycle limit to prevent infinite loops

5. **Multi-panel TUI** — `genos-agent/src/ui.rs`
   - GOP framebuffer rendering (pixel-level control via UEFI)
   - Tabs: Chat | Tasks | Memory | Research | System
   - Keyboard shortcuts:
     - Tab key to switch panels
     - F1 for help
     - Ctrl+C to interrupt current task
     - Ctrl+Z to background task
   - Status line across all panels with real-time updates

6. **Memory consolidation** (background task)
   - Runs periodically or on demand
   - Summarize old journal entries (>7 days)
   - Merge related memory keys
   - Prune entries with access_count=0 for 30 days
   - Rebuild semantic index over remaining entries

7. **Policy engine** — `genos-agent/src/policy.rs`
   - Require confirmation before:
     - `fs.delete(path)` on system paths
     - `fs.write(path)` to files modified in last 24h
     - Any `net.fetch` outside allowed domains
   - Rate limiting: `net.fetch` max N requests/minute (default 10)
   - Hard context-window ceiling: when `context_budget` exceeded, truncate oldest non-essential turns before consolidating
   - Safety checks: no tool calls allowed if model output is malformed JSON

8. **Tool versioning & audit log** — `\logs\tool_calls.jsonl`
   - Every tool call logged: `{ts, tool, args, ok, result, elapsed_ms}`
   - Model can review its own history: "What URLs did I fetch last session?"
   - User can audit exactly what the OS did

9. **Research outputs** to `\research\{topic}\`
   - Structured findings: `findings.jsonl` (one finding per line)
   - Summary: `summary.txt` (compressed, for re-reading)
   - Sources: `sources.json` (URLs, dates, credibility scores)
   - Related topics: `related.txt` (auto-linked topics from memory graph)

**Verification**:
- Multi-turn conversation spanning 100+ turns without degradation
- Background research job runs while REPL remains responsive
- User-defined tool (e.g., `calc.fibonacci(n)`) works correctly
- Memory consolidation reduces storage while preserving key facts
- Policy engine correctly blocks dangerous operations
- Multi-panel TUI renders all 5 tabs with live status

**ESP layout (complete)**:
```
/EFI/BOOT/BOOTX64.EFI        ← genos binary
/models/gemma4-e2b.gguf
/models/gemma4-e2b-vocab.json
/models/gemma4-e2b-merges.txt
/system/prompt.txt            ← Master system prompt
/system/identity.txt          ← User identity (L0 memory)
/system/config.toml           ← Sampling, policy params
/memory/facts.kv              ← L1: critical facts
/memory/store.jsonl           ← L2: semantic entries
/memory/archive/              ← L3: old entries >90 days
/logs/journal.jsonl           ← Conversation log
/logs/tool_calls.jsonl        ← Tool audit log
/research/{topic}/            ← Research outputs
/research/cache/              ← Web page cache
/tools/user_tools.json        ← User-defined tools
/data/                        ← User files
```

---

## Detailed Subsystems

### Memory Architecture

The naive Phase C plan (flat TF-IDF on `store.jsonl`) degrades as memory grows. genos uses a **four-layer stack**:

**L0 — Identity (~50 tokens)**
- Stored in `\system\identity.txt`
- "Who are you? What are your core capabilities?"
- Prepended to every system prompt
- Updated rarely, almost never changes mid-session
- Example: "You are genos, a bare-metal LLM OS. You can browse, remember facts, and guide user research."

**L1 — Critical Facts (~120 tokens)**
- Stored in `\memory\facts.kv` (key-value pairs)
- "What are the user's current projects? What decisions have been made?"
- Also prepended to every prompt (after L0)
- Regenerated by consolidation pass after every 20 turns
- If facts exceed 120 tokens, consolidation must compress them
- Example:
  ```
  user_name: Alice
  current_project: LanguageGenius
  tech_stack: Rust + UEFI
  last_research_topic: LLM Quantization
  context_window_budget: 4096
  ```

**L2 — Semantic Session Index (on-demand)**
- TF-IDF index over entries in `\memory\store.jsonl`
- Only retrieved when input keyword matches
- Top-K results (k=3) injected into prompt only on relevance hit
- Reduces context bloat while maintaining access to recent findings

**L3 — Deep Archive (on-demand)**
- Raw verbatim entries: `\logs\journal.jsonl` + `\memory\archive\`
- Never summarized
- Queried only on explicit user request: "Search my memory for...", "Show me when I decided..."

### Consolidation Pass (Runs Every 20 Turns)

The single most important missing piece in naive memory systems:

1. **Extract**: Read last N journal entries, use model to extract decisions, preferences, and facts
2. **Merge**: Add to `facts.kv`, deduplicating by key
3. **Conflict detection**: If new fact contradicts old (same key, different value):
   - Timestamp both: `{key: value_old, valid_until: now, superseded_by: new_entry_id}`
   - Keep old in `store.jsonl` for auditability
   - Use new value going forward
4. **Re-score**: Entries in `store.jsonl` are ranked by:
   - Recency (newer = higher)
   - Access frequency (queried more = higher)
   - Time decay (weight halves after 7 days)
5. **Prune**: Move entries >90 days old to `\memory\archive\`, keep index only in L3

### Temporal Entity Graph

Inspired by Zep's Graphiti, Obsidian's graph linking, and MemPalace's Wing/Room metaphor.

**Triple store format**:
```
(subject, relation, object, valid_from, valid_until)
```

**Examples**:
```
("Alice", "works_on", "LanguageGenius", 2026-01-01, null)
("LanguageGenius", "uses", "Rust", 2026-01-01, null)
("LanguageGenius", "uses", "UEFI", 2026-01-01, null)
("LanguageGenius", "status", "alpha", 2026-01-01, 2026-03-15)
("LanguageGenius", "status", "beta", 2026-03-15, null)
```

**Implementation**:
- Flat binary file with B-tree index over (subject, relation) for fast lookup
- Append-only: invalidations set `valid_until` timestamp, don't delete
- Queries always respect current time to get accurate state
- Historical queries supported: `graph.query(entity, as_of=2026-02-01)` → triples valid at that date

**Query patterns the model can leverage**:
- "Who am I working with?" → `(*, "collaborates_with", *, now)`
- "What do I use?" → `("project_x", "uses", *, now)`
- "When did we decide to use Postgres?" → `(*, *, "postgres", *)` with timestamp
- "What changed about project X?" → `("project_x", *, *, *)` sorted by time

---

### Web Intelligence Layer

#### `web.get_page(url)` Protocol

**Key principle**: Default to LLM-friendly content sources.

1. **Fetch `<host>/llms.txt`** first
   - If found: return it as-is (raw text optimized for LLMs)
2. **Fall back to `<host>/llm.txt`**
   - Alternative naming convention
3. **Fall back to HTML scraping**
   - Strip tags, extract semantic chunks
   - Return: `[{type: "title"|"heading"|"paragraph"|"code"|"table", text, depth?}, ...]`

**Chunk extraction** (no external crate):
```rust
pub enum ChunkType {
    Title,
    Heading { level: u8 },
    Paragraph,
    Code { lang: Option<String> },
    Table,
    List { ordered: bool },
}

pub struct Chunk {
    pub typ: ChunkType,
    pub text: String,
    pub tokens_estimate: usize,
}
```

**Context budget handling**:
- Count total estimated tokens in all chunks
- If `total > context_budget`, return only first N chunks
- Include `has_more: true` and a `page` cursor for pagination

**Domain credibility**:
```json
{
  "url": "https://example.com/article",
  "domain": "example.com",
  "credibility": "high"|"medium"|"low",
  "age_days": 42,
  "chunks": [...],
  "source_llms_txt": true|false
}
```
- Model uses credibility to reason about source trustworthiness

#### Web Tools

```rust
web.search(query, n?) -> Vec<SearchResult>
// Returns structured results, not HTML
// Single endpoint initially (e.g., DDG lite API, no JS)

web.read(url, focus?, page?, max_tokens?) -> PageContent
// focus: "main" strips nav/sidebar, "all" returns everything
// page: cursor for pagination
// max_tokens: hard limit

web.diff(url, since_timestamp) -> Changes
// Fetches page, compares cached version
// Returns only changed sections (for monitoring)

web.extract_links(url, filter?) -> Vec<Link>
// Returns {url, text, type: "internal"|"external"|"document"}

web.cache_read(url) -> Option<CachedPage>
// Return cached version if exists

web.cache_invalidate(url)
// Force fresh fetch on next read
```

---

### Tool-Call Protocol

**Canonical format** (must be strict, no variations):

**Model output** (tool invocation):
```json
{
  "tool": "web.read",
  "args": {
    "url": "https://example.com",
    "focus": "main",
    "max_tokens": 2000
  },
  "call_id": "c1"
}
```

**Dispatcher response** (success):
```json
{
  "call_id": "c1",
  "ok": true,
  "result": {
    "chunks": [
      {"type": "title", "text": "Example Article"},
      {"type": "paragraph", "text": "Some content..."}
    ],
    "total_tokens": 180,
    "has_more": false
  },
  "elapsed_ms": 412
}
```

**Dispatcher response** (error):
```json
{
  "call_id": "c1",
  "ok": false,
  "error": {
    "code": "NET_TIMEOUT",
    "message": "Request timed out after 5000ms",
    "retryable": true
  },
  "elapsed_ms": 5020
}
```

**Key design decisions**:
- `call_id` echoed back enables multi-step reasoning with backwards references
- `elapsed_ms` teaches the model which tools are slow (enables planning)
- `retryable` tells model whether to retry or try alternatives
- `ok: false` is not fatal — model should decide next action

---

### Self-Building Tool System (Phase E+)

#### `code.eval` — The Foundation

Model can write and execute code without a pre-built tool for it.

```rust
code.eval(lang, src, timeout_ms) -> Result<ExecResult>

pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub elapsed_ms: u64,
}
```

**Supported languages** (priority order):

1. **Lua 5.4** — vendored no_std Rust port
   - Safe sandbox: no `fs`, `net`, `sys` access
   - Libs: math, string, table, io (stdout only)
   - 250KB compiled footprint
   - Example: compound interest, date math, unit conversion

2. **Restricted expression evaluator**
   - Parse and evaluate mathematical expressions
   - `(3.14159 * r^2)` → returns number
   - ~200 lines of Rust

**Sandbox restrictions** (no escape):
- Lua interpreter gets no I/O except stdout
- No module loading (no `require`, no DLL injection)
- CPU timeout enforced via instruction counter (rough estimate)

#### Tool Composer

Model proposes new tools at runtime without Rust compilation:

**Model emits**:
```json
{
  "tool": "tool.define",
  "args": {
    "name": "calc.sum_list",
    "doc": "Sum a list of numbers",
    "parameters": [
      {"name": "numbers", "type": "array[number]", "doc": "Numbers to sum"}
    ],
    "output_schema": {"type": "number"},
    "implementation": {
      "type": "lua",
      "code": "function sum(t)\n  local s = 0\n  for _, v in ipairs(t) do\n    s = s + v\n  end\n  return s\nend\nreturn sum(numbers)"
    }
  },
  "call_id": "define1"
}
```

**Policy engine response**:
```
Create new tool 'calc.sum_list'? [y/n/always/never]
```

**On approval**:
1. Validate Lua syntax (parse test)
2. Write to `\tools\user_tools.json`
3. Register in tool registry at runtime
4. Model can call it immediately: `{"tool": "calc.sum_list", "args": {"numbers": [1, 2, 3]}, "call_id": "c42"}`

#### Tool Registry

All tools (built-in and user-defined) stored in a single registry:

```rust
pub struct ToolDef {
    pub name: &'static str,           // "web.read"
    pub namespace: &'static str,      // "web"
    pub doc: &'static str,            // Signature & description
    pub parameters: Vec<ParamDef>,
    pub requires_network: bool,
    pub requires_disk: bool,
    pub timeout_ms: u64,
}
```

**System prompt generation**:
- On boot, generate a compact tool manifest from the registry
- Include in every prompt's system section
- When new user tool is registered, regenerate manifest
- Model always has up-to-date tool list

---

### Runtime Awareness & System Context

#### Wall Clock — Non-Negotiable

UEFI provides `EFI_RUNTIME_SERVICES.GetTime()`. This is injected into every system prompt turn:

```
[2026-04-09T$TIME | RAM: $USED/$TOTAL | Uptime: $UP | Tokens/sec: $TPS]
```

Every memory entry, research finding, and entity graph triple requires a real timestamp. Without this, the model is temporally blind.

#### `sys.introspect()` Tool

```rust
sys.clock() -> {iso8601: String, unix_ts: u64, uefi_time: EfiTime}
sys.introspect() -> {
    ram_free_mb: u64,
    ram_used_mb: u64,
    tokens_per_sec: f32,         // Rolling average
    context_tokens_used: u32,
    context_tokens_budget: u32,
    uptime_secs: u64,
    active_tasks: Vec<String>,
    disk_free_mb: u64,
}
sys.context_budget() -> u32           // Exact tokens remaining
sys.gc() -> ()                        // Trigger garbage collection
```

**Model's usage pattern**:
- Before starting research: check `sys.introspect().ram_free_mb` — don't launch if <500MB free
- Before long response: check `sys.context_budget()` — if <1000 tokens remaining, trigger `sys.gc()`
- Periodically: call `sys.introspect()` and log to memory for runtime monitoring

#### Context Window Management

When `context_tokens_used > 0.75 * context_budget`:

1. Trigger `sys.gc()` automatically
2. `sys.gc()` calls model with compression prompt:
   ```
   Summarize the earliest 1000 tokens of this conversation.
   Extract key facts, decisions, and context needed to continue.
   Be complete but brief (max 200 tokens).
   
   [oldest 1000 tokens of conversation...]
   ```
3. Model emits summary
4. Oldest 1000 tokens replaced with 200-token summary
5. Model continues without losing continuity

This is how you get a persistent AI OS that runs for hours or days without degrading.

---

### Networking Upgrade & HTTPS

#### HTTP-Only (Phase B-C)

Current plan: no TLS, no HTTPS. This is defensible for Phase B-C:
- Most developer documentation (`llms.txt` sites) serve HTTP
- `llms.txt` is explicitly designed for plaintext content
- Eliminates 2-5MB bloat and massive complexity

#### BearSSL HTTPS (Phase D, optional)

**Option**: vendor a minimal TLS 1.2 implementation (45KB, supports `#![no_std]`).

Benefits:
- Access HTTPS-only content
- Credentials over TLS (future password manager tool)

Trade-off: 45KB on an already resource-constrained environment.

**Decision**: Implement in Phase D if bandwidth allows, otherwise stay HTTP-only.

#### DNS Resolution

UEFI's `EFI_DNS4_PROTOCOL` provides DNS. Implement minimal resolver in `genos-hal/src/net.rs`:

```rust
pub fn resolve_hostname(hostname: &str) -> Result<u32>  // Returns IPv4
```

#### Connection Pooling

For research tasks fetching multiple pages from same domain:
- Keep TCP connection alive between requests (HTTP/1.1 `Connection: keep-alive`)
- Reduces latency from ~400ms to ~50ms per subsequent request
- Implement simple connection cache in `genos-hal/src/net.rs`

---

### GGUF Loader & Model Format

#### GGUF v3 Parser

GGUF is well-documented and becoming industry standard for quantized model distribution.

**Format**:
```
Magic: "GGUF" (4 bytes)
Version: u32
Metadata count: u32
metadata[count]: {key_len: u32, key: str, value_type: u32, value: ...}
Alignment: 32-byte boundary
Tensor data (aligned)
```

**Implementation** — `genos-kernel/src/gguf.rs`:

```rust
pub struct GGUFModel {
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: Vec<TensorMeta>,
    pub data_offset: u64,
    pub data: Vec<u8>,  // Memory-mapped from FAT32
}

impl GGUFModel {
    pub fn parse(data: &[u8]) -> Result<Self> { ... }
    pub fn get_tensor(&self, name: &str) -> Result<&[u8]> { ... }
}
```

#### Gemma 4 E2B Architecture

Key aspects implemented in inference engine:

1. **Per-Layer Embeddings (PLE)**: token embeddings injected per layer (not just at start)
2. **Sliding-window attention**: local layers use window size 4096
3. **Full attention**: every 6th layer uses full global attention
4. **GQA**: 8 KV heads (vs. 32+ for query heads)
5. **GeGLU FFN**: `(x @ W_a) * (x @ W_b)` (different from SwiGLU used in Llama)

#### int4 Quantization (Q4_K_M)

**Block-wise quantization**:
- Block size: 32 weights per block
- Scale factor: f16 (one per block)
- Data: 4-bit quantized weights

**Dequantize-on-the-fly matmul**:
```rust
for block_idx in 0..num_blocks {
    let scale = scales[block_idx];
    let block_data = quantized_weights[block_idx * 16..(block_idx+1) * 16];
    // Unpack 4-bit values, scale, multiply
    for i in 0..32 {
        let w_quantized = unpack_int4(block_data, i);
        let w = (w_quantized as f32 - 8.0) * scale;  // rough formula
        result += input[i] * w;
    }
}
```

Never create full-precision weight matrix — always stay in int4 in memory.

#### AVX2 Matmul Kernels

```rust
#[target_feature(enable = "avx2")]
pub unsafe fn matmul_int4_avx2(
    output: &mut [f32],
    input: &[f32],
    weights_int4: &[u8],
    scales: &[f16],
    rows: usize,
    cols: usize,
) {
    // Use _mm256_* intrinsics
    // Dequantize in-register, multiply, accumulate
}
```

**Fallback**: pure-Rust path using scalar ops (slower, always works)

---

### QJL KV-Cache (Phase D)

**Problem**: At 128k context with int32 KV values, KV cache is ~4GB.

**Solution**: 3-bit keys + 2-bit values via Johnson-Lindenstrauss projection.

**Benefits**: ~6-7x compression, minimal quality loss (minimal difference in model output).

**Implementation**:
- Projection matrix: random ±1 / √d matrix (computed once at startup)
- For each sequence position: project K and V to lower dimension
- Store as int3 and int2
- On attention: un-project back to full dimension

---

## Master Prompt for Copilot/Claude

Paste this into a Copilot/Claude session as your "system" context or "custom instructions," then follow up with individual module requests:

---

> **SYSTEM CONTEXT FOR GENOS IMPLEMENTATION**
>
> You are helping implement **genos**, an experimental bare-metal LLM operating system in Rust.
>
> ## Project Vision
>
> **genos** boots directly from UEFI USB with NO underlying operating system. The LLM runtime is the kernel. Users interact through a TUI shell. The LLM is stateful, has long-term memory, browses the web intelligently, and can create tools at runtime.
>
> ## Hard Constraints
>
> - **Target**: `x86_64-unknown-uefi` (UEFI 2.5+)
> - **Language**: Rust nightly (or stable with minimal features)
> - **Runtime**: `#![no_std]` + `core` + `alloc` ONLY
> - **No host OS calls**: No POSIX, no threading, no syscalls
> - **Allocator**: `linked_list_allocator` or custom buddy allocator
> - **Platform**: QEMU/OVMF for dev/test
>
> ## Architecture (5 Rust Crates)
>
> 1. **genos-boot**: UEFI entry, panic handler, boot sequence
> 2. **genos-hal**: Hardware abstraction (screen, keyboard, disk, net, timers) — all UEFI
> 3. **genos-kernel**: LLM runtime (model loading, forward pass, tokenizer, sampler, system prompt)
> 4. **genos-tools**: Tools (fs, web, memory, code, research, sys)
> 5. **genos-agent**: REPL shell, scheduler, policy engine, journaling
>
> ## Current Status
>
> - **Phase A (COMPLETE)**: Stories15M forward pass boots in QEMU, REPL works
> - **Phase B (NEXT)**: Tool API, JSON protocol, HTTP client, system prompt
> - **Phase C**: Web + llms.txt, memory with consolidation, entity graph
> - **Phase D**: Gemma 4 E2B (GGUF loader), QJL KV-cache, AVX2 matmul, context window mgmt
> - **Phase E**: Auto-research, tool builder, multi-panel TUI, background tasks
>
> ## Key Design Decisions
>
> - **Custom forward pass** (not Candle): no_std requirement makes Candle infeasible
> - **HTTP-only networking**: No TLS in Phase B-C (2-5MB bloat); BearSSL in Phase D optional
> - **Cooperative multitasking**: UEFI has no preemptive scheduler; tasks yield voluntarily
> - **FAT32 for everything**: UEFI natively supports it; no custom FS needed
> - **JSON tool calls**: Parseable in no_std, works with Gemma 4's native function calling
> - **Four-layer memory**: L0 identity, L1 facts, L2 semantic index, L3 archive
> - **Consolidation pass**: Critical for memory quality; runs every 20 turns
> - **Temporal entity graph**: Model has access to facts with validity windows
>
> ## When Asking for Implementation
>
> For each module I ask about, provide:
> 1. **Brief design explanation** (2-3 sentences)
> 2. **Full Rust code** (no pseudocode)
> 3. **Any `Cargo.toml` changes**
> 4. **Simple tests or hosted-mode examples** using `#[cfg(feature = "hosted")]`
>
> **Constraints on your responses**:
> - Treat UEFI + firmware as the ONLY environment — no assumptions about a host OS
> - All I/O goes through `genos-hal`; use UEFI APIs only
> - Respect memory limits (target: 512MB heap for model + runtime)
> - No external crates without explicit approval (minimize deps)
> - Comments should explain the why, not just the what
>
> ## Example Quality Requests
>
> - "Implement `genos-hal/src/screen.rs` with UEFI SimpleTextOutput, printing UTF-8 to UCS-2"
> - "Implement the `ToolCall` enum and JSON parser for tool invocations"
> - "Implement `genos-tools/src/memory.rs` with the four-layer stack and consolidation pass"
> - "Implement `genos-kernel/src/gguf.rs`, parsing GGUF v3 headers and tensor metadata"
>
> When ready, I'll ask for specific modules. Start by asking what area to implement first.

---

---

## Key Decisions & Rationale

| Decision | Rationale | Impact |
|----------|-----------|--------|
| **Custom forward pass (not Candle)** | Candle requires std; writing from scratch for Llama/Gemma is cleaner | 2-3 weeks extra but full control, no external dependency issues |
| **HTTP-only in B-C, BearSSL later** | TLS = 2-5MB + complex state machine; most dev content is HTTP-friendly | Limits access initially, but `llms.txt` mitigates; Phase D adds HTTPS if needed |
| **Cooperative multitasking** | UEFI has no preemptive scheduler; interrupts require custom ARM handling | Tasks must yield voluntarily; simpler to reason about but requires disciplined code |
| **FAT32 everywhere** | UEFI natively supports it; no custom FS code needed | Limited to FAT32 features; good enough for Phase A-C, may need upgrade in Phase E |
| **JSON tool protocol** | Parseable in no_std, works with Gemma 4's native function calling | No fine-tuning needed; model understands JSON natively |
| **Four-layer memory** | Naive single-layer storage degrades as entries accumulate; consolidation is key | L0/L1 always in context (small), L2/L3 on demand → O(1) prompt overhead |
| **Gemma 4 E2B over larger** | 2.3B effective params fits in 4-8GB ram at int4; native function calling built-in | Smaller than llama-7B but instruction-capable; future-proofs for SBC ports |
| **QJL KV-cache** | Unquantized KV-cache is 4GB @ 128k context; QJL achieves 6-7x compression | Massive memory savings; minimal quality loss if done right |
| **AVX2 matmul kernels** | int4 inference on x86 is 10-100x slower without SIMD | Enables 10-20 tok/sec; makes model usable |

---

## Risks & Mitigations

| Risk | Severity | Mitigation | Phase |
|------|----------|-----------|-------|
| **Custom LLM math bugs** | HIGH | Unit test every op against PyTorch reference; hosted mode tests | B-D |
| **GGUF parsing errors** | HIGH | Start with simple models (Stories15M in GGUF format for testing), then scale to Gemma 4 | D |
| **int4 quality degradation** | MEDIUM | Use Q4_K_M (better quality than Q4_0); A/B test against fp32 | D |
| **Memory explosion** | MEDIUM | Consolidation pass is mandatory, not optional; test with 1M+ entries in archive | C-E |
| **Entity graph contradictions** | MEDIUM | Model learns to detect: `memory.graph_query(entity)` returns all versions; model sees history | C |
| **AVX2 unavailable on target** | LOW | Pure-Rust fallback path (20x slower but correct); detect at runtime | D |
| **Context window OOM** | MEDIUM | `sys.gc()` with compression; model checks `context_budget()` before large operations | D-E |
| **UEFI firmware inconsistency** | MEDIUM | Test on QEMU + real hardware; graceful degradation if features unavailable | A-E |
| **Networking latency** | LOW | Connection pooling, async-style I/O yields; not blocking issue | B-C |
| **Tokenizer vocab mismatch** | MEDIUM | Strict vocab file format validation; fallback to char-level if BPE load fails | C-D |

---

## Out of Scope

- GPU drivers (x86/NVIDIA/AMD)
- Multi-user privilege model
- Audio/video I/O
- ARM/RISC-V (future ports, not v1)
- Secure boot signing / attestation
- HTTPS/TLS (Phase B-C; deferred to Phase D)
- Package manager / app marketplace
- Full disk encryption
- Virtualization / sandboxed app environment

---

## External Tools & References

### Essential Rust Resources
- [UEFI-rs Tutorial](https://rust-osdev.github.io/uefi-rs/tutorial/app.html)
- [Phil Opp's OS Blog — Allocators](https://os.phil-opp.com/allocator-designs/)
- [Rust No_Std Book](https://docs.rust-embedded.org/book/)

### Model & Quantization
- [TurboQuant Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Gemma 4 E2B Docs](https://developers.googleblog.com/id/introducing-gemma-3-nano-multimodal-ai-on-your-devices/)
- [GGML/llama.cpp Codebase](https://github.com/ggerganov/llama.cpp) — reference for int4 matmul
- [QJL KV-Cache Paper](https://arxiv.org/abs/2411.08896)

### LLM Software Architecture
- [Zep Memory Framework (Python)](https://github.com/getzep/zep) — memory consolidation reference
- [MemPalace Research](https://arxiv.org/abs/2409.18410) — entity graph inspiration
- [llms.txt Standard](https://llmstxt.org)

### Development
- [QEMU + OVMF Setup](https://wiki.osdev.org/UEFI)
- [GDB Remote Debugging](https://sourceware.org/gdb/onlinedocs/gdb/Remote-Protocol.html)

---

## Quick Start for New Contributors

1. **Clone the repo**: `git clone https://github.com/n33levo/genos.git`
2. **Install Rust nightly**: `rustup install nightly && rustup target add x86_64-unknown-uefi`
3. **Install QEMU + OVMF**: `brew install qemu && brew install OVMF` (macOS) or equivalent
4. **Build Phase A**: `make build && make esp && make qemu`
5. **Expected result**: QEMU boots, prints Stories15M output, REPL prompt
6. **Next step**: Read the Phase B section, then ask Copilot/Claude for the first module implementation

---

**Last updated**: April 9, 2026  
**Repository**: https://github.com/n33levo/genos  
**Maintainer**: @n33levo
