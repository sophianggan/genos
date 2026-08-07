# genos

A bare-metal operating system where the LLM *is* the kernel.

Boots directly from USB via UEFI — no Linux, no Windows, no BSD. After boot, the only "system" is an LLM runtime plus a set of tools. The model receives structured tool calls and emits decisions. Users interact through a text shell. It builds up a persistent memory of you and your work over time.

Built entirely in Rust (`#![no_std]`, `x86_64-unknown-uefi`).

---

## What this is

Most "AI operating systems" are just chat interfaces bolted onto an existing kernel. This is different — there is no kernel underneath. The firmware hands control directly to `BOOTX64.EFI`, and from that point on the only thing running is the model.

The LLM doesn't assist the OS. The LLM **is** the OS.

It schedules work, decides what to read and write to disk, browses the web, remembers context across sessions, and can spawn background research tasks — all through a structured tool API with no human mediation.

---

## Architecture

Six Rust crates in the UEFI workspace plus one hosted companion, each with a narrow boundary:

```
genos-boot    UEFI entry point, init, panic handler
genos-hal     Hardware abstraction (screen, keyboard, disk, net, timers)
genos-kernel  LLM runtime — model loading, inference, tokenizer, sampler, tool protocol
genos-mcp     Provider-neutral MCP wire model, clients, server, policy, catalogs
genos-tools   Concrete tools: filesystem, web, memory, code, media
genos-agent   REPL shell, task scheduler, policy engine
bridge/       Hosted stdio/legacy MCP adapter (separate workspace)
```

The separation means `genos-kernel` has no dependency on UEFI at all — it compiles and tests on your host machine.

### MCP platform

GenOS can discover and use tools, resources, and prompts from arbitrary MCP servers configured in
`\system\mcp.toml`. Modern Streamable HTTP endpoints connect directly through hosted sockets or
UEFI HTTP. Local and legacy stdio servers use the loopback-only companion bridge. Every server has
an isolated client, namespaced catalog, credential reference, trust level, limits, exact-call
consent, output provenance, and redacted audit events.

Start with [the MCP quickstart](docs/MCP_QUICKSTART.md), use the
[configuration and compatibility reference](docs/MCP_CONFIGURATION.md) for exact support details,
then see the [architecture and threat model](docs/MCP_ARCHITECTURE.md).

---

## Model

**Phase A–C:** [Stories15M](https://huggingface.co/karpathy/tinyllamas) — a 15M-parameter Llama 2 model in the llama2.c binary format. Tiny, fast to load, good for proving the boot + inference pipeline works.

**Phase D (current):** [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) — Google's 2.3B effective parameter edge model (5.1B total with Per-Layer Embeddings). 35 layers, 262k vocab, 128k context. Hybrid sliding-window (512) + full attention (every 5th layer). 8:1 GQA. GeGLU FFN. At int4 quantization it needs ~3.2 GB RAM. Native function calling, reasoning (thinking mode). Full forward pass implemented with TurboQuant (PolarQuant weights + QJL-compressed KV cache).

---

## Current state

Phases A through D are done. The kernel now includes both the original Stories15M Llama 2 runtime and the full Gemma 4 E2B forward pass with TurboQuant quantization.

**Test status**: 107 integration + 26 kernel + 19 MCP-core + 6 bridge tests = **158 tests passing**.

Phase D additions (~1,770 LOC across 4 new files + 2 modified):
- `gguf.rs` — GGUF v3 parser (metadata, tensor info, aligned data extraction)
- `simd.rs` — Quantized math: F16/BF16/Q4_0/Q8_0/Q4_K dequant + matmul, GELU, RMSNorm, RoPE, softmax, logit softcapping, PolarQuant
- `kv_cache.rs` — QJL-compressed KV cache (3-bit keys ~9× compression, 2-bit values ~13× compression, sliding window)
- `gemma4.rs` — Complete Gemma 4 E2B forward pass (PLE injection, hybrid attention, p-RoPE, GQA, GeGLU, logit softcapping)
- `sampler.rs` — Extended with top-k, min-p, repetition penalty

The EFI binary builds clean:

```
genos-boot.efi   467 KB   x86_64 UEFI application
```

Boot sequence:
1. UEFI hands control to `genos-boot.efi`
2. Heap allocator initialized from UEFI memory map
3. Model weights + tokenizer loaded from `\models\` on the EFI FAT32 partition
4. BPE tokenizer and sampler initialized
5. REPL loop starts — user types prompts, model generates responses token-by-token

---

## Phases

### Phase A — Bare-metal LLM (done)

Boot → heap → screen/keyboard/disk HAL → full Llama 2 forward pass (RMSNorm, RoPE, GQA attention, SwiGLU FFN) → BPE tokenizer → top-p sampler → interactive REPL.

### Phase B — OS Kernel Layer

Turn the prototype into a real OS environment:
- HTTP client on UEFI's `EFI_HTTP_PROTOCOL`, with firmware HTTPS where available
- JSON tool-call protocol: `{"tool": "fs.read", "args": {"path": "..."}}`
- System prompt: tells the model it IS the OS, lists available tools, enforces JSON output
- Tool executor: `fs.read`, `fs.write`, `fs.list`, `net.fetch`
- Multi-turn loop: user input → tool calls → execute → feed back → final response

### Phase C — Web + Memory

- `web.get_page(url)` — checks `/llms.txt` then `/llm.txt` before scraping raw HTML (strips tags)
- Persistent memory: `memory.store(key, content, tags)` → `\memory\store.jsonl` on disk
- Fuzzy memory search: TF-IDF over stored entries, auto-injected into context
- Conversation journal: every turn logged to `\logs\journal.jsonl`, loaded on next boot
- Improved TUI: status bar (model, tokens/sec, RAM), scrollable output, input editing

### Phase D — Gemma 4 E2B + Quantization (done)

Replace Stories15M with a real instruction-following model:
- GGUF v3 loader: parse magic, version, metadata, tensor info, aligned data — supports all GGML quant types (F32, F16, BF16, Q4_0, Q8_0, Q4_K, PolarQuant)
- Complete Gemma 4 E2B forward pass (35 layers, 262k vocab, 128k context):
  - Per-Layer Embeddings (PLE) injection at every transformer block
  - Hybrid attention: 4 sliding-window (window=512) + 1 full attention (every 5th layer)
  - p-RoPE: standard theta=10000 for sliding layers, theta=1M with partial_rotary_factor=0.25 for full layers
  - 8:1 GQA (8 query heads, 1 KV head per layer)
  - GeGLU FFN (GELU gate × up → down) with double-wide intermediate
  - Logit softcapping at 30.0
- TurboQuant — two complementary compression systems:
  - **PolarQuant** (weights): polar coordinate decomposition, dequantize-on-the-fly during matmul
  - **QJL** (KV cache): 3-bit key compression (~9× savings), 2-bit value compression (~13× savings), sliding window support
- Sampling upgrades: top-k, min-p, repetition penalty (configurable via `\system\config.toml`)
- Native function calling: Gemma 4 E2B has built-in tool use — directly emits structured tool calls

### Phase E — Agentic Environment

- Provider-neutral MCP host/client/server foundation: dynamic discovery, tools, resources,
  prompts, stateless Streamable HTTP, legacy stdio bridge, policy firewall, and human consent
- `research.run(topic, depth)` — long-lived loop: search → read → summarize → store results in `\research\`
- Cooperative task scheduler: REPL + research jobs + memory consolidation run interleaved (no preemption)
- Tool builder: LLM designs new tools, user approves, stored as JSON dispatch configs
- Multi-panel TUI: Chat | Tasks | Memory | Research | System tabs
- Policy engine: destructive actions need confirmation, network rate limiting, context window management

---

## Disk layout (EFI partition, FAT32)

```
/EFI/BOOT/BOOTX64.EFI     genos binary (boots automatically)
/models/stories15m.bin     model weights (llama2.c format)
/models/tokenizer.bin      BPE tokenizer
/system/prompt.txt         system prompt (editable)
/system/config.toml        sampling params, model selection
/system/mcp.toml           provider-neutral MCP endpoint registry
/secrets/                  referenced MCP credentials (never committed)
/memory/store.jsonl        persistent memory entries
/logs/journal.jsonl        conversation log
/research/                 auto-research outputs
/data/                     user files
```

---

## Getting started

### Requirements

- Rust nightly (installed automatically via `rust-toolchain.toml`)
- QEMU (for testing in a VM before flashing to USB)

```bash
# macOS
brew install qemu

# Debian/Ubuntu
sudo apt install qemu-system-x86 ovmf
```

### Build

```bash
git clone https://github.com/sophianggan/genos
cd genos
make build
```

### Download model

```bash
make setup-model
```

Downloads Stories15M (~60 MB) from HuggingFace to `esp/models/`.

### Run in QEMU

```bash
make qemu
```

This builds the EFI binary, creates the ESP directory structure, and boots it in QEMU with OVMF firmware. On Apple Silicon (M1/M2/M3) it emulates x86\_64 via software — works fine, just slower than native.

### Flash to USB (run on real hardware)

```bash
make esp
sudo dd if=/dev/zero of=genos.img bs=1M count=256
# format as FAT32, copy esp/ contents
# or use the flash target:
make flash USB=/dev/sdX   # replace /dev/sdX with your drive
```

Boot your machine from USB. Most UEFI firmware will auto-detect `EFI/BOOT/BOOTX64.EFI`.

---

## Development

### Running tests on your host machine

The kernel crate (`genos-kernel`) has no UEFI dependencies. You can run unit tests on your normal machine:

```bash
make test
```

This compiles with `--features hosted` which enables `std` and runs tests for the transformer math, tokenizer, and sampler.

The workspace `.cargo/config.toml` sets `x86_64-unknown-uefi` as the default target. If you run kernel tests directly with cargo, you must specify the host target explicitly:

```bash
# macOS (Apple Silicon)
cargo test -p genos-kernel --features hosted --target aarch64-apple-darwin

# macOS (Intel)
cargo test -p genos-kernel --features hosted --target x86_64-apple-darwin

# Linux x86_64
cargo test -p genos-kernel --features hosted --target x86_64-unknown-linux-gnu
```

The integration tests in `tests/` already have their own `.cargo/config.toml` and work with `cargo test` directly.

### Adding a new tool

1. Add a function in `genos-tools/src/`
2. Register it in the tool executor in `genos-agent/src/repl.rs`
3. Add it to the system prompt in `resources/prompt.txt`

### Targeting a different model

The inference engine in `genos-kernel/src/inference.rs` implements the standard Llama 2 / llama2.c format. Any model in that format works. For Gemma 4 E2B (GGUF), Phase D will add a separate loader path.

---

## Project notes

No GPU support — CPU-only with SIMD acceleration (AVX2). GPU drivers from firmware would be an enormous scope increase. The Gemma 4 E2B at int4 is designed to run on phones, so modern CPU performance is workable.

HTTPS is available through the hosted platform TLS stack and through firmware that implements UEFI HTTPS. Certificate validation on bare metal follows the firmware trust store and therefore varies by machine; authenticated remote MCP endpoints fail closed unless their URL is HTTPS.

Cooperative multitasking only — no interrupts, no preemption. Tasks yield voluntarily between token generation steps. Simple and correct for a single-user environment.

---

## License

MIT
