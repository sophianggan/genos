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

Five Rust crates in a workspace, each with zero overlap:

```
genos-boot    UEFI entry point, init, panic handler
genos-hal     Hardware abstraction (screen, keyboard, disk, net, timers)
genos-kernel  LLM runtime — model loading, inference, tokenizer, sampler, tool protocol
genos-tools   Concrete tools: filesystem, web, memory, code, media
genos-agent   REPL shell, task scheduler, policy engine
```

The separation means `genos-kernel` has no dependency on UEFI at all — it compiles and tests on your host machine.

---

## Model

**Phase A–C:** [Stories15M](https://huggingface.co/karpathy/tinyllamas) — a 15M-parameter Llama 2 model in the llama2.c binary format. Tiny, fast to load, good for proving the boot + inference pipeline works.

**Phase D+:** [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) — Google's 2.3B effective parameter edge model (5.1B total with Per-Layer Embeddings). Designed for phones and edge devices. At int4 quantization it needs ~3.2 GB RAM. It has native function calling, reasoning (thinking mode), 128k context, and multimodal input (text + image + audio). This is what genos will run in production.

---

## Current state

Phase A is done and the EFI binary builds clean:

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
- HTTP client on UEFI's `EFI_HTTP_PROTOCOL` (no TLS needed yet)
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

### Phase D — Gemma 4 E2B + Quantization

Upgrade to a real model:
- GGUF loader for Gemma 4 E2B architecture: hybrid local/global attention, Per-Layer Embeddings, 262k vocab
- QJL quantized KV-cache (3-bit keys, 2-bit values) — ~5-6x memory savings
- int4 matmul kernels with AVX2 SIMD via `core::arch::x86_64`
- Top-k/top-p/temperature sampling, repetition penalty, configurable via `\system\config.toml`
- Native function calling: Gemma 4 E2B has built-in tool use support — the model directly emits structured tool calls without prompt engineering

### Phase E — Agentic Environment

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
git clone https://github.com/n33levo/genos
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

### Adding a new tool

1. Add a function in `genos-tools/src/`
2. Register it in the tool executor in `genos-agent/src/repl.rs`
3. Add it to the system prompt in `resources/prompt.txt`

### Targeting a different model

The inference engine in `genos-kernel/src/inference.rs` implements the standard Llama 2 / llama2.c format. Any model in that format works. For Gemma 4 E2B (GGUF), Phase D will add a separate loader path.

---

## Project notes

No GPU support — CPU-only with SIMD acceleration (AVX2). GPU drivers from firmware would be an enormous scope increase. The Gemma 4 E2B at int4 is designed to run on phones, so modern CPU performance is workable.

No TLS/HTTPS — the UEFI HTTP Boot protocol handles plain HTTP natively. Adding TLS would bloat the binary by 2-5 MB and add significant complexity. HTTP-only for now, with llms.txt-aware fetching for developer-friendly sites.

Cooperative multitasking only — no interrupts, no preemption. Tasks yield voluntarily between token generation steps. Simple and correct for a single-user environment.

---

## License

MIT
