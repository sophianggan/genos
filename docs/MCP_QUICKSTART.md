# Connect MCP servers to GenOS

GenOS is a provider-neutral MCP host. You register endpoints as data in
`\system\mcp.toml`; the OS discovers their tools, resources, and prompts at runtime. No provider
SDK or source-code change is required.

The protocol core targets MCP `2026-07-28`. Current Streamable HTTP servers connect directly.
Local and older stdio servers connect through the included bridge, which adapts the legacy
initialize lifecycle to GenOS's stateless client boundary.

## 1. Configure a remote Streamable HTTP server

Copy `resources/mcp.toml`, enable an entry, and replace the example URL:

```toml
[defaults]
timeout_ms = 15000
max_response_bytes = 1048576
require_approval = true

[[servers]]
id = "search"
transport = "streamable_http"
endpoint = "https://mcp.example.com/mcp"
enabled = true
auth = "bearer"
credential_ref = "file:\secrets\search.token"
trust = "untrusted"
allow_tools = true
allow_resources = true
allow_prompts = true
require_approval = true
allowed_tools = ["search", "fetch"]
```

Put only the token in `\secrets\search.token` on the EFI partition. Do not put a token in
`mcp.toml`: configuration accepts credential references, not inline secrets. API-key servers use:

```toml
auth = "api_key"
header_name = "X-API-Key"
credential_ref = "file:\secrets\service.key"
```

For an unauthenticated endpoint use `auth = "none"` and omit `credential_ref`. Authenticated and
remote endpoints must use HTTPS. Firmware connections use the machine's UEFI HTTP/HTTPS support
and trust configuration, so hardware support varies.

The configuration grammar also reserves `env:`, `oauth:`, and `firmware:` credential providers.
The EFI runtime currently implements `file:`. Environment variables belong to the hosted stdio
server process; OAuth and firmware-vault references require a separately installed credential
provider and fail closed when none is present.

## 2. Connect a local or legacy stdio server

Build the companion on a conventional operating system:

```bash
make bridge-build
cargo run --manifest-path bridge/Cargo.toml \
  --target "$(rustc -vV | awk '/^host:/{print $2}')" -- -- \
  npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/files
```

For a hosted GenOS process, configure `http://127.0.0.1:8787/mcp`. Bare-metal GenOS has a
different loopback interface, so expose the bridge through an authenticated HTTPS reverse proxy
and configure that proxy URL. The bridge itself deliberately refuses non-loopback binds.

Pass secrets needed by the stdio server in that child process's environment. The bridge does not
log request bodies, copy HTTP authorization into child variables, or expose a plaintext LAN
listener. On first discovery it automatically distinguishes modern stdio from legacy servers;
legacy state stays in the hosted process.

## 3. Install the configuration

`make esp` copies the sample to a new EFI system partition. If `esp/system/mcp.toml` already
exists, update it explicitly so local changes are not overwritten accidentally:

```bash
make esp
cp resources/mcp.toml esp/system/mcp.toml
mkdir -p esp/secrets
cp /path/to/search.token esp/secrets/search.token
make qemu-net
```

On physical hardware, copy the same `system/mcp.toml` and credential files to the FAT32 EFI
partition. Treat that partition as sensitive: file credentials are convenient compatibility
credentials, not a hardware-backed vault.

## 4. Use it from the agent

The model-facing host API is stable regardless of provider:

| Host tool | Purpose |
|---|---|
| `mcp.servers` | Show configured endpoints and connection state |
| `mcp.connect` | Discover one server and build its deterministic catalog |
| `mcp.tools` | List namespaced discovered tools |
| `mcp.call` | Invoke a tool after policy and exact-call approval |
| `mcp.resource` | Read a server resource |
| `mcp.prompt` | Retrieve a server prompt |

Ask the agent to connect the configured server, inspect its tools, then perform the task. Remote
tools are internally namespaced as `mcp.<server-id>.<tool-name>` so two integrations cannot shadow
one another. When policy requires consent, GenOS displays the exact server, tool, and arguments;
type `yes` to authorize only that call for 60 seconds.

## Policy defaults

Start restrictive and relax deliberately:

- `untrusted`: every tool call needs human approval.
- `read_only`: only tools explicitly annotated `readOnlyHint: true` can run.
- `trusted`: policy can allow calls without the untrusted-server rule; destructive annotations
  still require approval.
- `allowed_tools`: an empty list allows all discovered names; a non-empty list is an allowlist.
- `max_response_bytes`, timeout, per-turn budgets, and a circuit breaker are host-enforced.

Tool descriptions and results remain untrusted data even after transport authentication. GenOS
redacts common secret fields, labels output provenance, and never lets trust from one endpoint
authorize another.

## Troubleshooting

- `config_error`: fix the reported line in `\system\mcp.toml`; unknown keys fail closed.
- `credential_error`: verify the referenced file exists and contains UTF-8 without line breaks.
- `legacy_adapter_required`: place that endpoint behind the included bridge.
- `transport_error`: verify UEFI has an HTTP service binding, DNS/networking is available, and the
  firmware accepts the endpoint's TLS chain.
- empty catalog: verify the endpoint advertises the relevant capability and that `allow_tools`,
  `allow_resources`, or `allow_prompts` is enabled.

For protocol rationale, component boundaries, and the threat model, see
[`MCP_ARCHITECTURE.md`](MCP_ARCHITECTURE.md).
For every configuration key, compatibility boundary, and adapter extension point, see
[`MCP_CONFIGURATION.md`](MCP_CONFIGURATION.md).

For development, `make test-mcp` runs the MCP core and bridge structural suites without contacting
or requiring credentials for a third-party server.
