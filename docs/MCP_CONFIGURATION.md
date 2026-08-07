# MCP configuration and compatibility reference

This reference describes the MCP functionality that is implemented today, the deployment boundary
for each transport, and the exact `\system\mcp.toml` schema. It is intentionally explicit about
compatibility gaps so an endpoint never appears safer or more portable than it is.

## Compatibility matrix

| Server shape | GenOS path | Status | Notes |
|---|---|---|---|
| MCP `2026-07-28` Streamable HTTP | Direct `https://` endpoint | Supported | Stateless requests, discovery, tools, resources, prompts, pagination, MRTR results |
| MCP `2026-07-28` stdio | Hosted bridge | Supported | Bridge exposes one bounded HTTP request per stdio exchange |
| Legacy stateful stdio | Hosted bridge | Supported | Bridge negotiates `initialize`, sends `notifications/initialized`, and translates discovery |
| Legacy HTTP with sessions or SSE | Compatibility gateway | Not direct | Put a gateway in front that presents the `2026-07-28` stateless contract |
| Custom non-HTTP transport | New adapter | Extension point | Implement the transport boundary and select it in the agent runtime |
| GenOS as an MCP server | Embedded server router | Library support | The router exists; a deployment still needs to bind it to an inbound transport |

Both `streamable_http` and `stdio_bridge` configurations are HTTP from the EFI runtime's point of
view. `stdio_bridge` documents that the endpoint is owned by the companion process; GenOS never
launches a child process on bare metal.

## File location and loading

GenOS reads `\system\mcp.toml` once when the agent runtime starts. A missing file produces an empty
registry. A malformed file produces an empty registry plus a `config_error` returned by
`mcp.servers`. Unknown keys are rejected rather than ignored.

The parser supports the documented TOML subset: `[defaults]`, repeated `[[servers]]` sections,
quoted strings, booleans, positive integers, string arrays, and comments. It does not attempt to
interpret arbitrary TOML features.

## Defaults

```toml
[defaults]
timeout_ms = 15000
max_response_bytes = 1048576
require_approval = true
```

| Key | Type | Default | Meaning |
|---|---|---:|---|
| `timeout_ms` | integer | `15000` | Per-request transport deadline |
| `max_response_bytes` | integer | `1048576` | Maximum response body accepted from a server |
| `require_approval` | boolean | `true` | Require a human confirmation for each exact tool call |

Defaults are copied into each server when its section begins. A server-level value overrides the
copied default.

## Server fields

```toml
[[servers]]
id = "example"
transport = "streamable_http"
endpoint = "https://mcp.example.com/mcp"
enabled = true
auth = "bearer"
credential_ref = "file:\secrets\example.token"
trust = "untrusted"
allow_tools = true
allow_resources = true
allow_prompts = true
require_approval = true
timeout_ms = 15000
max_response_bytes = 1048576
allowed_tools = ["search", "fetch"]
```

| Key | Required | Values | Behavior |
|---|---|---|---|
| `id` | Yes | lowercase letters, digits, `-`, `_` | Stable namespace and policy identity; must be unique |
| `transport` | No | `streamable_http`, `stdio_bridge`, custom string | Defaults to `streamable_http`; custom values need a runtime adapter |
| `endpoint` | Yes | URL | HTTPS, or loopback HTTP for an unauthenticated local bridge |
| `enabled` | No | boolean | Defaults to `true`; disabled servers cannot discover or execute |
| `auth` | No | `none`, `bearer`, `api_key`, `oauth` | Defaults to `none` |
| `credential_ref` | Conditional | credential reference | Required for every auth mode except `none` |
| `header_name` | API key only | HTTP header name | Defaults to `X-API-Key` |
| `trust` | No | `untrusted`, `read_only`, `trusted` | Defaults to `untrusted` |
| `allow_tools` | No | boolean | Enables tool discovery and execution |
| `allow_resources` | No | boolean | Enables resource discovery and reads |
| `allow_prompts` | No | boolean | Enables prompt discovery and retrieval |
| `require_approval` | No | boolean | Server-specific exact-call approval rule |
| `timeout_ms` | No | integer | Server-specific request deadline |
| `max_response_bytes` | No | positive integer | Server-specific response bound |
| `allowed_tools` | No | string array | Empty permits discovered tools; non-empty is an exact-name allowlist |

Server IDs become model-facing namespaces such as `mcp.example.search`. The remote name remains
`search` on the wire. This prevents one server from shadowing another and makes every approval and
audit record attributable.

## Authentication modes

### No authentication

```toml
auth = "none"
```

Omit `credential_ref`. Remote cleartext HTTP is still refused.

### Bearer token

```toml
auth = "bearer"
credential_ref = "file:\secrets\example.token"
```

GenOS resolves the reference for each request and sends `Authorization: Bearer <value>`. The value
is not stored in the endpoint registry or audit journal.

### API key

```toml
auth = "api_key"
header_name = "X-API-Key"
credential_ref = "file:\secrets\example.key"
```

The configured header is marked sensitive and redacted from request summaries.

### OAuth profile

```toml
auth = "oauth"
credential_ref = "oauth:work-account"
```

This is a credential-provider contract, not an embedded browser flow. The current EFI resolver
fails closed until an OAuth authorization provider is installed. A provider must bind tokens to
the authorization issuer and MCP resource, validate audiences, and return only request-scoped
bearer material.

## Credential providers

| Reference | EFI resolver | Intended owner |
|---|---|---|
| `file:\secrets\name.token` | Implemented | FAT32 credential file |
| `env:NAME` | Not available on bare metal | Hosted process or bridge launcher |
| `oauth:profile` | Extension point | Hosted authorization broker or future UI |
| `firmware:variable` | Extension point | Hardware/firmware credential vault |

Credential files may end in CR/LF, which is trimmed while loading. The resulting credential must
be non-empty UTF-8 and cannot contain an embedded line break. Secret buffers and resolved
authorization values are cleared on drop on a best-effort basis.

Configuration files containing raw tokens are not supported. This is deliberate: a value without
a provider prefix fails parsing instead of being treated as a secret.

## Trust and consent

Trust is local policy metadata; it is not inferred from TLS or authentication.

- `untrusted` requires human approval for every tool call.
- `read_only` permits only tools whose server annotation explicitly contains
  `readOnlyHint: true`; absence of the annotation is a denial.
- `trusted` removes the untrusted-server rule, but `require_approval = true` and
  `destructiveHint: true` still require confirmation.

Approval is bound to the server ID, remote tool name, exact JSON arguments, and a 60-second expiry.
Changing any field invalidates it. Approval of one server never grants another server permission.
Resources and prompts pass through the output guard but do not execute a tool side effect.

Each server also receives a per-turn call budget and a circuit breaker. Three consecutive failures
open the circuit for 30 seconds. Results exceeding their configured limit are rejected or replaced
with a bounded truncation record before reaching the model.

## Transport requirements

Modern requests are JSON-RPC HTTP POSTs carrying:

- `Content-Type: application/json`
- `Accept: application/json, text/event-stream`
- `MCP-Protocol-Version`
- `Mcp-Method`
- `Mcp-Name` when a named primitive is addressed

The same protocol version, client identity, and client capabilities are present in request `_meta`.
The bridge and GenOS server router reject routing headers that disagree with the JSON body.

Authenticated endpoints and all non-loopback remote endpoints must use HTTPS. Hosted builds use
the platform TLS stack. Bare-metal builds use the firmware HTTP service binding and inherit that
firmware's network drivers, HTTPS implementation, certificate validation, and trust store.

The included bridge binds only `127.0.0.1` or `::1`. For a VM or physical GenOS machine, place an
authenticated HTTPS reverse proxy in front of the bridge; the host's loopback is not the guest's
loopback.

## Runtime sequence

1. `mcp.servers` reports parsed configuration and connection state.
2. `mcp.connect` sends `server/discover` and selects a compatible protocol.
3. GenOS retrieves each enabled primitive catalog, following at most 100 pages.
4. Catalog entries are sorted and exposed under the server namespace.
5. A tool call passes argument-shape and policy checks and, when required, exact human approval.
6. The credential is resolved into request-local headers and the transport performs one exchange.
7. The result is redacted, size-bounded, labeled with provenance, and returned as untrusted data.

Tool calls are not retried automatically because they are not assumed idempotent.

## Extending the platform

The generic seams are Rust traits rather than provider registries:

- Implement `genos_mcp::transport::HttpTransport` for another HTTP/TLS environment.
- Add a runtime transport selector before using a custom `transport` string.
- Implement `genos_mcp::security::CredentialResolver` for a vault or authorization broker.
- Implement `genos_mcp::server::McpService` to expose another GenOS capability set.
- Keep stateful or deprecated protocol behavior inside a bridge adapter instead of adding session
  state to the modern client core.

A new adapter must preserve response bounds, routing validation, credential origin binding,
redacted diagnostics, and one-client-per-server isolation. Provider-specific SDK types should not
cross these interfaces.

## Validation commands

```bash
make test-mcp
cargo check --workspace --target x86_64-unknown-uefi
make build
make bridge-build
```

The MCP tests are structural and credential-free. Live third-party conformance, OAuth UI flows,
firmware TLS interoperability, fuzzing, and load tests remain separate production-hardening work.
