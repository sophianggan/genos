# MCP architecture for GenOS

Status: accepted  
Target protocol: MCP `2026-07-28`  
Scope: provider-neutral MCP host, client, and server foundations for a `no_std` UEFI system

## Decision

GenOS will become an MCP host rather than a collection of vendor-specific integrations. Each
configured MCP server gets an isolated client context. The host discovers its primitives,
namespaces them, applies policy before calls and after results, and exposes approved capabilities
to the model. GenOS also exposes its native tools and resources through an MCP server adapter.

The implementation has four layers:

```text
LLM / REPL
    |
    v
GenOS MCP host
  - endpoint registry       one isolated client per server
  - catalog                 tools, resources, prompts
  - policy firewall         consent, least privilege, limits, taint
  - audit log               redacted, correlated events
    |
    +----------------------+-----------------------+
    v                      v                       v
direct HTTP adapter   hosted bridge adapter   GenOS server adapter
remote MCP servers   stdio/TLS MCP servers   native GenOS tools
```

The core must remain `#![no_std]` and dependency-light. The official Rust MCP SDK currently uses
Tokio, Serde, and a hosted async runtime, so it is appropriate for the companion bridge but not for
the EFI binary. The wire model and orchestration therefore live in a small `genos-mcp` crate that
uses the existing GenOS JSON implementation.

## Why this shape

The current MCP architecture assigns orchestration, consent, context aggregation, and security to
the host. It also specifies a one-to-one relationship between a client and a server so servers
cannot see the whole conversation or each other. GenOS will preserve those boundaries rather than
flattening every server into a shared, implicitly trusted registry.

MCP `2026-07-28` is stateless at the protocol layer. Every request carries its version, client
identity, and client capabilities. `server/discover` replaces the required initialization
handshake; `Mcp-Method` and `Mcp-Name` make HTTP requests routable; list and resource results are
cacheable; multi-round-trip requests carry explicit continuation state; tasks are an optional
extension. The core targets this model and keeps version negotiation explicit so a legacy adapter
can support `2025-11-25` and earlier servers without contaminating the modern core.

## Components

### Protocol core

`genos-mcp` owns JSON-RPC envelopes, protocol metadata, MCP errors, primitive descriptors,
pagination cursors, cache hints, and extension identifiers. It does not perform I/O and does not
know about credentials. This keeps wire compatibility independently testable.

### Endpoint registry

An endpoint is data, not code. Users add entries to `\\system\\mcp.toml` containing an ID,
transport, endpoint or bridge address, authentication reference, trust level, and limits. The
registry never contains raw secrets. It resolves a credential reference through a credential
provider at request time.

Names are exposed to the LLM as `mcp.<server>.<primitive>`. Namespacing prevents two servers from
silently shadowing each other and makes policy and audit events attributable.

### Client manager

The manager creates one client state per endpoint and owns:

- optional `server/discover` and protocol-version fallback;
- deterministic, paginated discovery of tools, resources, and prompts;
- TTL-aware catalogs whose private entries are never shared across identities;
- request IDs, bounded retries, deadlines, response-size limits, and cancellation;
- multi-round-trip continuation state and optional task handles;
- catalog invalidation when subscriptions or legacy notifications report changes.

Transport behavior is behind a trait. No policy decision is permitted inside a transport.

### Transport adapters

The direct adapter emits one HTTP POST per JSON-RPC request. Modern requests include
`MCP-Protocol-Version`, `Mcp-Method`, and, when applicable, `Mcp-Name`. HTTPS is mandatory for
non-loopback authenticated endpoints. Redirects may not change origins while carrying a secret.

Bare-metal GenOS uses the UEFI HTTP service binding directly, including caller-provided MCP and
authorization headers. HTTPS availability and certificate validation inherit the machine's
firmware implementation and trust store. Where firmware networking is absent or unsuitable, a
hosted reverse proxy terminates TLS and the companion bridge launches stdio servers behind a
narrow HTTP MCP endpoint. This is an explicit deployment boundary, not a claim that plaintext
HTTP is safe.

Custom transports can be added by implementing the same request/response trait. This is how GenOS
can support stdio, in-memory test transports, serial links, or future MCP transports without
changing protocol or policy code.

### GenOS MCP server

The inbound adapter maps GenOS native tools, resources, and prompts to MCP. It uses the same policy
engine as local LLM calls. The server surface is deny-by-default and can expose a narrower set of
capabilities than the local agent sees. It validates routing headers against JSON bodies, bounds
request sizes and nesting, and returns structured JSON-RPC errors without leaking credentials or
internal paths.

### Policy firewall

Tool descriptions and tool results are untrusted input. The firewall runs twice:

1. Input gate: resolve server and tool identity, validate arguments against schema, minimize
   disclosed data, enforce allowed scopes, require approval for side effects, and apply budgets.
2. Output gate: limit size and content types, label provenance, redact secrets, and mark external
   text as untrusted data before it reaches the prompt.

Trust is scoped to an endpoint and capability, not transitively inherited across servers. A result
from server A never grants permission to call server B. Sampling and roots are not enabled in new
deployments because the current protocol deprecates them; elicitation is handled through explicit
multi-round-trip user interaction. User approval is bound to the exact server, tool, normalized
arguments, and expiry.

### Credentials

Configuration contains references such as `env:GITHUB_TOKEN`, `file:\\secrets\\sentry.token`, or
`oauth:sentry`, never bearer values. Credential providers return short-lived material into a
request-local buffer. The host:

- refuses authenticated non-loopback HTTP;
- never forwards one server's token to another origin;
- binds OAuth client registration and tokens to the authorization-server issuer and resource;
- validates OAuth issuer responses and token audience;
- supports PKCE for public-client authorization;
- redacts authorization headers and configured sensitive JSON fields from logs;
- zeroes mutable secret buffers where the platform permits it.

Static bearer/API-key authentication remains available because many MCP servers use it, but is
clearly identified as a compatibility mode. OAuth and enterprise-managed authorization are
transport capabilities, not hard-coded vendors.

## Agent loop

```text
discover/cache catalog
        |
        v
model selects namespaced primitive
        |
        v
validate schema -> policy/consent -> resolve credential -> transport
        |                                      |
        | denied                               v
        +------------------------------ structured MCP result
                                               |
                                               v
                         output filter -> provenance envelope -> model
```

The model receives compact schemas only for enabled capabilities. Catalog retrieval is separate
from tool execution so prompt caching remains stable. ReAct supports interleaving reasoning with
actions, while Toolformer and Gorilla support selective tool use and retrieval of changing API
descriptions. In GenOS this translates to dynamic discovery plus an observe/decide/act loop, with
the deterministic host—not the model—responsible for security and protocol correctness.

## Failure model

Failures are typed and localized:

- protocol: invalid JSON-RPC, unsupported version, method/header mismatch;
- transport: DNS, connect, TLS, timeout, cancellation, response too large;
- authentication: missing credential, invalid issuer, expired token, insufficient scope;
- policy: disabled server, denied capability, approval required, budget exceeded;
- remote: MCP error or tool execution error;
- content: unsupported content type, schema mismatch, unsafe or oversized result.

Retries are allowed only for explicitly retryable transport failures and idempotent operations.
Tool calls are not assumed idempotent. Circuit breakers isolate repeatedly failing endpoints.

## Compatibility strategy

The modern core speaks `2026-07-28`. A compatibility layer may fall back after
`UnsupportedProtocolVersionError` and own all legacy lifecycle/session behavior. This prevents
legacy session IDs, server-sent event assumptions, and deprecated client features from leaking
into new code.

The official SDK is used as an interoperability oracle in hosted development and bridge tests,
not as the protocol architecture. Structural contract tests use recorded specification-shaped
messages and an in-memory transport. Live third-party tests are opt-in because they are slow,
credentialed, and unstable.

## Testing policy

The requested test expansion is structural rather than exhaustive. Tests cover:

- crate boundaries and `no_std` compilation;
- JSON-RPC envelope shapes and required metadata;
- required HTTP routing headers and secret redaction;
- endpoint/config parsing without embedded secrets;
- one-client-per-server namespacing and deterministic catalogs;
- policy defaults, approval boundaries, and non-transitive trust;
- request routing for tools, resources, prompts, discovery, and errors;
- mocked round trips and legacy fallback seams.

They intentionally do not attempt full OAuth conformance, production TLS verification, fuzzing,
load testing, every media type, or compatibility with every external server.

## Delivery sequence

1. Land this decision and threat model.
2. Add the `no_std` MCP protocol crate.
3. Add endpoint configuration and credential references.
4. Add transport contracts and modern HTTP framing.
5. Add discovery, versions, and isolated client state.
6. Add deterministic catalogs for tools, resources, and prompts.
7. Add call execution, caching, pagination, and MRTR states.
8. Add the GenOS MCP server adapter.
9. Add the policy firewall, trust labels, and audit redaction.
10. Wire the MCP host into the agent and boot configuration.
11. Add the hosted stdio/TLS bridge contract and setup examples.
12. Add structural tests, operator documentation, and validation commands.

## Sources and design evidence

Protocol requirements:

- [MCP specification `2026-07-28`](https://modelcontextprotocol.io/specification/2026-07-28)
- [MCP architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP versioning and compatibility](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP `2026-07-28` release rationale](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
- [Official Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk)

Agent/tool research:

- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023.
- Schick et al., [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761), NeurIPS 2023.
- Patil et al., [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334), 2023.

Security research:

- Zhan et al., [InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents](https://arxiv.org/abs/2403.02691), ACL Findings 2024.
- Liu et al., [Formalizing and Benchmarking Prompt Injection Attacks and Defenses](https://www.usenix.org/system/files/usenixsecurity24-liu-yupei.pdf), USENIX Security 2024.
- Maloyan and Namiot, [Breaking the Protocol: Security Analysis of MCP](https://arxiv.org/abs/2601.17549), 2026 preprint. Its protocol-specific findings are treated as threat-model input, not established consensus.

## Consequences

Positive consequences are broad interoperability, testable boundaries, no dependency on a vendor
catalog, and a security decision point outside the model. Costs are a new protocol crate, a bridge
for capabilities unavailable in UEFI, and explicit compatibility code for older servers. Full
production hardening still requires verified behavior across firmware HTTP implementations,
hardware-backed credential storage, an OAuth authorization UI, and conformance/fuzz testing; this
foundation makes those increments possible without redesigning the agent.
