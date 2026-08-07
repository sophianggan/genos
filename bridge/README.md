# GenOS MCP bridge

The bridge makes local stdio MCP servers available to a hosted GenOS process
through the same Streamable HTTP boundary used by remote servers. GenOS cannot
launch child processes on UEFI, so this companion runs on a conventional host.

Build and run it with any stdio MCP server command:

```bash
make bridge-build
cargo run --manifest-path bridge/Cargo.toml \
  --target "$(rustc -vV | awk '/^host:/{print $2}')" -- -- \
  npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/files
```

Then configure GenOS:

```toml
[[servers]]
id = "filesystem"
transport = "stdio_bridge"
endpoint = "http://127.0.0.1:8787/mcp"
auth = "none"
trust = "read_only"
require_approval = true
```

The built-in listener only binds loopback. For a VM, another machine, or real
bare-metal GenOS, put the bridge behind a mutually authenticated TLS reverse
proxy and use an `https://` endpoint. Do not expose its plaintext port to a LAN.
Credentials for the child server should be passed through that process's
environment; the bridge never converts HTTP authorization headers into child
environment variables.

The bridge is intentionally narrow: bounded HTTP/1.1 POST requests in, one
JSON-RPC message per stdio line out. It validates MCP routing headers against
the JSON body, rejects non-loopback binding, bounds requests and responses to
4 MiB, does not log message bodies, and leaves TLS/OAuth to the reverse proxy.
