#![no_std]

extern crate alloc;

pub mod config;
pub mod transport;
pub mod wire;

pub use config::{
    AuthConfig, ConfigError, CredentialRef, McpConfig, McpDefaults, ServerConfig, TransportKind,
    TrustLevel,
};
pub use transport::{
    Header, HttpRequest, HttpResponse, HttpTransport, McpHttpClient, TransportError,
    TransportErrorKind,
};
pub use wire::{
    ClientIdentity, McpError, RequestId, RequestMetadata, RpcRequest, RpcResponse,
    LATEST_PROTOCOL_VERSION,
};
