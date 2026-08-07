#![no_std]

extern crate alloc;

pub mod config;
pub mod wire;

pub use config::{
    AuthConfig, ConfigError, CredentialRef, McpConfig, McpDefaults, ServerConfig, TransportKind,
    TrustLevel,
};
pub use wire::{
    ClientIdentity, McpError, RequestId, RequestMetadata, RpcRequest, RpcResponse,
    LATEST_PROTOCOL_VERSION,
};
