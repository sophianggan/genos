#![no_std]

extern crate alloc;

pub mod client;
pub mod config;
pub mod execution;
pub mod primitives;
pub mod transport;
pub mod wire;

pub use client::{
    ClientError, ClientManager, ClientState, ClientStatus, Discovery, ServerIdentity,
};
pub use config::{
    AuthConfig, ConfigError, CredentialRef, McpConfig, McpDefaults, ServerConfig, TransportKind,
    TrustLevel,
};
pub use execution::{CallOutcome, Content, Continuation, ListKind, McpHostError, ToolInvocation};
pub use primitives::{
    CacheScope, Catalog, Page, PrimitiveError, Prompt, PromptArgument, Resource, ServerCatalog,
    Tool,
};
pub use transport::{
    Header, HttpRequest, HttpResponse, HttpTransport, McpHttpClient, TransportError,
    TransportErrorKind,
};
pub use wire::{
    ClientIdentity, McpError, RequestId, RequestMetadata, RpcRequest, RpcResponse,
    LATEST_PROTOCOL_VERSION,
};
