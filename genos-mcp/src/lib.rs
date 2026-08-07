#![no_std]

extern crate alloc;

pub mod client;
pub mod config;
pub mod execution;
pub mod primitives;
pub mod security;
pub mod server;
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
pub use security::{
    resolve_auth, ApprovalGrant, AuditEvent, AuditLog, CredentialError, CredentialResolver,
    GuardedResult, PolicyDecision, PolicyFirewall, Provenance, ResolvedAuth, Secret,
};
pub use server::{McpServer, McpService, RoutingMetadata, ServerCapabilities, ServerInfo};
pub use transport::{
    Header, HttpRequest, HttpResponse, HttpTransport, McpHttpClient, TransportError,
    TransportErrorKind,
};
pub use wire::{
    ClientIdentity, McpError, RequestId, RequestMetadata, RpcRequest, RpcResponse,
    LATEST_PROTOCOL_VERSION,
};
