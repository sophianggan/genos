#![no_std]

extern crate alloc;

pub mod wire;

pub use wire::{
    ClientIdentity, McpError, RequestId, RequestMetadata, RpcRequest, RpcResponse,
    LATEST_PROTOCOL_VERSION,
};
