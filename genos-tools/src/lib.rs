#![cfg_attr(not(feature = "hosted"), no_std)]

#[cfg(not(feature = "hosted"))]
extern crate alloc;

#[cfg(feature = "hosted")]
extern crate std as alloc;

pub mod fs;
pub mod graph;
pub mod journal;
pub mod mcp_transport;
pub mod memory;
pub mod net;
pub mod palace;
pub mod protocol;
pub mod search;
pub mod sys;
pub mod web;
