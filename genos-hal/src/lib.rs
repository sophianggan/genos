#![cfg_attr(not(feature = "hosted"), no_std)]

#[cfg(not(feature = "hosted"))]
extern crate alloc;

#[cfg(feature = "hosted")]
extern crate std as alloc;

pub mod disk;
pub mod keyboard;
pub mod net;
pub mod screen;
pub mod timer;
