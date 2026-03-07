//! Memory management utilities for OxiBLAS.
//!
//! This module provides:
//! - Aligned memory allocation (`alloc`)
//! - Stack-based temporary allocation (`stack`)
//! - Aligned vectors (`aligned_vec`)
//! - Memory pools (`pool`)
//! - NUMA-aware allocation (`numa`)
//! - Arena allocators (`arena`)

pub mod aligned_vec;
pub mod alloc;
pub mod arena;
#[cfg(feature = "std")]
pub mod numa;
pub mod pool;
pub mod stack;

// Re-export commonly used items
pub use aligned_vec::*;
pub use alloc::*;
pub use arena::*;
#[cfg(feature = "std")]
pub use numa::*;
pub use pool::*;
pub use stack::*;
