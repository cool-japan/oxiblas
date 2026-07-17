//! x86_64 SIMD implementations using SSE4.2, AVX2, and AVX512.
//!
//! This module provides SIMD register types and operations for x86_64
//! processors. It includes:
//! - SSE4.2 (128-bit): F64x2Sse, F32x4Sse
//! - AVX2 (256-bit): F64x4, F32x8
//! - AVX512 (512-bit): F64x8, F32x16
//!
//! Split out of a single `x86_64.rs` (see `functions` for the runtime feature
//! detection + scalar fallbacks shared by every tier). The submodules below are
//! implementation detail; the public surface is the register types re-exported
//! here.

// Allow these clippy lints for SIMD code (inherited by every submodule):
// - should_implement_trait: We use add/sub/mul/neg methods on trait implementations
// - missing_transmute_annotations: Transmutes in SIMD are clear from context
// - incompatible_msrv: AVX-512 intrinsics require newer Rust but we gate with runtime detection
// - needless_range_loop: Index-based loops are clearer for SIMD element access patterns
#![allow(clippy::should_implement_trait)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::needless_range_loop)]

mod f32x16_traits;
mod f32x4sse_traits;
mod f32x8_traits;
mod f64x2sse_traits;
mod f64x4_traits;
mod f64x8_traits;
pub mod functions;
pub mod types;
pub mod type_aliases;

// Re-export the public register types so the API stays `simd::x86_64::TypeName`.
pub use type_aliases::*;
pub use types::*;
