//! x86_64 SIMD implementations using SSE4.2, AVX2, and AVX512.
//!
//! This module provides SIMD register types and operations for x86_64
//! processors. It includes:
//! - SSE4.2 (128-bit): F64x2Sse, F32x4Sse
//! - AVX2 (256-bit): F64x4, F32x8
//! - AVX512 (512-bit): F64x8, F32x16

pub mod f32x16_traits;
pub mod f32x4sse_traits;
pub mod f32x8_traits;
pub mod f64x2sse_traits;
pub mod f64x4_traits;
pub mod f64x8_traits;
pub mod types;
pub mod macros;
pub mod type_aliases;
pub mod functions;

// Re-export all types
pub use types::*;
pub use type_aliases::*;
