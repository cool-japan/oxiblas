//! Scalar traits for numeric types used in OxiBLAS.
//!
//! This module defines the trait hierarchy for numeric types:
//! - `Scalar`: Base trait for all scalar types
//! - `Real`: Real number types (f32, f64, f16 with feature, QuadFloat with f128 feature)
//! - `ComplexScalar`: Complex number types
//! - `Field`: Field operations (complete algebraic structure)

mod batch;
mod complex;
mod complex_impl;
mod extended;
mod real_impl;
mod traits;

#[cfg(test)]
mod tests;

// Re-export all public items so the public API remains unchanged.

// Core traits
pub use traits::{ComplexScalar, Field, Real, Scalar};

// Complex utilities
pub use complex::{
    C32, C64, ComplexExt, I32, I64, ToComplex, c32, c64, from_polar, from_polar32, imag, imag_unit,
    imag_unit32, imag32, real, real32,
};

// Batch operations and classification
pub use batch::{
    ExtendedPrecision, HasFastFma, KBKSum, KahanSum, ScalarBatch, ScalarClass, ScalarClassify,
    SimdCompatible, UnrollHints, pairwise_sum,
};

// Extended precision types
#[cfg(feature = "f16")]
pub use half::f16;

#[cfg(feature = "f128")]
pub use extended::QuadFloat;
