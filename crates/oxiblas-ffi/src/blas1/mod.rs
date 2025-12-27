//! BLAS Level 1 FFI - Vector-Vector operations.
//!
//! Split into modules for maintainability:
//! - `real`: Real-valued operations (f32, f64)
//! - `complex`: Complex-valued operations (Complex32, Complex64)

pub mod complex;
pub mod real;

// Re-export all functions
pub use complex::*;
pub use real::*;
