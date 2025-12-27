//! C FFI bindings for OxiBLAS.
//!
//! This module provides C-compatible functions for BLAS and LAPACK operations.
//!
//! # Usage
//!
//! Link against the generated shared library (liboblas.so, liboblas.dylib, or oblas.dll)
//! and include the provided C header file.
//!
//! # Safety
//!
//! All functions in this module are `unsafe` as they work with raw pointers from C code.
//! The caller is responsible for ensuring:
//! - All pointers are valid and properly aligned
//! - Array dimensions are correct
//! - Memory is not aliased inappropriately

pub mod blas1;
pub mod blas2;
pub mod blas3;
pub mod lapack;
pub mod types;

// Re-export common types
pub use types::*;
