//! BLAS Level 3 FFI - Matrix-Matrix operations.
//!
//! Split into modules:
//! - `basic`: Basic operations (GEMM, TRSM, SYMM, HEMM)
//! - `triangular_rank`: Triangular and rank-k operations (TRMM, SYRK, HERK, SYR2K, HER2K)
//! - `complex`: Complex-specific variants (CTRMM, ZTRMM, CSYMM, ZSYMM, CSYRK, ZSYRK, CSYR2K, ZSYR2K)

pub mod basic;
pub mod complex;
pub mod triangular_rank;

// Re-export all functions
pub use basic::*;
pub use complex::*;
pub use triangular_rank::*;
