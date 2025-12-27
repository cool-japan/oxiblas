//! BLAS Level 2 FFI - Matrix-Vector operations.
//!
//! Split into modules by operation type:
//! - `general`: General matrix operations (GEMV, TRSV, GER, HEMV, HER)
//! - `symmetric_triangular`: Symmetric and triangular operations (SYMV, SYR, TRMV)
//! - `banded`: Banded matrix operations (GBMV, SBMV, HBMV, TBMV, TBSV)
//! - `packed`: Packed storage operations (SPMV, HPMV, TPMV, TPSV)
//! - `helpers`: Internal helper functions

pub mod banded;
pub mod general;
pub(crate) mod helpers;
pub mod packed;
pub mod symmetric_triangular;

// Re-export all public functions
pub use banded::*;
pub use general::*;
pub use packed::*;
pub use symmetric_triangular::*;
