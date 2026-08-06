//! CBLAS-compatible interface for BLAS-TESTER compatibility.
//!
//! Split into modules:
//! - `types`: Common CBLAS types and enums
//! - `basic`: Level 1, 2, 3 basic operations
//! - `triangular_symmetric`: Triangular and symmetric Level 3 operations
//! - `hermitian`: Hermitian Level 3 operations (hemm/herk/her2k)
//! - `level2_real`: Real Level 2 operations (symv/syr/syr2/trmv/trsv/ger)
//! - `complex_level1`: Complex Level 1 operations (scal/axpy/copy/swap/nrm2/asum/iamax)
//! - `complex_level2`: Complex Level 2 operations (gemv/hemv)
//! - `validate`: shared parameter validation for every C-ABI entry point
//!   (crate-internal; see its module docs for the xerbla-then-return contract)

pub mod basic;
pub mod complex_level1;
pub mod complex_level2;
pub mod hermitian;
pub mod level2_real;
pub mod triangular_symmetric;
pub mod types;
mod validate;

// Re-export everything
pub use basic::*;
pub use complex_level1::*;
pub use complex_level2::*;
pub use hermitian::*;
pub use level2_real::*;
pub use triangular_symmetric::*;
pub use types::*;
