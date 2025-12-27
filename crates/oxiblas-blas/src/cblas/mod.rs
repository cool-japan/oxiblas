//! CBLAS-compatible interface for BLAS-TESTER compatibility.
//!
//! Split into modules:
//! - `types`: Common CBLAS types and enums
//! - `basic`: Level 1, 2, 3 basic operations
//! - `triangular_symmetric`: Triangular and symmetric Level 3 operations

pub mod basic;
pub mod triangular_symmetric;
pub mod types;

// Re-export everything
pub use basic::*;
pub use triangular_symmetric::*;
pub use types::*;
