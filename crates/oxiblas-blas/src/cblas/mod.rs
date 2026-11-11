//! CBLAS-compatible C ABI.
//!
//! The `cblas_*` functions are `extern "C"` entry points that follow the
//! semantics of reference CBLAS for valid calls. They are not a drop-in
//! replacement for every part of the reference library:
//!
//! - There is no `cblas_xerbla`. Invalid arguments (negative dimensions,
//!   undersized leading dimensions, null pointers, ...) make the call a silent
//!   no-op instead of raising an error.
//! - The enum arguments (`CblasLayout`, `CblasTranspose`, ...) are Rust
//!   `#[repr(C)]` enums, so a value outside the set defined in `cblas.h` is
//!   undefined behaviour.
//! - The `i?amax` routines return `i32`, whereas `cblas.h` declares their
//!   return type as `CBLAS_INDEX`.
//!
//! Netlib's CBLAS error-exit tests pass out-of-range enum values and expect
//! `cblas_xerbla`, so they are not supported.
//!
//! Split into modules:
//! - `types`: Common CBLAS types and enums
//! - `basic`: Level 1, 2, 3 basic operations
//! - `triangular_symmetric`: Triangular and symmetric Level 3 operations
//! - `hermitian`: Hermitian Level 3 operations (hemm/herk/her2k)
//! - `level2_real`: Real Level 2 operations (symv/syr/syr2/trmv/trsv/ger)
//! - `complex_level1`: Complex Level 1 operations (scal/axpy/copy/swap/nrm2/asum/iamax)
//! - `complex_level2`: Complex Level 2 operations (gemv/hemv)
//! - `level1_rot`: Level-1 rotations and extras (rotg/rotmg/rot/rotm,
//!   crotg/zrotg, csrot/zdrot, sdsdot/dsdot, scabs1/dcabs1)
//! - `level3_ext`: `?gemmtr` (s/d/c/z) and complex symm/syrk/syr2k/trmm/trsm
//!   (c/z)
//! - `banded`: band-storage Level 2 (gbmv/sbmv/hbmv/tbmv/tbsv)
//! - `packed`: packed-storage Level 2 (spmv/spr/spr2/hpmv/hpr/hpr2/tpmv/tpsv)
//! - `complex_level2_ext`: complex Level 2 (geru/gerc/her/her2/trmv/trsv)
//! - `strided`: crate-internal gather/scatter, layout-view and quick-return
//!   helpers
//! - `validate`: shared parameter validation for every C-ABI entry point
//!   (crate-internal; see its module docs for the validate-then-return
//!   contract: invalid arguments and null pointers are a silent no-op, no
//!   `cblas_xerbla` call)

pub mod basic;
pub mod complex_level1;
pub mod complex_level2;
pub mod hermitian;
pub mod level1_rot;
pub mod level2_real;
pub mod level3_ext;
pub mod triangular_symmetric;
pub mod types;
mod validate;

// Issue #14: reference-CBLAS Level-2 entry points (band, packed, complex dense).
pub mod banded;
pub mod complex_level2_ext;
pub mod packed;
mod strided;

// Re-export everything
pub use basic::*;
pub use complex_level1::*;
pub use complex_level2::*;
pub use hermitian::*;
pub use level1_rot::*;
pub use level2_real::*;
pub use level3_ext::*;
pub use triangular_symmetric::*;
pub use types::*;

pub use banded::*;
pub use complex_level2_ext::*;
pub use packed::*;
