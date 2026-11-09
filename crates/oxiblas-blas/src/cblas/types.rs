//! CBLAS-compatible C ABI: the shared enums.
//!
//! This module defines the `#[repr(C)]` enums taken by the `cblas_*` entry
//! points, with the discriminants of `cblas.h`. The entry points follow the
//! semantics of reference CBLAS for valid calls. Because these are Rust enums,
//! a value outside the set defined here is undefined behaviour rather than an
//! error exit; there is no `cblas_xerbla` to report it to (see
//! [`crate::cblas`] for the other deviations from reference CBLAS).
//!
//! # Layout
//!
//! CBLAS supports both row-major and column-major layouts. This library
//! uses column-major (Fortran) layout internally, so row-major operations
//! are converted using the identity: op(A) in row-major = op(A^T) in column-major.
//!
//! # Performance
//!
//! This module only defines the shared CBLAS enums. The wrappers that use them
//! validate their arguments (invalid input is a silent no-op), map row-major
//! requests onto the column-major natives and call the native kernels of
//! [`crate::level1`], [`crate::level2`] and [`crate::level3`], gathering and
//! scattering strided vectors where a kernel needs contiguous input. A few
//! simple routines (real `gemv`, the complex dot products and `gemm`, Level 1
//! calls with a non-unit increment) run reference-order scalar loops in the
//! wrapper instead; see the individual modules.

/// CBLAS layout ordering.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasLayout {
    /// Row-major ordering (C-style).
    RowMajor = 101,
    /// Column-major ordering (Fortran-style).
    ColMajor = 102,
}

/// CBLAS transpose operation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasTranspose {
    /// No transpose.
    NoTrans = 111,
    /// Transpose.
    Trans = 112,
    /// Conjugate transpose.
    ConjTrans = 113,
}

/// CBLAS triangular type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasUplo {
    /// Upper triangular.
    Upper = 121,
    /// Lower triangular.
    Lower = 122,
}

/// CBLAS diagonal type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasDiag {
    /// Non-unit diagonal.
    NonUnit = 131,
    /// Unit diagonal.
    Unit = 132,
}

/// CBLAS side for operations like TRSM.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasSide {
    /// Left side.
    Left = 141,
    /// Right side.
    Right = 142,
}
