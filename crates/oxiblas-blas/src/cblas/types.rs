//! CBLAS-compatible interface for BLAS-TESTER compatibility.
//!
//! This module provides a C-compatible interface that follows the standard
//! CBLAS specification, enabling interoperability with BLAS test suites
//! like BLAS-TESTER.
//!
//! # Layout
//!
//! CBLAS supports both row-major and column-major layouts. This library
//! uses column-major (Fortran) layout internally, so row-major operations
//! are converted using the identity: op(A) in row-major = op(A^T) in column-major.
//!
//! # Performance
//!
//! For unit stride vectors (incx=1, incy=1), this module uses the optimized
//! internal BLAS implementations with SIMD acceleration. For non-unit strides,
//! scalar fallbacks are used.

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
