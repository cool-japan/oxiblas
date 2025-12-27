//! C-compatible types for FFI.

use std::ffi::c_int;

/// Complex single-precision type (C99 float _Complex compatible).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OblasComplex32 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

/// Complex double-precision type (C99 double _Complex compatible).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OblasComplex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

/// Matrix layout/ordering.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OblasLayout {
    /// Row-major ordering (C-style).
    RowMajor = 101,
    /// Column-major ordering (Fortran-style).
    ColMajor = 102,
}

/// Transpose operation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OblasTranspose {
    /// No transpose.
    NoTrans = 111,
    /// Transpose.
    Trans = 112,
    /// Conjugate transpose.
    ConjTrans = 113,
}

/// Upper/lower triangular.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OblasUplo {
    /// Upper triangular.
    Upper = 121,
    /// Lower triangular.
    Lower = 122,
}

/// Diagonal type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OblasDiag {
    /// Non-unit diagonal.
    NonUnit = 131,
    /// Unit diagonal.
    Unit = 132,
}

/// Side (left or right).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OblasSide {
    /// Left side.
    Left = 141,
    /// Right side.
    Right = 142,
}

/// Return code.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OblasReturn {
    /// Success.
    Success = 0,
    /// Invalid argument.
    InvalidArg = -1,
    /// Singular matrix.
    Singular = -2,
    /// Not converged.
    NotConverged = -3,
    /// Memory allocation failed.
    MemoryError = -4,
    /// Matrix is not positive definite.
    NotPosdef = -5,
}

impl From<OblasComplex32> for num_complex::Complex32 {
    fn from(c: OblasComplex32) -> Self {
        num_complex::Complex32::new(c.re, c.im)
    }
}

impl From<num_complex::Complex32> for OblasComplex32 {
    fn from(c: num_complex::Complex32) -> Self {
        OblasComplex32 { re: c.re, im: c.im }
    }
}

impl From<OblasComplex64> for num_complex::Complex64 {
    fn from(c: OblasComplex64) -> Self {
        num_complex::Complex64::new(c.re, c.im)
    }
}

impl From<num_complex::Complex64> for OblasComplex64 {
    fn from(c: num_complex::Complex64) -> Self {
        OblasComplex64 { re: c.re, im: c.im }
    }
}

// Type aliases for standard BLAS naming convention
/// Single-precision integer (for indices).
pub type BlasInt = c_int;

/// Index type (for array indexing).
pub type BlasIndex = usize;

/// Helper to convert C layout to Rust bool (true = row major).
#[inline]
pub fn layout_is_row_major(layout: OblasLayout) -> bool {
    layout == OblasLayout::RowMajor
}

/// Helper to get transpose mode.
#[inline]
pub fn get_transpose(trans: OblasTranspose) -> bool {
    trans != OblasTranspose::NoTrans
}
