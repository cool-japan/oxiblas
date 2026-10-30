//! CBLAS packed-storage Level-2 routines (issue #14): `cblas_?spmv`,
//! `cblas_?spr`, `cblas_?spr2`, `cblas_?hpmv`, `cblas_?hpr`, `cblas_?hpr2`,
//! `cblas_?tpmv`, `cblas_?tpsv`.
//!
//! Thin C-ABI wrappers over the native packed kernels of [`crate::level2`]
//! (`spmv`, `spr`, `spr2`, `hpmv`, `hpr`, `hpr2`, `tpmv`, `tpsv`). As in
//! `level2_real.rs`, the wrappers own only the CBLAS marshaling: `layout` and
//! enum mapping, strided (including negative) increments via gather/scatter,
//! and the reference quick returns. The packed array is handed to the kernel
//! as a slice of exactly `n*(n+1)/2` elements.
//!
//! # Packed storage and row-major layout
//!
//! Column-major packing stores the `uplo` triangle column by column:
//! `A[i][j]` at `i + j*(j+1)/2` (Upper, `i <= j`) or `i + j*(2n-j-1)/2`
//! (Lower, `i >= j`). Row-major packing stores it row by row: `A[i][j]` at
//! `j + i*(2n-i-1)/2` (Upper) or `j + i*(i+1)/2` (Lower) — exactly the
//! column-major packing of `A^T` in the opposite triangle. So, as in
//! reference CBLAS, no element is ever moved:
//!
//! * SPMV / SPR / SPR2: `A^T = A`, so only `uplo` flips.
//! * HPMV: `A^T = conj(A)`, so `uplo` flips and the whole update is
//!   conjugated (`alpha`, `beta`, `x`, `y` in, `y` out), as for
//!   `cblas_?hemv`.
//! * HPR: `uplo` flips and `x` is conjugated:
//!   `conj(A) += alpha*conj(x)*conj(x)^H` (`alpha` is real).
//! * HPR2: `uplo` flips and the vectors are swapped *and* conjugated —
//!   `hpr2(alpha, conj(y), conj(x))` with `alpha` unchanged, as in reference
//!   `cblas_chpr2.c`.
//! * TPMV / TPSV: `uplo` flips and `op` toggles; `ConjTrans` conjugates `x`
//!   around a `NoTrans` call, as in reference `cblas_ctpsv.c`.
//!
//! # Argument handling
//!
//! Every routine validates before touching memory: a negative `n`, a zero
//! increment and null pointers are a silent no-op (reference BLAS calls
//! `xerbla`; a `void` C ABI cannot report it and unwinding across it is UB);
//! `n == 0` returns immediately. The packed length `n*(n+1)/2` is formed with
//! checked arithmetic, and a length whose byte size would overflow — which no
//! real buffer can have — is treated as invalid. SPMV / HPMV then apply the
//! reference y-update prologue on the caller's `y` (`alpha == 0 && beta ==
//! 1`: untouched; `y := beta*y`, zero-filled without being read when `beta ==
//! 0`; `alpha == 0`: `AP` and `x` are never read). The rank updates return
//! with `AP` untouched when `alpha == 0`. Otherwise, like reference BLAS,
//! they skip the column of an exactly zero scaling element (SPR / HPR:
//! `x(j) == 0`; SPR2 / HPR2: `x(j) == y(j) == 0`; in `RowMajor` that column
//! is row `j` of the caller's triangle), so an Inf/NaN in the other operand
//! cannot reach it, and HPR / HPR2 set the imaginary part of every diagonal
//! element of `AP` to zero, a skipped column's included. The packed kernels
//! mirror their dense twins operation for operation: SPR / SPR2 give bitwise
//! the results of SYR / SYR2 on the same matrix, and HPR / HPR2 those of
//! HER / HER2 (bitwise for NaN-free input; with NaN input the NaN sign/payload
//! may differ).
//!
//! # Known deviations from reference BLAS
//!
//! * A `NonUnit` triangular solve stops at the first exactly-zero diagonal
//!   entry it reaches and writes back the partially solved `x`. Reference
//!   BLAS never stops. In the `NoTrans` forms it skips the division and the
//!   column update while the current `x(j)` is exactly zero, which gives a
//!   finite solution of a consistent singular system. Otherwise it divides by
//!   zero and propagates Inf/NaN; the `Trans` / `ConjTrans` forms always
//!   divide.
//! * The TPMV / TPSV kernels do not reproduce reference BLAS's per-column
//!   "skip when `x(j) == 0`" branch of the `NoTrans` forms, so an Inf/NaN in
//!   `AP` facing a zero `x(j)` may reach the result where reference BLAS
//!   skips it. Finite results agree to rounding.

use super::strided::{
    flip_uplo, gather, hermitian_view, packed_len, scatter, triangular_view, y_update_prologue,
};
use super::types::*;
use super::validate::inc_valid;
use crate::level2;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use oxiblas_core::scalar::{Field, Real};

// =============================================================================
// SPMV — y := alpha*A*x + beta*y, A real symmetric packed
// =============================================================================

/// Single precision SPMV: `y := alpha*A*x + beta*y` for an `n x n` symmetric
/// matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `ap` must be non-null, properly aligned for `f32`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `f32`, and readable at every
///   offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `f32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `ap` or `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sspmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    ap: *const f32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if ap.is_null() || x.is_null() || y.is_null() {
        return;
    }
    spmv_impl(
        layout,
        uplo,
        n as usize,
        alpha,
        ap,
        x,
        incx as isize,
        beta,
        y,
        incy as isize,
    );
}

/// Double precision SPMV: `y := alpha*A*x + beta*y` for an `n x n` symmetric
/// matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `ap` must be non-null, properly aligned for `f64`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `f64`, and readable at every
///   offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `f64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `ap` or `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dspmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    ap: *const f64,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if ap.is_null() || x.is_null() || y.is_null() {
        return;
    }
    spmv_impl(
        layout,
        uplo,
        n as usize,
        alpha,
        ap,
        x,
        incx as isize,
        beta,
        y,
        incy as isize,
    );
}

/// Shared SPMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn spmv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T,
    ap: *const T,
    x: *const T,
    incx: isize,
    beta: T,
    y: *mut T,
    incy: isize,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    if !unsafe { y_update_prologue(alpha, beta, y, n, incy) } {
        return;
    }
    // Row-major packing is the column-major packing of `A^T = A` in the
    // opposite triangle.
    let uplo = match layout {
        CblasLayout::ColMajor => uplo,
        CblasLayout::RowMajor => flip_uplo(uplo),
    };
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, false) };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    let mut y_vec = unsafe { gather(y, n, incy, false) };
    // SAFETY: `ap` is non-null, aligned and holds `len = n*(n+1)/2`
    // initialized elements per the caller contract, and `len` elements fit
    // in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts(ap, len) };
    let _ = level2::spmv(spmv_uplo(uplo), n, alpha, ap, &x_vec, T::one(), &mut y_vec);
    // SAFETY: same strided walk of `y` as the gather above.
    unsafe { scatter(y, &y_vec, incy, false) };
}

// =============================================================================
// SPR — A := alpha*x*x^T + A, A real symmetric packed
// =============================================================================

/// Single precision SPR: `A := alpha*x*x^T + A` for an `n x n` symmetric
/// matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `x` must be non-null, properly aligned for `f32`, and readable at every
///   offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `ap` must be non-null, properly aligned for `f32`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sspr(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    ap: *mut f32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if x.is_null() || ap.is_null() {
        return;
    }
    spr_impl(layout, uplo, n as usize, alpha, x, incx as isize, ap);
}

/// Double precision SPR: `A := alpha*x*x^T + A` for an `n x n` symmetric
/// matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `x` must be non-null, properly aligned for `f64`, and readable at every
///   offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `ap` must be non-null, properly aligned for `f64`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dspr(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    ap: *mut f64,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if x.is_null() || ap.is_null() {
        return;
    }
    spr_impl(layout, uplo, n as usize, alpha, x, incx as isize, ap);
}

/// Shared SPR body; the arguments are validated and non-null.
unsafe fn spr_impl<T: Real>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T,
    x: *const T,
    incx: isize,
    ap: *mut T,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    // Reference quick return: `AP` is not touched for `alpha == 0`.
    if alpha == T::zero() {
        return;
    }
    let uplo = match layout {
        CblasLayout::ColMajor => uplo,
        CblasLayout::RowMajor => flip_uplo(uplo),
    };
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, false) };
    // SAFETY: `ap` is non-null, aligned, exclusively borrowed and holds
    // `len = n*(n+1)/2` initialized elements per the caller contract, and
    // `len` elements fit in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts_mut(ap, len) };
    let _ = level2::spr(spr_uplo(uplo), n, alpha, &x_vec, ap);
}

// =============================================================================
// SPR2 — A := alpha*x*y^T + alpha*y*x^T + A, A real symmetric packed
// =============================================================================

/// Single precision SPR2: `A := alpha*x*y^T + alpha*y*x^T + A` for an
/// `n x n` symmetric matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `x` / `y` must be non-null, properly aligned for `f32`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` /
///   `incy` (a negative increment walks back-to-front from the far end
///   rather than out of bounds).
/// - `ap` must be non-null, properly aligned for `f32`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sspr2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
    ap: *mut f32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if x.is_null() || y.is_null() || ap.is_null() {
        return;
    }
    spr2_impl(
        layout,
        uplo,
        n as usize,
        alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        ap,
    );
}

/// Double precision SPR2: `A := alpha*x*y^T + alpha*y*x^T + A` for an
/// `n x n` symmetric matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `x` / `y` must be non-null, properly aligned for `f64`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` /
///   `incy` (a negative increment walks back-to-front from the far end
///   rather than out of bounds).
/// - `ap` must be non-null, properly aligned for `f64`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dspr2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    ap: *mut f64,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if x.is_null() || y.is_null() || ap.is_null() {
        return;
    }
    spr2_impl(
        layout,
        uplo,
        n as usize,
        alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        ap,
    );
}

/// Shared SPR2 body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn spr2_impl<T: Real>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T,
    x: *const T,
    incx: isize,
    y: *const T,
    incy: isize,
    ap: *mut T,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    // Reference quick return: `AP` is not touched for `alpha == 0`.
    if alpha == T::zero() {
        return;
    }
    let uplo = match layout {
        CblasLayout::ColMajor => uplo,
        CblasLayout::RowMajor => flip_uplo(uplo),
    };
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, false) };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    let y_vec = unsafe { gather(y, n, incy, false) };
    // SAFETY: `ap` is non-null, aligned, exclusively borrowed and holds
    // `len = n*(n+1)/2` initialized elements per the caller contract, and
    // `len` elements fit in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts_mut(ap, len) };
    let _ = level2::spr2(spr2_uplo(uplo), n, alpha, &x_vec, &y_vec, ap);
}

// =============================================================================
// HPMV — y := alpha*A*x + beta*y, A Hermitian packed
// =============================================================================

/// Complex single precision HPMV: `y := alpha*A*x + beta*y` for an `n x n`
/// Hermitian matrix `A` supplied as its packed `uplo` triangle (the
/// imaginary parts of the diagonal are assumed zero and not read).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `alpha` and `beta` must be non-null and each point to one valid,
///   properly aligned, initialized `Complex32` value.
/// - `ap` must be non-null, properly aligned for `Complex32`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `Complex32`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `ap`, `x`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_chpmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex32,
    ap: *const Complex32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || beta.is_null() || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }
    hpmv_impl(
        layout,
        uplo,
        n as usize,
        *alpha,
        ap,
        x,
        incx as isize,
        *beta,
        y,
        incy as isize,
    );
}

/// Complex double precision HPMV: `y := alpha*A*x + beta*y` for an `n x n`
/// Hermitian matrix `A` supplied as its packed `uplo` triangle (the
/// imaginary parts of the diagonal are assumed zero and not read).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `alpha` and `beta` must be non-null and each point to one valid,
///   properly aligned, initialized `Complex64` value.
/// - `ap` must be non-null, properly aligned for `Complex64`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `Complex64`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `ap`, `x`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zhpmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex64,
    ap: *const Complex64,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || beta.is_null() || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }
    hpmv_impl(
        layout,
        uplo,
        n as usize,
        *alpha,
        ap,
        x,
        incx as isize,
        *beta,
        y,
        incy as isize,
    );
}

/// Shared HPMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn hpmv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T,
    ap: *const T,
    x: *const T,
    incx: isize,
    beta: T,
    y: *mut T,
    incy: isize,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    if !unsafe { y_update_prologue(alpha, beta, y, n, incy) } {
        return;
    }
    // Row-major: the stored triangle is the packing of `conj(A)` in the
    // opposite triangle, so evaluate
    // `conj(y) = conj(alpha)*conj(A)*conj(x) + conj(y)` (`beta` was already
    // applied to `y`, which matches the reference up to the sign of zero
    // imaginary parts, see `y_update_prologue`).
    let (uplo, conj) = hermitian_view(layout, uplo);
    let alpha = if conj { alpha.conj() } else { alpha };
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    let mut y_vec = unsafe { gather(y, n, incy, conj) };
    // SAFETY: `ap` is non-null, aligned and holds `len = n*(n+1)/2`
    // initialized elements per the caller contract, and `len` elements fit
    // in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts(ap, len) };
    let _ = level2::hpmv(hpmv_uplo(uplo), n, alpha, ap, &x_vec, T::one(), &mut y_vec);
    // SAFETY: same strided walk of `y` as the gather above.
    unsafe { scatter(y, &y_vec, incy, conj) };
}

// =============================================================================
// HPR — A := alpha*x*x^H + A, A Hermitian packed, alpha real
// =============================================================================

/// Complex single precision HPR: `A := alpha*x*x^H + A` (real `alpha`) for an
/// `n x n` Hermitian matrix `A` supplied as its packed `uplo` triangle. When
/// `alpha != 0` the imaginary parts of the diagonal are set to zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `x` must be non-null, properly aligned for `Complex32`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `ap` must be non-null, properly aligned for `Complex32`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_chpr(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    x: *const Complex32,
    incx: i32,
    ap: *mut Complex32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if x.is_null() || ap.is_null() {
        return;
    }
    hpr_impl(layout, uplo, n as usize, alpha, x, incx as isize, ap);
}

/// Complex double precision HPR: `A := alpha*x*x^H + A` (real `alpha`) for an
/// `n x n` Hermitian matrix `A` supplied as its packed `uplo` triangle. When
/// `alpha != 0` the imaginary parts of the diagonal are set to zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `x` must be non-null, properly aligned for `Complex64`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `ap` must be non-null, properly aligned for `Complex64`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zhpr(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    x: *const Complex64,
    incx: i32,
    ap: *mut Complex64,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if x.is_null() || ap.is_null() {
        return;
    }
    hpr_impl(layout, uplo, n as usize, alpha, x, incx as isize, ap);
}

/// Shared HPR body; the arguments are validated and non-null.
unsafe fn hpr_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T::Real,
    x: *const T,
    incx: isize,
    ap: *mut T,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    // Reference quick return: `AP` (including any imaginary diagonal part) is
    // not touched for `alpha == 0`.
    if alpha.is_zero() {
        return;
    }
    // Row-major: `conj(A) += alpha*conj(x)*conj(x)^H` in the opposite
    // triangle.
    let (uplo, conj) = hermitian_view(layout, uplo);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `ap` is non-null, aligned, exclusively borrowed and holds
    // `len = n*(n+1)/2` initialized elements per the caller contract, and
    // `len` elements fit in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts_mut(ap, len) };
    let _ = level2::hpr(hpr_uplo(uplo), n, alpha, &x_vec, ap);
}

// =============================================================================
// HPR2 — A := alpha*x*y^H + conj(alpha)*y*x^H + A, A Hermitian packed
// =============================================================================

/// Complex single precision HPR2: `A := alpha*x*y^H + conj(alpha)*y*x^H + A`
/// for an `n x n` Hermitian matrix `A` supplied as its packed `uplo`
/// triangle. When `alpha != 0` the imaginary parts of the diagonal are set to
/// zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex32`, and
///   readable at every offset of the strided walk of `n` elements implied by
///   `incx` / `incy` (a negative increment walks back-to-front from the far
///   end rather than out of bounds).
/// - `ap` must be non-null, properly aligned for `Complex32`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_chpr2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    ap: *mut Complex32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || ap.is_null() {
        return;
    }
    hpr2_impl(
        layout,
        uplo,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        ap,
    );
}

/// Complex double precision HPR2: `A := alpha*x*y^H + conj(alpha)*y*x^H + A`
/// for an `n x n` Hermitian matrix `A` supplied as its packed `uplo`
/// triangle. When `alpha != 0` the imaginary parts of the diagonal are set to
/// zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increments,
/// enum flags) are validated before any pointer is dereferenced, but — like
/// every BLAS/LAPACK ABI — the raw pointers themselves are trusted. The
/// caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex64`, and
///   readable at every offset of the strided walk of `n` elements implied by
///   `incx` / `incy` (a negative increment walks back-to-front from the far
///   end rather than out of bounds).
/// - `ap` must be non-null, properly aligned for `Complex64`, and point to
///   `n*(n+1)/2` readable and writable, initialized elements (the packed
///   triangle).
/// - `ap` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zhpr2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    ap: *mut Complex64,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || ap.is_null() {
        return;
    }
    hpr2_impl(
        layout,
        uplo,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        ap,
    );
}

/// Shared HPR2 body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn hpr2_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T,
    x: *const T,
    incx: isize,
    y: *const T,
    incy: isize,
    ap: *mut T,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    // Reference quick return: `AP` (including any imaginary diagonal part) is
    // not touched for `alpha == 0`.
    if alpha == T::zero() {
        return;
    }
    // Row-major (reference `cblas_chpr2.c`): opposite triangle, operands
    // swapped and both conjugated, `alpha` unchanged —
    // `conj(A) += alpha*conj(y)*conj(x)^H + conj(alpha)*conj(x)*conj(y)^H`.
    let (uplo, conj) = hermitian_view(layout, uplo);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    let y_vec = unsafe { gather(y, n, incy, conj) };
    let (first, second) = if conj {
        (&y_vec, &x_vec)
    } else {
        (&x_vec, &y_vec)
    };
    // SAFETY: `ap` is non-null, aligned, exclusively borrowed and holds
    // `len = n*(n+1)/2` initialized elements per the caller contract, and
    // `len` elements fit in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts_mut(ap, len) };
    let _ = level2::hpr2(hpr2_uplo(uplo), n, alpha, first, second, ap);
}

// =============================================================================
// TPMV — x := op(A)*x, A triangular packed
// =============================================================================

/// Single precision TPMV: `x := op(A)*x` for an `n x n` triangular matrix `A`
/// supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `f32`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `f32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_stpmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const f32,
    x: *mut f32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpmv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Double precision TPMV: `x := op(A)*x` for an `n x n` triangular matrix `A`
/// supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `f64`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `f64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtpmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const f64,
    x: *mut f64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpmv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Complex single precision TPMV: `x := op(A)*x` for an `n x n` triangular
/// matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `Complex32`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctpmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpmv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Complex double precision TPMV: `x := op(A)*x` for an `n x n` triangular
/// matrix `A` supplied as its packed `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `Complex64`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztpmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpmv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Shared TPMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn tpmv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: usize,
    ap: *const T,
    x: *mut T,
    incx: isize,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    let (uplo, trans, conj) = triangular_view(layout, uplo, trans);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let mut x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `ap` is non-null, aligned and holds `len = n*(n+1)/2`
    // initialized elements per the caller contract, and `len` elements fit
    // in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts(ap, len) };
    let _ = level2::tpmv(
        tpmv_uplo(uplo),
        tpmv_trans(trans),
        tpmv_diag(diag),
        n,
        ap,
        &mut x_vec,
    );
    // SAFETY: same strided walk of `x` as the gather above.
    unsafe { scatter(x, &x_vec, incx, conj) };
}

// =============================================================================
// TPSV — solve op(A)*x = b (b overwritten by x), A triangular packed
// =============================================================================

/// Single precision TPSV: solves `op(A)*x = b` for an `n x n` triangular
/// matrix `A` supplied as its packed `uplo` triangle; `b` is overwritten by
/// `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `f32`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `f32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_stpsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const f32,
    x: *mut f32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpsv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Double precision TPSV: solves `op(A)*x = b` for an `n x n` triangular
/// matrix `A` supplied as its packed `uplo` triangle; `b` is overwritten by
/// `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `f64`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `f64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtpsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const f64,
    x: *mut f64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpsv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Complex single precision TPSV: solves `op(A)*x = b` for an `n x n`
/// triangular matrix `A` supplied as its packed `uplo` triangle; `b` is
/// overwritten by `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `Complex32`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctpsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpsv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Complex double precision TPSV: solves `op(A)*x = b` for an `n x n`
/// triangular matrix `A` supplied as its packed `uplo` triangle; `b` is
/// overwritten by `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, increment, enum
/// flags) are validated before any pointer is dereferenced, but — like every
/// BLAS/LAPACK ABI — the raw pointers themselves are trusted. The caller
/// must ensure:
///
/// - `ap` must be non-null, properly aligned for `Complex64`, and point to
///   `n*(n+1)/2` readable, initialized elements (the packed triangle).
/// - `x` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `ap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztpsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !inc_valid(incx) {
        return;
    }
    if ap.is_null() || x.is_null() {
        return;
    }
    tpsv_impl(layout, uplo, trans, diag, n as usize, ap, x, incx as isize);
}

/// Shared TPSV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn tpsv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: usize,
    ap: *const T,
    x: *mut T,
    incx: isize,
) {
    let Some(len) = packed_len::<T>(n) else {
        return;
    };
    let (uplo, trans, conj) = triangular_view(layout, uplo, trans);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let mut x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `ap` is non-null, aligned and holds `len = n*(n+1)/2`
    // initialized elements per the caller contract, and `len` elements fit
    // in `isize::MAX` bytes (`packed_len`).
    let ap = unsafe { core::slice::from_raw_parts(ap, len) };
    // The only possible error is an exactly zero `NonUnit` diagonal; as in
    // `cblas_?trsv`, the partially solved vector is written back (see the
    // module docs).
    let _ = level2::tpsv(
        tpsv_uplo(uplo),
        tpsv_trans(trans),
        tpsv_diag(diag),
        n,
        ap,
        &mut x_vec,
    );
    // SAFETY: same strided walk of `x` as the gather above.
    unsafe { scatter(x, &x_vec, incx, conj) };
}

// =============================================================================
// Enum mapping
// =============================================================================

#[inline]
fn spmv_uplo(uplo: CblasUplo) -> level2::SpmvUplo {
    match uplo {
        CblasUplo::Upper => level2::SpmvUplo::Upper,
        CblasUplo::Lower => level2::SpmvUplo::Lower,
    }
}

#[inline]
fn spr_uplo(uplo: CblasUplo) -> level2::SprUplo {
    match uplo {
        CblasUplo::Upper => level2::SprUplo::Upper,
        CblasUplo::Lower => level2::SprUplo::Lower,
    }
}

#[inline]
fn spr2_uplo(uplo: CblasUplo) -> level2::Spr2Uplo {
    match uplo {
        CblasUplo::Upper => level2::Spr2Uplo::Upper,
        CblasUplo::Lower => level2::Spr2Uplo::Lower,
    }
}

#[inline]
fn hpmv_uplo(uplo: CblasUplo) -> level2::HpmvUplo {
    match uplo {
        CblasUplo::Upper => level2::HpmvUplo::Upper,
        CblasUplo::Lower => level2::HpmvUplo::Lower,
    }
}

#[inline]
fn hpr_uplo(uplo: CblasUplo) -> level2::HprUplo {
    match uplo {
        CblasUplo::Upper => level2::HprUplo::Upper,
        CblasUplo::Lower => level2::HprUplo::Lower,
    }
}

#[inline]
fn hpr2_uplo(uplo: CblasUplo) -> level2::Hpr2Uplo {
    match uplo {
        CblasUplo::Upper => level2::Hpr2Uplo::Upper,
        CblasUplo::Lower => level2::Hpr2Uplo::Lower,
    }
}

#[inline]
fn tpmv_uplo(uplo: CblasUplo) -> level2::TpmvUplo {
    match uplo {
        CblasUplo::Upper => level2::TpmvUplo::Upper,
        CblasUplo::Lower => level2::TpmvUplo::Lower,
    }
}

#[inline]
fn tpmv_trans(trans: CblasTranspose) -> level2::TpmvTrans {
    match trans {
        CblasTranspose::NoTrans => level2::TpmvTrans::NoTrans,
        CblasTranspose::Trans => level2::TpmvTrans::Trans,
        CblasTranspose::ConjTrans => level2::TpmvTrans::ConjTrans,
    }
}

#[inline]
fn tpmv_diag(diag: CblasDiag) -> level2::TpmvDiag {
    match diag {
        CblasDiag::NonUnit => level2::TpmvDiag::NonUnit,
        CblasDiag::Unit => level2::TpmvDiag::Unit,
    }
}

#[inline]
fn tpsv_uplo(uplo: CblasUplo) -> level2::TpsvUplo {
    match uplo {
        CblasUplo::Upper => level2::TpsvUplo::Upper,
        CblasUplo::Lower => level2::TpsvUplo::Lower,
    }
}

#[inline]
fn tpsv_trans(trans: CblasTranspose) -> level2::TpsvTrans {
    match trans {
        CblasTranspose::NoTrans => level2::TpsvTrans::NoTrans,
        CblasTranspose::Trans => level2::TpsvTrans::Trans,
        CblasTranspose::ConjTrans => level2::TpsvTrans::ConjTrans,
    }
}

#[inline]
fn tpsv_diag(diag: CblasDiag) -> level2::TpsvDiag {
    match diag {
        CblasDiag::NonUnit => level2::TpsvDiag::NonUnit,
        CblasDiag::Unit => level2::TpsvDiag::Unit,
    }
}

#[cfg(test)]
#[path = "packed_tests.rs"]
mod tests;
