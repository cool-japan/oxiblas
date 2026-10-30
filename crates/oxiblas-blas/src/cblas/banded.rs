//! CBLAS band-storage Level-2 routines (issue #14): `cblas_?gbmv`,
//! `cblas_?sbmv`, `cblas_?hbmv`, `cblas_?tbmv`, `cblas_?tbsv`.
//!
//! Thin C-ABI wrappers over the native band kernels of [`crate::level2`]
//! (`gbmv`, `sbmv`, `hbmv`, `tbmv`, `tbsv`). As in `level2_real.rs`, the
//! wrappers own only the marshaling the slice-based column-major kernels do
//! not: CBLAS `layout` and enum mapping, strided (including negative)
//! increments via gather/scatter, and the reference quick returns.
//!
//! # Band storage and row-major layout
//!
//! Column-major band storage keeps column `j` of `A` in column `j` of an
//! `lda`-strided array: `A[i][j]` at `(ku + i - j) + j*lda` (GBMV),
//! `(k + i - j) + j*lda` (Upper) or `(i - j) + j*lda` (Lower). Row-major band
//! storage keeps row `i` of `A` in row `i` of the array: `A[i][j]` at
//! `i*lda + (kl + j - i)` (GBMV), `i*lda + (j - i)` (Upper) or
//! `i*lda + (k + j - i)` (Lower). Read column-major, that is exactly the band
//! storage of `A^T` — so, as in reference CBLAS, no element is ever moved:
//!
//! * GBMV: `A^T` is `n x m` with `kl` / `ku` swapped and `op` toggles
//!   (`NoTrans` <-> `Trans`); `ConjTrans` becomes `NoTrans` on conjugated
//!   data, `conj(y) = conj(alpha)*A^T*conj(x) + conj(beta)*conj(y)`.
//! * SBMV: `A^T = A`, so only `uplo` flips.
//! * HBMV: `A^T = conj(A)` in the opposite triangle, so `uplo` flips and the
//!   whole update is conjugated (`alpha`, `beta`, `x`, `y` in, `y` out), as
//!   for `cblas_?hemv`.
//! * TBMV / TBSV: `uplo` flips and `op` toggles; `ConjTrans` conjugates `x`
//!   around a `NoTrans` call, as in reference `cblas_ctbsv.c`.
//!
//! # Argument handling
//!
//! Every routine validates before touching memory: negative dimensions or
//! band counts, `lda < kl + ku + 1` (`k + 1`) and zero increments are a
//! silent no-op (reference BLAS calls `xerbla`; a `void` C ABI cannot report
//! it and unwinding across it is UB), as are null pointers; `m == 0` /
//! `n == 0` return immediately. The y-update routines (GBMV, SBMV, HBMV) then
//! apply the reference prologue on the caller's `y`: `alpha == 0 && beta ==
//! 1` returns with `y` untouched, `y := beta*y` (zero-filled without being
//! read when `beta == 0`), and `alpha == 0` returns without reading `A` or
//! `x`.
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
//! * The native kernels accumulate row-wise and do not reproduce reference
//!   BLAS's per-column "skip when `x(j) == 0`" branches, so an Inf/NaN in `A`
//!   facing a zero `x(j)` (TBMV / TBSV) may reach the result where reference
//!   BLAS skips it. Finite results agree to rounding.

use super::strided::{
    flip_uplo, gather, hermitian_view, scatter, transpose_view, triangular_view, y_update_prologue,
};
use super::types::*;
use super::validate::{band_lda_valid, inc_valid};
use crate::level2;
use num_complex::{Complex32, Complex64};
use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::MatRef;

// =============================================================================
// GBMV — y := alpha*op(A)*x + beta*y, A general m x n band (kl, ku)
// =============================================================================

/// Single precision GBMV: `y := alpha*op(A)*x + beta*y` for an `m x n` band
/// matrix `A` with `kl` sub- and `ku` super-diagonals.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band counts,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and readable at
///   offset `r + c*lda` for every `r <= kl + ku` and every `c < n`
///   (`ColMajor`) or `c < m` (`RowMajor`); the validation checks
///   `lda >= kl + ku + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f32`, and readable at every
///   offset of the strided walk of `n` (`NoTrans`) or `m` (otherwise)
///   elements implied by `incx` (a negative `incx` walks back-to-front from
///   the far end rather than out of bounds).
/// - `y` must be non-null, properly aligned for `f32`, and readable and
///   writable at every offset of the strided walk of `m` (`NoTrans`) or `n`
///   (otherwise) elements implied by `incy`.
/// - `y` must not overlap `a` or `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sgbmv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !band_lda_valid(kl, ku, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    gbmv_impl(
        layout,
        trans,
        m as usize,
        n as usize,
        kl as usize,
        ku as usize,
        alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        beta,
        y,
        incy as isize,
    );
}

/// Double precision GBMV: `y := alpha*op(A)*x + beta*y` for an `m x n` band
/// matrix `A` with `kl` sub- and `ku` super-diagonals.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band counts,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and readable at
///   offset `r + c*lda` for every `r <= kl + ku` and every `c < n`
///   (`ColMajor`) or `c < m` (`RowMajor`); the validation checks
///   `lda >= kl + ku + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f64`, and readable at every
///   offset of the strided walk of `n` (`NoTrans`) or `m` (otherwise)
///   elements implied by `incx` (a negative `incx` walks back-to-front from
///   the far end rather than out of bounds).
/// - `y` must be non-null, properly aligned for `f64`, and readable and
///   writable at every offset of the strided walk of `m` (`NoTrans`) or `n`
///   (otherwise) elements implied by `incy`.
/// - `y` must not overlap `a` or `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dgbmv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !band_lda_valid(kl, ku, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    gbmv_impl(
        layout,
        trans,
        m as usize,
        n as usize,
        kl as usize,
        ku as usize,
        alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        beta,
        y,
        incy as isize,
    );
}

/// Complex single precision GBMV: `y := alpha*op(A)*x + beta*y` for an
/// `m x n` band matrix `A` with `kl` sub- and `ku` super-diagonals.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band counts,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` must be non-null and each point to one valid,
///   properly aligned, initialized `Complex32` value.
/// - `a` must be non-null, properly aligned for `Complex32`, and readable at
///   offset `r + c*lda` for every `r <= kl + ku` and every `c < n`
///   (`ColMajor`) or `c < m` (`RowMajor`); the validation checks
///   `lda >= kl + ku + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex32`, and readable at
///   every offset of the strided walk of `n` (`NoTrans`) or `m` (otherwise)
///   elements implied by `incx` (a negative `incx` walks back-to-front from
///   the far end rather than out of bounds).
/// - `y` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `m` (`NoTrans`) or `n`
///   (otherwise) elements implied by `incy`.
/// - `y` must not overlap `a`, `x`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cgbmv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !band_lda_valid(kl, ku, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    gbmv_impl(
        layout,
        trans,
        m as usize,
        n as usize,
        kl as usize,
        ku as usize,
        *alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        *beta,
        y,
        incy as isize,
    );
}

/// Complex double precision GBMV: `y := alpha*op(A)*x + beta*y` for an
/// `m x n` band matrix `A` with `kl` sub- and `ku` super-diagonals.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band counts,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` must be non-null and each point to one valid,
///   properly aligned, initialized `Complex64` value.
/// - `a` must be non-null, properly aligned for `Complex64`, and readable at
///   offset `r + c*lda` for every `r <= kl + ku` and every `c < n`
///   (`ColMajor`) or `c < m` (`RowMajor`); the validation checks
///   `lda >= kl + ku + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex64`, and readable at
///   every offset of the strided walk of `n` (`NoTrans`) or `m` (otherwise)
///   elements implied by `incx` (a negative `incx` walks back-to-front from
///   the far end rather than out of bounds).
/// - `y` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `m` (`NoTrans`) or `n`
///   (otherwise) elements implied by `incy`.
/// - `y` must not overlap `a`, `x`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zgbmv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !band_lda_valid(kl, ku, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    gbmv_impl(
        layout,
        trans,
        m as usize,
        n as usize,
        kl as usize,
        ku as usize,
        *alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        *beta,
        y,
        incy as isize,
    );
}

/// Shared GBMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn gbmv_impl<T: Field>(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    x: *const T,
    incx: isize,
    beta: T,
    y: *mut T,
    incy: isize,
) {
    // The vector lengths follow the caller's `op(A)`, whatever the layout.
    let (lenx, leny) = if trans == CblasTranspose::NoTrans {
        (n, m)
    } else {
        (m, n)
    };
    // SAFETY: `y` spans `leny` strided elements per the caller contract.
    if !unsafe { y_update_prologue(alpha, beta, y, leny, incy) } {
        return;
    }
    let (op, conj) = transpose_view(layout, trans);
    // Column-major band shape of the stored array: the caller's `m x n`
    // (kl, ku) matrix, or its `n x m` (ku, kl) transpose for row-major.
    let (rows, cols, sub, sup) = match layout {
        CblasLayout::ColMajor => (m, n, kl, ku),
        CblasLayout::RowMajor => (n, m, ku, kl),
    };
    let alpha = if conj { alpha.conj() } else { alpha };
    // SAFETY: `x` spans `lenx` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, lenx, incx, conj) };
    // SAFETY: `y` spans `leny` strided elements per the caller contract; it
    // was just scaled in place (equal to scaling its conjugate up to the sign
    // of zero imaginary parts, see `y_update_prologue`).
    let mut y_vec = unsafe { gather(y, leny, incy, conj) };
    // SAFETY: `a` holds `cols` band columns of `kl + ku + 1 <= lda` rows
    // (validated), which is every offset the view can address.
    let ab = unsafe { MatRef::new(a, kl + ku + 1, cols, lda) };
    // The shapes are consistent by construction, so the native call cannot
    // fail; beta = 1 because `y` is already scaled.
    let _ = level2::gbmv(
        gbmv_trans(op),
        rows,
        cols,
        sub,
        sup,
        alpha,
        ab,
        &x_vec,
        T::one(),
        &mut y_vec,
    );
    // SAFETY: same strided walk of `y` as the gather above.
    unsafe { scatter(y, &y_vec, incy, conj) };
}

// =============================================================================
// SBMV — y := alpha*A*x + beta*y, A real symmetric band (k)
// =============================================================================

/// Single precision SBMV: `y := alpha*A*x + beta*y` for an `n x n` symmetric
/// band matrix `A` with `k` super-diagonals (only the `uplo` triangle is
/// stored and read).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band count,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f32`, and readable at every
///   offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `f32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `a` or `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ssbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    sbmv_impl(
        layout,
        uplo,
        n as usize,
        k as usize,
        alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        beta,
        y,
        incy as isize,
    );
}

/// Double precision SBMV: `y := alpha*A*x + beta*y` for an `n x n` symmetric
/// band matrix `A` with `k` super-diagonals (only the `uplo` triangle is
/// stored and read).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band count,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f64`, and readable at every
///   offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `f64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `a` or `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    sbmv_impl(
        layout,
        uplo,
        n as usize,
        k as usize,
        alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        beta,
        y,
        incy as isize,
    );
}

/// Shared SBMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn sbmv_impl<T: Real>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    x: *const T,
    incx: isize,
    beta: T,
    y: *mut T,
    incy: isize,
) {
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    if !unsafe { y_update_prologue(alpha, beta, y, n, incy) } {
        return;
    }
    // Row-major band storage read column-major is that of `A^T = A` in the
    // opposite triangle.
    let uplo = match layout {
        CblasLayout::ColMajor => uplo,
        CblasLayout::RowMajor => flip_uplo(uplo),
    };
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, false) };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    let mut y_vec = unsafe { gather(y, n, incy, false) };
    // SAFETY: `a` holds `n` band columns of `k + 1 <= lda` rows (validated).
    let ab = unsafe { MatRef::new(a, k + 1, n, lda) };
    let _ = level2::sbmv(
        sbmv_uplo(uplo),
        n,
        k,
        alpha,
        ab,
        &x_vec,
        T::one(),
        &mut y_vec,
    );
    // SAFETY: same strided walk of `y` as the gather above.
    unsafe { scatter(y, &y_vec, incy, false) };
}

// =============================================================================
// HBMV — y := alpha*A*x + beta*y, A Hermitian band (k)
// =============================================================================

/// Complex single precision HBMV: `y := alpha*A*x + beta*y` for an `n x n`
/// Hermitian band matrix `A` with `k` super-diagonals (only the `uplo`
/// triangle is stored and read; the imaginary parts of the diagonal are
/// assumed zero and not read).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band count,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` must be non-null and each point to one valid,
///   properly aligned, initialized `Complex32` value.
/// - `a` must be non-null, properly aligned for `Complex32`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex32`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `a`, `x`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_chbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    hbmv_impl(
        layout,
        uplo,
        n as usize,
        k as usize,
        *alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        *beta,
        y,
        incy as isize,
    );
}

/// Complex double precision HBMV: `y := alpha*A*x + beta*y` for an `n x n`
/// Hermitian band matrix `A` with `k` super-diagonals (only the `uplo`
/// triangle is stored and read; the imaginary parts of the diagonal are
/// assumed zero and not read).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, band count,
/// leading dimension, increments, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` must be non-null and each point to one valid,
///   properly aligned, initialized `Complex64` value.
/// - `a` must be non-null, properly aligned for `Complex64`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex64`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `y` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incy`.
/// - `y` must not overlap `a`, `x`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zhbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    hbmv_impl(
        layout,
        uplo,
        n as usize,
        k as usize,
        *alpha,
        a,
        lda as usize,
        x,
        incx as isize,
        *beta,
        y,
        incy as isize,
    );
}

/// Shared HBMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn hbmv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    x: *const T,
    incx: isize,
    beta: T,
    y: *mut T,
    incy: isize,
) {
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    if !unsafe { y_update_prologue(alpha, beta, y, n, incy) } {
        return;
    }
    // Row-major: the stored band is that of `conj(A)` in the opposite
    // triangle, so evaluate `conj(y) = conj(alpha)*conj(A)*conj(x) + conj(y)`
    // (`beta` was already applied to `y`, which matches the reference up to the
    // sign of zero imaginary parts, see `y_update_prologue`).
    let (uplo, conj) = hermitian_view(layout, uplo);
    let alpha = if conj { alpha.conj() } else { alpha };
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `y` spans `n` strided elements per the caller contract.
    let mut y_vec = unsafe { gather(y, n, incy, conj) };
    // SAFETY: `a` holds `n` band columns of `k + 1 <= lda` rows (validated).
    let ab = unsafe { MatRef::new(a, k + 1, n, lda) };
    let _ = level2::hbmv(
        hbmv_uplo(uplo),
        n,
        k,
        alpha,
        ab,
        &x_vec,
        T::one(),
        &mut y_vec,
    );
    // SAFETY: same strided walk of `y` as the gather above.
    unsafe { scatter(y, &y_vec, incy, conj) };
}

// =============================================================================
// TBMV — x := op(A)*x, A triangular band (k)
// =============================================================================

/// Single precision TBMV: `x := op(A)*x` for an `n x n` triangular band
/// matrix `A` with `k` off-diagonals in the `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_stbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbmv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Double precision TBMV: `x := op(A)*x` for an `n x n` triangular band
/// matrix `A` with `k` off-diagonals in the `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbmv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Complex single precision TBMV: `x := op(A)*x` for an `n x n` triangular
/// band matrix `A` with `k` off-diagonals in the `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex32`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const Complex32,
    lda: i32,
    x: *mut Complex32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbmv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Complex double precision TBMV: `x := op(A)*x` for an `n x n` triangular
/// band matrix `A` with `k` off-diagonals in the `uplo` triangle.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex64`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztbmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const Complex64,
    lda: i32,
    x: *mut Complex64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbmv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Shared TBMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn tbmv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: usize,
    k: usize,
    a: *const T,
    lda: usize,
    x: *mut T,
    incx: isize,
) {
    let (uplo, trans, conj) = triangular_view(layout, uplo, trans);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let mut x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `a` holds `n` band columns of `k + 1 <= lda` rows (validated).
    let ab = unsafe { MatRef::new(a, k + 1, n, lda) };
    let _ = level2::tbmv(
        tbmv_uplo(uplo),
        tbmv_trans(trans),
        tbmv_diag(diag),
        n,
        k,
        ab,
        &mut x_vec,
    );
    // SAFETY: same strided walk of `x` as the gather above.
    unsafe { scatter(x, &x_vec, incx, conj) };
}

// =============================================================================
// TBSV — solve op(A)*x = b (b overwritten by x), A triangular band (k)
// =============================================================================

/// Single precision TBSV: solves `op(A)*x = b` for an `n x n` triangular band
/// matrix `A` with `k` off-diagonals in the `uplo` triangle; `b` is
/// overwritten by `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_stbsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbsv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Double precision TBSV: solves `op(A)*x = b` for an `n x n` triangular band
/// matrix `A` with `k` off-diagonals in the `uplo` triangle; `b` is
/// overwritten by `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `f64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtbsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbsv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Complex single precision TBSV: solves `op(A)*x = b` for an `n x n`
/// triangular band matrix `A` with `k` off-diagonals in the `uplo` triangle;
/// `b` is overwritten by `x`. A singular `A` stops the solve; see the module
/// docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex32`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctbsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const Complex32,
    lda: i32,
    x: *mut Complex32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbsv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Complex double precision TBSV: solves `op(A)*x = b` for an `n x n`
/// triangular band matrix `A` with `k` off-diagonals in the `uplo` triangle;
/// `b` is overwritten by `x`. A singular `A` stops the solve; see the module
/// docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, band count,
/// leading dimension, increment, enum flags) are validated before any
/// pointer is dereferenced, but — like every BLAS/LAPACK ABI — the raw
/// pointers themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex64`, and readable at
///   offset `r + c*lda` for every `r <= k` and every `c < n`; the validation
///   checks `lda >= k + 1` but cannot verify the allocation size.
/// - `x` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztbsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    k: i32,
    a: *const Complex64,
    lda: i32,
    x: *mut Complex64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !band_lda_valid(k, 0, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    tbsv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        k as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Shared TBSV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn tbsv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: usize,
    k: usize,
    a: *const T,
    lda: usize,
    x: *mut T,
    incx: isize,
) {
    let (uplo, trans, conj) = triangular_view(layout, uplo, trans);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let mut x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `a` holds `n` band columns of `k + 1 <= lda` rows (validated).
    let ab = unsafe { MatRef::new(a, k + 1, n, lda) };
    // The only possible error is an exactly zero `NonUnit` diagonal; as in
    // `cblas_?trsv`, the partially solved vector is written back (see the
    // module docs).
    let _ = level2::tbsv(
        tbsv_uplo(uplo),
        tbsv_trans(trans),
        tbsv_diag(diag),
        n,
        k,
        ab,
        &mut x_vec,
    );
    // SAFETY: same strided walk of `x` as the gather above.
    unsafe { scatter(x, &x_vec, incx, conj) };
}

// =============================================================================
// Enum mapping
// =============================================================================

#[inline]
fn gbmv_trans(trans: CblasTranspose) -> level2::GbmvTrans {
    match trans {
        CblasTranspose::NoTrans => level2::GbmvTrans::NoTrans,
        CblasTranspose::Trans => level2::GbmvTrans::Trans,
        CblasTranspose::ConjTrans => level2::GbmvTrans::ConjTrans,
    }
}

#[inline]
fn sbmv_uplo(uplo: CblasUplo) -> level2::SbmvUplo {
    match uplo {
        CblasUplo::Upper => level2::SbmvUplo::Upper,
        CblasUplo::Lower => level2::SbmvUplo::Lower,
    }
}

#[inline]
fn hbmv_uplo(uplo: CblasUplo) -> level2::HbmvUplo {
    match uplo {
        CblasUplo::Upper => level2::HbmvUplo::Upper,
        CblasUplo::Lower => level2::HbmvUplo::Lower,
    }
}

#[inline]
fn tbmv_uplo(uplo: CblasUplo) -> level2::TbmvUplo {
    match uplo {
        CblasUplo::Upper => level2::TbmvUplo::Upper,
        CblasUplo::Lower => level2::TbmvUplo::Lower,
    }
}

#[inline]
fn tbmv_trans(trans: CblasTranspose) -> level2::TbmvTrans {
    match trans {
        CblasTranspose::NoTrans => level2::TbmvTrans::NoTrans,
        CblasTranspose::Trans => level2::TbmvTrans::Trans,
        CblasTranspose::ConjTrans => level2::TbmvTrans::ConjTrans,
    }
}

#[inline]
fn tbmv_diag(diag: CblasDiag) -> level2::TbmvDiag {
    match diag {
        CblasDiag::NonUnit => level2::TbmvDiag::NonUnit,
        CblasDiag::Unit => level2::TbmvDiag::Unit,
    }
}

#[inline]
fn tbsv_uplo(uplo: CblasUplo) -> level2::TbsvUplo {
    match uplo {
        CblasUplo::Upper => level2::TbsvUplo::Upper,
        CblasUplo::Lower => level2::TbsvUplo::Lower,
    }
}

#[inline]
fn tbsv_trans(trans: CblasTranspose) -> level2::TbsvTrans {
    match trans {
        CblasTranspose::NoTrans => level2::TbsvTrans::NoTrans,
        CblasTranspose::Trans => level2::TbsvTrans::Trans,
        CblasTranspose::ConjTrans => level2::TbsvTrans::ConjTrans,
    }
}

#[inline]
fn tbsv_diag(diag: CblasDiag) -> level2::TbsvDiag {
    match diag {
        CblasDiag::NonUnit => level2::TbsvDiag::NonUnit,
        CblasDiag::Unit => level2::TbsvDiag::Unit,
    }
}

#[cfg(test)]
#[path = "banded_tests.rs"]
mod tests;
