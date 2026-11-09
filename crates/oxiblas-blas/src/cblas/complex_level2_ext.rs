//! CBLAS complex dense Level-2 routines (issue #14): `cblas_?geru`,
//! `cblas_?gerc`, `cblas_?her`, `cblas_?her2`, `cblas_?trmv`, `cblas_?trsv`
//! for the `c` / `z` precisions (the real `?ger` / `?trmv` / `?trsv` live in
//! `level2_real`, complex `?gemv` / `?hemv` in `complex_level2`).
//!
//! Thin C-ABI wrappers over the native kernels of [`crate::level2`] (`ger`,
//! `gerc`, `her`, `her2`, `trmv`, `trsv_in_place`). As in `level2_real.rs`,
//! the wrappers own only the CBLAS marshaling: `layout` and enum mapping,
//! strided (including negative) increments via gather/scatter, and the
//! reference quick returns.
//!
//! # Row-major layout
//!
//! A row-major buffer read column-major (same `lda`) is `A^T`. Following
//! reference CBLAS, no element is moved:
//!
//! * GERU: `A^T += alpha*y*x^T`, i.e. `ger(alpha, y, x)` on the `n x m` view.
//! * GERC: `A^T += alpha*conj(y)*x^T`, i.e. `ger(alpha, conj(y), x)` on the
//!   `n x m` view (reference `cblas_cgerc.c` calls `cgeru` on a conjugated
//!   copy of `y`).
//! * HER: a Hermitian `A^T` is `conj(A)` in the opposite triangle, so `uplo`
//!   flips and `x` is conjugated: `conj(A) += alpha*conj(x)*conj(x)^H`.
//! * HER2: `uplo` flips and the vectors are swapped *and* conjugated —
//!   `her2(alpha, conj(y), conj(x))` with `alpha` unchanged, as in reference
//!   `cblas_cher2.c`.
//! * TRMV / TRSV: `uplo` flips and `op` toggles; `ConjTrans` conjugates `x`
//!   around a `NoTrans` call, as in reference `cblas_ctrmv.c`.
//!
//! # Argument handling
//!
//! Every routine validates before touching memory: negative dimensions,
//! `lda < max(1, m)` (`ColMajor` GERU / GERC; `max(1, n)` for `RowMajor`),
//! `lda < max(1, n)` (HER, HER2, TRMV, TRSV), zero increments and null
//! pointers are a silent no-op (reference BLAS calls `xerbla`; a `void` C ABI
//! cannot report it, and a panic reaching the `extern "C"` boundary would abort
//! the process); `m == 0` / `n == 0` return immediately. The rank updates
//! return with `A` untouched when `alpha == 0`. Otherwise, like reference BLAS,
//! they skip the column of an exactly zero scaling element, so an Inf/NaN in
//! the other operand cannot reach it: GERU / GERC skip the column of a zero
//! `y(j)` (in `RowMajor`, with the operands swapped, the row of a zero `x(i)`),
//! HER skips `x(j) == 0` and HER2 `x(j) == y(j) == 0`. HER / HER2 also set the
//! imaginary part of every diagonal element of the referenced triangle to
//! zero, a skipped column's included.
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
//! * The TRMV / TRSV kernels do not reproduce reference BLAS's per-column
//!   "skip when `x(j) == 0`" branch of the `NoTrans` forms, so an Inf/NaN in
//!   `A` facing a zero `x(j)` may reach the result where reference BLAS skips
//!   it. Finite results agree to rounding.

use super::strided::{gather, hermitian_view, scatter, triangular_view};
use super::types::*;
use super::validate::{gemv_params_valid, inc_valid, square_lda_valid};
use crate::level2;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

// =============================================================================
// GERU — A := alpha*x*y^T + A
// =============================================================================

/// Complex single precision GERU: `A := alpha*x*y^T + A` for an `m x n`
/// matrix `A` (unconjugated rank-1 update).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimension, increments) are validated before any pointer is dereferenced,
/// but — like every BLAS/LAPACK ABI — the raw pointers themselves are
/// trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex32`, and
///   readable at every offset of the strided walk of `m` / `n` elements
///   implied by `incx` / `incy` (a negative increment walks back-to-front
///   from the far end rather than out of bounds).
/// - `a` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every `lda`-strided offset of the `m x n` matrix described
///   by `layout` (the validation checks `lda` for consistency but cannot
///   verify the allocation size).
/// - `a` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cgeru(
    layout: CblasLayout,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    a: *mut Complex32,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !gemv_params_valid(layout, m, n, lda) {
        return;
    }
    if !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || a.is_null() {
        return;
    }
    ger_impl(
        layout,
        m as usize,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        a,
        lda as usize,
        false,
    );
}

/// Complex double precision GERU: `A := alpha*x*y^T + A` for an `m x n`
/// matrix `A` (unconjugated rank-1 update).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimension, increments) are validated before any pointer is dereferenced,
/// but — like every BLAS/LAPACK ABI — the raw pointers themselves are
/// trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex64`, and
///   readable at every offset of the strided walk of `m` / `n` elements
///   implied by `incx` / `incy` (a negative increment walks back-to-front
///   from the far end rather than out of bounds).
/// - `a` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every `lda`-strided offset of the `m x n` matrix described
///   by `layout` (the validation checks `lda` for consistency but cannot
///   verify the allocation size).
/// - `a` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zgeru(
    layout: CblasLayout,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    a: *mut Complex64,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !gemv_params_valid(layout, m, n, lda) {
        return;
    }
    if !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || a.is_null() {
        return;
    }
    ger_impl(
        layout,
        m as usize,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        a,
        lda as usize,
        false,
    );
}

// =============================================================================
// GERC — A := alpha*x*y^H + A
// =============================================================================

/// Complex single precision GERC: `A := alpha*x*y^H + A` for an `m x n`
/// matrix `A` (conjugated rank-1 update).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimension, increments) are validated before any pointer is dereferenced,
/// but — like every BLAS/LAPACK ABI — the raw pointers themselves are
/// trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex32`, and
///   readable at every offset of the strided walk of `m` / `n` elements
///   implied by `incx` / `incy` (a negative increment walks back-to-front
///   from the far end rather than out of bounds).
/// - `a` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every `lda`-strided offset of the `m x n` matrix described
///   by `layout` (the validation checks `lda` for consistency but cannot
///   verify the allocation size).
/// - `a` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cgerc(
    layout: CblasLayout,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    a: *mut Complex32,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !gemv_params_valid(layout, m, n, lda) {
        return;
    }
    if !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || a.is_null() {
        return;
    }
    ger_impl(
        layout,
        m as usize,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        a,
        lda as usize,
        true,
    );
}

/// Complex double precision GERC: `A := alpha*x*y^H + A` for an `m x n`
/// matrix `A` (conjugated rank-1 update).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimension, increments) are validated before any pointer is dereferenced,
/// but — like every BLAS/LAPACK ABI — the raw pointers themselves are
/// trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex64`, and
///   readable at every offset of the strided walk of `m` / `n` elements
///   implied by `incx` / `incy` (a negative increment walks back-to-front
///   from the far end rather than out of bounds).
/// - `a` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every `lda`-strided offset of the `m x n` matrix described
///   by `layout` (the validation checks `lda` for consistency but cannot
///   verify the allocation size).
/// - `a` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zgerc(
    layout: CblasLayout,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    a: *mut Complex64,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `m == 0`
    // or `n == 0` is the reference quick return.
    if m <= 0 || n <= 0 || !gemv_params_valid(layout, m, n, lda) {
        return;
    }
    if !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || a.is_null() {
        return;
    }
    ger_impl(
        layout,
        m as usize,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        a,
        lda as usize,
        true,
    );
}

/// Shared GERU (`conj_y == false`) / GERC (`conj_y == true`) body; the
/// arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn ger_impl<T: Field>(
    layout: CblasLayout,
    m: usize,
    n: usize,
    alpha: T,
    x: *const T,
    incx: isize,
    y: *const T,
    incy: isize,
    a: *mut T,
    lda: usize,
    conj_y: bool,
) {
    // Reference quick return: `A` is not touched for `alpha == 0`.
    if alpha == T::zero() {
        return;
    }
    // SAFETY: `x` spans `m` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, m, incx, false) };
    match layout {
        CblasLayout::ColMajor => {
            // SAFETY: `y` spans `n` strided elements per the caller contract.
            let y_vec = unsafe { gather(y, n, incy, false) };
            // SAFETY: `a` is an exclusively borrowed `m x n` matrix with
            // `lda >= max(1, m)` (validated).
            let a_mut = unsafe { MatMut::new(a, m, n, lda) };
            if conj_y {
                level2::gerc(alpha, &x_vec, &y_vec, a_mut);
            } else {
                level2::ger(alpha, &x_vec, &y_vec, a_mut);
            }
        }
        CblasLayout::RowMajor => {
            // The buffer read column-major is `A^T` (`n x m`):
            // `A^T += alpha*op(y)*x^T` with `op(y) = conj(y)` for GERC.
            // SAFETY: `y` spans `n` strided elements per the caller contract.
            let y_vec = unsafe { gather(y, n, incy, conj_y) };
            // SAFETY: `a` is an exclusively borrowed row-major `m x n`
            // matrix with `lda >= max(1, n)` (validated), i.e. an `n x m`
            // column-major one.
            let a_mut = unsafe { MatMut::new(a, n, m, lda) };
            level2::ger(alpha, &y_vec, &x_vec, a_mut);
        }
    }
}

// =============================================================================
// HER — A := alpha*x*x^H + A, A Hermitian, alpha real
// =============================================================================

/// Complex single precision HER: `A := alpha*x*x^H + A` (real `alpha`) for an
/// `n x n` Hermitian matrix `A` of which only the `uplo` triangle is
/// referenced. When `alpha != 0` the imaginary parts of the diagonal are set
/// to zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increment, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null, properly aligned for `Complex32`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `a` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every `lda`-strided offset of the `n x n` matrix (the
///   validation checks `lda >= max(1, n)` but cannot verify the allocation
///   size).
/// - `a` must not overlap `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cher(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    x: *const Complex32,
    incx: i32,
    a: *mut Complex32,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) {
        return;
    }
    if x.is_null() || a.is_null() {
        return;
    }
    her_impl(
        layout,
        uplo,
        n as usize,
        alpha,
        x,
        incx as isize,
        a,
        lda as usize,
    );
}

/// Complex double precision HER: `A := alpha*x*x^H + A` (real `alpha`) for an
/// `n x n` Hermitian matrix `A` of which only the `uplo` triangle is
/// referenced. When `alpha != 0` the imaginary parts of the diagonal are set
/// to zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increment, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null, properly aligned for `Complex64`, and readable at
///   every offset of the strided walk of `n` elements implied by `incx` (a
///   negative `incx` walks back-to-front from the far end rather than out of
///   bounds).
/// - `a` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every `lda`-strided offset of the `n x n` matrix (the
///   validation checks `lda >= max(1, n)` but cannot verify the allocation
///   size).
/// - `a` must not overlap `x`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zher(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    x: *const Complex64,
    incx: i32,
    a: *mut Complex64,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) {
        return;
    }
    if x.is_null() || a.is_null() {
        return;
    }
    her_impl(
        layout,
        uplo,
        n as usize,
        alpha,
        x,
        incx as isize,
        a,
        lda as usize,
    );
}

/// Shared HER body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn her_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T::Real,
    x: *const T,
    incx: isize,
    a: *mut T,
    lda: usize,
) {
    // Reference quick return: `A` (including any imaginary diagonal part) is
    // not touched for `alpha == 0`.
    if alpha.is_zero() {
        return;
    }
    // Row-major: `conj(A) += alpha*conj(x)*conj(x)^H` in the opposite
    // triangle.
    let (uplo, conj) = hermitian_view(layout, uplo);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `a` is an exclusively borrowed `n x n` matrix with
    // `lda >= max(1, n)` (validated); either layout is square.
    let a_mut = unsafe { MatMut::new(a, n, n, lda) };
    let _ = level2::her(her_uplo(uplo), alpha, &x_vec, a_mut);
}

// =============================================================================
// HER2 — A := alpha*x*y^H + conj(alpha)*y*x^H + A, A Hermitian
// =============================================================================

/// Complex single precision HER2: `A := alpha*x*y^H + conj(alpha)*y*x^H + A`
/// for an `n x n` Hermitian matrix `A` of which only the `uplo` triangle is
/// referenced. When `alpha != 0` the imaginary parts of the diagonal are set
/// to zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increments, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex32`, and
///   readable at every offset of the strided walk of `n` elements implied by
///   `incx` / `incy` (a negative increment walks back-to-front from the far
///   end rather than out of bounds).
/// - `a` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every `lda`-strided offset of the `n x n` matrix (the
///   validation checks `lda >= max(1, n)` but cannot verify the allocation
///   size).
/// - `a` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cher2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    a: *mut Complex32,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || a.is_null() {
        return;
    }
    her2_impl(
        layout,
        uplo,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        a,
        lda as usize,
    );
}

/// Complex double precision HER2: `A := alpha*x*y^H + conj(alpha)*y*x^H + A`
/// for an `n x n` Hermitian matrix `A` of which only the `uplo` triangle is
/// referenced. When `alpha != 0` the imaginary parts of the diagonal are set
/// to zero.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increments, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `x` / `y` must be non-null, properly aligned for `Complex64`, and
///   readable at every offset of the strided walk of `n` elements implied by
///   `incx` / `incy` (a negative increment walks back-to-front from the far
///   end rather than out of bounds).
/// - `a` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every `lda`-strided offset of the `n x n` matrix (the
///   validation checks `lda >= max(1, n)` but cannot verify the allocation
///   size).
/// - `a` must not overlap `alpha`, `x` or `y`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zher2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    a: *mut Complex64,
    lda: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if alpha.is_null() || x.is_null() || y.is_null() || a.is_null() {
        return;
    }
    her2_impl(
        layout,
        uplo,
        n as usize,
        *alpha,
        x,
        incx as isize,
        y,
        incy as isize,
        a,
        lda as usize,
    );
}

/// Shared HER2 body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn her2_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: T,
    x: *const T,
    incx: isize,
    y: *const T,
    incy: isize,
    a: *mut T,
    lda: usize,
) {
    // Reference quick return: `A` (including any imaginary diagonal part) is
    // not touched for `alpha == 0`.
    if alpha == T::zero() {
        return;
    }
    // Row-major (reference `cblas_cher2.c`): opposite triangle, operands
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
    // SAFETY: `a` is an exclusively borrowed `n x n` matrix with
    // `lda >= max(1, n)` (validated); either layout is square.
    let a_mut = unsafe { MatMut::new(a, n, n, lda) };
    let _ = level2::her2(her2_uplo(uplo), alpha, first, second, a_mut);
}

// =============================================================================
// TRMV — x := op(A)*x, A triangular
// =============================================================================

/// Complex single precision TRMV: `x := op(A)*x` for an `n x n` triangular
/// matrix `A` (only the `uplo` triangle is referenced).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increment, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex32`, and readable at
///   every `lda`-strided offset of the `n x n` matrix (the validation checks
///   `lda >= max(1, n)` but cannot verify the allocation size).
/// - `x` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctrmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const Complex32,
    lda: i32,
    x: *mut Complex32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    trmv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Complex double precision TRMV: `x := op(A)*x` for an `n x n` triangular
/// matrix `A` (only the `uplo` triangle is referenced).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increment, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex64`, and readable at
///   every `lda`-strided offset of the `n x n` matrix (the validation checks
///   `lda >= max(1, n)` but cannot verify the allocation size).
/// - `x` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztrmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const Complex64,
    lda: i32,
    x: *mut Complex64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    trmv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Shared TRMV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn trmv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: usize,
    a: *const T,
    lda: usize,
    x: *mut T,
    incx: isize,
) {
    let (uplo, trans, conj) = triangular_view(layout, uplo, trans);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let mut x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `a` is an `n x n` matrix with `lda >= max(1, n)` (validated).
    let a_ref = unsafe { MatRef::new(a, n, n, lda) };
    let _ = level2::trmv(
        a_ref,
        &mut x_vec,
        trmv_uplo(uplo),
        trmv_op(trans),
        trmv_diag(diag),
    );
    // SAFETY: same strided walk of `x` as the gather above.
    unsafe { scatter(x, &x_vec, incx, conj) };
}

// =============================================================================
// TRSV — solve op(A)*x = b (b overwritten by x), A triangular
// =============================================================================

/// Complex single precision TRSV: solves `op(A)*x = b` for an `n x n`
/// triangular matrix `A` (only the `uplo` triangle is referenced); `b` is
/// overwritten by `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increment, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex32`, and readable at
///   every `lda`-strided offset of the `n x n` matrix (the validation checks
///   `lda >= max(1, n)` but cannot verify the allocation size).
/// - `x` must be non-null, properly aligned for `Complex32`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctrsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const Complex32,
    lda: i32,
    x: *mut Complex32,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    trsv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Complex double precision TRSV: solves `op(A)*x = b` for an `n x n`
/// triangular matrix `A` (only the `uplo` triangle is referenced); `b` is
/// overwritten by `x`. A singular `A` stops the solve; see the module docs.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimension, leading
/// dimension, increment, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex64`, and readable at
///   every `lda`-strided offset of the `n x n` matrix (the validation checks
///   `lda >= max(1, n)` but cannot verify the allocation size).
/// - `x` must be non-null, properly aligned for `Complex64`, and readable and
///   writable at every offset of the strided walk of `n` elements implied by
///   `incx` (a negative `incx` walks back-to-front from the far end rather
///   than out of bounds).
/// - `x` must not overlap `a`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztrsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const Complex64,
    lda: i32,
    x: *mut Complex64,
    incx: i32,
) {
    // Invalid arguments are a silent no-op (reference BLAS: xerbla); `n == 0`
    // is the reference quick return.
    if n <= 0 || !square_lda_valid(n, lda) || !inc_valid(incx) {
        return;
    }
    if a.is_null() || x.is_null() {
        return;
    }
    trsv_impl(
        layout,
        uplo,
        trans,
        diag,
        n as usize,
        a,
        lda as usize,
        x,
        incx as isize,
    );
}

/// Shared TRSV body; the arguments are validated and non-null.
#[allow(clippy::too_many_arguments)]
unsafe fn trsv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: usize,
    a: *const T,
    lda: usize,
    x: *mut T,
    incx: isize,
) {
    let (uplo, trans, conj) = triangular_view(layout, uplo, trans);
    // SAFETY: `x` spans `n` strided elements per the caller contract.
    let mut x_vec = unsafe { gather(x, n, incx, conj) };
    // SAFETY: `a` is an `n x n` matrix with `lda >= max(1, n)` (validated).
    let a_ref = unsafe { MatRef::new(a, n, n, lda) };
    // The only possible error is an exactly zero `NonUnit` diagonal; as in
    // the real `cblas_?trsv`, the partially solved vector is written back
    // (see the module docs).
    let _ = level2::trsv_in_place(
        a_ref,
        &mut x_vec,
        trsv_mode(uplo),
        trsv_trans(trans),
        trsv_diag(diag),
    );
    // SAFETY: same strided walk of `x` as the gather above.
    unsafe { scatter(x, &x_vec, incx, conj) };
}

// =============================================================================
// Enum mapping
// =============================================================================

#[inline]
fn her_uplo(uplo: CblasUplo) -> level2::HerUplo {
    match uplo {
        CblasUplo::Upper => level2::HerUplo::Upper,
        CblasUplo::Lower => level2::HerUplo::Lower,
    }
}

#[inline]
fn her2_uplo(uplo: CblasUplo) -> level2::Her2Uplo {
    match uplo {
        CblasUplo::Upper => level2::Her2Uplo::Upper,
        CblasUplo::Lower => level2::Her2Uplo::Lower,
    }
}

#[inline]
fn trmv_uplo(uplo: CblasUplo) -> level2::TrmvUplo {
    match uplo {
        CblasUplo::Upper => level2::TrmvUplo::Upper,
        CblasUplo::Lower => level2::TrmvUplo::Lower,
    }
}

#[inline]
fn trmv_op(trans: CblasTranspose) -> level2::TrmvOp {
    match trans {
        CblasTranspose::NoTrans => level2::TrmvOp::NoTrans,
        CblasTranspose::Trans => level2::TrmvOp::Trans,
        CblasTranspose::ConjTrans => level2::TrmvOp::ConjTrans,
    }
}

#[inline]
fn trmv_diag(diag: CblasDiag) -> level2::DiagKind {
    match diag {
        CblasDiag::NonUnit => level2::DiagKind::NonUnit,
        CblasDiag::Unit => level2::DiagKind::Unit,
    }
}

#[inline]
fn trsv_mode(uplo: CblasUplo) -> level2::TriangularMode {
    match uplo {
        CblasUplo::Upper => level2::TriangularMode::Upper,
        CblasUplo::Lower => level2::TriangularMode::Lower,
    }
}

#[inline]
fn trsv_trans(trans: CblasTranspose) -> level2::TrsvTrans {
    match trans {
        CblasTranspose::NoTrans => level2::TrsvTrans::NoTrans,
        CblasTranspose::Trans => level2::TrsvTrans::Trans,
        CblasTranspose::ConjTrans => level2::TrsvTrans::ConjTrans,
    }
}

#[inline]
fn trsv_diag(diag: CblasDiag) -> level2::TrsvDiag {
    match diag {
        CblasDiag::NonUnit => level2::TrsvDiag::NonUnit,
        CblasDiag::Unit => level2::TrsvDiag::Unit,
    }
}

#[cfg(test)]
#[path = "complex_level2_ext_tests.rs"]
mod tests;
