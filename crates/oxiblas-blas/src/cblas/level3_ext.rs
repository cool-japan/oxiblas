//! CBLAS Level-3 entry points completing the reference interface (issue #14):
//! `?gemmtr` (all four precisions) and the complex `symm`/`syrk`/`syr2k`/
//! `trmm`/`trsm` (`c`/`z`).
//!
//! # Layout handling
//!
//! The natives are column-major. A row-major request is mapped onto the
//! column-major view of the same buffers (a row-major `r x c` matrix with
//! leading dimension `ld` *is* its `c x r` transpose in column-major order with
//! the same `ld`), exactly as netlib `cblas_*.c` (3.12) does:
//!
//! - `gemmtr`: flip `uplo`, pass `(B, ldb, transB)` first and
//!   `(A, lda, transA)` second; transposes map one-to-one (`ConjTrans` stays
//!   `ConjTrans`), nothing is conjugated.
//! - `symm`: flip `side` and `uplo`, swap `m`/`n`.
//! - `syrk`/`syr2k`: flip `uplo`, `NoTrans` <-> `Trans`; `A`/`B` keep their
//!   order.
//! - `trmm`/`trsm`: flip `side` and `uplo`, swap `m`/`n` (B is viewed as the
//!   column-major `n x m` matrix with the caller's `ldb`); `transa`, `diag`
//!   and `alpha` are unchanged. The order of `A` comes from the caller's
//!   (unflipped) side: `m` for `Left`, `n` for `Right`.
//!
//! # Complex SYRK/SYR2K and `ConjTrans`
//!
//! These are the *symmetric* updates (`A·A^T`, never `A^H`). Column-major
//! `ConjTrans` is rejected by reference CBLAS: 3.12 forwards it to Fortran as
//! `TRANS = 'C'`, which C/ZSYRK and C/ZSYR2K refuse (Fortran `INFO = 2` →
//! `cblas_xerbla`, which terminates the process). Here it is treated as an
//! invalid argument (silent no-op, C untouched). Row-major `ConjTrans` is
//! mapped to `'N'` exactly like `Trans`, as in reference CBLAS 3.12.0/3.12.1
//! (`cblas_csyrk.c` / `cblas_csyr2k.c` / `cblas_zsyrk.c` / `cblas_zsyr2k.c`).
//!
//! # Quick returns and special values
//!
//! - `n == 0` (and `m == 0` for the `m x n` routines): no-op.
//! - `gemmtr`/`syrk`/`syr2k`: `(alpha == 0 || k == 0) && beta == 1` leaves C
//!   bitwise untouched; otherwise `alpha == 0 || k == 0` performs only
//!   `C := beta·C` on the referenced triangle (A/B are never read, and
//!   `beta == 0` zero-fills without reading C). Reference `?syrk`/`?syr2k` have
//!   this quick return and the `alpha == 0` branch; the `gemmtr` quick return
//!   and the `k == 0` shortcut are deviations (see below).
//! - `symm` (reference semantics): `alpha == 0 && beta == 1` leaves C
//!   untouched; `alpha == 0` performs only `C := beta·C` (A/B are not read;
//!   `beta == 0` does not read C).
//! - `trmm`/`trsm`: `alpha == 0` sets `B := 0` without reading A or B
//!   (reference semantics). `trsm` otherwise scales B by `alpha` (skipped for
//!   `alpha == 1`) and solves in place.
//!
//! Invalid arguments (negative dimensions, undersized leading dimensions, the
//! rejected `ConjTrans` above) and null pointers make the call a silent no-op,
//! the convention of every wrapper in this crate (a C ABI cannot surface an
//! error and a panic reaching it would abort the process). A singular
//! triangular factor in `trsm` makes the native solve stop with an error that
//! is likewise ignored: B is left as the solver left it, and reference ?TRSM
//! makes no promise about B for a singular A either.
//!
//! # Deviations from the reference
//!
//! `?gemmtr` adds the `?gemm`-style quick return
//! `(alpha == 0 || k == 0) && beta == 1`, which ZGEMMTR 3.12.1 lacks (it
//! returns early only for `n == 0`). Together with the `k == 0` →
//! `C := beta·C` shortcut used by `gemmtr`, `syrk` and `syr2k`, this differs
//! from the reference only in three ways:
//!
//! - signed zeros: for `k == 0` the reference's transposed-`A` forms evaluate
//!   `alpha·0 + beta·C`, and for `alpha == 0` complex `gemmtr` evaluates the
//!   complex product `beta·C`, so the sign of a zero in C can differ;
//! - NaN or ±Inf `alpha` with `k == 0`: the reference's `alpha·0` is NaN in its
//!   transposed-`A` forms, whereas the shortcut ignores `alpha`;
//! - Inf components of C in complex `gemmtr` with `beta == 1`: the reference
//!   still evaluates `(1,0)·C` (for `alpha == 0`, and for `k == 0` with a
//!   transposed `A`), which turns Inf components into NaN, while the early
//!   return leaves C untouched.

use super::types::*;
use super::validate::{
    gemm_params_valid, symm_params_valid, syr2k_params_valid, syrk_params_valid, trmm_params_valid,
};
use crate::level3::{self, GemmKernel};
use num_complex::{Complex32, Complex64};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

// =============================================================================
// GEMMTR — C := alpha*op(A)*op(B) + beta*C, only the `uplo` triangle of C
// =============================================================================

/// Single precision GEMMTR: `C := alpha·op(A)·op(B) + beta·C`, updating only
/// the `uplo` triangle of the `n x n` matrix C (`op(A)` is `n x k`, `op(B)` is
/// `k x n`).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` is non-null, properly aligned for `f32`, and readable at every
///   `lda`-strided offset of the stored `A` (`n x k` for `NoTrans`, else
///   `k x n`, in `layout` order).
/// - `b` is non-null, properly aligned for `f32`, and readable at every
///   `ldb`-strided offset of the stored `B` (`k x n` for `NoTrans`, else
///   `n x k`, in `layout` order).
/// - `c` is non-null, properly aligned for `f32`, and readable and writable
///   at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a` or `b`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sgemmtr(
    layout: CblasLayout,
    uplo: CblasUplo,
    transa: CblasTranspose,
    transb: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    if n <= 0 || !gemm_params_valid(layout, transa, transb, n, n, k, lda, ldb, ldc) {
        return;
    }
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        gemmtr_impl(
            layout, uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

/// Double precision GEMMTR: `C := alpha·op(A)·op(B) + beta·C`, updating only
/// the `uplo` triangle of the `n x n` matrix C (`op(A)` is `n x k`, `op(B)` is
/// `k x n`).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` is non-null, properly aligned for `f64`, and readable at every
///   `lda`-strided offset of the stored `A` (`n x k` for `NoTrans`, else
///   `k x n`, in `layout` order).
/// - `b` is non-null, properly aligned for `f64`, and readable at every
///   `ldb`-strided offset of the stored `B` (`k x n` for `NoTrans`, else
///   `n x k`, in `layout` order).
/// - `c` is non-null, properly aligned for `f64`, and readable and writable
///   at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a` or `b`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dgemmtr(
    layout: CblasLayout,
    uplo: CblasUplo,
    transa: CblasTranspose,
    transb: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    if n <= 0 || !gemm_params_valid(layout, transa, transb, n, n, k, lda, ldb, ldc) {
        return;
    }
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        gemmtr_impl(
            layout, uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

/// Complex single precision GEMMTR: `C := alpha·op(A)·op(B) + beta·C`,
/// updating only the `uplo` triangle of the `n x n` matrix C (`op(X)` is `X`,
/// `X^T` or `X^H`).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex32`.
/// - `a` is non-null, properly aligned for `Complex32`, and readable at every
///   `lda`-strided offset of the stored `A` (`n x k` for `NoTrans`, else
///   `k x n`, in `layout` order).
/// - `b` is non-null, properly aligned for `Complex32`, and readable at every
///   `ldb`-strided offset of the stored `B` (`k x n` for `NoTrans`, else
///   `n x k`, in `layout` order).
/// - `c` is non-null, properly aligned for `Complex32`, and readable and
///   writable at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a`, `b`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cgemmtr(
    layout: CblasLayout,
    uplo: CblasUplo,
    transa: CblasTranspose,
    transb: CblasTranspose,
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *const Complex32,
    ldb: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    if n <= 0 || !gemm_params_valid(layout, transa, transb, n, n, k, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        gemmtr_impl(
            layout, uplo, transa, transb, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc,
        );
    }
}

/// Complex double precision GEMMTR: `C := alpha·op(A)·op(B) + beta·C`,
/// updating only the `uplo` triangle of the `n x n` matrix C (`op(X)` is `X`,
/// `X^T` or `X^H`).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex64`.
/// - `a` is non-null, properly aligned for `Complex64`, and readable at every
///   `lda`-strided offset of the stored `A` (`n x k` for `NoTrans`, else
///   `k x n`, in `layout` order).
/// - `b` is non-null, properly aligned for `Complex64`, and readable at every
///   `ldb`-strided offset of the stored `B` (`k x n` for `NoTrans`, else
///   `n x k`, in `layout` order).
/// - `c` is non-null, properly aligned for `Complex64`, and readable and
///   writable at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a`, `b`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zgemmtr(
    layout: CblasLayout,
    uplo: CblasUplo,
    transa: CblasTranspose,
    transb: CblasTranspose,
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *const Complex64,
    ldb: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    if n <= 0 || !gemm_params_valid(layout, transa, transb, n, n, k, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        gemmtr_impl(
            layout, uplo, transa, transb, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc,
        );
    }
}

/// Shared GEMMTR body (column-major mapping, quick returns, native `gemmt`).
///
/// # Safety
/// `n > 0`, `k >= 0` and the leading dimensions satisfy
/// `gemm_params_valid`; the pointers are non-null and cover their operands.
unsafe fn gemmtr_impl<T: Field + GemmKernel + bytemuck::Zeroable>(
    layout: CblasLayout,
    uplo: CblasUplo,
    transa: CblasTranspose,
    transb: CblasTranspose,
    n: i32,
    k: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    b: *const T,
    ldb: i32,
    beta: T,
    c: *mut T,
    ldc: i32,
) {
    let (n, k, ldc) = (n as usize, k as usize, ldc as usize);
    let (zero, one) = (T::zero(), T::one());
    if (alpha == zero || k == 0) && beta == one {
        return;
    }
    // Column-major problem: the row-major C buffer is C^T, whose `flip(uplo)`
    // triangle is the caller's `uplo` triangle, and
    // C^T = alpha·op(B)^T·op(A)^T + beta·C^T = alpha·op'(B')·op'(A') + beta·C^T
    // with B' / A' the column-major views of the caller's buffers.
    let (uplo_cm, ta, a_cm, lda_cm, tb, b_cm, ldb_cm) = match layout {
        CblasLayout::ColMajor => (uplo, transa, a, lda as usize, transb, b, ldb as usize),
        CblasLayout::RowMajor => (
            flip_uplo(uplo),
            transb,
            b,
            ldb as usize,
            transa,
            a,
            lda as usize,
        ),
    };
    // SAFETY: C is n x n with leading dimension `ldc >= max(1, n)` (validated).
    let mut c_mat = unsafe { MatMut::new(c, n, n, ldc) };
    if alpha == zero || k == 0 {
        scale_triangle(uplo_cm, beta, &mut c_mat);
        return;
    }
    let (a_rows, a_cols) = if ta == CblasTranspose::NoTrans {
        (n, k)
    } else {
        (k, n)
    };
    let (b_rows, b_cols) = if tb == CblasTranspose::NoTrans {
        (k, n)
    } else {
        (n, k)
    };
    // SAFETY: the column-major views have exactly the stored shapes whose
    // leading dimensions `gemm_params_valid` checked.
    let (a_mat, b_mat) = unsafe {
        (
            MatRef::new(a_cm, a_rows, a_cols, lda_cm),
            MatRef::new(b_cm, b_rows, b_cols, ldb_cm),
        )
    };
    // The shapes are consistent by construction, so the native cannot report
    // a dimension error; a void C ABI could not surface one anyway.
    if let Err(_err) = level3::gemmt(
        native_uplo(uplo_cm),
        native_trans(ta),
        native_trans(tb),
        alpha,
        a_mat,
        b_mat,
        beta,
        c_mat,
    ) {}
}

// =============================================================================
// Complex SYMM — C := alpha*A*B + beta*C or alpha*B*A + beta*C, A symmetric
// =============================================================================

/// Complex single precision SYMM: `C := alpha·A·B + beta·C` (`Left`) or
/// `C := alpha·B·A + beta·C` (`Right`) with `A` complex **symmetric**
/// (`A = A^T`, no conjugation); only the `uplo` triangle of `A` is read.
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex32`.
/// - `a` is non-null, properly aligned for `Complex32`, and readable at every
///   `lda`-strided offset of the `ka x ka` matrix A (`ka = m` for `Left`,
///   `n` for `Right`).
/// - `b` is non-null, properly aligned for `Complex32`, and readable at every
///   `ldb`-strided offset of the `m x n` matrix B (in `layout` order).
/// - `c` is non-null, properly aligned for `Complex32`, and readable and
///   writable at every `ldc`-strided offset of the `m x n` matrix C.
/// - `c` does not overlap `a`, `b`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_csymm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *const Complex32,
    ldb: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    if m <= 0 || n <= 0 || !symm_params_valid(layout, side, m, n, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        symm_impl(
            layout, side, uplo, m, n, *alpha, a, lda, b, ldb, *beta, c, ldc,
        );
    }
}

/// Complex double precision SYMM: `C := alpha·A·B + beta·C` (`Left`) or
/// `C := alpha·B·A + beta·C` (`Right`) with `A` complex **symmetric**
/// (`A = A^T`, no conjugation); only the `uplo` triangle of `A` is read.
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex64`.
/// - `a` is non-null, properly aligned for `Complex64`, and readable at every
///   `lda`-strided offset of the `ka x ka` matrix A (`ka = m` for `Left`,
///   `n` for `Right`).
/// - `b` is non-null, properly aligned for `Complex64`, and readable at every
///   `ldb`-strided offset of the `m x n` matrix B (in `layout` order).
/// - `c` is non-null, properly aligned for `Complex64`, and readable and
///   writable at every `ldc`-strided offset of the `m x n` matrix C.
/// - `c` does not overlap `a`, `b`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zsymm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *const Complex64,
    ldb: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    if m <= 0 || n <= 0 || !symm_params_valid(layout, side, m, n, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        symm_impl(
            layout, side, uplo, m, n, *alpha, a, lda, b, ldb, *beta, c, ldc,
        );
    }
}

/// Shared SYMM body (row-major mapping as in `cblas_dsymm`, native `symm`).
///
/// # Safety
/// `m, n > 0` and the leading dimensions satisfy `symm_params_valid`; the
/// pointers are non-null and cover their operands.
unsafe fn symm_impl<T: Field + GemmKernel + bytemuck::Zeroable>(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: i32,
    n: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    b: *const T,
    ldb: i32,
    beta: T,
    c: *mut T,
    ldc: i32,
) {
    let (m, n) = (m as usize, n as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);
    // Reference quick return: alpha == 0 && beta == 1 leaves C untouched.
    if alpha == T::zero() && beta == T::one() {
        return;
    }
    let (side_cm, uplo_cm, bm, bn) = match layout {
        CblasLayout::ColMajor => (side, uplo, m, n),
        CblasLayout::RowMajor => (flip_side(side), flip_uplo(uplo), n, m),
    };
    // Order of A from the caller's side (`m` for Left, `n` for Right), the
    // same `ka` that `symm_params_valid` checked `lda` against.
    let ka = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };
    // SAFETY: shapes and leading dimensions validated by the caller.
    let (a_mat, b_mat, c_mat) = unsafe {
        (
            MatRef::new(a, ka, ka, lda),
            MatRef::new(b, bm, bn, ldb),
            MatMut::new(c, bm, bn, ldc),
        )
    };
    // The native handles alpha == 0 (C := beta·C without reading A/B, beta == 0
    // without reading C); its dimension checks cannot fail by construction.
    if let Err(_err) = level3::symm(
        native_side(side_cm),
        native_uplo(uplo_cm),
        alpha,
        a_mat,
        b_mat,
        beta,
        c_mat,
    ) {}
}

// =============================================================================
// Complex SYRK — C := alpha*A*A^T + beta*C or alpha*A^T*A + beta*C
// =============================================================================

/// Complex single precision SYRK: `C := alpha·A·A^T + beta·C` (`NoTrans`) or
/// `C := alpha·A^T·A + beta·C` (`Trans`), updating only the `uplo` triangle
/// of the `n x n` matrix C (symmetric update: no conjugation).
///
/// Column-major `ConjTrans` is rejected by reference CBLAS (`cblas_xerbla`
/// terminates the process) and is a silent no-op here; row-major `ConjTrans`
/// is identical to `Trans`, as in reference CBLAS 3.12 (see the module docs).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex32`.
/// - `a` is non-null, properly aligned for `Complex32`, and readable at every
///   `lda`-strided offset of the stored `A` (`n x k` for `NoTrans`, else
///   `k x n`, in `layout` order).
/// - `c` is non-null, properly aligned for `Complex32`, and readable and
///   writable at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_csyrk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    if n <= 0
        || !complex_sym_rank_trans_valid(layout, trans)
        || !syrk_params_valid(layout, trans, n, k, lda, ldc)
    {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        syrk_impl(layout, uplo, trans, n, k, *alpha, a, lda, *beta, c, ldc);
    }
}

/// Complex double precision SYRK: `C := alpha·A·A^T + beta·C` (`NoTrans`) or
/// `C := alpha·A^T·A + beta·C` (`Trans`), updating only the `uplo` triangle
/// of the `n x n` matrix C (symmetric update: no conjugation).
///
/// Column-major `ConjTrans` is rejected by reference CBLAS (`cblas_xerbla`
/// terminates the process) and is a silent no-op here; row-major `ConjTrans`
/// is identical to `Trans`, as in reference CBLAS 3.12 (see the module docs).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex64`.
/// - `a` is non-null, properly aligned for `Complex64`, and readable at every
///   `lda`-strided offset of the stored `A` (`n x k` for `NoTrans`, else
///   `k x n`, in `layout` order).
/// - `c` is non-null, properly aligned for `Complex64`, and readable and
///   writable at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zsyrk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    if n <= 0
        || !complex_sym_rank_trans_valid(layout, trans)
        || !syrk_params_valid(layout, trans, n, k, lda, ldc)
    {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        syrk_impl(layout, uplo, trans, n, k, *alpha, a, lda, *beta, c, ldc);
    }
}

/// Shared complex SYRK body.
///
/// # Safety
/// `n > 0`, `k >= 0`, the leading dimensions satisfy `syrk_params_valid` and
/// `trans` passed [`complex_sym_rank_trans_valid`]; the pointers are non-null
/// and cover their operands.
unsafe fn syrk_impl<T: Field + GemmKernel + bytemuck::Zeroable>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    beta: T,
    c: *mut T,
    ldc: i32,
) {
    let (n, k, lda, ldc) = (n as usize, k as usize, lda as usize, ldc as usize);
    let (zero, one) = (T::zero(), T::one());
    if (alpha == zero || k == 0) && beta == one {
        return;
    }
    let (uplo_cm, trans_cm) = sym_rank_colmajor(layout, uplo, trans);
    // SAFETY: C is n x n with leading dimension `ldc >= max(1, n)` (validated).
    let mut c_mat = unsafe { MatMut::new(c, n, n, ldc) };
    if alpha == zero || k == 0 {
        scale_triangle(uplo_cm, beta, &mut c_mat);
        return;
    }
    let (a_rows, a_cols) = if trans_cm == CblasTranspose::NoTrans {
        (n, k)
    } else {
        (k, n)
    };
    // SAFETY: the column-major view of A has exactly the stored shape whose
    // leading dimension `syrk_params_valid` checked.
    let a_mat = unsafe { MatRef::new(a, a_rows, a_cols, lda) };
    // `trans_cm` is NoTrans or Trans, so the native accepts it; its dimension
    // checks cannot fail by construction.
    if let Err(_err) = level3::syrk(
        native_uplo(uplo_cm),
        native_trans(trans_cm),
        alpha,
        a_mat,
        beta,
        c_mat,
    ) {}
}

// =============================================================================
// Complex SYR2K — C := alpha*A*B^T + alpha*B*A^T + beta*C (or transposed)
// =============================================================================

/// Complex single precision SYR2K: `C := alpha·A·B^T + alpha·B·A^T + beta·C`
/// (`NoTrans`) or `C := alpha·A^T·B + alpha·B^T·A + beta·C` (`Trans`),
/// updating only the `uplo` triangle of the `n x n` matrix C (symmetric
/// update: no conjugation, the same `alpha` on both terms).
///
/// Column-major `ConjTrans` is rejected by reference CBLAS (`cblas_xerbla`
/// terminates the process) and is a silent no-op here; row-major `ConjTrans`
/// is identical to `Trans`, as in reference CBLAS 3.12 (see the module docs).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex32`.
/// - `a` and `b` are non-null, properly aligned for `Complex32`, and readable
///   at every `lda`- / `ldb`-strided offset of the stored operands (`n x k`
///   for `NoTrans`, else `k x n`, in `layout` order).
/// - `c` is non-null, properly aligned for `Complex32`, and readable and
///   writable at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a`, `b`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_csyr2k(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *const Complex32,
    ldb: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    if n <= 0
        || !complex_sym_rank_trans_valid(layout, trans)
        || !syr2k_params_valid(layout, trans, n, k, lda, ldb, ldc)
    {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        syr2k_impl(
            layout, uplo, trans, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc,
        );
    }
}

/// Complex double precision SYR2K: `C := alpha·A·B^T + alpha·B·A^T + beta·C`
/// (`NoTrans`) or `C := alpha·A^T·B + alpha·B^T·A + beta·C` (`Trans`),
/// updating only the `uplo` triangle of the `n x n` matrix C (symmetric
/// update: no conjugation, the same `alpha` on both terms).
///
/// Column-major `ConjTrans` is rejected by reference CBLAS (`cblas_xerbla`
/// terminates the process) and is a silent no-op here; row-major `ConjTrans`
/// is identical to `Trans`, as in reference CBLAS 3.12 (see the module docs).
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` and `beta` are non-null and each point to one valid, properly
///   aligned, initialized `Complex64`.
/// - `a` and `b` are non-null, properly aligned for `Complex64`, and readable
///   at every `lda`- / `ldb`-strided offset of the stored operands (`n x k`
///   for `NoTrans`, else `k x n`, in `layout` order).
/// - `c` is non-null, properly aligned for `Complex64`, and readable and
///   writable at every `ldc`-strided offset of the `n x n` matrix C.
/// - `c` does not overlap `a`, `b`, `alpha` or `beta`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zsyr2k(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *const Complex64,
    ldb: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    if n <= 0
        || !complex_sym_rank_trans_valid(layout, trans)
        || !syr2k_params_valid(layout, trans, n, k, lda, ldb, ldc)
    {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        syr2k_impl(
            layout, uplo, trans, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc,
        );
    }
}

/// Shared complex SYR2K body.
///
/// # Safety
/// `n > 0`, `k >= 0`, the leading dimensions satisfy `syr2k_params_valid` and
/// `trans` passed [`complex_sym_rank_trans_valid`]; the pointers are non-null
/// and cover their operands.
unsafe fn syr2k_impl<T: Field + GemmKernel + bytemuck::Zeroable>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    b: *const T,
    ldb: i32,
    beta: T,
    c: *mut T,
    ldc: i32,
) {
    let (n, k, ldc) = (n as usize, k as usize, ldc as usize);
    let (lda, ldb) = (lda as usize, ldb as usize);
    let (zero, one) = (T::zero(), T::one());
    if (alpha == zero || k == 0) && beta == one {
        return;
    }
    let (uplo_cm, trans_cm) = sym_rank_colmajor(layout, uplo, trans);
    // SAFETY: C is n x n with leading dimension `ldc >= max(1, n)` (validated).
    let mut c_mat = unsafe { MatMut::new(c, n, n, ldc) };
    if alpha == zero || k == 0 {
        scale_triangle(uplo_cm, beta, &mut c_mat);
        return;
    }
    let (ab_rows, ab_cols) = if trans_cm == CblasTranspose::NoTrans {
        (n, k)
    } else {
        (k, n)
    };
    // SAFETY: the column-major views of A and B have exactly the stored shape
    // whose leading dimensions `syr2k_params_valid` checked.
    let (a_mat, b_mat) = unsafe {
        (
            MatRef::new(a, ab_rows, ab_cols, lda),
            MatRef::new(b, ab_rows, ab_cols, ldb),
        )
    };
    // `trans_cm` is NoTrans or Trans; the native's dimension checks cannot
    // fail by construction.
    if let Err(_err) = level3::syr2k(
        native_uplo(uplo_cm),
        native_trans(trans_cm),
        alpha,
        a_mat,
        b_mat,
        beta,
        c_mat,
    ) {}
}

// =============================================================================
// Complex TRMM — B := alpha*op(A)*B or alpha*B*op(A), A triangular
// =============================================================================

/// Complex single precision TRMM: `B := alpha·op(A)·B` (`Left`) or
/// `B := alpha·B·op(A)` (`Right`) with `A` triangular and `op(A)` one of `A`,
/// `A^T`, `A^H`.
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` is non-null and points to one valid, properly aligned,
///   initialized `Complex32`.
/// - `a` is non-null, properly aligned for `Complex32`, and readable at every
///   `lda`-strided offset of the `ka x ka` matrix A (`ka = m` for `Left`,
///   `n` for `Right`).
/// - `b` is non-null, properly aligned for `Complex32`, and readable and
///   writable at every `ldb`-strided offset of the `m x n` matrix B (in
///   `layout` order).
/// - `b` does not overlap `a` or `alpha`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctrmm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *mut Complex32,
    ldb: i32,
) {
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if alpha.is_null() || a.is_null() || b.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        trmm_impl(
            layout, side, uplo, transa, diag, m, n, *alpha, a, lda, b, ldb,
        );
    }
}

/// Complex double precision TRMM: `B := alpha·op(A)·B` (`Left`) or
/// `B := alpha·B·op(A)` (`Right`) with `A` triangular and `op(A)` one of `A`,
/// `A^T`, `A^H`.
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` is non-null and points to one valid, properly aligned,
///   initialized `Complex64`.
/// - `a` is non-null, properly aligned for `Complex64`, and readable at every
///   `lda`-strided offset of the `ka x ka` matrix A (`ka = m` for `Left`,
///   `n` for `Right`).
/// - `b` is non-null, properly aligned for `Complex64`, and readable and
///   writable at every `ldb`-strided offset of the `m x n` matrix B (in
///   `layout` order).
/// - `b` does not overlap `a` or `alpha`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztrmm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *mut Complex64,
    ldb: i32,
) {
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if alpha.is_null() || a.is_null() || b.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        trmm_impl(
            layout, side, uplo, transa, diag, m, n, *alpha, a, lda, b, ldb,
        );
    }
}

/// Shared complex TRMM body (row-major mapping as in `cblas_dtrmm`, native
/// `trmm_in_place`, which handles `alpha == 0` without reading A or B).
///
/// # Safety
/// `m, n > 0` and the leading dimensions satisfy `trmm_params_valid`; the
/// pointers are non-null and cover their operands.
unsafe fn trmm_impl<T: Field + GemmKernel + bytemuck::Zeroable>(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    b: *mut T,
    ldb: i32,
) {
    use crate::level3::{TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo};

    let (m, n, lda, ldb) = (m as usize, n as usize, lda as usize, ldb as usize);
    let (side_cm, uplo_cm, bm, bn) = tri_colmajor(layout, side, uplo, m, n);
    let ka = tri_order(side, m, n);
    // SAFETY: shapes and leading dimensions validated by the caller.
    let (a_mat, b_mat) = unsafe { (MatRef::new(a, ka, ka, lda), MatMut::new(b, bm, bn, ldb)) };
    let side_n = match side_cm {
        CblasSide::Left => TrmmSide::Left,
        CblasSide::Right => TrmmSide::Right,
    };
    let uplo_n = match uplo_cm {
        CblasUplo::Upper => TrmmUplo::Upper,
        CblasUplo::Lower => TrmmUplo::Lower,
    };
    let trans_n = match transa {
        CblasTranspose::NoTrans => TrmmTrans::NoTrans,
        CblasTranspose::Trans => TrmmTrans::Trans,
        CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
    };
    let diag_n = match diag {
        CblasDiag::NonUnit => TrmmDiag::NonUnit,
        CblasDiag::Unit => TrmmDiag::Unit,
    };
    // The native's dimension checks cannot fail by construction.
    if let Err(_err) = level3::trmm_in_place(side_n, uplo_n, trans_n, diag_n, alpha, a_mat, b_mat) {
    }
}

// =============================================================================
// Complex TRSM — solve op(A)*X = alpha*B or X*op(A) = alpha*B, A triangular
// =============================================================================

/// Complex single precision TRSM: solves `op(A)·X = alpha·B` (`Left`) or
/// `X·op(A) = alpha·B` (`Right`) for `X`, overwriting B, with `A` triangular
/// and `op(A)` one of `A`, `A^T`, `A^H`.
///
/// A singular `A` (zero on a non-unit diagonal) makes the solve stop early;
/// as in reference ?TRSM the contents of B are then unspecified.
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` is non-null and points to one valid, properly aligned,
///   initialized `Complex32`.
/// - `a` is non-null, properly aligned for `Complex32`, and readable at every
///   `lda`-strided offset of the `ka x ka` matrix A (`ka = m` for `Left`,
///   `n` for `Right`).
/// - `b` is non-null, properly aligned for `Complex32`, and readable and
///   writable at every `ldb`-strided offset of the `m x n` matrix B (in
///   `layout` order).
/// - `b` does not overlap `a` or `alpha`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ctrsm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *mut Complex32,
    ldb: i32,
) {
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if alpha.is_null() || a.is_null() || b.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        trsm_impl(
            layout, side, uplo, transa, diag, m, n, *alpha, a, lda, b, ldb,
        );
    }
}

/// Complex double precision TRSM: solves `op(A)·X = alpha·B` (`Left`) or
/// `X·op(A) = alpha·B` (`Right`) for `X`, overwriting B, with `A` triangular
/// and `op(A)` one of `A`, `A^T`, `A^H`.
///
/// A singular `A` (zero on a non-unit diagonal) makes the solve stop early;
/// as in reference ?TRSM the contents of B are then unspecified.
///
/// # Safety
///
/// This is a C ABI entry point. Dimensions, leading dimensions and enum flags
/// are validated before any pointer is dereferenced, but the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` is non-null and points to one valid, properly aligned,
///   initialized `Complex64`.
/// - `a` is non-null, properly aligned for `Complex64`, and readable at every
///   `lda`-strided offset of the `ka x ka` matrix A (`ka = m` for `Left`,
///   `n` for `Right`).
/// - `b` is non-null, properly aligned for `Complex64`, and readable and
///   writable at every `ldb`-strided offset of the `m x n` matrix B (in
///   `layout` order).
/// - `b` does not overlap `a` or `alpha`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ztrsm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *mut Complex64,
    ldb: i32,
) {
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if alpha.is_null() || a.is_null() || b.is_null() {
        return;
    }
    // SAFETY: shapes validated, pointers non-null and covering their operands (contract).
    unsafe {
        trsm_impl(
            layout, side, uplo, transa, diag, m, n, *alpha, a, lda, b, ldb,
        );
    }
}

/// Precision dispatch for the native complex in-place triangular solve.
trait ComplexTrsm: Field + GemmKernel + bytemuck::Zeroable {
    /// Solves `op(A)·X = B` (`Left`) or `X·op(A) = B` (`Right`) in place on
    /// column-major views with the precision-specific native kernel.
    fn trsm_native(
        side: level3::Side,
        uplo: level3::Uplo,
        trans: level3::Trans,
        diag: level3::Diag,
        a: MatRef<'_, Self>,
        b: MatMut<'_, Self>,
    );
}

impl ComplexTrsm for Complex32 {
    fn trsm_native(
        side: level3::Side,
        uplo: level3::Uplo,
        trans: level3::Trans,
        diag: level3::Diag,
        a: MatRef<'_, Self>,
        b: MatMut<'_, Self>,
    ) {
        // Only a singular A can fail here (the shapes are consistent by
        // construction); B is then left as the solver left it (module docs).
        if let Err(_err) = level3::trsm_c32_in_place(side, uplo, trans, diag, a, b) {}
    }
}

impl ComplexTrsm for Complex64 {
    fn trsm_native(
        side: level3::Side,
        uplo: level3::Uplo,
        trans: level3::Trans,
        diag: level3::Diag,
        a: MatRef<'_, Self>,
        b: MatMut<'_, Self>,
    ) {
        // Only a singular A can fail here (the shapes are consistent by
        // construction); B is then left as the solver left it (module docs).
        if let Err(_err) = level3::trsm_c64_in_place(side, uplo, trans, diag, a, b) {}
    }
}

/// Shared complex TRSM body (row-major mapping as in `cblas_dtrsm`).
///
/// # Safety
/// `m, n > 0` and the leading dimensions satisfy `trmm_params_valid`; the
/// pointers are non-null and cover their operands.
unsafe fn trsm_impl<T: ComplexTrsm>(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    b: *mut T,
    ldb: i32,
) {
    let (m, n, lda, ldb) = (m as usize, n as usize, lda as usize, ldb as usize);
    let (side_cm, uplo_cm, bm, bn) = tri_colmajor(layout, side, uplo, m, n);
    let ka = tri_order(side, m, n);
    // SAFETY: shapes and leading dimensions validated by the caller.
    let (a_mat, mut b_mat) = unsafe { (MatRef::new(a, ka, ka, lda), MatMut::new(b, bm, bn, ldb)) };
    // Reference ?TRSM: alpha == 0 sets B := 0 without reading A or B.
    if alpha == T::zero() {
        b_mat.fill_zero();
        return;
    }
    // X := alpha·B first (a complex multiply by 1 + 0i is not an identity for
    // non-finite entries, so alpha == 1 skips it), then solve in place.
    if alpha != T::one() {
        b_mat.scale(alpha);
    }
    T::trsm_native(
        native_side(side_cm),
        native_uplo(uplo_cm),
        native_trans(transa),
        native_diag(diag),
        a_mat,
        b_mat,
    );
}

// =============================================================================
// Shared helpers (private: the glob re-export in `cblas/mod.rs` only exposes
// the `pub` entry points)
// =============================================================================

/// `C := beta·C` on the `uplo` triangle of the square column-major `c` — the
/// whole update when `alpha == 0` or `k == 0`. `beta == 0` zero-fills without
/// reading C (callers have already returned for `beta == 1`).
fn scale_triangle<T: Field>(uplo: CblasUplo, beta: T, c: &mut MatMut<'_, T>) {
    let n = c.nrows();
    let beta_is_zero = beta == T::zero();
    for j in 0..n {
        let (lo, hi) = match uplo {
            CblasUplo::Upper => (0, j + 1),
            CblasUplo::Lower => (j, n),
        };
        for i in lo..hi {
            let value = if beta_is_zero {
                T::zero()
            } else {
                beta * c[(i, j)]
            };
            c.set(i, j, value);
        }
    }
}

/// Reference CBLAS 3.12 accepts complex SYRK/SYR2K `ConjTrans` only in
/// row-major layout (mapped to `'N'` like `Trans`); in column-major layout it
/// reaches Fortran C/ZSYRK / C/ZSYR2K as `TRANS = 'C'`, which they reject.
fn complex_sym_rank_trans_valid(layout: CblasLayout, trans: CblasTranspose) -> bool {
    !(layout == CblasLayout::ColMajor && trans == CblasTranspose::ConjTrans)
}

/// Column-major `(uplo, trans)` of a SYRK/SYR2K call: row-major flips `uplo`
/// and maps `NoTrans -> Trans`, `Trans | ConjTrans -> NoTrans`. The result is
/// never `ConjTrans` (callers have rejected a column-major `ConjTrans`).
fn sym_rank_colmajor(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
) -> (CblasUplo, CblasTranspose) {
    match layout {
        CblasLayout::ColMajor => (uplo, trans),
        CblasLayout::RowMajor => (
            flip_uplo(uplo),
            match trans {
                CblasTranspose::NoTrans => CblasTranspose::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => CblasTranspose::NoTrans,
            },
        ),
    }
}

/// Column-major `(side, uplo, rows of B, cols of B)` of a TRMM/TRSM call:
/// row-major flips `side` and `uplo` and views B as the `n x m` transpose.
fn tri_colmajor(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: usize,
    n: usize,
) -> (CblasSide, CblasUplo, usize, usize) {
    match layout {
        CblasLayout::ColMajor => (side, uplo, m, n),
        CblasLayout::RowMajor => (flip_side(side), flip_uplo(uplo), n, m),
    }
}

/// Order of the triangular operand from the caller's (unflipped) side: `m`
/// for `Left`, `n` for `Right` — the `ka` that `trmm_params_valid` checks
/// `lda` against. It must not be derived from the flipped row-major side.
fn tri_order(side: CblasSide, m: usize, n: usize) -> usize {
    match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    }
}

fn flip_uplo(uplo: CblasUplo) -> CblasUplo {
    match uplo {
        CblasUplo::Upper => CblasUplo::Lower,
        CblasUplo::Lower => CblasUplo::Upper,
    }
}

fn flip_side(side: CblasSide) -> CblasSide {
    match side {
        CblasSide::Left => CblasSide::Right,
        CblasSide::Right => CblasSide::Left,
    }
}

fn native_uplo(uplo: CblasUplo) -> level3::Uplo {
    match uplo {
        CblasUplo::Upper => level3::Uplo::Upper,
        CblasUplo::Lower => level3::Uplo::Lower,
    }
}

fn native_side(side: CblasSide) -> level3::Side {
    match side {
        CblasSide::Left => level3::Side::Left,
        CblasSide::Right => level3::Side::Right,
    }
}

fn native_trans(trans: CblasTranspose) -> level3::Trans {
    match trans {
        CblasTranspose::NoTrans => level3::Trans::NoTrans,
        CblasTranspose::Trans => level3::Trans::Trans,
        CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
    }
}

fn native_diag(diag: CblasDiag) -> level3::Diag {
    match diag {
        CblasDiag::NonUnit => level3::Diag::NonUnit,
        CblasDiag::Unit => level3::Diag::Unit,
    }
}

#[cfg(test)]
#[path = "level3_ext_tests.rs"]
mod tests;
