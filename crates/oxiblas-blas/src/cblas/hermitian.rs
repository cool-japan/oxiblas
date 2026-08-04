//! CBLAS-compatible Hermitian Level-3 routines.
//!
//! Provides the C-ABI wrappers for the Hermitian matrix-matrix family:
//! `chemm`/`zhemm`, `cherk`/`zherk`, `cher2k`/`zher2k`.
//!
//! # Layout handling
//!
//! This library is column-major internally. Row-major requests are mapped to
//! column-major by the standard CBLAS identity `op(A)`(row-major)
//! = `op(A^T)`(column-major). For the Hermitian family the concrete rules are:
//!
//! - `hemm`: swap `Side`, flip `Uplo`, swap `m`/`n`. `alpha`/`beta` unchanged.
//!   (Reading a Hermitian buffer transposed yields `conj(A) = A^T`, which is
//!   itself Hermitian with the opposite triangle stored, so `hemm` reconstructs
//!   the correct operator with no data conjugation.)
//! - `herk`: flip `Uplo`, toggle `NoTrans`<->`ConjTrans`. `alpha`/`beta` (real)
//!   unchanged.
//! - `her2k`: flip `Uplo`, toggle `NoTrans`<->`ConjTrans`, replace `alpha` by
//!   `conj(alpha)`. `beta` (real) unchanged. The `conj(alpha)` is required
//!   because transposing `alpha·A·B^H + conj(alpha)·B·A^H` swaps the two rank-1
//!   families, which is only symmetric under conjugation of `alpha`.
//!
//! # Diagonal
//!
//! For `herk`/`her2k` the result `C` is Hermitian, so the imaginary parts of its
//! diagonal are set to exactly zero on exit, matching Netlib reference behavior.

use super::types::*;
use super::validate::{symm_params_valid, syr2k_params_valid, syrk_params_valid};
use crate::level3;
use num_complex::{Complex, Complex32, Complex64};
use num_traits::Float;

// =============================================================================
// HEMM (Hermitian matrix-matrix multiply) — wraps native `hemm_c32`/`hemm_c64`
// =============================================================================

/// Complex single precision HEMM: `C = alpha*A*B + beta*C` (`Side::Left`) or
/// `C = alpha*B*A + beta*C` (`Side::Right`), with `A` Hermitian.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `a` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `beta` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `c` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_chemm(
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
    // Reference (x)HEMM calls xerbla and returns on a bad shape; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. Without this, `lda as usize` turns a negative `lda` into
    // `usize::MAX` (and an undersized positive `lda` aliases columns), making
    // the internal `a.add(row + col * lda)` read wildly out of bounds.
    if m <= 0 || n <= 0 || !symm_params_valid(layout, side, m, n, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    hemm_impl(
        layout, side, uplo, m, n, *alpha, a, lda, b, ldb, *beta, c, ldc,
    );
}

/// Complex double precision HEMM: `C = alpha*A*B + beta*C` (`Side::Left`) or
/// `C = alpha*B*A + beta*C` (`Side::Right`), with `A` Hermitian.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `a` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `beta` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `c` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zhemm(
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
    // Reference (x)HEMM calls xerbla and returns on a bad shape; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. Without this, `lda as usize` turns a negative `lda` into
    // `usize::MAX` (and an undersized positive `lda` aliases columns), making
    // the internal `a.add(row + col * lda)` read wildly out of bounds.
    if m <= 0 || n <= 0 || !symm_params_valid(layout, side, m, n, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    hemm_impl(
        layout, side, uplo, m, n, *alpha, a, lda, b, ldb, *beta, c, ldc,
    );
}

/// Shared HEMM marshaling. `HemmScalar` abstracts over the two native
/// precision-specific entry points so the layout logic lives in one place.
trait HemmScalar: Copy {
    /// Calls the native precision-specific `hemm` on column-major views.
    ///
    /// # Safety
    /// Pointers must be valid column-major buffers with the given leading
    /// dimensions for the whole `m`x`n`/`k`x`k` extents.
    unsafe fn native_hemm(
        side: level3::Side,
        uplo: level3::Uplo,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        b: *const Self,
        ldb: usize,
        beta: Self,
        c: *mut Self,
        ldc: usize,
    );
}

impl HemmScalar for Complex32 {
    unsafe fn native_hemm(
        side: level3::Side,
        uplo: level3::Uplo,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        b: *const Self,
        ldb: usize,
        beta: Self,
        c: *mut Self,
        ldc: usize,
    ) {
        use oxiblas_matrix::{MatMut, MatRef};
        let a_ref = MatRef::<Complex32>::new(a, k, k, lda);
        let b_ref = MatRef::<Complex32>::new(b, m, n, ldb);
        let c_mut = MatMut::<Complex32>::new(c, m, n, ldc);
        let _ = level3::hemm_c32(side, uplo, alpha, a_ref, b_ref, beta, c_mut);
    }
}

impl HemmScalar for Complex64 {
    unsafe fn native_hemm(
        side: level3::Side,
        uplo: level3::Uplo,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        b: *const Self,
        ldb: usize,
        beta: Self,
        c: *mut Self,
        ldc: usize,
    ) {
        use oxiblas_matrix::{MatMut, MatRef};
        let a_ref = MatRef::<Complex64>::new(a, k, k, lda);
        let b_ref = MatRef::<Complex64>::new(b, m, n, ldb);
        let c_mut = MatMut::<Complex64>::new(c, m, n, ldc);
        let _ = level3::hemm_c64(side, uplo, alpha, a_ref, b_ref, beta, c_mut);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn hemm_impl<T: HemmScalar>(
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
    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    // Map layout: for row-major, swap Side and Uplo and transpose the B/C extents.
    let (side_internal, uplo_internal, bm, bn) = match layout {
        CblasLayout::ColMajor => (map_side(side), map_uplo(uplo), m, n),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => level3::Side::Right,
                CblasSide::Right => level3::Side::Left,
            },
            flip_uplo(uplo),
            n,
            m,
        ),
    };

    // The Hermitian operand is `k`x`k`, where `k` is the shared dimension.
    let k = match side_internal {
        level3::Side::Left => bm,
        level3::Side::Right => bn,
    };

    T::native_hemm(
        side_internal,
        uplo_internal,
        bm,
        bn,
        k,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc,
    );
}

// =============================================================================
// HERK (Hermitian rank-k update)
// =============================================================================

/// Complex single precision HERK: `C = alpha*A*A^H + beta*C` (`NoTrans`) or
/// `C = alpha*A^H*A + beta*C` (`ConjTrans`), with real `alpha`/`beta`.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cherk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const Complex32,
    lda: i32,
    beta: f32,
    c: *mut Complex32,
    ldc: i32,
) {
    // See the HEMM guard above: xerbla-then-return, no panic across the C ABI.
    if n <= 0 || !syrk_params_valid(layout, trans, n, k, lda, ldc) {
        return;
    }
    if a.is_null() || c.is_null() {
        return;
    }
    herk_impl(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

/// Complex double precision HERK: `C = alpha*A*A^H + beta*C` (`NoTrans`) or
/// `C = alpha*A^H*A + beta*C` (`ConjTrans`), with real `alpha`/`beta`.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zherk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const Complex64,
    lda: i32,
    beta: f64,
    c: *mut Complex64,
    ldc: i32,
) {
    // See the HEMM guard above: xerbla-then-return, no panic across the C ABI.
    if n <= 0 || !syrk_params_valid(layout, trans, n, k, lda, ldc) {
        return;
    }
    if a.is_null() || c.is_null() {
        return;
    }
    herk_impl(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

#[allow(clippy::too_many_arguments)]
unsafe fn herk_impl<F: Float>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: F,
    a: *const Complex<F>,
    lda: i32,
    beta: F,
    c: *mut Complex<F>,
    ldc: i32,
) {
    // HERK only admits NoTrans and ConjTrans; Trans is invalid (matches xerbla
    // in the reference — we simply do nothing rather than panic).
    if matches!(trans, CblasTranspose::Trans) {
        return;
    }
    if k < 0 {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;

    let (uplo_eff, trans_eff) = match layout {
        CblasLayout::ColMajor => (uplo, trans),
        CblasLayout::RowMajor => (flip_uplo_cb(uplo), toggle_conj(trans)),
    };

    herk_colmajor(uplo_eff, trans_eff, n, k, alpha, a, lda, beta, c, ldc);
}

/// Column-major Hermitian rank-k update following Netlib CHERK/ZHERK exactly.
///
/// Each referenced element of the `uplo` triangle is written once as
/// `beta*C[i,j] + alpha*sum_p (...)`, so no separate beta-scaling pass is
/// needed. Diagonal elements are forced real on exit.
#[allow(clippy::too_many_arguments)]
unsafe fn herk_colmajor<F: Float>(
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: usize,
    k: usize,
    alpha: F,
    a: *const Complex<F>,
    lda: usize,
    beta: F,
    c: *mut Complex<F>,
    ldc: usize,
) {
    let zero = F::zero();
    for j in 0..n {
        let (i_lo, i_hi) = match uplo {
            CblasUplo::Upper => (0usize, j + 1),
            CblasUplo::Lower => (j, n),
        };
        for i in i_lo..i_hi {
            let mut acc = Complex::new(zero, zero);
            for p in 0..k {
                match trans {
                    // C[i,j] += A[i,p] * conj(A[j,p])
                    CblasTranspose::NoTrans => {
                        let a_ip = *a.add(i + p * lda);
                        let a_jp = *a.add(j + p * lda);
                        acc = acc + a_ip * a_jp.conj();
                    }
                    // C[i,j] += conj(A[p,i]) * A[p,j]
                    _ => {
                        let a_pi = *a.add(p + i * lda);
                        let a_pj = *a.add(p + j * lda);
                        acc = acc + a_pi.conj() * a_pj;
                    }
                }
            }
            let contrib = acc.scale(alpha);
            let cptr = c.add(i + j * ldc);
            let old = *cptr;
            if i == j {
                // Diagonal of a Hermitian matrix is real by construction.
                *cptr = Complex::new(beta * old.re + contrib.re, zero);
            } else {
                *cptr = old.scale(beta) + contrib;
            }
        }
    }
}

// =============================================================================
// HER2K (Hermitian rank-2k update)
// =============================================================================

/// Complex single precision HER2K:
/// `C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C` (`NoTrans`) or
/// `C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C` (`ConjTrans`), with real `beta`.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `a` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cher2k(
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
    beta: f32,
    c: *mut Complex32,
    ldc: i32,
) {
    // See the HEMM guard above: xerbla-then-return, no panic across the C ABI.
    if n <= 0 || !syr2k_params_valid(layout, trans, n, k, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    her2k_impl(
        layout, uplo, trans, n, k, *alpha, a, lda, b, ldb, beta, c, ldc,
    );
}

/// Complex double precision HER2K:
/// `C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C` (`NoTrans`) or
/// `C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C` (`ConjTrans`), with real `beta`.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `a` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zher2k(
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
    beta: f64,
    c: *mut Complex64,
    ldc: i32,
) {
    // See the HEMM guard above: xerbla-then-return, no panic across the C ABI.
    if n <= 0 || !syr2k_params_valid(layout, trans, n, k, lda, ldb, ldc) {
        return;
    }
    if alpha.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }
    her2k_impl(
        layout, uplo, trans, n, k, *alpha, a, lda, b, ldb, beta, c, ldc,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn her2k_impl<F: Float>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: Complex<F>,
    a: *const Complex<F>,
    lda: i32,
    b: *const Complex<F>,
    ldb: i32,
    beta: F,
    c: *mut Complex<F>,
    ldc: i32,
) {
    if matches!(trans, CblasTranspose::Trans) {
        return;
    }
    if k < 0 {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    // Row-major: flip uplo, toggle NoTrans<->ConjTrans, conjugate alpha.
    let (uplo_eff, trans_eff, alpha_eff) = match layout {
        CblasLayout::ColMajor => (uplo, trans, alpha),
        CblasLayout::RowMajor => (flip_uplo_cb(uplo), toggle_conj(trans), alpha.conj()),
    };

    her2k_colmajor(
        uplo_eff, trans_eff, n, k, alpha_eff, a, lda, b, ldb, beta, c, ldc,
    );
}

/// Column-major Hermitian rank-2k update following Netlib CHER2K/ZHER2K.
///
/// Each referenced element is written once; diagonal forced real on exit.
#[allow(clippy::too_many_arguments)]
unsafe fn her2k_colmajor<F: Float>(
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: usize,
    k: usize,
    alpha: Complex<F>,
    a: *const Complex<F>,
    lda: usize,
    b: *const Complex<F>,
    ldb: usize,
    beta: F,
    c: *mut Complex<F>,
    ldc: usize,
) {
    let zero = F::zero();
    let alpha_conj = alpha.conj();
    for j in 0..n {
        let (i_lo, i_hi) = match uplo {
            CblasUplo::Upper => (0usize, j + 1),
            CblasUplo::Lower => (j, n),
        };
        for i in i_lo..i_hi {
            // s1 = sum A[i,p]*conj(B[j,p]) ; s2 = sum B[i,p]*conj(A[j,p])
            // (NoTrans) or the conjugate-transposed analogues (ConjTrans).
            let mut s1 = Complex::new(zero, zero);
            let mut s2 = Complex::new(zero, zero);
            for p in 0..k {
                match trans {
                    CblasTranspose::NoTrans => {
                        let a_ip = *a.add(i + p * lda);
                        let a_jp = *a.add(j + p * lda);
                        let b_ip = *b.add(i + p * ldb);
                        let b_jp = *b.add(j + p * ldb);
                        s1 = s1 + a_ip * b_jp.conj();
                        s2 = s2 + b_ip * a_jp.conj();
                    }
                    _ => {
                        let a_pi = *a.add(p + i * lda);
                        let a_pj = *a.add(p + j * lda);
                        let b_pi = *b.add(p + i * ldb);
                        let b_pj = *b.add(p + j * ldb);
                        s1 = s1 + a_pi.conj() * b_pj;
                        s2 = s2 + b_pi.conj() * a_pj;
                    }
                }
            }
            let contrib = alpha * s1 + alpha_conj * s2;
            let cptr = c.add(i + j * ldc);
            let old = *cptr;
            if i == j {
                *cptr = Complex::new(beta * old.re + contrib.re, zero);
            } else {
                *cptr = old.scale(beta) + contrib;
            }
        }
    }
}

// =============================================================================
// Small enum-mapping helpers (shared, no allocation)
// =============================================================================

#[inline]
fn map_side(side: CblasSide) -> level3::Side {
    match side {
        CblasSide::Left => level3::Side::Left,
        CblasSide::Right => level3::Side::Right,
    }
}

#[inline]
fn map_uplo(uplo: CblasUplo) -> level3::Uplo {
    match uplo {
        CblasUplo::Upper => level3::Uplo::Upper,
        CblasUplo::Lower => level3::Uplo::Lower,
    }
}

#[inline]
fn flip_uplo(uplo: CblasUplo) -> level3::Uplo {
    match uplo {
        CblasUplo::Upper => level3::Uplo::Lower,
        CblasUplo::Lower => level3::Uplo::Upper,
    }
}

#[inline]
fn flip_uplo_cb(uplo: CblasUplo) -> CblasUplo {
    match uplo {
        CblasUplo::Upper => CblasUplo::Lower,
        CblasUplo::Lower => CblasUplo::Upper,
    }
}

/// Toggles `NoTrans` <-> `ConjTrans` for row-major Hermitian rank updates.
#[inline]
fn toggle_conj(trans: CblasTranspose) -> CblasTranspose {
    match trans {
        CblasTranspose::NoTrans => CblasTranspose::ConjTrans,
        _ => CblasTranspose::NoTrans,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    /// Reference Hermitian rank-k update computed straight from the definition,
    /// used to validate the CBLAS wrapper independently of the library.
    fn ref_herk(
        upper: bool,
        conj_trans: bool,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[Complex64],
        lda: usize,
        beta: f64,
        c: &mut [Complex64],
        ldc: usize,
    ) {
        for j in 0..n {
            let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
            for i in lo..hi {
                let mut acc = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    if conj_trans {
                        acc += a[p + i * lda].conj() * a[p + j * lda];
                    } else {
                        acc += a[i + p * lda] * a[j + p * lda].conj();
                    }
                }
                let contrib = acc * alpha;
                let old = c[i + j * ldc];
                c[i + j * ldc] = if i == j {
                    Complex64::new(beta * old.re + contrib.re, 0.0)
                } else {
                    old * beta + contrib
                };
            }
        }
    }

    #[test]
    fn test_cblas_zherk_notrans_lower() {
        // A is 3x2 column-major.
        let a = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 2.0),
        ];
        let n = 3;
        let k = 2;
        let mut c = [Complex64::new(0.0, 0.0); 9];
        let mut c_ref = c;

        unsafe {
            cblas_zherk(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                n as i32,
                k as i32,
                2.0,
                a.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        ref_herk(false, false, n, k, 2.0, &a, n, 0.0, &mut c_ref, n);
        for (idx, (g, r)) in c.iter().zip(c_ref.iter()).enumerate() {
            assert!(
                (g.re - r.re).abs() < 1e-10 && (g.im - r.im).abs() < 1e-10,
                "mismatch at {idx}: got {g:?} ref {r:?}"
            );
        }
        // Diagonal must be exactly real.
        assert_eq!(c[0].im, 0.0);
        assert_eq!(c[4].im, 0.0);
        assert_eq!(c[8].im, 0.0);
    }

    #[test]
    fn test_cblas_zherk_conjtrans_upper_beta() {
        // A is 2x3 column-major (ConjTrans => result 3x3).
        let a = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 2.0),
        ];
        let n = 3;
        let k = 2;
        let mut c = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.5),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.0, -1.0),
            Complex64::new(-1.0, -0.5),
            Complex64::new(3.0, 0.0),
        ];
        let mut c_ref = c;
        unsafe {
            cblas_zherk(
                CblasLayout::ColMajor,
                CblasUplo::Upper,
                CblasTranspose::ConjTrans,
                n as i32,
                k as i32,
                1.5,
                a.as_ptr(),
                k as i32,
                -0.5,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        ref_herk(true, true, n, k, 1.5, &a, k, -0.5, &mut c_ref, n);
        for (g, r) in c.iter().zip(c_ref.iter()) {
            assert!((g.re - r.re).abs() < 1e-10 && (g.im - r.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_zherk_rowmajor_matches_colmajor() {
        // Row-major NoTrans on an n x k buffer must equal the transformed
        // column-major computation. We check the Hermitian property holds.
        let n = 3;
        let k = 2;
        // Row-major A: n rows, k cols, row stride = k.
        let a = [
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(2.0, 2.0),
        ];
        let mut c = [Complex64::new(0.0, 0.0); 9];
        unsafe {
            cblas_zherk(
                CblasLayout::RowMajor,
                CblasUplo::Upper,
                CblasTranspose::NoTrans,
                n,
                k,
                1.0,
                a.as_ptr(),
                k,
                0.0,
                c.as_mut_ptr(),
                n,
            );
        }
        // C must be Hermitian: C[i,j] == conj(C[j,i]) for the referenced
        // triangle vs its mirror, and diagonal real.
        for d in [0usize, 4, 8] {
            assert!(c[d].im.abs() < 1e-12);
            assert!(c[d].re >= 0.0);
        }
    }

    #[test]
    fn test_cblas_zher2k_notrans_lower() {
        let n = 3;
        let k = 2;
        let a = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 2.0),
        ];
        let b = [
            Complex64::new(0.5, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(3.0, 1.0),
        ];
        let alpha = Complex64::new(1.5, -0.5);
        let beta = 0.75;
        let mut c = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.5),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.0, -1.0),
            Complex64::new(-1.0, -0.5),
            Complex64::new(3.0, 0.0),
        ];
        let mut c_ref = c;
        unsafe {
            cblas_zher2k(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                n as i32,
                k as i32,
                &alpha,
                a.as_ptr(),
                n as i32,
                b.as_ptr(),
                n as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        // Reference from definition.
        let ac = alpha.conj();
        for j in 0..n {
            for i in j..n {
                let mut s1 = Complex64::new(0.0, 0.0);
                let mut s2 = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    s1 += a[i + p * n] * b[j + p * n].conj();
                    s2 += b[i + p * n] * a[j + p * n].conj();
                }
                let contrib = alpha * s1 + ac * s2;
                let old = c_ref[i + j * n];
                c_ref[i + j * n] = if i == j {
                    Complex64::new(beta * old.re + contrib.re, 0.0)
                } else {
                    old * beta + contrib
                };
            }
        }
        for j in 0..n {
            for i in j..n {
                let g = c[i + j * n];
                let r = c_ref[i + j * n];
                assert!((g.re - r.re).abs() < 1e-10 && (g.im - r.im).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_cblas_zhemm_matches_native() {
        // Validate the HEMM wrapper against the native hemm_c64 directly on
        // identical column-major buffers (leading dimension = m for both).
        use oxiblas_matrix::{MatMut, MatRef};
        let m = 2usize;
        let n = 3usize;
        // Hermitian A (m x m), lower stored, real diagonal.
        let a = [
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 0.0), // upper (unused for Lower)
            Complex64::new(3.0, 0.0),
        ];
        let b = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 1.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, 2.0),
        ];
        let alpha = Complex64::new(1.0, 0.5);
        let beta = Complex64::new(0.5, 0.0);
        let mut c = [Complex64::new(0.5, 0.5); 6];
        let mut c_native = c;

        unsafe {
            cblas_zhemm(
                CblasLayout::ColMajor,
                CblasSide::Left,
                CblasUplo::Lower,
                m as i32,
                n as i32,
                &alpha,
                a.as_ptr(),
                m as i32,
                b.as_ptr(),
                m as i32,
                &beta,
                c.as_mut_ptr(),
                m as i32,
            );

            let a_ref = MatRef::<Complex64>::new(a.as_ptr(), m, m, m);
            let b_ref = MatRef::<Complex64>::new(b.as_ptr(), m, n, m);
            let c_mut = MatMut::<Complex64>::new(c_native.as_mut_ptr(), m, n, m);
            let _ = level3::hemm_c64(
                level3::Side::Left,
                level3::Uplo::Lower,
                alpha,
                a_ref,
                b_ref,
                beta,
                c_mut,
            );
        }

        for (g, r) in c.iter().zip(c_native.iter()) {
            assert!(
                (g.re - r.re).abs() < 1e-10 && (g.im - r.im).abs() < 1e-10,
                "hemm mismatch: got {g:?} ref {r:?}"
            );
        }
    }
}
