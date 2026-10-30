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
//! For `herk`/`her2k` the result `C` is Hermitian, so whenever `C` is updated the
//! imaginary parts of its diagonal are set to exactly zero, matching Netlib reference
//! behavior. The reference quick return, `(alpha == 0 || k == 0) && beta == 1`, leaves
//! `C` completely untouched (even a non-real diagonal).
//!
//! # `alpha == 0` / `beta == 0`
//!
//! As in the reference, `alpha == 0` never reads `A` / `B` (the update degenerates to
//! `C := beta*C`) and `beta == 0` never reads `C`, so NaN / Inf in those operands do
//! not leak into the result.

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
    // See the HEMM guard above: validate-then-return (a silent no-op, no
    // `cblas_xerbla` call), no panic across the C ABI.
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
    // See the HEMM guard above: validate-then-return (a silent no-op, no
    // `cblas_xerbla` call), no panic across the C ABI.
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

/// Column-major Hermitian rank-k update with the Netlib CHERK/ZHERK quick return,
/// `alpha == 0` / `beta == 0` semantics and real diagonal.
///
/// Each referenced element of the `uplo` triangle is written once as
/// `beta*C[i,j] + alpha*sum_p (...)`, so no separate beta-scaling pass is
/// needed. Diagonal elements are forced real whenever C is written.
///
/// Reference semantics: for `(alpha == 0 || k == 0) && beta == 1` C is left
/// completely untouched (even a non-real diagonal); for `alpha == 0 || k == 0`
/// only `C := beta*C` remains and A is never read; `beta == 0` never reads C.
///
/// The operation forms are not the reference's, so NaN / Inf propagate differently.
/// Off the diagonal `beta` and `alpha` are applied componentwise (`Complex::scale`),
/// where the reference (gfortran, default floating-point flags) multiplies by
/// `(beta, 0)` / `(alpha, 0)` as complex numbers (for `ConjTrans` also when
/// `beta == 1`). The diagonal uses the reference's real `beta*Re(C[j,j])`. `NoTrans`
/// is a dot product per element, without the reference's column-wise axpy that skips
/// a zero `A[j,p]`. See the `herk`/`her2k` NaN/Inf item in TODO.md.
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
    let scale_only = alpha == zero || k == 0;
    if scale_only && beta == F::one() {
        return;
    }
    for j in 0..n {
        let (i_lo, i_hi) = match uplo {
            CblasUplo::Upper => (0usize, j + 1),
            CblasUplo::Lower => (j, n),
        };
        for i in i_lo..i_hi {
            let mut acc = Complex::new(zero, zero);
            if !scale_only {
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
            }
            let contrib = if scale_only {
                Complex::new(zero, zero)
            } else {
                acc.scale(alpha)
            };
            let cptr = c.add(i + j * ldc);
            // beta == 0 must not read C (NaN/garbage in an uninitialised C must not leak).
            if i == j {
                // Diagonal of a Hermitian matrix is real by construction.
                let base = if beta == zero {
                    zero
                } else {
                    beta * (*cptr).re
                };
                *cptr = Complex::new(base + contrib.re, zero);
            } else {
                let base = if beta == zero {
                    Complex::new(zero, zero)
                } else {
                    (*cptr).scale(beta)
                };
                *cptr = base + contrib;
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
    // See the HEMM guard above: validate-then-return (a silent no-op, no
    // `cblas_xerbla` call), no panic across the C ABI.
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
    // See the HEMM guard above: validate-then-return (a silent no-op, no
    // `cblas_xerbla` call), no panic across the C ABI.
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

/// Column-major Hermitian rank-2k update with the Netlib CHER2K/ZHER2K quick
/// return, `alpha == 0` / `beta == 0` semantics and real diagonal.
///
/// Each referenced element is written once; the diagonal is forced real whenever C is
/// written.
///
/// Reference semantics: for `(alpha == 0 || k == 0) && beta == 1` C is left
/// completely untouched (even a non-real diagonal); for `alpha == 0 || k == 0`
/// only `C := beta*C` remains and A and B are never read; `beta == 0` never
/// reads C.
///
/// The operation forms are not all the reference's, so NaN / Inf propagate
/// differently. Off the diagonal `beta` is applied componentwise (`Complex::scale`),
/// where the reference (gfortran, default floating-point flags) multiplies by
/// `(beta, 0)` as a complex number (for `ConjTrans` also when `beta == 1`). The
/// diagonal uses the reference's real `beta*Re(C[j,j])`. `NoTrans` forms the dot
/// products `s1` / `s2` and applies `alpha` / `conj(alpha)` to the sums, without the
/// reference's column-wise axpy with per-column `alpha` temporaries that skips a
/// column where `A[j,p]` and `B[j,p]` are both zero. See the `herk`/`her2k` NaN/Inf
/// item in TODO.md.
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
    let scale_only = alpha == Complex::new(zero, zero) || k == 0;
    if scale_only && beta == F::one() {
        return;
    }
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
            if !scale_only {
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
            }
            let contrib = if scale_only {
                Complex::new(zero, zero)
            } else {
                alpha * s1 + alpha_conj * s2
            };
            let cptr = c.add(i + j * ldc);
            // beta == 0 must not read C (NaN/garbage in an uninitialised C must not leak).
            if i == j {
                let base = if beta == zero {
                    zero
                } else {
                    beta * (*cptr).re
                };
                *cptr = Complex::new(base + contrib.re, zero);
            } else {
                let base = if beta == zero {
                    Complex::new(zero, zero)
                } else {
                    (*cptr).scale(beta)
                };
                *cptr = base + contrib;
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

    // =========================================================================
    // Level-3 audits for the complex wrappers.
    //
    // * #9 class: reference BLAS never reads A / B when alpha == 0 and never reads C
    //   when beta == 0; herk / her2k additionally leave C completely untouched for
    //   `(alpha == 0 || k == 0) && beta == 1` and otherwise keep a real diagonal.
    // * #7 class: RowMajor hemm (m != n) and herk / her2k (n != k) against a dense
    //   reference, for both layouts, with junk in the unreferenced triangle.
    // =========================================================================

    /// Real-scalar bridge so one harness drives the single and double precision entry points.
    trait TestFloat: Float {
        fn from_f64(v: f64) -> Self;
        fn to_f64(self) -> f64;
    }

    impl TestFloat for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
        fn to_f64(self) -> f64 {
            f64::from(self)
        }
    }

    impl TestFloat for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
        fn to_f64(self) -> f64 {
            self
        }
    }

    type HemmFn<F> = unsafe extern "C" fn(
        CblasLayout,
        CblasSide,
        CblasUplo,
        i32,
        i32,
        *const Complex<F>,
        *const Complex<F>,
        i32,
        *const Complex<F>,
        i32,
        *const Complex<F>,
        *mut Complex<F>,
        i32,
    );

    type HerkFn<F> = unsafe extern "C" fn(
        CblasLayout,
        CblasUplo,
        CblasTranspose,
        i32,
        i32,
        F,
        *const Complex<F>,
        i32,
        F,
        *mut Complex<F>,
        i32,
    );

    type Her2kFn<F> = unsafe extern "C" fn(
        CblasLayout,
        CblasUplo,
        CblasTranspose,
        i32,
        i32,
        *const Complex<F>,
        *const Complex<F>,
        i32,
        *const Complex<F>,
        i32,
        F,
        *mut Complex<F>,
        i32,
    );

    const LAYOUTS: [CblasLayout; 2] = [CblasLayout::RowMajor, CblasLayout::ColMajor];
    const SIDES: [CblasSide; 2] = [CblasSide::Left, CblasSide::Right];
    const UPLOS: [CblasUplo; 2] = [CblasUplo::Upper, CblasUplo::Lower];
    const PAD: f64 = 777.0;

    fn c64(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn narrow<F: TestFloat>(z: Complex64) -> Complex<F> {
        Complex::new(F::from_f64(z.re), F::from_f64(z.im))
    }

    fn widen<F: TestFloat>(z: Complex<F>) -> Complex64 {
        c64(z.re.to_f64(), z.im.to_f64())
    }

    /// Alternating non-finite poison (NaN and Inf in both components).
    fn poison(i: usize, j: usize) -> Complex64 {
        if (i + j) % 2 == 0 {
            c64(f64::NAN, f64::NAN)
        } else {
            c64(f64::INFINITY, f64::NEG_INFINITY)
        }
    }

    fn slot(layout: CblasLayout, i: usize, j: usize, ld: usize) -> usize {
        match layout {
            CblasLayout::ColMajor => i + j * ld,
            CblasLayout::RowMajor => i * ld + j,
        }
    }

    /// Stores a logical `rows x cols` matrix into a buffer padded with `PAD - PAD i`.
    fn store<F: TestFloat>(
        layout: CblasLayout,
        rows: usize,
        cols: usize,
        ld: usize,
        value: impl Fn(usize, usize) -> Complex64,
    ) -> Vec<Complex<F>> {
        let len = match layout {
            CblasLayout::ColMajor => ld * cols,
            CblasLayout::RowMajor => ld * rows,
        };
        let mut buf = vec![narrow::<F>(c64(PAD, -PAD)); len];
        for j in 0..cols {
            for i in 0..rows {
                buf[slot(layout, i, j, ld)] = narrow::<F>(value(i, j));
            }
        }
        buf
    }

    /// Every storage slot outside the logical `rows x cols` region must still hold the padding.
    fn assert_padding<F: TestFloat>(
        what: &str,
        layout: CblasLayout,
        buf: &[Complex<F>],
        rows: usize,
        cols: usize,
        ld: usize,
    ) {
        for (idx, v) in buf.iter().enumerate() {
            let (r, c) = match layout {
                CblasLayout::ColMajor => (idx % ld, idx / ld),
                CblasLayout::RowMajor => (idx / ld, idx % ld),
            };
            if r >= rows || c >= cols {
                assert_eq!(
                    widen(*v),
                    c64(PAD, -PAD),
                    "{what}: padding ({r},{c}) was written"
                );
            }
        }
    }

    fn in_tri(uplo: CblasUplo, i: usize, j: usize) -> bool {
        match uplo {
            CblasUplo::Upper => i <= j,
            CblasUplo::Lower => i >= j,
        }
    }

    /// Relative closeness; false for NaN on either side.
    fn close(got: Complex64, want: Complex64, tol: f64) -> bool {
        (got - want).norm() <= tol * (1.0 + want.norm())
    }

    /// Component-wise equality that treats NaN == NaN ("this element must come back as stored").
    fn same_or_nan(got: Complex64, want: Complex64) -> bool {
        let eq = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b;
        eq(got.re, want.re) && eq(got.im, want.im)
    }

    /// Logical Hermitian matrix: real diagonal and `H[j][i] = conj(H[i][j])`.
    fn herm_elem(i: usize, j: usize) -> Complex64 {
        if i == j {
            return c64(1.0 + 0.25 * i as f64, 0.0);
        }
        let (lo, hi) = (i.min(j), i.max(j));
        let g = c64(
            0.25 * ((3 * lo + 5 * hi) % 7) as f64 - 0.5,
            0.25 * ((2 * lo + 7 * hi) % 5) as f64 - 0.5,
        );
        if i < j { g } else { g.conj() }
    }

    fn gen_a(i: usize, j: usize) -> Complex64 {
        c64(
            0.5 * ((2 * i + 3 * j) % 7) as f64 - 1.0,
            0.25 * ((i + 5 * j) % 6) as f64 - 0.5,
        )
    }

    fn gen_b(i: usize, j: usize) -> Complex64 {
        c64(
            0.25 * ((i + 2 * j) % 5) as f64 - 0.5,
            0.25 * ((3 * i + j) % 4) as f64 - 0.25,
        )
    }

    /// C operand; the diagonal deliberately carries an imaginary part (0.3).
    fn gen_c0(i: usize, j: usize) -> Complex64 {
        if i == j {
            c64(0.5 + 0.125 * i as f64, 0.3)
        } else {
            c64(
                0.125 * ((2 * i + j) % 7) as f64,
                0.0625 * ((i + 3 * j) % 5) as f64 - 0.125,
            )
        }
    }

    /// One alpha / beta scenario. `poison_ab` replaces every stored element of A / B by
    /// non-finite values (only legal when alpha == 0); `poison_c` does the same for C
    /// (only legal when beta == 0).
    #[derive(Clone, Copy, Debug)]
    struct Scalars {
        alpha: Complex64,
        beta: Complex64,
        poison_ab: bool,
        poison_c: bool,
    }

    fn scalars(alpha: (f64, f64), beta: (f64, f64), poison_ab: bool, poison_c: bool) -> Scalars {
        Scalars {
            alpha: c64(alpha.0, alpha.1),
            beta: c64(beta.0, beta.1),
            poison_ab,
            poison_c,
        }
    }

    /// HEMM scenarios (complex alpha and beta). The last entry is the plain update.
    fn hemm_cases() -> [Scalars; 5] {
        [
            scalars((0.0, 0.0), (0.0, 0.0), true, true),
            scalars((0.0, 0.0), (2.0, 0.5), true, false),
            scalars((0.0, 0.0), (1.0, 0.0), true, false),
            scalars((1.5, -0.5), (0.0, 0.0), false, true),
            scalars((1.5, -0.5), (0.5, 0.25), false, false),
        ]
    }

    /// HERK / HER2K scenarios (real beta; HERK uses `alpha.re`). The last entry is the
    /// plain update; the one before it has `beta == 1` with a non-zero alpha.
    fn rank_cases() -> [Scalars; 6] {
        [
            scalars((0.0, 0.0), (0.0, 0.0), true, true),
            scalars((0.0, 0.0), (2.0, 0.0), true, false),
            scalars((0.0, 0.0), (1.0, 0.0), true, false),
            scalars((1.5, -0.5), (0.0, 0.0), false, true),
            scalars((1.5, -0.5), (1.0, 0.0), false, false),
            scalars((1.5, -0.5), (0.5, 0.0), false, false),
        ]
    }

    /// HEMM against a dense reference. The (70, 66) shape pushes the order of A past the
    /// native 3M-GEMM threshold; (3, 5) and (5, 3) are the non-square naive shapes.
    fn check_hemm<F: TestFloat>(func: HemmFn<F>, cases: &[Scalars], tol: f64) {
        for sc in cases {
            for layout in LAYOUTS {
                for side in SIDES {
                    for uplo in UPLOS {
                        for (m, n) in [(3usize, 5usize), (5, 3), (70, 66)] {
                            let k = match side {
                                CblasSide::Left => m,
                                CblasSide::Right => n,
                            };
                            let lda = k + 2;
                            let ldb = 3 + match layout {
                                CblasLayout::ColMajor => m,
                                CblasLayout::RowMajor => n,
                            };
                            let ldc = ldb + 1;
                            let a = store::<F>(layout, k, k, lda, |i, j| {
                                if sc.poison_ab {
                                    poison(i, j)
                                } else if i == j {
                                    // The imaginary part of the diagonal must be ignored.
                                    herm_elem(i, j) + c64(0.0, 9.0)
                                } else if in_tri(uplo, i, j) {
                                    herm_elem(i, j)
                                } else {
                                    c64(100.0, 100.0)
                                }
                            });
                            let b = store::<F>(layout, m, n, ldb, |i, j| {
                                if sc.poison_ab {
                                    poison(i + 1, j)
                                } else {
                                    gen_b(i, j)
                                }
                            });
                            let mut c = store::<F>(layout, m, n, ldc, |i, j| {
                                if sc.poison_c {
                                    poison(i, j)
                                } else {
                                    gen_c0(i, j)
                                }
                            });
                            let (alpha, beta) = (narrow::<F>(sc.alpha), narrow::<F>(sc.beta));
                            // SAFETY: buffers hold `ld * cols` / `ld * rows` elements with `ld`
                            // above the validated minimum; alpha / beta are valid locals.
                            unsafe {
                                func(
                                    layout,
                                    side,
                                    uplo,
                                    m as i32,
                                    n as i32,
                                    &alpha,
                                    a.as_ptr(),
                                    lda as i32,
                                    b.as_ptr(),
                                    ldb as i32,
                                    &beta,
                                    c.as_mut_ptr(),
                                    ldc as i32,
                                );
                            }
                            for j in 0..n {
                                for i in 0..m {
                                    let zero = c64(0.0, 0.0);
                                    let prod = if sc.alpha == zero {
                                        zero
                                    } else {
                                        match side {
                                            CblasSide::Left => (0..m)
                                                .map(|l| herm_elem(i, l) * gen_b(l, j))
                                                .sum::<Complex64>(),
                                            CblasSide::Right => (0..n)
                                                .map(|l| gen_b(i, l) * herm_elem(l, j))
                                                .sum::<Complex64>(),
                                        }
                                    };
                                    let c_term = if sc.beta == zero {
                                        zero
                                    } else {
                                        sc.beta * gen_c0(i, j)
                                    };
                                    let want = sc.alpha * prod + c_term;
                                    let got = widen(c[slot(layout, i, j, ldc)]);
                                    assert!(
                                        close(got, want, tol),
                                        "hemm {layout:?}/{side:?}/{uplo:?} {m}x{n} {sc:?}: \
                                         C[{i},{j}] = {got:?}, expected {want:?}"
                                    );
                                }
                            }
                            assert_padding("hemm C", layout, &c, m, n, ldc);
                        }
                    }
                }
            }
        }
    }

    /// HERK / HER2K against a dense reference (NoTrans and ConjTrans, both layouts, n != k
    /// in both directions plus the k == 0 case).
    fn check_rank_k<F: TestFloat>(herk: HerkFn<F>, her2k: Her2kFn<F>, cases: &[Scalars], tol: f64) {
        let zero = c64(0.0, 0.0);
        for sc in cases {
            for layout in LAYOUTS {
                for uplo in UPLOS {
                    for trans in [CblasTranspose::NoTrans, CblasTranspose::ConjTrans] {
                        for (n, k) in [(4usize, 3usize), (3, 4), (5, 0)] {
                            for two_k in [false, true] {
                                let notrans = trans == CblasTranspose::NoTrans;
                                let (ar, ac) = if notrans { (n, k) } else { (k, n) };
                                let lda = 2 + match layout {
                                    CblasLayout::ColMajor => ar,
                                    CblasLayout::RowMajor => ac,
                                };
                                let ldc = n + 3;
                                // u(i, l) = op(A)[i][l], op = identity or conjugate transpose.
                                let ua = |i: usize, l: usize| {
                                    if notrans {
                                        gen_a(i, l)
                                    } else {
                                        gen_a(l, i).conj()
                                    }
                                };
                                let ub = |i: usize, l: usize| {
                                    if notrans {
                                        gen_b(i, l)
                                    } else {
                                        gen_b(l, i).conj()
                                    }
                                };
                                // HERK takes a real alpha.
                                let alpha_eff = if two_k {
                                    sc.alpha
                                } else {
                                    c64(sc.alpha.re, 0.0)
                                };
                                let beta_re = sc.beta.re;
                                let a = store::<F>(layout, ar, ac, lda, |i, j| {
                                    if sc.poison_ab {
                                        poison(i, j)
                                    } else {
                                        gen_a(i, j)
                                    }
                                });
                                let b = store::<F>(layout, ar, ac, lda, |i, j| {
                                    if sc.poison_ab {
                                        poison(i + 1, j)
                                    } else {
                                        gen_b(i, j)
                                    }
                                });
                                let mut c = store::<F>(layout, n, n, ldc, |i, j| {
                                    if sc.poison_c {
                                        poison(i, j)
                                    } else {
                                        gen_c0(i, j)
                                    }
                                });
                                let alpha_c = narrow::<F>(alpha_eff);
                                // SAFETY: A / B hold `ld * cols` (col-major) or `ld * rows`
                                // (row-major) elements and C is n x n with ldc >= n.
                                unsafe {
                                    if two_k {
                                        her2k(
                                            layout,
                                            uplo,
                                            trans,
                                            n as i32,
                                            k as i32,
                                            &alpha_c,
                                            a.as_ptr(),
                                            lda as i32,
                                            b.as_ptr(),
                                            lda as i32,
                                            F::from_f64(beta_re),
                                            c.as_mut_ptr(),
                                            ldc as i32,
                                        );
                                    } else {
                                        herk(
                                            layout,
                                            uplo,
                                            trans,
                                            n as i32,
                                            k as i32,
                                            F::from_f64(alpha_eff.re),
                                            a.as_ptr(),
                                            lda as i32,
                                            F::from_f64(beta_re),
                                            c.as_mut_ptr(),
                                            ldc as i32,
                                        );
                                    }
                                }
                                // Reference quick return: C is not touched at all (not even a
                                // non-real diagonal) for (alpha == 0 || k == 0) && beta == 1.
                                let quick = (alpha_eff == zero || k == 0) && beta_re == 1.0;
                                for j in 0..n {
                                    for i in 0..n {
                                        let got = widen(c[slot(layout, i, j, ldc)]);
                                        if !in_tri(uplo, i, j) {
                                            let before = if sc.poison_c {
                                                poison(i, j)
                                            } else {
                                                gen_c0(i, j)
                                            };
                                            assert!(
                                                same_or_nan(got, before),
                                                "unreferenced C[{i},{j}] changed to {got:?}"
                                            );
                                            continue;
                                        }
                                        let want = if quick {
                                            gen_c0(i, j)
                                        } else {
                                            let acc = if alpha_eff == zero {
                                                zero
                                            } else if two_k {
                                                alpha_eff
                                                    * (0..k)
                                                        .map(|l| ua(i, l) * ub(j, l).conj())
                                                        .sum::<Complex64>()
                                                    + alpha_eff.conj()
                                                        * (0..k)
                                                            .map(|l| ub(i, l) * ua(j, l).conj())
                                                            .sum::<Complex64>()
                                            } else {
                                                alpha_eff
                                                    * (0..k)
                                                        .map(|l| ua(i, l) * ua(j, l).conj())
                                                        .sum::<Complex64>()
                                            };
                                            let c_term = if beta_re == 0.0 {
                                                zero
                                            } else {
                                                gen_c0(i, j) * beta_re
                                            };
                                            let val = acc + c_term;
                                            // Diagonal of a Hermitian result is real.
                                            if i == j { c64(val.re, 0.0) } else { val }
                                        };
                                        assert!(
                                            close(got, want, tol),
                                            "{} {layout:?}/{uplo:?}/{trans:?} n={n} k={k} {sc:?}: \
                                             C[{i},{j}] = {got:?}, expected {want:?}",
                                            if two_k { "her2k" } else { "herk" }
                                        );
                                    }
                                }
                                assert_padding("rank-k C", layout, &c, n, n, ldc);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_issue_9_cblas_zhemm_alpha_beta_zero_semantics() {
        check_hemm::<f64>(cblas_zhemm, &hemm_cases(), 1e-10);
    }

    #[test]
    fn test_issue_9_cblas_chemm_alpha_beta_zero_semantics() {
        check_hemm::<f32>(cblas_chemm, &hemm_cases(), 5e-4);
    }

    #[test]
    fn test_issue_9_cblas_zherk_zher2k_alpha_beta_zero_semantics() {
        check_rank_k::<f64>(cblas_zherk, cblas_zher2k, &rank_cases(), 1e-10);
    }

    #[test]
    fn test_issue_9_cblas_cherk_cher2k_alpha_beta_zero_semantics() {
        check_rank_k::<f32>(cblas_cherk, cblas_cher2k, &rank_cases(), 5e-4);
    }

    /// RowMajor / ColMajor hemm with m != n against the dense reference (plain update only).
    #[test]
    fn test_issue_7_row_major_hemm_non_square_audit() {
        let plain = hemm_cases();
        check_hemm::<f64>(cblas_zhemm, &plain[4..], 1e-10);
        check_hemm::<f32>(cblas_chemm, &plain[4..], 5e-4);
    }

    /// RowMajor / ColMajor herk and her2k (complex alpha, so the `conj(alpha)` of the
    /// row-major mapping matters) with n != k against the dense reference.
    #[test]
    fn test_issue_7_row_major_herk_her2k_non_square_audit() {
        let cases = rank_cases();
        check_rank_k::<f64>(cblas_zherk, cblas_zher2k, &cases[5..], 1e-10);
        check_rank_k::<f32>(cblas_cherk, cblas_cher2k, &cases[5..], 5e-4);
    }
}
