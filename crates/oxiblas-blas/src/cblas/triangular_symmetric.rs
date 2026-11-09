//! CBLAS-compatible C ABI: triangular and symmetric Level 3 routines.
//!
//! The `extern "C"` entry points in this module follow the semantics of
//! reference CBLAS for valid calls. Invalid arguments are a silent no-op:
//! there is no `cblas_xerbla` (see [`crate::cblas`] for the other deviations
//! from reference CBLAS).
//!
//! # Layout
//!
//! CBLAS supports both row-major and column-major layouts. This library
//! uses column-major (Fortran) layout internally, so row-major operations
//! are converted using the identity: op(A) in row-major = op(A^T) in column-major.
//!
//! # Performance
//!
//! The wrappers validate their arguments (invalid input is a silent no-op), map
//! row-major requests onto the column-major natives (flipping `side`/`uplo`,
//! swapping `m`/`n`) and call the native Level 3 kernels of
//! [`crate::level3`] (`trsm_in_place`, `trmm_in_place`, `syrk`, `syr2k`,
//! `symm`) on zero-copy views of the caller's buffers. These are matrix
//! routines, so there are no strided vectors to gather or scatter. `trsm`
//! scales B by `alpha` (one pass over B, skipped for `alpha == 1`) before the
//! in-place solve; `trmm` passes `alpha` to the native, which folds it into
//! its kernels. Both natives are sequential, like `cblas_?gemm`.

use super::types::*;
use super::validate::{
    symm_params_valid, syr2k_params_valid, syrk_params_valid, trmm_params_valid,
};
use crate::level3;

// Level 3 BLAS - TRSM (Triangular Solve with Multiple Right-Hand Sides)
// =============================================================================

/// Double precision TRSM: op(A) * X = alpha * B or X * op(A) = alpha * B.
///
/// Solves a triangular matrix equation. On exit, B is overwritten with X.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from and written to at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtrsm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    // Convert CBLAS parameters to internal format
    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => level3::Side::Left,
                CblasSide::Right => level3::Side::Right,
            },
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans => level3::Trans::Trans,
                CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => level3::Diag::NonUnit,
                CblasDiag::Unit => level3::Diag::Unit,
            },
        ),
        CblasLayout::RowMajor => {
            // Row-major: swap Left/Right and Upper/Lower
            (
                match side {
                    CblasSide::Left => level3::Side::Right,
                    CblasSide::Right => level3::Side::Left,
                },
                match uplo {
                    CblasUplo::Upper => level3::Uplo::Lower,
                    CblasUplo::Lower => level3::Uplo::Upper,
                },
                match transa {
                    CblasTranspose::NoTrans => level3::Trans::NoTrans,
                    CblasTranspose::Trans => level3::Trans::Trans,
                    CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
                },
                match diag {
                    CblasDiag::NonUnit => level3::Diag::NonUnit,
                    CblasDiag::Unit => level3::Diag::Unit,
                },
            )
        }
    };

    // Order of the triangular operand A, taken from the CALLER's side: `m` for a
    // Left operation, `n` for Right (the same `ka` that `trmm_params_valid`
    // checks `lda` against). It must not be derived from `side_internal`: in the
    // row-major case that side is flipped while `m`/`n` are the caller's
    // unswapped values (only the view of B below is swapped), which picked the
    // wrong order whenever m != n.
    let k = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    // Create matrix views
    let a_ref = MatRef::<f64>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m), // Swap for row-major
    };
    let mut b_mut = MatMut::<f64>::new(b, bm, bn, ldb);

    // Reference ?TRSM: alpha == 0 sets B := 0 without reading A or B. The prescale
    // below would turn NaN/Inf in B into NaN (x * 0) and the solve would read A.
    if alpha == 0.0 {
        b_mut.fill_zero();
        return;
    }

    // X := alpha·B first. As in reference ?TRSM, alpha == 1 skips this pass
    // (for real types it would be an exact identity anyway).
    if alpha != 1.0 {
        for j in 0..bn {
            if let Some(col) = b_mut.col_as_slice_mut(j) {
                for v in col.iter_mut() {
                    *v *= alpha;
                }
            }
        }
    }

    // Call internal TRSM
    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A void C ABI cannot surface a `Result`, so — matching the
    // no-op-on-error convention of the wrappers in `basic/` (validate-then-
    // return: a silent no-op, no `cblas_xerbla` call) — an error leaves B as the
    // internal routine left it and the routine returns. In practice only a
    // singular triangular factor is reachable here (the dimensions are correct
    // by construction), which matches reference (x)TRSM: it likewise makes no
    // guarantee about B for singular A.
    if let Err(_err) = level3::trsm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        a_ref,
        b_mut,
    ) {}
}

/// Single precision TRSM: op(A) * X = alpha * B or X * op(A) = alpha * B.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from and written to at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_strsm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *mut f32,
    ldb: i32,
) {
    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => level3::Side::Left,
                CblasSide::Right => level3::Side::Right,
            },
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans => level3::Trans::Trans,
                CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => level3::Diag::NonUnit,
                CblasDiag::Unit => level3::Diag::Unit,
            },
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => level3::Side::Right,
                CblasSide::Right => level3::Side::Left,
            },
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match transa {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans => level3::Trans::Trans,
                CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => level3::Diag::NonUnit,
                CblasDiag::Unit => level3::Diag::Unit,
            },
        ),
    };

    // Order of A from the caller's (unflipped) side; see `cblas_dtrsm`.
    let k = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m),
    };
    let mut b_mut = MatMut::<f32>::new(b, bm, bn, ldb);

    // alpha == 0: B := 0 without reading A or B (see `cblas_dtrsm`).
    if alpha == 0.0 {
        b_mut.fill_zero();
        return;
    }

    // X := alpha·B, skipped for alpha == 1 (see `cblas_dtrsm`).
    if alpha != 1.0 {
        for j in 0..bn {
            if let Some(col) = b_mut.col_as_slice_mut(j) {
                for v in col.iter_mut() {
                    *v *= alpha;
                }
            }
        }
    }

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A void C ABI cannot surface a `Result`, so — matching the
    // no-op-on-error convention of the wrappers in `basic/` (validate-then-
    // return: a silent no-op, no `cblas_xerbla` call) — an error leaves B as the
    // internal routine left it and the routine returns. In practice only a
    // singular triangular factor is reachable here (the dimensions are correct
    // by construction), which matches reference (x)TRSM: it likewise makes no
    // guarantee about B for singular A.
    if let Err(_err) = level3::trsm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        a_ref,
        b_mut,
    ) {}
}

// =============================================================================
// Level 3 BLAS - TRMM (Triangular Matrix-Matrix Multiply)
// =============================================================================

/// Double precision TRMM: B = alpha * op(A) * B or B = alpha * B * op(A).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from and written to at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtrmm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    use crate::level3::{TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo};

    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => TrmmSide::Left,
                CblasSide::Right => TrmmSide::Right,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Upper,
                CblasUplo::Lower => TrmmUplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => TrmmSide::Right,
                CblasSide::Right => TrmmSide::Left,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Lower,
                CblasUplo::Lower => TrmmUplo::Upper,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
    };

    // Order of A from the caller's (unflipped) side; see `cblas_dtrsm`.
    let k = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m),
    };
    let b_mut = MatMut::<f64>::new(b, bm, bn, ldb);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave B untouched.
    if let Err(_err) = level3::trmm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        alpha,
        a_ref,
        b_mut,
    ) {}
}

/// Single precision TRMM: B = alpha * op(A) * B or B = alpha * B * op(A).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from and written to at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_strmm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *mut f32,
    ldb: i32,
) {
    use crate::level3::{TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo};

    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if m <= 0 || n <= 0 || !trmm_params_valid(layout, side, m, n, lda, ldb) {
        return;
    }
    if a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => TrmmSide::Left,
                CblasSide::Right => TrmmSide::Right,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Upper,
                CblasUplo::Lower => TrmmUplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => TrmmSide::Right,
                CblasSide::Right => TrmmSide::Left,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Lower,
                CblasUplo::Lower => TrmmUplo::Upper,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
    };

    // Order of A from the caller's (unflipped) side; see `cblas_dtrsm`.
    let k = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m),
    };
    let b_mut = MatMut::<f32>::new(b, bm, bn, ldb);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave B untouched.
    if let Err(_err) = level3::trmm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        alpha,
        a_ref,
        b_mut,
    ) {}
}

// =============================================================================
// Level 3 BLAS - SYRK (Symmetric Rank-K Update)
// =============================================================================

/// Double precision SYRK: C = alpha * A * A^T + beta * C or C = alpha * A^T * A + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f64`, and point to
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
pub unsafe extern "C" fn cblas_dsyrk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if n <= 0 || !syrk_params_valid(layout, trans, n, k, lda, ldc) {
        return;
    }
    if a.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            // Swap Upper/Lower and NoTrans/Trans for row-major
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    // A dimensions depend on trans
    let (a_rows, a_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, a_rows, a_cols, lda);
    let c_mut = MatMut::<f64>::new(c, n, n, ldc);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave C untouched.
    if let Err(_err) = level3::syrk(uplo_internal, trans_internal, alpha, a_ref, beta, c_mut) {}
}

/// Single precision SYRK: C = alpha * A * A^T + beta * C or C = alpha * A^T * A + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f32`, and point to
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
pub unsafe extern "C" fn cblas_ssyrk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if n <= 0 || !syrk_params_valid(layout, trans, n, k, lda, ldc) {
        return;
    }
    if a.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    let (a_rows, a_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, a_rows, a_cols, lda);
    let c_mut = MatMut::<f32>::new(c, n, n, ldc);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave C untouched.
    if let Err(_err) = level3::syrk(uplo_internal, trans_internal, alpha, a_ref, beta, c_mut) {}
}

// =============================================================================
// Level 3 BLAS - SYR2K (Symmetric Rank-2K Update)
// =============================================================================

/// Double precision SYR2K: C = alpha * A * B^T + alpha * B * A^T + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f64`, and point to
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
pub unsafe extern "C" fn cblas_dsyr2k(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
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
    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if n <= 0 || !syr2k_params_valid(layout, trans, n, k, lda, ldb, ldc) {
        return;
    }
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    let (ab_rows, ab_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, ab_rows, ab_cols, lda);
    let b_ref = MatRef::<f64>::new(b, ab_rows, ab_cols, ldb);
    let c_mut = MatMut::<f64>::new(c, n, n, ldc);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave C untouched.
    if let Err(_err) = level3::syr2k(
        uplo_internal,
        trans_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    ) {}
}

/// Single precision SYR2K: C = alpha * A * B^T + alpha * B * A^T + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f32`, and point to
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
pub unsafe extern "C" fn cblas_ssyr2k(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
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
    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if n <= 0 || !syr2k_params_valid(layout, trans, n, k, lda, ldb, ldc) {
        return;
    }
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    let (ab_rows, ab_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, ab_rows, ab_cols, lda);
    let b_ref = MatRef::<f32>::new(b, ab_rows, ab_cols, ldb);
    let c_mut = MatMut::<f32>::new(c, n, n, ldc);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave C untouched.
    if let Err(_err) = level3::syr2k(
        uplo_internal,
        trans_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    ) {}
}

// =============================================================================
// Level 3 BLAS - SYMM (Symmetric Matrix-Matrix Multiply)
// =============================================================================

/// Double precision SYMM: C = alpha * A * B + beta * C or C = alpha * B * A + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f64`, and point to
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
pub unsafe extern "C" fn cblas_dsymm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    use crate::level3::{Side, Uplo};

    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if m <= 0 || n <= 0 || !symm_params_valid(layout, side, m, n, lda, ldb, ldc) {
        return;
    }
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (side_internal, uplo_internal, bm, bn) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => Side::Left,
                CblasSide::Right => Side::Right,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Upper,
                CblasUplo::Lower => Uplo::Lower,
            },
            m,
            n,
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => Side::Right,
                CblasSide::Right => Side::Left,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Lower,
                CblasUplo::Lower => Uplo::Upper,
            },
            n,
            m,
        ),
    };

    let k = match side_internal {
        Side::Left => bm,
        Side::Right => bn,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, k, k, lda);
    let b_ref = MatRef::<f64>::new(b, bm, bn, ldb);
    let c_mut = MatMut::<f64>::new(c, bm, bn, ldc);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave C untouched.
    if let Err(_err) = level3::symm(
        side_internal,
        uplo_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    ) {}
}

/// Single precision SYMM: C = alpha * A * B + beta * C or C = alpha * B * A + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f32`, and point to
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
pub unsafe extern "C" fn cblas_ssymm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    use crate::level3::{Side, Uplo};

    // Reference BLAS calls xerbla and returns on a bad shape; a void C ABI
    // cannot surface an error code and a panic reaching it would abort the
    // process, so we no-op instead. Without this, `lda as usize` below turns a
    // negative `lda` into `usize::MAX` (and an undersized positive `lda`
    // aliases columns), making the internal `a.add(row + col * lda)` indexing
    // run out of bounds.
    if m <= 0 || n <= 0 || !symm_params_valid(layout, side, m, n, lda, ldb, ldc) {
        return;
    }
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (side_internal, uplo_internal, bm, bn) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => Side::Left,
                CblasSide::Right => Side::Right,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Upper,
                CblasUplo::Lower => Uplo::Lower,
            },
            m,
            n,
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => Side::Right,
                CblasSide::Right => Side::Left,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Lower,
                CblasUplo::Lower => Uplo::Upper,
            },
            n,
            m,
        ),
    };

    let k = match side_internal {
        Side::Left => bm,
        Side::Right => bn,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, k, k, lda);
    let b_ref = MatRef::<f32>::new(b, bm, bn, ldb);
    let c_mut = MatMut::<f32>::new(c, bm, bn, ldc);

    // Inspect the internal result instead of discarding it with a bare
    // `let _ =`. A dimension error is impossible by construction; if one ever
    // occurred we mirror the no-op-on-error convention of `basic/` (a void C ABI
    // cannot surface a `Result`) and leave C untouched.
    if let Err(_err) = level3::symm(
        side_internal,
        uplo_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    ) {}
}

#[cfg(test)]
#[path = "triangular_symmetric_tests.rs"]
mod tests;
