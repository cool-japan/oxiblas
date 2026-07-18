//! CBLAS-compatible real Level-2 routines.
//!
//! Thin C-ABI wrappers around this crate's native Level-2 kernels
//! (`symv`, `syr`, `syr2`, `trmv`, `trsv`, `ger`). The wrappers own only the
//! marshaling concerns that the native (slice-based, column-major) kernels do
//! not: CBLAS `layout`, `uplo`/`trans`/`diag` enum mapping, and strided
//! (`incx`/`incy`, including negative) vector access.
//!
//! # Layout handling
//!
//! For a symmetric operand, reading a column-major buffer as its transpose is a
//! no-op mathematically, so row-major requests only flip `uplo`. For the
//! general rank-1 update `ger` (non-symmetric), row-major additionally swaps the
//! `x`/`y` operands and the `m`/`n` extents. For the triangular products/solves
//! row-major flips `uplo` and toggles the transpose flag.
//!
//! # Increment convention
//!
//! Following reference BLAS, a negative increment traverses the logical vector
//! from its last element toward the first: logical element `i` lives at pointer
//! offset `i*inc` when `inc >= 0`, and `(i - (n-1))*inc` when `inc < 0` (which
//! is `(n-1-i)*|inc| >= 0`). See `vec_offset`.

use super::types::*;
use crate::level2;
use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{MatMut, MatRef};

// =============================================================================
// Strided-vector helpers
// =============================================================================

/// Pointer offset (in elements) of logical vector index `i` for a length-`n`
/// vector accessed with increment `inc`, honoring the reference BLAS
/// negative-increment convention.
#[inline]
fn vec_offset(i: usize, n: usize, inc: isize) -> isize {
    if inc >= 0 {
        i as isize * inc
    } else {
        (i as isize - (n as isize - 1)) * inc
    }
}

/// Gathers a strided vector into a contiguous `Vec` for the slice-based kernels.
///
/// # Safety
/// `x` must be valid for reads at every offset produced by [`vec_offset`] for
/// `i in 0..n`.
unsafe fn gather<T: Copy>(x: *const T, n: usize, inc: isize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(*x.offset(vec_offset(i, n, inc)));
    }
    v
}

/// Scatters a contiguous slice back to a strided destination vector.
///
/// # Safety
/// `dst` must be valid for writes at every offset produced by [`vec_offset`]
/// for `i in 0..n`, and `src.len()` must be `n`.
unsafe fn scatter<T: Copy>(dst: *mut T, src: &[T], n: usize, inc: isize) {
    for (i, &val) in src.iter().enumerate().take(n) {
        *dst.offset(vec_offset(i, n, inc)) = val;
    }
}

// =============================================================================
// SYMV — symmetric matrix-vector multiply: y = alpha*A*x + beta*y
// =============================================================================

/// Double precision SYMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsymv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    symv_impl(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

/// Single precision SYMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ssymv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    symv_impl(layout, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

#[allow(clippy::too_many_arguments)]
unsafe fn symv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    x: *const T,
    incx: i32,
    beta: T,
    y: *mut T,
    incy: i32,
) {
    let n = n as usize;
    let lda = lda as usize;
    let uplo_internal =
        match (layout, uplo) {
            (CblasLayout::ColMajor, CblasUplo::Upper)
            | (CblasLayout::RowMajor, CblasUplo::Lower) => level2::SymvUplo::Upper,
            (CblasLayout::ColMajor, CblasUplo::Lower)
            | (CblasLayout::RowMajor, CblasUplo::Upper) => level2::SymvUplo::Lower,
        };

    let x_vec = gather(x, n, incx as isize);
    let mut y_vec = gather(y, n, incy as isize);
    // SAFETY: `a` is a valid pointer to an n x n matrix with leading
    // dimension `lda >= n`, matching the CBLAS contract for this routine.
    let a_ref = unsafe { MatRef::<T>::new(a, n, n, lda) };
    let _ = level2::symv(uplo_internal, alpha, a_ref, &x_vec, beta, &mut y_vec);
    scatter(y, &y_vec, n, incy as isize);
}

// =============================================================================
// SYR — symmetric rank-1 update: A = alpha*x*x^T + A
// =============================================================================

/// Double precision SYR.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsyr(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    a: *mut f64,
    lda: i32,
) {
    if n <= 0 {
        return;
    }
    syr_impl(layout, uplo, n, alpha, x, incx, a, lda);
}

/// Single precision SYR.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ssyr(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    a: *mut f32,
    lda: i32,
) {
    if n <= 0 {
        return;
    }
    syr_impl(layout, uplo, n, alpha, x, incx, a, lda);
}

#[allow(clippy::too_many_arguments)]
unsafe fn syr_impl<T: Real>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: T,
    x: *const T,
    incx: i32,
    a: *mut T,
    lda: i32,
) {
    let n = n as usize;
    let lda = lda as usize;
    let uplo_internal =
        match (layout, uplo) {
            (CblasLayout::ColMajor, CblasUplo::Upper)
            | (CblasLayout::RowMajor, CblasUplo::Lower) => level2::SyrUplo::Upper,
            (CblasLayout::ColMajor, CblasUplo::Lower)
            | (CblasLayout::RowMajor, CblasUplo::Upper) => level2::SyrUplo::Lower,
        };
    let x_vec = gather(x, n, incx as isize);
    let a_mut = MatMut::<T>::new(a, n, n, lda);
    let _ = level2::syr(uplo_internal, alpha, &x_vec, a_mut);
}

// =============================================================================
// SYR2 — symmetric rank-2 update: A = alpha*(x*y^T + y*x^T) + A
// =============================================================================

/// Double precision SYR2.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsyr2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    a: *mut f64,
    lda: i32,
) {
    if n <= 0 {
        return;
    }
    syr2_impl(layout, uplo, n, alpha, x, incx, y, incy, a, lda);
}

/// Single precision SYR2.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ssyr2(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
    a: *mut f32,
    lda: i32,
) {
    if n <= 0 {
        return;
    }
    syr2_impl(layout, uplo, n, alpha, x, incx, y, incy, a, lda);
}

#[allow(clippy::too_many_arguments)]
unsafe fn syr2_impl<T: Real>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: T,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    a: *mut T,
    lda: i32,
) {
    let n = n as usize;
    let lda = lda as usize;
    let uplo_internal =
        match (layout, uplo) {
            (CblasLayout::ColMajor, CblasUplo::Upper)
            | (CblasLayout::RowMajor, CblasUplo::Lower) => level2::Syr2Uplo::Upper,
            (CblasLayout::ColMajor, CblasUplo::Lower)
            | (CblasLayout::RowMajor, CblasUplo::Upper) => level2::Syr2Uplo::Lower,
        };
    let x_vec = gather(x, n, incx as isize);
    let y_vec = gather(y, n, incy as isize);
    let a_mut = MatMut::<T>::new(a, n, n, lda);
    let _ = level2::syr2(uplo_internal, alpha, &x_vec, &y_vec, a_mut);
}

// =============================================================================
// GER — general rank-1 update: A = alpha*x*y^T + A
// =============================================================================

/// Double precision GER.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dger(
    layout: CblasLayout,
    m: i32,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    a: *mut f64,
    lda: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    ger_impl(layout, m, n, alpha, x, incx, y, incy, a, lda);
}

/// Single precision GER.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sger(
    layout: CblasLayout,
    m: i32,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
    a: *mut f32,
    lda: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    ger_impl(layout, m, n, alpha, x, incx, y, incy, a, lda);
}

#[allow(clippy::too_many_arguments)]
unsafe fn ger_impl<T: Field>(
    layout: CblasLayout,
    m: i32,
    n: i32,
    alpha: T,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    a: *mut T,
    lda: i32,
) {
    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let x_vec = gather(x, m, incx as isize);
    let y_vec = gather(y, n, incy as isize);

    match layout {
        CblasLayout::ColMajor => {
            let a_mut = MatMut::<T>::new(a, m, n, lda);
            level2::ger(alpha, &x_vec, &y_vec, a_mut);
        }
        CblasLayout::RowMajor => {
            // A(row-major, m x n) reinterpreted column-major is A^T (n x m), and
            // (alpha*x*y^T)^T = alpha*y*x^T, so swap the operands and extents.
            let a_mut = MatMut::<T>::new(a, n, m, lda);
            level2::ger(alpha, &y_vec, &x_vec, a_mut);
        }
    }
}

// =============================================================================
// TRMV — triangular matrix-vector multiply: x = op(A)*x
// =============================================================================

/// Double precision TRMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtrmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    if n <= 0 {
        return;
    }
    trmv_impl(layout, uplo, trans, diag, n, a, lda, x, incx);
}

/// Single precision TRMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_strmv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
) {
    if n <= 0 {
        return;
    }
    trmv_impl(layout, uplo, trans, diag, n, a, lda, x, incx);
}

#[allow(clippy::too_many_arguments)]
unsafe fn trmv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const T,
    lda: i32,
    x: *mut T,
    incx: i32,
) {
    let n = n as usize;
    let lda = lda as usize;

    // Row-major flips uplo and toggles the transpose (reinterpreting the buffer
    // as A^T). For real types ConjTrans is equivalent to Trans.
    let (uplo_internal, op_internal) = match layout {
        CblasLayout::ColMajor => (map_trmv_uplo(uplo), map_trmv_op(trans)),
        CblasLayout::RowMajor => (map_trmv_uplo(flip_uplo(uplo)), toggle_trmv_op(trans)),
    };
    let diag_internal = match diag {
        CblasDiag::NonUnit => level2::DiagKind::NonUnit,
        CblasDiag::Unit => level2::DiagKind::Unit,
    };

    let mut x_vec = gather(x, n, incx as isize);
    // SAFETY: `a` is a valid pointer to an n x n matrix with leading
    // dimension `lda >= n`, matching the CBLAS contract for this routine.
    let a_ref = unsafe { MatRef::<T>::new(a, n, n, lda) };
    let _ = level2::trmv(a_ref, &mut x_vec, uplo_internal, op_internal, diag_internal);
    scatter(x, &x_vec, n, incx as isize);
}

// =============================================================================
// TRSV — triangular solve: op(A)*x = b (b overwritten by x)
// =============================================================================

/// Double precision TRSV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtrsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    if n <= 0 {
        return;
    }
    trsv_impl(layout, uplo, trans, diag, n, a, lda, x, incx);
}

/// Single precision TRSV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_strsv(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
) {
    if n <= 0 {
        return;
    }
    trsv_impl(layout, uplo, trans, diag, n, a, lda, x, incx);
}

#[allow(clippy::too_many_arguments)]
unsafe fn trsv_impl<T: Field>(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    n: i32,
    a: *const T,
    lda: i32,
    x: *mut T,
    incx: i32,
) {
    let n = n as usize;
    let lda = lda as usize;

    // Column-major: mode = uplo, trans passes through directly (NoTrans /
    // Trans / ConjTrans all map faithfully now that trsv_in_place supports
    // level2::TrsvTrans::ConjTrans natively).
    //
    // Row-major: flip uplo and reinterpret the buffer as A^T (swap
    // NoTrans<->Trans). NOTE: this reinterpretation trick only swaps
    // transposition, not conjugation, so a row-major ConjTrans request
    // cannot be expressed as a single TrsvTrans variant here (it would need
    // "conjugate without transposing", which the enum does not have) — this
    // pre-existing limitation (the row-major path already silently dropped
    // conjugation for both Trans and ConjTrans before this fix) is preserved
    // rather than expanded in scope; row-major ConjTrans still does not
    // conjugate. This is a known follow-up, not a regression.
    let (mode, trsv_trans) = match layout {
        CblasLayout::ColMajor => (
            map_trsv_mode(uplo),
            match trans {
                CblasTranspose::NoTrans => level2::TrsvTrans::NoTrans,
                CblasTranspose::Trans => level2::TrsvTrans::Trans,
                CblasTranspose::ConjTrans => level2::TrsvTrans::ConjTrans,
            },
        ),
        CblasLayout::RowMajor => (
            map_trsv_mode(flip_uplo(uplo)),
            if matches!(trans, CblasTranspose::NoTrans) {
                level2::TrsvTrans::Trans
            } else {
                level2::TrsvTrans::NoTrans
            },
        ),
    };
    let trsv_diag = match diag {
        CblasDiag::NonUnit => level2::TrsvDiag::NonUnit,
        CblasDiag::Unit => level2::TrsvDiag::Unit,
    };

    let mut x_vec = gather(x, n, incx as isize);

    // SAFETY: `a` is a valid pointer to an n x n matrix with leading
    // dimension `lda >= n`, matching the CBLAS contract for this routine.
    let a_ref = unsafe { MatRef::<T>::new(a, n, n, lda) };
    let _ = level2::trsv_in_place(a_ref, &mut x_vec, mode, trsv_trans, trsv_diag);
    scatter(x, &x_vec, n, incx as isize);
}

// =============================================================================
// Enum-mapping helpers
// =============================================================================

#[inline]
fn flip_uplo(uplo: CblasUplo) -> CblasUplo {
    match uplo {
        CblasUplo::Upper => CblasUplo::Lower,
        CblasUplo::Lower => CblasUplo::Upper,
    }
}

#[inline]
fn map_trmv_uplo(uplo: CblasUplo) -> level2::TrmvUplo {
    match uplo {
        CblasUplo::Upper => level2::TrmvUplo::Upper,
        CblasUplo::Lower => level2::TrmvUplo::Lower,
    }
}

#[inline]
fn map_trmv_op(trans: CblasTranspose) -> level2::TrmvOp {
    match trans {
        CblasTranspose::NoTrans => level2::TrmvOp::NoTrans,
        CblasTranspose::Trans => level2::TrmvOp::Trans,
        CblasTranspose::ConjTrans => level2::TrmvOp::ConjTrans,
    }
}

/// Toggles the operation for a row-major triangular product. Because the buffer
/// is reinterpreted as `A^T`, `NoTrans` becomes `Trans` and vice versa;
/// `ConjTrans` (equivalent to `Trans` for real types) toggles to `NoTrans`.
#[inline]
fn toggle_trmv_op(trans: CblasTranspose) -> level2::TrmvOp {
    match trans {
        CblasTranspose::NoTrans => level2::TrmvOp::Trans,
        CblasTranspose::Trans | CblasTranspose::ConjTrans => level2::TrmvOp::NoTrans,
    }
}

#[inline]
fn map_trsv_mode(uplo: CblasUplo) -> level2::TriangularMode {
    match uplo {
        CblasUplo::Upper => level2::TriangularMode::Upper,
        CblasUplo::Lower => level2::TriangularMode::Lower,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cblas_dsymv_matches_native() {
        // A symmetric 3x3, lower stored, column-major.
        let a = [
            2.0f64, 1.0, 0.5, // col 0
            0.0, 3.0, -1.0, // col 1 (upper part unused for Lower)
            0.0, 0.0, 4.0, // col 2
        ];
        let x = [1.0f64, -2.0, 0.5];
        let mut y = [0.25f64, 1.0, -0.5];
        let mut y_ref = y;
        let alpha = 1.5;
        let beta = 0.75;

        unsafe {
            cblas_dsymv(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                3,
                alpha,
                a.as_ptr(),
                3,
                x.as_ptr(),
                1,
                beta,
                y.as_mut_ptr(),
                1,
            );
            let a_ref = MatRef::<f64>::new(a.as_ptr(), 3, 3, 3);
            let _ = level2::symv(level2::SymvUplo::Lower, alpha, a_ref, &x, beta, &mut y_ref);
        }
        for (g, r) in y.iter().zip(y_ref.iter()) {
            assert!((g - r).abs() < 1e-12, "got {g} ref {r}");
        }
    }

    #[test]
    fn test_cblas_dsymv_strided_and_negative_inc() {
        // Compare an incx=2 / incy=-1 call against a manual dense reference.
        let a = [2.0f64, 1.0, 1.0, 3.0]; // 2x2 symmetric, full
        // x logical = [1, -1], stored with incx=2: [1, _, -1]
        let x_storage = [1.0f64, 99.0, -1.0];
        // y logical = [0.5, 2.0], stored with incy=-1 (reversed): index0 -> offset1
        let mut y_storage = [2.0f64, 0.5];
        let alpha = 1.0;
        let beta = 1.0;

        unsafe {
            cblas_dsymv(
                CblasLayout::ColMajor,
                CblasUplo::Upper,
                2,
                alpha,
                a.as_ptr(),
                2,
                x_storage.as_ptr(),
                2,
                beta,
                y_storage.as_mut_ptr(),
                -1,
            );
        }
        // Manual: y_logical = alpha*A*x + beta*y_logical
        // A = [[2,1],[1,3]], x=[1,-1] -> A*x = [1, -2]
        // y_logical_in = [0.5, 2.0] -> y_out = [1+0.5, -2+2.0] = [1.5, 0.0]
        // negative incy: logical index 0 at storage offset (0-1)*-1 = 1,
        // logical index 1 at storage offset (1-1)*-1 = 0.
        assert!((y_storage[1] - 1.5).abs() < 1e-12); // logical 0
        assert!((y_storage[0] - 0.0).abs() < 1e-12); // logical 1
    }

    #[test]
    fn test_cblas_dsymv_rowmajor() {
        // Row-major upper == column-major lower for a symmetric matrix; verify
        // the wrapper produces the same result as the column-major call.
        // Row-major upper storage of A=[[2,1],[1,3]]: (0,0)@0, (0,1)@1, (1,1)@3;
        // index 2 is the unreferenced lower slot.
        let a_upper_row = [2.0f64, 1.0, 0.0, 3.0];
        let a_full = [2.0f64, 1.0, 1.0, 3.0];
        let x = [1.0f64, 2.0];
        let mut y_row = [0.0f64, 0.0];
        let mut y_col = [0.0f64, 0.0];
        unsafe {
            cblas_dsymv(
                CblasLayout::RowMajor,
                CblasUplo::Upper,
                2,
                1.0,
                a_upper_row.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y_row.as_mut_ptr(),
                1,
            );
            cblas_dsymv(
                CblasLayout::ColMajor,
                CblasUplo::Upper,
                2,
                1.0,
                a_full.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y_col.as_mut_ptr(),
                1,
            );
        }
        for (r, c) in y_row.iter().zip(y_col.iter()) {
            assert!((r - c).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_dsyr_matches_native() {
        let a_init = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = [1.0f64, 2.0, 3.0];
        let mut a = a_init;
        let mut a_ref_buf = a_init;
        unsafe {
            cblas_dsyr(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                3,
                2.0,
                x.as_ptr(),
                1,
                a.as_mut_ptr(),
                3,
            );
            let a_mut = MatMut::<f64>::new(a_ref_buf.as_mut_ptr(), 3, 3, 3);
            let _ = level2::syr(level2::SyrUplo::Lower, 2.0, &x, a_mut);
        }
        for (g, r) in a.iter().zip(a_ref_buf.iter()) {
            assert!((g - r).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_dsyr2_matches_native() {
        let a_init = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = [1.0f64, 2.0, 3.0];
        let y = [0.5f64, -1.0, 2.0];
        let mut a = a_init;
        let mut a_ref_buf = a_init;
        unsafe {
            cblas_dsyr2(
                CblasLayout::ColMajor,
                CblasUplo::Upper,
                3,
                1.5,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                3,
            );
            let a_mut = MatMut::<f64>::new(a_ref_buf.as_mut_ptr(), 3, 3, 3);
            let _ = level2::syr2(level2::Syr2Uplo::Upper, 1.5, &x, &y, a_mut);
        }
        for (g, r) in a.iter().zip(a_ref_buf.iter()) {
            assert!((g - r).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_dger_matches_native() {
        let a_init = [0.0f64; 6]; // 2x3 column-major
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0, 5.0];
        let mut a = a_init;
        let mut a_ref_buf = a_init;
        unsafe {
            cblas_dger(
                CblasLayout::ColMajor,
                2,
                3,
                2.0,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                2,
            );
            let a_mut = MatMut::<f64>::new(a_ref_buf.as_mut_ptr(), 2, 3, 2);
            level2::ger(2.0, &x, &y, a_mut);
        }
        for (g, r) in a.iter().zip(a_ref_buf.iter()) {
            assert!((g - r).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_dger_rowmajor() {
        // Row-major 2x3: A[i][j] = alpha*x[i]*y[j].
        let mut a = [0.0f64; 6]; // row-major 2x3, row stride 3
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0, 5.0];
        unsafe {
            cblas_dger(
                CblasLayout::RowMajor,
                2,
                3,
                1.0,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                3,
            );
        }
        // Expected row-major: A[i*3+j] = x[i]*y[j]
        let expected = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
        for (g, e) in a.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "got {g} exp {e}");
        }
    }

    #[test]
    fn test_cblas_dtrmv_matches_native() {
        // Lower triangular 3x3, column-major, non-unit diagonal.
        let a = [2.0f64, 1.0, 3.0, 0.0, 4.0, -1.0, 0.0, 0.0, 5.0];
        let x = [1.0f64, 2.0, 3.0];
        let mut xg = x;
        let mut xr = x;
        unsafe {
            cblas_dtrmv(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                CblasDiag::NonUnit,
                3,
                a.as_ptr(),
                3,
                xg.as_mut_ptr(),
                1,
            );
            let a_ref = MatRef::<f64>::new(a.as_ptr(), 3, 3, 3);
            let _ = level2::trmv(
                a_ref,
                &mut xr,
                level2::TrmvUplo::Lower,
                level2::TrmvOp::NoTrans,
                level2::DiagKind::NonUnit,
            );
        }
        for (g, r) in xg.iter().zip(xr.iter()) {
            assert!((g - r).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_dtrsv_roundtrip_nonunit() {
        // Solve A x = b then multiply back with TRMV to recover b.
        let a = [2.0f64, 1.0, 3.0, 0.0, 4.0, -1.0, 0.0, 0.0, 5.0]; // lower, col-major
        let b = [4.0f64, 10.0, 8.0];
        let mut x = b;
        unsafe {
            cblas_dtrsv(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                CblasDiag::NonUnit,
                3,
                a.as_ptr(),
                3,
                x.as_mut_ptr(),
                1,
            );
            // multiply back
            cblas_dtrmv(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                CblasDiag::NonUnit,
                3,
                a.as_ptr(),
                3,
                x.as_mut_ptr(),
                1,
            );
        }
        for (g, r) in x.iter().zip(b.iter()) {
            assert!((g - r).abs() < 1e-10, "got {g} exp {r}");
        }
    }

    #[test]
    fn test_cblas_dtrsv_unit_diag() {
        // Unit lower-triangular solve: diagonal entries are treated as 1,
        // ignoring the stored diagonal values.
        // A_unit = [[1,0,0],[1,1,0],[3,-1,1]] ; b = [1, 3, 4]
        // Forward substitution: x0=1, x1=3-1=2, x2=4-3*1+1*2=3
        let a = [9.0f64, 1.0, 3.0, 0.0, 9.0, -1.0, 0.0, 0.0, 9.0]; // diag values are junk (9)
        let b = [1.0f64, 3.0, 4.0];
        let mut x = b;
        unsafe {
            cblas_dtrsv(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                CblasDiag::Unit,
                3,
                a.as_ptr(),
                3,
                x.as_mut_ptr(),
                1,
            );
        }
        let expected = [1.0f64, 2.0, 3.0];
        for (g, e) in x.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "got {g} exp {e}");
        }
    }
}
