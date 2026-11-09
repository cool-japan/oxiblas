//! Regression tests for CBLAS C-ABI parameter validation.
//!
//! Every test here feeds a malicious or spec-invalid argument to an
//! `extern "C"` entry point that previously passed it straight through to
//! `a.add(row + col * lda)`-style indexing. Before the guards existed, a
//! negative leading dimension became `usize::MAX` (wild out-of-bounds access)
//! and an undersized positive one aliased/overran the operand buffers.
//!
//! The contract being asserted is reference BLAS's: on an invalid argument the
//! routine calls `xerbla` and returns **without touching the operands** — so
//! the checks are "did not crash" *and* "output buffer unchanged". Panicking
//! is not an option: a panic reaching an `extern "C"` boundary aborts the
//! process.

use num_complex::{Complex32, Complex64};
use oxiblas_blas::cblas::*;

/// A 4x4 column-major operand, deliberately much smaller than the dimensions
/// the malicious calls below claim.
fn mat4<T: Copy>(fill: T) -> Vec<T> {
    vec![fill; 16]
}

// =============================================================================
// hermitian.rs — hemm / herk / her2k
// =============================================================================

#[test]
fn zhemm_negative_lda_is_a_noop() {
    let a = mat4(Complex64::new(1.0, 0.0));
    let b = mat4(Complex64::new(1.0, 0.0));
    let mut c = mat4(Complex64::new(7.0, 0.0));
    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::new(1.0, 0.0);

    unsafe {
        cblas_zhemm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            100,
            100,
            &alpha,
            a.as_ptr(),
            -1,
            b.as_ptr(),
            100,
            &beta,
            c.as_mut_ptr(),
            100,
        );
    }

    assert!(c.iter().all(|z| *z == Complex64::new(7.0, 0.0)));
}

#[test]
fn chemm_undersized_lda_is_a_noop() {
    let a = mat4(Complex32::new(1.0, 0.0));
    let b = mat4(Complex32::new(1.0, 0.0));
    let mut c = mat4(Complex32::new(7.0, 0.0));
    let alpha = Complex32::new(1.0, 0.0);
    let beta = Complex32::new(1.0, 0.0);

    // lda = 1 for a 4x4 Hermitian A: distinct (row, col) alias each other and
    // the last column runs past the end of `a`.
    unsafe {
        cblas_chemm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            4,
            4,
            &alpha,
            a.as_ptr(),
            1,
            b.as_ptr(),
            4,
            &beta,
            c.as_mut_ptr(),
            4,
        );
    }

    assert!(c.iter().all(|z| *z == Complex32::new(7.0, 0.0)));
}

#[test]
fn chemm_null_alpha_is_a_noop() {
    let a = mat4(Complex32::new(1.0, 0.0));
    let b = mat4(Complex32::new(1.0, 0.0));
    let mut c = mat4(Complex32::new(7.0, 0.0));
    let beta = Complex32::new(1.0, 0.0);

    unsafe {
        cblas_chemm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            4,
            4,
            core::ptr::null(),
            a.as_ptr(),
            4,
            b.as_ptr(),
            4,
            &beta,
            c.as_mut_ptr(),
            4,
        );
    }

    assert!(c.iter().all(|z| *z == Complex32::new(7.0, 0.0)));
}

#[test]
fn cherk_negative_k_is_a_noop() {
    let a = mat4(Complex32::new(1.0, 0.0));
    let mut c = mat4(Complex32::new(7.0, 0.0));

    unsafe {
        cblas_cherk(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            4,
            -1,
            1.0,
            a.as_ptr(),
            4,
            1.0,
            c.as_mut_ptr(),
            4,
        );
    }

    assert!(c.iter().all(|z| *z == Complex32::new(7.0, 0.0)));
}

#[test]
fn zherk_negative_ldc_is_a_noop() {
    let a = mat4(Complex64::new(1.0, 0.0));
    let mut c = mat4(Complex64::new(7.0, 0.0));

    unsafe {
        cblas_zherk(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            4,
            4,
            1.0,
            a.as_ptr(),
            4,
            1.0,
            c.as_mut_ptr(),
            -1,
        );
    }

    assert!(c.iter().all(|z| *z == Complex64::new(7.0, 0.0)));
}

#[test]
fn zher2k_negative_ldb_is_a_noop() {
    let a = mat4(Complex64::new(1.0, 0.0));
    let b = mat4(Complex64::new(1.0, 0.0));
    let mut c = mat4(Complex64::new(7.0, 0.0));
    let alpha = Complex64::new(1.0, 0.0);

    unsafe {
        cblas_zher2k(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            4,
            4,
            &alpha,
            a.as_ptr(),
            4,
            b.as_ptr(),
            -1,
            1.0,
            c.as_mut_ptr(),
            4,
        );
    }

    assert!(c.iter().all(|z| *z == Complex64::new(7.0, 0.0)));
}

// =============================================================================
// triangular_symmetric.rs — trsm / trmm / syrk / syr2k / symm
// =============================================================================

#[test]
fn dtrsm_negative_lda_is_a_noop() {
    let a = mat4(1.0f64);
    let mut b = mat4(7.0f64);

    unsafe {
        cblas_dtrsm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            100,
            100,
            1.0,
            a.as_ptr(),
            -1,
            b.as_mut_ptr(),
            100,
        );
    }

    assert!(b.iter().all(|v| *v == 7.0));
}

#[test]
fn strmm_undersized_ldb_is_a_noop() {
    let a = mat4(1.0f32);
    let mut b = mat4(7.0f32);

    unsafe {
        cblas_strmm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            4,
            4,
            1.0,
            a.as_ptr(),
            4,
            b.as_mut_ptr(),
            1,
        );
    }

    assert!(b.iter().all(|v| *v == 7.0));
}

#[test]
fn dsyrk_undersized_lda_is_a_noop() {
    let a = mat4(1.0f64);
    let mut c = mat4(7.0f64);

    unsafe {
        cblas_dsyrk(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            4,
            4,
            1.0,
            a.as_ptr(),
            1,
            1.0,
            c.as_mut_ptr(),
            4,
        );
    }

    assert!(c.iter().all(|v| *v == 7.0));
}

#[test]
fn ssyr2k_negative_lda_is_a_noop() {
    let a = mat4(1.0f32);
    let b = mat4(1.0f32);
    let mut c = mat4(7.0f32);

    unsafe {
        cblas_ssyr2k(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            4,
            4,
            1.0,
            a.as_ptr(),
            -1,
            b.as_ptr(),
            4,
            1.0,
            c.as_mut_ptr(),
            4,
        );
    }

    assert!(c.iter().all(|v| *v == 7.0));
}

#[test]
fn dsymm_null_c_is_a_noop() {
    let a = mat4(1.0f64);
    let b = mat4(1.0f64);

    unsafe {
        cblas_dsymm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            4,
            4,
            1.0,
            a.as_ptr(),
            4,
            b.as_ptr(),
            4,
            1.0,
            core::ptr::null_mut(),
            4,
        );
    }
}

#[test]
fn dsymm_negative_ldc_is_a_noop() {
    let a = mat4(1.0f64);
    let b = mat4(1.0f64);
    let mut c = mat4(7.0f64);

    unsafe {
        cblas_dsymm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            4,
            4,
            1.0,
            a.as_ptr(),
            4,
            b.as_ptr(),
            4,
            1.0,
            c.as_mut_ptr(),
            -1,
        );
    }

    assert!(c.iter().all(|v| *v == 7.0));
}

// =============================================================================
// complex_level2.rs — gemv / hemv
// =============================================================================

#[test]
fn cgemv_negative_lda_is_a_noop() {
    let a = mat4(Complex32::new(1.0, 0.0));
    let x = [Complex32::new(1.0, 0.0); 4];
    let mut y = vec![Complex32::new(7.0, 0.0); 4];
    let alpha = Complex32::new(1.0, 0.0);
    let beta = Complex32::new(1.0, 0.0);

    unsafe {
        cblas_cgemv(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            100,
            100,
            &alpha,
            a.as_ptr(),
            -1,
            x.as_ptr(),
            1,
            &beta,
            y.as_mut_ptr(),
            1,
        );
    }

    assert!(y.iter().all(|z| *z == Complex32::new(7.0, 0.0)));
}

#[test]
fn zhemv_zero_increment_is_a_noop() {
    let a = mat4(Complex64::new(1.0, 0.0));
    let x = [Complex64::new(1.0, 0.0); 4];
    let mut y = vec![Complex64::new(7.0, 0.0); 4];
    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::new(1.0, 0.0);

    unsafe {
        cblas_zhemv(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            4,
            &alpha,
            a.as_ptr(),
            4,
            x.as_ptr(),
            0,
            &beta,
            y.as_mut_ptr(),
            1,
        );
    }

    assert!(y.iter().all(|z| *z == Complex64::new(7.0, 0.0)));
}

// =============================================================================
// complex_level1.rs
// =============================================================================

#[test]
fn cscal_null_alpha_is_a_noop() {
    let mut x = vec![Complex32::new(7.0, 0.0); 4];

    unsafe {
        cblas_cscal(4, core::ptr::null(), x.as_mut_ptr(), 1);
    }

    assert!(x.iter().all(|z| *z == Complex32::new(7.0, 0.0)));
}

#[test]
fn test_issue_10_zaxpy_zero_increment_follows_reference_loop() {
    // A zero increment is valid for Level-1 routines (Netlib 3.12): with
    // incx == 0 every iteration reuses x[0], so y[i] = 7 + 1 * 1 = 8. Only the
    // Level-2 routines reject inc == 0 (see `zhemv_zero_increment_is_a_noop`).
    let x = [Complex64::new(1.0, 0.0); 4];
    let mut y = vec![Complex64::new(7.0, 0.0); 4];
    let alpha = Complex64::new(1.0, 0.0);

    unsafe {
        cblas_zaxpy(4, &alpha, x.as_ptr(), 0, y.as_mut_ptr(), 1);
    }

    assert!(y.iter().all(|z| *z == Complex64::new(8.0, 0.0)));
}

#[test]
fn icamax_null_vector_returns_zero() {
    // `iamax_c` seeds its running maximum with an unconditional `*x`, so a
    // null pointer here used to be an immediate null dereference.
    let idx = unsafe { cblas_icamax(4, core::ptr::null(), 1) };
    assert_eq!(idx, 0);
}

#[test]
fn scnrm2_null_vector_returns_zero() {
    let nrm = unsafe { cblas_scnrm2(4, core::ptr::null(), 1) };
    assert_eq!(nrm, 0.0);
}

// =============================================================================
// level2_real.rs — symv / syr / syr2 / ger / trmv / trsv
// =============================================================================

#[test]
fn dsymv_negative_lda_is_a_noop() {
    let a = mat4(1.0f64);
    let x = [1.0f64; 4];
    let mut y = vec![7.0f64; 4];

    unsafe {
        cblas_dsymv(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            100,
            1.0,
            a.as_ptr(),
            -1,
            x.as_ptr(),
            1,
            1.0,
            y.as_mut_ptr(),
            1,
        );
    }

    assert!(y.iter().all(|v| *v == 7.0));
}

#[test]
fn dger_undersized_lda_is_a_noop() {
    let x = [1.0f64; 4];
    let y = [1.0f64; 4];
    let mut a = mat4(7.0f64);

    unsafe {
        cblas_dger(
            CblasLayout::ColMajor,
            4,
            4,
            1.0,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a.as_mut_ptr(),
            1,
        );
    }

    assert!(a.iter().all(|v| *v == 7.0));
}

#[test]
fn dsyr_zero_increment_is_a_noop() {
    let x = [1.0f64; 4];
    let mut a = mat4(7.0f64);

    unsafe {
        cblas_dsyr(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            4,
            1.0,
            x.as_ptr(),
            0,
            a.as_mut_ptr(),
            4,
        );
    }

    assert!(a.iter().all(|v| *v == 7.0));
}

#[test]
fn dtrsv_undersized_lda_is_a_noop() {
    let a = mat4(1.0f64);
    let mut x = vec![7.0f64; 4];

    unsafe {
        cblas_dtrsv(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            4,
            a.as_ptr(),
            1,
            x.as_mut_ptr(),
            1,
        );
    }

    assert!(x.iter().all(|v| *v == 7.0));
}

#[test]
fn strmv_null_a_is_a_noop() {
    let mut x = vec![7.0f32; 4];

    unsafe {
        cblas_strmv(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            4,
            core::ptr::null(),
            4,
            x.as_mut_ptr(),
            1,
        );
    }

    assert!(x.iter().all(|v| *v == 7.0));
}

// =============================================================================
// The guards must not reject legitimate calls
// =============================================================================

#[test]
fn valid_calls_still_compute() {
    // dsymm: C = A * B with A = I (4x4, upper), B = 2*I -> C = 2*I.
    let mut a = [0.0f64; 16];
    let mut b = [0.0f64; 16];
    for i in 0..4 {
        a[i + i * 4] = 1.0;
        b[i + i * 4] = 2.0;
    }
    let mut c = vec![0.0f64; 16];

    unsafe {
        cblas_dsymm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            4,
            4,
            1.0,
            a.as_ptr(),
            4,
            b.as_ptr(),
            4,
            0.0,
            c.as_mut_ptr(),
            4,
        );
    }

    for i in 0..4 {
        assert!(
            (c[i + i * 4] - 2.0).abs() < 1e-12,
            "c[{i},{i}] = {}",
            c[i + i * 4]
        );
    }

    // A negative increment is spec-valid and must still be honored.
    let x = [1.0f64, 2.0, 3.0, 4.0];
    let mut y = vec![0.0f64; 4];
    unsafe {
        cblas_dcopy(4, x.as_ptr(), 1, y.as_mut_ptr(), -1);
    }
    assert_eq!(y, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn row_major_valid_calls_are_not_rejected() {
    // The guards run at the C-ABI boundary against the CALLER's layout and
    // UNFLIPPED arguments, so a row-major call whose leading dimensions are the
    // row lengths (not the column heights) must still be accepted. Getting this
    // backwards would silently turn every row-major call into a no-op.
    let sentinel = 7.0f64;

    // --- SYMM, row-major, Left: A is 2x2, B and C are 2x3 (ldb = ldc = 3).
    let a = [1.0f64, 0.0, 0.0, 1.0]; // identity
    let b = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut c = [sentinel; 6];
    unsafe {
        cblas_dsymm(
            CblasLayout::RowMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            2,
            3,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            3,
            0.0,
            c.as_mut_ptr(),
            3,
        );
    }
    assert!(
        c.iter().any(|v| *v != sentinel),
        "row-major dsymm was wrongly rejected as invalid"
    );

    // --- SYRK, row-major, NoTrans: A is n x k = 3x2, so lda = k = 2.
    let a = [1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut c = [sentinel; 9];
    unsafe {
        cblas_dsyrk(
            CblasLayout::RowMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            3,
            2,
            1.0,
            a.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            3,
        );
    }
    assert!(
        c.iter().any(|v| *v != sentinel),
        "row-major dsyrk was wrongly rejected as invalid"
    );

    // --- TRSM, row-major, Left, m != n: A is 3x3 upper (lda = 3), B is 3x2
    // (ldb = n = 2). A deliberately non-trivial triangular A plus exact value
    // assertions: a bare "something changed" check cannot tell a solved B from a
    // merely alpha-scaled one. T = [[2,1,3],[0,4,1],[0,0,5]] (the 99s are the
    // unreferenced lower triangle); B = T * X0 with X0 = [[1,2],[3,4],[5,6]], so the
    // solve of T X = 2 B must return 2 * X0.
    let a = [2.0f64, 1.0, 3.0, 99.0, 4.0, 1.0, 99.0, 99.0, 5.0];
    let mut b = [20.0f64, 26.0, 17.0, 22.0, 25.0, 30.0];
    // SAFETY: A is 3x3 (lda = 3) and B is 3x2 row-major (ldb = 2); both buffers are exact.
    unsafe {
        cblas_dtrsm(
            CblasLayout::RowMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            3,
            2,
            2.0,
            a.as_ptr(),
            3,
            b.as_mut_ptr(),
            2,
        );
    }
    for (got, want) in b.iter().zip([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]) {
        assert!(
            (got - want).abs() < 1e-12,
            "row-major dtrsm (m != n) returned {b:?}, expected 2 * X0"
        );
    }

    // --- GER, row-major: A is m x n = 2x3, so lda = n = 3.
    let x = [1.0f64, 1.0];
    let y = [1.0f64, 1.0, 1.0];
    let mut a = [sentinel; 6];
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
    assert!(
        a.iter().any(|v| *v != sentinel),
        "row-major dger was wrongly rejected as invalid"
    );

    // --- CGEMV, row-major: A is m x n = 2x3, so lda = n = 3.
    let a = [Complex64::new(1.0, 0.0); 6];
    let x = [Complex64::new(1.0, 0.0); 3];
    let mut y = [Complex64::new(sentinel, 0.0); 2];
    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::new(0.0, 0.0);
    unsafe {
        cblas_zgemv(
            CblasLayout::RowMajor,
            CblasTranspose::NoTrans,
            2,
            3,
            &alpha,
            a.as_ptr(),
            3,
            x.as_ptr(),
            1,
            &beta,
            y.as_mut_ptr(),
            1,
        );
    }
    assert!(
        y.iter().any(|z| *z != Complex64::new(sentinel, 0.0)),
        "row-major zgemv was wrongly rejected as invalid"
    );
}

/// Issue #7: row-major TRMM / TRSM with m != n on the Right side, with exact
/// values. Right-side A is n x n (n = 3 here, lda = 3) and B is m x n = 2x3
/// (ldb = n = 3). The strictly upper entries of A (99) are never referenced.
#[test]
fn test_issue_7_row_major_trmm_trsm_right_side_non_square_exact_values() {
    // T = [[2,0,0],[1,4,0],[3,1,5]] (lower), B = [[1,2,3],[4,5,6]].
    let a = [2.0f64, 99.0, 99.0, 1.0, 4.0, 99.0, 3.0, 1.0, 5.0];
    let x0 = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];

    // B := B * T = [[13,11,15],[31,26,30]]
    let mut b = x0;
    // SAFETY: A is 3x3 (lda = 3) and B is 2x3 row-major (ldb = 3); both buffers are exact.
    unsafe {
        cblas_dtrmm(
            CblasLayout::RowMajor,
            CblasSide::Right,
            CblasUplo::Lower,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            2,
            3,
            1.0,
            a.as_ptr(),
            3,
            b.as_mut_ptr(),
            3,
        );
    }
    assert_eq!(b, [13.0, 11.0, 15.0, 31.0, 26.0, 30.0]);

    // X * T = B must give back X0.
    // SAFETY: same buffers and shapes as the TRMM call above.
    unsafe {
        cblas_dtrsm(
            CblasLayout::RowMajor,
            CblasSide::Right,
            CblasUplo::Lower,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            2,
            3,
            1.0,
            a.as_ptr(),
            3,
            b.as_mut_ptr(),
            3,
        );
    }
    for (got, want) in b.iter().zip(x0) {
        assert!(
            (got - want).abs() < 1e-12,
            "dtrsm Right/Lower returned {b:?}"
        );
    }

    // B := B * T^T = [[2,9,20],[8,24,47]]
    let mut b = x0;
    // SAFETY: A is 3x3 (lda = 3) and B is 2x3 row-major (ldb = 3); both buffers are exact.
    unsafe {
        cblas_dtrmm(
            CblasLayout::RowMajor,
            CblasSide::Right,
            CblasUplo::Lower,
            CblasTranspose::Trans,
            CblasDiag::NonUnit,
            2,
            3,
            1.0,
            a.as_ptr(),
            3,
            b.as_mut_ptr(),
            3,
        );
    }
    assert_eq!(b, [2.0, 9.0, 20.0, 8.0, 24.0, 47.0]);

    // Single precision, Unit diagonal: B := B * unit(T) = B + strictly-lower part.
    // unit(T) = [[1,0,0],[1,1,0],[3,1,1]]; row0 = [1+2+9, 2+3, 3], row1 = [4+5+18, 5+6, 6].
    let af = [2.0f32, 99.0, 99.0, 1.0, 4.0, 99.0, 3.0, 1.0, 5.0];
    let mut bf = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // SAFETY: A is 3x3 (lda = 3) and B is 2x3 row-major (ldb = 3); both buffers are exact.
    unsafe {
        cblas_strmm(
            CblasLayout::RowMajor,
            CblasSide::Right,
            CblasUplo::Lower,
            CblasTranspose::NoTrans,
            CblasDiag::Unit,
            2,
            3,
            1.0,
            af.as_ptr(),
            3,
            bf.as_mut_ptr(),
            3,
        );
    }
    assert_eq!(bf, [12.0, 5.0, 3.0, 27.0, 11.0, 6.0]);
}

/// Issue #7 audit: row-major SYRK / SYR2K only flip `uplo` and `trans` (n and k
/// keep their meaning), so they were never affected by the TRMM/TRSM bug. Pin
/// that down for n != k in both layouts against a dense reference: only the
/// `uplo` triangle of C may change, and leading-dimension padding must survive.
#[test]
fn test_issue_7_row_major_syrk_syr2k_n_ne_k_audit() {
    const PAD: f64 = 777.0;
    let slot = |layout: CblasLayout, i: usize, j: usize, ld: usize| match layout {
        CblasLayout::ColMajor => i + j * ld,
        CblasLayout::RowMajor => i * ld + j,
    };
    let store = |layout: CblasLayout,
                 rows: usize,
                 cols: usize,
                 ld: usize,
                 f: &dyn Fn(usize, usize) -> f64| {
        let len = match layout {
            CblasLayout::ColMajor => ld * cols,
            CblasLayout::RowMajor => ld * rows,
        };
        let mut buf = vec![PAD; len];
        for j in 0..cols {
            for i in 0..rows {
                buf[slot(layout, i, j, ld)] = f(i, j);
            }
        }
        buf
    };

    let (alpha, beta) = (2.0f64, 0.5f64);
    let a_f = |i: usize, j: usize| 0.5 * ((2 * i + 3 * j) % 7) as f64 - 1.0;
    let b_f = |i: usize, j: usize| 0.25 * ((i + 5 * j) % 6) as f64 - 0.5;
    let c_f = |i: usize, j: usize| 0.125 * ((3 * i + j) % 5) as f64;

    for layout in [CblasLayout::RowMajor, CblasLayout::ColMajor] {
        for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
            for trans in [
                CblasTranspose::NoTrans,
                CblasTranspose::Trans,
                CblasTranspose::ConjTrans,
            ] {
                for (n, k) in [(4usize, 2usize), (2, 4), (3, 3)] {
                    // A and B are n x k for NoTrans and k x n otherwise.
                    let notrans = trans == CblasTranspose::NoTrans;
                    let (ar, ac) = if notrans { (n, k) } else { (k, n) };
                    let lda = 2 + match layout {
                        CblasLayout::ColMajor => ar,
                        CblasLayout::RowMajor => ac,
                    };
                    let ldc = n + 3;
                    let opa = |i: usize, l: usize| if notrans { a_f(i, l) } else { a_f(l, i) };
                    let opb = |i: usize, l: usize| if notrans { b_f(i, l) } else { b_f(l, i) };
                    let a = store(layout, ar, ac, lda, &a_f);
                    let b = store(layout, ar, ac, lda, &b_f);
                    let in_tri = |i: usize, j: usize| match uplo {
                        CblasUplo::Upper => i <= j,
                        CblasUplo::Lower => i >= j,
                    };

                    for two_k in [false, true] {
                        let mut c = store(layout, n, n, ldc, &c_f);
                        // SAFETY: A / B hold `ld * cols` (col-major) or `ld * rows`
                        // (row-major) elements and C is n x n with ldc >= n; every
                        // validated leading dimension is respected.
                        unsafe {
                            if two_k {
                                cblas_dsyr2k(
                                    layout,
                                    uplo,
                                    trans,
                                    n as i32,
                                    k as i32,
                                    alpha,
                                    a.as_ptr(),
                                    lda as i32,
                                    b.as_ptr(),
                                    lda as i32,
                                    beta,
                                    c.as_mut_ptr(),
                                    ldc as i32,
                                );
                            } else {
                                cblas_dsyrk(
                                    layout,
                                    uplo,
                                    trans,
                                    n as i32,
                                    k as i32,
                                    alpha,
                                    a.as_ptr(),
                                    lda as i32,
                                    beta,
                                    c.as_mut_ptr(),
                                    ldc as i32,
                                );
                            }
                        }
                        for j in 0..n {
                            for i in 0..n {
                                let want = if in_tri(i, j) {
                                    let acc: f64 = (0..k)
                                        .map(|l| {
                                            if two_k {
                                                opa(i, l) * opb(j, l) + opb(i, l) * opa(j, l)
                                            } else {
                                                opa(i, l) * opa(j, l)
                                            }
                                        })
                                        .sum();
                                    alpha * acc + beta * c_f(i, j)
                                } else {
                                    c_f(i, j)
                                };
                                let got = c[slot(layout, i, j, ldc)];
                                assert!(
                                    (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
                                    "{} {layout:?}/{uplo:?}/{trans:?} n={n} k={k}: \
                                     C[{i},{j}] = {got}, expected {want}",
                                    if two_k { "syr2k" } else { "syrk" }
                                );
                            }
                        }
                        // The padding rows / columns of C must be untouched.
                        for (idx, v) in c.iter().enumerate() {
                            let (r, col) = match layout {
                                CblasLayout::ColMajor => (idx % ldc, idx / ldc),
                                CblasLayout::RowMajor => (idx / ldc, idx % ldc),
                            };
                            if r >= n || col >= n {
                                assert_eq!(*v, PAD, "C padding ({r},{col}) was written");
                            }
                        }
                    }
                }
            }
        }
    }
}
