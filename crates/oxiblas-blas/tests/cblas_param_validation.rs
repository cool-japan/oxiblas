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
//! is not an option: unwinding across an `extern "C"` boundary is UB.

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
fn zaxpy_zero_increment_is_a_noop() {
    let x = [Complex64::new(1.0, 0.0); 4];
    let mut y = vec![Complex64::new(7.0, 0.0); 4];
    let alpha = Complex64::new(1.0, 0.0);

    unsafe {
        cblas_zaxpy(4, &alpha, x.as_ptr(), 0, y.as_mut_ptr(), 1);
    }

    assert!(y.iter().all(|z| *z == Complex64::new(7.0, 0.0)));
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

    // --- TRSM, row-major, Left: A is 3x3 (lda = 3), B is 3x2 (ldb = n = 2).
    let a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut b = [sentinel; 6];
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
    assert!(
        b.iter().any(|v| *v != sentinel),
        "row-major dtrsm was wrongly rejected as invalid"
    );

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
