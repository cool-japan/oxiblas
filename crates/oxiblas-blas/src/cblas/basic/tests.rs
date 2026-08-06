//! Tests for the CBLAS `basic` module (all four sub-areas).
//!
//! Regression tests for the negative-increment / beta==0 / parameter-
//! validation hardening. Every strided routine family is exercised with a
//! negative increment on a NON-palindromic vector so that forward vs.
//! reference back-to-front traversal produce observably different results,
//! and each is checked against an independent, manually reverse-indexed
//! reference (not a copy of the implementation under test).

use super::super::types::*;
use super::*;
use num_complex::{Complex32, Complex64};

const EPS: f64 = 1e-12;
const EPS_F32: f32 = 1e-6;

// ---- Level 1: two-vector routines (ddot/sdot/axpy/copy/swap) ----------

#[test]
fn test_ddot_negative_incx() {
    let x = [1.0f64, 2.0, 3.0, 4.0];
    let y = [10.0f64, 20.0, 30.0, 40.0];
    // Manual reference: incx=-1 reverses x, incy=+1 forward y.
    let expected: f64 = (0..4).map(|i| x[3 - i] * y[i]).sum();
    let got = unsafe { cblas_ddot(4, x.as_ptr(), -1, y.as_ptr(), 1) };
    assert!(
        (got - expected).abs() < EPS,
        "got {got}, expected {expected}"
    );
    // Sanity: differs from the naive forward dot (proves the fix matters).
    let forward: f64 = (0..4).map(|i| x[i] * y[i]).sum();
    assert!((expected - forward).abs() > 1.0);
}

#[test]
fn test_ddot_negative_both_mixed_stride() {
    // incx=-1, incy=-2 on independent, non-symmetric data.
    let x = [1.0f64, 2.0, 3.0]; // logical: x[2], x[1], x[0]
    let y = [5.0f64, 6.0, 7.0, 8.0, 9.0]; // incy=-2 ⇒ y[4], y[2], y[0]
    let expected = x[2] * y[4] + x[1] * y[2] + x[0] * y[0];
    let got = unsafe { cblas_ddot(3, x.as_ptr(), -1, y.as_ptr(), -2) };
    assert!(
        (got - expected).abs() < EPS,
        "got {got}, expected {expected}"
    );
}

#[test]
fn test_sdot_negative_incx() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = [10.0f32, 20.0, 30.0, 40.0];
    let expected: f32 = (0..4).map(|i| x[3 - i] * y[i]).sum();
    let got = unsafe { cblas_sdot(4, x.as_ptr(), -1, y.as_ptr(), 1) };
    assert!((got - expected).abs() < EPS_F32);
}

#[test]
fn test_daxpy_negative_incx() {
    let alpha = 2.0f64;
    let x = [1.0f64, 2.0, 3.0];
    let mut y = [10.0f64, 20.0, 30.0];
    let mut expected = y;
    // incx=-1 reverses x; incy=+1 forward y.
    for i in 0..3 {
        expected[i] += alpha * x[2 - i];
    }
    unsafe { cblas_daxpy(3, alpha, x.as_ptr(), -1, y.as_mut_ptr(), 1) };
    for i in 0..3 {
        assert!((y[i] - expected[i]).abs() < EPS, "index {i}");
    }
    // Observably different from forward axpy at both ends.
    assert!((y[0] - 16.0).abs() < EPS && (y[2] - 32.0).abs() < EPS);
}

#[test]
fn test_saxpy_negative_incy() {
    let alpha = 3.0f32;
    let x = [1.0f32, 2.0, 3.0];
    let mut y = [10.0f32, 20.0, 30.0];
    let mut expected = y;
    // incx=+1 forward x; incy=-1 reverses y writes.
    for i in 0..3 {
        expected[2 - i] += alpha * x[i];
    }
    unsafe { cblas_saxpy(3, alpha, x.as_ptr(), 1, y.as_mut_ptr(), -1) };
    for i in 0..3 {
        assert!((y[i] - expected[i]).abs() < EPS_F32, "index {i}");
    }
}

#[test]
fn test_dcopy_negative_incx_reverses() {
    let x = [1.0f64, 2.0, 3.0, 4.0];
    let mut y = [0.0f64; 4];
    unsafe { cblas_dcopy(4, x.as_ptr(), -1, y.as_mut_ptr(), 1) };
    // y = reverse(x)
    assert_eq!(y, [4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_scopy_negative_incx_reverses() {
    let x = [1.0f32, 2.0, 3.0];
    let mut y = [0.0f32; 3];
    unsafe { cblas_scopy(3, x.as_ptr(), -1, y.as_mut_ptr(), 1) };
    assert_eq!(y, [3.0, 2.0, 1.0]);
}

#[test]
fn test_dswap_negative_incx() {
    let mut x = [1.0f64, 2.0, 3.0];
    let mut y = [10.0f64, 20.0, 30.0];
    // Reference: swap x[2-i] <-> y[i] for i = 0,1,2.
    unsafe { cblas_dswap(3, x.as_mut_ptr(), -1, y.as_mut_ptr(), 1) };
    assert_eq!(x, [30.0, 20.0, 10.0]);
    assert_eq!(y, [3.0, 2.0, 1.0]);
}

#[test]
fn test_sswap_negative_incx() {
    let mut x = [1.0f32, 2.0, 3.0];
    let mut y = [10.0f32, 20.0, 30.0];
    unsafe { cblas_sswap(3, x.as_mut_ptr(), -1, y.as_mut_ptr(), 1) };
    assert_eq!(x, [30.0, 20.0, 10.0]);
    assert_eq!(y, [3.0, 2.0, 1.0]);
}

// ---- Level 1: single-vector routines (scal/nrm2/asum/iamax) -----------

#[test]
fn test_dscal_negative_incx_is_noop() {
    // Reference DSCAL is a no-op for incx <= 0.
    let mut x = [1.0f64, 2.0, 3.0];
    unsafe { cblas_dscal(3, 5.0, x.as_mut_ptr(), -1) };
    assert_eq!(x, [1.0, 2.0, 3.0]);
}

#[test]
fn test_dscal_positive_strided() {
    // Only elements at stride-2 positions are scaled.
    let mut x = [1.0f64, 9.0, 2.0, 9.0];
    unsafe { cblas_dscal(2, 10.0, x.as_mut_ptr(), 2) };
    assert_eq!(x, [10.0, 9.0, 20.0, 9.0]);
}

#[test]
fn test_dnrm2_negative_incx_is_zero() {
    let x = [3.0f64, 4.0];
    let got = unsafe { cblas_dnrm2(2, x.as_ptr(), -1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_dnrm2_strided() {
    let x = [3.0f64, 99.0, 4.0, 99.0];
    let got = unsafe { cblas_dnrm2(2, x.as_ptr(), 2) };
    assert!((got - 5.0).abs() < EPS);
}

#[test]
fn test_dnrm2_strided_no_overflow() {
    // Blue's scaled accumulation must not overflow where a naive Σx² would.
    // Squaring 1e200 overflows f64, but ||[1e200, 1e200]||_2 = √2·1e200.
    let x = [1e200f64, 0.0, 1e200, 0.0];
    let got = unsafe { cblas_dnrm2(2, x.as_ptr(), 2) };
    let expected = std::f64::consts::SQRT_2 * 1e200;
    assert!(got.is_finite(), "norm overflowed to {got}");
    assert!((got - expected).abs() / expected < 1e-12);
}

#[test]
fn test_snrm2_strided_no_overflow() {
    let x = [1e20f32, 0.0, 1e20, 0.0];
    let got = unsafe { cblas_snrm2(2, x.as_ptr(), 2) };
    let expected = std::f32::consts::SQRT_2 * 1e20;
    assert!(got.is_finite());
    assert!((got - expected).abs() / expected < 1e-5);
}

#[test]
fn test_dasum_negative_incx_is_zero() {
    let x = [1.0f64, -2.0, 3.0];
    let got = unsafe { cblas_dasum(3, x.as_ptr(), -1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_dasum_strided() {
    let x = [1.0f64, 9.0, -2.0, 9.0, 3.0, 9.0];
    let got = unsafe { cblas_dasum(3, x.as_ptr(), 2) };
    assert!((got - 6.0).abs() < EPS);
}

#[test]
fn test_idamax_negative_incx_is_zero() {
    let x = [1.0f64, -5.0, 3.0];
    let got = unsafe { cblas_idamax(3, x.as_ptr(), -1) };
    assert_eq!(got, 0);
}

#[test]
fn test_idamax_strided() {
    // Strided view over [1, -5, 3]; max |.| is at logical index 1.
    let x = [1.0f64, 9.0, -5.0, 9.0, 3.0, 9.0];
    let got = unsafe { cblas_idamax(3, x.as_ptr(), 2) };
    assert_eq!(got, 1);
}

// ---- Level 1: null-pointer / zero-increment guards ---------------------
//
// Wave 1 hardened level2_real.rs / complex_level1.rs / complex_level2.rs /
// hermitian.rs / triangular_symmetric.rs with `is_null()` and `inc_valid()`
// checks; ddot..sswap (this file) were the one Level-1 family still missing
// them. Every fast/strided path below unconditionally dereferences its
// pointer(s), so without the guard a null pointer or a zero increment
// reaching it is an instant out-of-bounds/UB access — these tests would
// crash the test binary (value-returning routines: silently return a wrong,
// non-identity result) if the corresponding guard were ever removed.

#[test]
fn test_ddot_null_x_returns_identity() {
    let y = [1.0f64, 2.0, 3.0];
    let got = unsafe { cblas_ddot(3, std::ptr::null(), 1, y.as_ptr(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_ddot_null_y_returns_identity() {
    let x = [1.0f64, 2.0, 3.0];
    let got = unsafe { cblas_ddot(3, x.as_ptr(), 1, std::ptr::null(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_ddot_zero_incx_returns_identity() {
    let x = [1.0f64, 2.0, 3.0];
    let y = [1.0f64, 2.0, 3.0];
    let got = unsafe { cblas_ddot(3, x.as_ptr(), 0, y.as_ptr(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_sdot_null_x_returns_identity() {
    let y = [1.0f32, 2.0, 3.0];
    let got = unsafe { cblas_sdot(3, std::ptr::null(), 1, y.as_ptr(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_sdot_zero_incy_returns_identity() {
    let x = [1.0f32, 2.0, 3.0];
    let y = [1.0f32, 2.0, 3.0];
    let got = unsafe { cblas_sdot(3, x.as_ptr(), 1, y.as_ptr(), 0) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_dnrm2_null_x_is_zero() {
    let got = unsafe { cblas_dnrm2(3, std::ptr::null(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_snrm2_null_x_is_zero() {
    let got = unsafe { cblas_snrm2(3, std::ptr::null(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_snrm2_negative_incx_is_zero() {
    let x = [3.0f32, 4.0];
    let got = unsafe { cblas_snrm2(2, x.as_ptr(), -1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_dasum_null_x_is_zero() {
    let got = unsafe { cblas_dasum(3, std::ptr::null(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_sasum_null_x_is_zero() {
    let got = unsafe { cblas_sasum(3, std::ptr::null(), 1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_sasum_negative_incx_is_zero() {
    let x = [1.0f32, -2.0, 3.0];
    let got = unsafe { cblas_sasum(3, x.as_ptr(), -1) };
    assert_eq!(got, 0.0);
}

#[test]
fn test_idamax_null_x_is_zero() {
    let got = unsafe { cblas_idamax(3, std::ptr::null(), 1) };
    assert_eq!(got, 0);
}

#[test]
fn test_isamax_null_x_is_zero() {
    let got = unsafe { cblas_isamax(3, std::ptr::null(), 1) };
    assert_eq!(got, 0);
}

#[test]
fn test_isamax_negative_incx_is_zero() {
    let x = [1.0f32, -5.0, 3.0];
    let got = unsafe { cblas_isamax(3, x.as_ptr(), -1) };
    assert_eq!(got, 0);
}

#[test]
fn test_dscal_null_x_is_noop() {
    // The only pointer argument is null; the guard must return before the
    // fast path's `slice::from_raw_parts_mut(null, 3)` (instant UB). Not
    // crashing the test process is the assertion.
    unsafe { cblas_dscal(3, 5.0, std::ptr::null_mut(), 1) };
}

#[test]
fn test_sscal_null_x_is_noop() {
    unsafe { cblas_sscal(3, 5.0, std::ptr::null_mut(), 1) };
}

#[test]
fn test_sscal_negative_incx_is_noop() {
    let mut x = [1.0f32, 2.0, 3.0];
    unsafe { cblas_sscal(3, 5.0, x.as_mut_ptr(), -1) };
    assert_eq!(x, [1.0, 2.0, 3.0]);
}

#[test]
fn test_daxpy_null_x_leaves_y_untouched() {
    let mut y = [9.0f64, 9.0, 9.0];
    unsafe { cblas_daxpy(3, 2.0, std::ptr::null(), 1, y.as_mut_ptr(), 1) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_daxpy_null_y_is_noop() {
    let x = [1.0f64, 2.0, 3.0];
    // y is null too: the assertion is that this does not dereference it.
    unsafe { cblas_daxpy(3, 2.0, x.as_ptr(), 1, std::ptr::null_mut(), 1) };
}

#[test]
fn test_daxpy_zero_incx_leaves_y_untouched() {
    let x = [1.0f64, 2.0, 3.0];
    let mut y = [9.0f64, 9.0, 9.0];
    unsafe { cblas_daxpy(3, 2.0, x.as_ptr(), 0, y.as_mut_ptr(), 1) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_saxpy_null_x_leaves_y_untouched() {
    let mut y = [9.0f32, 9.0, 9.0];
    unsafe { cblas_saxpy(3, 2.0, std::ptr::null(), 1, y.as_mut_ptr(), 1) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_saxpy_zero_incy_leaves_y_untouched() {
    let x = [1.0f32, 2.0, 3.0];
    let mut y = [9.0f32, 9.0, 9.0];
    unsafe { cblas_saxpy(3, 2.0, x.as_ptr(), 1, y.as_mut_ptr(), 0) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_dcopy_null_x_leaves_y_untouched() {
    let mut y = [9.0f64, 9.0, 9.0];
    unsafe { cblas_dcopy(3, std::ptr::null(), 1, y.as_mut_ptr(), 1) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_dcopy_zero_incx_leaves_y_untouched() {
    let x = [1.0f64, 2.0, 3.0];
    let mut y = [9.0f64, 9.0, 9.0];
    unsafe { cblas_dcopy(3, x.as_ptr(), 0, y.as_mut_ptr(), 1) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_scopy_null_y_is_noop() {
    let x = [1.0f32, 2.0, 3.0];
    unsafe { cblas_scopy(3, x.as_ptr(), 1, std::ptr::null_mut(), 1) };
}

#[test]
fn test_scopy_zero_incy_leaves_y_untouched() {
    let x = [1.0f32, 2.0, 3.0];
    let mut y = [9.0f32, 9.0, 9.0];
    unsafe { cblas_scopy(3, x.as_ptr(), 1, y.as_mut_ptr(), 0) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_dswap_null_x_leaves_y_untouched() {
    let mut y = [9.0f64, 9.0, 9.0];
    unsafe { cblas_dswap(3, std::ptr::null_mut(), 1, y.as_mut_ptr(), 1) };
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_dswap_zero_incx_leaves_buffers_untouched() {
    let mut x = [1.0f64, 2.0, 3.0];
    let mut y = [9.0f64, 9.0, 9.0];
    unsafe { cblas_dswap(3, x.as_mut_ptr(), 0, y.as_mut_ptr(), 1) };
    assert_eq!(x, [1.0, 2.0, 3.0]);
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

#[test]
fn test_sswap_null_y_leaves_x_untouched() {
    let mut x = [1.0f32, 2.0, 3.0];
    unsafe { cblas_sswap(3, x.as_mut_ptr(), 1, std::ptr::null_mut(), 1) };
    assert_eq!(x, [1.0, 2.0, 3.0]);
}

#[test]
fn test_sswap_zero_incy_leaves_buffers_untouched() {
    let mut x = [1.0f32, 2.0, 3.0];
    let mut y = [9.0f32, 9.0, 9.0];
    unsafe { cblas_sswap(3, x.as_mut_ptr(), 1, y.as_mut_ptr(), 0) };
    assert_eq!(x, [1.0, 2.0, 3.0]);
    assert_eq!(y, [9.0, 9.0, 9.0]);
}

// ---- Level 2: GEMV ----------------------------------------------------

#[test]
fn test_dgemv_positive_stride_sanity() {
    // A = [[1,2],[3,4]] col-major; y = A*x with x = [1,1] ⇒ [3,7].
    let a = [1.0f64, 3.0, 2.0, 4.0];
    let x = [1.0f64, 1.0];
    let mut y = [0.0f64; 2];
    unsafe {
        cblas_dgemv(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
    assert!((y[0] - 3.0).abs() < EPS && (y[1] - 7.0).abs() < EPS);
}

#[test]
fn test_dgemv_negative_inc_and_beta_zero() {
    // A = [[1,2],[3,4]] col-major. incx=-1 ⇒ x is read reversed; incy=-1 ⇒
    // y is written reversed. beta=0 must overwrite the NaN garbage in y.
    let a = [1.0f64, 3.0, 2.0, 4.0];
    let x = [10.0f64, 20.0]; // logical (reversed): [20, 10]
    let mut y = [f64::NAN; 2];
    // Independent reference.
    let x_log = [x[1], x[0]];
    let y_log = [
        a[0] * x_log[0] + a[2] * x_log[1], // row 0 · x
        a[1] * x_log[0] + a[3] * x_log[1], // row 1 · x
    ];
    // incy=-1 stores y_log[0] at y[1], y_log[1] at y[0].
    let expected = [y_log[1], y_log[0]];
    unsafe {
        cblas_dgemv(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            x.as_ptr(),
            -1,
            0.0,
            y.as_mut_ptr(),
            -1,
        );
    }
    assert!(y[0].is_finite() && y[1].is_finite(), "beta=0 left NaN in y");
    assert!((y[0] - expected[0]).abs() < EPS);
    assert!((y[1] - expected[1]).abs() < EPS);
}

#[test]
fn test_sgemv_negative_incx() {
    let a = [1.0f32, 3.0, 2.0, 4.0];
    let x = [10.0f32, 20.0];
    let mut y = [0.0f32; 2];
    let x_log = [x[1], x[0]];
    let expected = [
        a[0] * x_log[0] + a[2] * x_log[1],
        a[1] * x_log[0] + a[3] * x_log[1],
    ];
    unsafe {
        cblas_sgemv(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            x.as_ptr(),
            -1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
    assert!((y[0] - expected[0]).abs() < EPS_F32);
    assert!((y[1] - expected[1]).abs() < EPS_F32);
}

// ---- Complex dot products (zdotu/zdotc/cdotu/cdotc) -------------------

#[test]
fn test_zdotu_negative_incx() {
    let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let y = [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
    // incx=-1 reverses x, incy=+1 forward y.
    let expected = x[1] * y[0] + x[0] * y[1];
    let mut got = Complex64::new(0.0, 0.0);
    unsafe { cblas_zdotu_sub(2, x.as_ptr(), -1, y.as_ptr(), 1, &mut got) };
    assert!((got.re - expected.re).abs() < EPS);
    assert!((got.im - expected.im).abs() < EPS);
    // Different from forward dot.
    let forward = x[0] * y[0] + x[1] * y[1];
    assert!((expected.im - forward.im).abs() > 1.0);
}

#[test]
fn test_zdotc_negative_incx() {
    let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let y = [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
    // conj(x) reversed dotted with forward y.
    let expected = x[1].conj() * y[0] + x[0].conj() * y[1];
    let mut got = Complex64::new(0.0, 0.0);
    unsafe { cblas_zdotc_sub(2, x.as_ptr(), -1, y.as_ptr(), 1, &mut got) };
    assert!((got.re - expected.re).abs() < EPS);
    assert!((got.im - expected.im).abs() < EPS);
}

#[test]
fn test_cdotu_negative_incy() {
    let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    let y = [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];
    // incx=+1 forward x, incy=-1 reverses y.
    let expected = x[0] * y[1] + x[1] * y[0];
    let mut got = Complex32::new(0.0, 0.0);
    unsafe { cblas_cdotu_sub(2, x.as_ptr(), 1, y.as_ptr(), -1, &mut got) };
    assert!((got.re - expected.re).abs() < EPS_F32);
    assert!((got.im - expected.im).abs() < EPS_F32);
}

#[test]
fn test_cdotc_negative_incx() {
    let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    let y = [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];
    let expected = x[1].conj() * y[0] + x[0].conj() * y[1];
    let mut got = Complex32::new(0.0, 0.0);
    unsafe { cblas_cdotc_sub(2, x.as_ptr(), -1, y.as_ptr(), 1, &mut got) };
    assert!((got.re - expected.re).abs() < EPS_F32);
    assert!((got.im - expected.im).abs() < EPS_F32);
}

// ---- Complex dot products: null-pointer / zero-increment guards -------
//
// `dotu`/`dotc` used to be dereferenced for the write unconditionally,
// including on the `n <= 0` early-return path, with no null check anywhere
// and no `x`/`y` null or increment validation on the main path either —
// `cblas_zdotu_sub(0, x, 1, y, 1, ptr::null_mut())` was a null-pointer
// *write* on an otherwise spec-legal call. These pin the fix: a null
// output pointer must be a true no-op (nothing left to write through), and
// once the output pointer is valid, a null `x`/`y` or a zero increment
// must still land the documented identity rather than reading through a
// bad pointer.

#[test]
fn test_zdotu_sub_null_dotu_is_noop() {
    let x = [Complex64::new(1.0, 1.0)];
    let y = [Complex64::new(2.0, 2.0)];
    // Must not dereference the null output pointer.
    unsafe { cblas_zdotu_sub(1, x.as_ptr(), 1, y.as_ptr(), 1, std::ptr::null_mut()) };
}

#[test]
fn test_zdotu_sub_null_dotu_with_n_zero_is_noop() {
    // The specific historical bug: n<=0 used to write through `dotu`
    // unconditionally, before any null check existed.
    unsafe {
        cblas_zdotu_sub(
            0,
            std::ptr::null(),
            1,
            std::ptr::null(),
            1,
            std::ptr::null_mut(),
        )
    };
}

#[test]
fn test_zdotu_sub_null_x_writes_identity() {
    let y = [Complex64::new(1.0, 1.0)];
    let mut got = Complex64::new(9.0, 9.0);
    unsafe { cblas_zdotu_sub(1, std::ptr::null(), 1, y.as_ptr(), 1, &mut got) };
    assert_eq!(got, Complex64::new(0.0, 0.0));
}

#[test]
fn test_zdotu_sub_zero_incx_writes_identity() {
    let x = [Complex64::new(1.0, 1.0)];
    let y = [Complex64::new(1.0, 1.0)];
    let mut got = Complex64::new(9.0, 9.0);
    unsafe { cblas_zdotu_sub(1, x.as_ptr(), 0, y.as_ptr(), 1, &mut got) };
    assert_eq!(got, Complex64::new(0.0, 0.0));
}

#[test]
fn test_zdotc_sub_null_y_writes_identity() {
    let x = [Complex64::new(1.0, 1.0)];
    let mut got = Complex64::new(9.0, 9.0);
    unsafe { cblas_zdotc_sub(1, x.as_ptr(), 1, std::ptr::null(), 1, &mut got) };
    assert_eq!(got, Complex64::new(0.0, 0.0));
}

#[test]
fn test_zdotc_sub_zero_incy_writes_identity() {
    let x = [Complex64::new(1.0, 1.0)];
    let y = [Complex64::new(1.0, 1.0)];
    let mut got = Complex64::new(9.0, 9.0);
    unsafe { cblas_zdotc_sub(1, x.as_ptr(), 1, y.as_ptr(), 0, &mut got) };
    assert_eq!(got, Complex64::new(0.0, 0.0));
}

#[test]
fn test_cdotu_sub_null_dotu_is_noop() {
    let x = [Complex32::new(1.0, 1.0)];
    let y = [Complex32::new(2.0, 2.0)];
    unsafe { cblas_cdotu_sub(1, x.as_ptr(), 1, y.as_ptr(), 1, std::ptr::null_mut()) };
}

#[test]
fn test_cdotu_sub_null_x_writes_identity() {
    let y = [Complex32::new(1.0, 1.0)];
    let mut got = Complex32::new(9.0, 9.0);
    unsafe { cblas_cdotu_sub(1, std::ptr::null(), 1, y.as_ptr(), 1, &mut got) };
    assert_eq!(got, Complex32::new(0.0, 0.0));
}

#[test]
fn test_cdotc_sub_null_y_writes_identity() {
    let x = [Complex32::new(1.0, 1.0)];
    let mut got = Complex32::new(9.0, 9.0);
    unsafe { cblas_cdotc_sub(1, x.as_ptr(), 1, std::ptr::null(), 1, &mut got) };
    assert_eq!(got, Complex32::new(0.0, 0.0));
}

#[test]
fn test_cdotc_sub_zero_incx_writes_identity() {
    let x = [Complex32::new(1.0, 1.0)];
    let y = [Complex32::new(1.0, 1.0)];
    let mut got = Complex32::new(9.0, 9.0);
    unsafe { cblas_cdotc_sub(1, x.as_ptr(), 0, y.as_ptr(), 1, &mut got) };
    assert_eq!(got, Complex32::new(0.0, 0.0));
}

// ---- Level 3: beta==0 with uninitialized C ---------------------------

#[test]
fn test_dgemm_beta_zero_ignores_uninit_c() {
    // Hit the scalar transpose fallback (transa=Trans) with NaN in C and
    // beta=0: the NaN must be overwritten, not propagated via 0*NaN.
    let a = [1.0f64, 3.0, 2.0, 4.0]; // stored col-major 2x2
    let b = [1.0f64, 0.0, 0.0, 1.0]; // identity
    let mut c = [f64::NAN; 4];
    unsafe {
        cblas_dgemm(
            CblasLayout::ColMajor,
            CblasTranspose::Trans,
            CblasTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
    }
    // C = A^T (stored) = [[1,3],[2,4]] col-major ⇒ [1, 2, 3, 4].
    assert!(
        c.iter().all(|v| v.is_finite()),
        "beta=0 propagated NaN: {c:?}"
    );
    assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_zgemm_beta_zero_ignores_uninit_c() {
    let a = [Complex64::new(2.0, 0.0)];
    let b = [Complex64::new(3.0, 0.0)];
    let mut c = [Complex64::new(f64::NAN, f64::NAN)];
    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::new(0.0, 0.0);
    unsafe {
        cblas_zgemm(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            1,
            1,
            1,
            &alpha,
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            &beta,
            c.as_mut_ptr(),
            1,
        );
    }
    assert!(c[0].re.is_finite() && c[0].im.is_finite());
    assert!((c[0].re - 6.0).abs() < EPS && c[0].im.abs() < EPS);
}

// ---- Level 3: parameter validation (negative k / bad ld / null) -------

#[test]
fn test_dgemm_negative_k_is_noop() {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [1.0f64, 2.0, 3.0, 4.0];
    let mut c = [7.0f64; 4];
    unsafe {
        cblas_dgemm(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            2,
            2,
            -1, // invalid k
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            1.0,
            c.as_mut_ptr(),
            2,
        );
    }
    assert_eq!(c, [7.0; 4]);
}

#[test]
fn test_dgemm_undersized_ldc_is_noop() {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [1.0f64, 2.0, 3.0, 4.0];
    let mut c = [7.0f64; 4];
    unsafe {
        cblas_dgemm(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            1.0,
            c.as_mut_ptr(),
            1, // ldc < max(1, m) = 2
        );
    }
    assert_eq!(c, [7.0; 4]);
}

#[test]
fn test_dgemm_null_pointer_is_noop() {
    let b = [1.0f64, 2.0, 3.0, 4.0];
    let mut c = [7.0f64; 4];
    unsafe {
        cblas_dgemm(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            std::ptr::null(), // null A
            2,
            b.as_ptr(),
            2,
            1.0,
            c.as_mut_ptr(),
            2,
        );
    }
    assert_eq!(c, [7.0; 4]);
}

#[test]
fn test_dgemv_undersized_lda_is_noop() {
    let a = [1.0f64, 3.0, 2.0, 4.0];
    let x = [1.0f64, 1.0];
    let mut y = [7.0f64; 2];
    unsafe {
        cblas_dgemv(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            2,
            2,
            1.0,
            a.as_ptr(),
            1, // lda < max(1, m) = 2
            x.as_ptr(),
            1,
            1.0,
            y.as_mut_ptr(),
            1,
        );
    }
    assert_eq!(y, [7.0; 2]);
}
