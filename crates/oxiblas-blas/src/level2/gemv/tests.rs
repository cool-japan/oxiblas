//! Tests for `gemv` and friends (general matrix-vector multiply).

use super::*;
use oxiblas_matrix::Mat;

#[test]
fn test_gemv_basic() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let x = [1.0, 2.0, 3.0];
    let mut y = [0.0, 0.0];

    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    // y[0] = 1*1 + 2*2 + 3*3 = 14
    // y[1] = 4*1 + 5*2 + 6*3 = 32
    assert!((y[0] - 14.0).abs() < 1e-10);
    assert!((y[1] - 32.0).abs() < 1e-10);
}

#[test]
fn test_gemv_with_alpha_beta() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0];
    let mut y = [10.0, 10.0];

    // y = 2 * A * x + 3 * y
    // A*x = [3, 7]
    // y = 2*[3,7] + 3*[10,10] = [6,14] + [30,30] = [36, 44]
    gemv(GemvTrans::NoTrans, 2.0, a.as_ref(), &x, 3.0, &mut y);

    assert!((y[0] - 36.0).abs() < 1e-10);
    assert!((y[1] - 44.0).abs() < 1e-10);
}

#[test]
fn test_gemv_transpose() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let x = [1.0, 2.0]; // 2 elements for A^T (2×3 -> 3×2)
    let mut y = [0.0, 0.0, 0.0];

    // y = A^T * x
    // A^T = [[1,4], [2,5], [3,6]]
    // y[0] = 1*1 + 4*2 = 9
    // y[1] = 2*1 + 5*2 = 12
    // y[2] = 3*1 + 6*2 = 15
    gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    assert!((y[0] - 9.0).abs() < 1e-10);
    assert!((y[1] - 12.0).abs() < 1e-10);
    assert!((y[2] - 15.0).abs() < 1e-10);
}

#[test]
fn test_gemv_alpha_zero() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 2.0];
    let mut y = [10.0, 20.0];

    // y = 0 * A * x + 2 * y = [20, 40]
    gemv(GemvTrans::NoTrans, 0.0, a.as_ref(), &x, 2.0, &mut y);

    assert!((y[0] - 20.0).abs() < 1e-10);
    assert!((y[1] - 40.0).abs() < 1e-10);
}

#[test]
fn test_gemv_beta_zero() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 2.0];
    let mut y = [100.0, 200.0]; // Should be ignored

    // y = 1 * A * x + 0 * y
    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    // y[0] = 1*1 + 2*2 = 5
    // y[1] = 3*1 + 4*2 = 11
    assert!((y[0] - 5.0).abs() < 1e-10);
    assert!((y[1] - 11.0).abs() < 1e-10);
}

#[test]
fn test_gemv_identity() {
    let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
    let x = [1.0, 2.0, 3.0];
    let mut y = [0.0, 0.0, 0.0];

    gemv(GemvTrans::NoTrans, 1.0, eye.as_ref(), &x, 0.0, &mut y);

    assert_eq!(y, x);
}

#[test]
fn test_gemv_simple() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 2.0];

    let y = gemv_simple(a.as_ref(), &x);

    assert!((y[0] - 5.0).abs() < 1e-10);
    assert!((y[1] - 11.0).abs() < 1e-10);
}

#[test]
fn test_gemv_f32() {
    let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
    let x = [1.0f32, 2.0];
    let mut y = [0.0f32, 0.0];

    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    assert!((y[0] - 5.0).abs() < 1e-5);
    assert!((y[1] - 11.0).abs() < 1e-5);
}

#[test]
fn test_gemv_empty() {
    let a: Mat<f64> = Mat::zeros(0, 0);
    let x: [f64; 0] = [];
    let mut y: [f64; 0] = [];

    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);
    // Should not panic
}

#[test]
fn test_gemv_single() {
    let a = Mat::from_rows(&[&[3.0f64]]);
    let x = [2.0];
    let mut y = [0.0];

    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    assert!((y[0] - 6.0).abs() < 1e-10);
}

#[test]
fn test_gemv_tall_matrix() {
    // 4x2 matrix
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);
    let x = [1.0, 1.0];
    let mut y = [0.0, 0.0, 0.0, 0.0];

    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    assert!((y[0] - 3.0).abs() < 1e-10);
    assert!((y[1] - 7.0).abs() < 1e-10);
    assert!((y[2] - 11.0).abs() < 1e-10);
    assert!((y[3] - 15.0).abs() < 1e-10);
}

#[test]
fn test_gemv_wide_matrix() {
    // 2x4 matrix
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);
    let x = [1.0, 1.0, 1.0, 1.0];
    let mut y = [0.0, 0.0];

    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    assert!((y[0] - 10.0).abs() < 1e-10);
    assert!((y[1] - 26.0).abs() < 1e-10);
}

#[test]
fn test_gemv_parallel() {
    // Test with a larger matrix to trigger parallel execution
    let n = 512;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let x: Vec<f64> = vec![1.0; n];
    let mut y: Vec<f64> = vec![0.0; n];

    #[cfg(feature = "parallel")]
    {
        gemv_with_par(
            GemvTrans::NoTrans,
            1.0,
            a.as_ref(),
            &x,
            0.0,
            &mut y,
            Par::Rayon,
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);
    }

    // Each element of y should be n (sum of n ones)
    for i in 0..n {
        assert!(
            (y[i] - n as f64).abs() < 1e-10,
            "y[{}] = {}, expected {}",
            i,
            y[i],
            n
        );
    }
}

#[test]
fn test_gemv_parallel_transpose() {
    // Test parallel transpose
    let n = 512;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let x: Vec<f64> = vec![1.0; n];
    let mut y: Vec<f64> = vec![0.0; n];

    #[cfg(feature = "parallel")]
    {
        gemv_with_par(
            GemvTrans::Trans,
            1.0,
            a.as_ref(),
            &x,
            0.0,
            &mut y,
            Par::Rayon,
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y);
    }

    // Each element of y should be n (sum of n ones)
    for i in 0..n {
        assert!(
            (y[i] - n as f64).abs() < 1e-10,
            "y[{}] = {}, expected {}",
            i,
            y[i],
            n
        );
    }
}

#[test]
fn test_gemv_blocked_notrans() {
    // Test blocked GEMV with large matrix (> 4096 elements)
    let m = 128;
    let n = 64;
    let a: Mat<f64> = Mat::filled(m, n, 2.0);
    let x: Vec<f64> = vec![1.0; n];
    let mut y: Vec<f64> = vec![0.0; m];

    gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    // Each element of y should be n * 2.0 = 128.0
    for i in 0..m {
        assert!(
            (y[i] - 128.0).abs() < 1e-10,
            "y[{}] = {}, expected 128.0",
            i,
            y[i]
        );
    }
}

#[test]
fn test_gemv_blocked_trans() {
    // Test blocked GEMV transpose with large matrix
    let m = 128;
    let n = 64;
    let a: Mat<f64> = Mat::filled(m, n, 2.0);
    let x: Vec<f64> = vec![1.0; m];
    let mut y: Vec<f64> = vec![0.0; n];

    gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    // Each element of y should be m * 2.0 = 256.0
    for j in 0..n {
        assert!(
            (y[j] - 256.0).abs() < 1e-10,
            "y[{}] = {}, expected 256.0",
            j,
            y[j]
        );
    }
}

#[test]
fn test_gemv_blocked_with_alpha_beta() {
    // Test blocked GEMV with alpha and beta
    let n = 100;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let x: Vec<f64> = vec![1.0; n];
    let mut y: Vec<f64> = vec![10.0; n];

    // y = 2 * A * x + 0.5 * y
    // A * x = n * 1 = 100 for each element
    // y = 2 * 100 + 0.5 * 10 = 205
    gemv(GemvTrans::NoTrans, 2.0, a.as_ref(), &x, 0.5, &mut y);

    for i in 0..n {
        assert!(
            (y[i] - 205.0).abs() < 1e-10,
            "y[{}] = {}, expected 205.0",
            i,
            y[i]
        );
    }
}

// ========================================================================
// Fused GEMV operations tests
// ========================================================================

#[test]
fn test_gemv_add_basic() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0];
    let z = vec![10.0, 20.0];

    // result = 1.0 * A * x + z = [3, 7] + [10, 20] = [13, 27]
    let result = gemv_add(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, z);

    assert!((result[0] - 13.0).abs() < 1e-10);
    assert!((result[1] - 27.0).abs() < 1e-10);
}

#[test]
fn test_gemv_add_with_alpha() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0];
    let z = vec![10.0, 20.0];

    // result = 2.0 * A * x + z = 2*[3, 7] + [10, 20] = [6, 14] + [10, 20] = [16, 34]
    let result = gemv_add(GemvTrans::NoTrans, 2.0, a.as_ref(), &x, z);

    assert!((result[0] - 16.0).abs() < 1e-10);
    assert!((result[1] - 34.0).abs() < 1e-10);
}

#[test]
fn test_gemv_add_alpha_zero() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0];
    let z = vec![10.0, 20.0];

    // result = 0 * A * x + z = z
    let result = gemv_add(GemvTrans::NoTrans, 0.0, a.as_ref(), &x, z);

    assert!((result[0] - 10.0).abs() < 1e-10);
    assert!((result[1] - 20.0).abs() < 1e-10);
}

#[test]
fn test_gemv_add_transpose() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let x = [1.0, 1.0]; // Input for A^T (m elements)
    let z = vec![10.0, 20.0, 30.0]; // Output size n

    // A^T = [[1,4], [2,5], [3,6]]
    // A^T * x = [1+4, 2+5, 3+6] = [5, 7, 9]
    // result = 1.0 * [5, 7, 9] + [10, 20, 30] = [15, 27, 39]
    let result = gemv_add(GemvTrans::Trans, 1.0, a.as_ref(), &x, z);

    assert!((result[0] - 15.0).abs() < 1e-10);
    assert!((result[1] - 27.0).abs() < 1e-10);
    assert!((result[2] - 39.0).abs() < 1e-10);
}

#[test]
fn test_gemv_add_inplace_basic() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0];
    let mut y = vec![10.0, 20.0];

    // y += 1.0 * A * x -> y = [10, 20] + [3, 7] = [13, 27]
    gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y);

    assert!((y[0] - 13.0).abs() < 1e-10);
    assert!((y[1] - 27.0).abs() < 1e-10);
}

#[test]
fn test_gemv_add_inplace_twice() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0];
    let mut y = vec![0.0, 0.0];

    // First: y = [0, 0] + [3, 7] = [3, 7]
    gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y);
    assert!((y[0] - 3.0).abs() < 1e-10);
    assert!((y[1] - 7.0).abs() < 1e-10);

    // Second: y = [3, 7] + [3, 7] = [6, 14]
    gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y);
    assert!((y[0] - 6.0).abs() < 1e-10);
    assert!((y[1] - 14.0).abs() < 1e-10);
}

#[test]
fn test_gemv_add_inplace_dimension_mismatch_x_panics() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0, 1.0]; // wrong length: A has 2 columns
    let mut y = vec![0.0, 0.0];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y);
    }));
    assert!(
        result.is_err(),
        "gemv_add_inplace must reject x with the wrong length instead of reading out of bounds"
    );
}

#[test]
fn test_gemv_add_inplace_dimension_mismatch_y_panics() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let x = [1.0, 1.0];
    let mut y = vec![0.0, 0.0, 0.0]; // wrong length: A has 2 rows

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y);
    }));
    assert!(
        result.is_err(),
        "gemv_add_inplace must reject y with the wrong length instead of silently under/over-writing"
    );
}

// ========================================================================
// NaN/Inf propagation regression tests (reference-BLAS exactness).
//
// Reference DGEMV never special-cases x[i] == 0: every row contributes
// A[i,j] * x[i], so a row with A[i,j] == Inf and x[i] == 0.0 must
// propagate NaN (IEEE 754: 0 * Inf = NaN) into y[j]. A "skip rows where
// x[i] == 0" sparsity optimization would silently produce a different
// (wrong) numeric result for such inputs. These tests pin down that the
// Trans/ConjTrans code paths do NOT skip zero rows.
// ========================================================================

#[test]
fn test_gemv_trans_zero_x_does_not_suppress_nan() {
    // A = [[Inf, 1], [2, 3]], x = [0, 1]
    // Reference: y[0] = Inf*0 + 2*1 = NaN + 2 = NaN
    let a = Mat::from_rows(&[&[f64::INFINITY, 1.0f64], &[2.0, 3.0]]);
    let x = [0.0f64, 1.0];
    let mut y = [0.0f64, 0.0];

    gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y);

    assert!(
        y[0].is_nan(),
        "y[0] must be NaN (0 * Inf) per reference DGEMV, got {}",
        y[0]
    );
}

#[test]
fn test_gemv_add_inplace_trans_zero_x_does_not_suppress_nan_unblocked() {
    // Small matrix -> exercises gemv_add_unblocked_trans.
    let a = Mat::from_rows(&[&[f64::INFINITY, 1.0f64], &[2.0, 3.0]]);
    let x = [0.0f64, 1.0];
    let mut y = vec![0.0f64, 0.0];

    gemv_add_inplace(GemvTrans::Trans, 1.0, a.as_ref(), &x, &mut y);

    assert!(
        y[0].is_nan(),
        "unblocked gemv_add_inplace(Trans) must propagate NaN from a zero-x row, got {}",
        y[0]
    );
}

#[test]
fn test_gemv_add_inplace_conjtrans_zero_x_does_not_suppress_nan_unblocked() {
    let a = Mat::from_rows(&[&[f64::INFINITY, 1.0f64], &[2.0, 3.0]]);
    let x = [0.0f64, 1.0];
    let mut y = vec![0.0f64, 0.0];

    gemv_add_inplace(GemvTrans::ConjTrans, 1.0, a.as_ref(), &x, &mut y);

    assert!(
        y[0].is_nan(),
        "unblocked gemv_add_inplace(ConjTrans) must propagate NaN from a zero-x row, got {}",
        y[0]
    );
}

#[test]
fn test_gemv_add_blocked_trans_zero_x_does_not_suppress_nan() {
    // Directly exercise the blocked Trans kernel used for large matrices
    // (m*n > 4096 in the public gemv_add_inplace dispatcher).
    let m = 4;
    let n = 4;
    let mut a: Mat<f64> = Mat::zeros(m, n);
    a[(0, 0)] = f64::INFINITY;
    a[(1, 0)] = 2.0;
    let mut x = vec![0.0f64; m];
    x[1] = 1.0;
    let mut y = vec![0.0f64; n];

    gemv_add_blocked_trans(1.0, a.as_ref(), &x, &mut y, m, n);

    assert!(
        y[0].is_nan(),
        "blocked gemv_add Trans kernel must propagate NaN from a zero-x row, got {}",
        y[0]
    );
}

#[test]
fn test_gemv_add_blocked_conjtrans_zero_x_does_not_suppress_nan() {
    let m = 4;
    let n = 4;
    let mut a: Mat<f64> = Mat::zeros(m, n);
    a[(0, 0)] = f64::INFINITY;
    a[(1, 0)] = 2.0;
    let mut x = vec![0.0f64; m];
    x[1] = 1.0;
    let mut y = vec![0.0f64; n];

    gemv_add_blocked_conjtrans(1.0, a.as_ref(), &x, &mut y, m, n);

    assert!(
        y[0].is_nan(),
        "blocked gemv_add ConjTrans kernel must propagate NaN from a zero-x row, got {}",
        y[0]
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_gemv_parallel_trans_zero_x_does_not_suppress_nan() {
    // Directly exercise the private thread-local-accumulator kernel to
    // pin down that x[i] == 0 rows are not skipped during reduction.
    let m = 8;
    let n = 8;
    let mut a: Mat<f64> = Mat::zeros(m, n);
    a[(0, 0)] = f64::INFINITY;
    a[(1, 0)] = 2.0;
    let mut x = vec![0.0f64; m];
    x[1] = 1.0;
    let mut y = vec![0.0f64; n];

    gemv_parallel_trans(1.0, a.as_ref(), &x, &mut y, m, n, Par::RayonWith(2));

    assert!(
        y[0].is_nan(),
        "gemv_parallel_trans must propagate NaN from a zero-x row, got {}",
        y[0]
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_gemv_parallel_conjtrans_zero_x_does_not_suppress_nan() {
    let m = 8;
    let n = 8;
    let mut a: Mat<f64> = Mat::zeros(m, n);
    a[(0, 0)] = f64::INFINITY;
    a[(1, 0)] = 2.0;
    let mut x = vec![0.0f64; m];
    x[1] = 1.0;
    let mut y = vec![0.0f64; n];

    gemv_parallel_conjtrans(1.0, a.as_ref(), &x, &mut y, m, n, Par::RayonWith(2));

    assert!(
        y[0].is_nan(),
        "gemv_parallel_conjtrans must propagate NaN from a zero-x row, got {}",
        y[0]
    );
}

#[test]
fn test_gemv_sum2_basic() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
    let x = [1.0, 1.0];
    let z = [1.0, 1.0];

    // A*x = [3, 7]
    // B*z = [11, 15]
    // result = 1.0 * [3, 7] + 1.0 * [11, 15] = [14, 22]
    let result = gemv_sum2(1.0, a.as_ref(), &x, 1.0, b.as_ref(), &z);

    assert!((result[0] - 14.0).abs() < 1e-10);
    assert!((result[1] - 22.0).abs() < 1e-10);
}

#[test]
fn test_gemv_sum2_with_scalars() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
    let x = [1.0, 1.0];
    let z = [1.0, 1.0];

    // A*x = [3, 7]
    // B*z = [11, 15]
    // result = 2.0 * [3, 7] + 0.5 * [11, 15] = [6, 14] + [5.5, 7.5] = [11.5, 21.5]
    let result = gemv_sum2(2.0, a.as_ref(), &x, 0.5, b.as_ref(), &z);

    assert!((result[0] - 11.5).abs() < 1e-10);
    assert!((result[1] - 21.5).abs() < 1e-10);
}

#[test]
fn test_gemv_sum2_alpha_zero() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
    let x = [1.0, 1.0];
    let z = [1.0, 1.0];

    // result = 0 * A*x + 1.0 * B*z = [11, 15]
    let result = gemv_sum2(0.0, a.as_ref(), &x, 1.0, b.as_ref(), &z);

    assert!((result[0] - 11.0).abs() < 1e-10);
    assert!((result[1] - 15.0).abs() < 1e-10);
}

#[test]
fn test_gemv_sum2_beta_zero() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
    let x = [1.0, 1.0];
    let z = [1.0, 1.0];

    // result = 1.0 * A*x + 0 * B*z = [3, 7]
    let result = gemv_sum2(1.0, a.as_ref(), &x, 0.0, b.as_ref(), &z);

    assert!((result[0] - 3.0).abs() < 1e-10);
    assert!((result[1] - 7.0).abs() < 1e-10);
}

#[test]
fn test_gemv_add_blocked() {
    // Test with large matrix that triggers blocked implementation
    let n = 100;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let x: Vec<f64> = vec![1.0; n];
    let z: Vec<f64> = vec![5.0; n];

    // result = 1.0 * A * x + z
    // A * x = [n, n, ..., n] (each element = 100)
    // result = [100, 100, ...] + [5, 5, ...] = [105, 105, ...]
    let result = gemv_add(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, z);

    for i in 0..n {
        assert!(
            (result[i] - 105.0).abs() < 1e-10,
            "result[{}] = {}, expected 105.0",
            i,
            result[i]
        );
    }
}

#[test]
fn test_gemv_add_blocked_trans() {
    // Test blocked transpose
    let m = 100;
    let n = 80;
    let a: Mat<f64> = Mat::filled(m, n, 1.0);
    let x: Vec<f64> = vec![1.0; m];
    let z: Vec<f64> = vec![5.0; n];

    // A^T * x: each element = m = 100
    // result = 1.0 * [100, ...] + [5, ...] = [105, ...]
    let result = gemv_add(GemvTrans::Trans, 1.0, a.as_ref(), &x, z);

    for j in 0..n {
        assert!(
            (result[j] - 105.0).abs() < 1e-10,
            "result[{}] = {}, expected 105.0",
            j,
            result[j]
        );
    }
}

#[test]
fn test_gemv_sum2_large() {
    // Test gemv_sum2 with larger matrices
    let n = 100;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let b: Mat<f64> = Mat::filled(n, n, 2.0);
    let x: Vec<f64> = vec![1.0; n];
    let z: Vec<f64> = vec![1.0; n];

    // A*x = [100, 100, ...]
    // B*z = [200, 200, ...]
    // result = 1.0 * [100, ...] + 0.5 * [200, ...] = [100, ...] + [100, ...] = [200, ...]
    let result = gemv_sum2(1.0, a.as_ref(), &x, 0.5, b.as_ref(), &z);

    for i in 0..n {
        assert!(
            (result[i] - 200.0).abs() < 1e-10,
            "result[{}] = {}, expected 200.0",
            i,
            result[i]
        );
    }
}
