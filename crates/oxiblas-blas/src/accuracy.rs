#![allow(clippy::doc_markdown)] // Mathematical notation in docs

//! Accuracy and numerical error analysis for BLAS operations.
//!
//! This module provides utilities and tests for measuring the numerical accuracy
//! of BLAS operations, including:
//!
//! - **Forward error**: How close is our result to the exact result?
//! - **Backward error**: What perturbation in the input would make our result exact?
//! - **Relative error**: Error relative to the magnitude of the result
//!
//! # Error Metrics
//!
//! For a computed result ŷ vs exact result y:
//!
//! - **Forward error**: ||ŷ - y||
//! - **Relative forward error**: ||ŷ - y|| / ||y||
//! - **Backward error**: For y = Ax, find smallest ΔA such that ŷ = (A + ΔA)x
//!
//! # Accuracy Bounds
//!
//! For well-implemented BLAS operations with double precision:
//!
//! | Operation | Expected Bound | Notes |
//! |-----------|---------------|-------|
//! | DOT | n·ε | Accumulation error |
//! | GEMV | (m+k)·ε | Matrix-vector product |
//! | GEMM | (m+k+n)·ε | Matrix-matrix product |
//! | TRSM | k²·ε | Triangular solve |
//!
//! where ε = machine epsilon (~2.2e-16 for f64)

use num_complex::Complex64;
use num_traits::Float;

/// Machine epsilon for f64.
pub const EPSILON_F64: f64 = f64::EPSILON;

/// Machine epsilon for f32.
pub const EPSILON_F32: f32 = f32::EPSILON;

/// Computes the maximum absolute error between two vectors.
///
/// max_error = max_i |x_i - y_i|
#[inline]
pub fn max_absolute_error<T: Float>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    x.iter()
        .zip(y.iter())
        .map(|(a, b)| Float::abs(*a - *b))
        .fold(T::zero(), |acc, v| if v > acc { v } else { acc })
}

/// Computes the L2 (Euclidean) error between two vectors.
///
/// l2_error = sqrt(Σ (x_i - y_i)²)
#[inline]
pub fn l2_error<T: Float>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    let sum_sq: T = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let diff = *a - *b;
            diff * diff
        })
        .fold(T::zero(), |acc, v| acc + v);

    sum_sq.sqrt()
}

/// Computes the L∞ (infinity) norm of a vector.
///
/// l_inf = max_i |x_i|
#[inline]
pub fn linf_norm<T: Float>(x: &[T]) -> T {
    x.iter()
        .map(|v| Float::abs(*v))
        .fold(T::zero(), |acc, v| if v > acc { v } else { acc })
}

/// Computes the L2 (Euclidean) norm of a vector.
///
/// l2_norm = sqrt(Σ x_i²)
#[inline]
pub fn l2_norm<T: Float>(x: &[T]) -> T {
    let sum_sq: T = x.iter().map(|v| *v * *v).fold(T::zero(), |acc, v| acc + v);
    sum_sq.sqrt()
}

/// Computes the relative forward error.
///
/// rel_error = ||computed - exact|| / ||exact||
///
/// Returns infinity if exact is zero.
#[inline]
pub fn relative_forward_error<T: Float>(computed: &[T], exact: &[T]) -> T {
    let error = l2_error(computed, exact);
    let exact_norm = l2_norm(exact);

    if exact_norm > T::zero() {
        error / exact_norm
    } else if error > T::zero() {
        T::infinity()
    } else {
        T::zero()
    }
}

/// Computes componentwise relative error.
///
/// Returns max_i |computed_i - exact_i| / |exact_i|
///
/// Skips components where exact_i is zero.
#[inline]
pub fn max_relative_error<T: Float>(computed: &[T], exact: &[T]) -> T {
    assert_eq!(computed.len(), exact.len(), "Vectors must have same length");

    let eps = if std::mem::size_of::<T>() == 4 {
        T::from(EPSILON_F32).unwrap_or_else(|| T::epsilon())
    } else {
        T::from(EPSILON_F64).unwrap_or_else(|| T::epsilon())
    };

    computed
        .iter()
        .zip(exact.iter())
        .filter(|(_, e)| Float::abs(**e) > eps)
        .map(|(c, e)| Float::abs(*c - *e) / Float::abs(*e))
        .fold(T::zero(), |acc, v| if v > acc { v } else { acc })
}

/// Complex maximum absolute error.
#[inline]
pub fn max_absolute_error_c64(x: &[Complex64], y: &[Complex64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

/// Complex L2 error.
#[inline]
#[must_use]
pub fn l2_error_c64(x: &[Complex64], y: &[Complex64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    let sum_sq: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum();

    sum_sq.sqrt()
}

/// Complex L2 norm.
#[inline]
#[must_use]
pub fn l2_norm_c64(x: &[Complex64]) -> f64 {
    let sum_sq: f64 = x.iter().map(num_complex::Complex::norm_sqr).sum();
    sum_sq.sqrt()
}

/// Complex relative forward error.
#[inline]
#[must_use]
pub fn relative_forward_error_c64(computed: &[Complex64], exact: &[Complex64]) -> f64 {
    let error = l2_error_c64(computed, exact);
    let exact_norm = l2_norm_c64(exact);

    if exact_norm > 0.0 {
        error / exact_norm
    } else if error > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

// =============================================================================
// Reference Implementations for Accuracy Testing
// =============================================================================

/// Reference dot product using Kahan summation for accuracy comparison.
#[must_use]
pub fn dot_reference_f64(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for i in 0..x.len() {
        let y_val = x[i].mul_add(y[i], -c);
        let t = sum + y_val;
        c = (t - sum) - y_val;
        sum = t;
    }

    sum
}

/// Reference GEMV using naive triple loop for accuracy comparison.
pub fn gemv_reference_f64(
    trans: bool,
    alpha: f64,
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    let (m, n) = if trans {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };

    assert_eq!(x.len(), n, "x length must match matrix columns");
    assert_eq!(y.len(), m, "y length must match matrix rows");

    // Scale y by beta
    for yi in y.iter_mut() {
        *yi *= beta;
    }

    // Accumulate A*x with Kahan summation for each output element
    if trans {
        // y = alpha * A^T * x + beta * y
        for j in 0..m {
            let mut sum = 0.0;
            let mut c = 0.0;
            for i in 0..n {
                let prod = a[i + j * a_rows] * x[i];
                let y_val = prod - c;
                let t = sum + y_val;
                c = (t - sum) - y_val;
                sum = t;
            }
            y[j] += alpha * sum;
        }
    } else {
        // y = alpha * A * x + beta * y
        for i in 0..m {
            let mut sum = 0.0;
            let mut c = 0.0;
            for j in 0..n {
                let prod = a[i + j * a_rows] * x[j];
                let y_val = prod - c;
                let t = sum + y_val;
                c = (t - sum) - y_val;
                sum = t;
            }
            y[i] += alpha * sum;
        }
    }
}

/// Reference GEMM using naive triple loop for accuracy comparison.
pub fn gemm_reference_f64(
    alpha: f64,
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    b: &[f64],
    b_cols: usize,
    beta: f64,
    c: &mut [f64],
) {
    let m = a_rows;
    let k = a_cols;
    let n = b_cols;

    // Scale C by beta
    for ci in c.iter_mut() {
        *ci *= beta;
    }

    // Accumulate A*B with Kahan summation for each output element
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            let mut comp = 0.0;
            for p in 0..k {
                let prod = a[i + p * m] * b[p + j * k];
                let y = prod - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
            c[i + j * m] += alpha * sum;
        }
    }
}

// =============================================================================
// Error Bound Calculators
// =============================================================================

/// Theoretical error bound for dot product.
///
/// For n-element dot product: |error| ≤ n * ε * |x|·|y|
#[must_use]
pub fn dot_error_bound(n: usize, x_norm: f64, y_norm: f64) -> f64 {
    (n as f64) * EPSILON_F64 * x_norm * y_norm
}

/// Theoretical error bound for GEMV.
///
/// For m×k matrix-vector product: |error| ≤ (k+1) * ε * ||A||_F * ||x||
#[must_use]
pub fn gemv_error_bound(k: usize, a_frobenius: f64, x_norm: f64) -> f64 {
    ((k + 1) as f64) * EPSILON_F64 * a_frobenius * x_norm
}

/// Theoretical error bound for GEMM.
///
/// For m×k × k×n matrix product: |error| ≤ (k+1) * ε * ||A||_F * ||B||_F
#[must_use]
pub fn gemm_error_bound(k: usize, a_frobenius: f64, b_frobenius: f64) -> f64 {
    ((k + 1) as f64) * EPSILON_F64 * a_frobenius * b_frobenius
}

/// Compute Frobenius norm of a matrix stored in column-major order.
#[must_use]
pub fn frobenius_norm_f64(matrix: &[f64], rows: usize, cols: usize) -> f64 {
    assert_eq!(
        matrix.len(),
        rows * cols,
        "Matrix size must match dimensions"
    );

    let sum_sq: f64 = matrix.iter().map(|v| v * v).sum();
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level1::{dot, nrm2};
    use crate::level2::{GemvTrans, gemv};
    use crate::level3::gemm;
    use oxiblas_matrix::Mat;

    // ==========================================================================
    // Error Metric Tests
    // ==========================================================================

    #[test]
    fn test_max_absolute_error() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.5, 3.0];
        assert!((max_absolute_error(&x, &y) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_l2_error() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 2.0, 3.0];
        assert!((l2_error(&x, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_relative_forward_error() {
        let exact = [1.0, 2.0, 3.0];
        let computed = [1.0, 2.0, 3.0];
        assert!(relative_forward_error(&computed, &exact) < 1e-10);

        let computed_with_error = [1.001, 2.0, 3.0];
        let rel_err = relative_forward_error(&computed_with_error, &exact);
        assert!(rel_err > 0.0);
        assert!(rel_err < 0.01);
    }

    // ==========================================================================
    // DOT Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_dot_accuracy_random() {
        // Test with pseudo-random values
        let n = 1000;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();

        let computed = dot(&x, &y);
        let reference = dot_reference_f64(&x, &y);

        let rel_error = (computed - reference).abs() / reference.abs();

        // Should be within n * epsilon
        let bound = (n as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 10.0, // Allow 10x margin for implementation differences
            "DOT relative error {} exceeds bound {} * 10",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_dot_accuracy_ill_conditioned() {
        // Test with numbers of very different magnitudes
        let mut x = vec![1e15; 100];
        let mut y = vec![1e15; 100];
        x[50] = 1e-15;
        y[50] = 1e-15;

        let computed = dot(&x, &y);
        let reference = dot_reference_f64(&x, &y);

        // For ill-conditioned data, we expect larger relative error
        let rel_error = (computed - reference).abs() / reference.abs();

        // Should still be reasonable (within square root of condition number)
        assert!(
            rel_error < 1e-10,
            "DOT ill-conditioned relative error {} too large",
            rel_error
        );
    }

    #[test]
    fn test_dot_accuracy_cancellation() {
        // Test near-cancellation case
        let x: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let y: Vec<f64> = vec![1.0; 100];

        let computed = dot(&x, &y);
        // Should be 0 for even length
        assert!(
            computed.abs() < 1e-10,
            "DOT cancellation result {} should be 0",
            computed
        );
    }

    // ==========================================================================
    // GEMV Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_gemv_accuracy_random() {
        let m = 100;
        let n = 80;

        let a_data: Vec<f64> = (0..m * n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, n, &a_data);

        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let mut y_computed = vec![0.0; m];
        let mut y_reference = vec![0.0; m];

        gemv(
            GemvTrans::NoTrans,
            1.0,
            a.as_ref(),
            &x,
            0.0,
            &mut y_computed,
        );
        gemv_reference_f64(false, 1.0, &a_data, m, n, &x, 0.0, &mut y_reference);

        let rel_error = relative_forward_error(&y_computed, &y_reference);

        // Error bound: (k+1) * epsilon
        let bound = ((n + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0, // Allow generous margin
            "GEMV relative error {} exceeds bound {} * 100",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_gemv_accuracy_transpose() {
        let m = 80;
        let n = 100;

        let a_data: Vec<f64> = (0..m * n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, n, &a_data);

        let x: Vec<f64> = (0..m)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let mut y_computed = vec![0.0; n];
        let mut y_reference = vec![0.0; n];

        gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y_computed);
        gemv_reference_f64(true, 1.0, &a_data, m, n, &x, 0.0, &mut y_reference);

        let rel_error = relative_forward_error(&y_computed, &y_reference);
        let bound = ((m + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMV transpose relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    // ==========================================================================
    // GEMM Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_gemm_accuracy_random() {
        let m = 64;
        let k = 48;
        let n = 56;

        let a_data: Vec<f64> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, k, &a_data);

        let b_data: Vec<f64> = (0..k * n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let b = Mat::from_slice(k, n, &b_data);

        let mut c = Mat::zeros(m, n);
        let mut c_reference = vec![0.0; m * n];

        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
        gemm_reference_f64(1.0, &a_data, m, k, &b_data, n, 0.0, &mut c_reference);

        // Convert c to slice for comparison
        let c_computed: Vec<f64> = (0..m * n).map(|i| c[(i % m, i / m)]).collect();

        let rel_error = relative_forward_error(&c_computed, &c_reference);
        let bound = ((k + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMM relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_gemm_accuracy_large() {
        let m = 128;
        let k = 96;
        let n = 112;

        let a_data: Vec<f64> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, k, &a_data);

        let b_data: Vec<f64> = (0..k * n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let b = Mat::from_slice(k, n, &b_data);

        let mut c = Mat::zeros(m, n);
        let mut c_reference = vec![0.0; m * n];

        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
        gemm_reference_f64(1.0, &a_data, m, k, &b_data, n, 0.0, &mut c_reference);

        let c_computed: Vec<f64> = (0..m * n).map(|i| c[(i % m, i / m)]).collect();

        let rel_error = relative_forward_error(&c_computed, &c_reference);
        let bound = ((k + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMM large matrix relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_gemm_accuracy_with_alpha_beta() {
        let m = 32;
        let k = 24;
        let n = 28;

        let a_data: Vec<f64> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, k, &a_data);

        let b_data: Vec<f64> = (0..k * n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let b = Mat::from_slice(k, n, &b_data);

        // Initialize C with non-zero values
        let c_init: Vec<f64> = (0..m * n)
            .map(|i| ((i * 3 + 5) % 1000) as f64 / 1000.0)
            .collect();
        let mut c = Mat::from_slice(m, n, &c_init);
        let mut c_reference = c_init.clone();

        let alpha = 2.5;
        let beta = 0.3;

        gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
        gemm_reference_f64(alpha, &a_data, m, k, &b_data, n, beta, &mut c_reference);

        let c_computed: Vec<f64> = (0..m * n).map(|i| c[(i % m, i / m)]).collect();

        let rel_error = relative_forward_error(&c_computed, &c_reference);
        let bound = ((k + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMM with alpha/beta relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    // ==========================================================================
    // NRM2 Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_nrm2_accuracy() {
        let n = 1000;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();

        let computed = nrm2(&x);
        let reference = l2_norm(&x);

        let rel_error = (computed - reference).abs() / reference;
        assert!(
            rel_error < 1e-14,
            "NRM2 relative error {} too large",
            rel_error
        );
    }

    #[test]
    fn test_nrm2_accuracy_overflow_prevention() {
        // Test that we handle large values without overflow
        let x = vec![1e150; 4];
        let computed = nrm2(&x);
        let expected = 2.0 * 1e150;

        let rel_error = (computed - expected).abs() / expected;
        assert!(
            rel_error < 1e-14,
            "NRM2 overflow prevention error {} too large",
            rel_error
        );
    }

    #[test]
    fn test_nrm2_accuracy_underflow_prevention() {
        // Test that we handle small values without underflow
        let x = vec![1e-150; 4];
        let computed = nrm2(&x);
        let expected = 2.0 * 1e-150;

        let rel_error = (computed - expected).abs() / expected;
        assert!(
            rel_error < 1e-14,
            "NRM2 underflow prevention error {} too large",
            rel_error
        );
    }

    // ==========================================================================
    // Error Bound Tests
    // ==========================================================================

    #[test]
    fn test_error_bound_dot() {
        let n = 100;
        let x_norm = 10.0;
        let y_norm = 5.0;

        let bound = dot_error_bound(n, x_norm, y_norm);
        assert!(bound > 0.0);
        assert!(bound < 1.0); // Should be small for reasonable inputs
    }

    #[test]
    fn test_error_bound_gemv() {
        let k = 100;
        let a_frobenius = 50.0;
        let x_norm = 5.0;

        let bound = gemv_error_bound(k, a_frobenius, x_norm);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_error_bound_gemm() {
        let k = 100;
        let a_frobenius = 50.0;
        let b_frobenius = 40.0;

        let bound = gemm_error_bound(k, a_frobenius, b_frobenius);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_frobenius_norm() {
        // 2x2 matrix [[1, 2], [3, 4]] stored column-major as [1, 3, 2, 4]
        let matrix = [1.0, 3.0, 2.0, 4.0];
        let norm = frobenius_norm_f64(&matrix, 2, 2);
        let expected = (1.0 + 9.0 + 4.0 + 16.0_f64).sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }
}
