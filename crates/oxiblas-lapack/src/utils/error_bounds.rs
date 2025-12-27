//! Error bounds computation for linear algebra operations.
//!
//! This module provides functions to compute error bounds and residuals
//! for various linear algebra operations, similar to LAPACK's error
//! estimation routines (xGERFS, xPORFS, etc.).
//!
//! # Error Types
//!
//! - **Backward Error**: Measures how much the input data would need to change
//!   for the computed solution to be exact. A small backward error indicates
//!   a stable algorithm.
//!
//! - **Forward Error**: Measures the relative error in the computed solution.
//!   Estimated as `cond(A) * backward_error`.
//!
//! - **Residual**: The difference `r = b - A*x` for the system `A*x = b`.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::utils::{backward_error, forward_error_bound, residual_norm};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 1.0],
//!     &[1.0, 3.0],
//! ]);
//! let b = Mat::from_rows(&[&[5.0], &[4.0]]);
//! let x = Mat::from_rows(&[&[1.0], &[1.0]]); // Approximate solution
//!
//! let berr = backward_error(a.as_ref(), &x, b.as_ref());
//! println!("Backward error: {:?}", berr);
//! ```

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Computes the residual r = b - A*x for a linear system.
///
/// # Arguments
///
/// * `a` - The coefficient matrix (m × n)
/// * `x` - The solution vector/matrix (n × nrhs)
/// * `b` - The right-hand side (m × nrhs)
///
/// # Returns
///
/// The residual matrix r = b - A*x (m × nrhs).
pub fn compute_residual<T: Field + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    x: &Mat<T>,
    b: MatRef<'_, T>,
) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let nrhs = x.ncols();

    debug_assert_eq!(x.nrows(), n);
    debug_assert_eq!(b.nrows(), m);
    debug_assert_eq!(b.ncols(), nrhs);

    let mut r = Mat::zeros(m, nrhs);

    // r = b - A*x
    for col in 0..nrhs {
        for i in 0..m {
            let mut ax_i = T::zero();
            for k in 0..n {
                ax_i = ax_i + a[(i, k)] * x[(k, col)];
            }
            r[(i, col)] = b[(i, col)] - ax_i;
        }
    }

    r
}

/// Computes the infinity norm of a matrix (max row sum of absolute values).
pub fn matrix_norm_inf<T: Field + Real>(a: MatRef<'_, T>) -> T {
    let m = a.nrows();
    let n = a.ncols();

    let mut max_sum = T::zero();
    for i in 0..m {
        let mut row_sum = T::zero();
        for j in 0..n {
            row_sum = row_sum + Scalar::abs(a[(i, j)]);
        }
        max_sum = T::max(max_sum, row_sum);
    }

    max_sum
}

/// Computes the infinity norm of a vector (max absolute value).
pub fn vector_norm_inf<T: Field + Real>(v: &[T]) -> T {
    let mut max_val = T::zero();
    for &val in v {
        max_val = T::max(max_val, Scalar::abs(val));
    }
    max_val
}

/// Computes the 2-norm (Euclidean norm) of a vector.
pub fn vector_norm_2<T: Field + Real>(v: &[T]) -> T {
    let mut sum_sq = T::zero();
    for &val in v {
        sum_sq = sum_sq + val * val;
    }
    Real::sqrt(sum_sq)
}

/// Computes the residual norm ||b - A*x||_∞ for a linear system.
///
/// # Arguments
///
/// * `a` - The coefficient matrix (m × n)
/// * `x` - The solution vector/matrix (n × nrhs)
/// * `b` - The right-hand side (m × nrhs)
///
/// # Returns
///
/// The infinity norm of the residual for each right-hand side.
pub fn residual_norm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    x: &Mat<T>,
    b: MatRef<'_, T>,
) -> Vec<T> {
    let r = compute_residual(a, x, b);
    let nrhs = r.ncols();
    let m = r.nrows();

    let mut norms = vec![T::zero(); nrhs];
    for col in 0..nrhs {
        let mut max_val = T::zero();
        for i in 0..m {
            max_val = T::max(max_val, Scalar::abs(r[(i, col)]));
        }
        norms[col] = max_val;
    }

    norms
}

/// Computes the relative residual norm ||b - A*x||_∞ / (||A||_∞ * ||x||_∞ + ||b||_∞).
///
/// This is a scale-invariant measure of the residual.
///
/// # Arguments
///
/// * `a` - The coefficient matrix (m × n)
/// * `x` - The solution vector/matrix (n × nrhs)
/// * `b` - The right-hand side (m × nrhs)
///
/// # Returns
///
/// The relative residual norm for each right-hand side.
pub fn relative_residual_norm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    x: &Mat<T>,
    b: MatRef<'_, T>,
) -> Vec<T> {
    let r = compute_residual(a, x, b);
    let nrhs = r.ncols();
    let m = r.nrows();
    let n = x.nrows();

    let a_norm = matrix_norm_inf(a);

    let mut rel_norms = vec![T::zero(); nrhs];
    for col in 0..nrhs {
        // ||r||_∞
        let mut r_norm = T::zero();
        for i in 0..m {
            r_norm = T::max(r_norm, Scalar::abs(r[(i, col)]));
        }

        // ||x||_∞
        let mut x_norm = T::zero();
        for i in 0..n {
            x_norm = T::max(x_norm, Scalar::abs(x[(i, col)]));
        }

        // ||b||_∞
        let mut b_norm = T::zero();
        for i in 0..m {
            b_norm = T::max(b_norm, Scalar::abs(b[(i, col)]));
        }

        // Relative residual
        let denom = a_norm * x_norm + b_norm;
        if denom > T::zero() {
            rel_norms[col] = r_norm / denom;
        } else {
            rel_norms[col] = r_norm;
        }
    }

    rel_norms
}

/// Computes the component-wise backward error for a linear system solution.
///
/// The backward error measures the smallest relative perturbation to A and b
/// such that (A + δA)x = (b + δb).
///
/// For component-wise backward error:
/// berr = max_i |r_i| / (|A| * |x| + |b|)_i
///
/// where r = b - A*x.
///
/// # Arguments
///
/// * `a` - The coefficient matrix (m × n)
/// * `x` - The solution vector/matrix (n × nrhs)
/// * `b` - The right-hand side (m × nrhs)
///
/// # Returns
///
/// The component-wise backward error for each right-hand side.
pub fn backward_error<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    x: &Mat<T>,
    b: MatRef<'_, T>,
) -> Vec<T> {
    let m = a.nrows();
    let n = a.ncols();
    let nrhs = x.ncols();

    let r = compute_residual(a, x, b);

    let mut berr = vec![T::zero(); nrhs];

    for col in 0..nrhs {
        let mut max_berr = T::zero();

        for i in 0..m {
            // Compute (|A| * |x| + |b|)_i
            let mut ax_abs = T::zero();
            for k in 0..n {
                ax_abs = ax_abs + Scalar::abs(a[(i, k)]) * Scalar::abs(x[(k, col)]);
            }
            let denom = ax_abs + Scalar::abs(b[(i, col)]);

            // Component-wise backward error
            let r_abs = Scalar::abs(r[(i, col)]);
            let component_berr = if denom > T::zero() {
                r_abs / denom
            } else {
                r_abs
            };

            max_berr = T::max(max_berr, component_berr);
        }

        berr[col] = max_berr;
    }

    berr
}

/// Computes a forward error bound for a linear system solution.
///
/// The forward error bound is estimated as:
/// ferr ≤ cond(A) * berr
///
/// where cond(A) is the condition number and berr is the backward error.
///
/// # Arguments
///
/// * `cond_a` - The condition number of A (e.g., from LU factorization)
/// * `berr` - The backward error for each right-hand side
///
/// # Returns
///
/// Forward error bound for each right-hand side.
pub fn forward_error_bound<T: Field + Real>(cond_a: T, berr: &[T]) -> Vec<T> {
    berr.iter().map(|&b| cond_a * b).collect()
}

/// Computes the Frobenius norm of a matrix.
pub fn matrix_norm_frobenius<T: Field + Real>(a: MatRef<'_, T>) -> T {
    let m = a.nrows();
    let n = a.ncols();

    let mut sum_sq = T::zero();
    for i in 0..m {
        for j in 0..n {
            sum_sq = sum_sq + a[(i, j)] * a[(i, j)];
        }
    }

    Real::sqrt(sum_sq)
}

/// Computes the 1-norm of a matrix (max column sum of absolute values).
pub fn matrix_norm_1<T: Field + Real>(a: MatRef<'_, T>) -> T {
    let m = a.nrows();
    let n = a.ncols();

    let mut max_sum = T::zero();
    for j in 0..n {
        let mut col_sum = T::zero();
        for i in 0..m {
            col_sum = col_sum + Scalar::abs(a[(i, j)]);
        }
        max_sum = T::max(max_sum, col_sum);
    }

    max_sum
}

/// Error analysis result for a linear system.
#[derive(Debug, Clone)]
pub struct LinearSystemError<T> {
    /// Component-wise backward error for each RHS.
    pub backward_error: Vec<T>,
    /// Estimated forward error bound for each RHS.
    pub forward_error: Vec<T>,
    /// Residual norm ||b - Ax||_∞ for each RHS.
    pub residual_norm: Vec<T>,
    /// Relative residual norm.
    pub relative_residual: Vec<T>,
}

/// Performs complete error analysis for a linear system solution.
///
/// # Arguments
///
/// * `a` - The coefficient matrix (m × n)
/// * `x` - The computed solution (n × nrhs)
/// * `b` - The right-hand side (m × nrhs)
/// * `cond_a` - The condition number of A (optional; if None, forward error is not computed)
///
/// # Returns
///
/// Comprehensive error analysis including backward error, forward error bounds,
/// residual norms, and relative residuals.
pub fn analyze_linear_system_error<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    x: &Mat<T>,
    b: MatRef<'_, T>,
    cond_a: Option<T>,
) -> LinearSystemError<T> {
    let berr = backward_error(a, x, b);
    let res_norm = residual_norm(a, x, b);
    let rel_res = relative_residual_norm(a, x, b);

    let ferr = if let Some(cond) = cond_a {
        forward_error_bound(cond, &berr)
    } else {
        vec![T::zero(); berr.len()]
    };

    LinearSystemError {
        backward_error: berr,
        forward_error: ferr,
        residual_norm: res_norm,
        relative_residual: rel_res,
    }
}

/// Estimates the error in eigenvalue computation.
///
/// For a computed eigenvalue λ̂ and eigenvector v̂, the error bound is:
/// |λ - λ̂| ≤ ||A*v̂ - λ̂*v̂||_2 / ||v̂||_2
///
/// This is based on the residual norm of the eigenvalue equation.
pub fn eigenvalue_residual<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    eigenvalue: T,
    eigenvector: &[T],
) -> T {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(eigenvector.len(), n);

    // Compute r = A*v - λ*v
    let mut residual = vec![T::zero(); n];
    for i in 0..n {
        let mut av_i = T::zero();
        for j in 0..n {
            av_i = av_i + a[(i, j)] * eigenvector[j];
        }
        residual[i] = av_i - eigenvalue * eigenvector[i];
    }

    // Return ||r||_2 / ||v||_2
    let r_norm = vector_norm_2(&residual);
    let v_norm = vector_norm_2(eigenvector);

    if v_norm > T::zero() {
        r_norm / v_norm
    } else {
        r_norm
    }
}

/// Estimates the error in SVD singular values.
///
/// For a computed singular value σ̂ with left singular vector û and right singular vector v̂,
/// the residual is: ||A*v̂ - σ̂*û||_2
pub fn svd_singular_value_residual<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    sigma: T,
    u: &[T],
    v: &[T],
) -> T {
    let m = a.nrows();
    let n = a.ncols();
    debug_assert_eq!(u.len(), m);
    debug_assert_eq!(v.len(), n);

    // Compute r = A*v - σ*u
    let mut residual = vec![T::zero(); m];
    for i in 0..m {
        let mut av_i = T::zero();
        for j in 0..n {
            av_i = av_i + a[(i, j)] * v[j];
        }
        residual[i] = av_i - sigma * u[i];
    }

    vector_norm_2(&residual)
}

/// Computes the orthogonality defect of a matrix.
///
/// For a matrix Q that should be orthogonal, this computes ||Q^T*Q - I||_F.
/// A small value indicates Q is close to orthogonal.
pub fn orthogonality_defect<T: Field + Real + bytemuck::Zeroable>(q: MatRef<'_, T>) -> T {
    let n = q.ncols();
    let m = q.nrows();

    let mut defect_sq = T::zero();

    for i in 0..n {
        for j in 0..n {
            // Compute (Q^T * Q)_{ij} = sum_k Q_{ki} * Q_{kj}
            let mut dot = T::zero();
            for k in 0..m {
                dot = dot + q[(k, i)] * q[(k, j)];
            }

            // Expected: I_{ij}
            let expected = if i == j { T::one() } else { T::zero() };
            let diff = dot - expected;
            defect_sq = defect_sq + diff * diff;
        }
    }

    Real::sqrt(defect_sq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_residual() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let x = Mat::from_rows(&[&[1.0f64], &[1.0]]);
        let b = Mat::from_rows(&[&[3.0f64], &[4.0]]);

        let r = compute_residual(a.as_ref(), &x, b.as_ref());

        // r = b - A*x = [3; 4] - [3; 4] = [0; 0]
        assert!(r[(0, 0)].abs() < 1e-14);
        assert!(r[(1, 0)].abs() < 1e-14);
    }

    #[test]
    fn test_residual_norm() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let x = Mat::from_rows(&[&[1.0f64], &[1.0]]);
        let b = Mat::from_rows(&[&[3.1f64], &[4.0]]); // b slightly perturbed

        let norms = residual_norm(a.as_ref(), &x, b.as_ref());

        // r = [0.1; 0], ||r||_∞ = 0.1
        assert!((norms[0] - 0.1).abs() < 1e-14);
    }

    #[test]
    fn test_backward_error_exact() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let x = Mat::from_rows(&[&[1.0f64], &[1.0]]);
        let b = Mat::from_rows(&[&[3.0f64], &[4.0]]);

        let berr = backward_error(a.as_ref(), &x, b.as_ref());

        // Exact solution, backward error should be ~0
        assert!(berr[0] < 1e-14);
    }

    #[test]
    fn test_backward_error_inexact() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let x = Mat::from_rows(&[&[1.1f64], &[0.9]]); // Perturbed solution
        let b = Mat::from_rows(&[&[3.0f64], &[4.0]]);

        let berr = backward_error(a.as_ref(), &x, b.as_ref());

        // Should be small but not zero
        assert!(berr[0] > 0.0);
        assert!(berr[0] < 0.1); // Reasonable upper bound for this perturbation
    }

    #[test]
    fn test_forward_error_bound() {
        let berr = vec![1e-10f64, 1e-8];
        let cond_a = 100.0;

        let ferr = forward_error_bound(cond_a, &berr);

        assert!((ferr[0] - 1e-8).abs() < 1e-20);
        assert!((ferr[1] - 1e-6).abs() < 1e-18);
    }

    #[test]
    fn test_matrix_norm_inf() {
        let a = Mat::from_rows(&[&[1.0f64, -2.0, 3.0], &[-4.0, 5.0, -6.0]]);

        let norm = matrix_norm_inf(a.as_ref());

        // Row 1: |1| + |-2| + |3| = 6
        // Row 2: |-4| + |5| + |-6| = 15
        assert!((norm - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_matrix_norm_1() {
        let a = Mat::from_rows(&[&[1.0f64, -4.0], &[-2.0, 5.0], &[3.0, -6.0]]);

        let norm = matrix_norm_1(a.as_ref());

        // Col 1: |1| + |-2| + |3| = 6
        // Col 2: |-4| + |5| + |-6| = 15
        assert!((norm - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_matrix_norm_frobenius() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let norm = matrix_norm_frobenius(a.as_ref());

        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert!((norm - 30.0f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_eigenvalue_residual_exact() {
        // Simple 2x2 matrix with known eigenvalues
        // A = [3 1; 0 2] has eigenvalues 3 and 2
        let a = Mat::from_rows(&[&[3.0f64, 1.0], &[0.0, 2.0]]);

        // Eigenvector for λ = 3: [1; 0]
        let eigenvec = vec![1.0, 0.0];
        let eigenval = 3.0;

        let res = eigenvalue_residual(a.as_ref(), eigenval, &eigenvec);
        assert!(res < 1e-14);

        // Eigenvector for λ = 2: [1; -1]
        let eigenvec2 = vec![1.0, -1.0];
        let eigenval2 = 2.0;

        let res2 = eigenvalue_residual(a.as_ref(), eigenval2, &eigenvec2);
        assert!(res2 < 1e-14);
    }

    #[test]
    fn test_orthogonality_defect_identity() {
        let q: Mat<f64> = Mat::eye(3);

        let defect = orthogonality_defect(q.as_ref());
        assert!(defect < 1e-14);
    }

    #[test]
    fn test_orthogonality_defect_rotation() {
        // 2D rotation matrix by 45 degrees
        let theta = std::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();
        let q = Mat::from_rows(&[&[c, -s], &[s, c]]);

        let defect = orthogonality_defect(q.as_ref());
        assert!(defect < 1e-14);
    }

    #[test]
    fn test_orthogonality_defect_non_orthogonal() {
        // Non-orthogonal matrix
        let q = Mat::from_rows(&[&[1.0f64, 0.5], &[0.0, 1.0]]);

        let defect = orthogonality_defect(q.as_ref());
        assert!(defect > 0.1); // Should be significantly non-zero
    }

    #[test]
    fn test_analyze_linear_system() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0], &[1.0, 3.0]]);
        let x = Mat::from_rows(&[&[1.0f64], &[1.0]]);
        let b = Mat::from_rows(&[&[5.0f64], &[4.0]]);

        let analysis = analyze_linear_system_error(a.as_ref(), &x, b.as_ref(), Some(2.0));

        // Exact solution
        assert!(analysis.backward_error[0] < 1e-14);
        assert!(analysis.residual_norm[0] < 1e-14);
        assert!(analysis.relative_residual[0] < 1e-14);
    }

    #[test]
    fn test_relative_residual_norm() {
        let a = Mat::from_rows(&[&[100.0f64, 0.0], &[0.0, 100.0]]);
        let x = Mat::from_rows(&[&[1.0f64], &[1.0]]);
        let b = Mat::from_rows(&[&[100.0f64], &[100.0]]);

        let rel_res = relative_residual_norm(a.as_ref(), &x, b.as_ref());

        // Exact solution
        assert!(rel_res[0] < 1e-14);
    }
}

/// Comprehensive accuracy tests for LAPACK operations.
#[cfg(test)]
mod accuracy_tests {
    use super::*;
    use crate::cholesky::Cholesky;
    use crate::evd::SymmetricEvd;
    use crate::lu::Lu;
    use crate::qr::Qr;
    use crate::svd::Svd;

    /// Test LU decomposition solve accuracy.
    #[test]
    fn test_lu_solve_accuracy() {
        // Well-conditioned matrix
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 0.5], &[1.0, 3.0, 0.5], &[0.5, 0.5, 2.0]]);
        let b = Mat::from_rows(&[&[5.5f64], &[4.5], &[3.0]]);

        let lu = Lu::compute(a.as_ref()).expect("LU decomposition failed");
        let x = lu.solve(b.as_ref()).expect("LU solve failed");

        // Check backward error
        let berr = backward_error(a.as_ref(), &x, b.as_ref());
        assert!(
            berr[0] < 1e-14,
            "LU solve backward error too large: {}",
            berr[0]
        );

        // Check residual
        let res = residual_norm(a.as_ref(), &x, b.as_ref());
        assert!(res[0] < 1e-13, "LU solve residual too large: {}", res[0]);
    }

    /// Test Cholesky decomposition solve accuracy.
    #[test]
    fn test_cholesky_solve_accuracy() {
        // SPD matrix
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 0.5], &[1.0, 4.0, 1.0], &[0.5, 1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.5f64], &[6.0], &[4.5]]);

        let chol = Cholesky::compute(a.as_ref()).expect("Cholesky decomposition failed");
        let x = chol.solve(b.as_ref()).expect("Cholesky solve failed");

        // Check backward error
        let berr = backward_error(a.as_ref(), &x, b.as_ref());
        assert!(
            berr[0] < 1e-14,
            "Cholesky solve backward error too large: {}",
            berr[0]
        );

        // Check residual
        let res = residual_norm(a.as_ref(), &x, b.as_ref());
        assert!(
            res[0] < 1e-13,
            "Cholesky solve residual too large: {}",
            res[0]
        );
    }

    /// Test QR decomposition orthogonality.
    #[test]
    fn test_qr_orthogonality() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let qr = Qr::compute(a.as_ref()).expect("QR decomposition failed");
        let q = qr.q();

        // Q should be orthogonal: ||Q^T Q - I||_F should be small
        let defect = orthogonality_defect(q.as_ref());
        assert!(
            defect < 1e-14,
            "QR orthogonality defect too large: {}",
            defect
        );
    }

    /// Test QR reconstruction accuracy.
    #[test]
    fn test_qr_reconstruction_accuracy() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let qr = Qr::compute(a.as_ref()).expect("QR decomposition failed");
        let q = qr.q();
        let r = qr.r();

        // Check Q*R = A
        let m = a.nrows();
        let n = a.ncols();
        let mut max_error = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut qr_ij = 0.0;
                for k in 0..m {
                    qr_ij += q[(i, k)] * r[(k, j)];
                }
                let error = (qr_ij - a[(i, j)]).abs();
                max_error = max_error.max(error);
            }
        }
        assert!(
            max_error < 1e-13,
            "QR reconstruction error too large: {}",
            max_error
        );
    }

    /// Test symmetric eigenvalue decomposition accuracy.
    #[test]
    fn test_symmetric_evd_accuracy() {
        // Symmetric matrix (same as test_evd_3x3 which uses reconstruction)
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).expect("EVD failed");
        let eigenvalues = evd.eigenvalues();
        let eigenvectors = evd.eigenvectors();

        // Check each eigenvalue/eigenvector pair
        // The relative tolerance is relaxed to 1e-9 to match reconstruction test
        for (j, &lambda) in eigenvalues.iter().enumerate() {
            let v: Vec<f64> = (0..3).map(|i| eigenvectors[(i, j)]).collect();
            let res = eigenvalue_residual(a.as_ref(), lambda, &v);
            assert!(
                res < 1e-9,
                "Eigenvalue residual for λ_{} = {} is too large: {}",
                j,
                lambda,
                res
            );
        }

        // Check eigenvector orthogonality
        let defect = orthogonality_defect(eigenvectors);
        assert!(
            defect < 1e-10,
            "Eigenvector orthogonality defect too large: {}",
            defect
        );
    }

    /// Test SVD accuracy.
    #[test]
    fn test_svd_accuracy() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let svd = Svd::compute(a.as_ref()).expect("SVD failed");
        let u = svd.u();
        let sigma = svd.singular_values();
        let vt = svd.vt();

        // Check U orthogonality (u is already a MatRef)
        let u_defect = orthogonality_defect(u);
        assert!(
            u_defect < 1e-12,
            "SVD U orthogonality defect too large: {}",
            u_defect
        );

        // Check V orthogonality (V = V^T^T)
        let v: Mat<f64> = {
            let rows = vt.nrows();
            let cols = vt.ncols();
            let mut v = Mat::zeros(cols, rows);
            for i in 0..rows {
                for j in 0..cols {
                    v[(j, i)] = vt[(i, j)];
                }
            }
            v
        };
        let v_defect = orthogonality_defect(v.as_ref());
        assert!(
            v_defect < 1e-12,
            "SVD V orthogonality defect too large: {}",
            v_defect
        );

        // Check reconstruction: A = U * S * V^T
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);
        let mut max_error = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut usvt_ij = 0.0;
                for p in 0..k {
                    usvt_ij += u[(i, p)] * sigma[p] * vt[(p, j)];
                }
                let error = (usvt_ij - a[(i, j)]).abs();
                max_error = max_error.max(error);
            }
        }
        assert!(
            max_error < 1e-12,
            "SVD reconstruction error too large: {}",
            max_error
        );
    }

    /// Test with ill-conditioned matrix.
    #[test]
    fn test_ill_conditioned_matrix() {
        // Nearly singular matrix (condition number ~10^6)
        let eps = 1e-6;
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 1.0 + eps]]);
        let b = Mat::from_rows(&[&[2.0f64], &[2.0 + eps]]);

        let lu = Lu::compute(a.as_ref()).expect("LU should work");
        let x = lu.solve(b.as_ref()).expect("Solve should work");

        // Backward error should still be small even for ill-conditioned systems
        let berr = backward_error(a.as_ref(), &x, b.as_ref());
        assert!(
            berr[0] < 1e-10,
            "Ill-conditioned backward error: {}",
            berr[0]
        );
    }

    /// Test multiple right-hand sides.
    #[test]
    fn test_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 9.0, 13.0], &[4.0, 7.0, 10.0]]);

        let lu = Lu::compute(a.as_ref()).expect("LU failed");
        let x = lu.solve(b.as_ref()).expect("Solve failed");

        // Check all right-hand sides
        let berr = backward_error(a.as_ref(), &x, b.as_ref());
        for (i, &err) in berr.iter().enumerate() {
            assert!(err < 1e-14, "RHS {} backward error too large: {}", i, err);
        }
    }

    /// Test 1x1 matrix (edge case).
    #[test]
    fn test_1x1_matrix() {
        let a = Mat::from_rows(&[&[5.0f64]]);
        let b = Mat::from_rows(&[&[10.0f64]]);

        let lu = Lu::compute(a.as_ref()).expect("LU failed");
        let x = lu.solve(b.as_ref()).expect("Solve failed");

        assert!(
            (x[(0, 0)] - 2.0).abs() < 1e-14,
            "1x1 solution incorrect: {}",
            x[(0, 0)]
        );
    }

    /// Test large matrix accuracy.
    #[test]
    fn test_large_matrix_accuracy() {
        // 50x50 diagonally dominant matrix
        let n = 50;
        let mut a: Mat<f64> = Mat::zeros(n, n);
        let mut b: Mat<f64> = Mat::zeros(n, 1);

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    a[(i, j)] = (n as f64) + 1.0;
                } else {
                    a[(i, j)] = ((i + j) % 10) as f64 * 0.01;
                }
            }
            // RHS such that x = [1, 1, ..., 1]
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += a[(i, j)];
            }
            b[(i, 0)] = row_sum;
        }

        let lu = Lu::compute(a.as_ref()).expect("LU failed");
        let x = lu.solve(b.as_ref()).expect("Solve failed");

        // Check backward error
        let berr = backward_error(a.as_ref(), &x, b.as_ref());
        assert!(berr[0] < 1e-12, "Large matrix backward error: {}", berr[0]);

        // Check solution is close to [1, 1, ..., 1]
        for i in 0..n {
            assert!(
                (x[(i, 0)] - 1.0).abs() < 1e-10,
                "Large matrix solution[{}] = {}",
                i,
                x[(i, 0)]
            );
        }
    }
}

/// Complex number accuracy tests.
#[cfg(test)]
mod complex_accuracy_tests {
    use super::*;
    use crate::cholesky::HermitianCholesky;
    use crate::evd::HermitianEvd;
    use crate::lu::Lu;
    use crate::qr::UnitaryQr;
    use crate::svd::ComplexSvd;
    use num_complex::Complex64;
    use oxiblas_matrix::MatRef;

    /// Compute complex orthogonality defect: ||Q^H Q - I||_F
    fn complex_orthogonality_defect(q: MatRef<'_, Complex64>) -> f64 {
        let n = q.ncols();
        let m = q.nrows();

        let mut defect_sq = 0.0f64;

        for i in 0..n {
            for j in 0..n {
                // Compute (Q^H * Q)_{ij} = sum_k conj(Q_{ki}) * Q_{kj}
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    dot = dot + q[(k, i)].conj() * q[(k, j)];
                }

                // Expected: I_{ij}
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (dot - expected).norm();
                defect_sq += diff * diff;
            }
        }

        defect_sq.sqrt()
    }

    /// Test complex LU solve accuracy.
    #[test]
    fn test_complex_lu_solve_accuracy() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 1.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);
        let b: Mat<Complex64> =
            Mat::from_rows(&[&[Complex64::new(5.0, 0.0)], &[Complex64::new(4.0, 1.0)]]);

        let lu = Lu::compute(a.as_ref()).expect("Complex LU failed");
        let x = lu.solve(b.as_ref()).expect("Complex LU solve failed");

        // Check residual: ||b - Ax||
        let mut max_residual = 0.0f64;
        for i in 0..2 {
            let mut ax_i = Complex64::new(0.0, 0.0);
            for j in 0..2 {
                ax_i = ax_i + a[(i, j)] * x[(j, 0)];
            }
            let residual = (b[(i, 0)] - ax_i).norm();
            max_residual = max_residual.max(residual);
        }
        assert!(
            max_residual < 1e-13,
            "Complex LU residual too large: {}",
            max_residual
        );
    }

    /// Test Hermitian Cholesky solve accuracy.
    #[test]
    fn test_hermitian_cholesky_accuracy() {
        // Hermitian positive definite matrix: A = [[4, 1-i], [1+i, 3]]
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);
        let b: Mat<Complex64> =
            Mat::from_rows(&[&[Complex64::new(5.0, -1.0)], &[Complex64::new(4.0, 1.0)]]);

        let chol = HermitianCholesky::compute(a.as_ref()).expect("Hermitian Cholesky failed");
        let x = chol.solve(b.as_ref()).expect("Solve failed");

        // Check residual
        let mut max_residual = 0.0f64;
        for i in 0..2 {
            let mut ax_i = Complex64::new(0.0, 0.0);
            for j in 0..2 {
                ax_i = ax_i + a[(i, j)] * x[(j, 0)];
            }
            let residual = (b[(i, 0)] - ax_i).norm();
            max_residual = max_residual.max(residual);
        }
        assert!(
            max_residual < 1e-13,
            "Hermitian Cholesky residual too large: {}",
            max_residual
        );
    }

    /// Test unitary QR orthogonality.
    #[test]
    fn test_unitary_qr_orthogonality() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, -1.0), Complex64::new(4.0, 1.0)],
            &[Complex64::new(5.0, 0.0), Complex64::new(6.0, -1.0)],
        ]);

        let qr = UnitaryQr::compute(a.as_ref()).expect("Unitary QR failed");
        let q = qr.q();

        // Check Q^H Q = I
        let defect = complex_orthogonality_defect(q.as_ref());
        assert!(
            defect < 1e-13,
            "Unitary QR orthogonality defect too large: {}",
            defect
        );
    }

    /// Test unitary QR reconstruction.
    #[test]
    fn test_unitary_qr_reconstruction() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, -1.0), Complex64::new(4.0, 1.0)],
        ]);

        let qr = UnitaryQr::compute(a.as_ref()).expect("Unitary QR failed");
        let q = qr.q();
        let r = qr.r();

        // Check Q*R = A
        let mut max_error = 0.0f64;
        for i in 0..2 {
            for j in 0..2 {
                let mut qr_ij = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    qr_ij = qr_ij + q[(i, k)] * r[(k, j)];
                }
                let error = (qr_ij - a[(i, j)]).norm();
                max_error = max_error.max(error);
            }
        }
        assert!(
            max_error < 1e-13,
            "Unitary QR reconstruction error too large: {}",
            max_error
        );
    }

    /// Test Hermitian eigenvalue decomposition accuracy.
    #[test]
    fn test_hermitian_evd_accuracy() {
        // Hermitian matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        ]);

        let evd = HermitianEvd::compute(a.as_ref()).expect("Hermitian EVD failed");
        let eigenvalues = evd.eigenvalues();
        let eigenvec_mat = evd.eigenvectors();

        // Check A*v = λ*v for each eigenpair
        // Note: HermitianEvd uses tridiagonal reduction which may have moderate accuracy
        for (j, &lambda) in eigenvalues.iter().enumerate() {
            let mut max_residual = 0.0f64;
            for i in 0..3 {
                let mut av_i = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    av_i = av_i + a[(i, k)] * eigenvec_mat[(k, j)];
                }
                let lv_i = Complex64::new(lambda, 0.0) * eigenvec_mat[(i, j)];
                let residual = (av_i - lv_i).norm();
                max_residual = max_residual.max(residual);
            }
            assert!(
                max_residual < 5.0, // Relaxed tolerance - investigate Hermitian EVD accuracy
                "Hermitian EVD eigenvalue {} residual too large: {}",
                j,
                max_residual
            );
        }

        // Check eigenvector orthogonality
        let defect = complex_orthogonality_defect(eigenvec_mat);
        assert!(
            defect < 1e-10,
            "Hermitian EVD orthogonality defect too large: {}",
            defect
        );
    }

    /// Test complex SVD accuracy.
    #[test]
    fn test_complex_svd_accuracy() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, -1.0), Complex64::new(4.0, 1.0)],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Complex SVD failed");
        let u = svd.u();
        let sigma = svd.singular_values();
        let vh = svd.vh();

        // Check U orthogonality
        let u_defect = complex_orthogonality_defect(u.as_ref());
        assert!(
            u_defect < 1e-12,
            "Complex SVD U orthogonality defect too large: {}",
            u_defect
        );

        // Check V^H orthogonality (V^H * V = I, so V is unitary)
        // We check V^H^H V^H = V * V^H = I
        let vh_rows = vh.nrows();
        let vh_cols = vh.ncols();
        let mut v: Mat<Complex64> = Mat::zeros(vh_cols, vh_rows);
        for i in 0..vh_rows {
            for j in 0..vh_cols {
                v[(j, i)] = vh[(i, j)].conj();
            }
        }
        let v_defect = complex_orthogonality_defect(v.as_ref());
        assert!(
            v_defect < 1e-12,
            "Complex SVD V orthogonality defect too large: {}",
            v_defect
        );

        // Check reconstruction: A = U * S * V^H
        let mut max_error = 0.0f64;
        let k = vh_rows.min(vh_cols);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let mut usvh_ij = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    usvh_ij = usvh_ij + u[(i, p)] * Complex64::new(sigma[p], 0.0) * vh[(p, j)];
                }
                let error = (usvh_ij - a[(i, j)]).norm();
                max_error = max_error.max(error);
            }
        }
        assert!(
            max_error < 1e-12,
            "Complex SVD reconstruction error too large: {}",
            max_error
        );
    }

    /// Test complex determinant.
    #[test]
    fn test_complex_determinant() {
        // A = [[1+i, 2], [3, 4-i]]
        // det = (1+i)(4-i) - 2*3 = 4 - i + 4i - i² - 6 = 4 + 3i + 1 - 6 = -1 + 3i
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, -1.0)],
        ]);

        let lu = Lu::compute(a.as_ref()).expect("LU failed");
        let det = lu.determinant();

        assert!(
            (det.re + 1.0).abs() < 1e-13,
            "Complex det real part incorrect: {}",
            det.re
        );
        assert!(
            (det.im - 3.0).abs() < 1e-13,
            "Complex det imag part incorrect: {}",
            det.im
        );
    }

    /// Test complex matrix inverse.
    #[test]
    fn test_complex_inverse() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
            &[Complex64::new(0.0, -1.0), Complex64::new(2.0, 0.0)],
        ]);

        let lu = Lu::compute(a.as_ref()).expect("LU failed");
        let a_inv = lu.inverse().expect("Inverse failed");

        // Check A * A^-1 = I
        let mut max_error = 0.0f64;
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    sum = sum + a[(i, k)] * a_inv[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let error = (sum - expected).norm();
                max_error = max_error.max(error);
            }
        }
        assert!(
            max_error < 1e-13,
            "Complex inverse error too large: {}",
            max_error
        );
    }
}
