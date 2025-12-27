//! Expert symmetric linear system solvers.
//!
//! Provides enhanced solve routines for symmetric (possibly indefinite) matrices
//! with equilibration, condition estimation, and error bounds,
//! similar to LAPACK's xSYSVX routines.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::cholesky::{Ldlt, LdltError};
use crate::utils::{norm_1, norm_inf};

/// Error type for expert symmetric solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSymmetricSolveError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix and vector dimensions don't match.
    DimensionMismatch,
    /// Matrix is singular.
    SingularMatrix,
    /// Matrix is singular to working precision.
    SingularToWorkingPrecision,
}

impl core::fmt::Display for ExpertSymmetricSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::DimensionMismatch => write!(f, "Matrix and vector dimensions do not match"),
            Self::SingularMatrix => write!(f, "Matrix is singular"),
            Self::SingularToWorkingPrecision => {
                write!(f, "Matrix is singular to working precision")
            }
        }
    }
}

impl std::error::Error for ExpertSymmetricSolveError {}

impl From<LdltError> for ExpertSymmetricSolveError {
    fn from(e: LdltError) -> Self {
        match e {
            LdltError::Singular { .. } => Self::SingularMatrix,
            LdltError::NotSquare { .. } => Self::NotSquare,
            LdltError::DimensionMismatch { .. } => Self::DimensionMismatch,
        }
    }
}

/// Result of expert symmetric solve operation.
#[derive(Debug, Clone)]
pub struct ExpertSymmetricSolveResult<T: Scalar> {
    /// Solution matrix X.
    pub solution: Mat<T>,
    /// Reciprocal condition number estimate (1/kappa).
    pub rcond: T,
    /// Forward error bound for each right-hand side.
    pub forward_error: Vec<T>,
    /// Backward error bound for each right-hand side.
    pub backward_error: Vec<T>,
    /// Scaling factors (if equilibration was applied).
    pub scale: Option<Vec<T>>,
    /// Whether equilibration was applied.
    pub equilibrated: bool,
    /// Matrix inertia (positive, negative, zero eigenvalue counts).
    pub inertia: (usize, usize, usize),
}

/// Solves a symmetric linear system Ax = b with expert options.
///
/// This routine provides:
/// - LDL^T factorization (works for indefinite matrices)
/// - Optional equilibration (symmetric scaling) to improve condition
/// - Condition number estimation
/// - Forward and backward error bounds
/// - Matrix inertia information
///
/// # Arguments
///
/// * `a` - Symmetric coefficient matrix A (n×n)
/// * `b` - Right-hand side matrix B (n×nrhs)
/// * `equilibrate` - Whether to apply symmetric equilibration
///
/// # Returns
///
/// `ExpertSymmetricSolveResult` containing the solution and diagnostic information.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::expert_symmetric::solve_symmetric_expert;
/// use oxiblas_matrix::Mat;
///
/// // Indefinite symmetric matrix
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[2.0, 1.0],
/// ]);
/// let b = Mat::from_rows(&[&[3.0], &[3.0]]);
///
/// let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false).unwrap();
/// println!("Solution: {:?}", result.solution);
/// println!("Reciprocal condition: {}", result.rcond);
/// println!("Inertia: {:?}", result.inertia);
/// ```
pub fn solve_symmetric_expert<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    equilibrate: bool,
) -> Result<ExpertSymmetricSolveResult<T>, ExpertSymmetricSolveError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n {
        return Err(ExpertSymmetricSolveError::NotSquare);
    }
    if b.nrows() != n {
        return Err(ExpertSymmetricSolveError::DimensionMismatch);
    }

    // Handle empty case
    if n == 0 {
        return Ok(ExpertSymmetricSolveResult {
            solution: Mat::zeros(0, nrhs),
            rcond: T::one(),
            forward_error: vec![],
            backward_error: vec![],
            scale: None,
            equilibrated: false,
            inertia: (0, 0, 0),
        });
    }

    // Apply equilibration if requested
    let (a_scaled, b_scaled, scale, did_equilibrate) = if equilibrate {
        apply_symmetric_equilibration(a, b)
    } else {
        // Just copy
        let mut a_copy = Mat::zeros(n, n);
        let mut b_copy = Mat::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..n {
                a_copy[(i, j)] = a[(i, j)];
            }
            for j in 0..nrhs {
                b_copy[(i, j)] = b[(i, j)];
            }
        }
        (a_copy, b_copy, None, false)
    };

    // Compute 1-norm of A before factorization (for rcond)
    let anorm = norm_1(a_scaled.as_ref());

    // Compute LDL^T decomposition
    let ldlt = Ldlt::compute(a_scaled.as_ref())?;

    // Get inertia
    let inertia = ldlt.inertia();

    // Solve the system
    let x_scaled = ldlt.solve(b_scaled.as_ref())?;

    // Estimate reciprocal condition number
    let rcond = estimate_rcond_ldlt(&ldlt, anorm, n);

    // Check if matrix is singular to working precision
    let eps = <T as Scalar>::epsilon();
    if rcond < eps {
        return Err(ExpertSymmetricSolveError::SingularToWorkingPrecision);
    }

    // Compute error bounds
    let (forward_error, backward_error) =
        compute_error_bounds_ldlt(&a_scaled, &b_scaled, &x_scaled, &ldlt, n, nrhs);

    // Unscale solution if equilibration was applied
    let solution = if let Some(ref s) = scale {
        unscale_solution_symmetric(&x_scaled, s)
    } else {
        x_scaled
    };

    Ok(ExpertSymmetricSolveResult {
        solution,
        rcond,
        forward_error,
        backward_error,
        scale,
        equilibrated: did_equilibrate,
        inertia,
    })
}

/// Apply symmetric equilibration to matrix A and right-hand side b.
///
/// For symmetric matrices, we use D A D where D = diag(1/sqrt(|a_ii|)).
fn apply_symmetric_equilibration<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> (Mat<T>, Mat<T>, Option<Vec<T>>, bool) {
    let n = a.nrows();
    let nrhs = b.ncols();

    // Compute scaling: s[i] = 1 / sqrt(|a[i,i]|)
    let mut scale = vec![T::one(); n];
    let mut needs_scaling = false;

    for i in 0..n {
        let diag = Scalar::abs(a[(i, i)]);
        if diag > T::zero() {
            let s = T::one() / Real::sqrt(diag);
            // Only scale if it makes a significant difference
            if Scalar::abs(s - T::one()) > T::from_f64(0.1).unwrap_or(T::zero()) {
                scale[i] = s;
                needs_scaling = true;
            }
        }
    }

    if !needs_scaling {
        // No scaling needed, just copy
        let mut a_copy = Mat::zeros(n, n);
        let mut b_copy = Mat::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..n {
                a_copy[(i, j)] = a[(i, j)];
            }
            for j in 0..nrhs {
                b_copy[(i, j)] = b[(i, j)];
            }
        }
        return (a_copy, b_copy, None, false);
    }

    // Apply symmetric scaling: A' = D A D, b' = D b
    let mut a_scaled = Mat::zeros(n, n);
    let mut b_scaled = Mat::zeros(n, nrhs);

    for i in 0..n {
        for j in 0..n {
            a_scaled[(i, j)] = scale[i] * a[(i, j)] * scale[j];
        }
        for j in 0..nrhs {
            b_scaled[(i, j)] = scale[i] * b[(i, j)];
        }
    }

    (a_scaled, b_scaled, Some(scale), true)
}

/// Estimate reciprocal condition number for LDL^T factorization.
fn estimate_rcond_ldlt<T: Field + Real + bytemuck::Zeroable>(
    ldlt: &Ldlt<T>,
    anorm: T,
    n: usize,
) -> T {
    if anorm <= T::zero() || n == 0 {
        return T::zero();
    }

    // Estimate ||A^(-1)||_1 by solving A*y = e_j for several columns
    let mut ainv_norm_est = T::zero();

    for j in 0..n.min(5) {
        let mut e_j = Mat::zeros(n, 1);
        e_j[(j, 0)] = T::one();

        let y = match ldlt.solve(e_j.as_ref()) {
            Ok(y) => y,
            Err(_) => return T::zero(),
        };

        // Compute ||y||_1
        let mut y_norm = T::zero();
        for i in 0..n {
            y_norm = y_norm + Scalar::abs(y[(i, 0)]);
        }

        if y_norm > ainv_norm_est {
            ainv_norm_est = y_norm;
        }
    }

    // rcond = 1 / (||A||_1 * ||A^(-1)||_1)
    let kappa_est = anorm * ainv_norm_est;

    if kappa_est <= T::zero() {
        T::zero()
    } else {
        T::one() / kappa_est
    }
}

/// Compute forward and backward error bounds for LDL^T solve.
fn compute_error_bounds_ldlt<T: Field + Real + bytemuck::Zeroable>(
    a: &Mat<T>,
    b: &Mat<T>,
    x: &Mat<T>,
    ldlt: &Ldlt<T>,
    n: usize,
    nrhs: usize,
) -> (Vec<T>, Vec<T>) {
    let eps = <T as Scalar>::epsilon();
    let mut forward_error = vec![T::zero(); nrhs];
    let mut backward_error = vec![T::zero(); nrhs];

    for col in 0..nrhs {
        // Compute residual r = b - Ax
        let mut r = Mat::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            r[(i, 0)] = b[(i, col)] - ax_i;
        }

        // Backward error: ||r||_inf / (||A||_inf * ||x||_inf + ||b||_inf)
        let mut r_inf = T::zero();
        let mut x_inf = T::zero();
        let mut b_inf = T::zero();

        for i in 0..n {
            let abs_r = Scalar::abs(r[(i, 0)]);
            let abs_x = Scalar::abs(x[(i, col)]);
            let abs_b = Scalar::abs(b[(i, col)]);
            if abs_r > r_inf {
                r_inf = abs_r;
            }
            if abs_x > x_inf {
                x_inf = abs_x;
            }
            if abs_b > b_inf {
                b_inf = abs_b;
            }
        }

        let a_inf = norm_inf(a.as_ref());
        let denom = a_inf * x_inf + b_inf;

        if denom > T::zero() {
            backward_error[col] = r_inf / denom;
        } else {
            backward_error[col] = T::zero();
        }

        // Forward error estimate using iterative refinement
        let e = match ldlt.solve(r.as_ref()) {
            Ok(e) => e,
            Err(_) => {
                forward_error[col] = T::one();
                continue;
            }
        };

        // ||e||_inf / ||x||_inf
        let mut e_inf = T::zero();
        for i in 0..n {
            let abs_e = Scalar::abs(e[(i, 0)]);
            if abs_e > e_inf {
                e_inf = abs_e;
            }
        }

        if x_inf > T::zero() {
            forward_error[col] = e_inf / x_inf;
        } else {
            forward_error[col] = e_inf;
        }

        // Ensure at least machine epsilon
        if forward_error[col] < eps {
            forward_error[col] = eps;
        }
        if backward_error[col] < eps {
            backward_error[col] = eps;
        }
    }

    (forward_error, backward_error)
}

/// Unscale solution after symmetric equilibration.
fn unscale_solution_symmetric<T: Field + Real + bytemuck::Zeroable>(
    x: &Mat<T>,
    scale: &[T],
) -> Mat<T> {
    let n = x.nrows();
    let nrhs = x.ncols();

    let mut x_unscaled = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x_unscaled[(i, j)] = scale[i] * x[(i, j)];
        }
    }
    x_unscaled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_solve_symmetric_expert_positive_definite() {
        // A = [4 2; 2 5] is SPD
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // Verify Ax = b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!(approx_eq(ax0, b[(0, 0)], 1e-10));
        assert!(approx_eq(ax1, b[(1, 0)], 1e-10));

        // Inertia should be (2, 0, 0) for SPD
        assert_eq!(result.inertia, (2, 0, 0));
    }

    #[test]
    fn test_solve_symmetric_expert_indefinite() {
        // A = [1 2; 2 1] is indefinite (eigenvalues 3 and -1)
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 1.0]]);
        let b = Mat::from_rows(&[&[3.0], &[3.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // Verify Ax = b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!(approx_eq(ax0, b[(0, 0)], 1e-10));
        assert!(approx_eq(ax1, b[(1, 0)], 1e-10));

        // Inertia should be (1, 1, 0) for indefinite
        assert_eq!(result.inertia, (1, 1, 0));
    }

    #[test]
    fn test_solve_symmetric_expert_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // x should equal b for identity matrix
        for i in 0..3 {
            assert!(approx_eq(result.solution[(i, 0)], b[(i, 0)], 1e-10));
        }

        // rcond should be close to 1 for identity
        assert!(approx_eq(result.rcond, 1.0, 0.1));
    }

    #[test]
    fn test_solve_symmetric_expert_with_equilibration() {
        // Matrix with unbalanced diagonal but still well-conditioned after scaling
        let a = Mat::from_rows(&[&[100.0f64, 10.0], &[10.0, 100.0]]);
        let b = Mat::from_rows(&[&[110.0], &[110.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), true).unwrap();

        // Verify Ax = b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!(approx_eq(ax0, b[(0, 0)], 1e-8));
        assert!(approx_eq(ax1, b[(1, 0)], 1e-8));
    }

    #[test]
    fn test_solve_symmetric_expert_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0, 6.0], &[11.0, 9.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // Verify AX = B for both columns
        let x = &result.solution;
        for col in 0..2 {
            let ax0 = a[(0, 0)] * x[(0, col)] + a[(0, 1)] * x[(1, col)];
            let ax1 = a[(1, 0)] * x[(0, col)] + a[(1, 1)] * x[(1, col)];
            assert!(approx_eq(ax0, b[(0, col)], 1e-10));
            assert!(approx_eq(ax1, b[(1, col)], 1e-10));
        }

        // Should have error bounds for both columns
        assert_eq!(result.forward_error.len(), 2);
        assert_eq!(result.backward_error.len(), 2);
    }

    #[test]
    fn test_solve_symmetric_expert_singular() {
        // Singular matrix: [1 2; 2 4]
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            ExpertSymmetricSolveError::SingularMatrix
        );
    }

    #[test]
    fn test_solve_symmetric_expert_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0f32], &[11.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // Verify Ax ≈ b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!((ax0 - b[(0, 0)]).abs() < 1e-5);
        assert!((ax1 - b[(1, 0)]).abs() < 1e-5);
    }

    #[test]
    fn test_solve_symmetric_expert_negative_definite() {
        // Negative definite matrix: -I
        let a = Mat::from_rows(&[&[-1.0f64, 0.0], &[0.0, -1.0]]);
        let b = Mat::from_rows(&[&[-1.0], &[-2.0]]);

        let result = solve_symmetric_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // Solution should be [1, 2]
        assert!(approx_eq(result.solution[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(result.solution[(1, 0)], 2.0, 1e-10));

        // Inertia should be (0, 2, 0) for negative definite
        assert_eq!(result.inertia, (0, 2, 0));
    }
}
