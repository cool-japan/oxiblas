//! Expert Cholesky linear system solvers.
//!
//! Provides enhanced solve routines for symmetric positive definite matrices
//! with equilibration, condition estimation, and error bounds,
//! similar to LAPACK's xPOSVX routines.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::cholesky::{Cholesky, CholeskyError};
use crate::utils::{norm_1, norm_inf};

/// Error type for expert Cholesky solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCholeskySolveError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix and vector dimensions don't match.
    DimensionMismatch,
    /// Matrix is not positive definite.
    NotPositiveDefinite,
    /// Matrix is singular to working precision.
    SingularToWorkingPrecision,
}

impl core::fmt::Display for ExpertCholeskySolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::DimensionMismatch => write!(f, "Matrix and vector dimensions do not match"),
            Self::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            Self::SingularToWorkingPrecision => {
                write!(f, "Matrix is singular to working precision")
            }
        }
    }
}

impl std::error::Error for ExpertCholeskySolveError {}

impl From<CholeskyError> for ExpertCholeskySolveError {
    fn from(e: CholeskyError) -> Self {
        match e {
            CholeskyError::NotPositiveDefinite { .. } => Self::NotPositiveDefinite,
            CholeskyError::NotSquare { .. } => Self::NotSquare,
            CholeskyError::DimensionMismatch { .. } => Self::DimensionMismatch,
        }
    }
}

/// Result of expert Cholesky solve operation.
#[derive(Debug, Clone)]
pub struct ExpertCholeskySolveResult<T: Scalar> {
    /// Solution matrix X.
    pub solution: Mat<T>,
    /// Reciprocal condition number estimate (1/kappa).
    pub rcond: T,
    /// Forward error bound for each right-hand side.
    pub forward_error: Vec<T>,
    /// Backward error bound for each right-hand side.
    pub backward_error: Vec<T>,
    /// Scaling factors (if equilibration was applied).
    /// For SPD matrices, we use symmetric scaling: D A D where D is diagonal.
    pub scale: Option<Vec<T>>,
    /// Whether equilibration was applied.
    pub equilibrated: bool,
}

/// Solves a symmetric positive definite linear system Ax = b with expert options.
///
/// This routine provides:
/// - Optional equilibration (symmetric scaling) to improve condition
/// - Condition number estimation
/// - Forward and backward error bounds
///
/// # Arguments
///
/// * `a` - Symmetric positive definite coefficient matrix A (n×n)
/// * `b` - Right-hand side matrix B (n×nrhs)
/// * `equilibrate` - Whether to apply symmetric equilibration
///
/// # Returns
///
/// `ExpertCholeskySolveResult` containing the solution and diagnostic information.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::expert_cholesky::solve_cholesky_expert;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0],
///     &[2.0, 5.0],
/// ]);
/// let b = Mat::from_rows(&[&[8.0], &[11.0]]);
///
/// let result = solve_cholesky_expert(a.as_ref(), b.as_ref(), false).unwrap();
/// println!("Solution: {:?}", result.solution);
/// println!("Reciprocal condition: {}", result.rcond);
/// ```
pub fn solve_cholesky_expert<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    equilibrate: bool,
) -> Result<ExpertCholeskySolveResult<T>, ExpertCholeskySolveError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n {
        return Err(ExpertCholeskySolveError::NotSquare);
    }
    if b.nrows() != n {
        return Err(ExpertCholeskySolveError::DimensionMismatch);
    }

    // Handle empty case
    if n == 0 {
        return Ok(ExpertCholeskySolveResult {
            solution: Mat::zeros(0, nrhs),
            rcond: T::one(),
            forward_error: vec![],
            backward_error: vec![],
            scale: None,
            equilibrated: false,
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

    // Compute Cholesky decomposition
    let chol = Cholesky::compute(a_scaled.as_ref())?;

    // Solve the system
    let x_scaled = chol.solve(b_scaled.as_ref())?;

    // Estimate reciprocal condition number
    let rcond = estimate_rcond_cholesky(&chol, anorm, n);

    // Check if matrix is singular to working precision
    let eps = <T as Scalar>::epsilon();
    if rcond < eps {
        return Err(ExpertCholeskySolveError::SingularToWorkingPrecision);
    }

    // Compute error bounds
    let (forward_error, backward_error) =
        compute_error_bounds_cholesky(&a_scaled, &b_scaled, &x_scaled, &chol, n, nrhs);

    // Unscale solution if equilibration was applied
    let solution = if let Some(ref s) = scale {
        unscale_solution_symmetric(&x_scaled, s)
    } else {
        x_scaled
    };

    Ok(ExpertCholeskySolveResult {
        solution,
        rcond,
        forward_error,
        backward_error,
        scale,
        equilibrated: did_equilibrate,
    })
}

/// Apply symmetric equilibration to matrix A and right-hand side b.
///
/// For SPD matrices, we use D A D where D = diag(1/sqrt(a_ii)).
/// This makes the diagonal of the scaled matrix equal to 1.
fn apply_symmetric_equilibration<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> (Mat<T>, Mat<T>, Option<Vec<T>>, bool) {
    let n = a.nrows();
    let nrhs = b.ncols();

    // Compute scaling: s[i] = 1 / sqrt(a[i,i])
    let mut scale = vec![T::one(); n];
    let mut needs_scaling = false;

    for i in 0..n {
        let diag = a[(i, i)];
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

/// Estimate reciprocal condition number for Cholesky factorization.
///
/// Uses the relationship kappa(A) = kappa(L)^2 for A = LL^T.
fn estimate_rcond_cholesky<T: Field + Real + bytemuck::Zeroable>(
    chol: &Cholesky<T>,
    anorm: T,
    n: usize,
) -> T {
    if anorm <= T::zero() || n == 0 {
        return T::zero();
    }

    // For A = LL^T, we estimate ||A^(-1)||_1 by solving A*y = e_j for the
    // column that gives the largest norm.

    let mut ainv_norm_est = T::zero();

    // Try a few columns to estimate ||A^(-1)||_1
    for j in 0..n.min(5) {
        let mut e_j = Mat::zeros(n, 1);
        e_j[(j, 0)] = T::one();

        let y = match chol.solve(e_j.as_ref()) {
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

/// Compute forward and backward error bounds for Cholesky solve.
fn compute_error_bounds_cholesky<T: Field + Real + bytemuck::Zeroable>(
    a: &Mat<T>,
    b: &Mat<T>,
    x: &Mat<T>,
    chol: &Cholesky<T>,
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
        // Solve A*e = r to get error estimate
        let e = match chol.solve(r.as_ref()) {
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
///
/// If x' = D^(-1) A^(-1) D^(-1) (D b) = D^(-1) A^(-1) b,
/// then x = D x' (since A' = D A D, solution to A' x' = D b gives x = D x').
fn unscale_solution_symmetric<T: Field + Real + bytemuck::Zeroable>(
    x: &Mat<T>,
    scale: &[T],
) -> Mat<T> {
    let n = x.nrows();
    let nrhs = x.ncols();

    let mut x_unscaled = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            // x_true = D * x_scaled
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
    fn test_solve_cholesky_expert_simple() {
        // A = [4 2; 2 5] is SPD
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let result = solve_cholesky_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // Verify Ax = b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!(approx_eq(ax0, b[(0, 0)], 1e-10));
        assert!(approx_eq(ax1, b[(1, 0)], 1e-10));

        // rcond should be positive for well-conditioned matrix
        assert!(result.rcond > 0.0);
        assert!(result.rcond <= 1.0);
    }

    #[test]
    fn test_solve_cholesky_expert_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let result = solve_cholesky_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // x should equal b for identity matrix
        for i in 0..3 {
            assert!(approx_eq(result.solution[(i, 0)], b[(i, 0)], 1e-10));
        }

        // rcond should be close to 1 for identity
        assert!(approx_eq(result.rcond, 1.0, 0.1));
    }

    #[test]
    fn test_solve_cholesky_expert_with_equilibration() {
        // Matrix with unbalanced diagonal
        let a = Mat::from_rows(&[&[1000.0f64, 1.0], &[1.0, 0.01]]);
        let b = Mat::from_rows(&[&[1001.0], &[1.01]]);

        let result = solve_cholesky_expert(a.as_ref(), b.as_ref(), true).unwrap();

        // Verify Ax = b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!(approx_eq(ax0, b[(0, 0)], 1e-4));
        assert!(approx_eq(ax1, b[(1, 0)], 1e-4));

        // Equilibration should have been applied
        assert!(result.equilibrated);
        assert!(result.scale.is_some());
    }

    #[test]
    fn test_solve_cholesky_expert_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0, 6.0], &[11.0, 9.0]]);

        let result = solve_cholesky_expert(a.as_ref(), b.as_ref(), false).unwrap();

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
    fn test_solve_cholesky_expert_not_positive_definite() {
        // A = [1 2; 2 1] is not positive definite
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0]]);

        let result = solve_cholesky_expert(a.as_ref(), b.as_ref(), false);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            ExpertCholeskySolveError::NotPositiveDefinite
        );
    }

    #[test]
    fn test_solve_cholesky_expert_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0f32], &[11.0]]);

        let result = solve_cholesky_expert(a.as_ref(), b.as_ref(), false).unwrap();

        // Verify Ax ≈ b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!((ax0 - b[(0, 0)]).abs() < 1e-5);
        assert!((ax1 - b[(1, 0)]).abs() < 1e-5);
    }
}
