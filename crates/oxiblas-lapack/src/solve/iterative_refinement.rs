//! Iterative refinement for linear systems.
//!
//! Provides routines to improve the accuracy of computed solutions using
//! iterative refinement, similar to LAPACK's xGERFS routines.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::lu::Lu;
use crate::utils::norm_inf;

/// Error type for iterative refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementError {
    /// Matrix dimensions don't match.
    DimensionMismatch,
    /// LU factorization failed.
    FactorizationFailed,
    /// Refinement did not converge.
    NotConverged,
}

impl core::fmt::Display for RefinementError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DimensionMismatch => write!(f, "Matrix dimensions do not match"),
            Self::FactorizationFailed => write!(f, "LU factorization failed"),
            Self::NotConverged => write!(f, "Iterative refinement did not converge"),
        }
    }
}

impl std::error::Error for RefinementError {}

/// Result of iterative refinement.
#[derive(Debug, Clone)]
pub struct RefinementResult<T: Scalar> {
    /// Refined solution matrix X.
    pub solution: Mat<T>,
    /// Forward error bound for each right-hand side.
    /// Estimated relative forward error: ||x_true - x||_inf / ||x||_inf
    pub forward_error: Vec<T>,
    /// Backward error for each right-hand side.
    /// ||b - A*x||_inf / (||A||_inf * ||x||_inf + ||b||_inf)
    pub backward_error: Vec<T>,
    /// Number of refinement iterations performed for each RHS.
    pub iterations: Vec<usize>,
}

/// Maximum number of refinement iterations.
const MAX_ITERATIONS: usize = 5;

/// Convergence tolerance factor (relative to machine epsilon).
const CONVERGENCE_FACTOR: f64 = 10.0;

/// Refine a computed solution to a general linear system Ax = b.
///
/// Given the LU factorization of A and an initial solution x, this routine
/// performs iterative refinement to improve the accuracy of x.
///
/// # Arguments
///
/// * `a` - Original coefficient matrix A (n x n)
/// * `lu` - LU factorization of A
/// * `b` - Right-hand side matrix B (n x nrhs)
/// * `x` - Initial solution to refine (n x nrhs), modified in place
///
/// # Returns
///
/// `RefinementResult` containing the refined solution and error bounds.
///
/// # Algorithm
///
/// For each right-hand side:
/// 1. Compute residual r = b - A*x
/// 2. Solve A*d = r using LU factorization
/// 3. Update x = x + d
/// 4. Repeat until convergence or max iterations
///
/// # Example
///
/// ```
/// use oxiblas_lapack::lu::Lu;
/// use oxiblas_lapack::solve::iterative_refinement::refine_solution;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0],
///     &[2.0, 5.0],
/// ]);
/// let b = Mat::from_rows(&[&[8.0], &[11.0]]);
///
/// // Compute LU factorization and initial solution
/// let lu = Lu::compute(a.as_ref()).unwrap();
/// let mut x = lu.solve(b.as_ref()).unwrap();
///
/// // Refine the solution
/// let result = refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).unwrap();
/// println!("Forward error: {:?}", result.forward_error);
/// println!("Backward error: {:?}", result.backward_error);
/// ```
pub fn refine_solution<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    lu: &Lu<T>,
    b: MatRef<'_, T>,
    x: &mut Mat<T>,
) -> Result<RefinementResult<T>, RefinementError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n || b.nrows() != n || x.nrows() != n || x.ncols() != nrhs {
        return Err(RefinementError::DimensionMismatch);
    }

    if n == 0 || nrhs == 0 {
        return Ok(RefinementResult {
            solution: x.clone(),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
        });
    }

    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(CONVERGENCE_FACTOR).unwrap_or(eps);
    let a_inf = norm_inf(a);

    let mut forward_error = vec![T::zero(); nrhs];
    let mut backward_error = vec![T::zero(); nrhs];
    let mut iterations = vec![0usize; nrhs];

    for col in 0..nrhs {
        let mut prev_residual_norm = T::zero();

        for iter in 0..MAX_ITERATIONS {
            let mut r = Mat::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = T::zero();
                for j in 0..n {
                    ax_i = ax_i + a[(i, j)] * x[(j, col)];
                }
                r[(i, 0)] = b[(i, col)] - ax_i;
            }

            let mut r_inf = T::zero();
            for i in 0..n {
                let abs_r = Scalar::abs(r[(i, 0)]);
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            if iter > 0 && (r_inf <= tol * prev_residual_norm || r_inf <= eps) {
                iterations[col] = iter;
                break;
            }

            if r_inf <= eps {
                iterations[col] = 0;
                break;
            }

            let d = match lu.solve(r.as_ref()) {
                Ok(d) => d,
                Err(_) => break,
            };

            for i in 0..n {
                x[(i, col)] = x[(i, col)] + d[(i, 0)];
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }

        // Error bounds
        let mut r_inf = T::zero();
        let mut x_inf = T::zero();
        let mut b_inf = T::zero();

        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            let abs_r = Scalar::abs(b[(i, col)] - ax_i);
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

        let denom = a_inf * x_inf + b_inf;
        if denom > T::zero() {
            backward_error[col] = r_inf / denom;
        } else {
            backward_error[col] = T::zero();
        }

        let mut r_final = Mat::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = b[(i, col)] - ax_i;
        }

        let e = match lu.solve(r_final.as_ref()) {
            Ok(e) => e,
            Err(_) => {
                forward_error[col] = T::one();
                continue;
            }
        };

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

        if forward_error[col] < eps {
            forward_error[col] = eps;
        }
        if backward_error[col] < eps {
            backward_error[col] = eps;
        }
    }

    Ok(RefinementResult {
        solution: x.clone(),
        forward_error,
        backward_error,
        iterations,
    })
}

/// Refine a computed solution to a symmetric system Ax = b.
///
/// Uses LDL^T factorization for iterative refinement of symmetric systems.
///
/// # Arguments
///
/// * `a` - Original symmetric coefficient matrix A (n x n)
/// * `ldlt` - LDL^T factorization of A
/// * `b` - Right-hand side matrix B (n x nrhs)
/// * `x` - Initial solution to refine (n x nrhs), modified in place
pub fn refine_solution_symmetric<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    ldlt: &crate::cholesky::Ldlt<T>,
    b: MatRef<'_, T>,
    x: &mut Mat<T>,
) -> Result<RefinementResult<T>, RefinementError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n || b.nrows() != n || x.nrows() != n || x.ncols() != nrhs {
        return Err(RefinementError::DimensionMismatch);
    }

    if n == 0 || nrhs == 0 {
        return Ok(RefinementResult {
            solution: x.clone(),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
        });
    }

    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(CONVERGENCE_FACTOR).unwrap_or(eps);
    let a_inf = norm_inf(a);

    let mut forward_error = vec![T::zero(); nrhs];
    let mut backward_error = vec![T::zero(); nrhs];
    let mut iterations = vec![0usize; nrhs];

    for col in 0..nrhs {
        let mut prev_residual_norm = T::zero();

        for iter in 0..MAX_ITERATIONS {
            let mut r = Mat::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = T::zero();
                for j in 0..n {
                    ax_i = ax_i + a[(i, j)] * x[(j, col)];
                }
                r[(i, 0)] = b[(i, col)] - ax_i;
            }

            let mut r_inf = T::zero();
            for i in 0..n {
                let abs_r = Scalar::abs(r[(i, 0)]);
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            if iter > 0 && (r_inf <= tol * prev_residual_norm || r_inf <= eps) {
                iterations[col] = iter;
                break;
            }

            if r_inf <= eps {
                iterations[col] = 0;
                break;
            }

            let d = match ldlt.solve(r.as_ref()) {
                Ok(d) => d,
                Err(_) => break,
            };

            for i in 0..n {
                x[(i, col)] = x[(i, col)] + d[(i, 0)];
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }

        let mut r_inf = T::zero();
        let mut x_inf = T::zero();
        let mut b_inf = T::zero();

        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            let abs_r = Scalar::abs(b[(i, col)] - ax_i);
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

        let denom = a_inf * x_inf + b_inf;
        if denom > T::zero() {
            backward_error[col] = r_inf / denom;
        }

        let mut r_final = Mat::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = b[(i, col)] - ax_i;
        }

        let e = match ldlt.solve(r_final.as_ref()) {
            Ok(e) => e,
            Err(_) => {
                forward_error[col] = T::one();
                continue;
            }
        };

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

        if forward_error[col] < eps {
            forward_error[col] = eps;
        }
        if backward_error[col] < eps {
            backward_error[col] = eps;
        }
    }

    Ok(RefinementResult {
        solution: x.clone(),
        forward_error,
        backward_error,
        iterations,
    })
}

/// Refine a computed solution to an SPD system Ax = b.
///
/// Uses Cholesky factorization for iterative refinement of SPD systems.
pub fn refine_solution_cholesky<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    chol: &crate::cholesky::Cholesky<T>,
    b: MatRef<'_, T>,
    x: &mut Mat<T>,
) -> Result<RefinementResult<T>, RefinementError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n || b.nrows() != n || x.nrows() != n || x.ncols() != nrhs {
        return Err(RefinementError::DimensionMismatch);
    }

    if n == 0 || nrhs == 0 {
        return Ok(RefinementResult {
            solution: x.clone(),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
        });
    }

    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(CONVERGENCE_FACTOR).unwrap_or(eps);
    let a_inf = norm_inf(a);

    let mut forward_error = vec![T::zero(); nrhs];
    let mut backward_error = vec![T::zero(); nrhs];
    let mut iterations = vec![0usize; nrhs];

    for col in 0..nrhs {
        let mut prev_residual_norm = T::zero();

        for iter in 0..MAX_ITERATIONS {
            let mut r = Mat::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = T::zero();
                for j in 0..n {
                    ax_i = ax_i + a[(i, j)] * x[(j, col)];
                }
                r[(i, 0)] = b[(i, col)] - ax_i;
            }

            let mut r_inf = T::zero();
            for i in 0..n {
                let abs_r = Scalar::abs(r[(i, 0)]);
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            if iter > 0 && (r_inf <= tol * prev_residual_norm || r_inf <= eps) {
                iterations[col] = iter;
                break;
            }

            if r_inf <= eps {
                iterations[col] = 0;
                break;
            }

            let d = match chol.solve(r.as_ref()) {
                Ok(d) => d,
                Err(_) => break,
            };

            for i in 0..n {
                x[(i, col)] = x[(i, col)] + d[(i, 0)];
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }

        let mut r_inf = T::zero();
        let mut x_inf = T::zero();
        let mut b_inf = T::zero();

        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            let abs_r = Scalar::abs(b[(i, col)] - ax_i);
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

        let denom = a_inf * x_inf + b_inf;
        if denom > T::zero() {
            backward_error[col] = r_inf / denom;
        }

        let mut r_final = Mat::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = b[(i, col)] - ax_i;
        }

        let e = match chol.solve(r_final.as_ref()) {
            Ok(e) => e,
            Err(_) => {
                forward_error[col] = T::one();
                continue;
            }
        };

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

        if forward_error[col] < eps {
            forward_error[col] = eps;
        }
        if backward_error[col] < eps {
            backward_error[col] = eps;
        }
    }

    Ok(RefinementResult {
        solution: x.clone(),
        forward_error,
        backward_error,
        iterations,
    })
}

// =============================================================================
// Mixed-precision wrappers (backward compatibility)
// =============================================================================
//
// The full implementations live in `super::mixed_precision`. These thin
// wrappers preserve the original API surface of this module.

pub use super::mixed_precision::MixedPrecisionResult;

/// Solve Ax = b using mixed-precision LU iterative refinement.
///
/// Delegates to [`super::mixed_precision::mixed_precision_solve`].
pub fn mixed_precision_solve(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    super::mixed_precision::mixed_precision_solve(a, b)
}

/// Solve SPD system Ax = b using mixed-precision Cholesky refinement.
///
/// Delegates to [`super::mixed_precision::mixed_precision_solve_cholesky`].
pub fn mixed_precision_solve_cholesky(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    super::mixed_precision::mixed_precision_solve_cholesky(a, b)
}

/// Solve symmetric system Ax = b using mixed-precision LDL^T refinement.
///
/// Delegates to [`super::mixed_precision::mixed_precision_solve_symmetric`].
pub fn mixed_precision_solve_symmetric(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    super::mixed_precision::mixed_precision_solve_symmetric(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_refine_solution_simple() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let lu = Lu::compute(a.as_ref()).expect("lu should succeed");
        let mut x = lu.solve(b.as_ref()).expect("solve should succeed");

        let result =
            refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).expect("refine should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-10));
        assert!(approx_eq(ax1, 11.0, 1e-10));

        assert!(result.forward_error[0] < 1e-10);
        assert!(result.backward_error[0] < 1e-10);
    }

    #[test]
    fn test_refine_solution_ill_conditioned() {
        let a = Mat::from_rows(&[
            &[1.0f64, 1.0 / 2.0, 1.0 / 3.0],
            &[1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0],
            &[1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
        ]);
        let b = Mat::from_rows(&[&[1.0], &[1.0], &[1.0]]);

        let lu = Lu::compute(a.as_ref()).expect("lu should succeed");
        let mut x = lu.solve(b.as_ref()).expect("solve should succeed");

        let result =
            refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).expect("refine should succeed");

        assert_eq!(result.forward_error.len(), 1);
        assert_eq!(result.backward_error.len(), 1);
    }

    #[test]
    fn test_refine_solution_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0, 6.0], &[11.0, 9.0]]);

        let lu = Lu::compute(a.as_ref()).expect("lu should succeed");
        let mut x = lu.solve(b.as_ref()).expect("solve should succeed");

        let result =
            refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).expect("refine should succeed");

        assert_eq!(result.forward_error.len(), 2);
        assert_eq!(result.backward_error.len(), 2);
        assert_eq!(result.iterations.len(), 2);

        for col in 0..2 {
            let ax0 = 4.0 * result.solution[(0, col)] + 2.0 * result.solution[(1, col)];
            let ax1 = 2.0 * result.solution[(0, col)] + 5.0 * result.solution[(1, col)];
            assert!(approx_eq(ax0, b[(0, col)], 1e-10));
            assert!(approx_eq(ax1, b[(1, col)], 1e-10));
        }
    }

    #[test]
    fn test_refine_solution_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let lu = Lu::compute(a.as_ref()).expect("lu should succeed");
        let mut x = lu.solve(b.as_ref()).expect("solve should succeed");

        let result =
            refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).expect("refine should succeed");

        for i in 0..3 {
            assert!(approx_eq(result.solution[(i, 0)], b[(i, 0)], 1e-14));
        }
    }

    #[test]
    fn test_refine_solution_symmetric() {
        use crate::cholesky::Ldlt;

        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("ldlt should succeed");
        let mut x = ldlt.solve(b.as_ref()).expect("solve should succeed");

        let result = refine_solution_symmetric(a.as_ref(), &ldlt, b.as_ref(), &mut x)
            .expect("refine should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-10));
        assert!(approx_eq(ax1, 11.0, 1e-10));
    }

    #[test]
    fn test_refine_solution_cholesky() {
        use crate::cholesky::Cholesky;

        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let chol = Cholesky::compute(a.as_ref()).expect("cholesky should succeed");
        let mut x = chol.solve(b.as_ref()).expect("solve should succeed");

        let result = refine_solution_cholesky(a.as_ref(), &chol, b.as_ref(), &mut x)
            .expect("refine should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-10));
        assert!(approx_eq(ax1, 11.0, 1e-10));
    }

    #[test]
    fn test_refine_solution_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0f32], &[11.0]]);

        let lu = Lu::compute(a.as_ref()).expect("lu should succeed");
        let mut x = lu.solve(b.as_ref()).expect("solve should succeed");

        let result =
            refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).expect("refine should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-5);
        assert!((ax1 - 11.0).abs() < 1e-5);
    }
}
