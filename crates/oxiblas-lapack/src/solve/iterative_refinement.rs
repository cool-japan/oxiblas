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
/// * `a` - Original coefficient matrix A (n×n)
/// * `lu` - LU factorization of A
/// * `b` - Right-hand side matrix B (n×nrhs)
/// * `x` - Initial solution to refine (n×nrhs), modified in place
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

    // Handle empty case
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

    // Compute ||A||_inf for error bounds
    let a_inf = norm_inf(a);

    let mut forward_error = vec![T::zero(); nrhs];
    let mut backward_error = vec![T::zero(); nrhs];
    let mut iterations = vec![0usize; nrhs];

    // Process each right-hand side
    for col in 0..nrhs {
        let mut prev_residual_norm = T::zero();

        for iter in 0..MAX_ITERATIONS {
            // Compute residual r = b - A*x for this column
            let mut r = Mat::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = T::zero();
                for j in 0..n {
                    ax_i = ax_i + a[(i, j)] * x[(j, col)];
                }
                r[(i, 0)] = b[(i, col)] - ax_i;
            }

            // Compute residual norm
            let mut r_inf = T::zero();
            for i in 0..n {
                let abs_r = Scalar::abs(r[(i, 0)]);
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            // Check for convergence (but always do at least one iteration)
            if iter > 0 && (r_inf <= tol * prev_residual_norm || r_inf <= eps) {
                iterations[col] = iter;
                break;
            }

            // Already converged on first check
            if r_inf <= eps {
                iterations[col] = 0;
                break;
            }

            // Solve A*d = r
            let d = match lu.solve(r.as_ref()) {
                Ok(d) => d,
                Err(_) => break,
            };

            // Update x = x + d
            for i in 0..n {
                x[(i, col)] = x[(i, col)] + d[(i, 0)];
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }

        // Compute final error bounds
        // Backward error: ||b - A*x||_inf / (||A||_inf * ||x||_inf + ||b||_inf)
        let mut r_inf = T::zero();
        let mut x_inf = T::zero();
        let mut b_inf = T::zero();

        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            let residual = b[(i, col)] - ax_i;
            let abs_r = Scalar::abs(residual);
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

        // Forward error estimate
        // Solve A*e = r for error estimate
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

        // Ensure at least machine epsilon
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
/// * `a` - Original symmetric coefficient matrix A (n×n)
/// * `ldlt` - LDL^T factorization of A
/// * `b` - Right-hand side matrix B (n×nrhs)
/// * `x` - Initial solution to refine (n×nrhs), modified in place
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

            // Check for convergence (but always do at least one iteration)
            if iter > 0 && (r_inf <= tol * prev_residual_norm || r_inf <= eps) {
                iterations[col] = iter;
                break;
            }

            // Already converged on first check
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

        // Compute error bounds
        let mut r_inf = T::zero();
        let mut x_inf = T::zero();
        let mut b_inf = T::zero();

        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            let residual = b[(i, col)] - ax_i;
            let abs_r = Scalar::abs(residual);
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

            // Check for convergence (but always do at least one iteration)
            if iter > 0 && (r_inf <= tol * prev_residual_norm || r_inf <= eps) {
                iterations[col] = iter;
                break;
            }

            // Already converged on first check
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

        // Compute error bounds
        let mut r_inf = T::zero();
        let mut x_inf = T::zero();
        let mut b_inf = T::zero();

        for i in 0..n {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * x[(j, col)];
            }
            let residual = b[(i, col)] - ax_i;
            let abs_r = Scalar::abs(residual);
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
// Mixed Precision Iterative Refinement
// =============================================================================
//
// Mixed precision algorithms use lower precision (f32) for factorization
// while computing residuals and accumulating corrections in higher precision (f64).
// This achieves nearly f64 accuracy at lower computational cost for well-conditioned
// systems.

/// Mixed precision refinement result.
#[derive(Debug, Clone)]
pub struct MixedPrecisionResult {
    /// Refined solution in f64.
    pub solution: Mat<f64>,
    /// Forward error bound for each right-hand side.
    pub forward_error: Vec<f64>,
    /// Backward error for each right-hand side.
    pub backward_error: Vec<f64>,
    /// Number of refinement iterations performed for each RHS.
    pub iterations: Vec<usize>,
    /// Whether f64 accuracy was achieved.
    pub achieved_double_precision: bool,
}

/// Maximum iterations for mixed precision refinement.
const MIXED_MAX_ITERATIONS: usize = 30;

/// Solve Ax = b using mixed precision iterative refinement.
///
/// Uses f32 LU factorization with f64 residual computation to achieve
/// nearly f64 accuracy at reduced computational cost.
///
/// # Algorithm
///
/// 1. Factor A in f32 (working precision)
/// 2. Compute initial solution x₀ in f32, promote to f64
/// 3. For each iteration:
///    a. Compute residual r = b - Ax in f64
///    b. Convert r to f32, solve A*d = r using f32 factorization
///    c. Convert d to f64, update x = x + d
///    d. Check for convergence to f64 accuracy
///
/// # Arguments
///
/// * `a` - Coefficient matrix A (n×n) in f64
/// * `b` - Right-hand side matrix B (n×nrhs) in f64
///
/// # Returns
///
/// `MixedPrecisionResult` with solution in f64 and error bounds.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::iterative_refinement::mixed_precision_solve;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0],
///     &[2.0, 5.0],
/// ]);
/// let b = Mat::from_rows(&[&[8.0], &[11.0]]);
///
/// let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();
/// println!("Achieved double precision: {}", result.achieved_double_precision);
/// ```
pub fn mixed_precision_solve(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n || b.nrows() != n {
        return Err(RefinementError::DimensionMismatch);
    }

    // Handle empty case
    if n == 0 || nrhs == 0 {
        return Ok(MixedPrecisionResult {
            solution: Mat::zeros(n, nrhs),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
            achieved_double_precision: true,
        });
    }

    // Convert A to f32 for factorization
    let mut a_f32 = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_f32[(i, j)] = a[(i, j)] as f32;
        }
    }

    // LU factorization in f32
    let lu_f32 = match Lu::compute(a_f32.as_ref()) {
        Ok(lu) => lu,
        Err(_) => return Err(RefinementError::FactorizationFailed),
    };

    // Convert b to f32 for initial solve
    let mut b_f32 = Mat::<f32>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            b_f32[(i, j)] = b[(i, j)] as f32;
        }
    }

    // Initial solution in f32
    let x_f32 = match lu_f32.solve(b_f32.as_ref()) {
        Ok(x) => x,
        Err(_) => return Err(RefinementError::FactorizationFailed),
    };

    // Promote initial solution to f64
    let mut x = Mat::<f64>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x[(i, j)] = x_f32[(i, j)] as f64;
        }
    }

    // Machine epsilon for f64
    let eps_f64 = f64::EPSILON;
    let tol = eps_f64 * CONVERGENCE_FACTOR;

    // Compute ||A||_inf for error bounds
    let mut a_inf = 0.0f64;
    for i in 0..n {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            row_sum += a[(i, j)].abs();
        }
        if row_sum > a_inf {
            a_inf = row_sum;
        }
    }

    let mut forward_error = vec![0.0f64; nrhs];
    let mut backward_error = vec![0.0f64; nrhs];
    let mut iterations = vec![0usize; nrhs];
    let mut achieved_double = true;

    // Process each right-hand side
    for col in 0..nrhs {
        let mut prev_residual_norm = f64::MAX;

        for iter in 0..MIXED_MAX_ITERATIONS {
            // Compute residual r = b - A*x in f64 (high precision)
            let mut r_f64 = Mat::<f64>::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = 0.0f64;
                for j in 0..n {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                r_f64[(i, 0)] = b[(i, col)] - ax_i;
            }

            // Compute residual norm
            let mut r_inf = 0.0f64;
            for i in 0..n {
                let abs_r = r_f64[(i, 0)].abs();
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            // Check for convergence to f64 accuracy
            if r_inf <= tol || (iter > 0 && r_inf >= prev_residual_norm) {
                iterations[col] = iter;
                if r_inf > eps_f64 * 100.0 {
                    achieved_double = false;
                }
                break;
            }

            // Convert residual to f32 for correction solve
            let mut r_f32 = Mat::<f32>::zeros(n, 1);
            for i in 0..n {
                r_f32[(i, 0)] = r_f64[(i, 0)] as f32;
            }

            // Solve A*d = r in f32
            let d_f32 = match lu_f32.solve(r_f32.as_ref()) {
                Ok(d) => d,
                Err(_) => {
                    achieved_double = false;
                    break;
                }
            };

            // Update x = x + d in f64
            for i in 0..n {
                x[(i, col)] += d_f32[(i, 0)] as f64;
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }

        // Compute final error bounds in f64
        let mut r_inf = 0.0f64;
        let mut x_inf = 0.0f64;
        let mut b_inf = 0.0f64;

        for i in 0..n {
            let mut ax_i = 0.0f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            let residual = (b[(i, col)] - ax_i).abs();
            let abs_x = x[(i, col)].abs();
            let abs_b = b[(i, col)].abs();

            if residual > r_inf {
                r_inf = residual;
            }
            if abs_x > x_inf {
                x_inf = abs_x;
            }
            if abs_b > b_inf {
                b_inf = abs_b;
            }
        }

        // Backward error
        let denom = a_inf * x_inf + b_inf;
        if denom > 0.0 {
            backward_error[col] = r_inf / denom;
        } else {
            backward_error[col] = 0.0;
        }

        // Forward error estimate
        let mut r_final = Mat::<f32>::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = 0.0f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = (b[(i, col)] - ax_i) as f32;
        }

        if let Ok(e_f32) = lu_f32.solve(r_final.as_ref()) {
            let mut e_inf = 0.0f64;
            for i in 0..n {
                let abs_e = (e_f32[(i, 0)] as f64).abs();
                if abs_e > e_inf {
                    e_inf = abs_e;
                }
            }

            if x_inf > 0.0 {
                forward_error[col] = e_inf / x_inf;
            } else {
                forward_error[col] = e_inf;
            }
        } else {
            forward_error[col] = 1.0;
        }

        // Ensure at least machine epsilon
        if forward_error[col] < eps_f64 {
            forward_error[col] = eps_f64;
        }
        if backward_error[col] < eps_f64 {
            backward_error[col] = eps_f64;
        }
    }

    Ok(MixedPrecisionResult {
        solution: x,
        forward_error,
        backward_error,
        iterations,
        achieved_double_precision: achieved_double,
    })
}

/// Solve SPD system Ax = b using mixed precision Cholesky refinement.
///
/// Uses f32 Cholesky factorization with f64 residual computation.
///
/// # Arguments
///
/// * `a` - SPD coefficient matrix A (n×n) in f64
/// * `b` - Right-hand side matrix B (n×nrhs) in f64
pub fn mixed_precision_solve_cholesky(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n || b.nrows() != n {
        return Err(RefinementError::DimensionMismatch);
    }

    if n == 0 || nrhs == 0 {
        return Ok(MixedPrecisionResult {
            solution: Mat::zeros(n, nrhs),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
            achieved_double_precision: true,
        });
    }

    // Convert A to f32 for factorization
    let mut a_f32 = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_f32[(i, j)] = a[(i, j)] as f32;
        }
    }

    // Cholesky factorization in f32
    let chol_f32 = match crate::cholesky::Cholesky::compute(a_f32.as_ref()) {
        Ok(chol) => chol,
        Err(_) => return Err(RefinementError::FactorizationFailed),
    };

    // Convert b to f32 for initial solve
    let mut b_f32 = Mat::<f32>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            b_f32[(i, j)] = b[(i, j)] as f32;
        }
    }

    // Initial solution in f32
    let x_f32 = match chol_f32.solve(b_f32.as_ref()) {
        Ok(x) => x,
        Err(_) => return Err(RefinementError::FactorizationFailed),
    };

    // Promote initial solution to f64
    let mut x = Mat::<f64>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x[(i, j)] = x_f32[(i, j)] as f64;
        }
    }

    let eps_f64 = f64::EPSILON;
    let tol = eps_f64 * CONVERGENCE_FACTOR;

    // Compute ||A||_inf
    let mut a_inf = 0.0f64;
    for i in 0..n {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            row_sum += a[(i, j)].abs();
        }
        if row_sum > a_inf {
            a_inf = row_sum;
        }
    }

    let mut forward_error = vec![0.0f64; nrhs];
    let mut backward_error = vec![0.0f64; nrhs];
    let mut iterations = vec![0usize; nrhs];
    let mut achieved_double = true;

    for col in 0..nrhs {
        let mut prev_residual_norm = f64::MAX;

        for iter in 0..MIXED_MAX_ITERATIONS {
            // Compute residual r = b - A*x in f64
            let mut r_f64 = Mat::<f64>::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = 0.0f64;
                for j in 0..n {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                r_f64[(i, 0)] = b[(i, col)] - ax_i;
            }

            let mut r_inf = 0.0f64;
            for i in 0..n {
                let abs_r = r_f64[(i, 0)].abs();
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            if r_inf <= tol || (iter > 0 && r_inf >= prev_residual_norm) {
                iterations[col] = iter;
                if r_inf > eps_f64 * 100.0 {
                    achieved_double = false;
                }
                break;
            }

            // Convert residual to f32
            let mut r_f32 = Mat::<f32>::zeros(n, 1);
            for i in 0..n {
                r_f32[(i, 0)] = r_f64[(i, 0)] as f32;
            }

            // Solve A*d = r in f32
            let d_f32 = match chol_f32.solve(r_f32.as_ref()) {
                Ok(d) => d,
                Err(_) => {
                    achieved_double = false;
                    break;
                }
            };

            // Update x = x + d in f64
            for i in 0..n {
                x[(i, col)] += d_f32[(i, 0)] as f64;
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }

        // Compute final error bounds
        let mut r_inf = 0.0f64;
        let mut x_inf = 0.0f64;
        let mut b_inf = 0.0f64;

        for i in 0..n {
            let mut ax_i = 0.0f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            let residual = (b[(i, col)] - ax_i).abs();
            let abs_x = x[(i, col)].abs();
            let abs_b = b[(i, col)].abs();

            if residual > r_inf {
                r_inf = residual;
            }
            if abs_x > x_inf {
                x_inf = abs_x;
            }
            if abs_b > b_inf {
                b_inf = abs_b;
            }
        }

        let denom = a_inf * x_inf + b_inf;
        if denom > 0.0 {
            backward_error[col] = r_inf / denom;
        }

        // Forward error
        let mut r_final = Mat::<f32>::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = 0.0f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = (b[(i, col)] - ax_i) as f32;
        }

        if let Ok(e_f32) = chol_f32.solve(r_final.as_ref()) {
            let mut e_inf = 0.0f64;
            for i in 0..n {
                let abs_e = (e_f32[(i, 0)] as f64).abs();
                if abs_e > e_inf {
                    e_inf = abs_e;
                }
            }

            if x_inf > 0.0 {
                forward_error[col] = e_inf / x_inf;
            } else {
                forward_error[col] = e_inf;
            }
        } else {
            forward_error[col] = 1.0;
        }

        if forward_error[col] < eps_f64 {
            forward_error[col] = eps_f64;
        }
        if backward_error[col] < eps_f64 {
            backward_error[col] = eps_f64;
        }
    }

    Ok(MixedPrecisionResult {
        solution: x,
        forward_error,
        backward_error,
        iterations,
        achieved_double_precision: achieved_double,
    })
}

/// Solve symmetric system Ax = b using mixed precision LDL^T refinement.
///
/// Uses f32 LDL^T factorization with f64 residual computation.
///
/// # Arguments
///
/// * `a` - Symmetric coefficient matrix A (n×n) in f64
/// * `b` - Right-hand side matrix B (n×nrhs) in f64
pub fn mixed_precision_solve_symmetric(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n || b.nrows() != n {
        return Err(RefinementError::DimensionMismatch);
    }

    if n == 0 || nrhs == 0 {
        return Ok(MixedPrecisionResult {
            solution: Mat::zeros(n, nrhs),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
            achieved_double_precision: true,
        });
    }

    // Convert A to f32
    let mut a_f32 = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_f32[(i, j)] = a[(i, j)] as f32;
        }
    }

    // LDL^T factorization in f32
    let ldlt_f32 = match crate::cholesky::Ldlt::compute(a_f32.as_ref()) {
        Ok(ldlt) => ldlt,
        Err(_) => return Err(RefinementError::FactorizationFailed),
    };

    // Convert b to f32
    let mut b_f32 = Mat::<f32>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            b_f32[(i, j)] = b[(i, j)] as f32;
        }
    }

    // Initial solution in f32
    let x_f32 = match ldlt_f32.solve(b_f32.as_ref()) {
        Ok(x) => x,
        Err(_) => return Err(RefinementError::FactorizationFailed),
    };

    // Promote to f64
    let mut x = Mat::<f64>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x[(i, j)] = x_f32[(i, j)] as f64;
        }
    }

    let eps_f64 = f64::EPSILON;
    let tol = eps_f64 * CONVERGENCE_FACTOR;

    let mut a_inf = 0.0f64;
    for i in 0..n {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            row_sum += a[(i, j)].abs();
        }
        if row_sum > a_inf {
            a_inf = row_sum;
        }
    }

    let mut forward_error = vec![0.0f64; nrhs];
    let mut backward_error = vec![0.0f64; nrhs];
    let mut iterations = vec![0usize; nrhs];
    let mut achieved_double = true;

    for col in 0..nrhs {
        let mut prev_residual_norm = f64::MAX;

        for iter in 0..MIXED_MAX_ITERATIONS {
            let mut r_f64 = Mat::<f64>::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = 0.0f64;
                for j in 0..n {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                r_f64[(i, 0)] = b[(i, col)] - ax_i;
            }

            let mut r_inf = 0.0f64;
            for i in 0..n {
                let abs_r = r_f64[(i, 0)].abs();
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            if r_inf <= tol || (iter > 0 && r_inf >= prev_residual_norm) {
                iterations[col] = iter;
                if r_inf > eps_f64 * 100.0 {
                    achieved_double = false;
                }
                break;
            }

            let mut r_f32 = Mat::<f32>::zeros(n, 1);
            for i in 0..n {
                r_f32[(i, 0)] = r_f64[(i, 0)] as f32;
            }

            let d_f32 = match ldlt_f32.solve(r_f32.as_ref()) {
                Ok(d) => d,
                Err(_) => {
                    achieved_double = false;
                    break;
                }
            };

            for i in 0..n {
                x[(i, col)] += d_f32[(i, 0)] as f64;
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }

        // Compute error bounds
        let mut r_inf = 0.0f64;
        let mut x_inf = 0.0f64;
        let mut b_inf = 0.0f64;

        for i in 0..n {
            let mut ax_i = 0.0f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            let residual = (b[(i, col)] - ax_i).abs();
            let abs_x = x[(i, col)].abs();
            let abs_b = b[(i, col)].abs();

            if residual > r_inf {
                r_inf = residual;
            }
            if abs_x > x_inf {
                x_inf = abs_x;
            }
            if abs_b > b_inf {
                b_inf = abs_b;
            }
        }

        let denom = a_inf * x_inf + b_inf;
        if denom > 0.0 {
            backward_error[col] = r_inf / denom;
        }

        let mut r_final = Mat::<f32>::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = 0.0f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = (b[(i, col)] - ax_i) as f32;
        }

        if let Ok(e_f32) = ldlt_f32.solve(r_final.as_ref()) {
            let mut e_inf = 0.0f64;
            for i in 0..n {
                let abs_e = (e_f32[(i, 0)] as f64).abs();
                if abs_e > e_inf {
                    e_inf = abs_e;
                }
            }

            if x_inf > 0.0 {
                forward_error[col] = e_inf / x_inf;
            } else {
                forward_error[col] = e_inf;
            }
        } else {
            forward_error[col] = 1.0;
        }

        if forward_error[col] < eps_f64 {
            forward_error[col] = eps_f64;
        }
        if backward_error[col] < eps_f64 {
            backward_error[col] = eps_f64;
        }
    }

    Ok(MixedPrecisionResult {
        solution: x,
        forward_error,
        backward_error,
        iterations,
        achieved_double_precision: achieved_double,
    })
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

        let lu = Lu::compute(a.as_ref()).unwrap();
        let mut x = lu.solve(b.as_ref()).unwrap();

        let result = refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-10));
        assert!(approx_eq(ax1, 11.0, 1e-10));

        // Error bounds should be small
        assert!(result.forward_error[0] < 1e-10);
        assert!(result.backward_error[0] < 1e-10);
    }

    #[test]
    fn test_refine_solution_ill_conditioned() {
        // Hilbert matrix is ill-conditioned
        let a = Mat::from_rows(&[
            &[1.0f64, 1.0 / 2.0, 1.0 / 3.0],
            &[1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0],
            &[1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
        ]);
        let b = Mat::from_rows(&[&[1.0], &[1.0], &[1.0]]);

        let lu = Lu::compute(a.as_ref()).unwrap();
        let mut x = lu.solve(b.as_ref()).unwrap();

        let result = refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).unwrap();

        // At least refinement should have run
        assert!(result.forward_error.len() == 1);
        assert!(result.backward_error.len() == 1);
    }

    #[test]
    fn test_refine_solution_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0, 6.0], &[11.0, 9.0]]);

        let lu = Lu::compute(a.as_ref()).unwrap();
        let mut x = lu.solve(b.as_ref()).unwrap();

        let result = refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).unwrap();

        assert_eq!(result.forward_error.len(), 2);
        assert_eq!(result.backward_error.len(), 2);
        assert_eq!(result.iterations.len(), 2);

        // Verify both solutions
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

        let lu = Lu::compute(a.as_ref()).unwrap();
        let mut x = lu.solve(b.as_ref()).unwrap();

        let result = refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).unwrap();

        // For identity matrix, solution should be exact
        for i in 0..3 {
            assert!(approx_eq(result.solution[(i, 0)], b[(i, 0)], 1e-14));
        }
    }

    #[test]
    fn test_refine_solution_symmetric() {
        use crate::cholesky::Ldlt;

        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).unwrap();
        let mut x = ldlt.solve(b.as_ref()).unwrap();

        let result = refine_solution_symmetric(a.as_ref(), &ldlt, b.as_ref(), &mut x).unwrap();

        // Verify Ax = b
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

        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let mut x = chol.solve(b.as_ref()).unwrap();

        let result = refine_solution_cholesky(a.as_ref(), &chol, b.as_ref(), &mut x).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-10));
        assert!(approx_eq(ax1, 11.0, 1e-10));
    }

    #[test]
    fn test_refine_solution_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0f32], &[11.0]]);

        let lu = Lu::compute(a.as_ref()).unwrap();
        let mut x = lu.solve(b.as_ref()).unwrap();

        let result = refine_solution(a.as_ref(), &lu, b.as_ref(), &mut x).unwrap();

        // Verify Ax ≈ b
        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-5);
        assert!((ax1 - 11.0).abs() < 1e-5);
    }

    // ==========================================================================
    // Mixed Precision Tests
    // ==========================================================================

    #[test]
    fn test_mixed_precision_solve_simple() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax = b to f64 precision
        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-12));
        assert!(approx_eq(ax1, 11.0, 1e-12));

        // Should achieve double precision accuracy
        assert!(result.achieved_double_precision);
        assert!(result.backward_error[0] < 1e-14);
    }

    #[test]
    fn test_mixed_precision_solve_larger() {
        // 4x4 system
        let a = Mat::from_rows(&[
            &[10.0f64, 2.0, 1.0, 0.5],
            &[2.0, 8.0, 1.0, 0.3],
            &[1.0, 1.0, 6.0, 0.2],
            &[0.5, 0.3, 0.2, 4.0],
        ]);
        let b = Mat::from_rows(&[&[13.5], &[11.3], &[8.2], &[5.0]]);

        let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax ≈ b
        for i in 0..4 {
            let mut ax_i = 0.0f64;
            for j in 0..4 {
                ax_i += a[(i, j)] * result.solution[(j, 0)];
            }
            assert!(approx_eq(ax_i, b[(i, 0)], 1e-11));
        }

        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0, 6.0], &[11.0, 9.0]]);

        let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();

        assert_eq!(result.forward_error.len(), 2);
        assert_eq!(result.backward_error.len(), 2);
        assert_eq!(result.iterations.len(), 2);

        // Verify both solutions
        for col in 0..2 {
            let ax0 = 4.0 * result.solution[(0, col)] + 2.0 * result.solution[(1, col)];
            let ax1 = 2.0 * result.solution[(0, col)] + 5.0 * result.solution[(1, col)];
            assert!(approx_eq(ax0, b[(0, col)], 1e-12));
            assert!(approx_eq(ax1, b[(1, col)], 1e-12));
        }
    }

    #[test]
    fn test_mixed_precision_solve_ill_conditioned() {
        // Hilbert matrix - moderately ill-conditioned
        let a = Mat::from_rows(&[
            &[1.0f64, 1.0 / 2.0, 1.0 / 3.0],
            &[1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0],
            &[1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
        ]);
        let b = Mat::from_rows(&[&[1.0], &[1.0], &[1.0]]);

        let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();

        // Should still produce reasonable solution
        assert_eq!(result.forward_error.len(), 1);
        assert_eq!(result.backward_error.len(), 1);
    }

    #[test]
    fn test_mixed_precision_solve_cholesky() {
        // SPD matrix
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let result = mixed_precision_solve_cholesky(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-12));
        assert!(approx_eq(ax1, 11.0, 1e-12));

        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_cholesky_larger() {
        // 3x3 SPD
        let a = Mat::from_rows(&[&[10.0f64, 2.0, 1.0], &[2.0, 8.0, 2.0], &[1.0, 2.0, 6.0]]);
        let b = Mat::from_rows(&[&[13.0], &[12.0], &[9.0]]);

        let result = mixed_precision_solve_cholesky(a.as_ref(), b.as_ref()).unwrap();

        for i in 0..3 {
            let mut ax_i = 0.0f64;
            for j in 0..3 {
                ax_i += a[(i, j)] * result.solution[(j, 0)];
            }
            assert!(approx_eq(ax_i, b[(i, 0)], 1e-12));
        }
    }

    #[test]
    fn test_mixed_precision_solve_symmetric() {
        // Symmetric indefinite matrix
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, -3.0]]);
        let b = Mat::from_rows(&[&[8.0], &[-1.0]]);

        let result = mixed_precision_solve_symmetric(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] - 3.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-12));
        assert!(approx_eq(ax1, -1.0, 1e-12));
    }

    #[test]
    fn test_mixed_precision_empty_system() {
        let a = Mat::<f64>::zeros(0, 0);
        let b = Mat::<f64>::zeros(0, 1);

        let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();

        assert_eq!(result.solution.nrows(), 0);
        assert!(result.achieved_double_precision);
        assert!(result.forward_error.is_empty());
    }

    #[test]
    fn test_mixed_precision_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]); // wrong dimension

        let result = mixed_precision_solve(a.as_ref(), b.as_ref());

        assert!(matches!(result, Err(RefinementError::DimensionMismatch)));
    }

    #[test]
    fn test_mixed_precision_accuracy_improvement() {
        // Test that mixed precision improves over pure f32
        let a = Mat::from_rows(&[&[10.0f64, 1.0, 2.0], &[1.0, 8.0, 1.0], &[2.0, 1.0, 6.0]]);
        let b = Mat::from_rows(&[&[13.0], &[10.0], &[9.0]]);

        // Mixed precision should achieve f64 accuracy
        let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();

        // Verify high accuracy
        for i in 0..3 {
            let mut ax_i = 0.0f64;
            for j in 0..3 {
                ax_i += a[(i, j)] * result.solution[(j, 0)];
            }
            assert!(approx_eq(ax_i, b[(i, 0)], 1e-13));
        }

        // Should achieve double precision
        assert!(result.achieved_double_precision);
        assert!(result.backward_error[0] < 1e-14);
    }
}
