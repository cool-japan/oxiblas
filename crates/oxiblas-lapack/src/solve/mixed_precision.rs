//! Mixed-precision iterative refinement solvers.
//!
//! These routines factorize the coefficient matrix in single precision (f32)
//! while computing residuals and accumulating corrections in double precision
//! (f64). This achieves nearly f64 accuracy at reduced computational cost
//! for well-conditioned systems.
//!
//! # Supported factorizations
//!
//! - **LU**: General square systems via `mixed_precision_solve` / `mixed_precision_solve_lu`
//! - **Cholesky**: Symmetric positive-definite systems via `mixed_precision_solve_cholesky`
//! - **LDL^T**: Symmetric indefinite systems via `mixed_precision_solve_symmetric`
//! - **QR**: Least-squares problems via `mixed_precision_solve_qr`
//!
//! # Algorithm
//!
//! 1. Cast A to f32 and compute the factorization in f32
//! 2. Solve the system in f32 and promote the initial solution to f64
//! 3. Iteratively refine:
//!    - r = b - A*x  (computed in f64)
//!    - d = factor32 \ r  (solve correction in f32)
//!    - x = x + d  (accumulate in f64)
//!    - Check convergence: ||r||_inf < tol

use oxiblas_matrix::{Mat, MatRef};

use crate::lu::Lu;
use crate::qr::Qr;

use super::iterative_refinement::RefinementError;

/// Mixed-precision refinement result.
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

/// Maximum iterations for mixed-precision refinement.
const MIXED_MAX_ITERATIONS: usize = 30;

/// Convergence tolerance factor (relative to machine epsilon).
const CONVERGENCE_FACTOR: f64 = 10.0;

// ---------------------------------------------------------------------------
// Helper: compute infinity norm of matrix A (max absolute row sum)
// ---------------------------------------------------------------------------
fn matrix_inf_norm(a: MatRef<'_, f64>, n: usize) -> f64 {
    let mut a_inf = 0.0_f64;
    for i in 0..n {
        let mut row_sum = 0.0_f64;
        for j in 0..a.ncols() {
            row_sum += a[(i, j)].abs();
        }
        if row_sum > a_inf {
            a_inf = row_sum;
        }
    }
    a_inf
}

// ---------------------------------------------------------------------------
// Helper: compute final error bounds for a single RHS column
// ---------------------------------------------------------------------------
struct ErrorBounds {
    forward: f64,
    backward: f64,
}

fn compute_error_bounds_lu(
    a: MatRef<'_, f64>,
    x: &Mat<f64>,
    b: MatRef<'_, f64>,
    col: usize,
    n: usize,
    a_inf: f64,
    lu_f32: &Lu<f32>,
) -> ErrorBounds {
    let eps_f64 = f64::EPSILON;

    let mut r_inf = 0.0_f64;
    let mut x_inf = 0.0_f64;
    let mut b_inf = 0.0_f64;

    for i in 0..n {
        let mut ax_i = 0.0_f64;
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
    let backward = if denom > 0.0 {
        (r_inf / denom).max(eps_f64)
    } else {
        eps_f64
    };

    // Forward error estimate via f32 correction solve
    let mut r_final = Mat::<f32>::zeros(n, 1);
    for i in 0..n {
        let mut ax_i = 0.0_f64;
        for j in 0..n {
            ax_i += a[(i, j)] * x[(j, col)];
        }
        r_final[(i, 0)] = (b[(i, col)] - ax_i) as f32;
    }

    let forward = if let Ok(e_f32) = lu_f32.solve(r_final.as_ref()) {
        let mut e_inf = 0.0_f64;
        for i in 0..n {
            let abs_e = (e_f32[(i, 0)] as f64).abs();
            if abs_e > e_inf {
                e_inf = abs_e;
            }
        }
        let raw = if x_inf > 0.0 { e_inf / x_inf } else { e_inf };
        raw.max(eps_f64)
    } else {
        1.0
    };

    ErrorBounds { forward, backward }
}

// ---------------------------------------------------------------------------
// Core mixed-precision refinement loop for a square-system factorization.
// `solve_f32` is a closure that solves the system in f32 given an f32 RHS.
// ---------------------------------------------------------------------------
fn mixed_precision_refine_loop<F>(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    x: &mut Mat<f64>,
    n: usize,
    nrhs: usize,
    solve_f32: F,
) -> (Vec<usize>, bool)
where
    F: Fn(MatRef<'_, f32>) -> Option<Mat<f32>>,
{
    let eps_f64 = f64::EPSILON;
    let tol = eps_f64 * CONVERGENCE_FACTOR;

    let mut iterations = vec![0_usize; nrhs];
    let mut achieved_double = true;

    for col in 0..nrhs {
        let mut prev_residual_norm = f64::MAX;

        for iter in 0..MIXED_MAX_ITERATIONS {
            // Compute residual r = b - A*x in f64
            let mut r_f64 = Mat::<f64>::zeros(n, 1);
            for i in 0..n {
                let mut ax_i = 0.0_f64;
                for j in 0..n {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                r_f64[(i, 0)] = b[(i, col)] - ax_i;
            }

            // Infinity norm of residual
            let mut r_inf = 0.0_f64;
            for i in 0..n {
                let abs_r = r_f64[(i, 0)].abs();
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            // Convergence or stagnation check
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

            // Solve correction in f32
            let d_f32 = match solve_f32(r_f32.as_ref()) {
                Some(d) => d,
                None => {
                    achieved_double = false;
                    break;
                }
            };

            // Accumulate in f64
            for i in 0..n {
                x[(i, col)] += d_f32[(i, 0)] as f64;
            }

            prev_residual_norm = r_inf;
            iterations[col] = iter + 1;
        }
    }

    (iterations, achieved_double)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solve Ax = b using mixed-precision iterative refinement (LU).
///
/// Uses f32 LU factorization with f64 residual computation to achieve
/// nearly f64 accuracy at reduced computational cost.
///
/// # Algorithm
///
/// 1. Factor A in f32 (working precision)
/// 2. Compute initial solution x0 in f32, promote to f64
/// 3. For each iteration:
///    a. Compute residual r = b - Ax in f64
///    b. Convert r to f32, solve A*d = r using f32 factorization
///    c. Convert d to f64, update x = x + d
///    d. Check for convergence to f64 accuracy
///
/// # Arguments
///
/// * `a` - Coefficient matrix A (n x n) in f64
/// * `b` - Right-hand side matrix B (n x nrhs) in f64
///
/// # Returns
///
/// `MixedPrecisionResult` with solution in f64 and error bounds.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::mixed_precision::mixed_precision_solve;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0],
///     &[2.0, 5.0],
/// ]);
/// let b = Mat::from_rows(&[&[8.0], &[11.0]]);
///
/// let result = mixed_precision_solve(a.as_ref(), b.as_ref()).unwrap();
/// assert!(result.achieved_double_precision);
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

    if n == 0 || nrhs == 0 {
        return Ok(MixedPrecisionResult {
            solution: Mat::zeros(n, nrhs),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
            achieved_double_precision: true,
        });
    }

    // Cast A to f32 for factorization
    let mut a_f32 = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_f32[(i, j)] = a[(i, j)] as f32;
        }
    }

    let lu_f32 = Lu::compute(a_f32.as_ref()).map_err(|_| RefinementError::FactorizationFailed)?;

    // Initial solve in f32
    let mut b_f32 = Mat::<f32>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            b_f32[(i, j)] = b[(i, j)] as f32;
        }
    }

    let x_f32 = lu_f32
        .solve(b_f32.as_ref())
        .map_err(|_| RefinementError::FactorizationFailed)?;

    // Promote to f64
    let mut x = Mat::<f64>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x[(i, j)] = x_f32[(i, j)] as f64;
        }
    }

    let a_inf = matrix_inf_norm(a, n);

    // Iterative refinement loop
    let (iterations, achieved_double) =
        mixed_precision_refine_loop(a, b, &mut x, n, nrhs, |rhs| lu_f32.solve(rhs).ok());

    // Compute error bounds per RHS
    let mut forward_error = vec![0.0_f64; nrhs];
    let mut backward_error = vec![0.0_f64; nrhs];

    for col in 0..nrhs {
        let eb = compute_error_bounds_lu(a, &x, b, col, n, a_inf, &lu_f32);
        forward_error[col] = eb.forward;
        backward_error[col] = eb.backward;
    }

    Ok(MixedPrecisionResult {
        solution: x,
        forward_error,
        backward_error,
        iterations,
        achieved_double_precision: achieved_double,
    })
}

/// Alias for [`mixed_precision_solve`] -- LU-based mixed-precision solve.
///
/// Factorizes A in f32 via LU, then iteratively refines in f64.
pub fn mixed_precision_solve_lu(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    mixed_precision_solve(a, b)
}

/// Solve SPD system Ax = b using mixed-precision Cholesky refinement.
///
/// Uses f32 Cholesky factorization with f64 residual computation.
///
/// # Arguments
///
/// * `a` - SPD coefficient matrix A (n x n) in f64
/// * `b` - Right-hand side matrix B (n x nrhs) in f64
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::mixed_precision::mixed_precision_solve_cholesky;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0],
///     &[2.0, 5.0],
/// ]);
/// let b = Mat::from_rows(&[&[8.0], &[11.0]]);
///
/// let result = mixed_precision_solve_cholesky(a.as_ref(), b.as_ref()).unwrap();
/// assert!(result.achieved_double_precision);
/// ```
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

    // Cast A to f32
    let mut a_f32 = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_f32[(i, j)] = a[(i, j)] as f32;
        }
    }

    let chol_f32 = crate::cholesky::Cholesky::compute(a_f32.as_ref())
        .map_err(|_| RefinementError::FactorizationFailed)?;

    // Initial solve in f32
    let mut b_f32 = Mat::<f32>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            b_f32[(i, j)] = b[(i, j)] as f32;
        }
    }

    let x_f32 = chol_f32
        .solve(b_f32.as_ref())
        .map_err(|_| RefinementError::FactorizationFailed)?;

    let mut x = Mat::<f64>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x[(i, j)] = x_f32[(i, j)] as f64;
        }
    }

    let a_inf = matrix_inf_norm(a, n);

    let (iterations, achieved_double) =
        mixed_precision_refine_loop(a, b, &mut x, n, nrhs, |rhs| chol_f32.solve(rhs).ok());

    // Error bounds -- reuse LU error-bound logic structurally.
    // For Cholesky we compute the same norms; the forward error estimate
    // uses the Cholesky solve instead of LU, but we inline that here.
    let eps_f64 = f64::EPSILON;
    let mut forward_error = vec![0.0_f64; nrhs];
    let mut backward_error = vec![0.0_f64; nrhs];

    for col in 0..nrhs {
        let mut r_inf = 0.0_f64;
        let mut x_inf = 0.0_f64;
        let mut b_inf = 0.0_f64;

        for i in 0..n {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            let residual = (b[(i, col)] - ax_i).abs();
            if residual > r_inf {
                r_inf = residual;
            }
            let abs_x = x[(i, col)].abs();
            if abs_x > x_inf {
                x_inf = abs_x;
            }
            let abs_b = b[(i, col)].abs();
            if abs_b > b_inf {
                b_inf = abs_b;
            }
        }

        let denom = a_inf * x_inf + b_inf;
        backward_error[col] = if denom > 0.0 {
            (r_inf / denom).max(eps_f64)
        } else {
            eps_f64
        };

        // Forward error via Cholesky correction
        let mut r_final = Mat::<f32>::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = (b[(i, col)] - ax_i) as f32;
        }

        forward_error[col] = if let Ok(e_f32) = chol_f32.solve(r_final.as_ref()) {
            let mut e_inf = 0.0_f64;
            for i in 0..n {
                let abs_e = (e_f32[(i, 0)] as f64).abs();
                if abs_e > e_inf {
                    e_inf = abs_e;
                }
            }
            let raw = if x_inf > 0.0 { e_inf / x_inf } else { e_inf };
            raw.max(eps_f64)
        } else {
            1.0
        };
    }

    Ok(MixedPrecisionResult {
        solution: x,
        forward_error,
        backward_error,
        iterations,
        achieved_double_precision: achieved_double,
    })
}

/// Solve symmetric system Ax = b using mixed-precision LDL^T refinement.
///
/// Uses f32 LDL^T factorization with f64 residual computation.
///
/// # Arguments
///
/// * `a` - Symmetric coefficient matrix A (n x n) in f64
/// * `b` - Right-hand side matrix B (n x nrhs) in f64
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::mixed_precision::mixed_precision_solve_symmetric;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0],
///     &[2.0, -3.0],
/// ]);
/// let b = Mat::from_rows(&[&[8.0], &[-1.0]]);
///
/// let result = mixed_precision_solve_symmetric(a.as_ref(), b.as_ref()).unwrap();
/// ```
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

    let mut a_f32 = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_f32[(i, j)] = a[(i, j)] as f32;
        }
    }

    let ldlt_f32 = crate::cholesky::Ldlt::compute(a_f32.as_ref())
        .map_err(|_| RefinementError::FactorizationFailed)?;

    let mut b_f32 = Mat::<f32>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            b_f32[(i, j)] = b[(i, j)] as f32;
        }
    }

    let x_f32 = ldlt_f32
        .solve(b_f32.as_ref())
        .map_err(|_| RefinementError::FactorizationFailed)?;

    let mut x = Mat::<f64>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x[(i, j)] = x_f32[(i, j)] as f64;
        }
    }

    let a_inf = matrix_inf_norm(a, n);

    let (iterations, achieved_double) =
        mixed_precision_refine_loop(a, b, &mut x, n, nrhs, |rhs| ldlt_f32.solve(rhs).ok());

    let eps_f64 = f64::EPSILON;
    let mut forward_error = vec![0.0_f64; nrhs];
    let mut backward_error = vec![0.0_f64; nrhs];

    for col in 0..nrhs {
        let mut r_inf = 0.0_f64;
        let mut x_inf = 0.0_f64;
        let mut b_inf = 0.0_f64;

        for i in 0..n {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            let residual = (b[(i, col)] - ax_i).abs();
            if residual > r_inf {
                r_inf = residual;
            }
            if x[(i, col)].abs() > x_inf {
                x_inf = x[(i, col)].abs();
            }
            if b[(i, col)].abs() > b_inf {
                b_inf = b[(i, col)].abs();
            }
        }

        let denom = a_inf * x_inf + b_inf;
        backward_error[col] = if denom > 0.0 {
            (r_inf / denom).max(eps_f64)
        } else {
            eps_f64
        };

        let mut r_final = Mat::<f32>::zeros(n, 1);
        for i in 0..n {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            r_final[(i, 0)] = (b[(i, col)] - ax_i) as f32;
        }

        forward_error[col] = if let Ok(e_f32) = ldlt_f32.solve(r_final.as_ref()) {
            let mut e_inf = 0.0_f64;
            for i in 0..n {
                let abs_e = (e_f32[(i, 0)] as f64).abs();
                if abs_e > e_inf {
                    e_inf = abs_e;
                }
            }
            let raw = if x_inf > 0.0 { e_inf / x_inf } else { e_inf };
            raw.max(eps_f64)
        } else {
            1.0
        };
    }

    Ok(MixedPrecisionResult {
        solution: x,
        forward_error,
        backward_error,
        iterations,
        achieved_double_precision: achieved_double,
    })
}

/// Solve a least-squares problem min ||Ax - b||_2 using mixed-precision QR
/// refinement.
///
/// Factorizes A in f32 via QR decomposition, then iteratively refines the
/// solution in f64 by computing residuals in double precision and solving
/// corrections in single precision.
///
/// # Arguments
///
/// * `a` - Coefficient matrix A (m x n, m >= n) in f64
/// * `b` - Right-hand side matrix B (m x nrhs) in f64
///
/// # Returns
///
/// `MixedPrecisionResult` with the least-squares solution in f64.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::mixed_precision::mixed_precision_solve_qr;
/// use oxiblas_matrix::Mat;
///
/// // Overdetermined 3x2 system
/// let a = Mat::from_rows(&[
///     &[1.0_f64, 1.0],
///     &[1.0, 2.0],
///     &[1.0, 3.0],
/// ]);
/// let b = Mat::from_rows(&[&[6.0], &[8.0], &[10.0]]);
///
/// let result = mixed_precision_solve_qr(a.as_ref(), b.as_ref()).unwrap();
/// // solution ~ [4.0, 2.0]
/// assert!(result.achieved_double_precision);
/// ```
pub fn mixed_precision_solve_qr(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Result<MixedPrecisionResult, RefinementError> {
    let m = a.nrows();
    let n = a.ncols();
    let nrhs = b.ncols();

    if b.nrows() != m {
        return Err(RefinementError::DimensionMismatch);
    }

    if m < n {
        return Err(RefinementError::DimensionMismatch);
    }

    if m == 0 || n == 0 || nrhs == 0 {
        return Ok(MixedPrecisionResult {
            solution: Mat::zeros(n, nrhs),
            forward_error: vec![],
            backward_error: vec![],
            iterations: vec![],
            achieved_double_precision: true,
        });
    }

    // Cast A to f32 for QR factorization
    let mut a_f32 = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            a_f32[(i, j)] = a[(i, j)] as f32;
        }
    }

    let qr_f32 = Qr::compute(a_f32.as_ref()).map_err(|_| RefinementError::FactorizationFailed)?;

    // Initial least-squares solve in f32
    let mut b_f32 = Mat::<f32>::zeros(m, nrhs);
    for i in 0..m {
        for j in 0..nrhs {
            b_f32[(i, j)] = b[(i, j)] as f32;
        }
    }

    let x_f32 = qr_f32
        .solve_least_squares(b_f32.as_ref())
        .map_err(|_| RefinementError::FactorizationFailed)?;

    // Promote to f64
    let mut x = Mat::<f64>::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            x[(i, j)] = x_f32[(i, j)] as f64;
        }
    }

    let eps_f64 = f64::EPSILON;
    let tol = eps_f64 * CONVERGENCE_FACTOR;

    // Compute ||A||_inf for error bounds
    let a_inf = matrix_inf_norm(a, m);

    let mut forward_error = vec![0.0_f64; nrhs];
    let mut backward_error = vec![0.0_f64; nrhs];
    let mut iterations = vec![0_usize; nrhs];
    let mut achieved_double = true;

    // For least-squares the normal-equations residual is A^T (b - A x).
    // We refine by solving the correction via QR in f32.
    for col in 0..nrhs {
        let mut prev_residual_norm = f64::MAX;

        for iter in 0..MIXED_MAX_ITERATIONS {
            // Compute residual r = b - A*x in f64 (m-dimensional)
            let mut r_f64 = Mat::<f64>::zeros(m, 1);
            for i in 0..m {
                let mut ax_i = 0.0_f64;
                for j in 0..n {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                r_f64[(i, 0)] = b[(i, col)] - ax_i;
            }

            // Compute normal-equations residual norm: ||A^T r||_inf
            // This measures how far x is from the least-squares optimum.
            let mut atr_inf = 0.0_f64;
            for j in 0..n {
                let mut atr_j = 0.0_f64;
                for i in 0..m {
                    atr_j += a[(i, j)] * r_f64[(i, 0)];
                }
                let abs_val = atr_j.abs();
                if abs_val > atr_inf {
                    atr_inf = abs_val;
                }
            }

            // Also track plain residual norm for error bounds
            let mut r_inf = 0.0_f64;
            for i in 0..m {
                let abs_r = r_f64[(i, 0)].abs();
                if abs_r > r_inf {
                    r_inf = abs_r;
                }
            }

            // Convergence or stagnation
            if atr_inf <= tol || (iter > 0 && atr_inf >= prev_residual_norm) {
                iterations[col] = iter;
                if atr_inf > eps_f64 * 100.0 {
                    achieved_double = false;
                }
                break;
            }

            // Convert residual to f32 for correction solve
            let mut r_f32 = Mat::<f32>::zeros(m, 1);
            for i in 0..m {
                r_f32[(i, 0)] = r_f64[(i, 0)] as f32;
            }

            // Solve least-squares correction: min ||A*d - r|| in f32
            let d_f32 = match qr_f32.solve_least_squares(r_f32.as_ref()) {
                Ok(d) => d,
                Err(_) => {
                    achieved_double = false;
                    break;
                }
            };

            // Accumulate correction in f64
            for i in 0..n {
                x[(i, col)] += d_f32[(i, 0)] as f64;
            }

            prev_residual_norm = atr_inf;
            iterations[col] = iter + 1;
        }

        // Compute final error bounds
        let mut x_inf = 0.0_f64;
        let mut b_inf = 0.0_f64;

        // Residual r = b - Ax
        let mut r_inf_final = 0.0_f64;
        for i in 0..m {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            let residual = (b[(i, col)] - ax_i).abs();
            if residual > r_inf_final {
                r_inf_final = residual;
            }
            if b[(i, col)].abs() > b_inf {
                b_inf = b[(i, col)].abs();
            }
        }
        for i in 0..n {
            if x[(i, col)].abs() > x_inf {
                x_inf = x[(i, col)].abs();
            }
        }

        // Backward error
        let denom = a_inf * x_inf + b_inf;
        backward_error[col] = if denom > 0.0 {
            (r_inf_final / denom).max(eps_f64)
        } else {
            eps_f64
        };

        // Forward error estimate via f32 correction
        let mut r_final_f32 = Mat::<f32>::zeros(m, 1);
        for i in 0..m {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, col)];
            }
            r_final_f32[(i, 0)] = (b[(i, col)] - ax_i) as f32;
        }

        forward_error[col] = if let Ok(e_f32) = qr_f32.solve_least_squares(r_final_f32.as_ref()) {
            let mut e_inf = 0.0_f64;
            for i in 0..n {
                let abs_e = (e_f32[(i, 0)] as f64).abs();
                if abs_e > e_inf {
                    e_inf = abs_e;
                }
            }
            let raw = if x_inf > 0.0 { e_inf / x_inf } else { e_inf };
            raw.max(eps_f64)
        } else {
            1.0
        };
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

    // ======================================================================
    // LU mixed-precision tests
    // ======================================================================

    #[test]
    fn test_mixed_precision_solve_simple() {
        let a = Mat::from_rows(&[&[4.0_f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let result = mixed_precision_solve(a.as_ref(), b.as_ref()).expect("solve should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-12));
        assert!(approx_eq(ax1, 11.0, 1e-12));
        assert!(result.achieved_double_precision);
        assert!(result.backward_error[0] < 1e-14);
    }

    #[test]
    fn test_mixed_precision_solve_lu_alias() {
        let a = Mat::from_rows(&[&[4.0_f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let result =
            mixed_precision_solve_lu(a.as_ref(), b.as_ref()).expect("lu alias should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-12));
        assert!(approx_eq(ax1, 11.0, 1e-12));
        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_larger() {
        let a = Mat::from_rows(&[
            &[10.0_f64, 2.0, 1.0, 0.5],
            &[2.0, 8.0, 1.0, 0.3],
            &[1.0, 1.0, 6.0, 0.2],
            &[0.5, 0.3, 0.2, 4.0],
        ]);
        let b = Mat::from_rows(&[&[13.5], &[11.3], &[8.2], &[5.0]]);

        let result =
            mixed_precision_solve(a.as_ref(), b.as_ref()).expect("larger system should succeed");

        for i in 0..4 {
            let mut ax_i = 0.0_f64;
            for j in 0..4 {
                ax_i += a[(i, j)] * result.solution[(j, 0)];
            }
            assert!(approx_eq(ax_i, b[(i, 0)], 1e-11));
        }
        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0_f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0, 6.0], &[11.0, 9.0]]);

        let result =
            mixed_precision_solve(a.as_ref(), b.as_ref()).expect("multiple rhs should succeed");

        assert_eq!(result.forward_error.len(), 2);
        assert_eq!(result.backward_error.len(), 2);
        assert_eq!(result.iterations.len(), 2);

        for col in 0..2 {
            let ax0 = 4.0 * result.solution[(0, col)] + 2.0 * result.solution[(1, col)];
            let ax1 = 2.0 * result.solution[(0, col)] + 5.0 * result.solution[(1, col)];
            assert!(approx_eq(ax0, b[(0, col)], 1e-12));
            assert!(approx_eq(ax1, b[(1, col)], 1e-12));
        }
    }

    #[test]
    fn test_mixed_precision_solve_ill_conditioned() {
        // 3x3 Hilbert matrix
        let a = Mat::from_rows(&[
            &[1.0_f64, 1.0 / 2.0, 1.0 / 3.0],
            &[1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0],
            &[1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
        ]);
        let b = Mat::from_rows(&[&[1.0], &[1.0], &[1.0]]);

        let result =
            mixed_precision_solve(a.as_ref(), b.as_ref()).expect("ill-conditioned should succeed");

        assert_eq!(result.forward_error.len(), 1);
        assert_eq!(result.backward_error.len(), 1);
        // May need more iterations
        assert!(result.iterations[0] >= 1);
    }

    #[test]
    fn test_mixed_precision_empty_system() {
        let a = Mat::<f64>::zeros(0, 0);
        let b = Mat::<f64>::zeros(0, 1);

        let result =
            mixed_precision_solve(a.as_ref(), b.as_ref()).expect("empty system should succeed");

        assert_eq!(result.solution.nrows(), 0);
        assert!(result.achieved_double_precision);
        assert!(result.forward_error.is_empty());
    }

    #[test]
    fn test_mixed_precision_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0_f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let result = mixed_precision_solve(a.as_ref(), b.as_ref());
        assert!(matches!(result, Err(RefinementError::DimensionMismatch)));
    }

    #[test]
    fn test_mixed_precision_accuracy_improvement() {
        let a = Mat::from_rows(&[&[10.0_f64, 1.0, 2.0], &[1.0, 8.0, 1.0], &[2.0, 1.0, 6.0]]);
        let b = Mat::from_rows(&[&[13.0], &[10.0], &[9.0]]);

        let result =
            mixed_precision_solve(a.as_ref(), b.as_ref()).expect("accuracy test should succeed");

        for i in 0..3 {
            let mut ax_i = 0.0_f64;
            for j in 0..3 {
                ax_i += a[(i, j)] * result.solution[(j, 0)];
            }
            assert!(approx_eq(ax_i, b[(i, 0)], 1e-13));
        }
        assert!(result.achieved_double_precision);
        assert!(result.backward_error[0] < 1e-14);
    }

    #[test]
    fn test_mixed_precision_convergence_few_iterations() {
        // Well-conditioned diagonal-dominant system should converge in 1-3 iters
        let a = Mat::from_rows(&[&[10.0_f64, 1.0, 0.5], &[1.0, 10.0, 1.0], &[0.5, 1.0, 10.0]]);
        let b = Mat::from_rows(&[&[11.5], &[12.0], &[11.5]]);

        let result =
            mixed_precision_solve(a.as_ref(), b.as_ref()).expect("convergence test should succeed");

        assert!(
            result.iterations[0] <= 3,
            "expected <= 3 iterations, got {}",
            result.iterations[0]
        );
        assert!(result.achieved_double_precision);
    }

    // ======================================================================
    // Cholesky mixed-precision tests
    // ======================================================================

    #[test]
    fn test_mixed_precision_solve_cholesky_simple() {
        let a = Mat::from_rows(&[&[4.0_f64, 2.0], &[2.0, 5.0]]);
        let b = Mat::from_rows(&[&[8.0], &[11.0]]);

        let result = mixed_precision_solve_cholesky(a.as_ref(), b.as_ref())
            .expect("cholesky solve should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] + 5.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-12));
        assert!(approx_eq(ax1, 11.0, 1e-12));
        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_cholesky_larger() {
        let a = Mat::from_rows(&[&[10.0_f64, 2.0, 1.0], &[2.0, 8.0, 2.0], &[1.0, 2.0, 6.0]]);
        let b = Mat::from_rows(&[&[13.0], &[12.0], &[9.0]]);

        let result = mixed_precision_solve_cholesky(a.as_ref(), b.as_ref())
            .expect("cholesky larger should succeed");

        for i in 0..3 {
            let mut ax_i = 0.0_f64;
            for j in 0..3 {
                ax_i += a[(i, j)] * result.solution[(j, 0)];
            }
            assert!(approx_eq(ax_i, b[(i, 0)], 1e-12));
        }
    }

    // ======================================================================
    // Symmetric (LDL^T) mixed-precision tests
    // ======================================================================

    #[test]
    fn test_mixed_precision_solve_symmetric_simple() {
        let a = Mat::from_rows(&[&[4.0_f64, 2.0], &[2.0, -3.0]]);
        let b = Mat::from_rows(&[&[8.0], &[-1.0]]);

        let result = mixed_precision_solve_symmetric(a.as_ref(), b.as_ref())
            .expect("symmetric solve should succeed");

        let ax0 = 4.0 * result.solution[(0, 0)] + 2.0 * result.solution[(1, 0)];
        let ax1 = 2.0 * result.solution[(0, 0)] - 3.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 8.0, 1e-12));
        assert!(approx_eq(ax1, -1.0, 1e-12));
    }

    // ======================================================================
    // QR mixed-precision tests (least squares)
    // ======================================================================

    #[test]
    fn test_mixed_precision_solve_qr_exact_system() {
        // Square system solved via QR -- should be exact
        let a = Mat::from_rows(&[&[2.0_f64, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0], &[7.0]]);

        let result =
            mixed_precision_solve_qr(a.as_ref(), b.as_ref()).expect("qr exact should succeed");

        // Verify Ax = b
        let ax0 = 2.0 * result.solution[(0, 0)] + 1.0 * result.solution[(1, 0)];
        let ax1 = 1.0 * result.solution[(0, 0)] + 3.0 * result.solution[(1, 0)];
        assert!(approx_eq(ax0, 5.0, 1e-10));
        assert!(approx_eq(ax1, 7.0, 1e-10));
        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_qr_overdetermined() {
        // 3 equations, 2 unknowns -- points on a line
        let a = Mat::from_rows(&[&[1.0_f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0], &[8.0], &[10.0]]);

        let result = mixed_precision_solve_qr(a.as_ref(), b.as_ref())
            .expect("qr overdetermined should succeed");

        // Exact fit y = 4 + 2x
        assert!(approx_eq(result.solution[(0, 0)], 4.0, 1e-10));
        assert!(approx_eq(result.solution[(1, 0)], 2.0, 1e-10));
        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_qr_overdetermined_noisy() {
        // Points not exactly on a line
        let a = Mat::from_rows(&[&[1.0_f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0], &[8.0], &[11.0]]);

        let result =
            mixed_precision_solve_qr(a.as_ref(), b.as_ref()).expect("qr noisy should succeed");

        // Solution should match pure f64 lstsq closely
        let lstsq_result =
            crate::solve::lstsq(a.as_ref(), b.as_ref()).expect("lstsq should succeed");

        // For noisy (non-exact) least-squares, f32 factorization limits
        // how precisely the iterative refinement can converge. A tolerance
        // of 1e-7 is well within the expected range for mixed-precision QR.
        for i in 0..2 {
            assert!(
                approx_eq(result.solution[(i, 0)], lstsq_result.solution[(i, 0)], 1e-7),
                "solution[{i}] mismatch: mixed={} pure={}",
                result.solution[(i, 0)],
                lstsq_result.solution[(i, 0)],
            );
        }
    }

    #[test]
    fn test_mixed_precision_solve_qr_multiple_rhs() {
        let a = Mat::from_rows(&[&[1.0_f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0, 3.0], &[8.0, 5.0], &[10.0, 7.0]]);

        let result = mixed_precision_solve_qr(a.as_ref(), b.as_ref())
            .expect("qr multiple rhs should succeed");

        assert_eq!(result.iterations.len(), 2);

        // First col: y = 4 + 2x
        assert!(approx_eq(result.solution[(0, 0)], 4.0, 1e-10));
        assert!(approx_eq(result.solution[(1, 0)], 2.0, 1e-10));

        // Second col: y = 1 + 2x
        assert!(approx_eq(result.solution[(0, 1)], 1.0, 1e-10));
        assert!(approx_eq(result.solution[(1, 1)], 2.0, 1e-10));
    }

    #[test]
    fn test_mixed_precision_solve_qr_accuracy_vs_f64() {
        // Larger system -- verify mixed precision matches pure f64
        let a = Mat::from_rows(&[
            &[3.0_f64, 1.0, 0.5],
            &[1.0, 4.0, 1.0],
            &[0.5, 1.0, 5.0],
            &[1.0, 0.5, 0.3],
            &[0.2, 1.0, 0.8],
        ]);
        let b = Mat::from_rows(&[&[4.5], &[6.0], &[6.5], &[1.8], &[2.0]]);

        let result =
            mixed_precision_solve_qr(a.as_ref(), b.as_ref()).expect("qr accuracy should succeed");

        let lstsq_result =
            crate::solve::lstsq(a.as_ref(), b.as_ref()).expect("lstsq should succeed");

        for i in 0..3 {
            assert!(
                approx_eq(
                    result.solution[(i, 0)],
                    lstsq_result.solution[(i, 0)],
                    1e-10
                ),
                "solution[{i}] mismatch: mixed={} pure={}",
                result.solution[(i, 0)],
                lstsq_result.solution[(i, 0)],
            );
        }
    }

    #[test]
    fn test_mixed_precision_solve_qr_convergence_few_iterations() {
        // Well-conditioned overdetermined system
        let a = Mat::from_rows(&[&[10.0_f64, 1.0], &[1.0, 10.0], &[2.0, 1.0], &[1.0, 2.0]]);
        let b = Mat::from_rows(&[&[11.0], &[11.0], &[3.0], &[3.0]]);

        let result = mixed_precision_solve_qr(a.as_ref(), b.as_ref())
            .expect("qr convergence should succeed");

        assert!(
            result.iterations[0] <= 3,
            "expected <= 3 iterations for well-conditioned QR, got {}",
            result.iterations[0]
        );
        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_qr_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0_f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let result = mixed_precision_solve_qr(a.as_ref(), b.as_ref());
        assert!(matches!(result, Err(RefinementError::DimensionMismatch)));
    }

    #[test]
    fn test_mixed_precision_solve_qr_underdetermined_rejected() {
        // m < n should be rejected
        let a = Mat::from_rows(&[&[1.0_f64, 2.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0]]);

        let result = mixed_precision_solve_qr(a.as_ref(), b.as_ref());
        assert!(matches!(result, Err(RefinementError::DimensionMismatch)));
    }

    #[test]
    fn test_mixed_precision_solve_qr_empty() {
        let a = Mat::<f64>::zeros(0, 0);
        let b = Mat::<f64>::zeros(0, 1);

        let result =
            mixed_precision_solve_qr(a.as_ref(), b.as_ref()).expect("empty qr should succeed");

        assert_eq!(result.solution.nrows(), 0);
        assert!(result.achieved_double_precision);
    }

    #[test]
    fn test_mixed_precision_solve_qr_ill_conditioned() {
        // Near-collinear columns
        let a = Mat::from_rows(&[
            &[1.0_f64, 1.0001],
            &[1.0, 1.0002],
            &[1.0, 1.0003],
            &[1.0, 1.0004],
        ]);
        let b = Mat::from_rows(&[&[2.0001], &[2.0002], &[2.0003], &[2.0004]]);

        let result = mixed_precision_solve_qr(a.as_ref(), b.as_ref())
            .expect("ill-conditioned qr should succeed");

        // Should still produce a result (may not achieve double precision)
        assert_eq!(result.forward_error.len(), 1);
        assert_eq!(result.backward_error.len(), 1);
    }
}
