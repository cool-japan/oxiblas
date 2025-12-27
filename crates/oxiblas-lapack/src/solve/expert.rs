//! Expert linear system solvers.
//!
//! Provides enhanced solve routines with equilibration, condition estimation,
//! and error bounds, similar to LAPACK's xGESVX routines.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::lu::{Lu, LuError};
use crate::utils::{norm_1, norm_inf};

/// Error type for expert system solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSolveError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix and vector dimensions don't match.
    DimensionMismatch,
    /// Matrix is singular.
    SingularMatrix,
    /// Matrix is singular to working precision.
    SingularToWorkingPrecision,
}

impl core::fmt::Display for ExpertSolveError {
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

impl std::error::Error for ExpertSolveError {}

impl From<LuError> for ExpertSolveError {
    fn from(e: LuError) -> Self {
        match e {
            LuError::Singular { .. } => Self::SingularMatrix,
            LuError::NotSquare { .. } => Self::NotSquare,
            LuError::DimensionMismatch { .. } => Self::DimensionMismatch,
        }
    }
}

/// Equilibration type for scaling matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Equilibrate {
    /// No equilibration.
    None,
    /// Row equilibration (scale rows).
    Row,
    /// Column equilibration (scale columns).
    Column,
    /// Both row and column equilibration.
    Both,
}

/// Result of expert solve operation.
#[derive(Debug, Clone)]
pub struct ExpertSolveResult<T: Scalar> {
    /// Solution matrix X.
    pub solution: Mat<T>,
    /// Reciprocal condition number estimate (1/kappa).
    pub rcond: T,
    /// Forward error bound for each right-hand side.
    pub forward_error: Vec<T>,
    /// Backward error bound for each right-hand side.
    pub backward_error: Vec<T>,
    /// Row scaling factors (if equilibration was applied).
    pub row_scale: Option<Vec<T>>,
    /// Column scaling factors (if equilibration was applied).
    pub col_scale: Option<Vec<T>>,
    /// Equilibration type that was applied.
    pub equil_type: Equilibrate,
}

/// Solves a general linear system Ax = b with expert options.
///
/// This routine provides:
/// - Optional equilibration to improve condition
/// - Condition number estimation
/// - Forward and backward error bounds
///
/// # Arguments
///
/// * `a` - Square coefficient matrix A (n×n)
/// * `b` - Right-hand side matrix B (n×nrhs)
/// * `equil` - Equilibration type
///
/// # Returns
///
/// `ExpertSolveResult` containing the solution and diagnostic information.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::expert::{solve_expert, Equilibrate};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[2.0f64, 1.0],
///     &[1.0, 3.0],
/// ]);
/// let b = Mat::from_rows(&[&[5.0], &[7.0]]);
///
/// let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::None).unwrap();
/// println!("Solution: {:?}", result.solution);
/// println!("Reciprocal condition: {}", result.rcond);
/// ```
pub fn solve_expert<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    equil: Equilibrate,
) -> Result<ExpertSolveResult<T>, ExpertSolveError> {
    let n = a.nrows();
    let nrhs = b.ncols();

    if a.ncols() != n {
        return Err(ExpertSolveError::NotSquare);
    }
    if b.nrows() != n {
        return Err(ExpertSolveError::DimensionMismatch);
    }

    // Handle empty case
    if n == 0 {
        return Ok(ExpertSolveResult {
            solution: Mat::zeros(0, nrhs),
            rcond: T::one(),
            forward_error: vec![],
            backward_error: vec![],
            row_scale: None,
            col_scale: None,
            equil_type: Equilibrate::None,
        });
    }

    // Apply equilibration if requested
    let (a_scaled, b_scaled, row_scale, col_scale, actual_equil) = apply_equilibration(a, b, equil);

    // Compute 1-norm of A before factorization (for rcond)
    let anorm = norm_1(a_scaled.as_ref());

    // Compute LU decomposition
    let lu = Lu::compute(a_scaled.as_ref())?;

    // Solve the system
    let x_scaled = solve_multiple_with_lu(&lu, b_scaled.as_ref(), n, nrhs)?;

    // Estimate reciprocal condition number
    let rcond = estimate_rcond(&lu, anorm, n);

    // Check if matrix is singular to working precision
    let eps = <T as Scalar>::epsilon();
    if rcond < eps {
        return Err(ExpertSolveError::SingularToWorkingPrecision);
    }

    // Compute error bounds
    let (forward_error, backward_error) =
        compute_error_bounds(&a_scaled, &b_scaled, &x_scaled, &lu, n, nrhs);

    // Unscale solution if equilibration was applied
    let solution = unscale_solution(&x_scaled, &col_scale);

    Ok(ExpertSolveResult {
        solution,
        rcond,
        forward_error,
        backward_error,
        row_scale,
        col_scale,
        equil_type: actual_equil,
    })
}

/// Apply equilibration to matrix A and right-hand side b.
fn apply_equilibration<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    equil: Equilibrate,
) -> (Mat<T>, Mat<T>, Option<Vec<T>>, Option<Vec<T>>, Equilibrate) {
    let n = a.nrows();
    let nrhs = b.ncols();

    match equil {
        Equilibrate::None => {
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
            (a_copy, b_copy, None, None, Equilibrate::None)
        }
        Equilibrate::Row => {
            // Compute row scaling: R[i] = 1 / max_j |A[i,j]|
            let mut row_scale = vec![T::one(); n];
            let mut a_scaled = Mat::zeros(n, n);
            let mut b_scaled = Mat::zeros(n, nrhs);

            for i in 0..n {
                let mut row_max = T::zero();
                for j in 0..n {
                    let abs_val = Scalar::abs(a[(i, j)]);
                    if abs_val > row_max {
                        row_max = abs_val;
                    }
                }
                if row_max > T::zero() {
                    row_scale[i] = T::one() / row_max;
                }
            }

            // Apply scaling
            for i in 0..n {
                let r = row_scale[i];
                for j in 0..n {
                    a_scaled[(i, j)] = r * a[(i, j)];
                }
                for j in 0..nrhs {
                    b_scaled[(i, j)] = r * b[(i, j)];
                }
            }

            (a_scaled, b_scaled, Some(row_scale), None, Equilibrate::Row)
        }
        Equilibrate::Column => {
            // Compute column scaling: C[j] = 1 / max_i |A[i,j]|
            let mut col_scale = vec![T::one(); n];
            let mut a_scaled = Mat::zeros(n, n);
            let mut b_scaled = Mat::zeros(n, nrhs);

            for j in 0..n {
                let mut col_max = T::zero();
                for i in 0..n {
                    let abs_val = Scalar::abs(a[(i, j)]);
                    if abs_val > col_max {
                        col_max = abs_val;
                    }
                }
                if col_max > T::zero() {
                    col_scale[j] = T::one() / col_max;
                }
            }

            // Apply scaling (only to A, not b)
            for i in 0..n {
                for j in 0..n {
                    a_scaled[(i, j)] = col_scale[j] * a[(i, j)];
                }
                for j in 0..nrhs {
                    b_scaled[(i, j)] = b[(i, j)];
                }
            }

            (
                a_scaled,
                b_scaled,
                None,
                Some(col_scale),
                Equilibrate::Column,
            )
        }
        Equilibrate::Both => {
            // First compute row scaling
            let mut row_scale = vec![T::one(); n];
            for i in 0..n {
                let mut row_max = T::zero();
                for j in 0..n {
                    let abs_val = Scalar::abs(a[(i, j)]);
                    if abs_val > row_max {
                        row_max = abs_val;
                    }
                }
                if row_max > T::zero() {
                    row_scale[i] = T::one() / row_max;
                }
            }

            // Apply row scaling to get A'
            let mut a_temp = Mat::zeros(n, n);
            for i in 0..n {
                let r = row_scale[i];
                for j in 0..n {
                    a_temp[(i, j)] = r * a[(i, j)];
                }
            }

            // Then compute column scaling on A'
            let mut col_scale = vec![T::one(); n];
            for j in 0..n {
                let mut col_max = T::zero();
                for i in 0..n {
                    let abs_val = Scalar::abs(a_temp[(i, j)]);
                    if abs_val > col_max {
                        col_max = abs_val;
                    }
                }
                if col_max > T::zero() {
                    col_scale[j] = T::one() / col_max;
                }
            }

            // Apply column scaling
            let mut a_scaled = Mat::zeros(n, n);
            let mut b_scaled = Mat::zeros(n, nrhs);
            for i in 0..n {
                for j in 0..n {
                    a_scaled[(i, j)] = col_scale[j] * a_temp[(i, j)];
                }
                for j in 0..nrhs {
                    b_scaled[(i, j)] = row_scale[i] * b[(i, j)];
                }
            }

            (
                a_scaled,
                b_scaled,
                Some(row_scale),
                Some(col_scale),
                Equilibrate::Both,
            )
        }
    }
}

/// Solve multiple right-hand sides using LU factorization.
fn solve_multiple_with_lu<T: Field + Real + bytemuck::Zeroable>(
    lu: &Lu<T>,
    b: MatRef<'_, T>,
    n: usize,
    nrhs: usize,
) -> Result<Mat<T>, ExpertSolveError> {
    let mut x = Mat::zeros(n, nrhs);

    for col in 0..nrhs {
        let mut b_col = Mat::zeros(n, 1);
        for i in 0..n {
            b_col[(i, 0)] = b[(i, col)];
        }

        let x_col = lu.solve(b_col.as_ref()).map_err(|e| match e {
            LuError::Singular { .. } => ExpertSolveError::SingularMatrix,
            _ => ExpertSolveError::DimensionMismatch,
        })?;

        for i in 0..n {
            x[(i, col)] = x_col[(i, 0)];
        }
    }

    Ok(x)
}

/// Estimate reciprocal condition number using 1-norm estimation.
///
/// This implements the Hager-Higham algorithm for estimating ||A^(-1)||_1.
fn estimate_rcond<T: Field + Real + bytemuck::Zeroable>(lu: &Lu<T>, anorm: T, n: usize) -> T {
    if anorm <= T::zero() || n == 0 {
        return T::zero();
    }

    // Hager-Higham 1-norm estimation for ||A^(-1)||_1
    // Start with x = e_1 (first unit vector)
    let mut x = Mat::zeros(n, 1);
    x[(0, 0)] = T::one();

    // Solve A*v = x
    let mut v = match lu.solve(x.as_ref()) {
        Ok(v) => v,
        Err(_) => return T::zero(),
    };

    // gamma = ||v||_1
    let mut gamma = T::zero();
    for i in 0..n {
        gamma = gamma + Scalar::abs(v[(i, 0)]);
    }

    let mut ainv_norm_est = gamma;

    for _iter in 0..5 {
        // xi = sign(v)
        let mut xi = Mat::zeros(n, 1);
        for i in 0..n {
            xi[(i, 0)] = if v[(i, 0)] >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
        }

        // Solve A^T * z = xi
        let z = match lu.solve_transpose(xi.as_ref()) {
            Ok(z) => z,
            Err(_) => break,
        };

        // Find j = argmax |z_i|
        let mut max_z = T::zero();
        let mut j = 0;
        for i in 0..n {
            let abs_z = Scalar::abs(z[(i, 0)]);
            if abs_z > max_z {
                max_z = abs_z;
                j = i;
            }
        }

        // Convergence check: if |z_j| <= z^T * xi, we're done
        let mut zt_xi = T::zero();
        for i in 0..n {
            zt_xi = zt_xi + z[(i, 0)] * xi[(i, 0)];
        }

        if max_z <= Scalar::abs(zt_xi) {
            break;
        }

        // x = e_j (unit vector with 1 at position j)
        for i in 0..n {
            x[(i, 0)] = T::zero();
        }
        x[(j, 0)] = T::one();

        // Solve A*v = x
        v = match lu.solve(x.as_ref()) {
            Ok(v) => v,
            Err(_) => break,
        };

        // gamma = ||v||_1
        gamma = T::zero();
        for i in 0..n {
            gamma = gamma + Scalar::abs(v[(i, 0)]);
        }

        if gamma > ainv_norm_est {
            ainv_norm_est = gamma;
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

/// Compute forward and backward error bounds.
fn compute_error_bounds<T: Field + Real + bytemuck::Zeroable>(
    a: &Mat<T>,
    b: &Mat<T>,
    x: &Mat<T>,
    lu: &Lu<T>,
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
        let e = match lu.solve(r.as_ref()) {
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

/// Unscale solution after column equilibration.
fn unscale_solution<T: Field + Real + bytemuck::Zeroable>(
    x: &Mat<T>,
    col_scale: &Option<Vec<T>>,
) -> Mat<T> {
    let n = x.nrows();
    let nrhs = x.ncols();

    match col_scale {
        Some(scale) => {
            let mut x_unscaled = Mat::zeros(n, nrhs);
            for i in 0..n {
                for j in 0..nrhs {
                    // x_true = C * x_scaled (scale = 1/C, so x_true = x_scaled / scale = x_scaled * (1/scale))
                    // But our scale is already the reciprocal, so just multiply
                    x_unscaled[(i, j)] = scale[i] * x[(i, j)];
                }
            }
            x_unscaled
        }
        None => {
            // No scaling, just copy
            let mut x_copy = Mat::zeros(n, nrhs);
            for i in 0..n {
                for j in 0..nrhs {
                    x_copy[(i, j)] = x[(i, j)];
                }
            }
            x_copy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_solve_expert_simple() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0], &[7.0]]);

        let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::None).unwrap();

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
    fn test_solve_expert_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::None).unwrap();

        // x should equal b for identity matrix
        for i in 0..3 {
            assert!(approx_eq(result.solution[(i, 0)], b[(i, 0)], 1e-10));
        }

        // rcond should be 1 for identity
        assert!(approx_eq(result.rcond, 1.0, 0.1));
    }

    #[test]
    fn test_solve_expert_with_row_equilibration() {
        // Matrix with row imbalance
        let a = Mat::from_rows(&[&[1000.0f64, 1.0], &[1.0, 1000.0]]);
        let b = Mat::from_rows(&[&[1001.0], &[1001.0]]);

        let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::Row).unwrap();

        // Verify Ax = b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!(approx_eq(ax0, b[(0, 0)], 1e-6));
        assert!(approx_eq(ax1, b[(1, 0)], 1e-6));

        // Row scaling should have been applied
        assert!(result.row_scale.is_some());
    }

    #[test]
    fn test_solve_expert_multiple_rhs() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0, 3.0], &[7.0, 5.0]]);

        let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::None).unwrap();

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
    fn test_solve_expert_singular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0]]);

        let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::None);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_expert_ill_conditioned() {
        // Hilbert-like ill-conditioned matrix
        let a = Mat::from_rows(&[
            &[1.0f64, 1.0 / 2.0, 1.0 / 3.0],
            &[1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0],
            &[1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
        ]);
        let b = Mat::from_rows(&[&[1.0], &[1.0], &[1.0]]);

        let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::None).unwrap();

        // rcond should be small (ill-conditioned)
        assert!(result.rcond < 0.01);
    }

    #[test]
    fn test_solve_expert_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0f32], &[7.0]]);

        let result = solve_expert(a.as_ref(), b.as_ref(), Equilibrate::None).unwrap();

        // Verify Ax ≈ b
        let x = &result.solution;
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!((ax0 - b[(0, 0)]).abs() < 1e-5);
        assert!((ax1 - b[(1, 0)]).abs() < 1e-5);
    }
}
