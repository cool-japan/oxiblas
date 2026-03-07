//! General linear system solvers.
//!
//! Solves Ax = b for general (non-triangular) matrices using LU decomposition.

use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatRef};

use crate::lu::{Lu, LuError};

/// Error type for general system solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix and vector dimensions don't match.
    DimensionMismatch,
    /// Matrix is singular.
    SingularMatrix,
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::DimensionMismatch => write!(f, "Matrix and vector dimensions do not match"),
            Self::SingularMatrix => write!(f, "Matrix is singular"),
        }
    }
}

impl std::error::Error for SolveError {}

impl From<LuError> for SolveError {
    fn from(e: LuError) -> Self {
        match e {
            LuError::Singular { .. } => Self::SingularMatrix,
            LuError::NotSquare { .. } => Self::NotSquare,
            LuError::DimensionMismatch { .. } => Self::DimensionMismatch,
        }
    }
}

/// Solves a general linear system Ax = b.
///
/// Uses LU decomposition with partial pivoting.
///
/// # Arguments
///
/// * `a` - Square coefficient matrix A (n×n)
/// * `b` - Right-hand side vector b (n×1)
///
/// # Returns
///
/// Solution vector x such that Ax = b.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::solve;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[2.0f64, 1.0],
///     &[1.0, 3.0],
/// ]);
/// let b = Mat::from_rows(&[&[5.0], &[7.0]]);
///
/// let x = solve(a.as_ref(), b.as_ref()).unwrap();
/// // Check: Ax ≈ b
/// ```
pub fn solve<T: Field + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, SolveError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SolveError::NotSquare);
    }
    if b.nrows() != n {
        return Err(SolveError::DimensionMismatch);
    }

    // Compute LU decomposition
    let lu = Lu::compute(a)?;

    // Solve using LU
    let x = lu.solve(b)?;

    Ok(x)
}

/// Solves a general linear system AX = B for multiple right-hand sides.
///
/// Uses LU decomposition with partial pivoting.
///
/// # Arguments
///
/// * `a` - Square coefficient matrix A (n×n)
/// * `b` - Right-hand side matrix B (n×m)
///
/// # Returns
///
/// Solution matrix X (n×m) such that AX = B.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::solve_multiple;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[2.0f64, 1.0],
///     &[1.0, 3.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[5.0, 3.0],
///     &[7.0, 5.0],
/// ]);
///
/// let x = solve_multiple(a.as_ref(), b.as_ref()).unwrap();
/// // X has shape 2×2
/// ```
pub fn solve_multiple<T: Field + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, SolveError> {
    let n = a.nrows();
    let m = b.ncols();

    if a.ncols() != n {
        return Err(SolveError::NotSquare);
    }
    if b.nrows() != n {
        return Err(SolveError::DimensionMismatch);
    }

    // Compute LU decomposition once
    let lu = Lu::compute(a)?;

    // Solve for each column
    let mut x = Mat::zeros(n, m);

    for col in 0..m {
        // Extract column as a single-column matrix
        let mut b_col = Mat::zeros(n, 1);
        for i in 0..n {
            b_col[(i, 0)] = b[(i, col)];
        }

        // Solve
        let x_col = lu.solve(b_col.as_ref())?;

        // Copy result
        for i in 0..n {
            x[(i, col)] = x_col[(i, 0)];
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_solve_2x2() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0], &[7.0]]);

        let x = solve(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!(approx_eq(ax0, b[(0, 0)], 1e-10));
        assert!(approx_eq(ax1, b[(1, 0)], 1e-10));
    }

    #[test]
    fn test_solve_3x3() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 10.0], // Not singular
        ]);
        let b = Mat::from_rows(&[&[14.0], &[32.0], &[50.0]]);

        let x = solve(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax ≈ b
        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a[(i, j)] * x[(j, 0)];
            }
            assert!(approx_eq(ax_i, b[(i, 0)], 1e-8));
        }
    }

    #[test]
    fn test_solve_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let x = solve(a.as_ref(), b.as_ref()).unwrap();

        // x should equal b
        for i in 0..3 {
            assert!(approx_eq(x[(i, 0)], b[(i, 0)], 1e-10));
        }
    }

    #[test]
    fn test_solve_multiple_rhs() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0, 3.0], &[7.0, 5.0]]);

        let x = solve_multiple(a.as_ref(), b.as_ref()).unwrap();

        // Verify AX = B
        for col in 0..2 {
            for i in 0..2 {
                let mut ax_i = 0.0;
                for j in 0..2 {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                assert!(approx_eq(ax_i, b[(i, col)], 1e-10));
            }
        }
    }

    #[test]
    fn test_solve_singular() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0],
            &[2.0, 4.0], // Singular
        ]);
        let b = Mat::from_rows(&[&[1.0], &[2.0]]);

        let result = solve(a.as_ref(), b.as_ref());
        assert!(matches!(result, Err(SolveError::SingularMatrix)));
    }

    #[test]
    fn test_solve_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]); // Wrong size

        let result = solve(a.as_ref(), b.as_ref());
        assert!(matches!(result, Err(SolveError::DimensionMismatch)));
    }

    #[test]
    fn test_solve_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0f32], &[7.0]]);

        let x = solve(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax ≈ b
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];
        assert!((ax0 - b[(0, 0)]).abs() < 1e-5);
        assert!((ax1 - b[(1, 0)]).abs() < 1e-5);
    }
}
