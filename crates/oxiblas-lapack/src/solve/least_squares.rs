//! Least squares solvers.
//!
//! Solves overdetermined systems Ax ≈ b in the least squares sense,
//! minimizing ||Ax - b||₂.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::qr::{Qr, QrError};

/// Error type for least squares solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LstSqError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Dimension mismatch between A and b.
    DimensionMismatch,
    /// Matrix has more columns than rows (underdetermined).
    Underdetermined,
    /// QR decomposition failed.
    QrFailed,
}

impl core::fmt::Display for LstSqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::DimensionMismatch => write!(f, "Matrix and vector dimensions do not match"),
            Self::Underdetermined => {
                write!(f, "System is underdetermined (more columns than rows)")
            }
            Self::QrFailed => write!(f, "QR decomposition failed"),
        }
    }
}

impl std::error::Error for LstSqError {}

impl From<QrError> for LstSqError {
    fn from(_: QrError) -> Self {
        Self::QrFailed
    }
}

/// Result of least squares computation.
#[derive(Debug, Clone)]
pub struct LeastSquaresResult<T: Scalar> {
    /// Solution vector x that minimizes ||Ax - b||₂.
    pub solution: Mat<T>,
    /// Residual vector r = b - Ax.
    pub residual: Mat<T>,
    /// Squared norm of residual ||r||₂².
    pub residual_norm_sq: T,
    /// Effective rank of A (number of significant singular values).
    pub rank: usize,
}

/// Solves the least squares problem: minimize ||Ax - b||₂.
///
/// Uses QR decomposition: A = QR, then solves Rx = Q^T b.
///
/// # Arguments
///
/// * `a` - Coefficient matrix A (m×n where m ≥ n)
/// * `b` - Right-hand side vector b (m×1)
///
/// # Returns
///
/// `LeastSquaresResult` containing the solution and residual information.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::lstsq;
/// use oxiblas_matrix::Mat;
///
/// // Overdetermined system: 3 equations, 2 unknowns
/// let a = Mat::from_rows(&[
///     &[1.0f64, 1.0],
///     &[1.0, 2.0],
///     &[1.0, 3.0],
/// ]);
/// let b = Mat::from_rows(&[&[6.0], &[8.0], &[10.0]]);
///
/// let result = lstsq(a.as_ref(), b.as_ref()).unwrap();
/// let x = result.solution;
/// // x ≈ [4, 2] (best fit line y = 4 + 2x for points (1,6), (2,8), (3,10))
/// ```
pub fn lstsq<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<LeastSquaresResult<T>, LstSqError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(LstSqError::EmptyMatrix);
    }

    if b.nrows() != m {
        return Err(LstSqError::DimensionMismatch);
    }

    if m < n {
        return Err(LstSqError::Underdetermined);
    }

    let eps = <T as Scalar>::epsilon();

    // Compute QR decomposition
    let qr = Qr::compute(a)?;
    let q = qr.q();
    let r = qr.r();

    // Compute Q^T * b
    let mut qt_b = Mat::zeros(m, b.ncols());
    for i in 0..m {
        for j in 0..b.ncols() {
            let mut sum = T::zero();
            for k in 0..m {
                sum = sum + q[(k, i)] * b[(k, j)];
            }
            qt_b[(i, j)] = sum;
        }
    }

    // Determine effective rank by checking R's diagonal
    let mut rank = n;
    let r_max = if n > 0 {
        Scalar::abs(r[(0, 0)])
    } else {
        T::one()
    };
    let tol = eps * T::from_f64(m.max(n) as f64).unwrap() * r_max;

    for i in 0..n {
        if Scalar::abs(r[(i, i)]) < tol {
            rank = i;
            break;
        }
    }

    // Solve R_upper * x = (Q^T * b)_upper by back substitution
    // R_upper is the n×n upper triangular part of R
    let mut solution = Mat::zeros(n, b.ncols());

    for col in 0..b.ncols() {
        for i in (0..rank).rev() {
            let mut sum = qt_b[(i, col)];
            for j in (i + 1)..rank {
                sum = sum - r[(i, j)] * solution[(j, col)];
            }

            if Scalar::abs(r[(i, i)]) > eps {
                solution[(i, col)] = sum / r[(i, i)];
            }
        }
    }

    // Compute residual r = b - A * x
    let mut residual = Mat::zeros(m, b.ncols());
    let mut residual_norm_sq = T::zero();

    for col in 0..b.ncols() {
        for i in 0..m {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a[(i, j)] * solution[(j, col)];
            }
            residual[(i, col)] = b[(i, col)] - ax_i;
            residual_norm_sq = residual_norm_sq + residual[(i, col)] * residual[(i, col)];
        }
    }

    Ok(LeastSquaresResult {
        solution,
        residual,
        residual_norm_sq,
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_lstsq_exact() {
        // Square system - should be exact
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0]]);
        let b = Mat::from_rows(&[&[3.0], &[5.0]]);

        let result = lstsq(a.as_ref(), b.as_ref()).unwrap();
        let x = &result.solution;

        // x = [1, 2]
        assert!(approx_eq(x[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
        assert!(result.residual_norm_sq < 1e-10);
    }

    #[test]
    fn test_lstsq_overdetermined() {
        // Overdetermined system - points on a line
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0], &[8.0], &[10.0]]);

        let result = lstsq(a.as_ref(), b.as_ref()).unwrap();
        let x = &result.solution;

        // Line y = 4 + 2x fits perfectly
        assert!(approx_eq(x[(0, 0)], 4.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
        assert!(result.residual_norm_sq < 1e-10);
    }

    #[test]
    fn test_lstsq_with_residual() {
        // Points not exactly on a line
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0], &[8.0], &[11.0]]); // 11 instead of 10

        let result = lstsq(a.as_ref(), b.as_ref()).unwrap();

        // Should have some residual
        assert!(result.residual_norm_sq > 0.0);

        // Verify residual computation
        let mut manual_residual_sq = 0.0;
        for i in 0..3 {
            let ax_i = a[(i, 0)] * result.solution[(0, 0)] + a[(i, 1)] * result.solution[(1, 0)];
            let r_i = b[(i, 0)] - ax_i;
            manual_residual_sq += r_i * r_i;
        }
        assert!(approx_eq(
            result.residual_norm_sq,
            manual_residual_sq,
            1e-10
        ));
    }

    #[test]
    fn test_lstsq_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0], &[0.0, 0.0]]);
        let b = Mat::from_rows(&[&[3.0], &[4.0], &[0.0]]);

        let result = lstsq(a.as_ref(), b.as_ref()).unwrap();

        assert!(approx_eq(result.solution[(0, 0)], 3.0, 1e-10));
        assert!(approx_eq(result.solution[(1, 0)], 4.0, 1e-10));
    }

    #[test]
    fn test_lstsq_underdetermined() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0]]);

        let result = lstsq(a.as_ref(), b.as_ref());
        assert!(matches!(result, Err(LstSqError::Underdetermined)));
    }

    #[test]
    fn test_lstsq_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0]]);
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let result = lstsq(a.as_ref(), b.as_ref());
        assert!(matches!(result, Err(LstSqError::DimensionMismatch)));
    }

    #[test]
    fn test_lstsq_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0f32], &[8.0], &[10.0]]);

        let result = lstsq(a.as_ref(), b.as_ref()).unwrap();

        assert!((result.solution[(0, 0)] - 4.0).abs() < 1e-4);
        assert!((result.solution[(1, 0)] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_lstsq_multiple_rhs() {
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[6.0, 3.0], &[8.0, 5.0], &[10.0, 7.0]]);

        let result = lstsq(a.as_ref(), b.as_ref()).unwrap();

        // First column: y = 4 + 2x
        assert!(approx_eq(result.solution[(0, 0)], 4.0, 1e-10));
        assert!(approx_eq(result.solution[(1, 0)], 2.0, 1e-10));

        // Second column: y = 1 + 2x
        assert!(approx_eq(result.solution[(0, 1)], 1.0, 1e-10));
        assert!(approx_eq(result.solution[(1, 1)], 2.0, 1e-10));
    }
}
