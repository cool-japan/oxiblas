//! Matrix inverse and pseudoinverse computation.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::lu::{Lu, LuError};
use crate::svd::{Svd, SvdError};

/// Error type for matrix inversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix is singular.
    Singular,
}

impl core::fmt::Display for InvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::Singular => write!(f, "Matrix is singular"),
        }
    }
}

impl std::error::Error for InvError {}

impl From<LuError> for InvError {
    fn from(e: LuError) -> Self {
        match e {
            LuError::NotSquare { .. } => Self::NotSquare,
            LuError::Singular { .. } => Self::Singular,
            LuError::DimensionMismatch { .. } => Self::NotSquare,
        }
    }
}

/// Result of pseudoinverse computation.
#[derive(Debug, Clone)]
pub struct PinvResult<T: Scalar> {
    /// The pseudoinverse matrix A^+.
    pub pinv: Mat<T>,
    /// Numerical rank used in computation.
    pub rank: usize,
    /// Singular values (for diagnostics).
    pub singular_values: Vec<T>,
}

/// Computes the inverse of a square matrix.
///
/// Uses LU decomposition with partial pivoting.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The inverse matrix A^(-1) such that A * A^(-1) = I.
///
/// # Errors
///
/// Returns `InvError::NotSquare` if the matrix is not square.
/// Returns `InvError::Singular` if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::inv;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 7.0],
///     &[2.0, 6.0],
/// ]);
///
/// let a_inv = inv(a.as_ref()).unwrap();
///
/// // Verify: A * A^(-1) ≈ I
/// // det(A) = 10, so A^(-1) = [0.6, -0.7; -0.2, 0.4]
/// assert!((a_inv[(0, 0)] - 0.6).abs() < 1e-10);
/// assert!((a_inv[(0, 1)] + 0.7).abs() < 1e-10);
/// ```
pub fn inv<T: Field + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<Mat<T>, InvError> {
    let lu = Lu::compute(a)?;
    Ok(lu.inverse()?)
}

/// Computes the Moore-Penrose pseudoinverse of a matrix.
///
/// Uses SVD decomposition: A^+ = V * Σ^+ * U^T
/// where Σ^+ has 1/σ_i on the diagonal for non-zero singular values.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n, can be rectangular)
///
/// # Returns
///
/// `PinvResult` containing the pseudoinverse and rank information.
///
/// # Properties
///
/// The pseudoinverse satisfies:
/// 1. A * A^+ * A = A
/// 2. A^+ * A * A^+ = A^+
/// 3. (A * A^+)^T = A * A^+
/// 4. (A^+ * A)^T = A^+ * A
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::pinv;
/// use oxiblas_matrix::Mat;
///
/// // Tall matrix (overdetermined system)
/// let a = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 1.0],
///     &[1.0, 1.0],
/// ]);
///
/// let result = pinv(a.as_ref(), 1e-10).unwrap();
/// let a_pinv = result.pinv;
///
/// // A^+ has shape 2×3
/// assert_eq!(a_pinv.nrows(), 2);
/// assert_eq!(a_pinv.ncols(), 3);
/// ```
pub fn pinv<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: T,
) -> Result<PinvResult<T>, SvdError> {
    let svd = Svd::compute(a)?;
    let pinv_mat = svd.pseudoinverse(tol);
    let rank = svd.rank(tol);
    let singular_values = svd.singular_values().to_vec();

    Ok(PinvResult {
        pinv: pinv_mat,
        rank,
        singular_values,
    })
}

/// Computes the pseudoinverse using a default tolerance.
///
/// The default tolerance is `eps * max(m, n) * σ_max` where `eps` is machine
/// epsilon and `σ_max` is the largest singular value.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::inv;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let a_inv = inv(a.as_ref()).unwrap();
/// ```
pub fn pinv_default<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<PinvResult<T>, SvdError> {
    let svd = Svd::compute(a)?;
    let m = a.nrows();
    let n = a.ncols();

    // Default tolerance: eps * max(m, n) * sigma_max
    let eps = <T as Scalar>::epsilon();
    let sigma_max = if svd.singular_values().is_empty() {
        T::one()
    } else {
        svd.singular_values()[0]
    };
    let tol = eps * T::from_f64(m.max(n) as f64).unwrap_or(T::one()) * sigma_max;

    let pinv_mat = svd.pseudoinverse(tol);
    let rank = svd.rank(tol);
    let singular_values = svd.singular_values().to_vec();

    Ok(PinvResult {
        pinv: pinv_mat,
        rank,
        singular_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_inv_2x2() {
        let a = Mat::from_rows(&[&[4.0f64, 7.0], &[2.0, 6.0]]);

        let a_inv = inv(a.as_ref()).unwrap();

        // det(A) = 10, so A^(-1) = [6/10, -7/10; -2/10, 4/10]
        assert!(approx_eq(a_inv[(0, 0)], 0.6, 1e-10));
        assert!(approx_eq(a_inv[(0, 1)], -0.7, 1e-10));
        assert!(approx_eq(a_inv[(1, 0)], -0.2, 1e-10));
        assert!(approx_eq(a_inv[(1, 1)], 0.4, 1e-10));
    }

    #[test]
    fn test_inv_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 1.0, 4.0], &[5.0, 6.0, 0.0]]);

        let a_inv = inv(a.as_ref()).unwrap();

        // Verify A * A^(-1) = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += a[(i, k)] * a_inv[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-9),
                    "(A*A^-1)[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_inv_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let inv_eye = inv(eye.as_ref()).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(inv_eye[(i, j)], expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_inv_singular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let result = inv(a.as_ref());
        assert!(matches!(result, Err(InvError::Singular)));
    }

    #[test]
    fn test_inv_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = inv(a.as_ref());
        assert!(matches!(result, Err(InvError::NotSquare)));
    }

    #[test]
    fn test_pinv_square() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);

        let result = pinv(a.as_ref(), 1e-10).unwrap();

        // For diagonal matrix, pinv is element-wise reciprocal
        assert!(approx_eq(result.pinv[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(result.pinv[(1, 1)], 0.5, 1e-10));
        assert!(approx_eq(result.pinv[(0, 1)], 0.0, 1e-10));
        assert!(approx_eq(result.pinv[(1, 0)], 0.0, 1e-10));
    }

    #[test]
    fn test_pinv_tall() {
        // Tall matrix (3×2)
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0], &[0.0, 0.0]]);

        let result = pinv(a.as_ref(), 1e-10).unwrap();

        // Pinv should be 2×3
        assert_eq!(result.pinv.nrows(), 2);
        assert_eq!(result.pinv.ncols(), 3);

        // Verify A * A^+ * A = A
        let mut product = Mat::zeros(3, 2);
        // First: A * A^+
        let mut temp = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += a[(i, k)] * result.pinv[(k, j)];
                }
                temp[(i, j)] = sum;
            }
        }
        // Then: (A * A^+) * A
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += temp[(i, k)] * a[(k, j)];
                }
                product[(i, j)] = sum;
            }
        }

        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    approx_eq(product[(i, j)], a[(i, j)], 1e-9),
                    "A*A^+*A[{},{}] = {}, expected {}",
                    i,
                    j,
                    product[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_pinv_wide() {
        // Wide matrix (2×3)
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0]]);

        let result = pinv(a.as_ref(), 1e-10).unwrap();

        // Pinv should be 3×2
        assert_eq!(result.pinv.nrows(), 3);
        assert_eq!(result.pinv.ncols(), 2);
    }

    #[test]
    fn test_pinv_rank_deficient() {
        // Rank 1 matrix (3×3)
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]);

        let result = pinv(a.as_ref(), 1e-10).unwrap();

        // Should detect rank 1
        assert_eq!(result.rank, 1);

        // Verify A * A^+ * A ≈ A
        let mut product = Mat::zeros(3, 3);
        let mut temp = Mat::zeros(3, 3);

        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += a[(i, k)] * result.pinv[(k, j)];
                }
                temp[(i, j)] = sum;
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += temp[(i, k)] * a[(k, j)];
                }
                product[(i, j)] = sum;
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(product[(i, j)], a[(i, j)], 1e-9),
                    "A*A^+*A[{},{}] = {}, expected {}",
                    i,
                    j,
                    product[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_inv_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 7.0], &[2.0, 6.0]]);

        let a_inv = inv(a.as_ref()).unwrap();

        assert!((a_inv[(0, 0)] - 0.6).abs() < 1e-5);
        assert!((a_inv[(0, 1)] + 0.7).abs() < 1e-5);
    }
}
