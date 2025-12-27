//! Determinant computation.

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

use crate::lu::{Lu, LuError};

/// Error type for determinant computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix is singular.
    Singular,
}

impl core::fmt::Display for DetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::Singular => write!(f, "Matrix is singular"),
        }
    }
}

impl std::error::Error for DetError {}

impl From<LuError> for DetError {
    fn from(e: LuError) -> Self {
        match e {
            LuError::NotSquare { .. } => Self::NotSquare,
            LuError::Singular { .. } => Self::Singular,
            LuError::DimensionMismatch { .. } => Self::NotSquare,
        }
    }
}

/// Computes the determinant of a square matrix.
///
/// Uses LU decomposition internally. Returns an error if the matrix
/// is singular.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The determinant det(A).
///
/// # Errors
///
/// Returns `DetError::NotSquare` if the matrix is not square.
/// Returns `DetError::Singular` if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::det;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 7.0],
///     &[2.0, 6.0],
/// ]);
///
/// let d = det(a.as_ref()).unwrap();
/// assert!((d - 10.0).abs() < 1e-10); // det = 4*6 - 7*2 = 10
/// ```
pub fn det<T: Field + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, DetError> {
    let lu = Lu::compute(a)?;
    Ok(lu.determinant())
}

/// Computes the determinant using LU decomposition, returning the LU object as well.
///
/// This is useful when you need both the determinant and want to reuse the
/// LU decomposition for other operations (like solving systems).
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// A tuple of (determinant, LU decomposition).
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::det_lu;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[2.0f64, 1.0],
///     &[1.0, 3.0],
/// ]);
///
/// let (d, lu) = det_lu(a.as_ref()).unwrap();
/// // Can reuse lu for solving systems
/// let b = Mat::from_rows(&[&[5.0], &[7.0]]);
/// let x = lu.solve(b.as_ref()).unwrap();
/// ```
pub fn det_lu<T: Field + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<(T, Lu<T>), DetError> {
    let lu = Lu::compute(a)?;
    let d = lu.determinant();
    Ok((d, lu))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_det_2x2() {
        let a = Mat::from_rows(&[&[4.0f64, 7.0], &[2.0, 6.0]]);

        let d = det(a.as_ref()).unwrap();
        assert!(approx_eq(d, 10.0, 1e-10));
    }

    #[test]
    fn test_det_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let d = det(a.as_ref()).unwrap();
        // det = 1*(5*10-6*8) - 2*(4*10-6*7) + 3*(4*8-5*7)
        //     = 1*(50-48) - 2*(40-42) + 3*(32-35)
        //     = 2 + 4 - 9 = -3
        assert!(approx_eq(d, -3.0, 1e-10));
    }

    #[test]
    fn test_det_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let d = det(eye.as_ref()).unwrap();
        assert!(approx_eq(d, 1.0, 1e-10));
    }

    #[test]
    fn test_det_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 4.0]]);

        let d = det(a.as_ref()).unwrap();
        assert!(approx_eq(d, 24.0, 1e-10));
    }

    #[test]
    fn test_det_singular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let result = det(a.as_ref());
        assert!(matches!(result, Err(DetError::Singular)));
    }

    #[test]
    fn test_det_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = det(a.as_ref());
        assert!(matches!(result, Err(DetError::NotSquare)));
    }

    #[test]
    fn test_det_negative() {
        // Row swap changes sign
        let a = Mat::from_rows(&[&[0.0f64, 1.0], &[1.0, 0.0]]);

        let d = det(a.as_ref()).unwrap();
        assert!(approx_eq(d, -1.0, 1e-10));
    }

    #[test]
    fn test_det_lu_reuse() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);

        let (d, lu) = det_lu(a.as_ref()).unwrap();

        // det = 2*3 - 1*1 = 5
        assert!(approx_eq(d, 5.0, 1e-10));

        // Use LU to solve system
        let b = Mat::from_rows(&[&[5.0], &[7.0]]);
        let x = lu.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 2.0 * x[(0, 0)] + 1.0 * x[(1, 0)];
        let ax1 = 1.0 * x[(0, 0)] + 3.0 * x[(1, 0)];
        assert!(approx_eq(ax0, 5.0, 1e-10));
        assert!(approx_eq(ax1, 7.0, 1e-10));
    }

    #[test]
    fn test_det_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 7.0], &[2.0, 6.0]]);

        let d = det(a.as_ref()).unwrap();
        assert!((d - 10.0).abs() < 1e-5);
    }
}
