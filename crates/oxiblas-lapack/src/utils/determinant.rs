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
/// Uses LU decomposition internally. A singular (but square) matrix is
/// mathematically guaranteed to have determinant zero, so this function
/// returns `Ok(0.0)` for singular input rather than an error — matching
/// the behavior of reference LAPACK/numpy-style consumers, which rely on
/// `det(a) == 0` as the standard idiom for testing singularity.
///
/// Internally, `Lu::compute` reports a zero/near-zero pivot as
/// [`LuError::Singular`]. Rather than propagating that as a hard failure,
/// this function treats it as the honest numerical answer: a zero pivot
/// in Gaussian elimination means the (partially reduced) matrix has a
/// linearly dependent column, so the determinant of that submatrix — and
/// therefore of the original matrix — is exactly zero, without needing to
/// complete the rest of the factorization.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The determinant det(A). This is exactly `0.0` for singular matrices.
///
/// # Errors
///
/// Returns `DetError::NotSquare` if the matrix is not square. This is the
/// only error case: singular-but-square input yields `Ok(0.0)`.
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
///
/// // Singular matrices yield 0.0, not an error.
/// let singular = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[2.0, 4.0],
/// ]);
/// assert_eq!(det(singular.as_ref()).unwrap(), 0.0);
/// ```
pub fn det<T: Field + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, DetError> {
    match Lu::compute(a) {
        Ok(lu) => Ok(lu.determinant()),
        // A zero/near-zero pivot means the matrix (or the remaining Schur
        // complement) has a linearly dependent column, so the determinant
        // is exactly zero by construction. This is the correct, honest
        // value — not an error condition — for a square matrix.
        Err(LuError::Singular { .. }) => Ok(T::zero()),
        // Genuine structural errors (non-square input, etc.) still propagate.
        Err(e) => Err(e.into()),
    }
}

/// Computes the determinant using LU decomposition, returning the LU object as well.
///
/// This is useful when you need both the determinant and want to reuse the
/// LU decomposition for other operations (like solving systems).
///
/// Unlike [`det`], this function still returns `Err(DetError::Singular)` for
/// singular-but-square input: a singular matrix has no usable LU
/// factorization to hand back (forward/back substitution would divide by a
/// zero pivot), so there is no meaningful `Lu<T>` this function could return
/// alongside the determinant. Callers that only need the determinant value
/// (and want `0.0` rather than an error for singular input) should use
/// [`det`] instead.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// A tuple of (determinant, LU decomposition).
///
/// # Errors
///
/// Returns `DetError::NotSquare` if the matrix is not square.
/// Returns `DetError::Singular` if the matrix is singular (no reusable LU
/// factorization is available in that case).
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
    fn test_det_singular_proportional_rows() {
        // Row 2 = 2 * Row 1: linearly dependent rows, exactly singular.
        // det() must return the mathematically correct value 0.0, not an
        // error -- this is the standard idiom (numpy/LAPACK-style) for
        // testing singularity via the determinant.
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let d = det(a.as_ref()).expect("det() must not error on singular square input");
        assert_eq!(d, 0.0, "singular matrix must yield det == 0.0 exactly");
    }

    #[test]
    fn test_det_singular_identical_rows() {
        // Two exactly identical rows (rows 0 and 1) force a pivot to hit
        // exactly zero *after* a row swap has already occurred during
        // elimination, exercising the code path where the zero pivot shows
        // up in the interior of the factorization rather than trivially at
        // the first step.
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[1.0, 2.0, 3.0], &[4.0, 5.0, 7.0]]);

        let d = det(a.as_ref()).expect("det() must not error on singular square input");
        assert_eq!(
            d, 0.0,
            "two identical rows must yield det == 0.0 exactly, got {d}"
        );
    }

    #[test]
    fn test_det_small_magnitude_well_conditioned_not_falsely_singular() {
        // 1e-8 * I is perfectly well-conditioned (condition number 1) despite
        // every entry being tiny. It must NOT be treated as singular, and
        // det() must return the true (small but nonzero) value rather than
        // clamping to 0.0.
        let scale = 1.0e-8f64;
        let a = Mat::from_rows(&[&[scale, 0.0, 0.0], &[0.0, scale, 0.0], &[0.0, 0.0, scale]]);

        let d = det(a.as_ref()).expect("well-conditioned small-magnitude matrix must not error");
        let expected = scale * scale * scale;
        assert!(
            d != 0.0,
            "small-magnitude well-conditioned matrix must not be falsely flagged as det == 0"
        );
        assert!(
            ((d - expected) / expected).abs() < 1e-9,
            "det = {d}, expected ~= {expected}"
        );
    }

    #[test]
    fn test_det_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = det(a.as_ref());
        assert!(matches!(result, Err(DetError::NotSquare)));
    }

    #[test]
    fn test_det_lu_still_errors_on_singular() {
        // det_lu() must keep erroring on singular input: it hands back a
        // usable LU factorization for reuse (e.g. solve()), and no such
        // factorization exists for a singular matrix.
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let result = det_lu(a.as_ref());
        assert!(matches!(result, Err(DetError::Singular)));
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
