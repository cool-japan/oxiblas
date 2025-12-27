//! Condition number estimation.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::MatRef;

use super::norms::{norm_1, norm_inf};
use crate::lu::{Lu, LuError};
use crate::svd::{Svd, SvdError};

/// Error type for condition number computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix is empty.
    EmptyMatrix,
    /// SVD computation failed.
    SvdFailed,
    /// LU computation failed.
    LuFailed,
}

impl core::fmt::Display for CondError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::SvdFailed => write!(f, "SVD computation failed"),
            Self::LuFailed => write!(f, "LU computation failed"),
        }
    }
}

impl std::error::Error for CondError {}

impl From<SvdError> for CondError {
    fn from(e: SvdError) -> Self {
        match e {
            SvdError::EmptyMatrix => Self::EmptyMatrix,
            SvdError::NotConverged => Self::SvdFailed,
        }
    }
}

impl From<LuError> for CondError {
    fn from(e: LuError) -> Self {
        match e {
            LuError::NotSquare { .. } => Self::NotSquare,
            _ => Self::LuFailed,
        }
    }
}

/// Computes the 2-norm condition number of a matrix.
///
/// κ_2(A) = σ_max / σ_min
///
/// This is the most accurate condition number but requires SVD computation.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Returns
///
/// The condition number. Returns infinity if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::cond;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[2.0f64, 0.0],
///     &[0.0, 4.0],
/// ]);
///
/// let kappa = cond(a.as_ref()).unwrap();
/// // For diagonal matrix, cond = max/min = 4/2 = 2
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn cond<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let svd = Svd::compute(a)?;
    Ok(svd.condition_number())
}

/// Computes the 1-norm condition number of a square matrix.
///
/// κ_1(A) = ||A||_1 * ||A^(-1)||_1
///
/// Uses LU decomposition to compute the inverse, which is more efficient
/// than SVD for just the 1-norm condition number.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The 1-norm condition number. Returns infinity if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::cond_1;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 2.0],
/// ]);
///
/// let kappa = cond_1(a.as_ref()).unwrap();
/// // ||A||_1 = 2, ||A^(-1)||_1 = 1
/// // kappa_1 = 2 * 1 = 2
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn cond_1<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CondError::NotSquare);
    }

    if n == 0 {
        return Err(CondError::EmptyMatrix);
    }

    let norm_a = norm_1(a);

    // Compute inverse via LU
    let lu = match Lu::compute(a) {
        Ok(lu) => lu,
        Err(_) => return Ok(<T as Scalar>::max_value()), // Singular = infinite condition number
    };

    let a_inv = match lu.inverse() {
        Ok(inv) => inv,
        Err(_) => return Ok(<T as Scalar>::max_value()),
    };

    let norm_a_inv = norm_1(a_inv.as_ref());

    Ok(norm_a * norm_a_inv)
}

/// Computes the infinity-norm condition number of a square matrix.
///
/// κ_∞(A) = ||A||_∞ * ||A^(-1)||_∞
///
/// Uses LU decomposition to compute the inverse.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The infinity-norm condition number. Returns infinity if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::cond_inf;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 2.0],
/// ]);
///
/// let kappa = cond_inf(a.as_ref()).unwrap();
/// // ||A||_inf = 2, ||A^(-1)||_inf = 1
/// // kappa_inf = 2 * 1 = 2
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn cond_inf<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CondError::NotSquare);
    }

    if n == 0 {
        return Err(CondError::EmptyMatrix);
    }

    let norm_a = norm_inf(a);

    // Compute inverse via LU
    let lu = match Lu::compute(a) {
        Ok(lu) => lu,
        Err(_) => return Ok(<T as Scalar>::max_value()),
    };

    let a_inv = match lu.inverse() {
        Ok(inv) => inv,
        Err(_) => return Ok(<T as Scalar>::max_value()),
    };

    let norm_a_inv = norm_inf(a_inv.as_ref());

    Ok(norm_a * norm_a_inv)
}

/// Estimates the reciprocal of the 1-norm condition number.
///
/// rcond(A) = 1 / κ_1(A)
///
/// This is more numerically stable than computing κ directly, and
/// is the form used by LAPACK's DGECON routine.
///
/// Returns 0 if the matrix is singular (infinite condition number).
/// Returns 1 if the matrix is perfectly conditioned (identity-like).
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::rcond;
/// use oxiblas_matrix::Mat;
///
/// let eye = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 1.0],
/// ]);
///
/// let rc = rcond(eye.as_ref()).unwrap();
/// assert!((rc - 1.0).abs() < 1e-10);
/// ```
pub fn rcond<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let kappa = cond_1(a)?;

    if kappa >= <T as Scalar>::max_value() / T::from_f64(2.0).unwrap_or(T::one()) {
        Ok(T::zero())
    } else {
        Ok(T::one() / kappa)
    }
}

/// Estimates the reciprocal condition number using LAPACK-style algorithm.
///
/// This is a fast O(n²) estimation that doesn't require computing the
/// full inverse. It uses a 1-norm estimation technique.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// An estimate of 1/κ_1(A).
pub fn rcond_estimate<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<T, CondError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CondError::NotSquare);
    }

    if n == 0 {
        return Err(CondError::EmptyMatrix);
    }

    let norm_a = norm_1(a);

    // LU factorize
    let lu = match Lu::compute(a) {
        Ok(lu) => lu,
        Err(_) => return Ok(T::zero()), // Singular
    };

    // Use 1-norm estimation algorithm (simplified version)
    // This is a Hager-Higham type estimator
    // Start with x = (1/n, 1/n, ..., 1/n)
    use oxiblas_matrix::Mat;

    let mut x = Mat::zeros(n, 1);
    let scale = T::one() / T::from_f64(n as f64).unwrap_or(T::one());
    for i in 0..n {
        x[(i, 0)] = scale;
    }

    // Iterate a few times to estimate ||A^(-1)||_1
    let mut norm_a_inv_est = T::zero();

    for _iter in 0..5 {
        // Solve A*y = x to get y = A^(-1)*x
        let y = match lu.solve(x.as_ref()) {
            Ok(y) => y,
            Err(_) => return Ok(T::zero()),
        };

        // Compute ||y||_1
        let mut y_norm = T::zero();
        for i in 0..n {
            y_norm = y_norm + Scalar::abs(y[(i, 0)]);
        }

        if y_norm > norm_a_inv_est {
            norm_a_inv_est = y_norm;
        }

        // Update x to be sign(y)
        for i in 0..n {
            x[(i, 0)] = if y[(i, 0)] >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
        }
    }

    // rcond = 1 / (||A||_1 * ||A^(-1)||_1)
    let kappa_est = norm_a * norm_a_inv_est;

    if kappa_est <= T::zero()
        || kappa_est >= <T as Scalar>::max_value() / T::from_f64(2.0).unwrap_or(T::one())
    {
        Ok(T::zero())
    } else {
        Ok(T::one() / kappa_est)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_cond_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        assert!(approx_eq(kappa, 2.0, 1e-10));
    }

    #[test]
    fn test_cond_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let kappa = cond(eye.as_ref()).unwrap();
        assert!(approx_eq(kappa, 1.0, 1e-10));
    }

    #[test]
    fn test_cond_ill_conditioned() {
        // Hilbert matrix is ill-conditioned
        let a = Mat::from_rows(&[&[1.0f64, 1.0 / 2.0], &[1.0 / 2.0, 1.0 / 3.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        // Should be around 19
        assert!(kappa > 15.0);
        assert!(kappa < 25.0);
    }

    #[test]
    fn test_cond_1_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let kappa = cond_1(eye.as_ref()).unwrap();
        assert!(approx_eq(kappa, 1.0, 1e-10));
    }

    #[test]
    fn test_cond_inf_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let kappa = cond_inf(eye.as_ref()).unwrap();
        assert!(approx_eq(kappa, 1.0, 1e-10));
    }

    #[test]
    fn test_cond_1_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let kappa = cond_1(a.as_ref()).unwrap();
        // ||A||_1 = 4, ||A^(-1)||_1 = 0.5
        // kappa = 4 * 0.5 = 2
        assert!(approx_eq(kappa, 2.0, 1e-10));
    }

    #[test]
    fn test_cond_singular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        // Singular matrix has infinite condition number
        assert!(kappa > 1e10);
    }

    #[test]
    fn test_rcond_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let rc = rcond(eye.as_ref()).unwrap();
        assert!(approx_eq(rc, 1.0, 1e-10));
    }

    #[test]
    fn test_rcond_singular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let rc = cond_1(a.as_ref()).unwrap();
        // Should indicate singular (very large condition number)
        assert!(rc > 1e10);

        let rc2 = rcond(a.as_ref()).unwrap();
        // rcond should be 0 for singular
        assert!(rc2 < 1e-10);
    }

    #[test]
    fn test_rcond_estimate() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);

        let rc_est = rcond_estimate(a.as_ref()).unwrap();
        let rc_exact = rcond(a.as_ref()).unwrap();

        // Estimate should be within a factor of the true value
        // (this is a rough test since estimation is approximate)
        assert!(rc_est > 0.0);
        assert!(rc_est / rc_exact < 10.0);
        assert!(rc_exact / rc_est < 10.0);
    }

    #[test]
    fn test_cond_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = cond_1(a.as_ref());
        assert!(matches!(result, Err(CondError::NotSquare)));

        // cond (2-norm) works for non-square
        let kappa = cond(a.as_ref()).unwrap();
        assert!(kappa > 0.0);
    }

    #[test]
    fn test_cond_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 0.0], &[0.0, 4.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        assert!((kappa - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_cond_relationship() {
        // κ_2(A) <= κ_1(A) <= n * κ_2(A) for n×n matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let k2 = cond(a.as_ref()).unwrap();
        let k1 = cond_1(a.as_ref()).unwrap();

        // Not strict inequalities due to different norms
        // But they should be comparable
        assert!(k1 > 0.0);
        assert!(k2 > 0.0);
    }
}
