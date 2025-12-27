//! SYR2: Symmetric rank-2 update.
//!
//! Performs the symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
//! where A is a symmetric matrix stored in the specified triangle.

use oxiblas_core::scalar::Real;
use oxiblas_matrix::MatMut;

/// Specifies which triangle of the matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Syr2Uplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for symmetric rank-2 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Syr2Error {
    /// Matrix is not square.
    NotSquare,
    /// Dimension mismatch between matrix and vector x.
    DimensionMismatchX,
    /// Dimension mismatch between matrix and vector y.
    DimensionMismatchY,
}

impl core::fmt::Display for Syr2Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::DimensionMismatchX => {
                write!(f, "Dimension mismatch between matrix and vector x")
            }
            Self::DimensionMismatchY => {
                write!(f, "Dimension mismatch between matrix and vector y")
            }
        }
    }
}

impl std::error::Error for Syr2Error {}

/// Performs the symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// Only the specified triangle (upper or lower) of A is accessed and modified.
/// The other triangle is not referenced.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `alpha` - The scalar alpha
/// * `x` - The first vector
/// * `y` - The second vector
/// * `a` - The symmetric matrix A (only the specified triangle is used)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{syr2, Syr2Uplo};
/// use oxiblas_matrix::Mat;
///
/// // Start with a zero matrix
/// let mut a: Mat<f64> = Mat::zeros(3, 3);
/// let x = [1.0, 2.0, 3.0];
/// let y = [1.0, 0.0, 1.0];
///
/// // A = A + x * y^T + y * x^T (using lower triangle)
/// syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();
///
/// // A[0,0] = x[0]*y[0] + y[0]*x[0] = 1*1 + 1*1 = 2
/// assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
/// // A[1,0] = x[1]*y[0] + y[1]*x[0] = 2*1 + 0*1 = 2
/// assert!((a[(1, 0)] - 2.0).abs() < 1e-10);
/// ```
pub fn syr2<T: Real>(
    uplo: Syr2Uplo,
    alpha: T,
    x: &[T],
    y: &[T],
    a: MatMut<'_, T>,
) -> Result<(), Syr2Error> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(Syr2Error::NotSquare);
    }
    if n != x.len() {
        return Err(Syr2Error::DimensionMismatchX);
    }
    if n != y.len() {
        return Err(Syr2Error::DimensionMismatchY);
    }

    if n == 0 || alpha == T::zero() {
        return Ok(());
    }

    // For column-major storage: element (i,j) is at ptr + i + j*row_stride
    let a_ptr = a.as_ptr();
    let rs = a.row_stride();

    match uplo {
        Syr2Uplo::Lower => {
            // Update lower triangle: A[i,j] += alpha * (x[i]*y[j] + y[i]*x[j]) for i >= j
            for j in 0..n {
                let alpha_xj = alpha * x[j];
                let alpha_yj = alpha * y[j];
                for i in j..n {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        *ptr = *ptr + alpha_xj * y[i] + alpha_yj * x[i];
                    }
                }
            }
        }
        Syr2Uplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * (x[i]*y[j] + y[i]*x[j]) for i <= j
            for j in 0..n {
                let alpha_xj = alpha * x[j];
                let alpha_yj = alpha * y[j];
                for i in 0..=j {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        *ptr = *ptr + alpha_xj * y[i] + alpha_yj * x[i];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs the symmetric rank-2 update on a freshly allocated matrix.
///
/// Returns alpha * x * y^T + alpha * y * x^T as a fully symmetric matrix.
///
/// # Arguments
///
/// * `alpha` - The scalar alpha
/// * `x` - The first vector
/// * `y` - The second vector
///
/// # Returns
///
/// A new symmetric matrix containing alpha * (x * y^T + y * x^T).
///
/// # Errors
///
/// Returns `Syr2Error::DimensionMismatchY` if x and y have different lengths.
pub fn syr2_new<T: Real + bytemuck::Zeroable>(
    alpha: T,
    x: &[T],
    y: &[T],
) -> Result<oxiblas_matrix::Mat<T>, Syr2Error> {
    let n = x.len();
    if n != y.len() {
        return Err(Syr2Error::DimensionMismatchY);
    }

    let mut a = oxiblas_matrix::Mat::zeros(n, n);

    if n == 0 || alpha == T::zero() {
        return Ok(a);
    }

    // Create full symmetric matrix
    for i in 0..n {
        for j in 0..=i {
            let val = alpha * (x[i] * y[j] + y[i] * x[j]);
            a[(i, j)] = val;
            a[(j, i)] = val;
        }
    }

    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_syr2_lower_basic() {
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 1.0, 1.0];

        syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        // A[i,j] = x[i]*y[j] + y[i]*x[j]
        // A[0,0] = 1*1 + 1*1 = 2
        // A[1,0] = 2*1 + 1*1 = 3
        // A[1,1] = 2*1 + 1*2 = 4
        // A[2,0] = 3*1 + 1*1 = 4
        // A[2,1] = 3*1 + 1*2 = 5
        // A[2,2] = 3*1 + 1*3 = 6
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(2, 0)] - 4.0).abs() < 1e-10);
        assert!((a[(2, 1)] - 5.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 6.0).abs() < 1e-10);

        // Upper triangle should be unchanged (zero)
        assert!(a[(0, 1)].abs() < 1e-10);
        assert!(a[(0, 2)].abs() < 1e-10);
        assert!(a[(1, 2)].abs() < 1e-10);
    }

    #[test]
    fn test_syr2_upper_basic() {
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 1.0, 1.0];

        syr2(Syr2Uplo::Upper, 1.0, &x, &y, a.as_mut()).unwrap();

        // Upper triangle
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 3.0).abs() < 1e-10);
        assert!((a[(0, 2)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 2)] - 5.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 6.0).abs() < 1e-10);

        // Lower triangle should be unchanged
        assert!(a[(1, 0)].abs() < 1e-10);
        assert!(a[(2, 0)].abs() < 1e-10);
        assert!(a[(2, 1)].abs() < 1e-10);
    }

    #[test]
    fn test_syr2_with_alpha() {
        let mut a: Mat<f64> = Mat::zeros(2, 2);
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        syr2(Syr2Uplo::Lower, 2.0, &x, &y, a.as_mut()).unwrap();

        // A[i,j] = 2 * (x[i]*y[j] + y[i]*x[j])
        // A[0,0] = 2*(1*3 + 3*1) = 12
        // A[1,0] = 2*(2*3 + 4*1) = 20
        // A[1,1] = 2*(2*4 + 4*2) = 32
        assert!((a[(0, 0)] - 12.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 20.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2_same_vectors() {
        // When x == y, syr2 should give 2 * syr
        let mut a: Mat<f64> = Mat::zeros(2, 2);
        let x = [1.0, 2.0];

        syr2(Syr2Uplo::Lower, 1.0, &x, &x, a.as_mut()).unwrap();

        // A[i,j] = x[i]*x[j] + x[i]*x[j] = 2*x[i]*x[j]
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10); // 2*1*1
        assert!((a[(1, 0)] - 4.0).abs() < 1e-10); // 2*2*1
        assert!((a[(1, 1)] - 8.0).abs() < 1e-10); // 2*2*2
    }

    #[test]
    fn test_syr2_accumulate() {
        // Start with non-zero symmetric matrix
        let mut a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[2.0, 3.0, 0.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];

        syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        // Only A[1,0] should change: 2 + (0*1 + 1*1) = 3
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((a[(2, 0)] - 4.0).abs() < 1e-10);
        assert!((a[(2, 1)] - 5.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2_zero_alpha() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let original = a.clone();
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        syr2(Syr2Uplo::Lower, 0.0, &x, &y, a.as_mut()).unwrap();

        // Matrix unchanged
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - original[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_syr2_orthogonal_vectors() {
        // x and y orthogonal
        let mut a: Mat<f64> = Mat::zeros(2, 2);
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];

        syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        // A[0,0] = 1*0 + 0*1 = 0
        // A[1,0] = 0*0 + 1*1 = 1
        // A[1,1] = 0*1 + 1*0 = 0
        assert!(a[(0, 0)].abs() < 1e-10);
        assert!((a[(1, 0)] - 1.0).abs() < 1e-10);
        assert!(a[(1, 1)].abs() < 1e-10);
    }

    #[test]
    fn test_syr2_not_square() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        let result = syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut());
        assert!(matches!(result, Err(Syr2Error::NotSquare)));
    }

    #[test]
    fn test_syr2_dimension_mismatch_x() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let x = [1.0, 2.0, 3.0];
        let y = [3.0, 4.0];

        let result = syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut());
        assert!(matches!(result, Err(Syr2Error::DimensionMismatchX)));
    }

    #[test]
    fn test_syr2_dimension_mismatch_y() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let x = [1.0, 2.0];
        let y = [3.0, 4.0, 5.0];

        let result = syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut());
        assert!(matches!(result, Err(Syr2Error::DimensionMismatchY)));
    }

    #[test]
    fn test_syr2_empty() {
        let mut a: Mat<f64> = Mat::zeros(0, 0);
        let x: [f64; 0] = [];
        let y: [f64; 0] = [];

        syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();
    }

    #[test]
    fn test_syr2_1x1() {
        let mut a = Mat::from_rows(&[&[5.0f64]]);
        let x = [2.0];
        let y = [3.0];

        syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        // A[0,0] = 5 + (2*3 + 3*2) = 5 + 12 = 17
        assert!((a[(0, 0)] - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2_f32() {
        let mut a: Mat<f32> = Mat::zeros(2, 2);
        let x = [1.0f32, 2.0];
        let y = [3.0f32, 4.0];

        syr2(Syr2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        assert!((a[(0, 0)] - 6.0).abs() < 1e-5); // 1*3 + 3*1 = 6
        assert!((a[(1, 0)] - 10.0).abs() < 1e-5); // 2*3 + 4*1 = 10
        assert!((a[(1, 1)] - 16.0).abs() < 1e-5); // 2*4 + 4*2 = 16
    }

    #[test]
    fn test_syr2_negative_alpha() {
        let mut a: Mat<f64> = Mat::zeros(2, 2);
        let x = [1.0, 2.0];
        let y = [1.0, 1.0];

        syr2(Syr2Uplo::Lower, -1.0, &x, &y, a.as_mut()).unwrap();

        // A[i,j] = -1 * (x[i]*y[j] + y[i]*x[j])
        assert!((a[(0, 0)] + 2.0).abs() < 1e-10); // -(1*1 + 1*1)
        assert!((a[(1, 0)] + 3.0).abs() < 1e-10); // -(2*1 + 1*1)
        assert!((a[(1, 1)] + 4.0).abs() < 1e-10); // -(2*1 + 1*2)
    }

    #[test]
    fn test_syr2_new() {
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0];

        let a = syr2_new(1.0, &x, &y).unwrap();

        // Full symmetric matrix
        // A[0,0] = 1*3 + 3*1 = 6
        // A[0,1] = A[1,0] = 1*4 + 3*2 = 10
        // A[1,1] = 2*4 + 4*2 = 16
        assert!((a[(0, 0)] - 6.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 10.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 10.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2_new_dimension_mismatch() {
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0, 5.0];

        let result = syr2_new(1.0, &x, &y);
        assert!(matches!(result, Err(Syr2Error::DimensionMismatchY)));
    }

    #[test]
    fn test_syr2_new_with_alpha() {
        let x = [1.0f64, 1.0];
        let y = [1.0f64, 1.0];

        let a = syr2_new(2.0, &x, &y).unwrap();

        // A[i,j] = 2 * (x[i]*y[j] + y[i]*x[j]) = 2 * 2 = 4 for all elements
        assert!((a[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
    }
}
