//! SYR: Symmetric rank-1 update.
//!
//! Performs the symmetric rank-1 update: A = alpha * x * x^T + A
//! where A is a symmetric matrix stored in the specified triangle.

use oxiblas_core::scalar::Real;
use oxiblas_matrix::MatMut;

/// Specifies which triangle of the matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyrUplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for symmetric rank-1 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyrError {
    /// Matrix is not square.
    NotSquare,
    /// Dimension mismatch between matrix and vector.
    DimensionMismatch,
}

impl core::fmt::Display for SyrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between matrix and vector"),
        }
    }
}

impl std::error::Error for SyrError {}

/// Performs the symmetric rank-1 update: A = alpha * x * x^T + A
///
/// Only the specified triangle (upper or lower) of A is accessed and modified.
/// The other triangle is not referenced.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `alpha` - The scalar alpha
/// * `x` - The vector x
/// * `a` - The symmetric matrix A (only the specified triangle is used)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{syr, SyrUplo};
/// use oxiblas_matrix::Mat;
///
/// // Start with a symmetric matrix
/// let mut a = Mat::from_rows(&[
///     &[1.0f64, 2.0, 3.0],
///     &[2.0, 4.0, 5.0],
///     &[3.0, 5.0, 6.0],
/// ]);
/// let x = [1.0, 2.0, 3.0];
///
/// // A = A + x * x^T (using lower triangle)
/// syr(SyrUplo::Lower, 1.0, &x, a.as_mut()).unwrap();
///
/// // Lower triangle updated: A[i,j] += x[i] * x[j] for i >= j
/// // A[0,0] += 1*1 = 1, A[1,0] += 2*1 = 2, A[1,1] += 2*2 = 4, etc.
/// assert!((a[(0, 0)] - 2.0).abs() < 1e-10);  // 1 + 1*1
/// assert!((a[(1, 0)] - 4.0).abs() < 1e-10);  // 2 + 2*1
/// assert!((a[(1, 1)] - 8.0).abs() < 1e-10);  // 4 + 2*2
/// ```
pub fn syr<T: Real>(uplo: SyrUplo, alpha: T, x: &[T], a: MatMut<'_, T>) -> Result<(), SyrError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(SyrError::NotSquare);
    }
    if n != x.len() {
        return Err(SyrError::DimensionMismatch);
    }

    if n == 0 || alpha == T::zero() {
        return Ok(());
    }

    // Safety: we need mutable access
    // For column-major storage: element (i,j) is at ptr + i + j*row_stride
    let a_ptr = a.as_ptr();
    let rs = a.row_stride();

    match uplo {
        SyrUplo::Lower => {
            // Update lower triangle: A[i,j] += alpha * x[i] * x[j] for i >= j
            for j in 0..n {
                let alpha_xj = alpha * x[j];
                for i in j..n {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        *ptr += alpha_xj * x[i];
                    }
                }
            }
        }
        SyrUplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * x[i] * x[j] for i <= j
            for j in 0..n {
                let alpha_xj = alpha * x[j];
                for i in 0..=j {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        *ptr += alpha_xj * x[i];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs the symmetric rank-1 update on a freshly allocated matrix.
///
/// Returns A + alpha * x * x^T where the result is fully symmetric.
///
/// # Arguments
///
/// * `alpha` - The scalar alpha
/// * `x` - The vector x
///
/// # Returns
///
/// A new symmetric matrix containing alpha * x * x^T.
pub fn syr_new<T: Real + bytemuck::Zeroable>(alpha: T, x: &[T]) -> oxiblas_matrix::Mat<T> {
    let n = x.len();
    let mut a = oxiblas_matrix::Mat::zeros(n, n);

    if n == 0 || alpha == T::zero() {
        return a;
    }

    // Create full symmetric matrix
    for i in 0..n {
        for j in 0..=i {
            let val = alpha * x[i] * x[j];
            a[(i, j)] = val;
            a[(j, i)] = val;
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_syr_lower() {
        // Start with zero matrix
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];

        syr(SyrUplo::Lower, 1.0, &x, a.as_mut()).unwrap();

        // Lower triangle should be x * x^T
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10); // 1*1
        assert!((a[(1, 0)] - 2.0).abs() < 1e-10); // 2*1
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10); // 2*2
        assert!((a[(2, 0)] - 3.0).abs() < 1e-10); // 3*1
        assert!((a[(2, 1)] - 6.0).abs() < 1e-10); // 3*2
        assert!((a[(2, 2)] - 9.0).abs() < 1e-10); // 3*3

        // Upper triangle should be unchanged (zero)
        assert!((a[(0, 1)]).abs() < 1e-10);
        assert!((a[(0, 2)]).abs() < 1e-10);
        assert!((a[(1, 2)]).abs() < 1e-10);
    }

    #[test]
    fn test_syr_upper() {
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];

        syr(SyrUplo::Upper, 1.0, &x, a.as_mut()).unwrap();

        // Upper triangle should be x * x^T
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 2)] - 3.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 2)] - 6.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 9.0).abs() < 1e-10);

        // Lower triangle should be unchanged
        assert!((a[(1, 0)]).abs() < 1e-10);
        assert!((a[(2, 0)]).abs() < 1e-10);
        assert!((a[(2, 1)]).abs() < 1e-10);
    }

    #[test]
    fn test_syr_with_alpha() {
        let mut a: Mat<f64> = Mat::zeros(2, 2);
        let x = [1.0, 2.0];

        syr(SyrUplo::Lower, 2.0, &x, a.as_mut()).unwrap();

        // Lower triangle should be 2 * x * x^T
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10); // 2*1*1
        assert!((a[(1, 0)] - 4.0).abs() < 1e-10); // 2*2*1
        assert!((a[(1, 1)] - 8.0).abs() < 1e-10); // 2*2*2
    }

    #[test]
    fn test_syr_accumulate() {
        // Start with non-zero symmetric matrix
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 5.0], &[3.0, 5.0, 6.0]]);
        let x = [1.0, 1.0, 1.0];

        syr(SyrUplo::Lower, 1.0, &x, a.as_mut()).unwrap();

        // All lower triangle elements increased by 1
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10); // 1 + 1
        assert!((a[(1, 0)] - 3.0).abs() < 1e-10); // 2 + 1
        assert!((a[(1, 1)] - 5.0).abs() < 1e-10); // 4 + 1
        assert!((a[(2, 0)] - 4.0).abs() < 1e-10); // 3 + 1
        assert!((a[(2, 1)] - 6.0).abs() < 1e-10); // 5 + 1
        assert!((a[(2, 2)] - 7.0).abs() < 1e-10); // 6 + 1

        // Upper triangle unchanged
        assert!((a[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 2)] - 3.0).abs() < 1e-10);
        assert!((a[(1, 2)] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr_zero_alpha() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let original = a.clone();
        let x = [1.0, 2.0];

        syr(SyrUplo::Lower, 0.0, &x, a.as_mut()).unwrap();

        // Matrix unchanged
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - original[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_syr_not_square() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 2.0];

        let result = syr(SyrUplo::Lower, 1.0, &x, a.as_mut());
        assert!(matches!(result, Err(SyrError::NotSquare)));
    }

    #[test]
    fn test_syr_dimension_mismatch() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let x = [1.0, 2.0, 3.0];

        let result = syr(SyrUplo::Lower, 1.0, &x, a.as_mut());
        assert!(matches!(result, Err(SyrError::DimensionMismatch)));
    }

    #[test]
    fn test_syr_empty() {
        let mut a: Mat<f64> = Mat::zeros(0, 0);
        let x: [f64; 0] = [];

        syr(SyrUplo::Lower, 1.0, &x, a.as_mut()).unwrap();
    }

    #[test]
    fn test_syr_f32() {
        let mut a: Mat<f32> = Mat::zeros(2, 2);
        let x = [2.0f32, 3.0];

        syr(SyrUplo::Lower, 1.0, &x, a.as_mut()).unwrap();

        assert!((a[(0, 0)] - 4.0).abs() < 1e-5); // 2*2
        assert!((a[(1, 0)] - 6.0).abs() < 1e-5); // 3*2
        assert!((a[(1, 1)] - 9.0).abs() < 1e-5); // 3*3
    }

    #[test]
    fn test_syr_new() {
        let x = [1.0f64, 2.0, 3.0];

        let a = syr_new(1.0, &x);

        // Full symmetric matrix
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 2)] - 3.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 2)] - 6.0).abs() < 1e-10);
        assert!((a[(2, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(2, 1)] - 6.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr_negative_alpha() {
        let mut a: Mat<f64> = Mat::zeros(2, 2);
        let x = [1.0, 2.0];

        syr(SyrUplo::Lower, -1.0, &x, a.as_mut()).unwrap();

        // Lower triangle should be -x * x^T
        assert!((a[(0, 0)] + 1.0).abs() < 1e-10); // -1*1
        assert!((a[(1, 0)] + 2.0).abs() < 1e-10); // -2*1
        assert!((a[(1, 1)] + 4.0).abs() < 1e-10); // -2*2
    }
}
