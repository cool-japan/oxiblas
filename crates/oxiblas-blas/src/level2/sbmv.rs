//! SBMV: Symmetric Band Matrix-Vector multiply.
//!
//! Performs y = α·A·x + β·y where A is a symmetric band matrix.
//!
//! # Band Storage Format for Symmetric Matrices
//!
//! A symmetric band matrix with k superdiagonals (and k subdiagonals due to symmetry)
//! is stored in a compact format. Only the upper or lower triangle is stored.
//!
//! For upper triangle storage (k+1 rows, n columns):
//! - Row 0 contains elements k positions above the diagonal
//! - Row k contains the main diagonal
//! - Element A\[i,j\] (where j >= i) is stored at ab[k + i - j, j]
//!
//! Example: 5×5 symmetric matrix with k=2:
//! ```text
//!     [ a00  a01  a02   0    0  ]
//!     [ a01  a11  a12  a13   0  ]
//! A = [ a02  a12  a22  a23  a24 ]
//!     [  0   a13  a23  a33  a34 ]
//!     [  0    0   a24  a34  a44 ]
//!
//! Upper band storage (k+1=3 rows, n=5 columns):
//!     [  *    *   a02  a13  a24 ]  <- 2nd superdiagonal
//! AB = [  *   a01  a12  a23  a34 ]  <- 1st superdiagonal
//!     [ a00  a11  a22  a33  a44 ]  <- main diagonal
//! ```

use oxiblas_core::scalar::Real;
use oxiblas_matrix::MatRef;

/// Specifies which triangle of the symmetric band matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbmvUplo {
    /// Upper triangle is stored.
    Upper,
    /// Lower triangle is stored.
    Lower,
}

/// Error type for SBMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbmvError {
    /// Invalid matrix dimensions.
    InvalidDimensions,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for SbmvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "Invalid matrix dimensions"),
            Self::DimensionMismatchX => write!(f, "Vector x has wrong dimension"),
            Self::DimensionMismatchY => write!(f, "Vector y has wrong dimension"),
        }
    }
}

impl std::error::Error for SbmvError {}

/// Symmetric band matrix-vector multiply.
///
/// Performs y = α·A·x + β·y where A is an n×n symmetric band matrix with k
/// super-/sub-diagonals stored in band format.
///
/// # Arguments
///
/// * `uplo` - Whether upper or lower triangle is stored
/// * `n` - Order of the matrix A
/// * `k` - Number of super-diagonals (equals number of sub-diagonals for symmetric)
/// * `alpha` - Scalar multiplier for A·x
/// * `ab` - Band matrix in band storage format (k+1 rows, n columns)
/// * `x` - Input vector of length n
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector of length n (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{sbmv, SbmvUplo};
/// use oxiblas_matrix::Mat;
///
/// // 4×4 symmetric tridiagonal matrix (k=1):
/// // A = [[1, 2, 0, 0],
/// //      [2, 3, 4, 0],
/// //      [0, 4, 5, 6],
/// //      [0, 0, 6, 7]]
/// //
/// // Upper band storage (2 rows, 4 columns):
/// // AB = [[*, 2, 4, 6],     <- superdiagonal
/// //       [1, 3, 5, 7]]     <- main diagonal
///
/// let ab = Mat::from_rows(&[
///     &[0.0f64, 2.0, 4.0, 6.0],  // superdiagonal
///     &[1.0, 3.0, 5.0, 7.0],     // main diagonal
/// ]);
/// let x = [1.0f64, 1.0, 1.0, 1.0];
/// let mut y = [0.0f64; 4];
///
/// sbmv(SbmvUplo::Upper, 4, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();
///
/// // y = A * [1,1,1,1] = [1+2, 2+3+4, 4+5+6, 6+7] = [3, 9, 15, 13]
/// assert!((y[0] - 3.0).abs() < 1e-10);
/// assert!((y[1] - 9.0).abs() < 1e-10);
/// assert!((y[2] - 15.0).abs() < 1e-10);
/// assert!((y[3] - 13.0).abs() < 1e-10);
/// ```
pub fn sbmv<T: Real>(
    uplo: SbmvUplo,
    n: usize,
    k: usize,
    alpha: T,
    ab: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<(), SbmvError> {
    // Validate dimensions
    if ab.nrows() < k + 1 || ab.ncols() != n {
        return Err(SbmvError::InvalidDimensions);
    }
    if x.len() != n {
        return Err(SbmvError::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(SbmvError::DimensionMismatchY);
    }

    // Handle empty matrix
    if n == 0 {
        return Ok(());
    }

    // Scale y by beta
    if beta == T::zero() {
        for yi in y.iter_mut() {
            *yi = T::zero();
        }
    } else if beta != T::one() {
        for yi in y.iter_mut() {
            *yi = beta * *yi;
        }
    }

    // If alpha is zero, we're done
    if alpha == T::zero() {
        return Ok(());
    }

    match uplo {
        SbmvUplo::Upper => {
            // Upper triangle stored
            // ab[k + i - j, j] contains A[i, j] for i <= j <= min(i+k, n-1)
            for j in 0..n {
                let alpha_xj = alpha * x[j];

                // Diagonal element: ab[k, j]
                y[j] += alpha_xj * ab[(k, j)];

                // Off-diagonal elements (i < j)
                let i_start = j.saturating_sub(k);
                for i in i_start..j {
                    // ab[k + i - j, j] contains A[i, j]
                    let ab_row = k + i - j;
                    let aij = ab[(ab_row, j)];

                    // Due to symmetry: A[i,j] = A[j,i]
                    // y[i] += alpha * A[i,j] * x[j]
                    // y[j] += alpha * A[j,i] * x[i] = alpha * A[i,j] * x[i]
                    y[i] += alpha * aij * x[j];
                    y[j] += alpha * aij * x[i];
                }
            }
        }
        SbmvUplo::Lower => {
            // Lower triangle stored
            // ab[i - j, j] contains A[i, j] for max(0, j-k) <= i <= j
            // Actually for lower: ab[i - j, j] for j <= i <= min(j+k, n-1)
            for j in 0..n {
                let alpha_xj = alpha * x[j];

                // Diagonal element: ab[0, j]
                y[j] += alpha_xj * ab[(0, j)];

                // Off-diagonal elements (i > j)
                let i_end = (j + k + 1).min(n);
                for i in (j + 1)..i_end {
                    // ab[i - j, j] contains A[i, j]
                    let ab_row = i - j;
                    let aij = ab[(ab_row, j)];

                    // Due to symmetry
                    y[i] += alpha * aij * x[j];
                    y[j] += alpha * aij * x[i];
                }
            }
        }
    }

    Ok(())
}

/// Symmetric band matrix-vector multiply that allocates the result.
///
/// Computes y = α·A·x where A is a symmetric band matrix.
pub fn sbmv_new<T: Real>(
    uplo: SbmvUplo,
    n: usize,
    k: usize,
    alpha: T,
    ab: MatRef<'_, T>,
    x: &[T],
) -> Result<Vec<T>, SbmvError> {
    let mut y = vec![T::zero(); n];
    sbmv(uplo, n, k, alpha, ab, x, T::zero(), &mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_sbmv_upper_tridiagonal() {
        // 4×4 symmetric tridiagonal matrix (k=1):
        // A = [[1, 2, 0, 0],
        //      [2, 3, 4, 0],
        //      [0, 4, 5, 6],
        //      [0, 0, 6, 7]]
        let ab = Mat::from_rows(&[&[0.0f64, 2.0, 4.0, 6.0], &[1.0, 3.0, 5.0, 7.0]]);
        let x = [1.0f64, 1.0, 1.0, 1.0];
        let mut y = [0.0f64; 4];

        sbmv(SbmvUplo::Upper, 4, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        // y = A * [1,1,1,1] = [1+2, 2+3+4, 4+5+6, 6+7] = [3, 9, 15, 13]
        assert!(approx_eq(y[0], 3.0));
        assert!(approx_eq(y[1], 9.0));
        assert!(approx_eq(y[2], 15.0));
        assert!(approx_eq(y[3], 13.0));
    }

    #[test]
    fn test_sbmv_lower_tridiagonal() {
        // Same matrix stored in lower format
        // Lower band storage:
        // AB = [[1, 3, 5, 7],     <- main diagonal
        //       [2, 4, 6, *]]     <- subdiagonal
        let ab = Mat::from_rows(&[&[1.0f64, 3.0, 5.0, 7.0], &[2.0, 4.0, 6.0, 0.0]]);
        let x = [1.0f64, 1.0, 1.0, 1.0];
        let mut y = [0.0f64; 4];

        sbmv(SbmvUplo::Lower, 4, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        // Same result as upper
        assert!(approx_eq(y[0], 3.0));
        assert!(approx_eq(y[1], 9.0));
        assert!(approx_eq(y[2], 15.0));
        assert!(approx_eq(y[3], 13.0));
    }

    #[test]
    fn test_sbmv_diagonal_only() {
        // Diagonal matrix (k=0)
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        sbmv(SbmvUplo::Upper, 3, 0, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 6.0));
        assert!(approx_eq(y[2], 12.0));
    }

    #[test]
    fn test_sbmv_with_alpha_beta() {
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [1.0f64, 1.0, 1.0];

        // y = 2*A*x + 3*y
        sbmv(SbmvUplo::Upper, 3, 0, 2.0, ab.as_ref(), &x, 3.0, &mut y).unwrap();

        // A*x = [2, 6, 12]
        // y = 2*[2,6,12] + 3*[1,1,1] = [4+3, 12+3, 24+3] = [7, 15, 27]
        assert!(approx_eq(y[0], 7.0));
        assert!(approx_eq(y[1], 15.0));
        assert!(approx_eq(y[2], 27.0));
    }

    #[test]
    fn test_sbmv_pentadiagonal() {
        // 5×5 symmetric pentadiagonal (k=2)
        // Upper storage: [2nd superdiag, 1st superdiag, main diag]
        let ab = Mat::from_rows(&[
            &[0.0f64, 0.0, 1.0, 2.0, 3.0], // 2nd superdiagonal
            &[0.0, 4.0, 5.0, 6.0, 7.0],    // 1st superdiagonal
            &[8.0, 9.0, 10.0, 11.0, 12.0], // main diagonal
        ]);
        let x = [1.0f64; 5];
        let mut y = [0.0f64; 5];

        sbmv(SbmvUplo::Upper, 5, 2, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        // A = [[8, 4, 1, 0, 0],
        //      [4, 9, 5, 2, 0],
        //      [1, 5, 10, 6, 3],
        //      [0, 2, 6, 11, 7],
        //      [0, 0, 3, 7, 12]]
        // y = A * [1,1,1,1,1]
        assert!(approx_eq(y[0], 8.0 + 4.0 + 1.0));
        assert!(approx_eq(y[1], 4.0 + 9.0 + 5.0 + 2.0));
        assert!(approx_eq(y[2], 1.0 + 5.0 + 10.0 + 6.0 + 3.0));
        assert!(approx_eq(y[3], 2.0 + 6.0 + 11.0 + 7.0));
        assert!(approx_eq(y[4], 3.0 + 7.0 + 12.0));
    }

    #[test]
    fn test_sbmv_1x1() {
        let ab = Mat::from_rows(&[&[5.0f64]]);
        let x = [3.0f64];
        let mut y = [0.0f64];

        sbmv(SbmvUplo::Upper, 1, 0, 2.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 30.0)); // 2*5*3
    }

    #[test]
    fn test_sbmv_alpha_zero() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [1.0f64, 2.0, 3.0];

        sbmv(SbmvUplo::Upper, 3, 0, 0.0, ab.as_ref(), &x, 2.0, &mut y).unwrap();

        // y = 0*A*x + 2*y = 2*[1,2,3] = [2,4,6]
        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 4.0));
        assert!(approx_eq(y[2], 6.0));
    }

    #[test]
    fn test_sbmv_beta_zero() {
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 1.0, 1.0];
        let mut y = [100.0f64, 200.0, 300.0];

        sbmv(SbmvUplo::Upper, 3, 0, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 3.0));
        assert!(approx_eq(y[2], 4.0));
    }

    #[test]
    fn test_sbmv_empty() {
        let ab: Mat<f64> = Mat::zeros(1, 0);
        let x: [f64; 0] = [];
        let mut y: [f64; 0] = [];

        let result = sbmv(SbmvUplo::Upper, 0, 0, 1.0, ab.as_ref(), &x, 0.0, &mut y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sbmv_dimension_mismatch_x() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let x = [1.0f64, 2.0]; // Wrong size
        let mut y = [0.0f64; 3];

        let result = sbmv(SbmvUplo::Upper, 3, 0, 1.0, ab.as_ref(), &x, 0.0, &mut y);
        assert!(matches!(result, Err(SbmvError::DimensionMismatchX)));
    }

    #[test]
    fn test_sbmv_dimension_mismatch_y() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 2]; // Wrong size

        let result = sbmv(SbmvUplo::Upper, 3, 0, 1.0, ab.as_ref(), &x, 0.0, &mut y);
        assert!(matches!(result, Err(SbmvError::DimensionMismatchY)));
    }

    #[test]
    fn test_sbmv_invalid_dimensions() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0]]); // Only 1 row, but k+1=2 needed
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        let result = sbmv(SbmvUplo::Upper, 3, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y);
        assert!(matches!(result, Err(SbmvError::InvalidDimensions)));
    }

    #[test]
    fn test_sbmv_f32() {
        let ab = Mat::from_rows(&[&[0.0f32, 2.0, 4.0], &[1.0, 3.0, 5.0]]);
        let x = [1.0f32, 1.0, 1.0];
        let mut y = [0.0f32; 3];

        sbmv(SbmvUplo::Upper, 3, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        // A = [[1, 2, 0], [2, 3, 4], [0, 4, 5]]
        // y = A * [1,1,1] = [1+2, 2+3+4, 4+5] = [3, 9, 9]
        assert!((y[0] - 3.0).abs() < 1e-5);
        assert!((y[1] - 9.0).abs() < 1e-5);
        assert!((y[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_sbmv_new() {
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 2.0, 3.0];

        let y = sbmv_new(SbmvUplo::Upper, 3, 0, 1.0, ab.as_ref(), &x).unwrap();

        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 6.0));
        assert!(approx_eq(y[2], 12.0));
    }

    #[test]
    fn test_sbmv_specific_vector() {
        // Test with non-uniform vector
        let ab = Mat::from_rows(&[&[0.0f64, 2.0, 4.0, 6.0], &[1.0, 3.0, 5.0, 7.0]]);
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let mut y = [0.0f64; 4];

        sbmv(SbmvUplo::Upper, 4, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        // A = [[1, 2, 0, 0],
        //      [2, 3, 4, 0],
        //      [0, 4, 5, 6],
        //      [0, 0, 6, 7]]
        // y = A * [1,2,3,4]
        // y[0] = 1*1 + 2*2 = 5
        // y[1] = 2*1 + 3*2 + 4*3 = 20
        // y[2] = 4*2 + 5*3 + 6*4 = 47
        // y[3] = 6*3 + 7*4 = 46
        assert!(approx_eq(y[0], 5.0));
        assert!(approx_eq(y[1], 20.0));
        assert!(approx_eq(y[2], 47.0));
        assert!(approx_eq(y[3], 46.0));
    }
}
