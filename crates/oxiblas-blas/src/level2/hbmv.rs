//! HBMV: Hermitian Band Matrix-Vector multiply.
//!
//! Performs y = α·A·x + β·y where A is a Hermitian band matrix.
//!
//! # Band Storage Format for Hermitian Matrices
//!
//! A Hermitian band matrix with k super-/sub-diagonals is stored in a compact format.
//! Only the upper or lower triangle is stored. The diagonal elements are real.
//!
//! For upper triangle storage (k+1 rows, n columns):
//! - Element A\[i,j\] (where j >= i) is stored at ab[k + i - j, j]
//! - The conjugate A\[j,i\] = conj(A\[i,j\]) is inferred from symmetry

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Specifies which triangle of the Hermitian band matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbmvUplo {
    /// Upper triangle is stored.
    Upper,
    /// Lower triangle is stored.
    Lower,
}

/// Error type for HBMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbmvError {
    /// Invalid matrix dimensions.
    InvalidDimensions,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for HbmvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "Invalid matrix dimensions"),
            Self::DimensionMismatchX => write!(f, "Vector x has wrong dimension"),
            Self::DimensionMismatchY => write!(f, "Vector y has wrong dimension"),
        }
    }
}

impl std::error::Error for HbmvError {}

/// Hermitian band matrix-vector multiply.
///
/// Performs y = α·A·x + β·y where A is an n×n Hermitian band matrix with k
/// super-/sub-diagonals stored in band format.
///
/// For Hermitian matrices, A\[i,j\] = conj(A\[j,i\]) and diagonal elements are real.
///
/// # Arguments
///
/// * `uplo` - Whether upper or lower triangle is stored
/// * `n` - Order of the matrix A
/// * `k` - Number of super-diagonals (equals number of sub-diagonals)
/// * `alpha` - Scalar multiplier for A·x
/// * `ab` - Band matrix in band storage format (k+1 rows, n columns)
/// * `x` - Input vector of length n
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector of length n (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{hbmv, HbmvUplo};
/// use oxiblas_matrix::Mat;
/// use num_complex::Complex64;
///
/// fn c(re: f64, im: f64) -> Complex64 {
///     Complex64::new(re, im)
/// }
///
/// // 3×3 Hermitian tridiagonal matrix (k=1):
/// // A = [[1,    2+i,   0  ],
/// //      [2-i,  3,    4+2i],
/// //      [0,   4-2i,  5   ]]
/// //
/// // Upper band storage (2 rows, 3 columns):
/// // AB = [[*, 2+i, 4+2i],     <- superdiagonal
/// //       [1, 3,   5   ]]     <- main diagonal (real)
///
/// let ab = Mat::from_rows(&[
///     &[c(0.0, 0.0), c(2.0, 1.0), c(4.0, 2.0)],
///     &[c(1.0, 0.0), c(3.0, 0.0), c(5.0, 0.0)],
/// ]);
/// let x = [c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
/// let mut y = [c(0.0, 0.0); 3];
///
/// hbmv(HbmvUplo::Upper, 3, 1, c(1.0, 0.0), ab.as_ref(), &x, c(0.0, 0.0), &mut y).unwrap();
///
/// // y = A * [1,1,1]
/// // y[0] = 1 + (2+i) = 3+i
/// // y[1] = (2-i) + 3 + (4+2i) = 9+i
/// // y[2] = (4-2i) + 5 = 9-2i
/// assert!((y[0].re - 3.0).abs() < 1e-10);
/// assert!((y[0].im - 1.0).abs() < 1e-10);
/// ```
pub fn hbmv<T: Field>(
    uplo: HbmvUplo,
    n: usize,
    k: usize,
    alpha: T,
    ab: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<(), HbmvError> {
    // Validate dimensions
    if ab.nrows() < k + 1 || ab.ncols() != n {
        return Err(HbmvError::InvalidDimensions);
    }
    if x.len() != n {
        return Err(HbmvError::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(HbmvError::DimensionMismatchY);
    }

    // Handle empty matrix
    if n == 0 {
        return Ok(());
    }

    // Scale y by beta
    if beta.is_zero() {
        for yi in y.iter_mut() {
            *yi = T::zero();
        }
    } else if beta != T::one() {
        for yi in y.iter_mut() {
            *yi = beta * *yi;
        }
    }

    // If alpha is zero, we're done
    if alpha.is_zero() {
        return Ok(());
    }

    match uplo {
        HbmvUplo::Upper => {
            // Upper triangle stored
            for j in 0..n {
                let alpha_xj = alpha * x[j];

                // Diagonal element (assumed real, but we use it as-is)
                y[j] += alpha_xj * ab[(k, j)];

                // Off-diagonal elements (i < j)
                let i_start = j.saturating_sub(k);
                for i in i_start..j {
                    let ab_row = k + i - j;
                    let aij = ab[(ab_row, j)];

                    // y[i] += alpha * A[i,j] * x[j]
                    y[i] += alpha * aij * x[j];
                    // y[j] += alpha * conj(A[i,j]) * x[i] (due to Hermitian symmetry)
                    y[j] += alpha * aij.conj() * x[i];
                }
            }
        }
        HbmvUplo::Lower => {
            // Lower triangle stored
            for j in 0..n {
                let alpha_xj = alpha * x[j];

                // Diagonal element
                y[j] += alpha_xj * ab[(0, j)];

                // Off-diagonal elements (i > j)
                let i_end = (j + k + 1).min(n);
                for i in (j + 1)..i_end {
                    let ab_row = i - j;
                    let aij = ab[(ab_row, j)];

                    // y[i] += alpha * A[i,j] * x[j]
                    y[i] += alpha * aij * x[j];
                    // y[j] += alpha * conj(A[i,j]) * x[i] (due to Hermitian symmetry)
                    y[j] += alpha * aij.conj() * x[i];
                }
            }
        }
    }

    Ok(())
}

/// Hermitian band matrix-vector multiply that allocates the result.
///
/// Computes y = α·A·x where A is a Hermitian band matrix.
pub fn hbmv_new<T: Field>(
    uplo: HbmvUplo,
    n: usize,
    k: usize,
    alpha: T,
    ab: MatRef<'_, T>,
    x: &[T],
) -> Result<Vec<T>, HbmvError> {
    let mut y = vec![T::zero(); n];
    hbmv(uplo, n, k, alpha, ab, x, T::zero(), &mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn approx_eq_c(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_hbmv_upper_tridiagonal_complex() {
        // 3×3 Hermitian tridiagonal matrix (k=1):
        // A = [[1,    2+i,   0  ],
        //      [2-i,  3,    4+2i],
        //      [0,   4-2i,  5   ]]
        let ab = Mat::from_rows(&[
            &[c(0.0, 0.0), c(2.0, 1.0), c(4.0, 2.0)],
            &[c(1.0, 0.0), c(3.0, 0.0), c(5.0, 0.0)],
        ]);
        let x = [c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        hbmv(
            HbmvUplo::Upper,
            3,
            1,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y[0] = 1 + (2+i) = 3+i
        // y[1] = (2-i) + 3 + (4+2i) = 9+i
        // y[2] = (4-2i) + 5 = 9-2i
        assert!(approx_eq_c(y[0], c(3.0, 1.0)));
        assert!(approx_eq_c(y[1], c(9.0, 1.0)));
        assert!(approx_eq_c(y[2], c(9.0, -2.0)));
    }

    #[test]
    fn test_hbmv_lower_tridiagonal_complex() {
        // Same matrix stored in lower format
        // Lower band storage:
        // AB = [[1, 3, 5],         <- main diagonal
        //       [2-i, 4-2i, *]]    <- subdiagonal
        let ab = Mat::from_rows(&[
            &[c(1.0, 0.0), c(3.0, 0.0), c(5.0, 0.0)],
            &[c(2.0, -1.0), c(4.0, -2.0), c(0.0, 0.0)],
        ]);
        let x = [c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        hbmv(
            HbmvUplo::Lower,
            3,
            1,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // Same result as upper
        assert!(approx_eq_c(y[0], c(3.0, 1.0)));
        assert!(approx_eq_c(y[1], c(9.0, 1.0)));
        assert!(approx_eq_c(y[2], c(9.0, -2.0)));
    }

    #[test]
    fn test_hbmv_real_symmetric() {
        // For real types, HBMV should behave like SBMV
        let ab = Mat::from_rows(&[&[0.0f64, 2.0, 4.0], &[1.0, 3.0, 5.0]]);
        let x = [1.0f64, 1.0, 1.0];
        let mut y = [0.0f64; 3];

        hbmv(HbmvUplo::Upper, 3, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();

        // A = [[1, 2, 0],
        //      [2, 3, 4],
        //      [0, 4, 5]]
        // y = A * [1,1,1] = [3, 9, 9]
        assert!(approx_eq(y[0], 3.0));
        assert!(approx_eq(y[1], 9.0));
        assert!(approx_eq(y[2], 9.0));
    }

    #[test]
    fn test_hbmv_diagonal_complex() {
        // Diagonal Hermitian matrix (k=0) - diagonals must be real
        let ab = Mat::from_rows(&[&[c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)]]);
        let x = [c(1.0, 1.0), c(2.0, 0.0), c(0.0, 3.0)];
        let mut y = [c(0.0, 0.0); 3];

        hbmv(
            HbmvUplo::Upper,
            3,
            0,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y = diag(2,3,4) * x
        assert!(approx_eq_c(y[0], c(2.0, 2.0)));
        assert!(approx_eq_c(y[1], c(6.0, 0.0)));
        assert!(approx_eq_c(y[2], c(0.0, 12.0)));
    }

    #[test]
    fn test_hbmv_with_alpha_beta_complex() {
        let ab = Mat::from_rows(&[&[c(2.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(1.0, 1.0), c(1.0, -1.0)];

        // y = (1+i)*A*x + (1-i)*y
        hbmv(
            HbmvUplo::Upper,
            2,
            0,
            c(1.0, 1.0),
            ab.as_ref(),
            &x,
            c(1.0, -1.0),
            &mut y,
        )
        .unwrap();

        // A*x = [2, 3]
        // (1+i)*[2,3] = [2+2i, 3+3i]
        // (1-i)*[1+i, 1-i] = [(1-i)(1+i), (1-i)(1-i)] = [2, -2i]
        // y = [2+2i+2, 3+3i-2i] = [4+2i, 3+i]
        assert!(approx_eq_c(y[0], c(4.0, 2.0)));
        assert!(approx_eq_c(y[1], c(3.0, 1.0)));
    }

    #[test]
    fn test_hbmv_1x1() {
        let ab = Mat::from_rows(&[&[c(5.0, 0.0)]]);
        let x = [c(2.0, 1.0)];
        let mut y = [c(0.0, 0.0)];

        hbmv(
            HbmvUplo::Upper,
            1,
            0,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(10.0, 5.0)));
    }

    #[test]
    fn test_hbmv_alpha_zero() {
        let ab = Mat::from_rows(&[&[c(1.0, 0.0), c(2.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let mut y = [c(1.0, 1.0), c(2.0, 2.0)];

        hbmv(
            HbmvUplo::Upper,
            2,
            0,
            c(0.0, 0.0),
            ab.as_ref(),
            &x,
            c(2.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(2.0, 2.0)));
        assert!(approx_eq_c(y[1], c(4.0, 4.0)));
    }

    #[test]
    fn test_hbmv_beta_zero() {
        let ab = Mat::from_rows(&[&[c(2.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(100.0, 100.0), c(200.0, 200.0)];

        hbmv(
            HbmvUplo::Upper,
            2,
            0,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(2.0, 0.0)));
        assert!(approx_eq_c(y[1], c(3.0, 0.0)));
    }

    #[test]
    fn test_hbmv_empty() {
        let ab: Mat<Complex64> = Mat::zeros(1, 0);
        let x: [Complex64; 0] = [];
        let mut y: [Complex64; 0] = [];

        let result = hbmv(
            HbmvUplo::Upper,
            0,
            0,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_hbmv_dimension_mismatch_x() {
        let ab = Mat::from_rows(&[&[c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        let result = hbmv(
            HbmvUplo::Upper,
            3,
            0,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        );
        assert!(matches!(result, Err(HbmvError::DimensionMismatchX)));
    }

    #[test]
    fn test_hbmv_dimension_mismatch_y() {
        let ab = Mat::from_rows(&[&[c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        let result = hbmv(
            HbmvUplo::Upper,
            3,
            0,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        );
        assert!(matches!(result, Err(HbmvError::DimensionMismatchY)));
    }

    #[test]
    fn test_hbmv_invalid_dimensions() {
        let ab = Mat::from_rows(&[&[c(1.0, 0.0), c(2.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        let result = hbmv(
            HbmvUplo::Upper,
            3,
            1,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        );
        assert!(matches!(result, Err(HbmvError::InvalidDimensions)));
    }

    #[test]
    fn test_hbmv_new() {
        let ab = Mat::from_rows(&[&[c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];

        let y = hbmv_new(HbmvUplo::Upper, 3, 0, c(1.0, 0.0), ab.as_ref(), &x).unwrap();

        assert!(approx_eq_c(y[0], c(2.0, 0.0)));
        assert!(approx_eq_c(y[1], c(6.0, 0.0)));
        assert!(approx_eq_c(y[2], c(12.0, 0.0)));
    }

    #[test]
    fn test_hbmv_hermitian_property() {
        // Verify that A*x gives expected result for Hermitian matrix
        // where A[i,j] = conj(A[j,i])
        let ab = Mat::from_rows(&[
            &[c(0.0, 0.0), c(1.0, 2.0)], // off-diagonal: a01 = 1+2i
            &[c(3.0, 0.0), c(4.0, 0.0)], // diagonal (real)
        ]);
        let x = [c(1.0, 0.0), c(0.0, 1.0)]; // x = [1, i]
        let mut y = [c(0.0, 0.0); 2];

        hbmv(
            HbmvUplo::Upper,
            2,
            1,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // A = [[3, 1+2i], [1-2i, 4]]
        // y = A * [1, i]
        // y[0] = 3*1 + (1+2i)*i = 3 + i - 2 = 1 + i
        // y[1] = (1-2i)*1 + 4*i = 1 - 2i + 4i = 1 + 2i
        assert!(approx_eq_c(y[0], c(1.0, 1.0)));
        assert!(approx_eq_c(y[1], c(1.0, 2.0)));
    }
}
