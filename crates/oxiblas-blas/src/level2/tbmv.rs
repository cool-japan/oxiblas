//! TBMV: Triangular Band Matrix-Vector multiply.
//!
//! Performs x = op(A)·x where A is a triangular band matrix.
//!
//! # Band Storage Format for Triangular Matrices
//!
//! A triangular band matrix with k super-/sub-diagonals is stored in a compact format.
//!
//! For upper triangular with k superdiagonals (k+1 rows, n columns):
//! - Element A\[i,j\] (where i <= j <= i+k) is stored at ab[k + i - j, j]
//!
//! For lower triangular with k subdiagonals:
//! - Element A\[i,j\] (where i-k <= j <= i) is stored at ab[i - j, j]

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Specifies which triangle of the band matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbmvUplo {
    /// Upper triangle is stored.
    Upper,
    /// Lower triangle is stored.
    Lower,
}

/// Specifies the operation to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbmvTrans {
    /// No transpose: x = A·x
    NoTrans,
    /// Transpose: x = A^T·x
    Trans,
    /// Conjugate transpose: x = A^H·x
    ConjTrans,
}

/// Specifies whether the matrix has unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbmvDiag {
    /// Non-unit diagonal (use values from matrix).
    NonUnit,
    /// Unit diagonal (assume 1s on diagonal).
    Unit,
}

/// Error type for TBMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbmvError {
    /// Invalid matrix dimensions.
    InvalidDimensions,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
}

impl core::fmt::Display for TbmvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "Invalid matrix dimensions"),
            Self::DimensionMismatchX => write!(f, "Vector x has wrong dimension"),
        }
    }
}

impl std::error::Error for TbmvError {}

/// Triangular band matrix-vector multiply.
///
/// Performs x = op(A)·x where A is an n×n triangular band matrix with k
/// super-/sub-diagonals stored in band format.
///
/// # Arguments
///
/// * `uplo` - Whether upper or lower triangle is stored
/// * `trans` - The operation to apply: `NoTrans`, Trans, or `ConjTrans`
/// * `diag` - Whether diagonal is unit or non-unit
/// * `n` - Order of the matrix A
/// * `k` - Number of super-/sub-diagonals
/// * `ab` - Band matrix in band storage format (k+1 rows, n columns)
/// * `x` - Input/output vector of length n (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{tbmv, TbmvUplo, TbmvTrans, TbmvDiag};
/// use oxiblas_matrix::Mat;
///
/// // 4×4 upper triangular bidiagonal matrix (k=1):
/// // A = [[1, 2, 0, 0],
/// //      [0, 3, 4, 0],
/// //      [0, 0, 5, 6],
/// //      [0, 0, 0, 7]]
/// //
/// // Upper band storage (2 rows, 4 columns):
/// // AB = [[*, 2, 4, 6],     <- superdiagonal
/// //       [1, 3, 5, 7]]     <- main diagonal
///
/// let ab = Mat::from_rows(&[
///     &[0.0f64, 2.0, 4.0, 6.0],
///     &[1.0, 3.0, 5.0, 7.0],
/// ]);
/// let mut x = [1.0f64, 1.0, 1.0, 1.0];
///
/// tbmv(TbmvUplo::Upper, TbmvTrans::NoTrans, TbmvDiag::NonUnit, 4, 1, ab.as_ref(), &mut x).unwrap();
///
/// // x = A * [1,1,1,1] = [1+2, 3+4, 5+6, 7] = [3, 7, 11, 7]
/// assert!((x[0] - 3.0).abs() < 1e-10);
/// assert!((x[1] - 7.0).abs() < 1e-10);
/// assert!((x[2] - 11.0).abs() < 1e-10);
/// assert!((x[3] - 7.0).abs() < 1e-10);
/// ```
pub fn tbmv<T: Field>(
    uplo: TbmvUplo,
    trans: TbmvTrans,
    diag: TbmvDiag,
    n: usize,
    k: usize,
    ab: MatRef<'_, T>,
    x: &mut [T],
) -> Result<(), TbmvError> {
    // Validate dimensions
    if ab.nrows() < k + 1 || ab.ncols() != n {
        return Err(TbmvError::InvalidDimensions);
    }
    if x.len() != n {
        return Err(TbmvError::DimensionMismatchX);
    }

    // Handle empty matrix
    if n == 0 {
        return Ok(());
    }

    let use_unit_diag = matches!(diag, TbmvDiag::Unit);

    match (uplo, trans) {
        (TbmvUplo::Upper, TbmvTrans::NoTrans) => {
            // x = A * x for upper triangular
            // Process rows from top to bottom
            // For row i, x[i] = sum of A[i,j]*x[j] for j = i to min(i+k, n-1)
            for i in 0..n {
                // Diagonal element
                let mut temp = if use_unit_diag {
                    x[i]
                } else {
                    ab[(k, i)] * x[i]
                };

                // Superdiagonal elements (j > i)
                let j_end = (i + k + 1).min(n);
                for j in (i + 1)..j_end {
                    // A[i,j] is at ab[k + i - j, j]
                    let ab_row = k + i - j;
                    temp += ab[(ab_row, j)] * x[j];
                }

                x[i] = temp;
            }
        }
        (TbmvUplo::Upper, TbmvTrans::Trans) => {
            // x = A^T * x for upper triangular
            // A^T is lower triangular, process from bottom to top
            for i in (0..n).rev() {
                // Diagonal element

                let mut temp = if use_unit_diag {
                    x[i]
                } else {
                    ab[(k, i)] * x[i]
                };

                // Elements from upper triangle of A become subdiagonal in A^T
                // A^T[i,j] = A[j,i] for j < i (in band: max(0, i-k) <= j < i)
                let j_start = i.saturating_sub(k);
                for j in j_start..i {
                    // A[j,i] is at ab[k + j - i, i]
                    let ab_row = k + j - i;
                    temp += ab[(ab_row, i)] * x[j];
                }

                x[i] = temp;
            }
        }
        (TbmvUplo::Upper, TbmvTrans::ConjTrans) => {
            // x = A^H * x for upper triangular
            for i in (0..n).rev() {
                let mut temp = if use_unit_diag {
                    x[i]
                } else {
                    ab[(k, i)].conj() * x[i]
                };

                let j_start = i.saturating_sub(k);
                for j in j_start..i {
                    let ab_row = k + j - i;
                    temp += ab[(ab_row, i)].conj() * x[j];
                }

                x[i] = temp;
            }
        }
        (TbmvUplo::Lower, TbmvTrans::NoTrans) => {
            // x = A * x for lower triangular
            // Process rows from bottom to top
            // For row i, x[i] = sum of A[i,j]*x[j] for j = max(0, i-k) to i
            for i in (0..n).rev() {
                let mut temp = T::zero();

                // Subdiagonal elements (j < i)
                let j_start = i.saturating_sub(k);
                for j in j_start..i {
                    // A[i,j] is at ab[i - j, j]
                    let ab_row = i - j;
                    temp += ab[(ab_row, j)] * x[j];
                }

                // Diagonal element
                if use_unit_diag {
                    temp += x[i];
                } else {
                    temp += ab[(0, i)] * x[i];
                }

                x[i] = temp;
            }
        }
        (TbmvUplo::Lower, TbmvTrans::Trans) => {
            // x = A^T * x for lower triangular
            // A^T is upper triangular, process from top to bottom
            for i in 0..n {
                // Diagonal
                let mut temp = if use_unit_diag {
                    x[i]
                } else {
                    ab[(0, i)] * x[i]
                };

                // Elements from lower triangle of A become superdiagonal in A^T
                // A^T[i,j] = A[j,i] for j > i (in band: i < j <= min(i+k, n-1))
                let j_end = (i + k + 1).min(n);
                for j in (i + 1)..j_end {
                    // A[j,i] is at ab[j - i, i]
                    let ab_row = j - i;
                    temp += ab[(ab_row, i)] * x[j];
                }

                x[i] = temp;
            }
        }
        (TbmvUplo::Lower, TbmvTrans::ConjTrans) => {
            // x = A^H * x for lower triangular
            // A^H is upper triangular, process from top to bottom
            for i in 0..n {
                // Diagonal
                let mut temp = if use_unit_diag {
                    x[i]
                } else {
                    ab[(0, i)].conj() * x[i]
                };

                // A^H[i,j] = conj(A[j,i]) for j > i
                let j_end = (i + k + 1).min(n);
                for j in (i + 1)..j_end {
                    let ab_row = j - i;
                    temp += ab[(ab_row, i)].conj() * x[j];
                }

                x[i] = temp;
            }
        }
    }

    Ok(())
}

/// Triangular band matrix-vector multiply that allocates the result.
///
/// Computes y = op(A)·x where A is a triangular band matrix.
pub fn tbmv_new<T: Field + Clone>(
    uplo: TbmvUplo,
    trans: TbmvTrans,
    diag: TbmvDiag,
    n: usize,
    k: usize,
    ab: MatRef<'_, T>,
    x: &[T],
) -> Result<Vec<T>, TbmvError> {
    let mut y = x.to_vec();
    tbmv(uplo, trans, diag, n, k, ab, &mut y)?;
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

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn approx_eq_c(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    #[test]
    fn test_tbmv_upper_no_trans() {
        // 4×4 upper triangular bidiagonal (k=1)
        // A = [[1, 2, 0, 0], [0, 3, 4, 0], [0, 0, 5, 6], [0, 0, 0, 7]]
        let ab = Mat::from_rows(&[&[0.0f64, 2.0, 4.0, 6.0], &[1.0, 3.0, 5.0, 7.0]]);
        let mut x = [1.0f64, 1.0, 1.0, 1.0];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            4,
            1,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        // x = A * [1,1,1,1] = [1+2, 3+4, 5+6, 7]
        assert!(approx_eq(x[0], 3.0));
        assert!(approx_eq(x[1], 7.0));
        assert!(approx_eq(x[2], 11.0));
        assert!(approx_eq(x[3], 7.0));
    }

    #[test]
    fn test_tbmv_lower_no_trans() {
        // 4×4 lower triangular bidiagonal (k=1)
        // A = [[1, 0, 0, 0], [2, 3, 0, 0], [0, 4, 5, 0], [0, 0, 6, 7]]
        let ab = Mat::from_rows(&[&[1.0f64, 3.0, 5.0, 7.0], &[2.0, 4.0, 6.0, 0.0]]);
        let mut x = [1.0f64, 1.0, 1.0, 1.0];

        tbmv(
            TbmvUplo::Lower,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            4,
            1,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        // x = A * [1,1,1,1] = [1, 2+3, 4+5, 6+7]
        assert!(approx_eq(x[0], 1.0));
        assert!(approx_eq(x[1], 5.0));
        assert!(approx_eq(x[2], 9.0));
        assert!(approx_eq(x[3], 13.0));
    }

    #[test]
    fn test_tbmv_upper_trans() {
        // Upper triangular bidiagonal transpose
        let ab = Mat::from_rows(&[&[0.0f64, 2.0, 4.0, 6.0], &[1.0, 3.0, 5.0, 7.0]]);
        let mut x = [1.0f64, 1.0, 1.0, 1.0];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::Trans,
            TbmvDiag::NonUnit,
            4,
            1,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        // A^T = [[1, 0, 0, 0], [2, 3, 0, 0], [0, 4, 5, 0], [0, 0, 6, 7]]
        // x = A^T * [1,1,1,1] = [1, 2+3, 4+5, 6+7]
        assert!(approx_eq(x[0], 1.0));
        assert!(approx_eq(x[1], 5.0));
        assert!(approx_eq(x[2], 9.0));
        assert!(approx_eq(x[3], 13.0));
    }

    #[test]
    fn test_tbmv_unit_diagonal() {
        // Unit diagonal test
        let ab = Mat::from_rows(&[
            &[0.0f64, 2.0, 4.0],
            &[0.0, 0.0, 0.0], // Diagonal values ignored for unit diagonal
        ]);
        let mut x = [1.0f64, 1.0, 1.0];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::Unit,
            3,
            1,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        // A = [[1, 2, 0], [0, 1, 4], [0, 0, 1]] (unit diagonal)
        // x = A * [1,1,1] = [1+2, 1+4, 1]
        assert!(approx_eq(x[0], 3.0));
        assert!(approx_eq(x[1], 5.0));
        assert!(approx_eq(x[2], 1.0));
    }

    #[test]
    fn test_tbmv_diagonal_only() {
        // Diagonal matrix (k=0)
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let mut x = [1.0f64, 2.0, 3.0];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            3,
            0,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        assert!(approx_eq(x[0], 2.0));
        assert!(approx_eq(x[1], 6.0));
        assert!(approx_eq(x[2], 12.0));
    }

    #[test]
    fn test_tbmv_complex() {
        let ab = Mat::from_rows(&[&[c(0.0, 0.0), c(1.0, 1.0)], &[c(2.0, 0.0), c(3.0, 0.0)]]);
        let mut x = [c(1.0, 0.0), c(1.0, 0.0)];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            2,
            1,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        // A = [[2, 1+i], [0, 3]]
        // x = A * [1, 1] = [2 + (1+i), 3] = [3+i, 3]
        assert!(approx_eq_c(x[0], c(3.0, 1.0)));
        assert!(approx_eq_c(x[1], c(3.0, 0.0)));
    }

    #[test]
    fn test_tbmv_conj_trans() {
        let ab = Mat::from_rows(&[&[c(0.0, 0.0), c(1.0, 1.0)], &[c(2.0, 0.0), c(3.0, 0.0)]]);
        let mut x = [c(1.0, 0.0), c(1.0, 0.0)];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::ConjTrans,
            TbmvDiag::NonUnit,
            2,
            1,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        // A^H = [[2, 0], [1-i, 3]]
        // x = A^H * [1, 1] = [2, (1-i) + 3] = [2, 4-i]
        assert!(approx_eq_c(x[0], c(2.0, 0.0)));
        assert!(approx_eq_c(x[1], c(4.0, -1.0)));
    }

    #[test]
    fn test_tbmv_1x1() {
        let ab = Mat::from_rows(&[&[5.0f64]]);
        let mut x = [3.0f64];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            1,
            0,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        assert!(approx_eq(x[0], 15.0));
    }

    #[test]
    fn test_tbmv_empty() {
        let ab: Mat<f64> = Mat::zeros(1, 0);
        let mut x: [f64; 0] = [];

        let result = tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            0,
            0,
            ab.as_ref(),
            &mut x,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_tbmv_dimension_mismatch() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let mut x = [1.0f64, 2.0]; // Wrong size

        let result = tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            3,
            0,
            ab.as_ref(),
            &mut x,
        );
        assert!(matches!(result, Err(TbmvError::DimensionMismatchX)));
    }

    #[test]
    fn test_tbmv_invalid_dimensions() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0]]);
        let mut x = [1.0f64, 2.0, 3.0];

        let result = tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            3,
            1,
            ab.as_ref(),
            &mut x,
        );
        assert!(matches!(result, Err(TbmvError::InvalidDimensions)));
    }

    #[test]
    fn test_tbmv_f32() {
        let ab = Mat::from_rows(&[&[0.0f32, 2.0, 4.0], &[1.0, 3.0, 5.0]]);
        let mut x = [1.0f32, 1.0, 1.0];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            3,
            1,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        assert!((x[0] - 3.0).abs() < 1e-5);
        assert!((x[1] - 7.0).abs() < 1e-5);
        assert!((x[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_tbmv_new() {
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 2.0, 3.0];

        let y = tbmv_new(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            3,
            0,
            ab.as_ref(),
            &x,
        )
        .unwrap();

        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 6.0));
        assert!(approx_eq(y[2], 12.0));
    }

    #[test]
    fn test_tbmv_pentadiagonal_upper() {
        // 5×5 upper triangular with 2 superdiagonals
        let ab = Mat::from_rows(&[
            &[0.0f64, 0.0, 1.0, 2.0, 3.0], // 2nd superdiag
            &[0.0, 4.0, 5.0, 6.0, 7.0],    // 1st superdiag
            &[8.0, 9.0, 10.0, 11.0, 12.0], // main diag
        ]);
        let mut x = [1.0f64; 5];

        tbmv(
            TbmvUplo::Upper,
            TbmvTrans::NoTrans,
            TbmvDiag::NonUnit,
            5,
            2,
            ab.as_ref(),
            &mut x,
        )
        .unwrap();

        // A is upper triangular with bands
        // x[0] = 8 + 4 + 1 = 13  (no contribution from below)
        // x[1] = 9 + 5 + 2 = 16
        // x[2] = 10 + 6 + 3 = 19
        // x[3] = 11 + 7 = 18
        // x[4] = 12
        assert!(approx_eq(x[0], 13.0));
        assert!(approx_eq(x[1], 16.0));
        assert!(approx_eq(x[2], 19.0));
        assert!(approx_eq(x[3], 18.0));
        assert!(approx_eq(x[4], 12.0));
    }
}
