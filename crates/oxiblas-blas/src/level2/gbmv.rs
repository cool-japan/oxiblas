//! GBMV: General Band Matrix-Vector multiply.
//!
//! Performs matrix-vector operations using a band matrix stored in band format.
//!
//! y = α·A·x + β·y  or  y = α·A^T·x + β·y  or  y = α·A^H·x + β·y
//!
//! where A is an m×n band matrix with kl subdiagonals and ku superdiagonals.
//!
//! # Band Storage Format
//!
//! A band matrix with kl subdiagonals and ku superdiagonals is stored in a
//! compact format where row j of the storage array contains the j-th diagonal.
//!
//! For a matrix A with element a\[i,j\]:
//! - The element is stored at ab[ku + i - j, j]
//! - Only elements where |i - j| <= kl (for i > j) or |i - j| <= ku (for i < j)
//!   are stored
//!
//! Example: 5×5 matrix with kl=2, ku=1:
//! ```text
//!     [ a00  a01   0    0    0  ]
//!     [ a10  a11  a12   0    0  ]
//! A = [ a20  a21  a22  a23   0  ]
//!     [  0   a31  a32  a33  a34 ]
//!     [  0    0   a42  a43  a44 ]
//!
//! Band storage (kl+ku+1 rows, n columns):
//!     [  *   a01  a12  a23  a34 ]  <- superdiagonal (ku=1)
//! AB = [ a00  a11  a22  a33  a44 ]  <- main diagonal
//!     [ a10  a21  a32  a43   *  ]  <- subdiagonal 1
//!     [ a20  a31  a42   *    *  ]  <- subdiagonal 2 (kl=2)
//! ```

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Specifies the operation to be performed on the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbmvTrans {
    /// No transpose: y = α·A·x + β·y
    NoTrans,
    /// Transpose: y = α·A^T·x + β·y
    Trans,
    /// Conjugate transpose: y = α·A^H·x + β·y
    ConjTrans,
}

/// Error type for GBMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbmvError {
    /// Invalid matrix dimensions.
    InvalidDimensions,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
    /// Invalid band parameters (kl or ku).
    InvalidBandParams,
}

impl core::fmt::Display for GbmvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "Invalid matrix dimensions"),
            Self::DimensionMismatchX => write!(f, "Vector x has wrong dimension"),
            Self::DimensionMismatchY => write!(f, "Vector y has wrong dimension"),
            Self::InvalidBandParams => write!(f, "Invalid band parameters (kl or ku)"),
        }
    }
}

impl std::error::Error for GbmvError {}

/// General band matrix-vector multiply.
///
/// Performs y = α·op(A)·x + β·y where A is an m×n band matrix with kl
/// subdiagonals and ku superdiagonals stored in band format.
///
/// # Arguments
///
/// * `trans` - The operation to apply: `NoTrans`, Trans, or `ConjTrans`
/// * `m` - Number of rows of the original matrix A
/// * `n` - Number of columns of the original matrix A
/// * `kl` - Number of subdiagonals
/// * `ku` - Number of superdiagonals
/// * `alpha` - Scalar multiplier for op(A)·x
/// * `ab` - Band matrix in band storage format (kl+ku+1 rows, n columns)
/// * `x` - Input vector
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector (modified in place)
///
/// # Band Storage
///
/// The band matrix A is stored in a compact form with (kl + ku + 1) rows and n columns.
/// For element A\[i,j\], it is stored at ab[ku + i - j, j].
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{gbmv, GbmvTrans};
/// use oxiblas_matrix::Mat;
///
/// // 4×4 tridiagonal matrix (kl=1, ku=1):
/// // A = [[1, 2, 0, 0],
/// //      [3, 4, 5, 0],
/// //      [0, 6, 7, 8],
/// //      [0, 0, 9, 10]]
/// //
/// // Band storage (3 rows, 4 columns):
/// // AB = [[*, 2, 5, 8],     <- superdiagonal
/// //       [1, 4, 7, 10],    <- main diagonal
/// //       [3, 6, 9, *]]     <- subdiagonal
///
/// let ab = Mat::from_rows(&[
///     &[0.0f64, 2.0, 5.0, 8.0],   // superdiagonal (unused first element)
///     &[1.0, 4.0, 7.0, 10.0],      // main diagonal
///     &[3.0, 6.0, 9.0, 0.0],       // subdiagonal (unused last element)
/// ]);
/// let x = [1.0f64, 1.0, 1.0, 1.0];
/// let mut y = [0.0f64; 4];
///
/// gbmv(GbmvTrans::NoTrans, 4, 4, 1, 1, 1.0, ab.as_ref(), &x, 0.0, &mut y).unwrap();
///
/// // y = A * [1,1,1,1] = [1+2, 3+4+5, 6+7+8, 9+10] = [3, 12, 21, 19]
/// assert!((y[0] - 3.0).abs() < 1e-10);
/// assert!((y[1] - 12.0).abs() < 1e-10);
/// assert!((y[2] - 21.0).abs() < 1e-10);
/// assert!((y[3] - 19.0).abs() < 1e-10);
/// ```
pub fn gbmv<T: Field>(
    trans: GbmvTrans,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    ab: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<(), GbmvError> {
    // Validate band storage dimensions
    let expected_rows = kl + ku + 1;
    if ab.nrows() < expected_rows || ab.ncols() != n {
        return Err(GbmvError::InvalidDimensions);
    }

    // Validate vector dimensions based on transpose
    let (x_len, y_len) = match trans {
        GbmvTrans::NoTrans => (n, m),
        GbmvTrans::Trans | GbmvTrans::ConjTrans => (m, n),
    };

    if x.len() != x_len {
        return Err(GbmvError::DimensionMismatchX);
    }
    if y.len() != y_len {
        return Err(GbmvError::DimensionMismatchY);
    }

    // Handle empty matrix
    if m == 0 || n == 0 {
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

    match trans {
        GbmvTrans::NoTrans => {
            // y = alpha * A * x + beta * y
            // For each row i of A, compute the dot product with x
            for i in 0..m {
                let mut temp = T::zero();

                // Column range for row i: j in [max(0, i-kl), min(n-1, i+ku)]
                let j_start = i.saturating_sub(kl);
                let j_end = (i + ku + 1).min(n);

                for j in j_start..j_end {
                    // ab[ku + i - j, j] contains A[i, j]
                    let ab_row = ku + i - j;
                    temp += ab[(ab_row, j)] * x[j];
                }

                y[i] += alpha * temp;
            }
        }
        GbmvTrans::Trans => {
            // y = alpha * A^T * x + beta * y
            // y[j] = sum_i A[i,j] * x[i], so iterate over rows i of A
            for i in 0..m {
                let alpha_xi = alpha * x[i];

                // Column range for row i: j in [max(0, i-kl), min(n-1, i+ku)]
                let j_start = i.saturating_sub(kl);
                let j_end = (i + ku + 1).min(n);

                for j in j_start..j_end {
                    // ab[ku + i - j, j] contains A[i, j]
                    let ab_row = ku + i - j;
                    y[j] += alpha_xi * ab[(ab_row, j)];
                }
            }
        }
        GbmvTrans::ConjTrans => {
            // y = alpha * A^H * x + beta * y
            // y[j] = sum_i conj(A[i,j]) * x[i]
            for i in 0..m {
                let alpha_xi = alpha * x[i];

                let j_start = i.saturating_sub(kl);
                let j_end = (i + ku + 1).min(n);

                for j in j_start..j_end {
                    let ab_row = ku + i - j;
                    y[j] += alpha_xi * ab[(ab_row, j)].conj();
                }
            }
        }
    }

    Ok(())
}

/// General band matrix-vector multiply that allocates the result.
///
/// Computes y = α·op(A)·x where A is an m×n band matrix.
///
/// # Arguments
///
/// * `trans` - The operation to apply
/// * `m` - Number of rows of the original matrix A
/// * `n` - Number of columns of the original matrix A
/// * `kl` - Number of subdiagonals
/// * `ku` - Number of superdiagonals
/// * `alpha` - Scalar multiplier
/// * `ab` - Band matrix in band storage format
/// * `x` - Input vector
///
/// # Returns
///
/// A new vector containing α·op(A)·x.
pub fn gbmv_new<T: Field>(
    trans: GbmvTrans,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    ab: MatRef<'_, T>,
    x: &[T],
) -> Result<Vec<T>, GbmvError> {
    let y_len = match trans {
        GbmvTrans::NoTrans => m,
        GbmvTrans::Trans | GbmvTrans::ConjTrans => n,
    };
    let mut y = vec![T::zero(); y_len];
    gbmv(trans, m, n, kl, ku, alpha, ab, x, T::zero(), &mut y)?;
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
    fn test_gbmv_tridiagonal() {
        // 4×4 tridiagonal matrix (kl=1, ku=1):
        // A = [[1, 2, 0, 0],
        //      [3, 4, 5, 0],
        //      [0, 6, 7, 8],
        //      [0, 0, 9, 10]]
        let ab = Mat::from_rows(&[
            &[0.0f64, 2.0, 5.0, 8.0],
            &[1.0, 4.0, 7.0, 10.0],
            &[3.0, 6.0, 9.0, 0.0],
        ]);
        let x = [1.0f64, 1.0, 1.0, 1.0];
        let mut y = [0.0f64; 4];

        gbmv(
            GbmvTrans::NoTrans,
            4,
            4,
            1,
            1,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // y = A * [1,1,1,1] = [3, 12, 21, 19]
        assert!(approx_eq(y[0], 3.0));
        assert!(approx_eq(y[1], 12.0));
        assert!(approx_eq(y[2], 21.0));
        assert!(approx_eq(y[3], 19.0));
    }

    #[test]
    fn test_gbmv_transpose() {
        // Same tridiagonal matrix
        let ab = Mat::from_rows(&[
            &[0.0f64, 2.0, 5.0, 8.0],
            &[1.0, 4.0, 7.0, 10.0],
            &[3.0, 6.0, 9.0, 0.0],
        ]);
        let x = [1.0f64, 1.0, 1.0, 1.0];
        let mut y = [0.0f64; 4];

        gbmv(
            GbmvTrans::Trans,
            4,
            4,
            1,
            1,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // y = A^T * [1,1,1,1]
        // A^T = [[1, 3, 0, 0],
        //        [2, 4, 6, 0],
        //        [0, 5, 7, 9],
        //        [0, 0, 8, 10]]
        // y = [1+3, 2+4+6, 5+7+9, 8+10] = [4, 12, 21, 18]
        assert!(approx_eq(y[0], 4.0));
        assert!(approx_eq(y[1], 12.0));
        assert!(approx_eq(y[2], 21.0));
        assert!(approx_eq(y[3], 18.0));
    }

    #[test]
    fn test_gbmv_diagonal_only() {
        // Diagonal matrix (kl=0, ku=0)
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            0,
            0,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // y = diag(2,3,4) * [1,2,3] = [2, 6, 12]
        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 6.0));
        assert!(approx_eq(y[2], 12.0));
    }

    #[test]
    fn test_gbmv_upper_bidiagonal() {
        // Upper bidiagonal (kl=0, ku=1)
        // A = [[1, 2, 0],
        //      [0, 3, 4],
        //      [0, 0, 5]]
        let ab = Mat::from_rows(&[
            &[0.0f64, 2.0, 4.0], // superdiagonal
            &[1.0, 3.0, 5.0],    // main diagonal
        ]);
        let x = [1.0f64, 1.0, 1.0];
        let mut y = [0.0f64; 3];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            0,
            1,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // y = [1+2, 3+4, 5] = [3, 7, 5]
        assert!(approx_eq(y[0], 3.0));
        assert!(approx_eq(y[1], 7.0));
        assert!(approx_eq(y[2], 5.0));
    }

    #[test]
    fn test_gbmv_lower_bidiagonal() {
        // Lower bidiagonal (kl=1, ku=0)
        // A = [[1, 0, 0],
        //      [2, 3, 0],
        //      [0, 4, 5]]
        let ab = Mat::from_rows(&[
            &[1.0f64, 3.0, 5.0], // main diagonal
            &[2.0, 4.0, 0.0],    // subdiagonal
        ]);
        let x = [1.0f64, 1.0, 1.0];
        let mut y = [0.0f64; 3];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            1,
            0,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // y = [1, 2+3, 4+5] = [1, 5, 9]
        assert!(approx_eq(y[0], 1.0));
        assert!(approx_eq(y[1], 5.0));
        assert!(approx_eq(y[2], 9.0));
    }

    #[test]
    fn test_gbmv_with_alpha_beta() {
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [1.0f64, 1.0, 1.0];

        // y = 2*A*x + 3*y
        // A*x = [2, 6, 12]
        // y = 2*[2,6,12] + 3*[1,1,1] = [4+3, 12+3, 24+3] = [7, 15, 27]
        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            0,
            0,
            2.0,
            ab.as_ref(),
            &x,
            3.0,
            &mut y,
        )
        .unwrap();

        assert!(approx_eq(y[0], 7.0));
        assert!(approx_eq(y[1], 15.0));
        assert!(approx_eq(y[2], 27.0));
    }

    #[test]
    fn test_gbmv_rectangular() {
        // 3×4 band matrix with kl=1, ku=1
        // A = [[1, 2, 0, 0],
        //      [3, 4, 5, 0],
        //      [0, 6, 7, 8]]
        let ab = Mat::from_rows(&[
            &[0.0f64, 2.0, 5.0, 8.0],
            &[1.0, 4.0, 7.0, 0.0],
            &[3.0, 6.0, 0.0, 0.0],
        ]);
        let x = [1.0f64, 1.0, 1.0, 1.0];
        let mut y = [0.0f64; 3];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            4,
            1,
            1,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // y = [1+2, 3+4+5, 6+7+8] = [3, 12, 21]
        assert!(approx_eq(y[0], 3.0));
        assert!(approx_eq(y[1], 12.0));
        assert!(approx_eq(y[2], 21.0));
    }

    #[test]
    fn test_gbmv_complex() {
        // Complex tridiagonal
        let ab = Mat::from_rows(&[
            &[c(0.0, 0.0), c(1.0, 1.0), c(2.0, 0.0)],
            &[c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)],
            &[c(1.0, -1.0), c(1.0, 0.0), c(0.0, 0.0)],
        ]);
        let x = [c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            1,
            1,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // Row 0: 1 + (1+i) = 2+i
        // Row 1: (1-i) + 2 + 2 = 5-i
        // Row 2: 1 + 3 = 4
        assert!(approx_eq_c(y[0], c(2.0, 1.0)));
        assert!(approx_eq_c(y[1], c(5.0, -1.0)));
        assert!(approx_eq_c(y[2], c(4.0, 0.0)));
    }

    #[test]
    fn test_gbmv_conj_trans() {
        // Complex matrix for conjugate transpose test
        let ab = Mat::from_rows(&[
            &[c(0.0, 0.0), c(1.0, 1.0)],
            &[c(2.0, 0.0), c(3.0, 0.0)],
            &[c(1.0, 2.0), c(0.0, 0.0)],
        ]);
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        gbmv(
            GbmvTrans::ConjTrans,
            2,
            2,
            1,
            1,
            c(1.0, 0.0),
            ab.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // A^H * x where A = [[2, 1+i], [1+2i, 3]]
        // A^H = [[2, 1-2i], [1-i, 3]]
        // y[0] = 2*1 + (1-2i)*1 = 3-2i
        // y[1] = (1-i)*1 + 3*1 = 4-i
        assert!(approx_eq_c(y[0], c(3.0, -2.0)));
        assert!(approx_eq_c(y[1], c(4.0, -1.0)));
    }

    #[test]
    fn test_gbmv_alpha_zero() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [1.0f64, 2.0, 3.0];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            0,
            0,
            0.0,
            ab.as_ref(),
            &x,
            2.0,
            &mut y,
        )
        .unwrap();

        // y = 0*A*x + 2*y = 2*[1,2,3] = [2,4,6]
        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 4.0));
        assert!(approx_eq(y[2], 6.0));
    }

    #[test]
    fn test_gbmv_beta_zero() {
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 1.0, 1.0];
        let mut y = [100.0f64, 200.0, 300.0];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            0,
            0,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // y is overwritten: [2, 3, 4]
        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 3.0));
        assert!(approx_eq(y[2], 4.0));
    }

    #[test]
    fn test_gbmv_1x1() {
        let ab = Mat::from_rows(&[&[5.0f64]]);
        let x = [3.0f64];
        let mut y = [0.0f64];

        gbmv(
            GbmvTrans::NoTrans,
            1,
            1,
            0,
            0,
            2.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        assert!(approx_eq(y[0], 30.0)); // 2*5*3
    }

    #[test]
    fn test_gbmv_empty() {
        let ab: Mat<f64> = Mat::zeros(1, 0);
        let x: [f64; 0] = [];
        let mut y: [f64; 0] = [];

        let result = gbmv(
            GbmvTrans::NoTrans,
            0,
            0,
            0,
            0,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gbmv_dimension_mismatch_x() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let x = [1.0f64, 2.0]; // Wrong size
        let mut y = [0.0f64; 3];

        let result = gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            0,
            0,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        );
        assert!(matches!(result, Err(GbmvError::DimensionMismatchX)));
    }

    #[test]
    fn test_gbmv_dimension_mismatch_y() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 2]; // Wrong size

        let result = gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            0,
            0,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        );
        assert!(matches!(result, Err(GbmvError::DimensionMismatchY)));
    }

    #[test]
    fn test_gbmv_invalid_dimensions() {
        let ab = Mat::from_rows(&[&[1.0f64, 2.0]]); // Only 1 row, but kl+ku+1=3
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        let result = gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            1,
            1,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        );
        assert!(matches!(result, Err(GbmvError::InvalidDimensions)));
    }

    #[test]
    fn test_gbmv_f32() {
        let ab = Mat::from_rows(&[&[0.0f32, 2.0, 4.0], &[1.0, 3.0, 5.0], &[2.0, 4.0, 0.0]]);
        let x = [1.0f32, 1.0, 1.0];
        let mut y = [0.0f32; 3];

        gbmv(
            GbmvTrans::NoTrans,
            3,
            3,
            1,
            1,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        assert!((y[0] - 3.0).abs() < 1e-5); // 1+2
        assert!((y[1] - 9.0).abs() < 1e-5); // 2+3+4
        assert!((y[2] - 9.0).abs() < 1e-5); // 4+5
    }

    #[test]
    fn test_gbmv_new() {
        let ab = Mat::from_rows(&[&[2.0f64, 3.0, 4.0]]);
        let x = [1.0f64, 2.0, 3.0];

        let y = gbmv_new(GbmvTrans::NoTrans, 3, 3, 0, 0, 1.0, ab.as_ref(), &x).unwrap();

        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 6.0));
        assert!(approx_eq(y[2], 12.0));
    }

    #[test]
    fn test_gbmv_pentadiagonal() {
        // 5×5 pentadiagonal matrix (kl=2, ku=2)
        // More complex band structure
        let ab = Mat::from_rows(&[
            &[0.0f64, 0.0, 1.0, 2.0, 3.0],  // 2nd superdiagonal
            &[0.0, 4.0, 5.0, 6.0, 7.0],     // 1st superdiagonal
            &[8.0, 9.0, 10.0, 11.0, 12.0],  // main diagonal
            &[13.0, 14.0, 15.0, 16.0, 0.0], // 1st subdiagonal
            &[17.0, 18.0, 19.0, 0.0, 0.0],  // 2nd subdiagonal
        ]);
        let x = [1.0f64; 5];
        let mut y = [0.0f64; 5];

        gbmv(
            GbmvTrans::NoTrans,
            5,
            5,
            2,
            2,
            1.0,
            ab.as_ref(),
            &x,
            0.0,
            &mut y,
        )
        .unwrap();

        // Row 0: 8 + 4 + 1 = 13
        // Row 1: 13 + 9 + 5 + 2 = 29
        // Row 2: 17 + 14 + 10 + 6 + 3 = 50
        // Row 3: 18 + 15 + 11 + 7 = 51
        // Row 4: 19 + 16 + 12 = 47
        assert!(approx_eq(y[0], 13.0));
        assert!(approx_eq(y[1], 29.0));
        assert!(approx_eq(y[2], 50.0));
        assert!(approx_eq(y[3], 51.0));
        assert!(approx_eq(y[4], 47.0));
    }
}
