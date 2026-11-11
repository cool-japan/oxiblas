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
/// The other triangle is not referenced. As in reference BLAS `?syr2`, column `j`
/// is updated as `A[i,j] = A[i,j] + x[i] * (alpha * y[j]) + y[i] * (alpha * x[j])`
/// (evaluated left to right), and a column with `x[j]` and `y[j]` both zero is
/// skipped entirely, so an infinite or NaN neighbour cannot reach it. If
/// `n == 0` or `alpha == 0` the matrix is left untouched.
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
                // Reference ?SYR2 skips the column when x(j) == 0 and y(j) == 0, so
                // an infinite or NaN neighbour cannot reach it as `Inf * 0`.
                if x[j] == T::zero() && y[j] == T::zero() {
                    continue;
                }
                // Reference temps and evaluation order:
                // A(i,j) + x(i)*temp1 + y(i)*temp2, left to right.
                let temp1 = alpha * y[j];
                let temp2 = alpha * x[j];
                for i in j..n {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i, j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        *ptr = *ptr + x[i] * temp1 + y[i] * temp2;
                    }
                }
            }
        }
        Syr2Uplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * (x[i]*y[j] + y[i]*x[j]) for i <= j
            for j in 0..n {
                // Column skip for x(j) == 0 && y(j) == 0, as in the lower branch.
                if x[j] == T::zero() && y[j] == T::zero() {
                    continue;
                }
                let temp1 = alpha * y[j];
                let temp2 = alpha * x[j];
                for i in 0..=j {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i <= j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        *ptr = *ptr + x[i] * temp1 + y[i] * temp2;
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
/// The lower triangle is computed by [`syr2`] with [`Syr2Uplo::Lower`] on a zero
/// matrix, so it has the reference `?syr2` semantics (`A[i,j] = 0 + x[i] *
/// (alpha * y[j]) + y[i] * (alpha * x[j])`, evaluated left to right), and the
/// strictly upper triangle mirrors it, `A[j,i] = A[i,j]`. In particular:
///
/// * A column with `x[j] == 0` and `y[j] == 0` is skipped and stays zero, so an
///   infinite or NaN neighbour cannot reach it. Like the reference this is order
///   dependent: `x = [0, Inf]`, `y = [0, 1]` gives `A[1,0] = A[0,1] = 0`, but
///   `x = [Inf, 0]`, `y = [1, 0]` gives `A[1,0] = NaN` (from
///   `y[1] * (alpha * x[0]) = 0 * Inf`), mirrored into `A[0,1]`.
/// * Every entry is accumulated onto `+0`, so a `-0` result (a sum of `-0`
///   products) is stored as `+0`.
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

    // Reference-semantics lower triangle. `a` is `n x n` and both vectors have
    // length `n`, so this cannot fail; an error would be propagated as is.
    syr2(Syr2Uplo::Lower, alpha, x, y, a.as_mut())?;

    // Mirror the strictly lower triangle into the upper one.
    for j in 0..n {
        for i in (j + 1)..n {
            a[(j, i)] = a[(i, j)];
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

    /// Issue #14: reference ?SYR2 skips column `j` when `x(j) == 0` and
    /// `y(j) == 0`, so an infinite neighbour cannot reach it as `Inf * 0 = NaN`. A
    /// column with only one of the two zero is updated.
    #[test]
    fn test_issue_14_syr2_zero_column_skip() {
        let inf = f64::INFINITY;
        let lower = [
            &[1.0f64, 9.0, 9.0][..],
            &[0.5, 2.0, 9.0],
            &[0.25, 0.75, 3.0],
        ];

        // Lower: column 0 (x0 == y0 == 0) holds row 1, where x1 is infinite.
        let mut a = Mat::from_rows(&lower);
        let before = a.clone();
        let (x, y) = ([0.0, inf, 1.0], [-0.0, 2.0, 3.0]);
        syr2(Syr2Uplo::Lower, 1.5, &x, &y, a.as_mut()).unwrap();
        for i in 0..3 {
            assert_eq!(
                a[(i, 0)].to_bits(),
                before[(i, 0)].to_bits(),
                "Lower ({i},0)"
            );
        }
        // Column 1: temp2 = 1.5 * x1 = Inf reaches A11 and A21.
        assert_eq!((a[(1, 1)], a[(2, 1)]), (inf, inf));
        // Column 2: A22 = 3 + x2 * (1.5 * y2) + y2 * (1.5 * x2).
        assert_eq!(a[(2, 2)], 3.0 + 1.0 * 4.5 + 3.0 * 1.5);
        assert_eq!((a[(0, 1)], a[(0, 2)], a[(1, 2)]), (9.0, 9.0, 9.0));

        // Only x0 == 0: column 0 is updated, A10 = 0.5 + x1 * (1.5 * 5) + y1 * 0.
        let mut a = Mat::from_rows(&lower);
        syr2(Syr2Uplo::Lower, 1.5, &x, &[5.0, 2.0, 3.0], a.as_mut()).unwrap();
        assert_eq!(a[(1, 0)], inf);

        // Upper: column 2 (x2 == y2 == 0) holds row 1, where x1 is infinite.
        let mut a = Mat::from_rows(&[&[1.0f64, 0.5, 0.25], &[9.0, 2.0, 0.75], &[9.0, 9.0, 3.0]]);
        let before = a.clone();
        let (x, y) = ([1.0, inf, 0.0], [3.0, 2.0, -0.0]);
        syr2(Syr2Uplo::Upper, 1.5, &x, &y, a.as_mut()).unwrap();
        for i in 0..3 {
            assert_eq!(
                a[(i, 2)].to_bits(),
                before[(i, 2)].to_bits(),
                "Upper ({i},2)"
            );
        }
        assert_eq!((a[(0, 1)], a[(1, 1)]), (inf, inf));
        assert_eq!(a[(0, 0)], 1.0 + 1.0 * 4.5 + 3.0 * 1.5);
        assert_eq!((a[(1, 0)], a[(2, 0)], a[(2, 1)]), (9.0, 9.0, 9.0));
    }

    /// Issue #14: the element update follows reference ?SYR2,
    /// `A(i,j) + x(i)*(alpha*y(j)) + y(i)*(alpha*x(j))`, evaluated left to right.
    /// With `A10 = 1`, `x1*temp1 = 2^-53` rounds away and `y1*temp2 = 2^-52`
    /// survives, giving `1 + 2^-52`; adding the `y` term first would round the sum
    /// up to `1 + 2^-51`.
    #[test]
    fn test_issue_14_syr2_reference_evaluation_order() {
        let e = f64::EPSILON;
        let mut a = Mat::from_rows(&[&[1.0f64, 9.0], &[1.0, 1.0]]);
        syr2(Syr2Uplo::Lower, 1.0, &[1.0, 1.0], &[e / 2.0, e], a.as_mut()).unwrap();
        assert_eq!(a[(1, 0)], 1.0 + e);
    }

    /// `syr2_new` builds its lower triangle with `syr2(Lower, ..)` on a zero matrix,
    /// so the reference zero-column skip applies: for `x = [0, Inf]`, `y = [0, 1]`
    /// column 0 is skipped, `A[1,0]` stays `0` and is mirrored into `A[0,1]`. (The
    /// old direct product `alpha * (x[1] * y[0] + y[1] * x[0]) = Inf * 0 + 0` made
    /// both NaN.)
    #[test]
    fn test_syr2_new_skips_zero_column() {
        let inf = f64::INFINITY;

        let a = syr2_new(1.0, &[0.0, inf], &[0.0, 1.0]).unwrap();
        assert_eq!((a[(0, 0)], a[(1, 0)], a[(0, 1)]), (0.0, 0.0, 0.0));
        assert_eq!(a[(1, 1)], inf);

        // The lower-triangle semantics are order dependent, as in the reference:
        // with x = [Inf, 0], y = [1, 0] column 0 is not skipped and A[1,0] picks up
        // `y[1] * (alpha * x[0]) = 0 * Inf` (NaN, mirrored), while the skipped
        // column 1 stays zero.
        let a = syr2_new(1.0, &[inf, 0.0], &[1.0, 0.0]).unwrap();
        assert_eq!(a[(0, 0)], inf);
        assert!(a[(1, 0)].is_nan() && a[(0, 1)].is_nan());
        assert_eq!(a[(1, 1)], 0.0);
    }

    /// Every lower entry of `syr2_new` is `(0 + p1) + p2` (a skipped column stays
    /// `+0`), so a `-0` sum of products is stored as `+0`. The old direct products
    /// kept the `-0`.
    #[test]
    fn test_syr2_new_negative_zero_product_is_positive_zero() {
        let a = syr2_new(-1.0, &[1.0f64, 0.0], &[1.0, 0.0]).unwrap();
        assert_eq!(a[(0, 0)], -2.0);
        for (i, j) in [(1, 0), (0, 1), (1, 1)] {
            assert_eq!(a[(i, j)].to_bits(), 0.0f64.to_bits(), "({i},{j})");
        }
    }

    /// `syr2_new` equals `syr2(Lower, ..)` on a zero matrix bit for bit (reference
    /// temps and evaluation order), mirrored into the upper triangle; mismatched
    /// lengths still report `DimensionMismatchY`.
    #[test]
    fn test_syr2_new_matches_lower_syr2_on_zero_matrix() {
        let x = [1.25f64, -0.3, 0.0, 2.0e-3, 7.5];
        let y = [0.5f64, 0.0, 0.0, -4.25, 1.0e-3];
        let alpha = -0.37;
        let a = syr2_new(alpha, &x, &y).unwrap();
        let mut lower: Mat<f64> = Mat::zeros(5, 5);
        syr2(Syr2Uplo::Lower, alpha, &x, &y, lower.as_mut()).unwrap();
        for j in 0..5 {
            for i in j..5 {
                assert_eq!(a[(i, j)].to_bits(), lower[(i, j)].to_bits(), "({i},{j})");
                assert_eq!(a[(j, i)].to_bits(), lower[(i, j)].to_bits(), "({j},{i})");
            }
        }
        assert_eq!(
            syr2_new(alpha, &x, &y[..4]).map(|m| m.nrows()),
            Err(Syr2Error::DimensionMismatchY)
        );
    }
}
