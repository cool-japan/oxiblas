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
/// The other triangle is not referenced. As in reference BLAS `?syr`, column `j`
/// is updated as `A[i,j] += x[i] * (alpha * x[j])`, and a column with
/// `x[j] == 0` is skipped entirely, so an infinite or NaN `x[i]` cannot reach it
/// as `x[i] * 0`. If `n == 0` or `alpha == 0` the matrix is left untouched.
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
                // Reference ?SYR skips the column when x(j) == 0, so an infinite
                // or NaN x(i) cannot reach it as `x(i) * (alpha * 0)`.
                if x[j] == T::zero() {
                    continue;
                }
                let alpha_xj = alpha * x[j];
                for i in j..n {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i, j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
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
                // Column skip for x(j) == 0, as in the lower branch.
                if x[j] == T::zero() {
                    continue;
                }
                let alpha_xj = alpha * x[j];
                for i in 0..=j {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i <= j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
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
/// Returns alpha * x * x^T as a fully symmetric matrix.
///
/// The lower triangle is computed by [`syr`] with [`SyrUplo::Lower`] on a zero
/// matrix, so it has the reference `?syr` semantics (`A[i,j] = 0 + x[i] *
/// (alpha * x[j])`), and the strictly upper triangle mirrors it,
/// `A[j,i] = A[i,j]`. In particular:
///
/// * A column with `x[j] == 0` is skipped and stays zero, so an infinite or NaN
///   `x[i]` cannot reach it. Like the reference this is order dependent:
///   `x = [0, Inf]` gives `A[1,0] = A[0,1] = 0`, but `x = [Inf, 0]` gives
///   `A[1,0] = 0 * (alpha * Inf) = NaN`, mirrored into `A[0,1]`.
/// * Every entry is accumulated onto `+0`, so a `-0` product is stored as `+0`.
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

    // `a` is `n x n` and `x.len() == n`, so `syr` cannot fail. Should a future
    // dimension bug make it fail anyway, debug builds stop at the assertion;
    // release builds return the zero matrix (`syr` reports its errors before
    // writing) instead of panicking.
    let updated = syr(SyrUplo::Lower, alpha, x, a.as_mut());
    debug_assert!(
        updated.is_ok(),
        "syr_new: syr rejected an {n} x {n} matrix with x.len() == {n}: {updated:?}"
    );
    if updated.is_ok() {
        // Mirror the strictly lower triangle into the upper one.
        for j in 0..n {
            for i in (j + 1)..n {
                a[(j, i)] = a[(i, j)];
            }
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

    /// Issue #14: reference ?SYR skips column `j` when `x(j) == 0`, so an infinite
    /// `x(i)` cannot reach it as `x(i) * (alpha * 0) = NaN`.
    #[test]
    fn test_issue_14_syr_zero_column_skip() {
        let inf = f64::INFINITY;

        // Lower: column 0 (x0 == 0) holds row 1, where x1 is infinite.
        let x = [0.0, inf, 2.0];
        let mut a = Mat::from_rows(&[&[1.0f64, 9.0, 9.0], &[0.5, 2.0, 9.0], &[0.25, 0.75, 3.0]]);
        let before = a.clone();
        syr(SyrUplo::Lower, 1.5, &x, a.as_mut()).unwrap();
        for i in 0..3 {
            assert_eq!(
                a[(i, 0)].to_bits(),
                before[(i, 0)].to_bits(),
                "Lower ({i},0)"
            );
        }
        // Column 1: temp = 1.5 * Inf, so A11 and A21 become Inf.
        assert_eq!(a[(1, 1)], inf);
        assert_eq!(a[(2, 1)], inf);
        assert_eq!(a[(2, 2)], 3.0 + (1.5 * 2.0) * 2.0);
        assert_eq!((a[(0, 1)], a[(0, 2)], a[(1, 2)]), (9.0, 9.0, 9.0));

        // Upper: column 2 (x2 == -0) holds row 1, where x1 is infinite.
        let x = [2.0, inf, -0.0];
        let mut a = Mat::from_rows(&[&[1.0f64, 0.5, 0.25], &[9.0, 2.0, 0.75], &[9.0, 9.0, 3.0]]);
        let before = a.clone();
        syr(SyrUplo::Upper, 1.5, &x, a.as_mut()).unwrap();
        for i in 0..3 {
            assert_eq!(
                a[(i, 2)].to_bits(),
                before[(i, 2)].to_bits(),
                "Upper ({i},2)"
            );
        }
        assert_eq!(a[(0, 1)], inf);
        assert_eq!(a[(1, 1)], inf);
        assert_eq!(a[(0, 0)], 1.0 + (1.5 * 2.0) * 2.0);
        assert_eq!((a[(1, 0)], a[(2, 0)], a[(2, 1)]), (9.0, 9.0, 9.0));
    }

    /// `syr_new` builds its lower triangle with `syr(Lower, ..)` on a zero matrix, so
    /// the reference zero-column skip applies: for `x = [0, Inf]` column 0 is
    /// skipped, `A[1,0]` stays `0` and is mirrored into `A[0,1]`. (The old direct
    /// product `alpha * x[1] * x[0] = Inf * 0` made both NaN.)
    #[test]
    fn test_syr_new_skips_zero_column() {
        let inf = f64::INFINITY;

        let a = syr_new(1.0, &[0.0, inf]);
        assert_eq!((a[(0, 0)], a[(1, 0)], a[(0, 1)]), (0.0, 0.0, 0.0));
        assert_eq!(a[(1, 1)], inf);

        // The lower-triangle semantics are order dependent, as in the reference:
        // with x = [Inf, 0] column 0 is not skipped and A[1,0] = x[1] * (alpha *
        // x[0]) = 0 * Inf is NaN (mirrored), while the skipped column 1 stays zero.
        let a = syr_new(1.0, &[inf, 0.0]);
        assert_eq!(a[(0, 0)], inf);
        assert!(a[(1, 0)].is_nan() && a[(0, 1)].is_nan());
        assert_eq!(a[(1, 1)], 0.0);
    }

    /// Every lower entry of `syr_new` is `0 + product` (a skipped column stays `+0`),
    /// so a `-0` product is stored as `+0`. The old direct products kept the `-0`.
    #[test]
    fn test_syr_new_negative_zero_product_is_positive_zero() {
        let a = syr_new(-1.0, &[1.0f64, 0.0]);
        assert_eq!(a[(0, 0)], -1.0);
        for (i, j) in [(1, 0), (0, 1), (1, 1)] {
            assert_eq!(a[(i, j)].to_bits(), 0.0f64.to_bits(), "({i},{j})");
        }
    }

    /// `syr_new` equals `syr(Lower, ..)` on a zero matrix bit for bit (reference
    /// temp `alpha * x[j]` and evaluation order), mirrored into the upper triangle.
    #[test]
    fn test_syr_new_matches_lower_syr_on_zero_matrix() {
        let x = [1.25f64, -0.3, 0.0, 2.0e-3, 7.5];
        let alpha = -0.37;
        let a = syr_new(alpha, &x);
        let mut lower: Mat<f64> = Mat::zeros(5, 5);
        syr(SyrUplo::Lower, alpha, &x, lower.as_mut()).unwrap();
        for j in 0..5 {
            for i in j..5 {
                assert_eq!(a[(i, j)].to_bits(), lower[(i, j)].to_bits(), "({i},{j})");
                assert_eq!(a[(j, i)].to_bits(), lower[(i, j)].to_bits(), "({j},{i})");
            }
        }
    }
}
