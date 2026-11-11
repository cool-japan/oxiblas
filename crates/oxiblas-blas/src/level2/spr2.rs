//! SPR2: Symmetric packed rank-2 update.
//!
//! Performs the symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
//! where A is a symmetric matrix stored in packed format.
//!
//! # Packed Storage Format
//!
//! For an n×n symmetric matrix, only n*(n+1)/2 elements are stored.
//!
//! **Upper triangle (column-major):**
//! ```text
//! ap = [a11, a12, a22, a13, a23, a33, ...]
//! Index: ap[i + j*(j+1)/2] = a(i,j) for 0 <= i <= j
//! ```
//!
//! **Lower triangle (column-major):**
//! ```text
//! ap = [a11, a21, a31, ..., a22, a32, ..., a33, ...]
//! Index: ap[i + (2*n - j - 1)*j/2] = a(i,j) for i >= j
//! ```

use oxiblas_core::scalar::Real;

/// Specifies which triangle of the matrix is stored in packed format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spr2Uplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for symmetric packed rank-2 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spr2Error {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for Spr2Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPackedSize => {
                write!(f, "Packed array has wrong size (expected n*(n+1)/2)")
            }
            Self::DimensionMismatchX => write!(f, "Vector x has wrong dimension"),
            Self::DimensionMismatchY => write!(f, "Vector y has wrong dimension"),
        }
    }
}

impl std::error::Error for Spr2Error {}

/// Computes the index into upper packed array for element (i, j) where i <= j.
#[inline]
fn upper_packed_index(i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    i + j * (j + 1) / 2
}

/// Computes the index into lower packed array for element (i, j) where i >= j.
#[inline]
fn lower_packed_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i >= j);
    i + (2 * n - j - 1) * j / 2
}

/// Performs the symmetric packed rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// Only the specified triangle (upper or lower) is stored and updated. As in
/// reference BLAS `?spr2`, column `j` is updated as
/// `A[i,j] = A[i,j] + x[i] * (alpha * y[j]) + y[i] * (alpha * x[j])` (evaluated
/// left to right), and a column with `x[j]` and `y[j]` both zero is skipped
/// entirely, so an infinite or NaN neighbour cannot reach it. If `n == 0` or
/// `alpha == 0` the packed array is left untouched.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `n` - Order of the symmetric matrix A
/// * `alpha` - The scalar alpha
/// * `x` - The first vector x of length n
/// * `y` - The second vector y of length n
/// * `ap` - The packed symmetric matrix (n*(n+1)/2 elements)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{spr2, Spr2Uplo};
///
/// // n=2: need 2*3/2 = 3 elements
/// let mut ap = [0.0f64; 3];
/// let x = [1.0, 2.0];
/// let y = [3.0, 4.0];
///
/// // A = A + alpha * (x * y^T + y * x^T)
/// spr2(Spr2Uplo::Upper, 2, 1.0, &x, &y, &mut ap).unwrap();
///
/// // x*y^T + y*x^T = [[2*1*3, 1*4+2*3], [2*1+3*2, 2*2*4]] = [[6, 10], [10, 16]]
/// // Upper packed: [6, 10, 16]
/// assert!((ap[0] - 6.0).abs() < 1e-10);
/// assert!((ap[1] - 10.0).abs() < 1e-10);
/// assert!((ap[2] - 16.0).abs() < 1e-10);
/// ```
pub fn spr2<T: Real>(
    uplo: Spr2Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    y: &[T],
    ap: &mut [T],
) -> Result<(), Spr2Error> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(Spr2Error::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(Spr2Error::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(Spr2Error::DimensionMismatchY);
    }

    if n == 0 || alpha == T::zero() {
        return Ok(());
    }

    match uplo {
        Spr2Uplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * (x[i]*y[j] + y[i]*x[j]) for i <= j
            for j in 0..n {
                // Reference ?SPR2 skips the column when x(j) == 0 and y(j) == 0, so
                // an infinite or NaN neighbour cannot reach it as `Inf * 0`.
                if x[j] == T::zero() && y[j] == T::zero() {
                    continue;
                }
                // Reference temps and evaluation order:
                // AP(k) + x(i)*temp1 + y(i)*temp2, left to right.
                let temp1 = alpha * y[j];
                let temp2 = alpha * x[j];
                for i in 0..=j {
                    let kij = upper_packed_index(i, j);
                    ap[kij] = ap[kij] + x[i] * temp1 + y[i] * temp2;
                }
            }
        }
        Spr2Uplo::Lower => {
            // Update lower triangle: A[i,j] += alpha * (x[i]*y[j] + y[i]*x[j]) for i >= j
            for j in 0..n {
                // Column skip for x(j) == 0 && y(j) == 0, as in the upper branch.
                if x[j] == T::zero() && y[j] == T::zero() {
                    continue;
                }
                let temp1 = alpha * y[j];
                let temp2 = alpha * x[j];
                for i in j..n {
                    let kij = lower_packed_index(i, j, n);
                    ap[kij] = ap[kij] + x[i] * temp1 + y[i] * temp2;
                }
            }
        }
    }

    Ok(())
}

/// Creates a new packed symmetric matrix from alpha * (x * y^T + y * x^T).
///
/// # Arguments
///
/// * `uplo` - Specifies whether to store upper or lower triangle
/// * `alpha` - The scalar alpha
/// * `x` - The first vector x
/// * `y` - The second vector y
///
/// # Returns
///
/// A new packed array containing alpha * (x * y^T + y * x^T).
pub fn spr2_new<T: Real>(uplo: Spr2Uplo, alpha: T, x: &[T], y: &[T]) -> Result<Vec<T>, Spr2Error> {
    let n = x.len();
    if y.len() != n {
        return Err(Spr2Error::DimensionMismatchY);
    }

    let packed_size = n * (n + 1) / 2;
    let mut ap = vec![T::zero(); packed_size];

    if n == 0 || alpha == T::zero() {
        return Ok(ap);
    }

    spr2(uplo, n, alpha, x, y, &mut ap)?;
    Ok(ap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_spr2_upper() {
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        spr2(Spr2Uplo::Upper, 2, 1.0, &x, &y, &mut ap).unwrap();

        // x*y^T + y*x^T = [[2*1*3, 1*4+2*3], [3*2+4*1, 2*2*4]]
        //               = [[6, 10], [10, 16]]
        // Upper packed: [a00, a01, a11] = [6, 10, 16]
        assert!(approx_eq(ap[0], 6.0)); // 2*1*3
        assert!(approx_eq(ap[1], 10.0)); // 1*4 + 2*3
        assert!(approx_eq(ap[2], 16.0)); // 2*2*4
    }

    #[test]
    fn test_spr2_lower() {
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        spr2(Spr2Uplo::Lower, 2, 1.0, &x, &y, &mut ap).unwrap();

        // Lower packed: [a00, a10, a11] = [6, 10, 16]
        assert!(approx_eq(ap[0], 6.0));
        assert!(approx_eq(ap[1], 10.0));
        assert!(approx_eq(ap[2], 16.0));
    }

    #[test]
    fn test_spr2_with_alpha() {
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        spr2(Spr2Uplo::Upper, 2, 2.0, &x, &y, &mut ap).unwrap();

        // 2 * (x*y^T + y*x^T) = 2 * [[6, 10], [10, 16]] = [[12, 20], [20, 32]]
        assert!(approx_eq(ap[0], 12.0));
        assert!(approx_eq(ap[1], 20.0));
        assert!(approx_eq(ap[2], 32.0));
    }

    #[test]
    fn test_spr2_accumulate() {
        let mut ap = [1.0f64, 2.0, 3.0];
        let x = [1.0, 1.0];
        let y = [1.0, 1.0];

        spr2(Spr2Uplo::Upper, 2, 1.0, &x, &y, &mut ap).unwrap();

        // x*y^T + y*x^T = [[2, 2], [2, 2]]
        // Result: [1+2, 2+2, 3+2] = [3, 4, 5]
        assert!(approx_eq(ap[0], 3.0));
        assert!(approx_eq(ap[1], 4.0));
        assert!(approx_eq(ap[2], 5.0));
    }

    #[test]
    fn test_spr2_zero_alpha() {
        let mut ap = [1.0f64, 2.0, 3.0];
        let original = ap;
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        spr2(Spr2Uplo::Upper, 2, 0.0, &x, &y, &mut ap).unwrap();

        // Unchanged
        for i in 0..3 {
            assert!(approx_eq(ap[i], original[i]));
        }
    }

    #[test]
    fn test_spr2_x_equals_y() {
        // When x == y, result is 2 * x * x^T
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];

        spr2(Spr2Uplo::Upper, 2, 1.0, &x, &x, &mut ap).unwrap();

        // 2 * x * x^T = [[2, 4], [4, 8]]
        assert!(approx_eq(ap[0], 2.0));
        assert!(approx_eq(ap[1], 4.0));
        assert!(approx_eq(ap[2], 8.0));
    }

    #[test]
    fn test_spr2_dimension_errors() {
        // Wrong packed size
        let mut ap = [0.0f64; 5]; // Should be 6 for n=3
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        assert_eq!(
            spr2(Spr2Uplo::Upper, 3, 1.0, &x, &y, &mut ap),
            Err(Spr2Error::InvalidPackedSize)
        );

        // Wrong x dimension
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0];
        assert_eq!(
            spr2(Spr2Uplo::Upper, 2, 1.0, &x, &y, &mut ap),
            Err(Spr2Error::DimensionMismatchX)
        );

        // Wrong y dimension
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];
        let y = [4.0, 5.0, 6.0];
        assert_eq!(
            spr2(Spr2Uplo::Upper, 2, 1.0, &x, &y, &mut ap),
            Err(Spr2Error::DimensionMismatchY)
        );
    }

    #[test]
    fn test_spr2_empty() {
        let mut ap: [f64; 0] = [];
        let x: [f64; 0] = [];
        let y: [f64; 0] = [];

        spr2(Spr2Uplo::Upper, 0, 1.0, &x, &y, &mut ap).unwrap();
    }

    #[test]
    fn test_spr2_1x1() {
        let mut ap = [0.0f64];
        let x = [2.0];
        let y = [3.0];

        spr2(Spr2Uplo::Upper, 1, 1.0, &x, &y, &mut ap).unwrap();

        // 2*2*3 = 12
        assert!(approx_eq(ap[0], 12.0));
    }

    #[test]
    fn test_spr2_new_upper() {
        let x = [1.0f64, 2.0];
        let y = [3.0, 4.0];
        let ap = spr2_new(Spr2Uplo::Upper, 1.0, &x, &y).unwrap();

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0], 6.0));
        assert!(approx_eq(ap[1], 10.0));
        assert!(approx_eq(ap[2], 16.0));
    }

    #[test]
    fn test_spr2_new_lower() {
        let x = [1.0f64, 2.0];
        let y = [3.0, 4.0];
        let ap = spr2_new(Spr2Uplo::Lower, 1.0, &x, &y).unwrap();

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0], 6.0));
        assert!(approx_eq(ap[1], 10.0));
        assert!(approx_eq(ap[2], 16.0));
    }

    #[test]
    fn test_spr2_negative_alpha() {
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        spr2(Spr2Uplo::Upper, 2, -1.0, &x, &y, &mut ap).unwrap();

        assert!(approx_eq(ap[0], -6.0));
        assert!(approx_eq(ap[1], -10.0));
        assert!(approx_eq(ap[2], -16.0));
    }

    #[test]
    fn test_spr2_f32() {
        let mut ap = [0.0f32; 3];
        let x = [1.0f32, 2.0];
        let y = [3.0, 4.0];

        spr2(Spr2Uplo::Upper, 2, 1.0, &x, &y, &mut ap).unwrap();

        assert!((ap[0] - 6.0).abs() < 1e-5);
        assert!((ap[1] - 10.0).abs() < 1e-5);
        assert!((ap[2] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_spr2_3x3() {
        let mut ap = [0.0f64; 6];
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];

        spr2(Spr2Uplo::Upper, 3, 1.0, &x, &y, &mut ap).unwrap();

        // x*y^T + y*x^T:
        // [0,0] = 2*1*4 = 8
        // [0,1] = 1*5 + 4*2 = 13
        // [1,1] = 2*2*5 = 20
        // [0,2] = 1*6 + 4*3 = 18
        // [1,2] = 2*6 + 5*3 = 27
        // [2,2] = 2*3*6 = 36
        // Upper packed: [8, 13, 20, 18, 27, 36]
        assert!(approx_eq(ap[0], 8.0));
        assert!(approx_eq(ap[1], 13.0));
        assert!(approx_eq(ap[2], 20.0));
        assert!(approx_eq(ap[3], 18.0));
        assert!(approx_eq(ap[4], 27.0));
        assert!(approx_eq(ap[5], 36.0));
    }

    /// Issue #14: reference ?SPR2 skips column `j` when `x(j) == 0` and
    /// `y(j) == 0`, so an infinite neighbour cannot reach it as `Inf * 0 = NaN`. A
    /// column with only one of the two zero is updated.
    #[test]
    fn test_issue_14_spr2_zero_column_skip() {
        let inf = f64::INFINITY;
        let ap0 = [1.0f64, 0.5, 0.25, 2.0, 0.75, 3.0];

        // Lower packed [A00, A10, A20, A11, A21, A22]: column 0 (x0 == y0 == 0)
        // holds row 1, where x1 is infinite.
        let mut ap = ap0;
        let (x, y) = ([0.0, inf, 1.0], [-0.0, 2.0, 3.0]);
        spr2(Spr2Uplo::Lower, 3, 1.5, &x, &y, &mut ap).unwrap();
        for k in 0..3 {
            assert_eq!(ap[k].to_bits(), ap0[k].to_bits(), "Lower slot {k}");
        }
        // Column 1: temp2 = 1.5 * x1 = Inf reaches A11 and A21.
        assert_eq!((ap[3], ap[4]), (inf, inf));
        assert_eq!(ap[5], 3.0 + 1.0 * 4.5 + 3.0 * 1.5);

        // Only x0 == 0: column 0 is updated, A10 = 0.5 + x1 * (1.5 * 5) + y1 * 0.
        let mut ap = ap0;
        spr2(Spr2Uplo::Lower, 3, 1.5, &x, &[5.0, 2.0, 3.0], &mut ap).unwrap();
        assert_eq!(ap[1], inf);

        // Upper packed [A00, A01, A11, A02, A12, A22]: column 2 (x2 == y2 == 0)
        // holds row 1, where x1 is infinite.
        let mut ap = ap0;
        let (x, y) = ([1.0, inf, 0.0], [3.0, 2.0, -0.0]);
        spr2(Spr2Uplo::Upper, 3, 1.5, &x, &y, &mut ap).unwrap();
        for k in 3..6 {
            assert_eq!(ap[k].to_bits(), ap0[k].to_bits(), "Upper slot {k}");
        }
        assert_eq!((ap[1], ap[2]), (inf, inf));
        assert_eq!(ap[0], 1.0 + 1.0 * 4.5 + 3.0 * 1.5);
    }

    /// Issue #14: the element update follows reference ?SPR2,
    /// `AP(k) + x(i)*(alpha*y(j)) + y(i)*(alpha*x(j))`, evaluated left to right (see
    /// the `syr2` twin of this test for the rounding argument).
    #[test]
    fn test_issue_14_spr2_reference_evaluation_order() {
        let e = f64::EPSILON;
        // Lower packed [A00, A10, A11].
        let mut ap = [1.0f64, 1.0, 1.0];
        spr2(Spr2Uplo::Lower, 2, 1.0, &[1.0, 1.0], &[e / 2.0, e], &mut ap).unwrap();
        assert_eq!(ap[1], 1.0 + e);
    }
}
