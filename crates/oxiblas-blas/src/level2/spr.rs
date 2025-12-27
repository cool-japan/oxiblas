//! SPR: Symmetric packed rank-1 update.
//!
//! Performs the symmetric rank-1 update: A = alpha * x * x^T + A
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
pub enum SprUplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for symmetric packed rank-1 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprError {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector has wrong dimension.
    DimensionMismatch,
}

impl core::fmt::Display for SprError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPackedSize => {
                write!(f, "Packed array has wrong size (expected n*(n+1)/2)")
            }
            Self::DimensionMismatch => write!(f, "Vector has wrong dimension"),
        }
    }
}

impl std::error::Error for SprError {}

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

/// Performs the symmetric packed rank-1 update: A = alpha * x * x^T + A
///
/// Only the specified triangle (upper or lower) is stored and updated.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `n` - Order of the symmetric matrix A
/// * `alpha` - The scalar alpha
/// * `x` - The vector x of length n
/// * `ap` - The packed symmetric matrix (n*(n+1)/2 elements)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{spr, SprUplo};
///
/// // n=3: need 3*4/2 = 6 elements
/// // Start with zero packed array (upper)
/// let mut ap = [0.0f64; 6];
/// let x = [1.0, 2.0, 3.0];
///
/// // A = A + x * x^T
/// spr(SprUplo::Upper, 3, 1.0, &x, &mut ap).unwrap();
///
/// // Upper packed: [a00, a01, a11, a02, a12, a22]
/// // x*x^T = [[1,2,3],[2,4,6],[3,6,9]]
/// // Upper: [1, 2, 4, 3, 6, 9]
/// assert!((ap[0] - 1.0).abs() < 1e-10); // a00 = 1*1
/// assert!((ap[1] - 2.0).abs() < 1e-10); // a01 = 1*2
/// assert!((ap[2] - 4.0).abs() < 1e-10); // a11 = 2*2
/// assert!((ap[3] - 3.0).abs() < 1e-10); // a02 = 1*3
/// assert!((ap[4] - 6.0).abs() < 1e-10); // a12 = 2*3
/// assert!((ap[5] - 9.0).abs() < 1e-10); // a22 = 3*3
/// ```
pub fn spr<T: Real>(
    uplo: SprUplo,
    n: usize,
    alpha: T,
    x: &[T],
    ap: &mut [T],
) -> Result<(), SprError> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(SprError::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(SprError::DimensionMismatch);
    }

    if n == 0 || alpha == T::zero() {
        return Ok(());
    }

    match uplo {
        SprUplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * x[i] * x[j] for i <= j
            for j in 0..n {
                let alpha_xj = alpha * x[j];
                for i in 0..=j {
                    let kij = upper_packed_index(i, j);
                    ap[kij] += alpha_xj * x[i];
                }
            }
        }
        SprUplo::Lower => {
            // Update lower triangle: A[i,j] += alpha * x[i] * x[j] for i >= j
            for j in 0..n {
                let alpha_xj = alpha * x[j];
                for i in j..n {
                    let kij = lower_packed_index(i, j, n);
                    ap[kij] += alpha_xj * x[i];
                }
            }
        }
    }

    Ok(())
}

/// Creates a new packed symmetric matrix from x * x^T.
///
/// # Arguments
///
/// * `uplo` - Specifies whether to store upper or lower triangle
/// * `alpha` - The scalar alpha
/// * `x` - The vector x
///
/// # Returns
///
/// A new packed array containing alpha * x * x^T.
pub fn spr_new<T: Real>(uplo: SprUplo, alpha: T, x: &[T]) -> Vec<T> {
    let n = x.len();
    let packed_size = n * (n + 1) / 2;
    let mut ap = vec![T::zero(); packed_size];

    if n == 0 || alpha == T::zero() {
        return ap;
    }

    let _ = spr(uplo, n, alpha, x, &mut ap);
    ap
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_spr_upper() {
        let mut ap = [0.0f64; 6];
        let x = [1.0, 2.0, 3.0];

        spr(SprUplo::Upper, 3, 1.0, &x, &mut ap).unwrap();

        // Upper packed: [a00, a01, a11, a02, a12, a22]
        // x*x^T upper: [1, 2, 4, 3, 6, 9]
        assert!(approx_eq(ap[0], 1.0)); // 1*1
        assert!(approx_eq(ap[1], 2.0)); // 1*2
        assert!(approx_eq(ap[2], 4.0)); // 2*2
        assert!(approx_eq(ap[3], 3.0)); // 1*3
        assert!(approx_eq(ap[4], 6.0)); // 2*3
        assert!(approx_eq(ap[5], 9.0)); // 3*3
    }

    #[test]
    fn test_spr_lower() {
        let mut ap = [0.0f64; 6];
        let x = [1.0, 2.0, 3.0];

        spr(SprUplo::Lower, 3, 1.0, &x, &mut ap).unwrap();

        // Lower packed: [a00, a10, a20, a11, a21, a22]
        // x*x^T lower: [1, 2, 3, 4, 6, 9]
        assert!(approx_eq(ap[0], 1.0)); // 1*1
        assert!(approx_eq(ap[1], 2.0)); // 2*1
        assert!(approx_eq(ap[2], 3.0)); // 3*1
        assert!(approx_eq(ap[3], 4.0)); // 2*2
        assert!(approx_eq(ap[4], 6.0)); // 3*2
        assert!(approx_eq(ap[5], 9.0)); // 3*3
    }

    #[test]
    fn test_spr_with_alpha() {
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];

        spr(SprUplo::Upper, 2, 2.0, &x, &mut ap).unwrap();

        // Upper packed: [a00, a01, a11]
        // 2 * x*x^T = [2, 4, 8]
        assert!(approx_eq(ap[0], 2.0)); // 2*1*1
        assert!(approx_eq(ap[1], 4.0)); // 2*1*2
        assert!(approx_eq(ap[2], 8.0)); // 2*2*2
    }

    #[test]
    fn test_spr_accumulate() {
        // Start with non-zero packed matrix
        let mut ap = [1.0f64, 2.0, 4.0, 3.0, 6.0, 9.0];
        let x = [1.0, 1.0, 1.0];

        spr(SprUplo::Upper, 3, 1.0, &x, &mut ap).unwrap();

        // All elements increase by 1
        assert!(approx_eq(ap[0], 2.0)); // 1 + 1
        assert!(approx_eq(ap[1], 3.0)); // 2 + 1
        assert!(approx_eq(ap[2], 5.0)); // 4 + 1
        assert!(approx_eq(ap[3], 4.0)); // 3 + 1
        assert!(approx_eq(ap[4], 7.0)); // 6 + 1
        assert!(approx_eq(ap[5], 10.0)); // 9 + 1
    }

    #[test]
    fn test_spr_zero_alpha() {
        let mut ap = [1.0f64, 2.0, 3.0];
        let original = ap;
        let x = [1.0, 2.0];

        spr(SprUplo::Upper, 2, 0.0, &x, &mut ap).unwrap();

        // Unchanged
        for i in 0..3 {
            assert!(approx_eq(ap[i], original[i]));
        }
    }

    #[test]
    fn test_spr_dimension_errors() {
        // Wrong packed size
        let mut ap = [0.0f64; 5]; // Should be 6 for n=3
        let x = [1.0, 2.0, 3.0];
        assert_eq!(
            spr(SprUplo::Upper, 3, 1.0, &x, &mut ap),
            Err(SprError::InvalidPackedSize)
        );

        // Wrong x dimension
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0, 3.0];
        assert_eq!(
            spr(SprUplo::Upper, 2, 1.0, &x, &mut ap),
            Err(SprError::DimensionMismatch)
        );
    }

    #[test]
    fn test_spr_empty() {
        let mut ap: [f64; 0] = [];
        let x: [f64; 0] = [];

        spr(SprUplo::Upper, 0, 1.0, &x, &mut ap).unwrap();
    }

    #[test]
    fn test_spr_1x1() {
        let mut ap = [0.0f64];
        let x = [3.0];

        spr(SprUplo::Upper, 1, 2.0, &x, &mut ap).unwrap();

        assert!(approx_eq(ap[0], 18.0)); // 2 * 3 * 3
    }

    #[test]
    fn test_spr_new_upper() {
        let x = [1.0f64, 2.0, 3.0];
        let ap = spr_new(SprUplo::Upper, 1.0, &x);

        assert_eq!(ap.len(), 6);
        assert!(approx_eq(ap[0], 1.0));
        assert!(approx_eq(ap[1], 2.0));
        assert!(approx_eq(ap[2], 4.0));
        assert!(approx_eq(ap[3], 3.0));
        assert!(approx_eq(ap[4], 6.0));
        assert!(approx_eq(ap[5], 9.0));
    }

    #[test]
    fn test_spr_new_lower() {
        let x = [1.0f64, 2.0, 3.0];
        let ap = spr_new(SprUplo::Lower, 1.0, &x);

        assert_eq!(ap.len(), 6);
        assert!(approx_eq(ap[0], 1.0));
        assert!(approx_eq(ap[1], 2.0));
        assert!(approx_eq(ap[2], 3.0));
        assert!(approx_eq(ap[3], 4.0));
        assert!(approx_eq(ap[4], 6.0));
        assert!(approx_eq(ap[5], 9.0));
    }

    #[test]
    fn test_spr_negative_alpha() {
        let mut ap = [0.0f64; 3];
        let x = [1.0, 2.0];

        spr(SprUplo::Upper, 2, -1.0, &x, &mut ap).unwrap();

        assert!(approx_eq(ap[0], -1.0)); // -1*1
        assert!(approx_eq(ap[1], -2.0)); // -1*2
        assert!(approx_eq(ap[2], -4.0)); // -2*2
    }

    #[test]
    fn test_spr_f32() {
        let mut ap = [0.0f32; 3];
        let x = [2.0f32, 3.0];

        spr(SprUplo::Upper, 2, 1.0, &x, &mut ap).unwrap();

        assert!((ap[0] - 4.0).abs() < 1e-5); // 2*2
        assert!((ap[1] - 6.0).abs() < 1e-5); // 2*3
        assert!((ap[2] - 9.0).abs() < 1e-5); // 3*3
    }

    #[test]
    fn test_spr_4x4() {
        // 4x4 matrix: packed size = 10
        let mut ap = [0.0f64; 10];
        let x = [1.0, 2.0, 3.0, 4.0];

        spr(SprUplo::Upper, 4, 1.0, &x, &mut ap).unwrap();

        // Upper packed: [a00, a01, a11, a02, a12, a22, a03, a13, a23, a33]
        assert!(approx_eq(ap[0], 1.0)); // 1*1
        assert!(approx_eq(ap[1], 2.0)); // 1*2
        assert!(approx_eq(ap[2], 4.0)); // 2*2
        assert!(approx_eq(ap[3], 3.0)); // 1*3
        assert!(approx_eq(ap[4], 6.0)); // 2*3
        assert!(approx_eq(ap[5], 9.0)); // 3*3
        assert!(approx_eq(ap[6], 4.0)); // 1*4
        assert!(approx_eq(ap[7], 8.0)); // 2*4
        assert!(approx_eq(ap[8], 12.0)); // 3*4
        assert!(approx_eq(ap[9], 16.0)); // 4*4
    }
}
