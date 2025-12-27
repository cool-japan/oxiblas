//! Symmetric packed matrix-vector multiply (SPMV).
//!
//! Computes y = α·A·x + β·y where A is a symmetric matrix stored in packed format.
//!
//! # Packed Storage Format
//!
//! For an n×n symmetric matrix, only n*(n+1)/2 elements are stored in a 1D array.
//!
//! **Upper triangle (column-major):**
//! ```text
//! ap = [a11, a12, a22, a13, a23, a33, a14, a24, a34, a44, ...]
//!       col1  col2        col3              col4
//! Index: ap[i + j*(j+1)/2] = a(i,j) for 0 <= i <= j
//! ```
//!
//! **Lower triangle (column-major):**
//! ```text
//! ap = [a11, a21, a31, a41, a22, a32, a42, a33, a43, a44, ...]
//!       col1              col2        col3      col4
//! Index: ap[i + (2*n - j - 1)*j/2] = a(i,j) for i >= j
//! ```

use oxiblas_core::scalar::Field;

/// Specifies which triangle of the symmetric matrix is stored in packed format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmvUplo {
    /// Upper triangle is stored.
    Upper,
    /// Lower triangle is stored.
    Lower,
}

/// Error type for SPMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmvError {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for SpmvError {
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

impl std::error::Error for SpmvError {}

/// Computes the index into upper packed array for element (i, j) where i <= j.
///
/// ap[i + j*(j+1)/2] = a(i,j)
#[inline]
fn upper_packed_index(i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    i + j * (j + 1) / 2
}

/// Computes the index into lower packed array for element (i, j) where i >= j.
///
/// ap[i + (2*n - j - 1)*j/2] = a(i,j)
#[inline]
fn lower_packed_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i >= j);
    i + (2 * n - j - 1) * j / 2
}

/// Symmetric packed matrix-vector multiply.
///
/// Computes y = α·A·x + β·y where A is an n×n symmetric matrix stored in packed format.
///
/// Only the triangle specified by `uplo` is stored. The other triangle
/// is inferred from symmetry (A\[i,j\] = A\[j,i\]).
///
/// # Arguments
///
/// * `uplo` - Specifies whether upper or lower triangle is stored
/// * `n` - Order of the symmetric matrix A
/// * `alpha` - Scalar multiplier for A·x
/// * `ap` - Packed symmetric matrix (n*(n+1)/2 elements)
/// * `x` - Input vector of length n
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector of length n (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{spmv, SpmvUplo};
///
/// // Symmetric matrix A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
/// // Upper packed: [a11, a12, a22, a13, a23, a33] = [4, 2, 5, 1, 3, 6]
/// let ap = [4.0f64, 2.0, 5.0, 1.0, 3.0, 6.0];
/// let x = [1.0f64, 2.0, 3.0];
/// let mut y = [0.0f64; 3];
///
/// spmv(SpmvUplo::Upper, 3, 1.0, &ap, &x, 0.0, &mut y).unwrap();
///
/// // y = A*x = [4*1+2*2+1*3, 2*1+5*2+3*3, 1*1+3*2+6*3] = [11, 21, 25]
/// assert!((y[0] - 11.0).abs() < 1e-10);
/// assert!((y[1] - 21.0).abs() < 1e-10);
/// assert!((y[2] - 25.0).abs() < 1e-10);
/// ```
pub fn spmv<T: Field>(
    uplo: SpmvUplo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<(), SpmvError> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(SpmvError::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(SpmvError::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(SpmvError::DimensionMismatchY);
    }

    // Handle special cases
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

    // Compute y = y + alpha * A * x
    match uplo {
        SpmvUplo::Upper => {
            // Upper triangle stored: access ap[i + j*(j+1)/2] for i <= j
            for j in 0..n {
                let temp1 = alpha * x[j];
                let mut temp2 = T::zero();

                // Elements above diagonal (i < j)
                for i in 0..j {
                    let kij = upper_packed_index(i, j);
                    y[i] += temp1 * ap[kij];
                    temp2 += ap[kij] * x[i];
                }

                // Diagonal element
                let kjj = upper_packed_index(j, j);
                y[j] = y[j] + temp1 * ap[kjj] + alpha * temp2;
            }
        }
        SpmvUplo::Lower => {
            // Lower triangle stored: access ap[i + (2*n - j - 1)*j/2] for i >= j
            for j in 0..n {
                let temp1 = alpha * x[j];
                let mut temp2 = T::zero();

                // Diagonal element
                let kjj = lower_packed_index(j, j, n);
                y[j] += temp1 * ap[kjj];

                // Elements below diagonal (i > j)
                for i in (j + 1)..n {
                    let kij = lower_packed_index(i, j, n);
                    y[i] += temp1 * ap[kij];
                    temp2 += ap[kij] * x[i];
                }

                y[j] += alpha * temp2;
            }
        }
    }

    Ok(())
}

/// New-style symmetric packed matrix-vector multiply that allocates the result.
///
/// Computes y = α·A·x where A is a symmetric n×n matrix stored in packed format.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{spmv_new, SpmvUplo};
///
/// // Symmetric matrix A = [[4, 2], [2, 5]]
/// // Upper packed: [a11, a12, a22] = [4, 2, 5]
/// let ap = [4.0f64, 2.0, 5.0];
/// let x = [1.0f64, 2.0];
///
/// let y = spmv_new(SpmvUplo::Upper, 2, 1.0, &ap, &x).unwrap();
///
/// // y = A*x = [4*1+2*2, 2*1+5*2] = [8, 12]
/// assert!((y[0] - 8.0).abs() < 1e-10);
/// assert!((y[1] - 12.0).abs() < 1e-10);
/// ```
pub fn spmv_new<T: Field>(
    uplo: SpmvUplo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
) -> Result<Vec<T>, SpmvError> {
    let mut y = vec![T::zero(); n];
    spmv(uplo, n, alpha, ap, x, T::zero(), &mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_upper_packed_index() {
        // For 3x3 upper packed: [a00, a01, a11, a02, a12, a22]
        assert_eq!(upper_packed_index(0, 0), 0); // a00
        assert_eq!(upper_packed_index(0, 1), 1); // a01
        assert_eq!(upper_packed_index(1, 1), 2); // a11
        assert_eq!(upper_packed_index(0, 2), 3); // a02
        assert_eq!(upper_packed_index(1, 2), 4); // a12
        assert_eq!(upper_packed_index(2, 2), 5); // a22
    }

    #[test]
    fn test_lower_packed_index() {
        // For 3x3 lower packed: [a00, a10, a20, a11, a21, a22]
        let n = 3;
        assert_eq!(lower_packed_index(0, 0, n), 0); // a00
        assert_eq!(lower_packed_index(1, 0, n), 1); // a10
        assert_eq!(lower_packed_index(2, 0, n), 2); // a20
        assert_eq!(lower_packed_index(1, 1, n), 3); // a11
        assert_eq!(lower_packed_index(2, 1, n), 4); // a21
        assert_eq!(lower_packed_index(2, 2, n), 5); // a22
    }

    #[test]
    fn test_spmv_upper_basic() {
        // Symmetric matrix A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        // Upper packed: [a00, a01, a11, a02, a12, a22] = [4, 2, 5, 1, 3, 6]
        let ap = [4.0f64, 2.0, 5.0, 1.0, 3.0, 6.0];
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        spmv(SpmvUplo::Upper, 3, 1.0, &ap, &x, 0.0, &mut y).unwrap();

        // y = A*x = [4*1+2*2+1*3, 2*1+5*2+3*3, 1*1+3*2+6*3]
        //         = [4+4+3, 2+10+9, 1+6+18] = [11, 21, 25]
        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 21.0));
        assert!(approx_eq(y[2], 25.0));
    }

    #[test]
    fn test_spmv_lower_basic() {
        // Symmetric matrix A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        // Lower packed: [a00, a10, a20, a11, a21, a22] = [4, 2, 1, 5, 3, 6]
        let ap = [4.0f64, 2.0, 1.0, 5.0, 3.0, 6.0];
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        spmv(SpmvUplo::Lower, 3, 1.0, &ap, &x, 0.0, &mut y).unwrap();

        // Same result as upper
        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 21.0));
        assert!(approx_eq(y[2], 25.0));
    }

    #[test]
    fn test_spmv_with_alpha_beta() {
        // A = [[2, 1], [1, 3]]
        // Upper packed: [2, 1, 3]
        let ap = [2.0f64, 1.0, 3.0];
        let x = [1.0f64, 2.0];
        let mut y = [1.0f64, 1.0];

        // y = 2*A*x + 3*y
        // A*x = [2*1+1*2, 1*1+3*2] = [4, 7]
        // y = 2*[4,7] + 3*[1,1] = [8+3, 14+3] = [11, 17]
        spmv(SpmvUplo::Upper, 2, 2.0, &ap, &x, 3.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 17.0));
    }

    #[test]
    fn test_spmv_identity() {
        // Identity matrix packed: [1, 0, 1, 0, 0, 1]
        let ap = [1.0f64, 0.0, 1.0, 0.0, 0.0, 1.0];
        let x = [2.0f64, 3.0, 4.0];
        let mut y = [0.0f64; 3];

        spmv(SpmvUplo::Upper, 3, 1.0, &ap, &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 3.0));
        assert!(approx_eq(y[2], 4.0));
    }

    #[test]
    fn test_spmv_2x2() {
        // A = [[3, 2], [2, 5]]
        // Upper packed: [3, 2, 5]
        let ap = [3.0f64, 2.0, 5.0];
        let x = [1.0f64, 1.0];
        let mut y = [0.0f64; 2];

        spmv(SpmvUplo::Upper, 2, 1.0, &ap, &x, 0.0, &mut y).unwrap();

        // y = [3+2, 2+5] = [5, 7]
        assert!(approx_eq(y[0], 5.0));
        assert!(approx_eq(y[1], 7.0));
    }

    #[test]
    fn test_spmv_1x1() {
        let ap = [5.0f64];
        let x = [3.0f64];
        let mut y = [0.0f64];

        spmv(SpmvUplo::Upper, 1, 2.0, &ap, &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 30.0)); // 2 * 5 * 3
    }

    #[test]
    fn test_spmv_empty() {
        let ap: [f64; 0] = [];
        let x: [f64; 0] = [];
        let mut y: [f64; 0] = [];

        let result = spmv(SpmvUplo::Upper, 0, 1.0, &ap, &x, 0.0, &mut y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spmv_dimension_errors() {
        // Wrong packed size
        let ap = [1.0f64, 2.0, 3.0, 4.0]; // Should be 6 for n=3
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];
        assert_eq!(
            spmv(SpmvUplo::Upper, 3, 1.0, &ap, &x, 0.0, &mut y),
            Err(SpmvError::InvalidPackedSize)
        );

        // Wrong x dimension
        let ap = [1.0f64, 2.0, 3.0]; // Correct for n=2
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 2];
        assert_eq!(
            spmv(SpmvUplo::Upper, 2, 1.0, &ap, &x, 0.0, &mut y),
            Err(SpmvError::DimensionMismatchX)
        );

        // Wrong y dimension
        let ap = [1.0f64, 2.0, 3.0];
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 3];
        assert_eq!(
            spmv(SpmvUplo::Upper, 2, 1.0, &ap, &x, 0.0, &mut y),
            Err(SpmvError::DimensionMismatchY)
        );
    }

    #[test]
    fn test_spmv_alpha_zero() {
        let ap = [2.0f64, 1.0, 3.0];
        let x = [1.0f64, 2.0];
        let mut y = [5.0f64, 6.0];

        // y = 0*A*x + 2*y = [10, 12]
        spmv(SpmvUplo::Upper, 2, 0.0, &ap, &x, 2.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 10.0));
        assert!(approx_eq(y[1], 12.0));
    }

    #[test]
    fn test_spmv_beta_zero() {
        let ap = [2.0f64, 1.0, 3.0];
        let x = [1.0f64, 2.0];
        let mut y = [100.0f64, 200.0]; // Should be overwritten

        spmv(SpmvUplo::Upper, 2, 1.0, &ap, &x, 0.0, &mut y).unwrap();

        // y = A*x = [2+2, 1+6] = [4, 7]
        assert!(approx_eq(y[0], 4.0));
        assert!(approx_eq(y[1], 7.0));
    }

    #[test]
    fn test_spmv_new() {
        let ap = [4.0f64, 2.0, 5.0, 1.0, 3.0, 6.0];
        let x = [1.0f64, 2.0, 3.0];

        let y = spmv_new(SpmvUplo::Upper, 3, 1.0, &ap, &x).unwrap();

        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 21.0));
        assert!(approx_eq(y[2], 25.0));
    }

    #[test]
    fn test_spmv_f32() {
        let ap = [2.0f32, 1.0, 3.0];
        let x = [1.0f32, 2.0];
        let mut y = [0.0f32; 2];

        spmv(SpmvUplo::Upper, 2, 1.0f32, &ap, &x, 0.0f32, &mut y).unwrap();

        assert!((y[0] - 4.0).abs() < 1e-5);
        assert!((y[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_spmv_4x4() {
        // A = [[1, 2, 3, 4],
        //      [2, 5, 6, 7],
        //      [3, 6, 8, 9],
        //      [4, 7, 9, 10]]
        // Upper packed: [1, 2, 5, 3, 6, 8, 4, 7, 9, 10]
        let ap = [1.0f64, 2.0, 5.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 10.0];
        let x = [1.0f64, 1.0, 1.0, 1.0];
        let mut y = [0.0f64; 4];

        spmv(SpmvUplo::Upper, 4, 1.0, &ap, &x, 0.0, &mut y).unwrap();

        // y[0] = 1+2+3+4 = 10
        // y[1] = 2+5+6+7 = 20
        // y[2] = 3+6+8+9 = 26
        // y[3] = 4+7+9+10 = 30
        assert!(approx_eq(y[0], 10.0));
        assert!(approx_eq(y[1], 20.0));
        assert!(approx_eq(y[2], 26.0));
        assert!(approx_eq(y[3], 30.0));
    }

    #[test]
    fn test_spmv_lower_4x4() {
        // Same matrix but lower packed
        // Lower packed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let ap = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x = [1.0f64, 1.0, 1.0, 1.0];
        let mut y = [0.0f64; 4];

        spmv(SpmvUplo::Lower, 4, 1.0, &ap, &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 10.0));
        assert!(approx_eq(y[1], 20.0));
        assert!(approx_eq(y[2], 26.0));
        assert!(approx_eq(y[3], 30.0));
    }
}
