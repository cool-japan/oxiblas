//! Hermitian packed matrix-vector multiply (HPMV).
//!
//! Computes y = α·A·x + β·y where A is a Hermitian matrix stored in packed format.
//!
//! # Packed Storage Format
//!
//! For an n×n Hermitian matrix, only n*(n+1)/2 elements are stored in a 1D array.
//! The diagonal elements are stored with their real part only (imaginary part is zero).
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

use oxiblas_core::scalar::Field;

/// Specifies which triangle of the Hermitian matrix is stored in packed format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpmvUplo {
    /// Upper triangle is stored.
    Upper,
    /// Lower triangle is stored.
    Lower,
}

/// Error type for HPMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpmvError {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for HpmvError {
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

impl std::error::Error for HpmvError {}

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

/// Hermitian packed matrix-vector multiply.
///
/// Computes y = α·A·x + β·y where A is an n×n Hermitian matrix stored in packed format.
///
/// For a Hermitian matrix, A\[i,j\] = conj(A\[j,i\]) and diagonal elements are real.
/// Only the triangle specified by `uplo` is stored.
///
/// # Arguments
///
/// * `uplo` - Specifies whether upper or lower triangle is stored
/// * `n` - Order of the Hermitian matrix A
/// * `alpha` - Scalar multiplier for A·x
/// * `ap` - Packed Hermitian matrix (n*(n+1)/2 elements)
/// * `x` - Input vector of length n
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector of length n (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{hpmv, HpmvUplo};
/// use num_complex::Complex64;
///
/// fn c(re: f64, im: f64) -> Complex64 {
///     Complex64::new(re, im)
/// }
///
/// // Hermitian matrix A = [[2, 1+i], [1-i, 3]]
/// // Upper packed: [a11, a12, a22] = [2, 1+i, 3]
/// let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
/// let x = [c(1.0, 0.0), c(1.0, 0.0)];
/// let mut y = [c(0.0, 0.0); 2];
///
/// hpmv(HpmvUplo::Upper, 2, c(1.0, 0.0), &ap, &x, c(0.0, 0.0), &mut y).unwrap();
///
/// // y = A*x = [2+(1+i), (1-i)+3] = [3+i, 4-i]
/// assert!((y[0].re - 3.0).abs() < 1e-10);
/// assert!((y[0].im - 1.0).abs() < 1e-10);
/// assert!((y[1].re - 4.0).abs() < 1e-10);
/// assert!((y[1].im - (-1.0)).abs() < 1e-10);
/// ```
pub fn hpmv<T: Field>(
    uplo: HpmvUplo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<(), HpmvError> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(HpmvError::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(HpmvError::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(HpmvError::DimensionMismatchY);
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
    // For Hermitian matrix, A[i,j] = conj(A[j,i])
    match uplo {
        HpmvUplo::Upper => {
            // Upper triangle stored
            for j in 0..n {
                let temp1 = alpha * x[j];
                let mut temp2 = T::zero();

                // Elements above diagonal (i < j)
                for i in 0..j {
                    let kij = upper_packed_index(i, j);
                    y[i] += temp1 * ap[kij];
                    temp2 += ap[kij].conj() * x[i];
                }

                // Diagonal element (real for Hermitian)
                let kjj = upper_packed_index(j, j);
                y[j] = y[j] + temp1 * ap[kjj] + alpha * temp2;
            }
        }
        HpmvUplo::Lower => {
            // Lower triangle stored
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
                    temp2 += ap[kij].conj() * x[i];
                }

                y[j] += alpha * temp2;
            }
        }
    }

    Ok(())
}

/// New-style Hermitian packed matrix-vector multiply that allocates the result.
///
/// Computes y = α·A·x where A is a Hermitian n×n matrix stored in packed format.
pub fn hpmv_new<T: Field>(
    uplo: HpmvUplo,
    n: usize,
    alpha: T,
    ap: &[T],
    x: &[T],
) -> Result<Vec<T>, HpmvError> {
    let mut y = vec![T::zero(); n];
    hpmv(uplo, n, alpha, ap, x, T::zero(), &mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn approx_eq_c(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    #[test]
    fn test_upper_packed_index() {
        // For 3x3 upper packed: [a00, a01, a11, a02, a12, a22]
        assert_eq!(upper_packed_index(0, 0), 0);
        assert_eq!(upper_packed_index(0, 1), 1);
        assert_eq!(upper_packed_index(1, 1), 2);
        assert_eq!(upper_packed_index(0, 2), 3);
        assert_eq!(upper_packed_index(1, 2), 4);
        assert_eq!(upper_packed_index(2, 2), 5);
    }

    #[test]
    fn test_lower_packed_index() {
        // For 3x3 lower packed: [a00, a10, a20, a11, a21, a22]
        let n = 3;
        assert_eq!(lower_packed_index(0, 0, n), 0);
        assert_eq!(lower_packed_index(1, 0, n), 1);
        assert_eq!(lower_packed_index(2, 0, n), 2);
        assert_eq!(lower_packed_index(1, 1, n), 3);
        assert_eq!(lower_packed_index(2, 1, n), 4);
        assert_eq!(lower_packed_index(2, 2, n), 5);
    }

    #[test]
    fn test_hpmv_upper_basic() {
        // Hermitian matrix A = [[2, 1+i], [1-i, 3]]
        // Upper packed: [2, 1+i, 3]
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        hpmv(
            HpmvUplo::Upper,
            2,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y = A*x = [2*1+(1+i)*1, (1-i)*1+3*1] = [3+i, 4-i]
        assert!(approx_eq_c(y[0], c(3.0, 1.0)));
        assert!(approx_eq_c(y[1], c(4.0, -1.0)));
    }

    #[test]
    fn test_hpmv_lower_basic() {
        // Hermitian matrix A = [[2, 1+i], [1-i, 3]]
        // Lower packed: [2, 1-i, 3]
        let ap = [c(2.0, 0.0), c(1.0, -1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        hpmv(
            HpmvUplo::Lower,
            2,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // Same result as upper
        assert!(approx_eq_c(y[0], c(3.0, 1.0)));
        assert!(approx_eq_c(y[1], c(4.0, -1.0)));
    }

    #[test]
    fn test_hpmv_real_matrix() {
        // Real symmetric matrix (special case of Hermitian)
        // A = [[2, 1], [1, 3]]
        // Upper packed: [2, 1, 3]
        let ap = [c(2.0, 0.0), c(1.0, 0.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        hpmv(
            HpmvUplo::Upper,
            2,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y = A*x = [2*1+1*2, 1*1+3*2] = [4, 7]
        assert!(approx_eq_c(y[0], c(4.0, 0.0)));
        assert!(approx_eq_c(y[1], c(7.0, 0.0)));
    }

    #[test]
    fn test_hpmv_with_alpha_beta() {
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(1.0, 0.0), c(1.0, 0.0)];

        // y = 2*A*x + i*y
        // A*x = [3+i, 4-i]
        // y = 2*[3+i, 4-i] + i*[1, 1] = [6+2i+i, 8-2i+i] = [6+3i, 8-i]
        hpmv(
            HpmvUplo::Upper,
            2,
            c(2.0, 0.0),
            &ap,
            &x,
            c(0.0, 1.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(6.0, 3.0)));
        assert!(approx_eq_c(y[1], c(8.0, -1.0)));
    }

    #[test]
    fn test_hpmv_identity() {
        // Identity matrix packed: [1, 0, 1]
        let ap = [c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
        let x = [c(2.0, 3.0), c(4.0, 5.0)];
        let mut y = [c(0.0, 0.0); 2];

        hpmv(
            HpmvUplo::Upper,
            2,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(2.0, 3.0)));
        assert!(approx_eq_c(y[1], c(4.0, 5.0)));
    }

    #[test]
    fn test_hpmv_1x1() {
        let ap = [c(5.0, 0.0)];
        let x = [c(2.0, 1.0)];
        let mut y = [c(0.0, 0.0)];

        hpmv(
            HpmvUplo::Upper,
            1,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y = 5 * (2+i) = 10 + 5i
        assert!(approx_eq_c(y[0], c(10.0, 5.0)));
    }

    #[test]
    fn test_hpmv_empty() {
        let ap: [Complex64; 0] = [];
        let x: [Complex64; 0] = [];
        let mut y: [Complex64; 0] = [];

        let result = hpmv(
            HpmvUplo::Upper,
            0,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_hpmv_dimension_errors() {
        // Wrong packed size
        let ap = [c(1.0, 0.0), c(2.0, 0.0)]; // Should be 3 for n=2
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];
        assert_eq!(
            hpmv(
                HpmvUplo::Upper,
                2,
                c(1.0, 0.0),
                &ap,
                &x,
                c(0.0, 0.0),
                &mut y
            ),
            Err(HpmvError::InvalidPackedSize)
        );

        // Wrong x dimension
        let ap = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];
        assert_eq!(
            hpmv(
                HpmvUplo::Upper,
                2,
                c(1.0, 0.0),
                &ap,
                &x,
                c(0.0, 0.0),
                &mut y
            ),
            Err(HpmvError::DimensionMismatchX)
        );

        // Wrong y dimension
        let ap = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];
        assert_eq!(
            hpmv(
                HpmvUplo::Upper,
                2,
                c(1.0, 0.0),
                &ap,
                &x,
                c(0.0, 0.0),
                &mut y
            ),
            Err(HpmvError::DimensionMismatchY)
        );
    }

    #[test]
    fn test_hpmv_alpha_zero() {
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let mut y = [c(5.0, 0.0), c(6.0, 0.0)];

        // y = 0*A*x + 2*y = [10, 12]
        hpmv(
            HpmvUplo::Upper,
            2,
            c(0.0, 0.0),
            &ap,
            &x,
            c(2.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(10.0, 0.0)));
        assert!(approx_eq_c(y[1], c(12.0, 0.0)));
    }

    #[test]
    fn test_hpmv_beta_zero() {
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(100.0, 100.0), c(200.0, 200.0)]; // Should be overwritten

        hpmv(
            HpmvUplo::Upper,
            2,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(3.0, 1.0)));
        assert!(approx_eq_c(y[1], c(4.0, -1.0)));
    }

    #[test]
    fn test_hpmv_new() {
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];

        let y = hpmv_new(HpmvUplo::Upper, 2, c(1.0, 0.0), &ap, &x).unwrap();

        assert!(approx_eq_c(y[0], c(3.0, 1.0)));
        assert!(approx_eq_c(y[1], c(4.0, -1.0)));
    }

    #[test]
    fn test_hpmv_3x3_upper() {
        // 3x3 Hermitian matrix
        // A = [[1, 2+i, 3-2i], [2-i, 4, 1+i], [3+2i, 1-i, 5]]
        // Upper packed: [1, 2+i, 4, 3-2i, 1+i, 5]
        let ap = [
            c(1.0, 0.0),
            c(2.0, 1.0),
            c(4.0, 0.0),
            c(3.0, -2.0),
            c(1.0, 1.0),
            c(5.0, 0.0),
        ];
        let x = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        hpmv(
            HpmvUplo::Upper,
            3,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y = A * [1, 0, 0] = first column of A = [1, 2-i, 3+2i]
        assert!(approx_eq_c(y[0], c(1.0, 0.0)));
        assert!(approx_eq_c(y[1], c(2.0, -1.0)));
        assert!(approx_eq_c(y[2], c(3.0, 2.0)));
    }

    #[test]
    fn test_hpmv_3x3_lower() {
        // 3x3 Hermitian matrix
        // A = [[1, 2+i, 3-2i], [2-i, 4, 1+i], [3+2i, 1-i, 5]]
        // Lower packed: [1, 2-i, 3+2i, 4, 1-i, 5]
        let ap = [
            c(1.0, 0.0),
            c(2.0, -1.0),
            c(3.0, 2.0),
            c(4.0, 0.0),
            c(1.0, -1.0),
            c(5.0, 0.0),
        ];
        let x = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        hpmv(
            HpmvUplo::Lower,
            3,
            c(1.0, 0.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y = A * [1, 0, 0] = first column of A = [1, 2-i, 3+2i]
        assert!(approx_eq_c(y[0], c(1.0, 0.0)));
        assert!(approx_eq_c(y[1], c(2.0, -1.0)));
        assert!(approx_eq_c(y[2], c(3.0, 2.0)));
    }

    #[test]
    fn test_hpmv_complex_alpha() {
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        // y = (1+i)*A*x where A*x = [3+i, 4-i]
        // y = (1+i)*[3+i, 4-i] = [3+i+3i-1, 4-i+4i+1] = [2+4i, 5+3i]
        hpmv(
            HpmvUplo::Upper,
            2,
            c(1.0, 1.0),
            &ap,
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(2.0, 4.0)));
        assert!(approx_eq_c(y[1], c(5.0, 3.0)));
    }
}
