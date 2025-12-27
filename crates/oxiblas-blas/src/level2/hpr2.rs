//! HPR2: Hermitian packed rank-2 update.
//!
//! Performs the Hermitian rank-2 update: A = alpha * x * y^H + conj(alpha) * y * x^H + A
//! where A is a Hermitian matrix stored in packed format.
//!
//! # Packed Storage Format
//!
//! For an n×n Hermitian matrix, only n*(n+1)/2 elements are stored.
//! The diagonal elements are real (imaginary part is zero).
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

/// Specifies which triangle of the matrix is stored in packed format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hpr2Uplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for Hermitian packed rank-2 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hpr2Error {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for Hpr2Error {
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

impl std::error::Error for Hpr2Error {}

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

/// Performs the Hermitian packed rank-2 update: A = alpha * x * y^H + conj(alpha) * y * x^H + A
///
/// Only the specified triangle (upper or lower) is stored and updated.
/// For a Hermitian matrix, A\[i,j\] = conj(A\[j,i\]) and diagonal is real.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `n` - Order of the Hermitian matrix A
/// * `alpha` - The complex scalar alpha
/// * `x` - The first vector x of length n
/// * `y` - The second vector y of length n
/// * `ap` - The packed Hermitian matrix (n*(n+1)/2 elements)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{hpr2, Hpr2Uplo};
/// use num_complex::Complex64;
///
/// fn c(re: f64, im: f64) -> Complex64 {
///     Complex64::new(re, im)
/// }
///
/// // n=2: need 2*3/2 = 3 elements
/// let mut ap = [c(0.0, 0.0); 3];
/// let x = [c(1.0, 0.0), c(0.0, 1.0)];
/// let y = [c(1.0, 0.0), c(1.0, 0.0)];
///
/// // A = alpha * x * y^H + conj(alpha) * y * x^H + A
/// hpr2(Hpr2Uplo::Upper, 2, c(1.0, 0.0), &x, &y, &mut ap).unwrap();
///
/// // For real alpha=1:
/// // x * y^H: [[1, 1], [i, i]]
/// // y * x^H: [[1, -i], [1, -i]]
/// // Sum: [[2, 1-i], [1+i, 0]]
/// // Upper packed: [2, 1-i, 0]
/// assert!((ap[0].re - 2.0).abs() < 1e-10);
/// assert!((ap[0].im).abs() < 1e-10);       // diagonal is real
/// assert!((ap[1].re - 1.0).abs() < 1e-10);
/// assert!((ap[1].im - (-1.0)).abs() < 1e-10);
/// assert!(ap[2].re.abs() < 1e-10);         // diagonal = 0
/// assert!(ap[2].im.abs() < 1e-10);         // diagonal is real
/// ```
pub fn hpr2<T: Field>(
    uplo: Hpr2Uplo,
    n: usize,
    alpha: T,
    x: &[T],
    y: &[T],
    ap: &mut [T],
) -> Result<(), Hpr2Error> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(Hpr2Error::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(Hpr2Error::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(Hpr2Error::DimensionMismatchY);
    }

    if n == 0 || alpha.is_zero() {
        return Ok(());
    }

    let alpha_conj = alpha.conj();

    match uplo {
        Hpr2Uplo::Upper => {
            // Update upper triangle:
            // A[i,j] += alpha * x[i] * conj(y[j]) + conj(alpha) * y[i] * conj(x[j]) for i <= j
            for j in 0..n {
                let y_j_conj = y[j].conj();
                let x_j_conj = x[j].conj();
                let alpha_yj_conj = alpha * y_j_conj;
                let alpha_conj_xj_conj = alpha_conj * x_j_conj;

                for i in 0..j {
                    let kij = upper_packed_index(i, j);
                    ap[kij] = ap[kij] + alpha_yj_conj * x[i] + alpha_conj_xj_conj * y[i];
                }

                // Diagonal: A[j,j] += alpha * x[j] * conj(y[j]) + conj(alpha) * y[j] * conj(x[j])
                // This is 2 * Re(alpha * x[j] * conj(y[j])), which is real
                let kjj = upper_packed_index(j, j);
                let term = alpha * x[j] * y_j_conj;
                // term + conj(term) = 2 * Re(term)
                let diag_update = T::from_real(term.real() + term.real());
                ap[kjj] = T::from_real(ap[kjj].real()) + diag_update;
            }
        }
        Hpr2Uplo::Lower => {
            // Update lower triangle:
            // A[i,j] += alpha * x[i] * conj(y[j]) + conj(alpha) * y[i] * conj(x[j]) for i >= j
            for j in 0..n {
                let y_j_conj = y[j].conj();
                let x_j_conj = x[j].conj();
                let alpha_yj_conj = alpha * y_j_conj;
                let alpha_conj_xj_conj = alpha_conj * x_j_conj;

                // Diagonal: result must be real
                let kjj = lower_packed_index(j, j, n);
                let term = alpha * x[j] * y_j_conj;
                let diag_update = T::from_real(term.real() + term.real());
                ap[kjj] = T::from_real(ap[kjj].real()) + diag_update;

                for i in (j + 1)..n {
                    let kij = lower_packed_index(i, j, n);
                    ap[kij] = ap[kij] + alpha_yj_conj * x[i] + alpha_conj_xj_conj * y[i];
                }
            }
        }
    }

    Ok(())
}

/// Creates a new packed Hermitian matrix from alpha * x * y^H + conj(alpha) * y * x^H.
///
/// # Arguments
///
/// * `uplo` - Specifies whether to store upper or lower triangle
/// * `alpha` - The complex scalar alpha
/// * `x` - The first vector x
/// * `y` - The second vector y
///
/// # Returns
///
/// A new packed array containing alpha * x * y^H + conj(alpha) * y * x^H.
pub fn hpr2_new<T: Field>(uplo: Hpr2Uplo, alpha: T, x: &[T], y: &[T]) -> Result<Vec<T>, Hpr2Error> {
    let n = x.len();
    if y.len() != n {
        return Err(Hpr2Error::DimensionMismatchY);
    }

    let packed_size = n * (n + 1) / 2;
    let mut ap = vec![T::zero(); packed_size];

    if n == 0 || alpha.is_zero() {
        return Ok(ap);
    }

    hpr2(uplo, n, alpha, x, y, &mut ap)?;
    Ok(ap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

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
    fn test_hpr2_upper_real_alpha() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(0.0, 1.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0)];

        hpr2(Hpr2Uplo::Upper, 2, c(1.0, 0.0), &x, &y, &mut ap).unwrap();

        // x * y^H = [[1*1, 1*1], [i*1, i*1]] = [[1, 1], [i, i]]
        // y * x^H = [[1*1, 1*(-i)], [1*1, 1*(-i)]] = [[1, -i], [1, -i]]
        // Sum: [[2, 1-i], [1+i, 2i-i]] but wait, let me recalculate
        // x * y^H + y * x^H:
        // [0,0]: 1*1 + 1*1 = 2
        // [0,1]: 1*1 + (-i)*1 = 1-i
        // [1,0]: i*1 + 1*1 = 1+i (but we store upper)
        // [1,1]: i*1 + 1*(-i) = i - i = 0... wait
        // Let me recalculate:
        // [1,1] = x[1]*conj(y[1]) + y[1]*conj(x[1]) = i*1 + 1*(-i) = i - i = 0
        // Hmm that's 0, let me verify
        // Actually: x[1] = i, y[1] = 1
        // x[1]*conj(y[1]) = i*1 = i
        // y[1]*conj(x[1]) = 1*(-i) = -i
        // Sum = i + (-i) = 0

        // Upper packed: [a00, a01, a11] = [2, 1-i, 0]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(1.0, -1.0)));
        assert!(approx_eq(ap[2].re, 0.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_lower_real_alpha() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(0.0, 1.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0)];

        hpr2(Hpr2Uplo::Lower, 2, c(1.0, 0.0), &x, &y, &mut ap).unwrap();

        // Lower stores conjugate of upper for off-diagonal
        // Lower packed: [a00, a10, a11] = [2, 1+i, 0]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(1.0, 1.0)));
        assert!(approx_eq(ap[2].re, 0.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_complex_alpha() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0)];

        // alpha = 1+i, conj(alpha) = 1-i
        hpr2(Hpr2Uplo::Upper, 2, c(1.0, 1.0), &x, &y, &mut ap).unwrap();

        // x * y^H = [[1, 1], [1, 1]]
        // alpha * x * y^H = (1+i) * [[1, 1], [1, 1]] = [[1+i, 1+i], [1+i, 1+i]]
        // y * x^H = [[1, 1], [1, 1]]
        // conj(alpha) * y * x^H = (1-i) * [[1, 1], [1, 1]] = [[1-i, 1-i], [1-i, 1-i]]
        // Sum: [[2, 2], [2, 2]]
        // Upper packed: [2, 2, 2]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq(ap[1].re, 2.0));
        assert!(approx_eq(ap[1].im, 0.0));
        assert!(approx_eq(ap[2].re, 2.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_x_equals_y() {
        // When x == y, result is 2 * Re(alpha) * x * x^H
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 1.0), c(2.0, 0.0)];

        hpr2(Hpr2Uplo::Upper, 2, c(1.0, 0.0), &x, &x, &mut ap).unwrap();

        // 2 * x * x^H:
        // x * x^H = [[2, 2+2i], [2-2i, 4]]
        // 2 * x * x^H = [[4, 4+4i], [4-4i, 8]]
        // Upper packed: [4, 4+4i, 8]
        assert!(approx_eq(ap[0].re, 4.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(4.0, 4.0)));
        assert!(approx_eq(ap[2].re, 8.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_real_vectors() {
        // Real vectors - same as SPR2
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let y = [c(3.0, 0.0), c(4.0, 0.0)];

        hpr2(Hpr2Uplo::Upper, 2, c(1.0, 0.0), &x, &y, &mut ap).unwrap();

        // Same as SPR2: [6, 10, 16]
        assert!(approx_eq(ap[0].re, 6.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq(ap[1].re, 10.0));
        assert!(approx_eq(ap[1].im, 0.0));
        assert!(approx_eq(ap[2].re, 16.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_accumulate() {
        let mut ap = [c(1.0, 0.0), c(2.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0)];

        hpr2(Hpr2Uplo::Upper, 2, c(1.0, 0.0), &x, &y, &mut ap).unwrap();

        // x * y^H + y * x^H = [[2, 2], [2, 2]]
        // Result: [1+2, 2+1+2, 3+2] = [3, 4+i (but off-diag stays complex), 5]
        // Wait, off-diagonal should add properly
        // ap[1] = (2+i) + 2 = 4+i
        assert!(approx_eq(ap[0].re, 3.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(4.0, 1.0)));
        assert!(approx_eq(ap[2].re, 5.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_zero_alpha() {
        let mut ap = [c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0)];
        let original = ap;
        let x = [c(1.0, 1.0), c(2.0, 2.0)];
        let y = [c(3.0, 3.0), c(4.0, 4.0)];

        hpr2(Hpr2Uplo::Upper, 2, c(0.0, 0.0), &x, &y, &mut ap).unwrap();

        // Unchanged
        for i in 0..3 {
            assert!(approx_eq_c(ap[i], original[i]));
        }
    }

    #[test]
    fn test_hpr2_dimension_errors() {
        // Wrong packed size
        let mut ap = [c(0.0, 0.0); 5]; // Should be 6 for n=3
        let x = [c(1.0, 0.0); 3];
        let y = [c(1.0, 0.0); 3];
        assert_eq!(
            hpr2(Hpr2Uplo::Upper, 3, c(1.0, 0.0), &x, &y, &mut ap),
            Err(Hpr2Error::InvalidPackedSize)
        );

        // Wrong x dimension
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0); 3];
        let y = [c(1.0, 0.0); 2];
        assert_eq!(
            hpr2(Hpr2Uplo::Upper, 2, c(1.0, 0.0), &x, &y, &mut ap),
            Err(Hpr2Error::DimensionMismatchX)
        );

        // Wrong y dimension
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0); 2];
        let y = [c(1.0, 0.0); 3];
        assert_eq!(
            hpr2(Hpr2Uplo::Upper, 2, c(1.0, 0.0), &x, &y, &mut ap),
            Err(Hpr2Error::DimensionMismatchY)
        );
    }

    #[test]
    fn test_hpr2_empty() {
        let mut ap: [Complex64; 0] = [];
        let x: [Complex64; 0] = [];
        let y: [Complex64; 0] = [];

        hpr2(Hpr2Uplo::Upper, 0, c(1.0, 0.0), &x, &y, &mut ap).unwrap();
    }

    #[test]
    fn test_hpr2_1x1() {
        let mut ap = [c(0.0, 0.0)];
        let x = [c(2.0, 1.0)];
        let y = [c(3.0, 0.0)];

        hpr2(Hpr2Uplo::Upper, 1, c(1.0, 0.0), &x, &y, &mut ap).unwrap();

        // x[0] * conj(y[0]) + y[0] * conj(x[0])
        // = (2+i)*3 + 3*(2-i) = 6+3i + 6-3i = 12
        assert!(approx_eq(ap[0].re, 12.0));
        assert!(approx_eq(ap[0].im, 0.0));
    }

    #[test]
    fn test_hpr2_new_upper() {
        let x = [c(1.0, 0.0), c(0.0, 1.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0)];
        let ap = hpr2_new(Hpr2Uplo::Upper, c(1.0, 0.0), &x, &y).unwrap();

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(1.0, -1.0)));
        assert!(approx_eq(ap[2].re, 0.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_new_lower() {
        let x = [c(1.0, 0.0), c(0.0, 1.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0)];
        let ap = hpr2_new(Hpr2Uplo::Lower, c(1.0, 0.0), &x, &y).unwrap();

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(1.0, 1.0)));
        assert!(approx_eq(ap[2].re, 0.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr2_3x3() {
        let mut ap = [c(0.0, 0.0); 6];
        let x = [c(1.0, 0.0), c(0.0, 1.0), c(1.0, 1.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];

        hpr2(Hpr2Uplo::Upper, 3, c(1.0, 0.0), &x, &y, &mut ap).unwrap();

        // x * y^H:
        // [0,0] = 1*1 = 1
        // [0,1] = 1*1 = 1
        // [0,2] = 1*1 = 1
        // [1,0] = i*1 = i
        // [1,1] = i*1 = i
        // [1,2] = i*1 = i
        // [2,0] = (1+i)*1 = 1+i
        // [2,1] = (1+i)*1 = 1+i
        // [2,2] = (1+i)*1 = 1+i

        // y * x^H:
        // [0,0] = 1*1 = 1
        // [0,1] = 1*(-i) = -i
        // [0,2] = 1*(1-i) = 1-i
        // etc.

        // Sum diagonal:
        // [0,0] = 1 + 1 = 2
        // [1,1] = i + (-i) = 0
        // [2,2] = (1+i) + (1-i) = 2

        // Upper packed: [a00, a01, a11, a02, a12, a22]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        // a01 = 1 + (-i) = 1-i
        assert!(approx_eq_c(ap[1], c(1.0, -1.0)));
        assert!(approx_eq(ap[2].re, 0.0));
        assert!(approx_eq(ap[2].im, 0.0));
        // a02 = 1 + (1-i) = 2-i
        assert!(approx_eq_c(ap[3], c(2.0, -1.0)));
        // a12 = i + (1-i) = 1
        assert!(approx_eq_c(ap[4], c(1.0, 0.0)));
        assert!(approx_eq(ap[5].re, 2.0));
        assert!(approx_eq(ap[5].im, 0.0));
    }
}
