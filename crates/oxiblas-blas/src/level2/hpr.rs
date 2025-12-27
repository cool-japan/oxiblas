//! HPR: Hermitian packed rank-1 update.
//!
//! Performs the Hermitian rank-1 update: A = alpha * x * x^H + A
//! where A is a Hermitian matrix stored in packed format.
//! Note: alpha must be real for the result to remain Hermitian.
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

use num_traits::Zero;
use oxiblas_core::scalar::Field;

/// Specifies which triangle of the matrix is stored in packed format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HprUplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for Hermitian packed rank-1 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HprError {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector has wrong dimension.
    DimensionMismatch,
}

impl core::fmt::Display for HprError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPackedSize => {
                write!(f, "Packed array has wrong size (expected n*(n+1)/2)")
            }
            Self::DimensionMismatch => write!(f, "Vector has wrong dimension"),
        }
    }
}

impl std::error::Error for HprError {}

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

/// Performs the Hermitian packed rank-1 update: A = alpha * x * x^H + A
///
/// Only the specified triangle (upper or lower) is stored and updated.
/// For a Hermitian matrix, A\[i,j\] = conj(A\[j,i\]) and diagonal is real.
///
/// Note: alpha should be real for the result to remain Hermitian.
/// The function uses the real part of alpha for the diagonal update.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `n` - Order of the Hermitian matrix A
/// * `alpha` - The real scalar alpha (real part of T used for diagonal)
/// * `x` - The vector x of length n
/// * `ap` - The packed Hermitian matrix (n*(n+1)/2 elements)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{hpr, HprUplo};
/// use num_complex::Complex64;
///
/// fn c(re: f64, im: f64) -> Complex64 {
///     Complex64::new(re, im)
/// }
///
/// // n=2: need 2*3/2 = 3 elements
/// let mut ap = [c(0.0, 0.0); 3];
/// let x = [c(1.0, 1.0), c(2.0, 0.0)];
///
/// // A = A + x * x^H
/// hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap).unwrap();
///
/// // x * x^H = [[|1+i|^2, (1+i)*2], [2*(1-i), |2|^2]]
/// //         = [[2, 2+2i], [2-2i, 4]]
/// // Upper packed: [a00, a01, a11] = [2, 2+2i, 4]
/// assert!((ap[0].re - 2.0).abs() < 1e-10); // |1+i|^2 = 2
/// assert!((ap[0].im).abs() < 1e-10);       // diagonal is real
/// assert!((ap[1].re - 2.0).abs() < 1e-10); // (1+i)*2 = 2+2i
/// assert!((ap[1].im - 2.0).abs() < 1e-10);
/// assert!((ap[2].re - 4.0).abs() < 1e-10); // |2|^2 = 4
/// assert!((ap[2].im).abs() < 1e-10);       // diagonal is real
/// ```
pub fn hpr<T: Field>(
    uplo: HprUplo,
    n: usize,
    alpha: T::Real,
    x: &[T],
    ap: &mut [T],
) -> Result<(), HprError> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(HprError::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(HprError::DimensionMismatch);
    }

    if n == 0 || alpha == T::Real::zero() {
        return Ok(());
    }

    match uplo {
        HprUplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * x[i] * conj(x[j]) for i <= j
            for j in 0..n {
                let x_j_conj = x[j].conj();
                let alpha_xj_conj = T::from_real(alpha) * x_j_conj;

                for i in 0..j {
                    let kij = upper_packed_index(i, j);
                    ap[kij] += alpha_xj_conj * x[i];
                }

                // Diagonal: A[j,j] += alpha * x[j] * conj(x[j]) = alpha * |x[j]|^2
                // Result must be real
                let kjj = upper_packed_index(j, j);
                let norm_sq = x[j].abs_sq();
                ap[kjj] = T::from_real(ap[kjj].real() + alpha * norm_sq);
            }
        }
        HprUplo::Lower => {
            // Update lower triangle: A[i,j] += alpha * x[i] * conj(x[j]) for i >= j
            for j in 0..n {
                let x_j_conj = x[j].conj();
                let alpha_xj_conj = T::from_real(alpha) * x_j_conj;

                // Diagonal: A[j,j] += alpha * |x[j]|^2 (must be real)
                let kjj = lower_packed_index(j, j, n);
                let norm_sq = x[j].abs_sq();
                ap[kjj] = T::from_real(ap[kjj].real() + alpha * norm_sq);

                for i in (j + 1)..n {
                    let kij = lower_packed_index(i, j, n);
                    ap[kij] += alpha_xj_conj * x[i];
                }
            }
        }
    }

    Ok(())
}

/// Creates a new packed Hermitian matrix from x * x^H.
///
/// # Arguments
///
/// * `uplo` - Specifies whether to store upper or lower triangle
/// * `alpha` - The real scalar alpha
/// * `x` - The vector x
///
/// # Returns
///
/// A new packed array containing alpha * x * x^H.
pub fn hpr_new<T: Field>(uplo: HprUplo, alpha: T::Real, x: &[T]) -> Vec<T> {
    let n = x.len();
    let packed_size = n * (n + 1) / 2;
    let mut ap = vec![T::zero(); packed_size];

    if n == 0 || alpha == T::Real::zero() {
        return ap;
    }

    let _ = hpr(uplo, n, alpha, x, &mut ap);
    ap
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
    fn test_hpr_upper_basic() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 1.0), c(2.0, 0.0)];

        hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap).unwrap();

        // x * x^H = [[|1+i|^2, (1+i)*2], [2*(1-i), |2|^2]]
        //         = [[2, 2+2i], [2-2i, 4]]
        // Upper packed: [a00, a01, a11] = [2, 2+2i, 4]
        assert!(approx_eq(ap[0].re, 2.0)); // |1+i|^2 = 2
        assert!(approx_eq(ap[0].im, 0.0)); // diagonal is real
        assert!(approx_eq_c(ap[1], c(2.0, 2.0))); // (1+i)*2
        assert!(approx_eq(ap[2].re, 4.0)); // |2|^2 = 4
        assert!(approx_eq(ap[2].im, 0.0)); // diagonal is real
    }

    #[test]
    fn test_hpr_lower_basic() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 1.0), c(2.0, 0.0)];

        hpr(HprUplo::Lower, 2, 1.0, &x, &mut ap).unwrap();

        // x * x^H = [[2, 2+2i], [2-2i, 4]]
        // Lower packed: [a00, a10, a11] = [2, 2-2i, 4]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(2.0, -2.0))); // (1-i)*2 = 2-2i
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_with_alpha() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(0.0, 1.0)];

        hpr(HprUplo::Upper, 2, 2.0, &x, &mut ap).unwrap();

        // x * x^H = [[1, -i], [i, 1]]
        // 2 * x * x^H = [[2, -2i], [2i, 2]]
        // Upper packed: [2, -2i, 2]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(0.0, -2.0)));
        assert!(approx_eq(ap[2].re, 2.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_accumulate() {
        let mut ap = [c(1.0, 0.0), c(2.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];

        hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap).unwrap();

        // x * x^H = [[1, 1], [1, 1]]
        // New: [2, 3+i, 4]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0)); // diagonal stays real
        assert!(approx_eq_c(ap[1], c(3.0, 1.0)));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_real_vector() {
        // Real vector - result is symmetric
        let mut ap = [c(0.0, 0.0); 6];
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];

        hpr(HprUplo::Upper, 3, 1.0, &x, &mut ap).unwrap();

        // Same as SPR for real vectors
        assert!(approx_eq(ap[0].re, 1.0));
        assert!(approx_eq(ap[1].re, 2.0));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[3].re, 3.0));
        assert!(approx_eq(ap[4].re, 6.0));
        assert!(approx_eq(ap[5].re, 9.0));
    }

    #[test]
    fn test_hpr_zero_alpha() {
        let mut ap = [c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0)];
        let original = ap;
        let x = [c(1.0, 1.0), c(2.0, 2.0)];

        hpr(HprUplo::Upper, 2, 0.0, &x, &mut ap).unwrap();

        // Unchanged
        for i in 0..3 {
            assert!(approx_eq_c(ap[i], original[i]));
        }
    }

    #[test]
    fn test_hpr_dimension_errors() {
        // Wrong packed size
        let mut ap = [c(0.0, 0.0); 5]; // Should be 6 for n=3
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        assert_eq!(
            hpr(HprUplo::Upper, 3, 1.0, &x, &mut ap),
            Err(HprError::InvalidPackedSize)
        );

        // Wrong x dimension
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        assert_eq!(
            hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap),
            Err(HprError::DimensionMismatch)
        );
    }

    #[test]
    fn test_hpr_empty() {
        let mut ap: [Complex64; 0] = [];
        let x: [Complex64; 0] = [];

        hpr(HprUplo::Upper, 0, 1.0, &x, &mut ap).unwrap();
    }

    #[test]
    fn test_hpr_1x1() {
        let mut ap = [c(0.0, 0.0)];
        let x = [c(2.0, 1.0)];

        hpr(HprUplo::Upper, 1, 1.0, &x, &mut ap).unwrap();

        // |2+i|^2 = 4 + 1 = 5
        assert!(approx_eq(ap[0].re, 5.0));
        assert!(approx_eq(ap[0].im, 0.0));
    }

    #[test]
    fn test_hpr_new_upper() {
        let x = [c(1.0, 1.0), c(2.0, 0.0)];
        let ap = hpr_new(HprUplo::Upper, 1.0, &x);

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(2.0, 2.0)));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_new_lower() {
        let x = [c(1.0, 1.0), c(2.0, 0.0)];
        let ap = hpr_new(HprUplo::Lower, 1.0, &x);

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(2.0, -2.0)));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_negative_alpha() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(0.0, 1.0)];

        hpr(HprUplo::Upper, 2, -1.0, &x, &mut ap).unwrap();

        // -1 * x * x^H = [[-1, i], [-i, -1]]
        assert!(approx_eq(ap[0].re, -1.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(0.0, 1.0)));
        assert!(approx_eq(ap[2].re, -1.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_3x3() {
        let mut ap = [c(0.0, 0.0); 6];
        let x = [c(1.0, 0.0), c(0.0, 1.0), c(1.0, 1.0)];

        hpr(HprUplo::Upper, 3, 1.0, &x, &mut ap).unwrap();

        // x * x^H:
        // [0,0] = 1*1 = 1
        // [0,1] = 1*(-i) = -i
        // [0,2] = 1*(1-i) = 1-i
        // [1,1] = i*(-i) = 1
        // [1,2] = i*(1-i) = i+1 = 1+i
        // [2,2] = (1+i)*(1-i) = 1+1 = 2

        // Upper packed: [a00, a01, a11, a02, a12, a22]
        assert!(approx_eq(ap[0].re, 1.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(0.0, -1.0)));
        assert!(approx_eq(ap[2].re, 1.0));
        assert!(approx_eq(ap[2].im, 0.0));
        assert!(approx_eq_c(ap[3], c(1.0, -1.0)));
        assert!(approx_eq_c(ap[4], c(1.0, 1.0)));
        assert!(approx_eq(ap[5].re, 2.0));
        assert!(approx_eq(ap[5].im, 0.0));
    }
}
