//! HER: Hermitian rank-1 update.
//!
//! Performs the Hermitian rank-1 update: A = alpha * x * x^H + A
//! where A is a Hermitian matrix stored in the specified triangle.
//!
//! For real types, this is equivalent to the symmetric rank-1 update (SYR).

use num_traits::Zero;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatMut;

/// Specifies which triangle of the matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HerUplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for Hermitian rank-1 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HerError {
    /// Matrix is not square.
    NotSquare,
    /// Dimension mismatch between matrix and vector.
    DimensionMismatch,
}

impl core::fmt::Display for HerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between matrix and vector"),
        }
    }
}

impl std::error::Error for HerError {}

/// Performs the Hermitian rank-1 update: A = alpha * x * x^H + A
///
/// Only the specified triangle (upper or lower) of A is accessed and modified.
/// The other triangle is not referenced. The diagonal elements of A are assumed
/// to be real, and the imaginary parts are not modified.
///
/// Note: alpha must be real. For complex alpha, use geru instead.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `alpha` - The real scalar alpha
/// * `x` - The vector x (may be complex)
/// * `a` - The Hermitian matrix A (only the specified triangle is used)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{her, HerUplo};
/// use oxiblas_matrix::Mat;
///
/// // For real types, HER is equivalent to SYR
/// let mut a = Mat::from_rows(&[
///     &[1.0f64, 0.0, 0.0],
///     &[0.0, 1.0, 0.0],
///     &[0.0, 0.0, 1.0],
/// ]);
/// let x = [1.0, 2.0, 3.0];
///
/// // A = A + x * x^H (using lower triangle)
/// her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();
///
/// assert!((a[(0, 0)] - 2.0).abs() < 1e-10);  // 1 + 1*1
/// assert!((a[(1, 0)] - 2.0).abs() < 1e-10);  // 0 + 2*1
/// assert!((a[(1, 1)] - 5.0).abs() < 1e-10);  // 1 + 2*2
/// ```
pub fn her<T: Field>(
    uplo: HerUplo,
    alpha: T::Real,
    x: &[T],
    a: MatMut<'_, T>,
) -> Result<(), HerError>
where
    T::Real: Copy,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(HerError::NotSquare);
    }
    if n != x.len() {
        return Err(HerError::DimensionMismatch);
    }

    if n == 0 || alpha.is_zero() {
        return Ok(());
    }

    // Safety: we need mutable access
    // For column-major storage: element (i,j) is at ptr + i + j*row_stride
    let a_ptr = a.as_ptr();
    let rs = a.row_stride();

    match uplo {
        HerUplo::Lower => {
            // Update lower triangle: A[i,j] += alpha * x[i] * conj(x[j]) for i >= j
            for j in 0..n {
                let alpha_xj_conj = T::from_real(alpha) * x[j].conj();
                for i in j..n {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: ensure result is real
                            // A[i,i] += alpha * |x[i]|^2
                            let val = alpha_xj_conj * x[i];
                            *ptr += T::from_real(val.real());
                        } else {
                            *ptr += alpha_xj_conj * x[i];
                        }
                    }
                }
            }
        }
        HerUplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * x[i] * conj(x[j]) for i <= j
            for j in 0..n {
                let alpha_xj_conj = T::from_real(alpha) * x[j].conj();
                for i in 0..=j {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: ensure result is real
                            let val = alpha_xj_conj * x[i];
                            *ptr += T::from_real(val.real());
                        } else {
                            *ptr += alpha_xj_conj * x[i];
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs the Hermitian rank-1 update on a freshly allocated matrix.
///
/// Returns alpha * x * x^H as a fully Hermitian matrix.
///
/// # Arguments
///
/// * `alpha` - The real scalar alpha
/// * `x` - The vector x
///
/// # Returns
///
/// A new Hermitian matrix containing alpha * x * x^H.
pub fn her_new<T: Field + bytemuck::Zeroable>(alpha: T::Real, x: &[T]) -> oxiblas_matrix::Mat<T>
where
    T::Real: Copy,
{
    let n = x.len();
    let mut a = oxiblas_matrix::Mat::zeros(n, n);

    if n == 0 || alpha.is_zero() {
        return a;
    }

    // Create full Hermitian matrix
    for i in 0..n {
        for j in 0..=i {
            let val = T::from_real(alpha) * x[i] * x[j].conj();
            if i == j {
                // Diagonal must be real
                a[(i, j)] = T::from_real(val.real());
            } else {
                a[(i, j)] = val;
                a[(j, i)] = val.conj();
            }
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_her_lower_real() {
        // For real types, HER is equivalent to SYR
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];

        her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();

        // Lower triangle should be x * x^T
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(2, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(2, 1)] - 6.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_her_upper_real() {
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];

        her(HerUplo::Upper, 1.0, &x, a.as_mut()).unwrap();

        // Upper triangle should be x * x^T
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 2)] - 3.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 2)] - 6.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_her_complex() {
        // Complex case: A += x * x^H
        let mut a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();

        // A[0,0] = |x[0]|^2 = 1^2 + 1^2 = 2 (real)
        // A[1,0] = x[1] * conj(x[0]) = (2-i)(1-i) = 2 - 2i - i + i^2 = 1 - 3i
        // A[1,1] = |x[1]|^2 = 2^2 + 1^2 = 5 (real)
        assert!((a[(0, 0)].re - 2.0).abs() < 1e-10);
        assert!(a[(0, 0)].im.abs() < 1e-10);

        assert!((a[(1, 0)].re - 1.0).abs() < 1e-10);
        assert!((a[(1, 0)].im + 3.0).abs() < 1e-10);

        assert!((a[(1, 1)].re - 5.0).abs() < 1e-10);
        assert!(a[(1, 1)].im.abs() < 1e-10);
    }

    #[test]
    fn test_her_complex_upper() {
        let mut a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        her(HerUplo::Upper, 1.0, &x, a.as_mut()).unwrap();

        // A[0,0] = |x[0]|^2 = 2 (real)
        // A[0,1] = x[0] * conj(x[1]) = (1+i)(2+i) = 2 + i + 2i + i^2 = 1 + 3i
        // A[1,1] = |x[1]|^2 = 5 (real)
        assert!((a[(0, 0)].re - 2.0).abs() < 1e-10);
        assert!(a[(0, 0)].im.abs() < 1e-10);

        assert!((a[(0, 1)].re - 1.0).abs() < 1e-10);
        assert!((a[(0, 1)].im - 3.0).abs() < 1e-10);

        assert!((a[(1, 1)].re - 5.0).abs() < 1e-10);
        assert!(a[(1, 1)].im.abs() < 1e-10);
    }

    #[test]
    fn test_her_with_alpha() {
        let mut a: Mat<f64> = Mat::zeros(2, 2);
        let x = [1.0, 2.0];

        her(HerUplo::Lower, 2.0, &x, a.as_mut()).unwrap();

        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_her_accumulate() {
        let mut a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let x = [1.0, 1.0];

        her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();

        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_her_zero_alpha() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let original = a.clone();
        let x = [1.0, 2.0];

        her(HerUplo::Lower, 0.0, &x, a.as_mut()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - original[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_her_not_square() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 2.0];

        let result = her(HerUplo::Lower, 1.0, &x, a.as_mut());
        assert!(matches!(result, Err(HerError::NotSquare)));
    }

    #[test]
    fn test_her_dimension_mismatch() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let x = [1.0, 2.0, 3.0];

        let result = her(HerUplo::Lower, 1.0, &x, a.as_mut());
        assert!(matches!(result, Err(HerError::DimensionMismatch)));
    }

    #[test]
    fn test_her_empty() {
        let mut a: Mat<f64> = Mat::zeros(0, 0);
        let x: [f64; 0] = [];

        her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();
    }

    #[test]
    fn test_her_f32() {
        let mut a: Mat<f32> = Mat::zeros(2, 2);
        let x = [2.0f32, 3.0];

        her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();

        assert!((a[(0, 0)] - 4.0).abs() < 1e-5);
        assert!((a[(1, 0)] - 6.0).abs() < 1e-5);
        assert!((a[(1, 1)] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_her_new_real() {
        let x = [1.0f64, 2.0, 3.0];

        let a = her_new(1.0, &x);

        // Full Hermitian (symmetric for real) matrix
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

    // Note: test_her_new_complex is omitted because Complex64 doesn't implement
    // bytemuck::Zeroable, which is required by Mat::zeros used in her_new.
    // For complex types, use the in-place `her` function instead.
}
