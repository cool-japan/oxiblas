//! HER2: Hermitian rank-2 update.
//!
//! Performs the Hermitian rank-2 update: A = alpha * x * y^H + conj(alpha) * y * x^H + A
//! where A is a Hermitian matrix stored in the specified triangle.
//!
//! For real types, this is equivalent to the symmetric rank-2 update (SYR2).

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatMut;

/// Specifies which triangle of the matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Her2Uplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for Hermitian rank-2 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Her2Error {
    /// Matrix is not square.
    NotSquare,
    /// Dimension mismatch between matrix and vector x.
    DimensionMismatchX,
    /// Dimension mismatch between matrix and vector y.
    DimensionMismatchY,
}

impl core::fmt::Display for Her2Error {
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

impl std::error::Error for Her2Error {}

/// Performs the Hermitian rank-2 update: A = alpha * x * y^H + conj(alpha) * y * x^H + A
///
/// Only the specified triangle (upper or lower) of A is accessed and modified.
/// The other triangle is not referenced. The diagonal elements of A are assumed
/// to be real, and remain real after the update.
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `alpha` - The complex scalar alpha
/// * `x` - The first vector
/// * `y` - The second vector
/// * `a` - The Hermitian matrix A (only the specified triangle is used)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{her2, Her2Uplo};
/// use oxiblas_matrix::Mat;
///
/// // For real types, HER2 is equivalent to SYR2
/// let mut a: Mat<f64> = Mat::zeros(2, 2);
/// let x = [1.0, 2.0];
/// let y = [3.0, 4.0];
///
/// // A = A + x * y^T + y * x^T (using lower triangle)
/// her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();
///
/// // A[0,0] = 1*3 + 3*1 = 6
/// assert!((a[(0, 0)] - 6.0).abs() < 1e-10);
/// // A[1,0] = 2*3 + 4*1 = 10
/// assert!((a[(1, 0)] - 10.0).abs() < 1e-10);
/// ```
pub fn her2<T: Field>(
    uplo: Her2Uplo,
    alpha: T,
    x: &[T],
    y: &[T],
    a: MatMut<'_, T>,
) -> Result<(), Her2Error> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(Her2Error::NotSquare);
    }
    if n != x.len() {
        return Err(Her2Error::DimensionMismatchX);
    }
    if n != y.len() {
        return Err(Her2Error::DimensionMismatchY);
    }

    if n == 0 || alpha.is_zero() {
        return Ok(());
    }

    // For column-major storage: element (i,j) is at ptr + i + j*row_stride
    let a_ptr = a.as_ptr();
    let rs = a.row_stride();

    // conj(alpha) for the second term
    let alpha_conj = alpha.conj();

    match uplo {
        Her2Uplo::Lower => {
            // Update lower triangle: A[i,j] += alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j])
            // for i >= j
            for j in 0..n {
                let alpha_yj_conj = alpha * y[j].conj();
                let alpha_conj_xj_conj = alpha_conj * x[j].conj();
                for i in j..n {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: ensure result is real
                            // A[i,i] += alpha*x[i]*conj(y[i]) + conj(alpha)*y[i]*conj(x[i])
                            // For Hermitian, this sum should be real
                            let val = alpha_yj_conj * x[i] + alpha_conj_xj_conj * y[i];
                            *ptr += T::from_real(val.real());
                        } else {
                            *ptr = *ptr + alpha_yj_conj * x[i] + alpha_conj_xj_conj * y[i];
                        }
                    }
                }
            }
        }
        Her2Uplo::Upper => {
            // Update upper triangle: A[i,j] += alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j])
            // for i <= j
            for j in 0..n {
                let alpha_yj_conj = alpha * y[j].conj();
                let alpha_conj_xj_conj = alpha_conj * x[j].conj();
                for i in 0..=j {
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: ensure result is real
                            let val = alpha_yj_conj * x[i] + alpha_conj_xj_conj * y[i];
                            *ptr += T::from_real(val.real());
                        } else {
                            *ptr = *ptr + alpha_yj_conj * x[i] + alpha_conj_xj_conj * y[i];
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs the Hermitian rank-2 update on a freshly allocated matrix.
///
/// Returns alpha * x * y^H + conj(alpha) * y * x^H as a fully Hermitian matrix.
///
/// # Arguments
///
/// * `alpha` - The complex scalar alpha
/// * `x` - The first vector
/// * `y` - The second vector
///
/// # Returns
///
/// A new Hermitian matrix.
///
/// # Errors
///
/// Returns `Her2Error::DimensionMismatchY` if x and y have different lengths.
pub fn her2_new<T: Field + bytemuck::Zeroable>(
    alpha: T,
    x: &[T],
    y: &[T],
) -> Result<oxiblas_matrix::Mat<T>, Her2Error> {
    let n = x.len();
    if n != y.len() {
        return Err(Her2Error::DimensionMismatchY);
    }

    let mut a = oxiblas_matrix::Mat::zeros(n, n);

    if n == 0 || alpha.is_zero() {
        return Ok(a);
    }

    let alpha_conj = alpha.conj();

    // Create full Hermitian matrix
    for i in 0..n {
        for j in 0..=i {
            // A[i,j] = alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j])
            let val = alpha * x[i] * y[j].conj() + alpha_conj * y[i] * x[j].conj();
            if i == j {
                // Diagonal must be real
                a[(i, j)] = T::from_real(val.real());
            } else {
                a[(i, j)] = val;
                a[(j, i)] = val.conj();
            }
        }
    }

    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn approx_eq_c(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    #[test]
    fn test_her2_lower_real() {
        // For real types, HER2 is equivalent to SYR2
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 1.0, 1.0];

        her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        // A[i,j] = x[i]*y[j] + y[i]*x[j]
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(2, 0)] - 4.0).abs() < 1e-10);
        assert!((a[(2, 1)] - 5.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2_upper_real() {
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 1.0, 1.0];

        her2(Her2Uplo::Upper, 1.0, &x, &y, a.as_mut()).unwrap();

        // Upper triangle
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 3.0).abs() < 1e-10);
        assert!((a[(0, 2)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 2)] - 5.0).abs() < 1e-10);
        assert!((a[(2, 2)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2_complex_lower() {
        // Complex case: A += alpha*x*y^H + conj(alpha)*y*x^H
        let mut a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        let x = [c(1.0, 1.0), c(2.0, -1.0)];
        let y = [c(1.0, 0.0), c(0.0, 1.0)];

        her2(Her2Uplo::Lower, c(1.0, 0.0), &x, &y, a.as_mut()).unwrap();

        // With alpha = 1:
        // A[0,0] = x[0]*conj(y[0]) + y[0]*conj(x[0])
        //        = (1+i)*(1) + (1)*(1-i)
        //        = (1+i) + (1-i) = 2  (real)
        assert!((a[(0, 0)].re - 2.0).abs() < 1e-10);
        assert!(a[(0, 0)].im.abs() < 1e-10);

        // A[1,0] = x[1]*conj(y[0]) + y[1]*conj(x[0])
        //        = (2-i)*(1) + (i)*(1-i)
        //        = (2-i) + (i - i^2) = (2-i) + (i + 1) = 3
        assert!((a[(1, 0)].re - 3.0).abs() < 1e-10);
        assert!(a[(1, 0)].im.abs() < 1e-10);

        // A[1,1] = x[1]*conj(y[1]) + y[1]*conj(x[1])
        //        = (2-i)*(-i) + (i)*(2+i)
        //        = -2i + i^2 + 2i + i^2
        //        = -2i - 1 + 2i - 1 = -2  (real)
        assert!((a[(1, 1)].re + 2.0).abs() < 1e-10);
        assert!(a[(1, 1)].im.abs() < 1e-10);
    }

    #[test]
    fn test_her2_complex_upper() {
        let mut a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        let x = [c(1.0, 1.0), c(2.0, -1.0)];
        let y = [c(1.0, 0.0), c(0.0, 1.0)];

        her2(Her2Uplo::Upper, c(1.0, 0.0), &x, &y, a.as_mut()).unwrap();

        // A[0,0] = 2 (same as lower)
        assert!((a[(0, 0)].re - 2.0).abs() < 1e-10);
        assert!(a[(0, 0)].im.abs() < 1e-10);

        // A[0,1] = x[0]*conj(y[1]) + y[0]*conj(x[1])
        //        = (1+i)*(-i) + (1)*(2+i)
        //        = -i - i^2 + 2 + i = -i + 1 + 2 + i = 3
        assert!((a[(0, 1)].re - 3.0).abs() < 1e-10);
        assert!(a[(0, 1)].im.abs() < 1e-10);

        // A[1,1] = -2 (same as lower)
        assert!((a[(1, 1)].re + 2.0).abs() < 1e-10);
        assert!(a[(1, 1)].im.abs() < 1e-10);
    }

    #[test]
    fn test_her2_complex_alpha() {
        // Complex alpha
        let mut a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        let x = [c(1.0, 0.0), c(0.0, 1.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0)];

        // alpha = i
        her2(Her2Uplo::Lower, c(0.0, 1.0), &x, &y, a.as_mut()).unwrap();

        // A[0,0] = alpha*x[0]*conj(y[0]) + conj(alpha)*y[0]*conj(x[0])
        //        = i*(1)*(1) + (-i)*(1)*(1)
        //        = i - i = 0
        assert!(approx_eq_c(a[(0, 0)], c(0.0, 0.0)));

        // A[1,0] = alpha*x[1]*conj(y[0]) + conj(alpha)*y[1]*conj(x[0])
        //        = i*(i)*(1) + (-i)*(1)*(1)
        //        = i^2 - i = -1 - i
        assert!(approx_eq_c(a[(1, 0)], c(-1.0, -1.0)));

        // A[1,1] = alpha*x[1]*conj(y[1]) + conj(alpha)*y[1]*conj(x[1])
        //        = i*(i)*(1) + (-i)*(1)*(-i)
        //        = i^2 + i^2 = -2  (real)
        assert!((a[(1, 1)].re + 2.0).abs() < 1e-10);
        assert!(a[(1, 1)].im.abs() < 1e-10);
    }

    #[test]
    fn test_her2_same_vectors_complex() {
        // When x == y, her2 gives 2*Re(alpha)*her
        let mut a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        let x = [c(1.0, 1.0), c(2.0, 0.0)];

        her2(Her2Uplo::Lower, c(1.0, 0.0), &x, &x, a.as_mut()).unwrap();

        // A[0,0] = x[0]*conj(x[0]) + x[0]*conj(x[0]) = 2*|x[0]|^2 = 2*(1+1) = 4
        assert!((a[(0, 0)].re - 4.0).abs() < 1e-10);
        assert!(a[(0, 0)].im.abs() < 1e-10);

        // A[1,0] = x[1]*conj(x[0]) + x[1]*conj(x[0]) = 2*x[1]*conj(x[0])
        //        = 2*(2)*(1-i) = 4-4i
        assert!(approx_eq_c(a[(1, 0)], c(4.0, -4.0)));

        // A[1,1] = 2*|x[1]|^2 = 2*4 = 8
        assert!((a[(1, 1)].re - 8.0).abs() < 1e-10);
        assert!(a[(1, 1)].im.abs() < 1e-10);
    }

    #[test]
    fn test_her2_accumulate() {
        let mut a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];

        her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        // A[0,0] unchanged: 1 + 0 = 1
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        // A[1,0] = 0 + (0*0 + 1*1) = 1
        assert!((a[(1, 0)] - 1.0).abs() < 1e-10);
        // A[1,1] unchanged: 1 + 0 = 1
        assert!((a[(1, 1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2_zero_alpha() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let original = a.clone();
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        her2(Her2Uplo::Lower, 0.0, &x, &y, a.as_mut()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - original[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_her2_not_square() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        let result = her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut());
        assert!(matches!(result, Err(Her2Error::NotSquare)));
    }

    #[test]
    fn test_her2_dimension_mismatch_x() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let x = [1.0, 2.0, 3.0];
        let y = [3.0, 4.0];

        let result = her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut());
        assert!(matches!(result, Err(Her2Error::DimensionMismatchX)));
    }

    #[test]
    fn test_her2_dimension_mismatch_y() {
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);
        let x = [1.0, 2.0];
        let y = [3.0, 4.0, 5.0];

        let result = her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut());
        assert!(matches!(result, Err(Her2Error::DimensionMismatchY)));
    }

    #[test]
    fn test_her2_empty() {
        let mut a: Mat<f64> = Mat::zeros(0, 0);
        let x: [f64; 0] = [];
        let y: [f64; 0] = [];

        her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();
    }

    #[test]
    fn test_her2_1x1() {
        let mut a = Mat::from_rows(&[&[5.0f64]]);
        let x = [2.0];
        let y = [3.0];

        her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        // A[0,0] = 5 + (2*3 + 3*2) = 5 + 12 = 17
        assert!((a[(0, 0)] - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2_f32() {
        let mut a: Mat<f32> = Mat::zeros(2, 2);
        let x = [1.0f32, 2.0];
        let y = [3.0f32, 4.0];

        her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();

        assert!((a[(0, 0)] - 6.0).abs() < 1e-5);
        assert!((a[(1, 0)] - 10.0).abs() < 1e-5);
        assert!((a[(1, 1)] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_her2_new_real() {
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0];

        let a = her2_new(1.0, &x, &y).unwrap();

        // Full symmetric matrix (for real types)
        assert!((a[(0, 0)] - 6.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 10.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 10.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2_new_dimension_mismatch() {
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0, 5.0];

        let result = her2_new(1.0, &x, &y);
        assert!(matches!(result, Err(Her2Error::DimensionMismatchY)));
    }

    #[test]
    fn test_her2_diagonal_remains_real() {
        // Verify that diagonal elements remain real for complex Hermitian matrix
        let mut a: Mat<Complex64> = Mat::filled(3, 3, Complex64::new(0.0, 0.0));

        // Set some initial diagonal values
        a[(0, 0)] = c(1.0, 0.0);
        a[(1, 1)] = c(2.0, 0.0);
        a[(2, 2)] = c(3.0, 0.0);

        let x = [c(1.0, 2.0), c(3.0, -1.0), c(0.0, 1.0)];
        let y = [c(2.0, 1.0), c(-1.0, 2.0), c(1.0, 1.0)];

        her2(Her2Uplo::Lower, c(1.0, 0.5), &x, &y, a.as_mut()).unwrap();

        // All diagonal elements should remain real
        assert!(a[(0, 0)].im.abs() < 1e-10);
        assert!(a[(1, 1)].im.abs() < 1e-10);
        assert!(a[(2, 2)].im.abs() < 1e-10);
    }
}
