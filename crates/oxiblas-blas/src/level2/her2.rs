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

/// Rewrites the diagonal element `(j, j)` as `real(A[j,j])`, i.e. drops its imaginary part.
///
/// # Safety
///
/// `a_ptr.add(j + j * rs)` must be a valid element of an exclusively borrowed matrix view.
#[inline]
unsafe fn drop_diagonal_imag<T: Field>(a_ptr: *const T, rs: usize, j: usize) {
    let ptr = a_ptr.add(j + j * rs).cast_mut();
    *ptr = T::from_real((*ptr).real());
}

/// Performs the Hermitian rank-2 update: A = alpha * x * y^H + conj(alpha) * y * x^H + A
///
/// Only the specified triangle (upper or lower) of A is accessed and modified.
/// The other triangle is not referenced. The diagonal elements of A are assumed
/// to be real; as in reference BLAS `?her2`, the imaginary part of every diagonal
/// element is *set to zero* by the update (`A[j,j] = real(A[j,j]) + real(update)`).
/// Also as in the reference, a column with `x[j]` and `y[j]` both zero is skipped
/// entirely (only `A[j,j] = real(A[j,j])` is applied), so an infinite or NaN neighbour
/// cannot reach it. If `n == 0` or `alpha == 0` the matrix is left completely
/// untouched (including an imaginary diagonal).
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

    // The column temps are the reference ones, `TEMP1 = ALPHA*DCONJG(Y(J))` and
    // `TEMP2 = DCONJG(ALPHA*X(J))`: conjugating the product rather than its factors
    // gives the reference sign of an exact-zero imaginary part.
    match uplo {
        Her2Uplo::Lower => {
            // Update lower triangle: A[i,j] += alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j])
            // for i >= j
            for j in 0..n {
                // Reference ?HER2 skips the column update when x(j) == 0 and y(j) == 0 (so an
                // infinite or NaN neighbour cannot reach it as `x(i) * conj(0)`) and only
                // drops the imaginary part of the diagonal.
                if x[j] == T::zero() && y[j] == T::zero() {
                    // SAFETY: `a` is an exclusive `MatMut` view and `j < n`.
                    unsafe { drop_diagonal_imag(a_ptr, rs, j) };
                    continue;
                }
                let alpha_yj_conj = alpha * y[j].conj();
                let conj_alpha_xj = (alpha * x[j]).conj();
                for i in j..n {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i, j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: A[i,i] = real(A[i,i]) + real(alpha*x[i]*conj(y[i])
                            // + conj(alpha)*y[i]*conj(x[i])). The sum is real for a Hermitian
                            // update; like reference ?HER2 the stored imaginary part is
                            // dropped, so the diagonal is exactly real afterwards.
                            let val = alpha_yj_conj * x[i] + conj_alpha_xj * y[i];
                            *ptr = T::from_real((*ptr).real() + val.real());
                        } else {
                            *ptr = *ptr + alpha_yj_conj * x[i] + conj_alpha_xj * y[i];
                        }
                    }
                }
            }
        }
        Her2Uplo::Upper => {
            // Update upper triangle: A[i,j] += alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j])
            // for i <= j
            for j in 0..n {
                // Column skip for x(j) == 0 && y(j) == 0, as in the lower branch.
                if x[j] == T::zero() && y[j] == T::zero() {
                    // SAFETY: `a` is an exclusive `MatMut` view and `j < n`.
                    unsafe { drop_diagonal_imag(a_ptr, rs, j) };
                    continue;
                }
                let alpha_yj_conj = alpha * y[j].conj();
                let conj_alpha_xj = (alpha * x[j]).conj();
                for i in 0..=j {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i <= j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: real(A[i,i]) + real(update); stored imaginary
                            // part dropped, as in reference ?HER2.
                            let val = alpha_yj_conj * x[i] + conj_alpha_xj * y[i];
                            *ptr = T::from_real((*ptr).real() + val.real());
                        } else {
                            *ptr = *ptr + alpha_yj_conj * x[i] + conj_alpha_xj * y[i];
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
/// The lower triangle is computed by [`her2`] with [`Her2Uplo::Lower`] on a zero
/// matrix, so it has the reference `?her2` semantics
/// (`A[i,j] = 0 + x[i] * temp1 + y[i] * temp2`, evaluated left to right, with
/// `temp1 = alpha * conj(y[j])` and `temp2 = conj(alpha * x[j])`), and the
/// strictly upper triangle is its conjugated mirror, `A[j,i] = conj(A[i,j])`. In
/// particular:
///
/// * A column with `x[j] == 0` and `y[j] == 0` is skipped and stays zero, so an
///   infinite or NaN neighbour cannot reach it. Like the reference this is order
///   dependent: `x = [0, Inf]`, `y = [0, 1]` gives `A[1,0] = A[0,1] = 0`, but
///   `x = [Inf, 0]`, `y = [1, 0]` gives `A[1,0] = NaN` (from
///   `y[1] * conj(alpha * x[0]) = 0 * Inf`), mirrored into `A[0,1]`.
/// * The diagonal is exactly real.
/// * Every lower entry is accumulated onto `+0`, so a `-0` result (a sum of `-0`
///   products) is stored as `+0`. The upper entries are exact conjugates, so a
///   zero imaginary part there has the opposite sign.
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

    // Reference-semantics lower triangle. `a` is `n x n` and both vectors have
    // length `n`, so this cannot fail; an error would be propagated as is.
    her2(Her2Uplo::Lower, alpha, x, y, a.as_mut())?;

    // Mirror the strictly lower triangle into the upper one, conjugated.
    for j in 0..n {
        for i in (j + 1)..n {
            a[(j, i)] = a[(i, j)].conj();
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

    // Issue #13: reference ?HER2 sets the imaginary part of every diagonal element
    // to zero (A(j,j) = real(A(j,j)) + real(x(j)*temp1 + y(j)*temp2), or
    // real(A(j,j)) when x(j) and y(j) are both zero) whenever n > 0 and alpha != 0.

    /// Reporter's repro: x = y = [1], alpha = 1, A = [2+0.7i] -> 4+0i (not 4+0.7i).
    #[test]
    fn test_issue_13_her2_zeroes_diagonal_imaginary_part() {
        for uplo in [Her2Uplo::Lower, Her2Uplo::Upper] {
            let mut a: Mat<Complex64> = Mat::filled(1, 1, c(2.0, 0.7));
            let x = [c(1.0, 0.0)];
            her2(uplo, c(1.0, 0.0), &x, &x, a.as_mut()).unwrap();
            assert_eq!(a[(0, 0)], c(4.0, 0.0), "{uplo:?}");
        }
    }

    /// x(j) == y(j) == 0 still rewrites the diagonal as `real(A(j,j))`, and the
    /// unreferenced triangle is left alone.
    #[test]
    fn test_issue_13_her2_zero_vector_element_still_zeroes_diagonal_imaginary_part() {
        let junk = c(9.0, 9.0);
        // Upper triangle stored; the strict lower triangle is junk that must survive.
        let mut a: Mat<Complex64> = Mat::from_rows(&[
            &[c(1.0, 0.5), c(0.2, 0.1), c(0.3, -0.2)],
            &[junk, c(2.0, 0.7), c(0.4, 0.6)],
            &[junk, junk, c(3.0, -0.8)],
        ]);
        let x = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0)];
        let y = [c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];

        her2(Her2Uplo::Upper, c(1.0, 0.0), &x, &y, a.as_mut()).unwrap();

        // A00 = 1 + (1*1 + 1*1) = 3; A11 = real(2+0.7i) = 2 (x1 = y1 = 0);
        // A22 = 3 + real(x2*conj(y2) + y2*conj(x2)) = 3 + real(i + (-i)) = 3.
        assert_eq!(a[(0, 0)], c(3.0, 0.0));
        assert_eq!(a[(1, 1)], c(2.0, 0.0));
        assert_eq!(a[(2, 2)], c(3.0, 0.0));
        // A01 += x0*conj(y1) + y0*conj(x1) = 0.
        assert_eq!(a[(0, 1)], c(0.2, 0.1));
        // A02: x0*conj(y2) = 1*1 = 1, y0*conj(x2) = 1*(-i) = -i.
        assert_eq!(a[(0, 2)], c(0.3 + 1.0, -0.2 - 1.0));
        // A12: x1 = y1 = 0 -> x1*conj(y2) + y1*conj(x2) = 0.
        assert_eq!(a[(1, 2)], c(0.4, 0.6));
        assert_eq!(a[(1, 0)], junk);
        assert_eq!(a[(2, 0)], junk);
        assert_eq!(a[(2, 1)], junk);
    }

    /// Reference ?HER2 skips the column update only when BOTH x(j) and y(j) are zero (then
    /// just `A(j,j) = real(A(j,j))`): an infinite neighbour must not reach that column
    /// (`x(i) * conj(0)` would be `Inf * 0 = NaN`), but a zero x(j) with a non-zero y(j)
    /// is still updated. Real arithmetic keeps the expected values exact.
    #[test]
    fn test_issue_13_her2_skips_column_only_when_x_and_y_are_zero() {
        let x = [0.0f64, f64::INFINITY, 2.0];
        let y = [0.0f64, 1.0, 3.0];
        let mut a = Mat::from_rows(&[&[1.0f64, 9.0, 9.0], &[0.5, 2.0, 9.0], &[0.25, 0.75, 3.0]]);
        her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();
        // Column 0 (x0 == y0 == 0) is skipped.
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(1, 0)], 0.5);
        assert_eq!(a[(2, 0)], 0.25);
        // Column 1: temp1 = y1 = 1, temp2 = x1 = Inf -> A21 = 0.75 + 2*1 + 3*Inf, A11 = 2 + Inf + Inf.
        assert_eq!(a[(2, 1)], f64::INFINITY);
        assert_eq!(a[(1, 1)], f64::INFINITY);
        // Column 2: A22 = 3 + x2*y2 + y2*x2 = 3 + 6 + 6.
        assert_eq!(a[(2, 2)], 15.0);
        assert_eq!((a[(0, 1)], a[(0, 2)], a[(1, 2)]), (9.0, 9.0, 9.0));

        // x(0) == 0 but y(0) != 0: the column is NOT skipped.
        let x = [0.0f64, 1.0];
        let y = [1.0f64, 0.0];
        let mut a = Mat::from_rows(&[&[1.0f64, 9.0], &[0.5, 2.0]]);
        her2(Her2Uplo::Lower, 1.0, &x, &y, a.as_mut()).unwrap();
        // A10 = 0.5 + x1*y0 + y1*x0 = 0.5 + 1.
        assert_eq!(a[(1, 0)], 1.5);
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(1, 1)], 2.0);
    }

    /// Complex counterpart: the skipped column stays untouched and its diagonal loses its
    /// imaginary part (whatever happens in the non-skipped columns is not asserted).
    #[test]
    fn test_issue_13_her2_complex_zero_column_untouched_with_real_diagonal() {
        let junk = c(9.0, 9.0);
        let mut a: Mat<Complex64> = Mat::from_rows(&[
            &[c(1.0, 0.5), c(0.2, 0.1), c(0.3, -0.2)],
            &[junk, c(2.0, 0.7), c(0.4, 0.6)],
            &[junk, junk, c(3.0, -0.8)],
        ]);
        let x = [c(2.0, 0.0), c(f64::INFINITY, 0.0), c(0.0, 0.0)];
        let y = [c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)];
        her2(Her2Uplo::Upper, c(1.0, 0.0), &x, &y, a.as_mut()).unwrap();
        // Column 2 (x2 == y2 == 0) is skipped.
        assert_eq!(a[(0, 2)], c(0.3, -0.2));
        assert_eq!(a[(1, 2)], c(0.4, 0.6));
        assert_eq!(a[(2, 2)], c(3.0, 0.0));
    }

    /// alpha == 0 (or n == 0) leaves A untouched, including an imaginary diagonal.
    #[test]
    fn test_issue_13_her2_zero_alpha_leaves_complex_diagonal_untouched() {
        let mut a: Mat<Complex64> =
            Mat::from_rows(&[&[c(1.0, 0.5), c(0.2, 0.1)], &[c(0.3, 0.2), c(2.0, -0.7)]]);
        let before = a.clone();
        let x = [c(1.0, 1.0), c(2.0, 2.0)];
        her2(Her2Uplo::Lower, c(0.0, 0.0), &x, &x, a.as_mut()).unwrap();
        for j in 0..2 {
            for i in 0..2 {
                assert_eq!(a[(i, j)], before[(i, j)], "({i},{j})");
            }
        }
    }

    /// A purely imaginary alpha: the Hermitian sum on the diagonal is real, so the
    /// stored imaginary part must still end up zero.
    #[test]
    fn test_issue_13_her2_complex_alpha_zeroes_diagonal_imaginary_part() {
        let mut a: Mat<Complex64> = Mat::filled(2, 2, c(1.0, 0.25));
        let x = [c(1.0, 1.0), c(2.0, -1.0)];
        let y = [c(0.5, 0.0), c(0.0, 1.0)];

        her2(Her2Uplo::Lower, c(0.0, 1.0), &x, &y, a.as_mut()).unwrap();

        assert_eq!(a[(0, 0)].im, 0.0);
        assert_eq!(a[(1, 1)].im, 0.0);
        // A00 = 1 + real(alpha*x0*conj(y0) + conj(alpha)*y0*conj(x0))
        //     = 1 + real(i*(1+i)*0.5 + (-i)*0.5*(1-i)) = 1 + real((-0.5+0.5i) + (-0.5-0.5i)) = 0.
        assert!((a[(0, 0)].re - 0.0).abs() < 1e-12, "{:?}", a[(0, 0)]);
    }

    fn bits(z: Complex64) -> (u64, u64) {
        (z.re.to_bits(), z.im.to_bits())
    }

    /// Reference ?HER2 / ?HPR2 form `TEMP2 = DCONJG(ALPHA*X(J))`. With alpha = 1+i
    /// and x(j) = 1-i, `re(alpha)*im(x(j)) + im(alpha)*re(x(j)) == 0` exactly, so
    /// `conj(alpha*x(j)) = (2, -0)` while `conj(alpha)*conj(x(j)) = (2, +0)`. The
    /// off-diagonal `A(i,j) + x(i)*temp1 + y(i)*temp2` below has a running imaginary
    /// part of -0 and `im(y(i)) = -0`, so it keeps the sign of that zero and must
    /// match the reference expression bit for bit, in `her2` and in `hpr2`.
    #[test]
    fn test_issue_13_her2_temp2_matches_reference_signed_zero() {
        use crate::level2::{Hpr2Uplo, hpr2};
        let alpha = c(1.0, 1.0);
        let (xj, yj) = (c(1.0, -1.0), c(1.0, 0.0));
        let (xi, yi) = (c(-0.0, -0.0), c(0.25, -0.0));
        let aij = c(0.5, -0.0);
        // Reference temps: TEMP1 = ALPHA*DCONJG(Y(J)), TEMP2 = DCONJG(ALPHA*X(J)).
        let temp1 = alpha * yj.conj();
        let temp2 = (alpha * xj).conj();
        assert_eq!(bits(temp2), bits(c(2.0, -0.0)));
        // A(I,J) + X(I)*TEMP1 + Y(I)*TEMP2, evaluated left to right as in ZHER2.
        let want = aij + xi * temp1 + yi * temp2;
        assert_eq!(bits(want), bits(c(1.0, -0.0)));
        // The data discriminates: conjugating the factors instead flips the zero.
        let conj_factors = alpha.conj() * xj.conj();
        assert_eq!(bits(conj_factors), bits(c(2.0, 0.0)));
        assert_ne!(bits(aij + xi * temp1 + yi * conj_factors), bits(want));

        let (d0, d1, junk) = (c(1.0, 0.0), c(2.0, 0.0), c(9.0, 9.0));
        // Lower: (i, j) = (1, 0); packed [A00, A10, A11].
        let (x, y) = ([xj, xi], [yj, yi]);
        let mut a: Mat<Complex64> = Mat::from_rows(&[&[d0, junk], &[aij, d1]]);
        her2(Her2Uplo::Lower, alpha, &x, &y, a.as_mut()).unwrap();
        assert_eq!(bits(a[(1, 0)]), bits(want), "her2 Lower");
        let mut ap = [d0, aij, d1];
        hpr2(Hpr2Uplo::Lower, 2, alpha, &x, &y, &mut ap).unwrap();
        assert_eq!(bits(ap[1]), bits(want), "hpr2 Lower");
        // Upper: (i, j) = (0, 1); packed [A00, A01, A11].
        let (x, y) = ([xi, xj], [yi, yj]);
        let mut a: Mat<Complex64> = Mat::from_rows(&[&[d0, aij], &[junk, d1]]);
        her2(Her2Uplo::Upper, alpha, &x, &y, a.as_mut()).unwrap();
        assert_eq!(bits(a[(0, 1)]), bits(want), "her2 Upper");
        let mut ap = [d0, aij, d1];
        hpr2(Hpr2Uplo::Upper, 2, alpha, &x, &y, &mut ap).unwrap();
        assert_eq!(bits(ap[1]), bits(want), "hpr2 Upper");
    }

    /// `her2_new` builds its lower triangle with `her2(Lower, ..)` on a zero matrix,
    /// so the reference zero-column skip applies: for `x = [0, Inf]`, `y = [0, 1]`
    /// column 0 is skipped, `A[1,0]` stays `0` and is mirrored into `A[0,1]`. (The
    /// old direct products `alpha * x[1] * conj(y[0]) = Inf * 0` made both NaN.)
    #[test]
    fn test_her2_new_skips_zero_column() {
        let inf = f64::INFINITY;

        let a = her2_new(1.0, &[0.0f64, inf], &[0.0, 1.0]).unwrap();
        assert_eq!((a[(0, 0)], a[(1, 0)], a[(0, 1)]), (0.0, 0.0, 0.0));
        assert_eq!(a[(1, 1)], inf);

        let x = [c(0.0, 0.0), c(inf, 0.0)];
        let y = [c(0.0, 0.0), c(1.0, 0.0)];
        let a = her2_new(c(1.0, 0.0), &x, &y).unwrap();
        assert_eq!(a[(0, 0)], c(0.0, 0.0));
        assert_eq!(a[(1, 0)], c(0.0, 0.0));
        assert_eq!(a[(0, 1)], c(0.0, 0.0));
        assert_eq!(a[(1, 1)].im.to_bits(), 0.0f64.to_bits());

        // The lower-triangle semantics are order dependent, as in the reference:
        // with x = [Inf, 0], y = [1, 0] column 0 is not skipped and A[1,0] picks up
        // `y[1] * conj(alpha * x[0]) = 0 * Inf` (NaN, mirrored), while the skipped
        // column 1 stays zero.
        let a = her2_new(1.0, &[inf, 0.0f64], &[1.0, 0.0]).unwrap();
        assert_eq!(a[(0, 0)], inf);
        assert!(a[(1, 0)].is_nan() && a[(0, 1)].is_nan());
        assert_eq!(a[(1, 1)], 0.0);
    }

    /// Every lower entry of `her2_new` is `(0 + p1) + p2` (a skipped column stays
    /// `+0`), so a `-0` sum of products is stored as `+0`. The old direct sums kept
    /// the `-0`.
    #[test]
    fn test_her2_new_negative_zero_product_is_positive_zero() {
        let a = her2_new(-1.0, &[1.0f64, 0.0], &[1.0, 0.0]).unwrap();
        assert_eq!(a[(0, 0)], -2.0);
        for (i, j) in [(1, 0), (0, 1), (1, 1)] {
            assert_eq!(a[(i, j)].to_bits(), 0.0f64.to_bits(), "({i},{j})");
        }
    }

    /// `her2_new` equals `her2(Lower, ..)` on a zero matrix bit for bit (reference
    /// temps and evaluation order), with the strictly lower triangle mirrored,
    /// conjugated, into the upper one; mismatched lengths still report
    /// `DimensionMismatchY`.
    #[test]
    fn test_her2_new_matches_lower_her2_on_zero_matrix() {
        let x = [c(1.0, 0.5), c(-0.3, 2.0), c(0.0, 0.0), c(0.7, -1.1)];
        let y = [c(0.2, -0.4), c(0.0, 0.0), c(0.0, 0.0), c(-1.5, 0.25)];
        let alpha = c(0.37, -1.2);
        let a = her2_new(alpha, &x, &y).unwrap();
        let mut lower: Mat<Complex64> = Mat::zeros(4, 4);
        her2(Her2Uplo::Lower, alpha, &x, &y, lower.as_mut()).unwrap();
        for j in 0..4 {
            assert_eq!(bits(a[(j, j)]), bits(lower[(j, j)]), "({j},{j})");
            for i in (j + 1)..4 {
                assert_eq!(bits(a[(i, j)]), bits(lower[(i, j)]), "({i},{j})");
                assert_eq!(bits(a[(j, i)]), bits(lower[(i, j)].conj()), "({j},{i})");
            }
        }
        assert_eq!(
            her2_new(alpha, &x, &y[..3]).map(|m| m.nrows()),
            Err(Her2Error::DimensionMismatchY)
        );
    }
}
