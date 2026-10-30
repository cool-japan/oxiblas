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

/// The column temp `alpha * conj(x)` of ?HER / ?HPR for a real `alpha`: the
/// complex product `(alpha, 0) * conj(x)`. That is how default-flag gfortran
/// evaluates the reference `TEMP = ALPHA*DCONJG(X(J))`, because Fortran converts
/// the real operand to complex before multiplying. An infinite `x` therefore
/// gives a NaN component (`0 * Inf`), as in the reference. For real `T` the
/// result is `alpha * x`.
#[inline]
pub(super) fn real_alpha_conj<T: Field>(alpha: T::Real, x: T) -> T {
    T::from_real(alpha) * x.conj()
}

/// Performs the Hermitian rank-1 update: A = alpha * x * x^H + A
///
/// Only the specified triangle (upper or lower) of A is accessed and modified.
/// The other triangle is not referenced. The diagonal elements of A are assumed
/// to be real; as in reference BLAS `?her`, the imaginary part of every diagonal
/// element is *set to zero* by the update (`A[j,j] = real(A[j,j]) + alpha * |x[j]|^2`).
/// Also as in the reference, a column with `x[j] == 0` is skipped entirely (only
/// `A[j,j] = real(A[j,j])` is applied), so an infinite or NaN `x[i]` cannot reach it.
/// If `n == 0` or `alpha == 0` the matrix is left completely untouched (including an
/// imaginary diagonal). The column temp is the complex product
/// `(alpha, 0) * conj(x[j])`, so, as in reference `?her`, an infinite `x[j]` gives a
/// NaN diagonal.
///
/// Note: alpha is real (`T::Real`); the Hermitian update with a complex scalar is
/// the rank-2 update [`her2`](crate::level2::her2).
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
                // Reference ?HER skips the column update when x(j) == 0 (so an infinite or
                // NaN x(i) cannot reach it as `x(i) * conj(0)`) and only drops the
                // imaginary part of the diagonal.
                if x[j] == T::zero() {
                    // SAFETY: `a` is an exclusive `MatMut` view and `j < n`.
                    unsafe { drop_diagonal_imag(a_ptr, rs, j) };
                    continue;
                }
                let alpha_xj_conj = real_alpha_conj(alpha, x[j]);
                for i in j..n {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i, j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: A[i,i] = real(A[i,i]) + alpha * |x[i]|^2.
                            // Like reference ?HER the stored imaginary part is
                            // dropped, so the diagonal is exactly real afterwards.
                            let val = alpha_xj_conj * x[i];
                            *ptr = T::from_real((*ptr).real() + val.real());
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
                // Column skip for x(j) == 0, as in the lower branch.
                if x[j] == T::zero() {
                    // SAFETY: `a` is an exclusive `MatMut` view and `j < n`.
                    unsafe { drop_diagonal_imag(a_ptr, rs, j) };
                    continue;
                }
                let alpha_xj_conj = real_alpha_conj(alpha, x[j]);
                for i in 0..=j {
                    // SAFETY: `a` is an exclusive `MatMut` view and `i <= j < n`, so
                    // `a_ptr.add(i + j * rs)` is an in-bounds element of that view.
                    unsafe {
                        let ptr = a_ptr.add(i + j * rs).cast_mut();
                        if i == j {
                            // Diagonal: A[i,i] = real(A[i,i]) + alpha * |x[i]|^2
                            // (stored imaginary part dropped, as in reference ?HER).
                            let val = alpha_xj_conj * x[i];
                            *ptr = T::from_real((*ptr).real() + val.real());
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

    #[test]
    fn test_her_new_complex() {
        // num-complex's `bytemuck` feature (enabled workspace-wide) makes
        // `Complex64: bytemuck::Zeroable`, so `her_new` works for complex types.
        let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];

        let a = her_new(1.0, &x);

        // A = x * x^H = [[2, (1+i)(2+i)], [(2-i)(1-i), 5]] = [[2, 1+3i], [1-3i, 5]].
        assert_eq!(a[(0, 0)], Complex64::new(2.0, 0.0));
        assert_eq!(a[(1, 0)], Complex64::new(1.0, -3.0));
        assert_eq!(a[(0, 1)], Complex64::new(1.0, 3.0));
        assert_eq!(a[(1, 1)], Complex64::new(5.0, 0.0));
    }

    // Issue #13: reference ?HER sets the imaginary part of every diagonal element
    // to zero (A(j,j) = real(A(j,j)) + real(x(j)*temp), or real(A(j,j)) when x(j)
    // is zero) whenever n > 0 and alpha != 0.

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Reporter's repro: n = 1, alpha = 1, x = [1], A = [2+0.7i] -> 3+0i (not 3+0.7i).
    #[test]
    fn test_issue_13_her_zeroes_diagonal_imaginary_part() {
        for uplo in [HerUplo::Lower, HerUplo::Upper] {
            let mut a: Mat<Complex64> = Mat::filled(1, 1, c(2.0, 0.7));
            her(uplo, 1.0, &[c(1.0, 0.0)], a.as_mut()).unwrap();
            assert_eq!(a[(0, 0)], c(3.0, 0.0), "{uplo:?}");
        }
    }

    /// x(j) == 0 still rewrites the diagonal as `real(A(j,j))` (imaginary part
    /// dropped), and the unreferenced triangle is left alone.
    #[test]
    fn test_issue_13_her_zero_x_element_still_zeroes_diagonal_imaginary_part() {
        let junk = c(9.0, 9.0);
        // Lower triangle stored; the strict upper triangle is junk that must survive.
        let mut a: Mat<Complex64> = Mat::from_rows(&[
            &[c(1.0, 0.5), junk, junk],
            &[c(0.2, 0.1), c(2.0, 0.7), junk],
            &[c(0.3, -0.2), c(0.4, 0.6), c(3.0, -0.8)],
        ]);
        let x = [c(1.0, 1.0), c(0.0, 0.0), c(2.0, -1.0)];

        her(HerUplo::Lower, 0.5, &x, a.as_mut()).unwrap();

        // temp0 = 0.5*conj(1+i) = 0.5-0.5i; A20 += x2*temp0 = (2-i)(0.5-0.5i) = 0.5-1.5i.
        // A00 = 1 + real((1+i)*temp0) = 2; A11 = real(2+0.7i) = 2 (x1 == 0);
        // A22 = 3 + real((2-i)*0.5*(2+i)) = 3 + 2.5 = 5.5.
        assert_eq!(a[(0, 0)], c(2.0, 0.0));
        assert_eq!(a[(1, 0)], c(0.2, 0.1));
        assert_eq!(a[(1, 1)], c(2.0, 0.0));
        assert_eq!(a[(2, 0)], c(0.3 + 0.5, -0.2 - 1.5));
        assert_eq!(a[(2, 1)], c(0.4, 0.6));
        assert_eq!(a[(2, 2)], c(5.5, 0.0));
        assert_eq!(a[(0, 1)], junk);
        assert_eq!(a[(0, 2)], junk);
        assert_eq!(a[(1, 2)], junk);
    }

    #[test]
    fn test_issue_13_her_upper_zeroes_every_diagonal_imaginary_part() {
        let mut a: Mat<Complex64> =
            Mat::from_rows(&[&[c(1.0, 0.5), c(0.2, 0.1)], &[c(7.0, 7.0), c(2.0, -0.7)]]);
        let x = [c(0.0, 0.0), c(0.0, 2.0)];

        her(HerUplo::Upper, 1.0, &x, a.as_mut()).unwrap();

        // x0 == 0 -> A00 = real(A00) = 1 and A01 is untouched (temp0 = 0);
        // x1 = 2i -> A11 = 2 + |x1|^2 = 6, A01 += x0*conj(x1)*alpha = 0.
        assert_eq!(a[(0, 0)], c(1.0, 0.0));
        assert_eq!(a[(0, 1)], c(0.2, 0.1));
        assert_eq!(a[(1, 1)], c(6.0, 0.0));
        assert_eq!(a[(1, 0)], c(7.0, 7.0));
    }

    /// alpha == 0 (or n == 0) leaves A untouched, including an imaginary diagonal.
    #[test]
    fn test_issue_13_her_zero_alpha_leaves_complex_diagonal_untouched() {
        let mut a: Mat<Complex64> =
            Mat::from_rows(&[&[c(1.0, 0.5), c(0.2, 0.1)], &[c(0.3, 0.2), c(2.0, -0.7)]]);
        let before = a.clone();
        her(HerUplo::Lower, 0.0, &[c(1.0, 1.0), c(2.0, 2.0)], a.as_mut()).unwrap();
        for j in 0..2 {
            for i in 0..2 {
                assert_eq!(a[(i, j)], before[(i, j)], "({i},{j})");
            }
        }
    }

    /// Reference ?HER skips the whole column update when x(j) == 0 and only rewrites the
    /// diagonal as `real(A(j,j))`: an infinite x(i) elsewhere must not reach that column
    /// (`x(i) * conj(0)` would be `Inf * 0 = NaN`). Real arithmetic keeps the expected
    /// values exact.
    #[test]
    fn test_issue_13_her_skips_column_of_zero_x_element() {
        let x = [0.0f64, f64::INFINITY, 2.0];

        // Lower: column 0 (x0 == 0) is skipped.
        let mut a = Mat::from_rows(&[&[1.0f64, 9.0, 9.0], &[0.5, 2.0, 9.0], &[0.25, 0.75, 3.0]]);
        her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(1, 0)], 0.5);
        assert_eq!(a[(2, 0)], 0.25);
        // Column 1: temp = alpha * x1 = Inf, so A21 = 0.75 + 2*Inf and A11 = 2 + Inf*Inf.
        assert_eq!(a[(2, 1)], f64::INFINITY);
        assert_eq!(a[(1, 1)], f64::INFINITY);
        // Column 2: A22 = 3 + 2*2.
        assert_eq!(a[(2, 2)], 7.0);
        // The unreferenced triangle is untouched.
        assert_eq!((a[(0, 1)], a[(0, 2)], a[(1, 2)]), (9.0, 9.0, 9.0));

        // Upper: column 2 (x2 == 0 here) is skipped.
        let x = [2.0f64, f64::INFINITY, 0.0];
        let mut a = Mat::from_rows(&[&[1.0f64, 0.5, 0.25], &[9.0, 2.0, 0.75], &[9.0, 9.0, 3.0]]);
        her(HerUplo::Upper, 1.0, &x, a.as_mut()).unwrap();
        assert_eq!(a[(0, 2)], 0.25);
        assert_eq!(a[(1, 2)], 0.75);
        assert_eq!(a[(2, 2)], 3.0);
        assert_eq!(a[(0, 0)], 5.0);
        assert_eq!(a[(0, 1)], f64::INFINITY);
        assert_eq!(a[(1, 1)], f64::INFINITY);
        assert_eq!((a[(1, 0)], a[(2, 0)], a[(2, 1)]), (9.0, 9.0, 9.0));
    }

    /// Complex counterpart: the skipped column stays untouched (whatever is going on in
    /// the other columns) and its diagonal loses its imaginary part.
    #[test]
    fn test_issue_13_her_complex_zero_x_column_untouched_with_real_diagonal() {
        let junk = c(9.0, 9.0);
        let mut a: Mat<Complex64> = Mat::from_rows(&[
            &[c(1.0, 0.5), junk, junk],
            &[c(0.2, 0.1), c(2.0, 0.7), junk],
            &[c(0.3, -0.2), c(0.4, 0.6), c(3.0, -0.8)],
        ]);
        let x = [c(0.0, 0.0), c(f64::INFINITY, 0.0), c(2.0, 0.0)];
        her(HerUplo::Lower, 1.0, &x, a.as_mut()).unwrap();
        assert_eq!(a[(0, 0)], c(1.0, 0.0));
        assert_eq!(a[(1, 0)], c(0.2, 0.1));
        assert_eq!(a[(2, 0)], c(0.3, -0.2));
    }

    fn bits(z: Complex64) -> (u64, u64) {
        (z.re.to_bits(), z.im.to_bits())
    }

    /// Issue #13: reference ?HER forms `TEMP = ALPHA*DCONJG(X(J))` as the complex
    /// product `(alpha, 0) * conj(x(j))` (Fortran converts the real `ALPHA` to
    /// complex), so an infinite `x(j)` puts `0 * Inf = NaN` into one component of
    /// the temp and the diagonal `DBLE(A(J,J)) + DBLE(X(J)*TEMP)` becomes
    /// `(NaN, +0)`. The expected values follow zher.f; NaN bit patterns are not
    /// asserted (the default NaN differs between targets).
    #[test]
    fn test_issue_13_her_infinite_x_diagonal_matches_reference() {
        let inf = f64::INFINITY;
        for alpha in [0.5, -2.0] {
            for xj in [c(inf, 0.0), c(0.0, inf), c(-inf, 1.0)] {
                for uplo in [HerUplo::Lower, HerUplo::Upper] {
                    let mut a: Mat<Complex64> = Mat::from_rows(&[
                        &[c(1.0, 0.5), c(2.0, -1.0)],
                        &[c(2.0, 1.0), c(3.0, 0.5)],
                    ]);
                    her(uplo, alpha, &[c(0.0, 0.0), xj], a.as_mut()).unwrap();
                    let what = format!("{uplo:?} alpha={alpha} x1={xj}");
                    // A11 = real(A11) + real(x1 * temp), where temp holds a NaN part.
                    assert!(a[(1, 1)].re.is_nan(), "{what}");
                    assert_eq!(a[(1, 1)].im.to_bits(), 0.0f64.to_bits(), "{what}");
                    // Column 0 (x0 == 0) is skipped; only its diagonal loses the
                    // imaginary part.
                    assert_eq!(bits(a[(0, 0)]), bits(c(1.0, 0.0)), "{what}");
                    match uplo {
                        // A01 += x0 * temp: 0 times a NaN and an infinite part.
                        HerUplo::Upper => {
                            assert!(a[(0, 1)].re.is_nan() && a[(0, 1)].im.is_nan(), "{what}");
                        }
                        // Column 0 is skipped and column 1 has no subdiagonal.
                        HerUplo::Lower => {
                            assert_eq!(bits(a[(1, 0)]), bits(c(2.0, 1.0)), "{what}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_issue_13_her_complex32_zeroes_diagonal_imaginary_part() {
        use num_complex::Complex32;
        let mut a: Mat<Complex32> = Mat::filled(1, 1, Complex32::new(2.0, 0.7));
        her(
            HerUplo::Lower,
            1.0f32,
            &[Complex32::new(1.0, 0.0)],
            a.as_mut(),
        )
        .unwrap();
        assert_eq!(a[(0, 0)], Complex32::new(3.0, 0.0));
    }
}
