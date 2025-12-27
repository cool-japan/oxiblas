//! Hermitian matrix-vector multiply (HEMV).
//!
//! Computes y = α·A·x + β·y where A is a Hermitian matrix.

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Specifies which triangle of the Hermitian matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HemvUplo {
    /// Upper triangle is stored.
    Upper,
    /// Lower triangle is stored.
    Lower,
}

/// Error type for HEMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HemvError {
    /// Matrix is not square.
    NotSquare,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for HemvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::DimensionMismatchX => write!(f, "Vector x has wrong dimension"),
            Self::DimensionMismatchY => write!(f, "Vector y has wrong dimension"),
        }
    }
}

impl std::error::Error for HemvError {}

/// Hermitian matrix-vector multiply.
///
/// Computes y = α·A·x + β·y where A is a Hermitian n×n matrix.
///
/// For a Hermitian matrix, A\[i,j\] = conj(A\[j,i\]) and diagonal elements are real.
/// Only the triangle specified by `uplo` is accessed.
///
/// # Arguments
///
/// * `uplo` - Specifies whether upper or lower triangle is stored
/// * `alpha` - Scalar multiplier for A·x
/// * `a` - Hermitian matrix (only one triangle is read)
/// * `x` - Input vector of length n
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector of length n (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{hemv, HemvUplo};
/// use oxiblas_matrix::Mat;
/// use num_complex::Complex64;
///
/// fn c(re: f64, im: f64) -> Complex64 {
///     Complex64::new(re, im)
/// }
///
/// // Hermitian matrix A = [[2, 1+i], [1-i, 3]]
/// let a = Mat::from_rows(&[
///     &[c(2.0, 0.0), c(1.0, 1.0)],
///     &[c(0.0, 0.0), c(3.0, 0.0)], // Lower part ignored for Upper
/// ]);
/// let x = [c(1.0, 0.0), c(1.0, 0.0)];
/// let mut y = [c(0.0, 0.0); 2];
///
/// hemv(HemvUplo::Upper, c(1.0, 0.0), a.as_ref(), &x, c(0.0, 0.0), &mut y).unwrap();
///
/// // y = A*x = [2+(1+i), (1-i)+3] = [3+i, 4-i]
/// assert!((y[0].re - 3.0).abs() < 1e-10);
/// assert!((y[0].im - 1.0).abs() < 1e-10);
/// assert!((y[1].re - 4.0).abs() < 1e-10);
/// assert!((y[1].im - (-1.0)).abs() < 1e-10);
/// ```
pub fn hemv<T: Field>(
    uplo: HemvUplo,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<(), HemvError> {
    let n = a.nrows();

    // Validate dimensions
    if a.ncols() != n {
        return Err(HemvError::NotSquare);
    }
    if x.len() != n {
        return Err(HemvError::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(HemvError::DimensionMismatchY);
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
        HemvUplo::Upper => {
            // Upper triangle stored: access a[i,j] for i <= j
            for i in 0..n {
                let mut temp1 = T::zero();
                let alpha_xi = alpha * x[i];

                // Diagonal element (must be real for Hermitian, but we use it as-is)
                y[i] += alpha * a[(i, i)] * x[i];

                // Off-diagonal elements (j > i)
                for j in (i + 1)..n {
                    let aij = a[(i, j)];
                    // A[i,j] * x[j] contributes to y[i]
                    temp1 += aij * x[j];
                    // conj(A[i,j]) * x[i] = A[j,i] * x[i] contributes to y[j]
                    y[j] += alpha_xi * aij.conj();
                }

                y[i] += alpha * temp1;
            }
        }
        HemvUplo::Lower => {
            // Lower triangle stored: access a[i,j] for i >= j
            for i in 0..n {
                let mut temp1 = T::zero();
                let alpha_xi = alpha * x[i];

                // Diagonal element
                y[i] += alpha * a[(i, i)] * x[i];

                // Off-diagonal elements (j < i)
                for j in 0..i {
                    let aij = a[(i, j)];
                    // A[i,j] * x[j] contributes to y[i]
                    temp1 += aij * x[j];
                    // conj(A[i,j]) * x[i] = A[j,i] * x[i] contributes to y[j]
                    y[j] += alpha_xi * aij.conj();
                }

                y[i] += alpha * temp1;
            }
        }
    }

    Ok(())
}

/// New-style Hermitian matrix-vector multiply that allocates the result.
///
/// Computes y = α·A·x where A is a Hermitian n×n matrix.
pub fn hemv_new<T: Field>(
    uplo: HemvUplo,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
) -> Result<Vec<T>, HemvError> {
    let n = a.nrows();
    let mut y = vec![T::zero(); n];
    hemv(uplo, alpha, a, x, T::zero(), &mut y)?;
    Ok(y)
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
    fn test_hemv_upper_basic() {
        // Hermitian matrix A = [[2, 1+i], [1-i, 3]]
        let a = Mat::from_rows(&[&[c(2.0, 0.0), c(1.0, 1.0)], &[c(0.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        hemv(
            HemvUplo::Upper,
            c(1.0, 0.0),
            a.as_ref(),
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
    fn test_hemv_lower_basic() {
        // Hermitian matrix A = [[2, 1+i], [1-i, 3]]
        let a = Mat::from_rows(&[&[c(2.0, 0.0), c(0.0, 0.0)], &[c(1.0, -1.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        hemv(
            HemvUplo::Lower,
            c(1.0, 0.0),
            a.as_ref(),
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
    fn test_hemv_real_matrix() {
        // Real symmetric matrix (special case of Hermitian)
        let a = Mat::from_rows(&[&[c(2.0, 0.0), c(1.0, 0.0)], &[c(0.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(2.0, 0.0)];
        let mut y = [c(0.0, 0.0); 2];

        hemv(
            HemvUplo::Upper,
            c(1.0, 0.0),
            a.as_ref(),
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
    fn test_hemv_with_alpha_beta() {
        let a = Mat::from_rows(&[&[c(2.0, 0.0), c(1.0, 1.0)], &[c(0.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(1.0, 0.0)];
        let mut y = [c(1.0, 0.0), c(1.0, 0.0)];

        // y = 2*A*x + i*y
        // A*x = [3+i, 4-i]
        // y = 2*[3+i, 4-i] + i*[1, 1] = [6+2i+i, 8-2i+i] = [6+3i, 8-i]
        hemv(
            HemvUplo::Upper,
            c(2.0, 0.0),
            a.as_ref(),
            &x,
            c(0.0, 1.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(6.0, 3.0)));
        assert!(approx_eq_c(y[1], c(8.0, -1.0)));
    }

    #[test]
    fn test_hemv_identity() {
        let a = Mat::from_rows(&[&[c(1.0, 0.0), c(0.0, 0.0)], &[c(0.0, 0.0), c(1.0, 0.0)]]);
        let x = [c(2.0, 3.0), c(4.0, 5.0)];
        let mut y = [c(0.0, 0.0); 2];

        hemv(
            HemvUplo::Upper,
            c(1.0, 0.0),
            a.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        assert!(approx_eq_c(y[0], c(2.0, 3.0)));
        assert!(approx_eq_c(y[1], c(4.0, 5.0)));
    }

    #[test]
    fn test_hemv_1x1() {
        let a = Mat::from_rows(&[&[c(5.0, 0.0)]]);
        let x = [c(2.0, 1.0)];
        let mut y = [c(0.0, 0.0)];

        hemv(
            HemvUplo::Upper,
            c(1.0, 0.0),
            a.as_ref(),
            &x,
            c(0.0, 0.0),
            &mut y,
        )
        .unwrap();

        // y = 5 * (2+i) = 10 + 5i
        assert!(approx_eq_c(y[0], c(10.0, 5.0)));
    }

    #[test]
    fn test_hemv_dimension_errors() {
        let a = Mat::from_rows(&[&[c(1.0, 0.0), c(0.0, 0.0)], &[c(0.0, 0.0), c(1.0, 0.0)]]);

        let x = [c(1.0, 0.0); 3];
        let mut y = [c(0.0, 0.0); 2];
        assert_eq!(
            hemv(
                HemvUplo::Upper,
                c(1.0, 0.0),
                a.as_ref(),
                &x,
                c(0.0, 0.0),
                &mut y
            ),
            Err(HemvError::DimensionMismatchX)
        );

        let x = [c(1.0, 0.0); 2];
        let mut y = [c(0.0, 0.0); 3];
        assert_eq!(
            hemv(
                HemvUplo::Upper,
                c(1.0, 0.0),
                a.as_ref(),
                &x,
                c(0.0, 0.0),
                &mut y
            ),
            Err(HemvError::DimensionMismatchY)
        );
    }

    #[test]
    fn test_hemv_new() {
        let a = Mat::from_rows(&[&[c(2.0, 0.0), c(1.0, 1.0)], &[c(0.0, 0.0), c(3.0, 0.0)]]);
        let x = [c(1.0, 0.0), c(1.0, 0.0)];

        let y = hemv_new(HemvUplo::Upper, c(1.0, 0.0), a.as_ref(), &x).unwrap();

        assert!(approx_eq_c(y[0], c(3.0, 1.0)));
        assert!(approx_eq_c(y[1], c(4.0, -1.0)));
    }

    #[test]
    fn test_hemv_3x3() {
        // 3x3 Hermitian matrix
        // A = [[1, 2+i, 3-2i], [2-i, 4, 1+i], [3+2i, 1-i, 5]]
        let a = Mat::from_rows(&[
            &[c(1.0, 0.0), c(2.0, 1.0), c(3.0, -2.0)],
            &[c(0.0, 0.0), c(4.0, 0.0), c(1.0, 1.0)],
            &[c(0.0, 0.0), c(0.0, 0.0), c(5.0, 0.0)],
        ]);
        let x = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
        let mut y = [c(0.0, 0.0); 3];

        hemv(
            HemvUplo::Upper,
            c(1.0, 0.0),
            a.as_ref(),
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
}
