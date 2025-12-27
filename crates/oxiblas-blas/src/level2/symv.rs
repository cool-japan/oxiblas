//! Symmetric matrix-vector multiply (SYMV).
//!
//! Computes y = α·A·x + β·y where A is a symmetric matrix.

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Specifies which triangle of the symmetric matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymvUplo {
    /// Upper triangle is stored.
    Upper,
    /// Lower triangle is stored.
    Lower,
}

/// Error type for SYMV operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymvError {
    /// Matrix is not square.
    NotSquare,
    /// Vector x has wrong dimension.
    DimensionMismatchX,
    /// Vector y has wrong dimension.
    DimensionMismatchY,
}

impl core::fmt::Display for SymvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::DimensionMismatchX => write!(f, "Vector x has wrong dimension"),
            Self::DimensionMismatchY => write!(f, "Vector y has wrong dimension"),
        }
    }
}

impl std::error::Error for SymvError {}

/// Symmetric matrix-vector multiply.
///
/// Computes y = α·A·x + β·y where A is a symmetric n×n matrix.
///
/// Only the triangle specified by `uplo` is accessed. The other triangle
/// is inferred from symmetry (A\[i,j\] = A\[j,i\]).
///
/// # Arguments
///
/// * `uplo` - Specifies whether upper or lower triangle is stored
/// * `alpha` - Scalar multiplier for A·x
/// * `a` - Symmetric matrix (only one triangle is read)
/// * `x` - Input vector of length n
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector of length n (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{symv, SymvUplo};
/// use oxiblas_matrix::Mat;
///
/// // Symmetric matrix A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
/// // Store only upper triangle
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0, 1.0],
///     &[0.0, 5.0, 3.0],  // Lower part ignored
///     &[0.0, 0.0, 6.0],
/// ]);
/// let x = [1.0f64, 2.0, 3.0];
/// let mut y = [0.0f64; 3];
///
/// symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y).unwrap();
///
/// // y = A*x = [4*1+2*2+1*3, 2*1+5*2+3*3, 1*1+3*2+6*3] = [11, 21, 25]
/// assert!((y[0] - 11.0).abs() < 1e-10);
/// assert!((y[1] - 21.0).abs() < 1e-10);
/// assert!((y[2] - 25.0).abs() < 1e-10);
/// ```
pub fn symv<T: Field>(
    uplo: SymvUplo,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> Result<(), SymvError> {
    let n = a.nrows();

    // Validate dimensions
    if a.ncols() != n {
        return Err(SymvError::NotSquare);
    }
    if x.len() != n {
        return Err(SymvError::DimensionMismatchX);
    }
    if y.len() != n {
        return Err(SymvError::DimensionMismatchY);
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
    // For symmetric matrix, A[i,j] = A[j,i]
    match uplo {
        SymvUplo::Upper => {
            // Upper triangle stored: access a[i,j] for i <= j
            for i in 0..n {
                let mut temp1 = T::zero();
                let temp2 = alpha * x[i];

                // Diagonal element
                y[i] += alpha * a[(i, i)] * x[i];

                // Off-diagonal elements (j > i)
                for j in (i + 1)..n {
                    let aij = a[(i, j)];
                    temp1 += aij * x[j];
                    y[j] += temp2 * aij;
                }

                y[i] += alpha * temp1;
            }
        }
        SymvUplo::Lower => {
            // Lower triangle stored: access a[i,j] for i >= j
            for i in 0..n {
                let mut temp1 = T::zero();
                let temp2 = alpha * x[i];

                // Diagonal element
                y[i] += alpha * a[(i, i)] * x[i];

                // Off-diagonal elements (j < i)
                for j in 0..i {
                    let aij = a[(i, j)];
                    temp1 += aij * x[j];
                    y[j] += temp2 * aij;
                }

                y[i] += alpha * temp1;
            }
        }
    }

    Ok(())
}

/// New-style symmetric matrix-vector multiply that allocates the result.
///
/// Computes y = α·A·x where A is a symmetric n×n matrix.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{symv_new, SymvUplo};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 2.0, 1.0],
///     &[0.0, 5.0, 3.0],
///     &[0.0, 0.0, 6.0],
/// ]);
/// let x = [1.0f64, 2.0, 3.0];
///
/// let y = symv_new(SymvUplo::Upper, 1.0, a.as_ref(), &x).unwrap();
///
/// assert!((y[0] - 11.0).abs() < 1e-10);
/// ```
pub fn symv_new<T: Field>(
    uplo: SymvUplo,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
) -> Result<Vec<T>, SymvError> {
    let n = a.nrows();
    let mut y = vec![T::zero(); n];
    symv(uplo, alpha, a, x, T::zero(), &mut y)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_symv_upper_basic() {
        // Symmetric matrix A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[0.0, 5.0, 3.0], &[0.0, 0.0, 6.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y).unwrap();

        // y = A*x = [4*1+2*2+1*3, 2*1+5*2+3*3, 1*1+3*2+6*3]
        //         = [4+4+3, 2+10+9, 1+6+18] = [11, 21, 25]
        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 21.0));
        assert!(approx_eq(y[2], 25.0));
    }

    #[test]
    fn test_symv_lower_basic() {
        // Symmetric matrix A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        let a = Mat::from_rows(&[&[4.0f64, 0.0, 0.0], &[2.0, 5.0, 0.0], &[1.0, 3.0, 6.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];

        symv(SymvUplo::Lower, 1.0, a.as_ref(), &x, 0.0, &mut y).unwrap();

        // Same result as upper
        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 21.0));
        assert!(approx_eq(y[2], 25.0));
    }

    #[test]
    fn test_symv_with_alpha_beta() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let x = [1.0f64, 2.0];
        let mut y = [1.0f64, 1.0];

        // y = 2*A*x + 3*y
        // A*x = [2*1+1*2, 1*1+3*2] = [4, 7]
        // y = 2*[4,7] + 3*[1,1] = [8+3, 14+3] = [11, 17]
        symv(SymvUplo::Upper, 2.0, a.as_ref(), &x, 3.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 17.0));
    }

    #[test]
    fn test_symv_identity() {
        // Identity matrix
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let x = [2.0f64, 3.0, 4.0];
        let mut y = [0.0f64; 3];

        symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 2.0));
        assert!(approx_eq(y[1], 3.0));
        assert!(approx_eq(y[2], 4.0));
    }

    #[test]
    fn test_symv_2x2() {
        // 2x2 symmetric
        let a = Mat::from_rows(&[&[3.0f64, 2.0], &[0.0, 5.0]]);
        let x = [1.0f64, 1.0];
        let mut y = [0.0f64; 2];

        symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y).unwrap();

        // y = [3+2, 2+5] = [5, 7]
        assert!(approx_eq(y[0], 5.0));
        assert!(approx_eq(y[1], 7.0));
    }

    #[test]
    fn test_symv_1x1() {
        let a = Mat::from_rows(&[&[5.0f64]]);
        let x = [3.0f64];
        let mut y = [0.0f64];

        symv(SymvUplo::Upper, 2.0, a.as_ref(), &x, 0.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 30.0)); // 2 * 5 * 3
    }

    #[test]
    fn test_symv_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let x: [f64; 0] = [];
        let mut y: [f64; 0] = [];

        let result = symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_symv_dimension_errors() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[0.0, 3.0]]);

        // Wrong x dimension
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 2];
        assert_eq!(
            symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y),
            Err(SymvError::DimensionMismatchX)
        );

        // Wrong y dimension
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 3];
        assert_eq!(
            symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y),
            Err(SymvError::DimensionMismatchY)
        );
    }

    #[test]
    fn test_symv_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0]]);
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 2];

        assert_eq!(
            symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y),
            Err(SymvError::NotSquare)
        );
    }

    #[test]
    fn test_symv_alpha_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[0.0, 3.0]]);
        let x = [1.0f64, 2.0];
        let mut y = [5.0f64, 6.0];

        // y = 0*A*x + 2*y = [10, 12]
        symv(SymvUplo::Upper, 0.0, a.as_ref(), &x, 2.0, &mut y).unwrap();

        assert!(approx_eq(y[0], 10.0));
        assert!(approx_eq(y[1], 12.0));
    }

    #[test]
    fn test_symv_beta_zero() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let x = [1.0f64, 2.0];
        let mut y = [100.0f64, 200.0]; // Should be overwritten

        symv(SymvUplo::Upper, 1.0, a.as_ref(), &x, 0.0, &mut y).unwrap();

        // y = A*x = [2+2, 1+6] = [4, 7]
        assert!(approx_eq(y[0], 4.0));
        assert!(approx_eq(y[1], 7.0));
    }

    #[test]
    fn test_symv_new() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[0.0, 5.0, 3.0], &[0.0, 0.0, 6.0]]);
        let x = [1.0f64, 2.0, 3.0];

        let y = symv_new(SymvUplo::Upper, 1.0, a.as_ref(), &x).unwrap();

        assert!(approx_eq(y[0], 11.0));
        assert!(approx_eq(y[1], 21.0));
        assert!(approx_eq(y[2], 25.0));
    }

    #[test]
    fn test_symv_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 1.0], &[0.0, 3.0]]);
        let x = [1.0f32, 2.0];
        let mut y = [0.0f32; 2];

        symv(SymvUplo::Upper, 1.0f32, a.as_ref(), &x, 0.0f32, &mut y).unwrap();

        assert!((y[0] - 4.0).abs() < 1e-5);
        assert!((y[1] - 7.0).abs() < 1e-5);
    }
}
