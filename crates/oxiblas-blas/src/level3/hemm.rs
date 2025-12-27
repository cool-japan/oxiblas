//! HEMM: Hermitian Matrix-Matrix Multiply.
//!
//! Computes C = α·A·B + β·C (side = Left) or C = α·B·A + β·C (side = Right)
//! where A is Hermitian.
//!
//! # Operation
//!
//! - For `Side::Left`: C = α·A·B + β·C where A is m×m Hermitian, B is m×n, C is m×n
//! - For `Side::Right`: C = α·B·A + β·C where B is m×n, A is n×n Hermitian, C is m×n
//!
//! Only the specified triangle (upper or lower) of A is accessed.
//! The diagonal of A is assumed to be real.
//!
//! # Optimization
//!
//! For larger matrices (n >= 32), uses optimized 3M complex GEMM after expanding the
//! Hermitian matrix to a full dense matrix. This provides significant speedup compared
//! to the naive implementation.

use crate::level3::complex_gemm::{gemm3m_c32, gemm3m_c64};
use crate::level3::trsm::{Side, Uplo};
use num_complex::{Complex32, Complex64};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Error type for HEMM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HemmError {
    /// Matrix A is not square.
    NotSquare,
    /// Dimension mismatch between matrices.
    DimensionMismatch,
}

impl core::fmt::Display for HemmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix A is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between matrices"),
        }
    }
}

impl std::error::Error for HemmError {}

/// Performs Hermitian matrix-matrix multiplication.
///
/// C = α·A·B + β·C (when side = Left)
/// C = α·B·A + β·C (when side = Right)
///
/// where A is Hermitian and only the `uplo` triangle is accessed.
/// For Hermitian matrices: A\[i,j\] = conj(A\[j,i\]).
///
/// # Arguments
///
/// * `side` - Which side A appears on (Left: C = α·A·B + β·C, Right: C = α·B·A + β·C)
/// * `uplo` - Which triangle of A is stored (Upper or Lower)
/// * `alpha` - Scalar multiplier for A·B or B·A
/// * `a` - The Hermitian matrix A
/// * `b` - The general matrix B
/// * `beta` - Scalar multiplier for C
/// * `c` - The output matrix C (updated in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::hemm::{hemm, HemmError};
/// use oxiblas_blas::level3::trsm::{Side, Uplo};
/// use oxiblas_matrix::Mat;
/// use num_complex::Complex64;
///
/// // A = [[2, 1-i], [1+i, 3]] (Hermitian, stored as lower triangle)
/// let a = Mat::from_rows(&[
///     &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
///     &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
/// ]);
///
/// let b = Mat::from_rows(&[
///     &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
///     &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
/// ]);
///
/// let mut c = Mat::zeros(2, 2);
/// hemm(Side::Left, Uplo::Lower, Complex64::new(1.0, 0.0), a.as_ref(), b.as_ref(),
///      Complex64::new(0.0, 0.0), c.as_mut()).unwrap();
/// ```
pub fn hemm<T: Field>(
    side: Side,
    uplo: Uplo,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
) -> Result<(), HemmError> {
    // Validate A is square
    if a.nrows() != a.ncols() {
        return Err(HemmError::NotSquare);
    }

    let m = c.nrows();
    let n = c.ncols();
    let ka = a.nrows();

    // Validate dimensions based on side
    match side {
        Side::Left => {
            // C = α·A·B + β·C: A is m×m, B is m×n, C is m×n
            if a.nrows() != m || b.nrows() != m || b.ncols() != n {
                return Err(HemmError::DimensionMismatch);
            }
        }
        Side::Right => {
            // C = α·B·A + β·C: B is m×n, A is n×n, C is m×n
            if a.nrows() != n || b.nrows() != m || b.ncols() != n {
                return Err(HemmError::DimensionMismatch);
            }
        }
    }

    // Handle empty cases
    if m == 0 || n == 0 {
        return Ok(());
    }

    // Helper to get Hermitian element: A[i,j] = conj(A[j,i]) for off-diagonal
    let get_a = |i: usize, j: usize| -> T {
        match uplo {
            Uplo::Lower => {
                if i >= j {
                    a[(i, j)]
                } else {
                    a[(j, i)].conj()
                }
            }
            Uplo::Upper => {
                if i <= j {
                    a[(i, j)]
                } else {
                    a[(j, i)].conj()
                }
            }
        }
    };

    match side {
        Side::Left => {
            // C = α·A·B + β·C where A is m×m Hermitian
            for j in 0..n {
                for i in 0..m {
                    let mut sum = T::zero();
                    for k in 0..ka {
                        sum += get_a(i, k) * b[(k, j)];
                    }
                    let val = alpha * sum + beta * c[(i, j)];
                    c.set(i, j, val);
                }
            }
        }
        Side::Right => {
            // C = α·B·A + β·C where A is n×n Hermitian
            for j in 0..n {
                for i in 0..m {
                    let mut sum = T::zero();
                    for k in 0..ka {
                        sum += b[(i, k)] * get_a(k, j);
                    }
                    let val = alpha * sum + beta * c[(i, j)];
                    c.set(i, j, val);
                }
            }
        }
    }

    Ok(())
}

/// Performs Hermitian matrix-matrix multiplication and returns the result.
///
/// This is a convenience function that allocates a new output matrix.
///
/// # Arguments
///
/// * `side` - Which side A appears on
/// * `uplo` - Which triangle of A is stored
/// * `alpha` - Scalar multiplier
/// * `a` - The Hermitian matrix A
/// * `b` - The general matrix B
///
/// # Returns
///
/// A new matrix C = α·A·B (side = Left) or C = α·B·A (side = Right).
pub fn hemm_new<T: Field + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, HemmError> {
    let m = b.nrows();
    let n = b.ncols();

    let mut c = Mat::zeros(m, n);
    hemm(side, uplo, alpha, a, b, T::zero(), c.as_mut())?;

    Ok(c)
}

// HEMM threshold for using GEMM optimization
// For matrices >= 64, the 3M GEMM method becomes beneficial despite overhead.
// The overhead is ~3% for extraction/combination, but 3M reduces 4 GEMMs to 3.
const HEMM_GEMM_THRESHOLD: usize = 64;

/// Performs optimized Hermitian matrix-matrix multiplication for Complex64.
///
/// Uses the 3M complex GEMM method for larger matrices (size >= 32).
///
/// C = α·A·B + β·C (when side = Left)
/// C = α·B·A + β·C (when side = Right)
///
/// # Arguments
///
/// * `side` - Which side A appears on
/// * `uplo` - Which triangle of A is stored
/// * `alpha` - Scalar multiplier
/// * `a` - The Hermitian matrix A
/// * `b` - The general matrix B
/// * `beta` - Scalar multiplier for C
/// * `c` - The output matrix C (updated in place)
pub fn hemm_c64(
    side: Side,
    uplo: Uplo,
    alpha: Complex64,
    a: MatRef<'_, Complex64>,
    b: MatRef<'_, Complex64>,
    beta: Complex64,
    c: MatMut<'_, Complex64>,
) -> Result<(), HemmError> {
    // Validate A is square
    if a.nrows() != a.ncols() {
        return Err(HemmError::NotSquare);
    }

    let m = c.nrows();
    let n = c.ncols();
    let ka = a.nrows();

    // Validate dimensions based on side
    match side {
        Side::Left => {
            if a.nrows() != m || b.nrows() != m || b.ncols() != n {
                return Err(HemmError::DimensionMismatch);
            }
        }
        Side::Right => {
            if a.nrows() != n || b.nrows() != m || b.ncols() != n {
                return Err(HemmError::DimensionMismatch);
            }
        }
    }

    if m == 0 || n == 0 {
        return Ok(());
    }

    if ka >= HEMM_GEMM_THRESHOLD {
        // Use optimized complex GEMM (3M method)
        hemm_via_gemm_c64(side, uplo, alpha, a, b, beta, c, ka)
    } else {
        // For small matrices, use generic implementation
        hemm(side, uplo, alpha, a, b, beta, c)
    }
}

/// HEMM via optimized complex GEMM for Complex64.
fn hemm_via_gemm_c64(
    side: Side,
    uplo: Uplo,
    alpha: Complex64,
    a: MatRef<'_, Complex64>,
    b: MatRef<'_, Complex64>,
    beta: Complex64,
    c: MatMut<'_, Complex64>,
    ka: usize,
) -> Result<(), HemmError> {
    // Expand Hermitian matrix to full matrix
    // For Hermitian: A[i,j] = conj(A[j,i]) for off-diagonal elements
    let mut a_full: Mat<Complex64> = Mat::zeros(ka, ka);
    match uplo {
        Uplo::Lower => {
            for i in 0..ka {
                a_full[(i, i)] = a[(i, i)];
                for j in 0..i {
                    let val = a[(i, j)];
                    a_full[(i, j)] = val;
                    a_full[(j, i)] = val.conj();
                }
            }
        }
        Uplo::Upper => {
            for i in 0..ka {
                a_full[(i, i)] = a[(i, i)];
                for j in (i + 1)..ka {
                    let val = a[(i, j)];
                    a_full[(i, j)] = val;
                    a_full[(j, i)] = val.conj();
                }
            }
        }
    }

    // Use optimized complex GEMM (3M method)
    match side {
        Side::Left => gemm3m_c64(alpha, a_full.as_ref(), b, beta, c),
        Side::Right => gemm3m_c64(alpha, b, a_full.as_ref(), beta, c),
    }

    Ok(())
}

/// Performs optimized Hermitian matrix-matrix multiplication for Complex32.
///
/// Uses the 3M complex GEMM method for larger matrices (size >= 32).
///
/// C = α·A·B + β·C (when side = Left)
/// C = α·B·A + β·C (when side = Right)
pub fn hemm_c32(
    side: Side,
    uplo: Uplo,
    alpha: Complex32,
    a: MatRef<'_, Complex32>,
    b: MatRef<'_, Complex32>,
    beta: Complex32,
    c: MatMut<'_, Complex32>,
) -> Result<(), HemmError> {
    // Validate A is square
    if a.nrows() != a.ncols() {
        return Err(HemmError::NotSquare);
    }

    let m = c.nrows();
    let n = c.ncols();
    let ka = a.nrows();

    // Validate dimensions based on side
    match side {
        Side::Left => {
            if a.nrows() != m || b.nrows() != m || b.ncols() != n {
                return Err(HemmError::DimensionMismatch);
            }
        }
        Side::Right => {
            if a.nrows() != n || b.nrows() != m || b.ncols() != n {
                return Err(HemmError::DimensionMismatch);
            }
        }
    }

    if m == 0 || n == 0 {
        return Ok(());
    }

    if ka >= HEMM_GEMM_THRESHOLD {
        // Use optimized complex GEMM (3M method)
        hemm_via_gemm_c32(side, uplo, alpha, a, b, beta, c, ka)
    } else {
        // For small matrices, use generic implementation
        hemm(side, uplo, alpha, a, b, beta, c)
    }
}

/// HEMM via optimized complex GEMM for Complex32.
fn hemm_via_gemm_c32(
    side: Side,
    uplo: Uplo,
    alpha: Complex32,
    a: MatRef<'_, Complex32>,
    b: MatRef<'_, Complex32>,
    beta: Complex32,
    c: MatMut<'_, Complex32>,
    ka: usize,
) -> Result<(), HemmError> {
    // Expand Hermitian matrix to full matrix
    let mut a_full: Mat<Complex32> = Mat::zeros(ka, ka);
    match uplo {
        Uplo::Lower => {
            for i in 0..ka {
                a_full[(i, i)] = a[(i, i)];
                for j in 0..i {
                    let val = a[(i, j)];
                    a_full[(i, j)] = val;
                    a_full[(j, i)] = val.conj();
                }
            }
        }
        Uplo::Upper => {
            for i in 0..ka {
                a_full[(i, i)] = a[(i, i)];
                for j in (i + 1)..ka {
                    let val = a[(i, j)];
                    a_full[(i, j)] = val;
                    a_full[(j, i)] = val.conj();
                }
            }
        }
    }

    // Use optimized complex GEMM (3M method)
    match side {
        Side::Left => gemm3m_c32(alpha, a_full.as_ref(), b, beta, c),
        Side::Right => gemm3m_c32(alpha, b, a_full.as_ref(), beta, c),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn cplx(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn test_hemm_left_lower() {
        // A = [[2, 1-i], [1+i, 3]] (Hermitian, lower triangle stored)
        // A is stored as:
        // [[2, 0], [1+i, 3]]
        let a = Mat::from_rows(&[
            &[cplx(2.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 1.0), cplx(3.0, 0.0)],
        ]);

        // B = [[1, 0], [1, 1]]
        let b = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 0.0), cplx(1.0, 0.0)],
        ]);

        // A·B = [[2*1+(1-i)*1, 2*0+(1-i)*1], [(1+i)*1+3*1, (1+i)*0+3*1]]
        //     = [[2+1-i, 1-i], [1+i+3, 3]]
        //     = [[3-i, 1-i], [4+i, 3]]
        let mut c = Mat::zeros(2, 2);
        hemm(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)].re - 3.0).abs() < 1e-10);
        assert!((c[(0, 0)].im - (-1.0)).abs() < 1e-10);
        assert!((c[(0, 1)].re - 1.0).abs() < 1e-10);
        assert!((c[(0, 1)].im - (-1.0)).abs() < 1e-10);
        assert!((c[(1, 0)].re - 4.0).abs() < 1e-10);
        assert!((c[(1, 0)].im - 1.0).abs() < 1e-10);
        assert!((c[(1, 1)].re - 3.0).abs() < 1e-10);
        assert!((c[(1, 1)].im - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hemm_left_upper() {
        // A = [[2, 1-i], [1+i, 3]] (Hermitian, upper triangle stored)
        // A is stored as:
        // [[2, 1-i], [0, 3]]
        let a = Mat::from_rows(&[
            &[cplx(2.0, 0.0), cplx(1.0, -1.0)],
            &[cplx(0.0, 0.0), cplx(3.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 0.0), cplx(1.0, 0.0)],
        ]);

        let mut c = Mat::zeros(2, 2);
        hemm(
            Side::Left,
            Uplo::Upper,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        )
        .unwrap();

        // Same result as lower triangle test
        assert!((c[(0, 0)].re - 3.0).abs() < 1e-10);
        assert!((c[(0, 0)].im - (-1.0)).abs() < 1e-10);
        assert!((c[(1, 0)].re - 4.0).abs() < 1e-10);
        assert!((c[(1, 0)].im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hemm_real_reduces_to_symm() {
        // For real matrices, HEMM should give the same result as SYMM
        let a = Mat::from_rows(&[
            &[cplx(2.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 0.0), cplx(3.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(2.0, 0.0)],
            &[cplx(3.0, 0.0), cplx(4.0, 0.0)],
        ]);

        let mut c = Mat::zeros(2, 2);
        hemm(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        )
        .unwrap();

        // A = [[2, 1], [1, 3]] (symmetric since it's real)
        // A·B = [[2*1+1*3, 2*2+1*4], [1*1+3*3, 1*2+3*4]] = [[5, 8], [10, 14]]
        assert!((c[(0, 0)].re - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)].re - 8.0).abs() < 1e-10);
        assert!((c[(1, 0)].re - 10.0).abs() < 1e-10);
        assert!((c[(1, 1)].re - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_hemm_identity() {
        // A = I (identity is Hermitian)
        let a = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(0.0, 0.0), cplx(1.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[cplx(1.0, 2.0), cplx(3.0, 4.0)],
            &[cplx(5.0, 6.0), cplx(7.0, 8.0)],
        ]);

        // I·B = B
        let mut c = Mat::zeros(2, 2);
        hemm(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((c[(0, 0)].im - 2.0).abs() < 1e-10);
        assert!((c[(1, 1)].re - 7.0).abs() < 1e-10);
        assert!((c[(1, 1)].im - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_hemm_with_alpha() {
        let a = Mat::from_rows(&[
            &[cplx(2.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 0.0), cplx(3.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(2.0, 0.0)],
            &[cplx(3.0, 0.0), cplx(4.0, 0.0)],
        ]);

        // 2 * A·B = [[10, 16], [20, 28]]
        let mut c = Mat::zeros(2, 2);
        hemm(
            Side::Left,
            Uplo::Lower,
            cplx(2.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)].re - 10.0).abs() < 1e-10);
        assert!((c[(0, 1)].re - 16.0).abs() < 1e-10);
        assert!((c[(1, 0)].re - 20.0).abs() < 1e-10);
        assert!((c[(1, 1)].re - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_hemm_not_square() {
        let a = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)],
            &[cplx(4.0, 0.0), cplx(5.0, 0.0), cplx(6.0, 0.0)],
        ]);
        let b = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(2.0, 0.0)],
            &[cplx(3.0, 0.0), cplx(4.0, 0.0)],
        ]);
        let mut c = Mat::zeros(2, 2);

        let result = hemm(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        );
        assert!(matches!(result, Err(HemmError::NotSquare)));
    }

    #[test]
    fn test_hemm_empty() {
        let a: Mat<Complex64> = Mat::zeros(0, 0);
        let b: Mat<Complex64> = Mat::zeros(0, 0);
        let mut c: Mat<Complex64> = Mat::zeros(0, 0);

        let result = hemm(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_hemm_new() {
        let a = Mat::from_rows(&[
            &[cplx(2.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 0.0), cplx(3.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(2.0, 0.0)],
            &[cplx(3.0, 0.0), cplx(4.0, 0.0)],
        ]);

        let c = hemm_new(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((c[(0, 0)].re - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)].re - 8.0).abs() < 1e-10);
        assert!((c[(1, 0)].re - 10.0).abs() < 1e-10);
        assert!((c[(1, 1)].re - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_hemm_c64_basic() {
        // Test optimized hemm_c64 with small matrix (uses generic path)
        let a = Mat::from_rows(&[
            &[cplx(2.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 1.0), cplx(3.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[cplx(1.0, 0.0), cplx(0.0, 0.0)],
            &[cplx(1.0, 0.0), cplx(1.0, 0.0)],
        ]);

        let mut c = Mat::zeros(2, 2);
        hemm_c64(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c.as_mut(),
        )
        .unwrap();

        // Same result as test_hemm_left_lower
        assert!((c[(0, 0)].re - 3.0).abs() < 1e-10);
        assert!((c[(0, 0)].im - (-1.0)).abs() < 1e-10);
        assert!((c[(1, 0)].re - 4.0).abs() < 1e-10);
        assert!((c[(1, 0)].im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hemm_c64_large() {
        // Test optimized hemm_c64 with larger matrix (uses GEMM path)
        let n = 64;
        let mut a: Mat<Complex64> = Mat::zeros(n, n);
        let mut b: Mat<Complex64> = Mat::zeros(n, n);

        // Create Hermitian matrix A (lower triangle)
        for i in 0..n {
            a[(i, i)] = cplx((i + 1) as f64, 0.0); // Real diagonal
            for j in 0..i {
                a[(i, j)] = cplx(0.01 * (i + j) as f64, 0.001 * (i * j) as f64);
            }
        }

        // Create general matrix B
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = cplx(0.01 * (i + j + 1) as f64, 0.0);
            }
        }

        // Compute with hemm_c64 (uses optimized GEMM path for n >= 32)
        let mut c_opt: Mat<Complex64> = Mat::zeros(n, n);
        hemm_c64(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c_opt.as_mut(),
        )
        .unwrap();

        // Compute with generic hemm for comparison
        let mut c_ref: Mat<Complex64> = Mat::zeros(n, n);
        hemm(
            Side::Left,
            Uplo::Lower,
            cplx(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            cplx(0.0, 0.0),
            c_ref.as_mut(),
        )
        .unwrap();

        // Compare results
        for i in 0..n {
            for j in 0..n {
                let diff_re = (c_opt[(i, j)].re - c_ref[(i, j)].re).abs();
                let diff_im = (c_opt[(i, j)].im - c_ref[(i, j)].im).abs();
                assert!(
                    diff_re < 1e-8,
                    "Mismatch at ({},{}) re: opt={}, ref={}",
                    i,
                    j,
                    c_opt[(i, j)].re,
                    c_ref[(i, j)].re
                );
                assert!(
                    diff_im < 1e-8,
                    "Mismatch at ({},{}) im: opt={}, ref={}",
                    i,
                    j,
                    c_opt[(i, j)].im,
                    c_ref[(i, j)].im
                );
            }
        }
    }

    #[test]
    fn test_hemm_c32_basic() {
        use num_complex::Complex32;

        fn c32(re: f32, im: f32) -> Complex32 {
            Complex32::new(re, im)
        }

        // Test optimized hemm_c32 with small matrix
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[c32(2.0, 0.0), c32(0.0, 0.0)],
            &[c32(1.0, 1.0), c32(3.0, 0.0)],
        ]);

        let b: Mat<Complex32> = Mat::from_rows(&[
            &[c32(1.0, 0.0), c32(0.0, 0.0)],
            &[c32(1.0, 0.0), c32(1.0, 0.0)],
        ]);

        let mut c: Mat<Complex32> = Mat::zeros(2, 2);
        hemm_c32(
            Side::Left,
            Uplo::Lower,
            c32(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            c32(0.0, 0.0),
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)].re - 3.0).abs() < 1e-5);
        assert!((c[(0, 0)].im - (-1.0)).abs() < 1e-5);
        assert!((c[(1, 0)].re - 4.0).abs() < 1e-5);
        assert!((c[(1, 0)].im - 1.0).abs() < 1e-5);
    }
}
