//! SYMM: Symmetric Matrix-Matrix Multiply.
//!
//! Computes C = α·A·B + β·C (side = Left) or C = α·B·A + β·C (side = Right)
//! where A is symmetric.
//!
//! # Operation
//!
//! - For `Side::Left`: C = α·A·B + β·C where A is m×m symmetric, B is m×n, C is m×n
//! - For `Side::Right`: C = α·B·A + β·C where B is m×n, A is n×n symmetric, C is m×n
//!
//! Only the specified triangle (upper or lower) of A is accessed.
//!
//! # Optimization
//!
//! For larger matrices (n >= 32), uses optimized GEMM after expanding the symmetric
//! matrix to a full dense matrix. This provides significant speedup compared to
//! the naive implementation.

use crate::level3::gemm::gemm;
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::trsm::{Side, Uplo};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Error type for SYMM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmError {
    /// Matrix A is not square.
    NotSquare,
    /// Dimension mismatch between matrices.
    DimensionMismatch,
}

impl core::fmt::Display for SymmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix A is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between matrices"),
        }
    }
}

impl std::error::Error for SymmError {}

/// Performs symmetric matrix-matrix multiplication.
///
/// C = α·A·B + β·C (when side = Left)
/// C = α·B·A + β·C (when side = Right)
///
/// where A is symmetric and only the `uplo` triangle is accessed.
///
/// # Arguments
///
/// * `side` - Which side A appears on (Left: C = α·A·B + β·C, Right: C = α·B·A + β·C)
/// * `uplo` - Which triangle of A is stored (Upper or Lower)
/// * `alpha` - Scalar multiplier for A·B or B·A
/// * `a` - The symmetric matrix A
/// * `b` - The general matrix B
/// * `beta` - Scalar multiplier for C
/// * `c` - The output matrix C (updated in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::symm::{symm, SymmError};
/// use oxiblas_blas::level3::trsm::{Side, Uplo};
/// use oxiblas_matrix::Mat;
///
/// // A = [[2, 1], [1, 3]] (2×2 symmetric matrix, lower triangle stored)
/// let a = Mat::from_rows(&[
///     &[2.0f64, 0.0],
///     &[1.0, 3.0],
/// ]);
///
/// // B = [[1, 2], [3, 4]] (2×2 general matrix)
/// let b = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// // C = A·B
/// let mut c = Mat::zeros(2, 2);
/// symm(Side::Left, Uplo::Lower, 1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut()).unwrap();
///
/// // A·B = [[2*1+1*3, 2*2+1*4], [1*1+3*3, 1*2+3*4]] = [[5, 8], [10, 14]]
/// assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
/// assert!((c[(0, 1)] - 8.0).abs() < 1e-10);
/// assert!((c[(1, 0)] - 10.0).abs() < 1e-10);
/// assert!((c[(1, 1)] - 14.0).abs() < 1e-10);
/// ```
pub fn symm<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) -> Result<(), SymmError> {
    // Validate A is square
    if a.nrows() != a.ncols() {
        return Err(SymmError::NotSquare);
    }

    let m = c.nrows();
    let n = c.ncols();
    let ka = a.nrows();

    // Validate dimensions based on side
    match side {
        Side::Left => {
            // C = α·A·B + β·C: A is m×m, B is m×n, C is m×n
            if a.nrows() != m || b.nrows() != m || b.ncols() != n {
                return Err(SymmError::DimensionMismatch);
            }
        }
        Side::Right => {
            // C = α·B·A + β·C: B is m×n, A is n×n, C is m×n
            if a.nrows() != n || b.nrows() != m || b.ncols() != n {
                return Err(SymmError::DimensionMismatch);
            }
        }
    }

    // Handle empty cases
    if m == 0 || n == 0 {
        return Ok(());
    }

    // Use GEMM-based optimization for larger matrices
    const GEMM_THRESHOLD: usize = 32;
    if ka >= GEMM_THRESHOLD {
        symm_via_gemm(side, uplo, alpha, a, b, beta, c, m, n, ka)
    } else {
        symm_naive(side, uplo, alpha, a, b, beta, c, m, n, ka)
    }
}

/// GEMM-based SYMM for larger matrices.
///
/// Expands the symmetric matrix A to a full dense matrix and uses optimized GEMM.
fn symm_via_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    _m: usize,
    _n: usize,
    ka: usize,
) -> Result<(), SymmError> {
    // Expand symmetric matrix A to full dense matrix
    let mut a_full: Mat<T> = Mat::zeros(ka, ka);
    match uplo {
        Uplo::Lower => {
            // Lower triangle stored: copy lower and mirror to upper
            for i in 0..ka {
                for j in 0..=i {
                    let val = a[(i, j)];
                    a_full[(i, j)] = val;
                    a_full[(j, i)] = val;
                }
            }
        }
        Uplo::Upper => {
            // Upper triangle stored: copy upper and mirror to lower
            for i in 0..ka {
                for j in i..ka {
                    let val = a[(i, j)];
                    a_full[(i, j)] = val;
                    a_full[(j, i)] = val;
                }
            }
        }
    }

    // Use optimized GEMM: C = alpha * A_full * B + beta * C
    match side {
        Side::Left => {
            // C = α·A·B + β·C: A is m×m, B is m×n
            gemm(alpha, a_full.as_ref(), b, beta, c);
        }
        Side::Right => {
            // C = α·B·A + β·C: B is m×n, A is n×n
            gemm(alpha, b, a_full.as_ref(), beta, c);
        }
    }

    Ok(())
}

/// Naive SYMM implementation for small matrices.
fn symm_naive<T: Field>(
    side: Side,
    uplo: Uplo,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    m: usize,
    n: usize,
    ka: usize,
) -> Result<(), SymmError> {
    // Helper to get symmetric element
    let get_a = |i: usize, j: usize| -> T {
        match uplo {
            Uplo::Lower => {
                if i >= j {
                    a[(i, j)]
                } else {
                    a[(j, i)]
                }
            }
            Uplo::Upper => {
                if i <= j {
                    a[(i, j)]
                } else {
                    a[(j, i)]
                }
            }
        }
    };

    match side {
        Side::Left => {
            // C = α·A·B + β·C where A is m×m symmetric
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
            // C = α·B·A + β·C where A is n×n symmetric
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

/// Performs symmetric matrix-matrix multiplication and returns the result.
///
/// This is a convenience function that allocates a new output matrix.
///
/// # Arguments
///
/// * `side` - Which side A appears on
/// * `uplo` - Which triangle of A is stored
/// * `alpha` - Scalar multiplier
/// * `a` - The symmetric matrix A
/// * `b` - The general matrix B
///
/// # Returns
///
/// A new matrix C = α·A·B (side = Left) or C = α·B·A (side = Right).
pub fn symm_new<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, SymmError> {
    let m = b.nrows();
    let n = b.ncols();

    let mut c = Mat::zeros(m, n);
    symm(side, uplo, alpha, a, b, T::zero(), c.as_mut())?;

    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symm_left_lower() {
        // A = [[2, 1], [1, 3]] (symmetric, stored as lower triangle)
        let a = Mat::from_rows(&[
            &[2.0f64, 0.0], // upper part ignored
            &[1.0, 3.0],
        ]);

        // B = [[1, 2], [3, 4]]
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // C = A·B = [[2*1+1*3, 2*2+1*4], [1*1+3*3, 1*2+3*4]] = [[5, 8], [10, 14]]
        let mut c = Mat::zeros(2, 2);
        symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 8.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_left_upper() {
        // A = [[2, 1], [1, 3]] (symmetric, stored as upper triangle)
        let a = Mat::from_rows(&[
            &[2.0f64, 1.0],
            &[0.0, 3.0], // lower part ignored
        ]);

        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        symm(
            Side::Left,
            Uplo::Upper,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 8.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_right_lower() {
        // A = [[2, 1], [1, 3]] (symmetric)
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);

        // B = [[1, 2], [3, 4]]
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // C = B·A = [[1*2+2*1, 1*1+2*3], [3*2+4*1, 3*1+4*3]] = [[4, 7], [10, 15]]
        let mut c = Mat::zeros(2, 2);
        symm(
            Side::Right,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 7.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_right_upper() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);

        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        symm(
            Side::Right,
            Uplo::Upper,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 7.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_with_alpha() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);

        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // 2 * A·B = [[10, 16], [20, 28]]
        let mut c = Mat::zeros(2, 2);
        symm(
            Side::Left,
            Uplo::Lower,
            2.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 16.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 20.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_with_beta() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);

        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // C starts as [[1, 1], [1, 1]]
        // Result = A·B + 2*C = [[5, 8], [10, 14]] + [[2, 2], [2, 2]] = [[7, 10], [12, 16]]
        let mut c = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 1.0]]);
        symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            2.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 7.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 12.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_identity() {
        // A = I (identity is symmetric)
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // I·B = B
        let mut c = Mat::zeros(2, 2);
        symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]); // 2×3, not square
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut c = Mat::zeros(2, 2);

        let result = symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(SymmError::NotSquare)));
    }

    #[test]
    fn test_symm_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 3.0]]); // 2×2
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]); // 3×2
        let mut c = Mat::zeros(3, 2);

        let result = symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(SymmError::DimensionMismatch)));
    }

    #[test]
    fn test_symm_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let b: Mat<f64> = Mat::zeros(0, 0);
        let mut c: Mat<f64> = Mat::zeros(0, 0);

        let result = symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_symm_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        symm(
            Side::Left,
            Uplo::Lower,
            1.0f32,
            a.as_ref(),
            b.as_ref(),
            0.0f32,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 5.0).abs() < 1e-5);
        assert!((c[(0, 1)] - 8.0).abs() < 1e-5);
        assert!((c[(1, 0)] - 10.0).abs() < 1e-5);
        assert!((c[(1, 1)] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_symm_3x3() {
        // A = [[1, 2, 3], [2, 4, 5], [3, 5, 6]] (symmetric)
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[2.0, 4.0, 0.0], &[3.0, 5.0, 6.0]]);

        // B = [[1], [1], [1]]
        let b = Mat::from_rows(&[&[1.0f64], &[1.0], &[1.0]]);

        // A·B = [[1+2+3], [2+4+5], [3+5+6]] = [[6], [11], [14]]
        let mut c = Mat::zeros(3, 1);
        symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 6.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_new() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let c = symm_new(Side::Left, Uplo::Lower, 1.0, a.as_ref(), b.as_ref()).unwrap();

        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 8.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_symm_large_gemm_path() {
        // Test with n >= 32 to exercise the GEMM-based optimization path
        let n = 64;
        let m = 48;

        // Create symmetric matrix A (n×n) - stored as lower triangle
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0; // Diagonal dominance for stability
            for j in 0..i {
                let val = 0.1 / ((i + j + 1) as f64);
                a[(i, j)] = val;
            }
        }

        // Create general matrix B (n×m)
        let mut b: Mat<f64> = Mat::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                b[(i, j)] = (i + j) as f64 * 0.01;
            }
        }

        // Compute C = A·B using optimized SYMM
        let mut c: Mat<f64> = Mat::zeros(n, m);
        symm(
            Side::Left,
            Uplo::Lower,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Verify by computing expected result manually for a few elements
        // C[0,0] = A[0,:] · B[:,0] = A[0,0] * B[0,0] + A[0,1] * B[1,0] + ...
        let mut expected_00 = 0.0;
        for k in 0..n {
            let a_val = if k == 0 { a[(0, 0)] } else { a[(k, 0)] }; // A is symmetric
            expected_00 += a_val * b[(k, 0)];
        }
        assert!(
            (c[(0, 0)] - expected_00).abs() < 1e-8,
            "c[(0,0)] = {}, expected = {}",
            c[(0, 0)],
            expected_00
        );

        // Verify another element
        let mut expected_10 = 0.0;
        for k in 0..n {
            let a_val = if k <= 1 { a[(1, k)] } else { a[(k, 1)] }; // A is symmetric
            expected_10 += a_val * b[(k, 0)];
        }
        assert!(
            (c[(1, 0)] - expected_10).abs() < 1e-8,
            "c[(1,0)] = {}, expected = {}",
            c[(1, 0)],
            expected_10
        );
    }
}
