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
//! When the order of A is at least 32 (`m` for `Side::Left`, `n` for
//! `Side::Right`; the other dimension of C does not matter), uses optimized GEMM
//! after expanding the symmetric matrix to a full dense matrix. This provides
//! significant speedup compared to the naive implementation.

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

/// Applies the "scale only" update `C := beta * C` that reference ?SYMM / ?HEMM
/// perform when `alpha == 0`: `beta == 0` zero-fills C without reading it and
/// `beta == 1` leaves it untouched. A and B are never read on this path, so NaN
/// or Inf in them cannot leak into C.
pub(crate) fn scale_c_by_beta<T: Field>(c: &mut MatMut<'_, T>, beta: T) {
    if beta == T::zero() {
        c.fill_zero();
    } else if beta != T::one() {
        c.scale(beta);
    }
}

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
    mut c: MatMut<'_, T>,
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

    // Reference ?SYMM: with alpha == 0, A and B are never read and C := beta * C
    // (beta == 0 zero-fills C without reading it; beta == 1 leaves it untouched).
    if alpha == T::zero() {
        scale_c_by_beta(&mut c, beta);
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

    // beta == 0 must NOT read C: an uninitialised C may hold NaN/Inf and `0 * NaN`
    // would poison the result (reference ?SYMM overwrites C in that case).
    let beta_is_zero = beta == T::zero();

    match side {
        Side::Left => {
            // C = α·A·B + β·C where A is m×m symmetric
            for j in 0..n {
                for i in 0..m {
                    let mut sum = T::zero();
                    for k in 0..ka {
                        sum += get_a(i, k) * b[(k, j)];
                    }
                    let val = if beta_is_zero {
                        alpha * sum
                    } else {
                        alpha * sum + beta * c[(i, j)]
                    };
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
                    let val = if beta_is_zero {
                        alpha * sum
                    } else {
                        alpha * sum + beta * c[(i, j)]
                    };
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

    // Issue #9 class (complex symmetric, not reachable through the real-only cblas
    // wrappers): alpha == 0 never reads A / B and beta == 0 never reads C, on both the
    // naive (order 3 / 4) and the GEMM-routed (order >= 32) paths.
    #[test]
    fn test_issue_9_symm_complex_alpha_beta_zero_semantics() {
        use num_complex::Complex64;
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let inf = Complex64::new(f64::INFINITY, f64::NEG_INFINITY);
        let zero = Complex64::new(0.0, 0.0);
        let sym = |i: usize, j: usize| {
            Complex64::new(
                ((3 * i.min(j) + 5 * i.max(j)) % 7) as f64 * 0.125 + if i == j { 1.0 } else { 0.0 },
                0.0625 * ((i + j) % 3) as f64,
            )
        };
        let b_f = |i: usize, j: usize| {
            Complex64::new(
                0.25 * ((i + 2 * j) % 5) as f64 - 0.5,
                0.125 * ((3 * i + j) % 4) as f64,
            )
        };
        let c_f = |i: usize, j: usize| Complex64::new(0.125 * ((2 * i + j) % 7) as f64, 0.25);
        // (alpha, beta, A/B poisoned, C poisoned)
        let scenarios = [
            (0.0, 0.0, true, true),
            (0.0, 2.0, true, false),
            (0.0, 1.0, true, false),
            (2.0, 0.0, false, true),
            (2.0, 0.5, false, false),
        ];
        for (m, n) in [(3usize, 4usize), (40, 38)] {
            for side in [Side::Left, Side::Right] {
                for uplo in [Uplo::Lower, Uplo::Upper] {
                    for (alpha, beta, poison_ab, poison_c) in scenarios {
                        let ka = if side == Side::Left { m } else { n };
                        let mut a: Mat<Complex64> = Mat::zeros(ka, ka);
                        for j in 0..ka {
                            for i in 0..ka {
                                let in_tri = match uplo {
                                    Uplo::Lower => i >= j,
                                    Uplo::Upper => i <= j,
                                };
                                a[(i, j)] = if poison_ab {
                                    nan
                                } else if in_tri {
                                    sym(i, j)
                                } else {
                                    Complex64::new(100.0, 100.0)
                                };
                            }
                        }
                        let mut b: Mat<Complex64> = Mat::zeros(m, n);
                        let mut c: Mat<Complex64> = Mat::zeros(m, n);
                        for j in 0..n {
                            for i in 0..m {
                                b[(i, j)] = if poison_ab { inf } else { b_f(i, j) };
                                c[(i, j)] = if poison_c { nan } else { c_f(i, j) };
                            }
                        }
                        symm(
                            side,
                            uplo,
                            Complex64::new(alpha, 0.0),
                            a.as_ref(),
                            b.as_ref(),
                            Complex64::new(beta, 0.0),
                            c.as_mut(),
                        )
                        .unwrap();
                        for j in 0..n {
                            for i in 0..m {
                                let prod = if alpha == 0.0 {
                                    zero
                                } else if side == Side::Left {
                                    (0..m).map(|l| sym(i, l) * b_f(l, j)).sum::<Complex64>()
                                } else {
                                    (0..n).map(|l| b_f(i, l) * sym(l, j)).sum::<Complex64>()
                                };
                                let c_term = if beta == 0.0 { zero } else { c_f(i, j) * beta };
                                let want = prod * alpha + c_term;
                                let got = c[(i, j)];
                                assert!(
                                    (got - want).norm() <= 1e-10 * (1.0 + want.norm()),
                                    "symm {side:?}/{uplo:?} {m}x{n} alpha={alpha} beta={beta}: \
                                     C[{i},{j}] = {got:?}, expected {want:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
