//! GEMMT: General matrix-matrix multiplication with triangular result update.
//!
//! Computes C = α·op(A)·op(B) + β·C where only the specified triangle of C is updated.
//!
//! # Operation
//!
//! - `C = α·A·B + β·C` (when transA = `NoTrans`, transB = `NoTrans`)
//! - `C = α·A^T·B + β·C` (when transA = Trans, transB = `NoTrans`)
//! - `C = α·A·B^T + β·C` (when transA = `NoTrans`, transB = Trans)
//! - `C = α·A^T·B^T + β·C` (when transA = Trans, transB = Trans)
//!
//! Only the specified triangle (upper or lower) of C is computed and stored.
//! This is useful when:
//! - Computing symmetric products (e.g., A·A^T for covariance)
//! - Only half of the result is needed for subsequent operations
//! - Memory bandwidth optimization when result is symmetric
//!
//! # Performance Notes
//!
//! For large matrices, this operation uses optimized GEMM kernels and then
//! extracts the required triangle. For small matrices, a direct computation
//! avoiding unnecessary work is used.

use crate::level3::gemm::gemm;
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::trsm::{Trans, Uplo};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Error type for GEMMT operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmtError {
    /// Matrix C is not square.
    NotSquare,
    /// Dimension mismatch between A, B, and C.
    DimensionMismatch,
}

impl core::fmt::Display for GemmtError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix C must be square"),
            Self::DimensionMismatch => write!(f, "Matrix dimensions are incompatible"),
        }
    }
}

impl std::error::Error for GemmtError {}

/// Performs general matrix-matrix multiplication with triangular result update.
///
/// C = α·op(A)·op(B) + β·C
///
/// Only the `uplo` triangle of C is written.
///
/// # Arguments
///
/// * `uplo` - Which triangle of C to update (Upper or Lower)
/// * `trans_a` - Operation on A (`NoTrans`, Trans, or `ConjTrans`)
/// * `trans_b` - Operation on B (`NoTrans`, Trans, or `ConjTrans`)
/// * `alpha` - Scalar multiplier for op(A)·op(B)
/// * `a` - First input matrix A
/// * `b` - Second input matrix B
/// * `beta` - Scalar multiplier for C
/// * `c` - The output matrix C (updated in place)
///
/// # Dimensions
///
/// After applying transpositions:
/// - op(A) has dimensions n × k
/// - op(B) has dimensions k × n
/// - C has dimensions n × n (must be square)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::gemmt::{gemmt, GemmtError};
/// use oxiblas_blas::level3::trsm::{Trans, Uplo};
/// use oxiblas_matrix::Mat;
///
/// // Compute A·B^T where result is stored only in lower triangle
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
///     &[5.0, 6.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 1.0],
///     &[1.0, 1.0],
/// ]);
///
/// let mut c = Mat::zeros(3, 3);
/// gemmt(Uplo::Lower, Trans::NoTrans, Trans::Trans, 1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut()).unwrap();
///
/// // Lower triangle contains A·B^T
/// assert!((c[(0, 0)] - 1.0).abs() < 1e-10); // 1*1 + 2*0 = 1
/// assert!((c[(1, 0)] - 3.0).abs() < 1e-10); // 3*1 + 4*0 = 3
/// assert!((c[(2, 0)] - 5.0).abs() < 1e-10); // 5*1 + 6*0 = 5
/// ```
pub fn gemmt<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) -> Result<(), GemmtError> {
    // Validate C is square
    let n = c.nrows();
    if c.ncols() != n {
        return Err(GemmtError::NotSquare);
    }

    // Get dimensions after applying transpositions
    let (a_rows, a_cols) = match trans_a {
        Trans::NoTrans => (a.nrows(), a.ncols()),
        Trans::Trans | Trans::ConjTrans => (a.ncols(), a.nrows()),
    };

    let (b_rows, b_cols) = match trans_b {
        Trans::NoTrans => (b.nrows(), b.ncols()),
        Trans::Trans | Trans::ConjTrans => (b.ncols(), b.nrows()),
    };

    // op(A) should be n × k
    // op(B) should be k × n
    if a_rows != n {
        return Err(GemmtError::DimensionMismatch);
    }
    let k = a_cols;
    if b_rows != k || b_cols != n {
        return Err(GemmtError::DimensionMismatch);
    }

    // Handle empty cases
    if n == 0 {
        return Ok(());
    }

    // Choose strategy based on matrix size
    const GEMM_THRESHOLD: usize = 32;

    if n >= GEMM_THRESHOLD && k >= 8 {
        gemmt_via_gemm(uplo, trans_a, trans_b, alpha, a, b, beta, c, n, k)
    } else {
        gemmt_naive(uplo, trans_a, trans_b, alpha, a, b, beta, c, n, k)
    }
}

/// GEMMT via optimized GEMM for larger matrices.
fn gemmt_via_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), GemmtError> {
    // For GEMM, we need the matrices in non-transposed form
    // We'll create temporary matrices if needed

    // Create op(A) explicitly
    let op_a: Mat<T> = match trans_a {
        Trans::NoTrans => {
            let mut t = Mat::zeros(n, k);
            for i in 0..n {
                for j in 0..k {
                    t[(i, j)] = a[(i, j)];
                }
            }
            t
        }
        Trans::Trans => {
            let mut t = Mat::zeros(n, k);
            for i in 0..n {
                for j in 0..k {
                    t[(i, j)] = a[(j, i)];
                }
            }
            t
        }
        Trans::ConjTrans => {
            let mut t = Mat::zeros(n, k);
            for i in 0..n {
                for j in 0..k {
                    t[(i, j)] = a[(j, i)].conj();
                }
            }
            t
        }
    };

    // Create op(B) explicitly
    let op_b: Mat<T> = match trans_b {
        Trans::NoTrans => {
            let mut t = Mat::zeros(k, n);
            for i in 0..k {
                for j in 0..n {
                    t[(i, j)] = b[(i, j)];
                }
            }
            t
        }
        Trans::Trans => {
            let mut t = Mat::zeros(k, n);
            for i in 0..k {
                for j in 0..n {
                    t[(i, j)] = b[(j, i)];
                }
            }
            t
        }
        Trans::ConjTrans => {
            let mut t = Mat::zeros(k, n);
            for i in 0..k {
                for j in 0..n {
                    t[(i, j)] = b[(j, i)].conj();
                }
            }
            t
        }
    };

    // Compute full result using GEMM
    let mut temp: Mat<T> = Mat::zeros(n, n);
    gemm(
        alpha,
        op_a.as_ref(),
        op_b.as_ref(),
        T::zero(),
        temp.as_mut(),
    );

    // Copy triangle to C with beta scaling
    match uplo {
        Uplo::Lower => {
            for j in 0..n {
                for i in j..n {
                    let val = if beta == T::zero() {
                        temp[(i, j)]
                    } else {
                        temp[(i, j)] + beta * c[(i, j)]
                    };
                    c.set(i, j, val);
                }
            }
        }
        Uplo::Upper => {
            for j in 0..n {
                for i in 0..=j {
                    let val = if beta == T::zero() {
                        temp[(i, j)]
                    } else {
                        temp[(i, j)] + beta * c[(i, j)]
                    };
                    c.set(i, j, val);
                }
            }
        }
    }

    Ok(())
}

/// Direct GEMMT computation for small matrices.
///
/// This avoids the overhead of full GEMM and only computes the required triangle.
fn gemmt_naive<T: Field>(
    uplo: Uplo,
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), GemmtError> {
    // Scale C by beta (only the relevant triangle)
    if beta == T::zero() {
        match uplo {
            Uplo::Lower => {
                for j in 0..n {
                    for i in j..n {
                        c.set(i, j, T::zero());
                    }
                }
            }
            Uplo::Upper => {
                for j in 0..n {
                    for i in 0..=j {
                        c.set(i, j, T::zero());
                    }
                }
            }
        }
    } else if beta != T::one() {
        match uplo {
            Uplo::Lower => {
                for j in 0..n {
                    for i in j..n {
                        c.set(i, j, beta * c[(i, j)]);
                    }
                }
            }
            Uplo::Upper => {
                for j in 0..n {
                    for i in 0..=j {
                        c.set(i, j, beta * c[(i, j)]);
                    }
                }
            }
        }
    }

    // Early return if alpha is zero
    if alpha == T::zero() {
        return Ok(());
    }

    // Helper to access op(A)[i, l]
    let get_op_a = |i: usize, l: usize| -> T {
        match trans_a {
            Trans::NoTrans => a[(i, l)],
            Trans::Trans => a[(l, i)],
            Trans::ConjTrans => a[(l, i)].conj(),
        }
    };

    // Helper to access op(B)[l, j]
    let get_op_b = |l: usize, j: usize| -> T {
        match trans_b {
            Trans::NoTrans => b[(l, j)],
            Trans::Trans => b[(j, l)],
            Trans::ConjTrans => b[(j, l)].conj(),
        }
    };

    // Compute C += alpha * op(A) * op(B)
    // C[i,j] += alpha * sum_l op(A)[i,l] * op(B)[l,j]
    match uplo {
        Uplo::Lower => {
            // For lower triangle: i >= j
            for j in 0..n {
                for i in j..n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum += get_op_a(i, l) * get_op_b(l, j);
                    }
                    let val = c[(i, j)] + alpha * sum;
                    c.set(i, j, val);
                }
            }
        }
        Uplo::Upper => {
            // For upper triangle: i <= j
            for j in 0..n {
                for i in 0..=j {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum += get_op_a(i, l) * get_op_b(l, j);
                    }
                    let val = c[(i, j)] + alpha * sum;
                    c.set(i, j, val);
                }
            }
        }
    }

    Ok(())
}

/// Performs GEMMT and returns the result as a new matrix.
///
/// This is a convenience function that allocates a new output matrix.
/// The non-computed triangle is filled with zeros.
///
/// # Arguments
///
/// * `uplo` - Which triangle to compute and store
/// * `trans_a` - Operation on A
/// * `trans_b` - Operation on B
/// * `alpha` - Scalar multiplier
/// * `a` - First input matrix
/// * `b` - Second input matrix
///
/// # Returns
///
/// A new matrix with the result in the specified triangle.
pub fn gemmt_new<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, GemmtError> {
    // Determine n from the matrices
    let n = match trans_a {
        Trans::NoTrans => a.nrows(),
        Trans::Trans | Trans::ConjTrans => a.ncols(),
    };

    let mut c = Mat::zeros(n, n);
    gemmt(uplo, trans_a, trans_b, alpha, a, b, T::zero(), c.as_mut())?;
    Ok(c)
}

/// Performs GEMMT and fills both triangles for convenience (symmetric result).
///
/// This assumes the result should be symmetric and mirrors the computed
/// triangle to the other side.
///
/// # Arguments
///
/// * `uplo` - Which triangle to compute (the other will be mirrored)
/// * `trans_a` - Operation on A
/// * `trans_b` - Operation on B
/// * `alpha` - Scalar multiplier
/// * `a` - First input matrix
/// * `b` - Second input matrix
///
/// # Returns
///
/// A symmetric matrix with the result.
pub fn gemmt_symmetric<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, GemmtError> {
    let n = match trans_a {
        Trans::NoTrans => a.nrows(),
        Trans::Trans | Trans::ConjTrans => a.ncols(),
    };

    let mut c = Mat::zeros(n, n);
    gemmt(uplo, trans_a, trans_b, alpha, a, b, T::zero(), c.as_mut())?;

    // Mirror the computed triangle to the other side
    match uplo {
        Uplo::Lower => {
            for j in 0..n {
                for i in 0..j {
                    c[(i, j)] = c[(j, i)];
                }
            }
        }
        Uplo::Upper => {
            for j in 0..n {
                for i in (j + 1)..n {
                    c[(i, j)] = c[(j, i)];
                }
            }
        }
    }

    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemmt_lower_no_trans() {
        // A = [[1, 2], [3, 4], [5, 6]] (3×2)
        // B = [[1, 2, 3], [4, 5, 6]] (2×3)
        // A·B = [[9, 12, 15], [19, 26, 33], [29, 40, 51]]
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Check lower triangle
        assert!((c[(0, 0)] - 9.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 19.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 26.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 29.0).abs() < 1e-10);
        assert!((c[(2, 1)] - 40.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 51.0).abs() < 1e-10);

        // Upper triangle should be zero (not computed)
        assert!((c[(0, 1)] - 0.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 0.0).abs() < 1e-10);
        assert!((c[(1, 2)] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_upper_no_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        gemmt(
            Uplo::Upper,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Check upper triangle
        assert!((c[(0, 0)] - 9.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 12.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 15.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 26.0).abs() < 1e-10);
        assert!((c[(1, 2)] - 33.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 51.0).abs() < 1e-10);

        // Lower triangle should be zero
        assert!((c[(1, 0)] - 0.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 0.0).abs() < 1e-10);
        assert!((c[(2, 1)] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_trans_a() {
        // A = [[1, 2], [3, 4]] (2×2)
        // A^T = [[1, 3], [2, 4]]
        // B = [[1, 0], [0, 1]] (2×2)
        // A^T·B = [[1, 3], [2, 4]]
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let mut c = Mat::zeros(2, 2);
        gemmt(
            Uplo::Lower,
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_trans_b() {
        // A = [[1, 2], [3, 4]] (2×2)
        // B = [[1, 0], [0, 1]] (2×2)
        // B^T = B (identity)
        // A·B^T = A·B = [[1, 2], [3, 4]]
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let mut c = Mat::zeros(2, 2);
        gemmt(
            Uplo::Upper,
            Trans::NoTrans,
            Trans::Trans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_both_trans() {
        // A = [[1, 2, 3], [4, 5, 6]] (2×3)
        // B = [[1, 4], [2, 5], [3, 6]] (3×2)
        // A^T = [[1, 4], [2, 5], [3, 6]] (3×2)
        // B^T = [[1, 2, 3], [4, 5, 6]] (2×3)
        // A^T·B^T = 3×3 matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 4.0], &[2.0, 5.0], &[3.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        gemmt(
            Uplo::Lower,
            Trans::Trans,
            Trans::Trans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // A^T·B^T[0,0] = 1*1 + 4*4 = 17
        assert!((c[(0, 0)] - 17.0).abs() < 1e-10);
        // A^T·B^T[1,0] = 2*1 + 5*4 = 22
        assert!((c[(1, 0)] - 22.0).abs() < 1e-10);
        // A^T·B^T[2,0] = 3*1 + 6*4 = 27
        assert!((c[(2, 0)] - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_with_alpha() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let mut c = Mat::zeros(2, 2);
        gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            2.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // 2 * A·B = 2 * [[1, 2], [3, 4]] = [[2, 4], [6, 8]]
        assert!((c[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 6.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_with_beta() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let mut c = Mat::from_rows(&[&[10.0f64, 20.0], &[30.0, 40.0]]);
        gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.5,
            c.as_mut(),
        )
        .unwrap();

        // A·B + 0.5*C = [[1, 2], [3, 4]] + 0.5*[[10, 20], [30, 40]]
        // Lower triangle: [[1+5, ...], [3+15, 4+20]] = [[6, ...], [18, 24]]
        assert!((c[(0, 0)] - 6.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 18.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_new() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let c = gemmt_new(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        // A·B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_symmetric() {
        // Compute A·A^T which should be symmetric
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let c = gemmt_symmetric(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::Trans,
            1.0,
            a.as_ref(),
            a.as_ref(),
        )
        .unwrap();

        // A·A^T = [[1*1+2*2, 1*3+2*4], [3*1+4*2, 3*3+4*4]] = [[5, 11], [11, 25]]
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let c = gemmt_new(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        // I·B = B
        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_empty() {
        let a: Mat<f64> = Mat::zeros(0, 3);
        let b: Mat<f64> = Mat::zeros(3, 0);
        let mut c: Mat<f64> = Mat::zeros(0, 0);
        let result = gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gemmt_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let mut c = Mat::zeros(2, 2);

        // B is 2×3, but we need 2×2 for C to be 2×2
        let result = gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(GemmtError::DimensionMismatch)));
    }

    #[test]
    fn test_gemmt_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut c = Mat::zeros(2, 3); // Not square

        let result = gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(GemmtError::NotSquare)));
    }

    #[test]
    fn test_gemmt_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f32, 0.0], &[0.0, 1.0]]);

        let mut c = Mat::zeros(2, 2);
        gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0f32,
            a.as_ref(),
            b.as_ref(),
            0.0f32,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 1.0).abs() < 1e-5);
        assert!((c[(1, 0)] - 3.0).abs() < 1e-5);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemmt_larger() {
        // Test with a larger matrix to trigger GEMM path
        let n = 40;
        let k = 10;
        let mut a = Mat::<f64>::zeros(n, k);
        let mut b = Mat::<f64>::zeros(k, n);

        for i in 0..n {
            for j in 0..k {
                a[(i, j)] = (i * k + j + 1) as f64;
            }
        }
        for i in 0..k {
            for j in 0..n {
                b[(i, j)] = ((i + 1) * (j + 1)) as f64;
            }
        }

        let mut c = Mat::zeros(n, n);
        gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Verify a few elements by manual computation
        // C[0,0] = sum_l A[0,l] * B[l,0] = sum_l (l+1) * (l+1) * 1
        //        = 1*1 + 2*2 + 3*3 + ... + 10*10 = 385
        let expected_00: f64 = (1..=k).map(|l| (l * l) as f64).sum();
        assert!((c[(0, 0)] - expected_00).abs() < 1e-8);
    }

    #[test]
    fn test_gemmt_alpha_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::from_rows(&[&[10.0f64, 20.0], &[30.0, 40.0]]);
        gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            0.0,
            a.as_ref(),
            b.as_ref(),
            1.0,
            c.as_mut(),
        )
        .unwrap();

        // With alpha=0 and beta=1, C should remain unchanged in lower triangle
        assert!((c[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 30.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemmt_beta_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let mut c = Mat::from_rows(&[&[100.0f64, 200.0], &[300.0, 400.0]]);
        gemmt(
            Uplo::Lower,
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // With beta=0, original C values should be ignored
        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
    }
}
