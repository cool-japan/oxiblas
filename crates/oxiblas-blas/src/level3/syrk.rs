//! SYRK: Symmetric Rank-K update.
//!
//! Computes C = α·A·A^T + β·C (or C = α·A^T·A + β·C) where C is symmetric.
//!
//! # Operation
//!
//! - For `Trans::NoTrans`: C = α·A·A^T + β·C where A is n×k, C is n×n
//! - For `Trans::Trans`: C = α·A^T·A + β·C where A is k×n, C is n×n
//!
//! Only the specified triangle (upper or lower) of C is updated.

use crate::level3::gemm::gemm;
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::trsm::{Trans, Uplo};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Error type for SYRK operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyrkError {
    /// Matrix C is not square.
    NotSquare,
    /// Dimension mismatch.
    DimensionMismatch,
}

impl core::fmt::Display for SyrkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix C is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for SyrkError {}

/// Performs the symmetric rank-k update.
///
/// C = α·A·A^T + β·C (when trans = `NoTrans`)
/// C = α·A^T·A + β·C (when trans = Trans)
///
/// Only the `uplo` triangle of C is written.
///
/// # Arguments
///
/// * `uplo` - Which triangle of C to update (Upper or Lower)
/// * `trans` - Operation on A (`NoTrans` or Trans)
/// * `alpha` - Scalar multiplier for A·A^T
/// * `a` - The input matrix A
/// * `beta` - Scalar multiplier for C
/// * `c` - The symmetric output matrix C (updated in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::syrk::{syrk, SyrkError};
/// use oxiblas_blas::level3::trsm::{Trans, Uplo};
/// use oxiblas_matrix::Mat;
///
/// // A = [[1, 2], [3, 4], [5, 6]] (3×2 matrix)
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
///     &[5.0, 6.0],
/// ]);
///
/// // C = A·A^T is 3×3
/// let mut c = Mat::zeros(3, 3);
///
/// syrk(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref(), 0.0, c.as_mut()).unwrap();
///
/// // Lower triangle of C = A·A^T:
/// // [1*1+2*2,        ,        ]   [5,   ,   ]
/// // [3*1+4*2, 3*3+4*4,        ] = [11, 25,  ]
/// // [5*1+6*2, 5*3+6*4, 5*5+6*6]   [17, 39, 61]
/// assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
/// assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
/// assert!((c[(2, 2)] - 61.0).abs() < 1e-10);
/// ```
pub fn syrk<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) -> Result<(), SyrkError> {
    // Validate inputs
    let n = c.nrows();
    if c.ncols() != n {
        return Err(SyrkError::NotSquare);
    }

    // Determine k and validate A dimensions
    let k = match trans {
        Trans::NoTrans => {
            if a.nrows() != n {
                return Err(SyrkError::DimensionMismatch);
            }
            a.ncols()
        }
        Trans::Trans | Trans::ConjTrans => {
            if a.ncols() != n {
                return Err(SyrkError::DimensionMismatch);
            }
            a.nrows()
        }
    };

    // Handle empty cases
    if n == 0 {
        return Ok(());
    }

    // For larger matrices, use optimized GEMM
    // Threshold: when GEMM overhead is amortized
    const GEMM_THRESHOLD: usize = 32;

    if n >= GEMM_THRESHOLD && k >= 8 {
        syrk_via_gemm(uplo, trans, alpha, a, beta, c, n, k)
    } else {
        syrk_naive(uplo, trans, alpha, a, beta, c, n, k)
    }
}

/// SYRK via optimized GEMM for larger matrices.
fn syrk_via_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), SyrkError> {
    // Create transposed copy of A
    let a_t: Mat<T> = match trans {
        Trans::NoTrans => {
            // A is n×k, need A^T which is k×n
            let mut t = Mat::zeros(k, n);
            for i in 0..n {
                for j in 0..k {
                    t[(j, i)] = a[(i, j)];
                }
            }
            t
        }
        Trans::Trans | Trans::ConjTrans => {
            // A is k×n, need A^T which is n×k
            let mut t = Mat::zeros(n, k);
            for i in 0..k {
                for j in 0..n {
                    t[(j, i)] = a[(i, j)];
                }
            }
            t
        }
    };

    // Compute full result using GEMM
    // For NoTrans: A (n×k) × A^T (k×n) → temp (n×n)
    // For Trans: A^T (n×k) × A (k×n) → temp (n×n)
    let mut temp: Mat<T> = Mat::zeros(n, n);
    match trans {
        Trans::NoTrans => {
            // C = α·A·A^T: A is n×k, A^T is k×n
            gemm(alpha, a, a_t.as_ref(), T::zero(), temp.as_mut());
        }
        Trans::Trans | Trans::ConjTrans => {
            // C = α·A^T·A: A^T (transposed view) is n×k, A is k×n
            gemm(alpha, a_t.as_ref(), a, T::zero(), temp.as_mut());
        }
    }

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

/// Direct SYRK computation for small matrices.
fn syrk_naive<T: Field>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), SyrkError> {
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

    // Compute C += alpha * A * A^T or C += alpha * A^T * A
    match trans {
        Trans::NoTrans => {
            // C += alpha * A * A^T
            // C[i,j] += alpha * sum_l A[i,l] * A[j,l]
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for l in 0..k {
                            let temp = alpha * a[(j, l)];
                            for i in j..n {
                                let val = c[(i, j)] + temp * a[(i, l)];
                                c.set(i, j, val);
                            }
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for l in 0..k {
                            let temp = alpha * a[(j, l)];
                            for i in 0..=j {
                                let val = c[(i, j)] + temp * a[(i, l)];
                                c.set(i, j, val);
                            }
                        }
                    }
                }
            }
        }
        Trans::Trans | Trans::ConjTrans => {
            // C += alpha * A^T * A
            // C[i,j] += alpha * sum_l A[l,i] * A[l,j]
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut temp = T::zero();
                            for l in 0..k {
                                temp += a[(l, i)] * a[(l, j)];
                            }
                            let val = c[(i, j)] + alpha * temp;
                            c.set(i, j, val);
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            let mut temp = T::zero();
                            for l in 0..k {
                                temp += a[(l, i)] * a[(l, j)];
                            }
                            let val = c[(i, j)] + alpha * temp;
                            c.set(i, j, val);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs symmetric rank-k update and returns the result.
///
/// This is a convenience function that allocates a new output matrix.
///
/// # Arguments
///
/// * `uplo` - Which triangle to compute and store
/// * `trans` - Operation on A
/// * `alpha` - Scalar multiplier for A·A^T
/// * `a` - The input matrix A
///
/// # Returns
///
/// A new symmetric matrix C = α·A·A^T (or α·A^T·A if trans = Trans).
/// The result is fully symmetric (both triangles are filled).
pub fn syrk_new<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
) -> Result<Mat<T>, SyrkError> {
    let n = match trans {
        Trans::NoTrans => a.nrows(),
        Trans::Trans | Trans::ConjTrans => a.ncols(),
    };

    let mut c = Mat::zeros(n, n);
    syrk(uplo, trans, alpha, a, T::zero(), c.as_mut())?;

    // Fill in the other triangle for convenience
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
    fn test_syrk_lower_no_trans() {
        // A = [[1, 2], [3, 4], [5, 6]] (3×2)
        // A·A^T = [[5, 11, 17], [11, 25, 39], [17, 39, 61]]
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Check lower triangle
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 17.0).abs() < 1e-10);
        assert!((c[(2, 1)] - 39.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 61.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_upper_no_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        syrk(
            Uplo::Upper,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Check upper triangle
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 11.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 17.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
        assert!((c[(1, 2)] - 39.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 61.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_lower_trans() {
        // A = [[1, 2, 3], [4, 5, 6]] (2×3)
        // A^T·A = [[17, 22, 27], [22, 29, 36], [27, 36, 45]]
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        syrk(Uplo::Lower, Trans::Trans, 1.0, a.as_ref(), 0.0, c.as_mut()).unwrap();

        // Check lower triangle
        assert!((c[(0, 0)] - 17.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 29.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 27.0).abs() < 1e-10);
        assert!((c[(2, 1)] - 36.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_upper_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        syrk(Uplo::Upper, Trans::Trans, 1.0, a.as_ref(), 0.0, c.as_mut()).unwrap();

        // Check upper triangle
        assert!((c[(0, 0)] - 17.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 27.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 29.0).abs() < 1e-10);
        assert!((c[(1, 2)] - 36.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_with_alpha() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // A·A^T = [[5, 11], [11, 25]]
        // 2 * A·A^T = [[10, 22], [22, 50]]
        let mut c = Mat::zeros(2, 2);
        syrk(
            Uplo::Lower,
            Trans::NoTrans,
            2.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_with_beta() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // C initially = [[1, 2], [3, 4]]
        // A·A^T = [[5, 11], [11, 25]]
        // Result = A·A^T + 2*C = [[5+2, ...], [11+6, 25+8]] = [[7, ...], [17, 33]]
        let mut c = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            2.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 7.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 17.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 33.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_new() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let c = syrk_new(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref()).unwrap();

        // Should be fully symmetric
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_identity_like() {
        // A = I, so A·A^T = I
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let mut c = Mat::zeros(2, 2);
        syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 0.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk_empty() {
        let a: Mat<f64> = Mat::zeros(0, 3);
        let mut c: Mat<f64> = Mat::zeros(0, 0);
        let result = syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_syrk_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut c = Mat::zeros(3, 3); // Wrong size

        let result = syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(SyrkError::DimensionMismatch)));
    }

    #[test]
    fn test_syrk_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut c = Mat::zeros(2, 3); // Not square

        let result = syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(SyrkError::NotSquare)));
    }

    #[test]
    fn test_syrk_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0f32,
            a.as_ref(),
            0.0f32,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 5.0).abs() < 1e-5);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-5);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_syrk_larger() {
        // Test with a larger matrix to ensure correctness
        let n = 10;
        let k = 5;
        let mut a = Mat::<f64>::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                a[(i, j)] = (i * k + j + 1) as f64;
            }
        }

        let mut c = Mat::zeros(n, n);
        syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Verify by computing A·A^T manually for a few elements
        // C[0,0] = sum_l A[0,l]^2 = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55
        assert!((c[(0, 0)] - 55.0).abs() < 1e-10);

        // C[1,0] = sum_l A[1,l] * A[0,l] = 6*1 + 7*2 + 8*3 + 9*4 + 10*5 = 6+14+24+36+50 = 130
        assert!((c[(1, 0)] - 130.0).abs() < 1e-10);
    }
}
