//! HER2K: Hermitian Rank-2K update.
//!
//! Computes C = α·A·B^H + conj(α)·B·A^H + β·C (or the conjugate transposed variant)
//! where C is Hermitian.
//!
//! # Operation
//!
//! - For `Trans::NoTrans`: C = α·A·B^H + conj(α)·B·A^H + β·C where A, B are n×k, C is n×n
//! - For `Trans::ConjTrans`: C = α·A^H·B + conj(α)·B^H·A + β·C where A, B are k×n, C is n×n
//!
//! Only the specified triangle (upper or lower) of C is updated.
//! The diagonal of C is always real.
//!
//! # Note
//!
//! For real types (f32, f64), HER2K behaves identically to SYR2K since
//! conjugation has no effect on real numbers. This implementation uses
//! the optimized GEMM path for real types (f32, f64) via the `GemmKernel` trait.

use crate::level3::gemm::gemm;
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::trsm::{Trans, Uplo};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Error type for HER2K operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Her2kError {
    /// Matrix C is not square.
    NotSquare,
    /// Dimension mismatch.
    DimensionMismatch,
    /// Invalid transpose option (`Trans::Trans` not allowed for HER2K).
    InvalidTrans,
}

impl core::fmt::Display for Her2kError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix C is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
            Self::InvalidTrans => write!(
                f,
                "Invalid transpose option: HER2K only accepts NoTrans or ConjTrans"
            ),
        }
    }
}

impl std::error::Error for Her2kError {}

/// Performs the Hermitian rank-2k update.
///
/// C = α·A·B^H + conj(α)·B·A^H + β·C (when trans = `NoTrans`)
/// C = α·A^H·B + conj(α)·B^H·A + β·C (when trans = `ConjTrans`)
///
/// Only the `uplo` triangle of C is written.
///
/// # Arguments
///
/// * `uplo` - Which triangle of C to update (Upper or Lower)
/// * `trans` - Operation on A and B (`NoTrans` or `ConjTrans`)
/// * `alpha` - Scalar multiplier for A·B^H
/// * `a` - The first input matrix A
/// * `b` - The second input matrix B
/// * `beta` - Scalar multiplier for C (must be real for Hermitian result)
/// * `c` - The Hermitian output matrix C (updated in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::her2k::{her2k, Her2kError};
/// use oxiblas_blas::level3::trsm::{Trans, Uplo};
/// use oxiblas_matrix::Mat;
///
/// // For real types, HER2K is equivalent to SYR2K
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[5.0f64, 6.0],
///     &[7.0, 8.0],
/// ]);
///
/// let mut c = Mat::zeros(2, 2);
/// her2k(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut()).unwrap();
///
/// // C = A·B^H + B·A^H = A·B^T + B·A^T (for real types)
/// assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
/// assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
/// assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
/// ```
pub fn her2k<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) -> Result<(), Her2kError> {
    // Validate transpose option - only NoTrans and ConjTrans are valid for HER2K
    if trans == Trans::Trans {
        return Err(Her2kError::InvalidTrans);
    }

    // Validate C is square
    let n = c.nrows();
    if c.ncols() != n {
        return Err(Her2kError::NotSquare);
    }

    // Determine k and validate A, B dimensions
    let k = match trans {
        Trans::NoTrans => {
            if a.nrows() != n || b.nrows() != n {
                return Err(Her2kError::DimensionMismatch);
            }
            if a.ncols() != b.ncols() {
                return Err(Her2kError::DimensionMismatch);
            }
            a.ncols()
        }
        Trans::ConjTrans => {
            if a.ncols() != n || b.ncols() != n {
                return Err(Her2kError::DimensionMismatch);
            }
            if a.nrows() != b.nrows() {
                return Err(Her2kError::DimensionMismatch);
            }
            a.nrows()
        }
        Trans::Trans => unreachable!(), // Already checked above
    };

    // Handle empty cases
    if n == 0 {
        return Ok(());
    }

    // Use GEMM-based optimization for larger matrices with real types
    // For real types, HER2K is equivalent to SYR2K since conjugation has no effect
    const GEMM_THRESHOLD: usize = 32;
    if n >= GEMM_THRESHOLD && k >= 8 {
        her2k_via_gemm(uplo, trans, alpha, a, b, beta, c, n, k)
    } else {
        her2k_naive(uplo, trans, alpha, a, b, beta, c, n, k)
    }
}

/// GEMM-based HER2K for larger matrices.
///
/// For real types (f32, f64), HER2K is equivalent to SYR2K since conjugation
/// has no effect on real numbers. This uses the same optimization as SYR2K.
fn her2k_via_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), Her2kError> {
    let alpha_conj = alpha.conj();

    match trans {
        Trans::NoTrans => {
            // A is n×k, B is n×k
            // Compute A·B^H + conj(α)/α · B·A^H
            // For real: A·B^T + B·A^T (same as SYR2K)
            // Create B^H (k×n) and A^H (k×n)
            let mut b_h: Mat<T> = Mat::zeros(k, n);
            for i in 0..n {
                for j in 0..k {
                    b_h[(j, i)] = b[(i, j)].conj();
                }
            }
            let mut a_h: Mat<T> = Mat::zeros(k, n);
            for i in 0..n {
                for j in 0..k {
                    a_h[(j, i)] = a[(i, j)].conj();
                }
            }

            // Compute temp = α·A·B^H + conj(α)·B·A^H using two GEMMs
            let mut temp: Mat<T> = Mat::zeros(n, n);
            gemm(alpha, a, b_h.as_ref(), T::zero(), temp.as_mut()); // temp = α·A·B^H
            gemm(alpha_conj, b, a_h.as_ref(), T::one(), temp.as_mut()); // temp += conj(α)·B·A^H

            // Copy triangle with beta scaling
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            c.set(i, j, beta * c[(i, j)] + temp[(i, j)]);
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            c.set(i, j, beta * c[(i, j)] + temp[(i, j)]);
                        }
                    }
                }
            }
        }
        Trans::ConjTrans => {
            // A is k×n, B is k×n
            // Compute A^H·B + conj(α)/α · B^H·A
            // Create A^H (n×k) and B^H (n×k)
            let mut a_h: Mat<T> = Mat::zeros(n, k);
            for i in 0..k {
                for j in 0..n {
                    a_h[(j, i)] = a[(i, j)].conj();
                }
            }
            let mut b_h: Mat<T> = Mat::zeros(n, k);
            for i in 0..k {
                for j in 0..n {
                    b_h[(j, i)] = b[(i, j)].conj();
                }
            }

            // Compute temp = α·A^H·B + conj(α)·B^H·A using two GEMMs
            let mut temp: Mat<T> = Mat::zeros(n, n);
            gemm(alpha, a_h.as_ref(), b, T::zero(), temp.as_mut()); // temp = α·A^H·B
            gemm(alpha_conj, b_h.as_ref(), a, T::one(), temp.as_mut()); // temp += conj(α)·B^H·A

            // Copy triangle with beta scaling
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            c.set(i, j, beta * c[(i, j)] + temp[(i, j)]);
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            c.set(i, j, beta * c[(i, j)] + temp[(i, j)]);
                        }
                    }
                }
            }
        }
        Trans::Trans => unreachable!(),
    }

    Ok(())
}

/// Naive HER2K implementation for small matrices.
fn her2k_naive<T: Field>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), Her2kError> {
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

    // Compute C += alpha * A * B^H + conj(alpha) * B * A^H
    // or C += alpha * A^H * B + conj(alpha) * B^H * A
    let alpha_conj = alpha.conj();

    match trans {
        Trans::NoTrans => {
            // C += alpha * A * B^H + conj(alpha) * B * A^H
            // C[i,j] += alpha * sum_l A[i,l] * conj(B[j,l]) + conj(alpha) * sum_l B[i,l] * conj(A[j,l])
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for l in 0..k {
                            let temp1 = alpha * b[(j, l)].conj();
                            let temp2 = alpha_conj * a[(j, l)].conj();
                            for i in j..n {
                                let val = c[(i, j)] + a[(i, l)] * temp1 + b[(i, l)] * temp2;
                                c.set(i, j, val);
                            }
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for l in 0..k {
                            let temp1 = alpha * b[(j, l)].conj();
                            let temp2 = alpha_conj * a[(j, l)].conj();
                            for i in 0..=j {
                                let val = c[(i, j)] + a[(i, l)] * temp1 + b[(i, l)] * temp2;
                                c.set(i, j, val);
                            }
                        }
                    }
                }
            }
        }
        Trans::ConjTrans => {
            // C += alpha * A^H * B + conj(alpha) * B^H * A
            // C[i,j] += alpha * sum_l conj(A[l,i]) * B[l,j] + conj(alpha) * sum_l conj(B[l,i]) * A[l,j]
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut temp = T::zero();
                            for l in 0..k {
                                temp = temp
                                    + alpha * a[(l, i)].conj() * b[(l, j)]
                                    + alpha_conj * b[(l, i)].conj() * a[(l, j)];
                            }
                            let val = c[(i, j)] + temp;
                            c.set(i, j, val);
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            let mut temp = T::zero();
                            for l in 0..k {
                                temp = temp
                                    + alpha * a[(l, i)].conj() * b[(l, j)]
                                    + alpha_conj * b[(l, i)].conj() * a[(l, j)];
                            }
                            let val = c[(i, j)] + temp;
                            c.set(i, j, val);
                        }
                    }
                }
            }
        }
        Trans::Trans => unreachable!(),
    }

    Ok(())
}

/// Performs Hermitian rank-2k update and returns the result.
///
/// This is a convenience function that allocates a new output matrix.
///
/// # Arguments
///
/// * `uplo` - Which triangle to compute and store
/// * `trans` - Operation on A and B (`NoTrans` or `ConjTrans`)
/// * `alpha` - Scalar multiplier
/// * `a` - The first input matrix A
/// * `b` - The second input matrix B
///
/// # Returns
///
/// A new Hermitian matrix C = α·A·B^H + conj(α)·B·A^H (or the conjugate transposed variant).
/// The result is fully Hermitian (both triangles are filled).
pub fn her2k_new<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, Her2kError> {
    let n = match trans {
        Trans::NoTrans => a.nrows(),
        Trans::ConjTrans => a.ncols(),
        Trans::Trans => return Err(Her2kError::InvalidTrans),
    };

    let mut c = Mat::zeros(n, n);
    her2k(uplo, trans, alpha, a, b, T::zero(), c.as_mut())?;

    // Fill in the other triangle with conjugate values for Hermitian property
    match uplo {
        Uplo::Lower => {
            for j in 0..n {
                for i in 0..j {
                    c[(i, j)] = c[(j, i)].conj();
                }
            }
        }
        Uplo::Upper => {
            for j in 0..n {
                for i in (j + 1)..n {
                    c[(i, j)] = c[(j, i)].conj();
                }
            }
        }
    }

    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    // For real types, HER2K should behave like SYR2K
    #[test]
    fn test_her2k_real_lower_no_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        her2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Same as SYR2K for real types
        assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2k_real_upper_no_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        her2k(
            Uplo::Upper,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2k_real_lower_conj_trans() {
        // For real types, ConjTrans is the same as Trans
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Mat::from_rows(&[&[7.0f64, 8.0, 9.0], &[10.0, 11.0, 12.0]]);

        let mut c = Mat::zeros(3, 3);
        her2k(
            Uplo::Lower,
            Trans::ConjTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // A^H·B[0,0] = 1*7 + 4*10 = 47
        // B^H·A[0,0] = 7*1 + 10*4 = 47
        // Sum[0,0] = 94
        assert!((c[(0, 0)] - 94.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2k_with_alpha() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        her2k(
            Uplo::Lower,
            Trans::NoTrans,
            0.5,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Results should be halved
        assert!((c[(0, 0)] - 17.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 31.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 53.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2k_with_beta() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let mut c = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        her2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            2.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 36.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 108.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2k_new() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let c = her2k_new(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref(), b.as_ref()).unwrap();

        // Should be fully Hermitian (symmetric for real types)
        assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2k_invalid_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        let result = her2k(
            Uplo::Lower,
            Trans::Trans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(Her2kError::InvalidTrans)));
    }

    #[test]
    fn test_her2k_same_matrix() {
        // When A = B, her2k should give 2 * herk
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        her2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // A·A^H + A·A^H = 2 * A·A^H (for real types)
        assert!((c[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_her2k_empty() {
        let a: Mat<f64> = Mat::zeros(0, 3);
        let b: Mat<f64> = Mat::zeros(0, 3);
        let mut c: Mat<f64> = Mat::zeros(0, 0);
        let result = her2k(
            Uplo::Lower,
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
    fn test_her2k_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0], &[9.0, 10.0]]);
        let mut c = Mat::zeros(2, 2);

        let result = her2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(Her2kError::DimensionMismatch)));
    }

    #[test]
    fn test_her2k_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let mut c = Mat::zeros(2, 3);

        let result = her2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(Her2kError::NotSquare)));
    }

    #[test]
    fn test_her2k_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f32, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        her2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0f32,
            a.as_ref(),
            b.as_ref(),
            0.0f32,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 34.0).abs() < 1e-5);
        assert!((c[(1, 0)] - 62.0).abs() < 1e-5);
        assert!((c[(1, 1)] - 106.0).abs() < 1e-5);
    }

    // Test with Complex numbers would go here when Complex is fully implemented
    // For now, we test that real types work correctly with HER2K
}
