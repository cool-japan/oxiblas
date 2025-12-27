//! HERK: Hermitian Rank-K update.
//!
//! Computes C = α·A·A^H + β·C (or C = α·A^H·A + β·C) where C is Hermitian.
//!
//! # Operation
//!
//! - For `Trans::NoTrans`: C = α·A·A^H + β·C where A is n×k, C is n×n
//! - For `Trans::ConjTrans`: C = α·A^H·A + β·C where A is k×n, C is n×n
//!
//! Only the specified triangle (upper or lower) of C is updated.
//! The diagonal of C is always real.
//!
//! # Note
//!
//! For real types (f32, f64), HERK behaves identically to SYRK since
//! conjugation has no effect on real numbers. This implementation uses
//! the optimized SYRK path for real types (f32, f64) via the `GemmKernel` trait.

use crate::level3::gemm::gemm;
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::trsm::{Trans, Uplo};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Error type for HERK operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HerkError {
    /// Matrix C is not square.
    NotSquare,
    /// Dimension mismatch.
    DimensionMismatch,
    /// Invalid transpose option (`Trans::Trans` not allowed for HERK).
    InvalidTrans,
}

impl core::fmt::Display for HerkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix C is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
            Self::InvalidTrans => write!(
                f,
                "Invalid transpose option: HERK only accepts NoTrans or ConjTrans"
            ),
        }
    }
}

impl std::error::Error for HerkError {}

/// Performs the Hermitian rank-k update.
///
/// C = α·A·A^H + β·C (when trans = `NoTrans`)
/// C = α·A^H·A + β·C (when trans = `ConjTrans`)
///
/// Only the `uplo` triangle of C is written.
///
/// # Arguments
///
/// * `uplo` - Which triangle of C to update (Upper or Lower)
/// * `trans` - Operation on A (`NoTrans` or `ConjTrans`)
/// * `alpha` - Scalar multiplier for A·A^H (must be real for Hermitian result)
/// * `a` - The input matrix A
/// * `beta` - Scalar multiplier for C (must be real for Hermitian result)
/// * `c` - The Hermitian output matrix C (updated in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::herk::{herk, HerkError};
/// use oxiblas_blas::level3::trsm::{Trans, Uplo};
/// use oxiblas_matrix::Mat;
///
/// // For real types, HERK is equivalent to SYRK
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let mut c = Mat::zeros(2, 2);
/// herk(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref(), 0.0, c.as_mut()).unwrap();
///
/// // C = A·A^H = A·A^T (for real types)
/// assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
/// assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
/// assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
/// ```
pub fn herk<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) -> Result<(), HerkError> {
    // Validate transpose option - only NoTrans and ConjTrans are valid for HERK
    if trans == Trans::Trans {
        return Err(HerkError::InvalidTrans);
    }

    // Validate C is square
    let n = c.nrows();
    if c.ncols() != n {
        return Err(HerkError::NotSquare);
    }

    // Determine k and validate A dimensions
    let k = match trans {
        Trans::NoTrans => {
            if a.nrows() != n {
                return Err(HerkError::DimensionMismatch);
            }
            a.ncols()
        }
        Trans::ConjTrans => {
            if a.ncols() != n {
                return Err(HerkError::DimensionMismatch);
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
    // For real types, HERK is equivalent to SYRK since conjugation has no effect
    const GEMM_THRESHOLD: usize = 32;
    if n >= GEMM_THRESHOLD && k >= 8 {
        herk_via_gemm(uplo, trans, alpha, a, beta, c, n, k)
    } else {
        herk_naive(uplo, trans, alpha, a, beta, c, n, k)
    }
}

/// GEMM-based HERK for larger matrices.
///
/// For real types (f32, f64), HERK is equivalent to SYRK since conjugation
/// has no effect on real numbers. This uses the same optimization as SYRK.
fn herk_via_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), HerkError> {
    // Create transposed/conjugate-transposed copy based on trans mode
    // For real types, A^H = A^T
    match trans {
        Trans::NoTrans => {
            // A is n×k, compute A·A^H
            // For real types: A·A^H = A·A^T
            // Create A^T (k×n)
            let mut a_t: Mat<T> = Mat::zeros(k, n);
            for i in 0..n {
                for j in 0..k {
                    // For real types, conj() is identity; for complex, we conjugate
                    a_t[(j, i)] = a[(i, j)].conj();
                }
            }

            // Compute temp = α·A·A^H via GEMM
            let mut temp: Mat<T> = Mat::zeros(n, n);
            gemm(alpha, a, a_t.as_ref(), T::zero(), temp.as_mut());

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
            // A is k×n, compute A^H·A
            // For real types: A^H·A = A^T·A
            // Create A^H (n×k)
            let mut a_h: Mat<T> = Mat::zeros(n, k);
            for i in 0..k {
                for j in 0..n {
                    a_h[(j, i)] = a[(i, j)].conj();
                }
            }

            // Compute temp = α·A^H·A via GEMM
            let mut temp: Mat<T> = Mat::zeros(n, n);
            gemm(alpha, a_h.as_ref(), a, T::zero(), temp.as_mut());

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

/// Naive HERK implementation for small matrices.
fn herk_naive<T: Field>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), HerkError> {
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

    // Compute C += alpha * A * A^H or C += alpha * A^H * A
    match trans {
        Trans::NoTrans => {
            // C += alpha * A * A^H
            // C[i,j] += alpha * sum_l A[i,l] * conj(A[j,l])
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for l in 0..k {
                            let temp = alpha * a[(j, l)].conj();
                            for i in j..n {
                                let val = c[(i, j)] + a[(i, l)] * temp;
                                c.set(i, j, val);
                            }
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for l in 0..k {
                            let temp = alpha * a[(j, l)].conj();
                            for i in 0..=j {
                                let val = c[(i, j)] + a[(i, l)] * temp;
                                c.set(i, j, val);
                            }
                        }
                    }
                }
            }
        }
        Trans::ConjTrans => {
            // C += alpha * A^H * A
            // C[i,j] += alpha * sum_l conj(A[l,i]) * A[l,j]
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut temp = T::zero();
                            for l in 0..k {
                                temp += a[(l, i)].conj() * a[(l, j)];
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
                                temp += a[(l, i)].conj() * a[(l, j)];
                            }
                            let val = c[(i, j)] + alpha * temp;
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

/// Performs Hermitian rank-k update and returns the result.
///
/// This is a convenience function that allocates a new output matrix.
///
/// # Arguments
///
/// * `uplo` - Which triangle to compute and store
/// * `trans` - Operation on A (`NoTrans` or `ConjTrans`)
/// * `alpha` - Scalar multiplier for A·A^H
/// * `a` - The input matrix A
///
/// # Returns
///
/// A new Hermitian matrix C = α·A·A^H (or α·A^H·A if trans = `ConjTrans`).
/// The result is fully Hermitian (both triangles are filled).
pub fn herk_new<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
) -> Result<Mat<T>, HerkError> {
    let n = match trans {
        Trans::NoTrans => a.nrows(),
        Trans::ConjTrans => a.ncols(),
        Trans::Trans => return Err(HerkError::InvalidTrans),
    };

    let mut c = Mat::zeros(n, n);
    herk(uplo, trans, alpha, a, T::zero(), c.as_mut())?;

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

    // For real types, HERK should behave like SYRK
    #[test]
    fn test_herk_real_lower_no_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        herk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Same as SYRK for real types
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 17.0).abs() < 1e-10);
        assert!((c[(2, 1)] - 39.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 61.0).abs() < 1e-10);
    }

    #[test]
    fn test_herk_real_upper_no_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        herk(
            Uplo::Upper,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 11.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 17.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
        assert!((c[(1, 2)] - 39.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 61.0).abs() < 1e-10);
    }

    #[test]
    fn test_herk_real_lower_conj_trans() {
        // For real types, ConjTrans is the same as Trans
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let mut c = Mat::zeros(3, 3);
        herk(
            Uplo::Lower,
            Trans::ConjTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // A^H·A = A^T·A for real types
        assert!((c[(0, 0)] - 17.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 29.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 27.0).abs() < 1e-10);
        assert!((c[(2, 1)] - 36.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_herk_with_alpha() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        herk(
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
    fn test_herk_with_beta() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        herk(
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
    fn test_herk_new() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let c = herk_new(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref()).unwrap();

        // Should be fully Hermitian (symmetric for real types)
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_herk_invalid_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        let result = herk(Uplo::Lower, Trans::Trans, 1.0, a.as_ref(), 0.0, c.as_mut());
        assert!(matches!(result, Err(HerkError::InvalidTrans)));
    }

    #[test]
    fn test_herk_empty() {
        let a: Mat<f64> = Mat::zeros(0, 3);
        let mut c: Mat<f64> = Mat::zeros(0, 0);
        let result = herk(
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
    fn test_herk_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut c = Mat::zeros(3, 3);

        let result = herk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(HerkError::DimensionMismatch)));
    }

    #[test]
    fn test_herk_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut c = Mat::zeros(2, 3);

        let result = herk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(HerkError::NotSquare)));
    }

    #[test]
    fn test_herk_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        herk(
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
    fn test_herk_larger() {
        let n = 10;
        let k = 5;
        let mut a = Mat::<f64>::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                a[(i, j)] = (i * k + j + 1) as f64;
            }
        }

        let mut c = Mat::zeros(n, n);
        herk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Verify by computing A·A^H manually for a few elements
        assert!((c[(0, 0)] - 55.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 130.0).abs() < 1e-10);
    }

    // Test with Complex numbers would go here when Complex is implemented
    // For now, we test that real types work correctly with HERK
}
