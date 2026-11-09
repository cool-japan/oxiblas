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
//! Whenever C is updated its diagonal is made real. As in reference (Z/C)HERK,
//! `(alpha == 0 || k == 0) && beta == 1` is a quick return that leaves C completely
//! untouched (even a non-real diagonal).
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

/// The two operations HERK performs, decoded once from the public [`Trans`]
/// argument (whose third variant, `Trans::Trans`, HERK rejects), so the
/// private helpers can never be handed the invalid one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HerkOp {
    /// `C = α·A·Aᴴ + β·C` (`Trans::NoTrans`), A is n×k.
    NoTrans,
    /// `C = α·Aᴴ·A + β·C` (`Trans::ConjTrans`), A is k×n.
    ConjTrans,
}

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
/// * `alpha` - Scalar multiplier for A·A^H. Per the reference (Z/C)HERK contract
///   this is a REAL scalar; for a complex `T` any imaginary component is discarded.
/// * `a` - The input matrix A
/// * `beta` - Scalar multiplier for C. Also a REAL scalar per (Z/C)HERK; for a
///   complex `T` any imaginary component is discarded.
/// * `c` - The Hermitian output matrix C (updated in place). Its diagonal is written
///   as a real value whenever C is updated; for `(alpha == 0 || k == 0) && beta == 1`
///   (the reference quick return) C is left completely untouched.
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
    let op = match trans {
        Trans::NoTrans => HerkOp::NoTrans,
        Trans::ConjTrans => HerkOp::ConjTrans,
        Trans::Trans => return Err(HerkError::InvalidTrans),
    };

    // Reference (Z/C)HERK declare BOTH α and β as REAL scalars: the update
    // C = α·A·Aᴴ + β·C is Hermitian only when both scalars are real. This crate is
    // generic over the field and cannot express "real scalar" in the type, so we
    // discard any imaginary component up front, exactly reproducing the reference
    // contract where those parameters simply have no imaginary part. This is a no-op
    // for real element types (`from_real(x.real()) == x`).
    let alpha = T::from_real(alpha.real());
    let beta = T::from_real(beta.real());

    // Validate C is square
    let n = c.nrows();
    if c.ncols() != n {
        return Err(HerkError::NotSquare);
    }

    // Determine k and validate A dimensions
    let k = match op {
        HerkOp::NoTrans => {
            if a.nrows() != n {
                return Err(HerkError::DimensionMismatch);
            }
            a.ncols()
        }
        HerkOp::ConjTrans => {
            if a.ncols() != n {
                return Err(HerkError::DimensionMismatch);
            }
            a.nrows()
        }
    };

    // Handle empty cases
    if n == 0 {
        return Ok(());
    }

    // Reference quick return: for (alpha == 0 || k == 0) && beta == 1 nothing changes and C
    // is left completely untouched (even a non-real diagonal; the real-diagonal projection
    // only happens on the paths that actually update C).
    if (alpha == T::zero() || k == 0) && beta == T::one() {
        return Ok(());
    }

    // Use GEMM-based optimization for larger matrices with real types
    // For real types, HERK is equivalent to SYRK since conjugation has no effect.
    // alpha == 0 never reads A (C := beta * C), so it stays on the naive path.
    const GEMM_THRESHOLD: usize = 32;
    if n >= GEMM_THRESHOLD && k >= 8 && alpha != T::zero() {
        herk_via_gemm(uplo, op, alpha, a, beta, c, n, k)
    } else {
        herk_naive(uplo, op, alpha, a, beta, c, n, k)
    }
}

/// Writes `temp`'s `uplo` triangle into `c` as `β·C + temp`, enforcing the two
/// invariants the reference (Z/C)HERK guarantee:
///
/// * **`β == 0` must not read `C`.** The result is `temp` alone; reading the old
///   `C` and forming `0·C` would let NaN/garbage in an uninitialised `C` buffer
///   poison the output (`0·NaN == NaN`).
/// * **The diagonal is real** (this helper only runs when C is actually updated; the
///   reference quick return never gets here). A Hermitian matrix has a real diagonal, but the
///   GEMM accumulation of `A·Aᴴ` can leave rounding noise in the imaginary part,
///   so every `i == j` entry is projected back onto the reals.
fn write_herk_triangle<T: Field>(
    uplo: Uplo,
    beta: T,
    temp: &Mat<T>,
    c: &mut MatMut<'_, T>,
    n: usize,
) {
    let beta_is_zero = beta == T::zero();
    for j in 0..n {
        // Row range covers only the requested triangle (including the diagonal).
        let i_range = match uplo {
            Uplo::Lower => j..n,
            Uplo::Upper => 0..(j + 1),
        };
        for i in i_range {
            let combined = if beta_is_zero {
                temp[(i, j)]
            } else {
                temp[(i, j)] + beta * c[(i, j)]
            };
            let val = if i == j {
                T::from_real(combined.real())
            } else {
                combined
            };
            c.set(i, j, val);
        }
    }
}

/// GEMM-based HERK for larger matrices.
///
/// For real types (f32, f64), HERK is equivalent to SYRK since conjugation
/// has no effect on real numbers. This uses the same optimization as SYRK.
fn herk_via_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    op: HerkOp,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), HerkError> {
    // Create transposed/conjugate-transposed copy based on trans mode and compute
    // `temp = α·A·Aᴴ` (or `α·Aᴴ·A`). For real types, A^H = A^T.
    let mut temp: Mat<T> = Mat::zeros(n, n);
    match op {
        HerkOp::NoTrans => {
            // A is n×k, compute A·A^H via A^T (k×n).
            let mut a_t: Mat<T> = Mat::zeros(k, n);
            for i in 0..n {
                for j in 0..k {
                    // For real types, conj() is identity; for complex, we conjugate.
                    a_t[(j, i)] = a[(i, j)].conj();
                }
            }
            gemm(alpha, a, a_t.as_ref(), T::zero(), temp.as_mut());
        }
        HerkOp::ConjTrans => {
            // A is k×n, compute A^H·A via A^H (n×k).
            let mut a_h: Mat<T> = Mat::zeros(n, k);
            for i in 0..k {
                for j in 0..n {
                    a_h[(j, i)] = a[(i, j)].conj();
                }
            }
            gemm(alpha, a_h.as_ref(), a, T::zero(), temp.as_mut());
        }
    }

    // Copy triangle with beta scaling, honouring the β==0 / real-diagonal invariants.
    write_herk_triangle(uplo, beta, &temp, &mut c, n);

    Ok(())
}

/// Naive HERK implementation for small matrices.
fn herk_naive<T: Field>(
    uplo: Uplo,
    op: HerkOp,
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

    // Compute C += alpha * A * A^H or C += alpha * A^H * A (skipped when alpha == 0).
    if alpha != T::zero() {
        match op {
            HerkOp::NoTrans => {
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
            HerkOp::ConjTrans => {
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
        }
    }

    // The diagonal of a Hermitian matrix is real. β-scaling of a noisy input and the
    // α·A·Aᴴ accumulation can both leave rounding residue in the imaginary part, so
    // project every diagonal entry back onto the reals (reference ZHERK also stores a
    // real diagonal). No-op for real element types. The operation form is not ZHERK's:
    // ZHERK scales only `DBLE(C(J,J))` by β (and adds the real `ALPHA*RTEMP` for
    // `ConjTrans`), while the β pass above multiplies the whole complex entry by β + 0i
    // (and `ConjTrans` adds the complex α·temp), so a non-finite imaginary part (of
    // C(j,j), or of temp for `ConjTrans`) makes the real part NaN (see the herk/her2k
    // NaN/Inf item in TODO.md).
    for i in 0..n {
        c.set(i, i, T::from_real(c[(i, i)].real()));
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

    /// Regression: `beta == 0` must not read C. A NaN left in the (conceptually
    /// uninitialised) output buffer must not leak into the result through `0*NaN`.
    /// Uses n>=32, k>=8 so the GEMM path (`herk_via_gemm`) is exercised — that is the
    /// path that previously always computed `beta * c[(i,j)] + temp`.
    #[test]
    fn test_herk_beta_zero_ignores_nan_c() {
        let n = 32usize;
        let k = 8usize;
        let mut a = Mat::<f64>::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                a[(i, j)] = (i as f64) * 0.25 - (j as f64) + 1.0;
            }
        }

        let mut c = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                c[(i, j)] = f64::NAN;
            }
        }

        herk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Reference result: C = A·Aᵀ, entirely NaN-free.
        for j in 0..n {
            for i in j..n {
                assert!(
                    c[(i, j)].is_finite(),
                    "NaN leaked into C[{i},{j}] with beta==0"
                );
                let mut expected = 0.0;
                for l in 0..k {
                    expected += a[(i, l)] * a[(j, l)];
                }
                assert!((c[(i, j)] - expected).abs() < 1e-9);
            }
        }
    }

    /// Regression: the same β==0 guard on the small/naive path.
    #[test]
    fn test_herk_naive_beta_zero_ignores_nan_c() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut c = Mat::from_rows(&[&[f64::NAN, f64::NAN], &[f64::NAN, f64::NAN]]);

        herk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 11.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 25.0).abs() < 1e-10);
    }

    // Issue #9 class: reference ?HERK never reads A when alpha == 0 (or k == 0) and never
    // reads C when beta == 0. The quick return `(alpha == 0 || k == 0) && beta == 1` leaves C
    // completely untouched (even a non-real diagonal); otherwise the diagonal ends up real.
    #[test]
    fn test_issue_9_herk_alpha_beta_zero_semantics() {
        use num_complex::Complex64;
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let zero = Complex64::new(0.0, 0.0);
        let same = |got: Complex64, want: Complex64| {
            let eq = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b;
            eq(got.re, want.re) && eq(got.im, want.im)
        };
        let a_f = |i: usize, j: usize| {
            Complex64::new(
                0.5 * ((2 * i + 3 * j) % 7) as f64 - 1.0,
                0.25 * ((i + 5 * j) % 6) as f64 - 0.5,
            )
        };
        let c_f = |i: usize, j: usize| {
            Complex64::new(
                0.5 + 0.125 * (i + j) as f64,
                if i == j {
                    0.3
                } else {
                    0.0625 * ((i + 3 * j) % 5) as f64
                },
            )
        };
        // (alpha, beta, A poisoned, C poisoned)
        let scenarios = [
            (0.0, 0.0, true, true),
            (0.0, 2.0, true, false),
            (0.0, 1.0, true, false),
            (1.5, 0.0, false, true),
            (1.5, 1.0, false, false),
            (1.5, 0.5, false, false),
        ];
        // (40, 24) crosses the GEMM threshold (n >= 32, k >= 8) and the blocked GEMM path.
        for (n, k) in [(4usize, 3usize), (40, 24), (5, 0)] {
            for uplo in [Uplo::Lower, Uplo::Upper] {
                for trans in [Trans::NoTrans, Trans::ConjTrans] {
                    for (alpha, beta, poison_a, poison_c) in scenarios {
                        let notrans = trans == Trans::NoTrans;
                        let (ar, ac) = if notrans { (n, k) } else { (k, n) };
                        let u = |i: usize, l: usize| {
                            if notrans { a_f(i, l) } else { a_f(l, i).conj() }
                        };
                        let mut a: Mat<Complex64> = Mat::zeros(ar, ac);
                        for j in 0..ac {
                            for i in 0..ar {
                                a[(i, j)] = if poison_a { nan } else { a_f(i, j) };
                            }
                        }
                        let mut c: Mat<Complex64> = Mat::zeros(n, n);
                        for j in 0..n {
                            for i in 0..n {
                                c[(i, j)] = if poison_c { nan } else { c_f(i, j) };
                            }
                        }
                        herk(
                            uplo,
                            trans,
                            Complex64::new(alpha, 0.0),
                            a.as_ref(),
                            Complex64::new(beta, 0.0),
                            c.as_mut(),
                        )
                        .unwrap();
                        let quick = (alpha == 0.0 || k == 0) && beta == 1.0;
                        for j in 0..n {
                            for i in 0..n {
                                let in_tri = match uplo {
                                    Uplo::Lower => i >= j,
                                    Uplo::Upper => i <= j,
                                };
                                let got = c[(i, j)];
                                if !in_tri {
                                    let before = if poison_c { nan } else { c_f(i, j) };
                                    assert!(same(got, before), "unreferenced C[{i},{j}] = {got:?}");
                                    continue;
                                }
                                let want = if quick {
                                    c_f(i, j)
                                } else {
                                    let acc = if alpha == 0.0 {
                                        zero
                                    } else {
                                        (0..k).map(|l| u(i, l) * u(j, l).conj()).sum::<Complex64>()
                                            * alpha
                                    };
                                    let c_term = if beta == 0.0 { zero } else { c_f(i, j) * beta };
                                    let val = acc + c_term;
                                    if i == j {
                                        Complex64::new(val.re, 0.0)
                                    } else {
                                        val
                                    }
                                };
                                assert!(
                                    (got - want).norm() <= 1e-10 * (1.0 + want.norm()),
                                    "herk {uplo:?}/{trans:?} n={n} k={k} alpha={alpha} beta={beta}: \
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
