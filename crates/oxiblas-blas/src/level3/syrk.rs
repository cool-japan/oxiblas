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
//!
//! # Algorithm
//!
//! Larger problems run on the triangle-only blocked kernel shared with SYR2K
//! and GEMMT: the GEMM packers read `A` in place (the transposed operand
//! through their transposed fast path) and the GEMM micro-kernel computes
//! only the micro-tiles that meet the `uplo` triangle, accumulating straight
//! into C. No transposed copy of `A` and no `n × n` temporary are formed.
//! Small problems, and complex `NoTrans` problems of every size (where the
//! column updates of the loops measured faster than the scalar complex
//! micro-kernel), use the loops of reference ?SYRK; the measured switch-over
//! points are documented on the private `use_blocked`.
//!
//! # `ConjTrans` and complex element types
//!
//! SYRK is the *symmetric* rank-k update and always uses the plain transpose
//! `A^T` (never the conjugate transpose `A^H`). The conjugate-transpose form is
//! the domain of the Hermitian family (HERK/HER2K), so `Trans::ConjTrans` is
//! meaningless for a complex-valued SYRK. Reference C/ZSYRK reject `'C'` outright,
//! while reference S/DSYRK accept `'C'` and treat it exactly as `'T'` because
//! conjugation is the identity on reals. This implementation mirrors that: for
//! real element types `ConjTrans` is accepted and behaves like `Trans`, and for
//! complex element types it is rejected with [`SyrkError::InvalidTrans`].

use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::tri_gemm::{KSegment, tri_update};
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
    /// Invalid transpose option: `Trans::ConjTrans` is not allowed for a
    /// complex-valued symmetric rank-k update (SYRK uses `A^T`, not `A^H`).
    InvalidTrans,
}

impl core::fmt::Display for SyrkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix C is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
            Self::InvalidTrans => write!(
                f,
                "Invalid transpose option: complex SYRK only accepts NoTrans or Trans (ConjTrans is Hermitian-only)"
            ),
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
    // SYRK computes the *symmetric* update using A^T. For complex element types the
    // conjugate transpose A^H is a different operation (that of HERK), so silently
    // treating ConjTrans as Trans would return a plausible-but-wrong result. Reject
    // it explicitly, matching reference C/ZSYRK. For real element types conjugation
    // is a no-op, so ConjTrans is legal and behaves exactly like Trans (matching
    // reference S/DSYRK), and `T::is_real()` keeps it out of this branch.
    if trans == Trans::ConjTrans && !T::is_real() {
        return Err(SyrkError::InvalidTrans);
    }

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

    // alpha == 0 never reads A (reference ?SYRK: C := beta * C on the triangle);
    // both paths honour that, the naive one is simply cheaper for it.
    if alpha != T::zero() && use_blocked::<T>(trans, n, k, beta == T::zero()) {
        syrk_blocked(uplo, trans, alpha, a, beta, c);
        Ok(())
    } else {
        syrk_naive(uplo, trans, alpha, a, beta, c, n, k)
    }
}

/// Whether [`syrk`] runs the blocked triangle-only kernel (otherwise the
/// reference loops of `syrk_naive`) for an `n × n` C and inner dimension `k`.
///
/// Chosen from the time of the blocked kernel over the time of the loops,
/// measured on an Apple M3 (one thread, f64 and f32; the rules follow f64,
/// where the loops fare relatively better). With `beta == 0` (`beta_is_zero`):
///
/// - real `NoTrans` (the loops are column updates `C[:, j] += t·A[:, l]`):
///   the kernel wins from `k >= 8` at `n = 6..12` (ratio 0.68-0.86), from
///   `k >= 4` at `n = 16..32` (0.70-0.88), from `k >= 2` at `n = 48..128`
///   (0.73-0.93) and for every `k` from `n = 256` (0.72-0.75 at `k = 1`);
///   with `k = 1` the loops stay ahead up to `n = 128` (1.14 at `n = 64`);
/// - real `Trans` (the loops form dot products, `A[:, i]·A[:, j]`): from
///   `k >= 8` at `n = 12` (0.80), from `k >= 4` at `n = 16` (0.78) and for
///   every `k` from `n = 24` (0.74 at `k = 1`);
/// - complex `NoTrans`: never. Against the column updates of the loops, the
///   scalar `2 × 2` complex micro-kernel does not pay off: with `k >= 16` the
///   kernel took 0.98-1.08 times as long for Complex64 at `n = 32..1024`
///   (0.81-0.97 at `n = 8..16`, under a microsecond) and 1.2-2.4 times as
///   long for Complex32 at `n >= 16`; smaller `k` favour the loops more;
/// - complex `Trans` (dot products): from `k >= 128` at `n >= 32`
///   (Complex64: 0.78-0.97 for `n = 32..1024`, except 1.11 at
///   `n = k = 1024`).
///
/// With `beta != 0` the kernel also scales the triangle up front and reads C
/// back in every tile, which small `k` cannot amortize (`beta = 0.5`, f64,
/// `n = 16..64`): real `NoTrans` wins from `k >= 8` at `n >= 16` (0.61-0.83)
/// and from `k >= 4` at `n = 64` (0.92), but took 1.10-1.50 times as long
/// with `k = 2..3` at `n = 48..64`; real `Trans` wins from `k >= 8` at
/// `n = 16` (0.88), `k >= 3` at `n = 24` (0.77), `k >= 2` at `n = 32` (0.88)
/// and for every `k` from `n = 48` (0.65-0.73 at `k = 1`). Shapes without a
/// `beta != 0` measurement keep the loops (the path they took before the
/// blocked kernel existed). The complex rule needs a long `k`, which
/// amortizes the extra pass, and applies unchanged.
fn use_blocked<T: Field>(trans: Trans, n: usize, k: usize, beta_is_zero: bool) -> bool {
    match (T::is_real(), trans, beta_is_zero) {
        (true, Trans::NoTrans, true) => {
            (n >= 6 && k >= 8) || (n >= 16 && k >= 4) || (n >= 48 && k >= 2) || n >= 256
        }
        (true, Trans::NoTrans, false) => (n >= 16 && k >= 8) || (n >= 64 && k >= 4),
        (true, Trans::Trans | Trans::ConjTrans, true) => {
            (n >= 12 && k >= 8) || (n >= 16 && k >= 4) || n >= 24
        }
        (true, Trans::Trans | Trans::ConjTrans, false) => {
            (n >= 16 && k >= 8) || (n >= 24 && k >= 3) || (n >= 32 && k >= 2) || n >= 48
        }
        (false, Trans::NoTrans, _) => false,
        (false, Trans::Trans | Trans::ConjTrans, _) => n >= 32 && k >= 128,
    }
}

/// Blocked SYRK: one triangle-only pass of the GEMM packers and micro-kernel.
///
/// `NoTrans` computes `C := beta·C + alpha·A·Aᵀ` as the k-segment `(A, Aᵀ)`,
/// `Trans` (and, for real types, `ConjTrans`) computes `alpha·Aᵀ·A` as
/// `(Aᵀ, A)`. Both operands are packed straight from `A` (the transposed one
/// through the packers' transposed fast path); no transposed copy and no
/// `n × n` temporary are formed, and only the micro-tiles that meet the
/// `uplo` triangle are computed.
fn syrk_blocked<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) {
    let segment = match trans {
        Trans::NoTrans => KSegment::new(a, Trans::NoTrans, a, Trans::Trans),
        // `ConjTrans` only reaches here for real types (complex is rejected
        // above), where it is the plain transpose.
        Trans::Trans | Trans::ConjTrans => KSegment::new(a, Trans::Trans, a, Trans::NoTrans),
    };
    tri_update(uplo, alpha, &[segment], beta, c);
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
#[path = "syrk_family_tests.rs"]
mod family_tests;

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

    /// For real element types, `ConjTrans` is accepted (not rejected) and must compute
    /// exactly the same result as `Trans`, since conjugation is the identity on reals.
    /// This is the documented "no-op" branch of the ConjTrans validation. (Complex
    /// element types are rejected with `SyrkError::InvalidTrans`; that branch is
    /// covered by `complex_syrk_conj_trans_rejected` in `syrk_family_tests.rs`.)
    #[test]
    fn test_syrk_conj_trans_real_matches_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let mut c_trans = Mat::zeros(3, 3);
        syrk(
            Uplo::Lower,
            Trans::Trans,
            1.0,
            a.as_ref(),
            0.0,
            c_trans.as_mut(),
        )
        .unwrap();

        let mut c_conj = Mat::zeros(3, 3);
        syrk(
            Uplo::Lower,
            Trans::ConjTrans,
            1.0,
            a.as_ref(),
            0.0,
            c_conj.as_mut(),
        )
        .unwrap();

        for j in 0..3 {
            for i in j..3 {
                assert!((c_trans[(i, j)] - c_conj[(i, j)]).abs() < 1e-12);
            }
        }
    }

    /// Regression: `beta == 0` must not read C on the blocked path (`syrk_blocked`); a
    /// NaN in the output buffer must not survive via `0*NaN`.
    #[test]
    fn test_syrk_beta_zero_ignores_nan_c() {
        let n = 32usize;
        let k = 8usize;
        let mut a = Mat::<f64>::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                a[(i, j)] = (i as f64) * 0.5 - (j as f64) + 1.0;
            }
        }

        let mut c = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                c[(i, j)] = f64::NAN;
            }
        }

        syrk(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        for j in 0..n {
            for i in j..n {
                assert!(c[(i, j)].is_finite(), "NaN leaked into C[{i},{j}]");
            }
        }
    }
}
