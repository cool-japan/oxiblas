//! SYR2K: Symmetric Rank-2K update.
//!
//! Computes C = α·A·B^T + α·B·A^T + β·C (or the transposed variant) where C is symmetric.
//!
//! # Operation
//!
//! - For `Trans::NoTrans`: C = α·A·B^T + α·B·A^T + β·C where A, B are n×k, C is n×n
//! - For `Trans::Trans`: C = α·A^T·B + α·B^T·A + β·C where A, B are k×n, C is n×n
//!
//! Only the specified triangle (upper or lower) of C is updated.
//!
//! # Algorithm
//!
//! Larger problems run in one call of the triangle-only blocked kernel shared
//! with SYRK and GEMMT, as the rank-`2k` product `alpha·[A B]·[B A]^T` (or its
//! transposed form): every k-block is packed straight from `A` or `B` by the
//! GEMM packers and only the micro-tiles that meet the `uplo` triangle are
//! computed, accumulating straight into C (no transposed copies and no
//! `n × n` temporary). Each k-block is one sweep over the triangle, and no
//! k-block spans both segments, so C is swept at least twice. Small problems, and complex
//! `NoTrans` problems of every size, use the loops of reference ?SYR2K; the
//! measured switch-over points are documented on the private `use_blocked`.
//! `Trans::ConjTrans` is treated as `Trans` (no conjugation) for every
//! element type.

use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::tri_gemm::{KSegment, tri_update};
use crate::level3::trsm::{Trans, Uplo};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Error type for SYR2K operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Syr2kError {
    /// Matrix C is not square.
    NotSquare,
    /// Dimension mismatch.
    DimensionMismatch,
}

impl core::fmt::Display for Syr2kError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix C is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for Syr2kError {}

/// Performs the symmetric rank-2k update.
///
/// C = α·A·B^T + α·B·A^T + β·C (when trans = `NoTrans`)
/// C = α·A^T·B + α·B^T·A + β·C (when trans = Trans)
///
/// Only the `uplo` triangle of C is written.
///
/// # Arguments
///
/// * `uplo` - Which triangle of C to update (Upper or Lower)
/// * `trans` - Operation on A and B (`NoTrans` or Trans)
/// * `alpha` - Scalar multiplier for A·B^T + B·A^T
/// * `a` - The first input matrix A
/// * `b` - The second input matrix B
/// * `beta` - Scalar multiplier for C
/// * `c` - The symmetric output matrix C (updated in place)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::syr2k::{syr2k, Syr2kError};
/// use oxiblas_blas::level3::trsm::{Trans, Uplo};
/// use oxiblas_matrix::Mat;
///
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
/// syr2k(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut()).unwrap();
///
/// // C = A·B^T + B·A^T
/// // A·B^T = [[17, 23], [39, 53]]
/// // B·A^T = [[17, 39], [23, 53]]
/// // Sum = [[34, 62], [62, 106]]
/// assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
/// assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
/// assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
/// ```
pub fn syr2k<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) -> Result<(), Syr2kError> {
    // Validate C is square
    let n = c.nrows();
    if c.ncols() != n {
        return Err(Syr2kError::NotSquare);
    }

    // Determine k and validate A, B dimensions
    let k = match trans {
        Trans::NoTrans => {
            if a.nrows() != n || b.nrows() != n {
                return Err(Syr2kError::DimensionMismatch);
            }
            if a.ncols() != b.ncols() {
                return Err(Syr2kError::DimensionMismatch);
            }
            a.ncols()
        }
        Trans::Trans | Trans::ConjTrans => {
            if a.ncols() != n || b.ncols() != n {
                return Err(Syr2kError::DimensionMismatch);
            }
            if a.nrows() != b.nrows() {
                return Err(Syr2kError::DimensionMismatch);
            }
            a.nrows()
        }
    };

    // Handle empty cases
    if n == 0 {
        return Ok(());
    }

    // alpha == 0 never reads A or B (reference ?SYR2K: C := beta * C on the
    // triangle); both paths honour that, the naive one is simply cheaper for it.
    if alpha != T::zero() && use_blocked::<T>(trans, n, k, beta == T::zero()) {
        syr2k_blocked(uplo, trans, alpha, a, b, beta, c);
        Ok(())
    } else {
        syr2k_naive(uplo, trans, alpha, a, b, beta, c, n, k)
    }
}

/// Whether [`syr2k`] runs the blocked triangle-only kernel (otherwise the
/// reference loops of `syr2k_naive`) for an `n × n` C and inner dimension `k`.
///
/// Chosen from the time of the blocked kernel over the time of the loops,
/// measured on an Apple M3 (one thread, f64 and f32; the rules follow f64,
/// where the loops fare relatively better). With `beta == 0` (`beta_is_zero`):
///
/// - real `NoTrans` (the loops are column updates): the kernel wins from
///   `k >= 16` at `n = 8..12` (0.82-0.83), from `k >= 6` at `n = 16..64`
///   (0.80-0.91), from `k >= 4` at `n = 128..256` (0.74-0.87) and from
///   `k >= 2` at `n = 512` (0.78); with `k = 1` the loops stay ahead up to
///   `n = 512` (1.04);
/// - real `Trans` (dot products): from `k >= 16` at `n = 12..16` (0.69-0.85),
///   from `k >= 4` at `n = 24..32` (0.64-0.86) and for every `k` from
///   `n = 48` (0.76-0.80 at `k = 1`);
/// - complex `NoTrans`: never. Against the column updates of the loops, the
///   scalar `2 × 2` complex micro-kernel does not pay off: the kernel took
///   1.12-1.8 times as long for Complex64 (`n` up to 1024, `k >= 8`);
/// - complex `Trans` (dot products): from `k >= 128` at `n >= 32`
///   (0.88-0.97, Complex32).
///
/// With `beta != 0` the kernel also scales the triangle up front and reads C
/// back in every tile of both k-segments, which small `k` cannot amortize
/// (`beta = 0.5`, f64, `n = 16..64`): real `NoTrans` wins from `k >= 8` at
/// `n >= 24` (0.69-0.91; 1.01 at `n = 16`), real `Trans` from `k >= 8` at
/// `n >= 24` (0.84-0.91) and from `k >= 2` at `n >= 48` (0.65-0.88), but took
/// 1.08-1.09 times as long with `k = 4` at `n = 24` and with `k = 1` at
/// `n = 48`. Shapes without a `beta != 0` measurement keep the loops (the
/// path they took before the blocked kernel existed); `k >= 16` at `n >= 16`
/// is taken for both `trans`, as it amortizes the extra pass twice as well as
/// `k = 8`. The complex rule needs a long `k` and applies unchanged.
fn use_blocked<T: Field>(trans: Trans, n: usize, k: usize, beta_is_zero: bool) -> bool {
    match (T::is_real(), trans, beta_is_zero) {
        (true, Trans::NoTrans, true) => {
            (n >= 8 && k >= 16)
                || (n >= 16 && k >= 6)
                || (n >= 128 && k >= 4)
                || (n >= 512 && k >= 2)
        }
        (true, Trans::NoTrans, false) => (n >= 16 && k >= 16) || (n >= 24 && k >= 8),
        (true, Trans::Trans | Trans::ConjTrans, true) => {
            (n >= 12 && k >= 16) || (n >= 24 && k >= 4) || n >= 48
        }
        (true, Trans::Trans | Trans::ConjTrans, false) => {
            (n >= 16 && k >= 16) || (n >= 24 && k >= 8) || (n >= 48 && k >= 2)
        }
        (false, Trans::NoTrans, _) => false,
        (false, Trans::Trans | Trans::ConjTrans, _) => n >= 32 && k >= 128,
    }
}

/// Blocked SYR2K: one triangle-only call of the GEMM packers and
/// micro-kernel over `k' = 2k`.
///
/// `NoTrans` computes `C := beta·C + alpha·[A B]·[B A]ᵀ` as the two k-segments
/// `(A, Bᵀ)` and `(B, Aᵀ)`; `Trans` (and `ConjTrans`, which this routine has
/// always treated as the plain transpose) computes `alpha·[A B]ᵀ·[B A]` as
/// `(Aᵀ, B)` and `(Bᵀ, A)`. Every k-block is packed straight from `A` or `B`
/// according to the segment its k range falls in, so neither the
/// concatenations nor any transposed copy or `n × n` temporary are formed,
/// and C is accumulated in place (one sweep of the triangle per k-block, so
/// at least one per segment).
fn syr2k_blocked<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) {
    let segments = match trans {
        Trans::NoTrans => [
            KSegment::new(a, Trans::NoTrans, b, Trans::Trans),
            KSegment::new(b, Trans::NoTrans, a, Trans::Trans),
        ],
        Trans::Trans | Trans::ConjTrans => [
            KSegment::new(a, Trans::Trans, b, Trans::NoTrans),
            KSegment::new(b, Trans::Trans, a, Trans::NoTrans),
        ],
    };
    tri_update(uplo, alpha, &segments, beta, c);
}

/// Naive SYR2K implementation for small matrices.
fn syr2k_naive<T: Field>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    n: usize,
    k: usize,
) -> Result<(), Syr2kError> {
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

    // Compute C += alpha * (A * B^T + B * A^T) or C += alpha * (A^T * B + B^T * A)
    match trans {
        Trans::NoTrans => {
            // C += alpha * A * B^T + alpha * B * A^T
            // C[i,j] += alpha * sum_l (A[i,l] * B[j,l] + B[i,l] * A[j,l])
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for l in 0..k {
                            let temp1 = alpha * b[(j, l)];
                            let temp2 = alpha * a[(j, l)];
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
                            let temp1 = alpha * b[(j, l)];
                            let temp2 = alpha * a[(j, l)];
                            for i in 0..=j {
                                let val = c[(i, j)] + a[(i, l)] * temp1 + b[(i, l)] * temp2;
                                c.set(i, j, val);
                            }
                        }
                    }
                }
            }
        }
        Trans::Trans | Trans::ConjTrans => {
            // C += alpha * A^T * B + alpha * B^T * A
            // C[i,j] += alpha * sum_l (A[l,i] * B[l,j] + B[l,i] * A[l,j])
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut temp = T::zero();
                            for l in 0..k {
                                temp = temp + a[(l, i)] * b[(l, j)] + b[(l, i)] * a[(l, j)];
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
                                temp = temp + a[(l, i)] * b[(l, j)] + b[(l, i)] * a[(l, j)];
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

/// Performs symmetric rank-2k update and returns the result.
///
/// This is a convenience function that allocates a new output matrix.
///
/// # Arguments
///
/// * `uplo` - Which triangle to compute and store
/// * `trans` - Operation on A and B
/// * `alpha` - Scalar multiplier
/// * `a` - The first input matrix A
/// * `b` - The second input matrix B
///
/// # Returns
///
/// A new symmetric matrix C = α·A·B^T + α·B·A^T (or the transposed variant).
/// The result is fully symmetric (both triangles are filled).
pub fn syr2k_new<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, Syr2kError> {
    let n = match trans {
        Trans::NoTrans => a.nrows(),
        Trans::Trans | Trans::ConjTrans => a.ncols(),
    };

    let mut c = Mat::zeros(n, n);
    syr2k(uplo, trans, alpha, a, b, T::zero(), c.as_mut())?;

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
    fn test_syr2k_lower_no_trans() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // A·B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]] = [[17, 23], [39, 53]]
        // B·A^T = [[5*1+6*2, 5*3+6*4], [7*1+8*2, 7*3+8*4]] = [[17, 39], [23, 53]]
        // A·B^T + B·A^T = [[34, 62], [62, 106]]
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        syr2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Check lower triangle
        assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2k_upper_no_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        syr2k(
            Uplo::Upper,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // Check upper triangle
        assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2k_lower_trans() {
        // A = [[1, 2, 3], [4, 5, 6]] (2×3), B = [[7, 8, 9], [10, 11, 12]] (2×3)
        // A^T·B = [[1*7+4*10, 1*8+4*11, 1*9+4*12], ...] (3×3)
        // B^T·A = [[7*1+10*4, ...], ...]
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Mat::from_rows(&[&[7.0f64, 8.0, 9.0], &[10.0, 11.0, 12.0]]);

        let mut c = Mat::zeros(3, 3);
        syr2k(
            Uplo::Lower,
            Trans::Trans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // A^T·B[0,0] = 1*7 + 4*10 = 47
        // B^T·A[0,0] = 7*1 + 10*4 = 47
        // Sum[0,0] = 94
        assert!((c[(0, 0)] - 94.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2k_with_alpha() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        syr2k(
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
    fn test_syr2k_with_beta() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let mut c = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        syr2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            2.0,
            c.as_mut(),
        )
        .unwrap();

        // C = A·B^T + B·A^T + 2*C
        // [34, 62] + [2, 0] = [36, 62]
        // [62, 106] + [0, 2] = [62, 108]
        assert!((c[(0, 0)] - 36.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 108.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2k_new() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let c = syr2k_new(Uplo::Lower, Trans::NoTrans, 1.0, a.as_ref(), b.as_ref()).unwrap();

        // Should be fully symmetric
        assert!((c[(0, 0)] - 34.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 62.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 106.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2k_same_matrix() {
        // When A = B, syr2k should give 2 * syrk
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let mut c = Mat::zeros(2, 2);
        syr2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            a.as_ref(),
            0.0,
            c.as_mut(),
        )
        .unwrap();

        // A·A^T + A·A^T = 2 * A·A^T
        // A·A^T = [[5, 11], [11, 25]]
        // 2 * A·A^T = [[10, 22], [22, 50]]
        assert!((c[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2k_empty() {
        let a: Mat<f64> = Mat::zeros(0, 3);
        let b: Mat<f64> = Mat::zeros(0, 3);
        let mut c: Mat<f64> = Mat::zeros(0, 0);
        let result = syr2k(
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
    fn test_syr2k_dimension_mismatch_a() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0], &[9.0, 10.0]]);
        let mut c = Mat::zeros(2, 2);

        let result = syr2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(Syr2kError::DimensionMismatch)));
    }

    #[test]
    fn test_syr2k_dimension_mismatch_k() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Mat::from_rows(&[&[7.0f64, 8.0], &[9.0, 10.0]]);
        let mut c = Mat::zeros(2, 2);

        let result = syr2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(Syr2kError::DimensionMismatch)));
    }

    #[test]
    fn test_syr2k_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let mut c = Mat::zeros(2, 3);

        let result = syr2k(
            Uplo::Lower,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
        );
        assert!(matches!(result, Err(Syr2kError::NotSquare)));
    }

    #[test]
    fn test_syr2k_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f32, 6.0], &[7.0, 8.0]]);

        let mut c = Mat::zeros(2, 2);
        syr2k(
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
}
