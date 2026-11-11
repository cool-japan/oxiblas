//! TRMM: Triangular matrix-matrix multiply.
//!
//! Performs B = alpha * op(A) * B (left) or B = alpha * B * op(A) (right)
//! where A is a triangular matrix.
//!
//! The multiply is blocked and in place, and touches only the referenced
//! triangle: diagonal blocks of A are packed (the unreferenced triangle is
//! never read, nor the diagonal for `TrmmDiag::Unit`) and multiplied by
//! register-tiled kernels, and off-diagonal blocks are applied with one
//! in-place GEMM on views of A and B, ordered so that every block of B is
//! read before it is overwritten. For an `m × n` B this performs the
//! multiply-adds of a triangular product, about `m²·n/2` (Left, A of order
//! `m`) or `m·n²/2` (Right, A of order `n`), half of a dense GEMM of the
//! same shape, and allocates only the packed diagonal blocks and a small
//! micro-panel buffer. The shared machinery lives in
//! [`mod@crate::level3::trsm`] (private `tri_*` modules).

use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::trsm::tri_driver::trmm_apply;
use crate::level3::trsm::{Side, Trans, Uplo};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Specifies which side the triangular matrix is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmmSide {
    /// B = alpha * op(A) * B (A is on the left).
    Left,
    /// B = alpha * B * op(A) (A is on the right).
    Right,
}

/// Specifies which triangle of the matrix is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmmUplo {
    /// Lower triangular matrix.
    Lower,
    /// Upper triangular matrix.
    Upper,
}

/// Specifies the operation on the triangular matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmmTrans {
    /// No transpose.
    NoTrans,
    /// Transpose.
    Trans,
    /// Conjugate transpose (for complex types).
    ConjTrans,
}

/// Specifies whether the matrix has unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmmDiag {
    /// Non-unit diagonal (use actual diagonal values).
    NonUnit,
    /// Unit diagonal (assume diagonal is all ones).
    Unit,
}

/// Error type for triangular matrix multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmmError {
    /// Matrix A is not square.
    NotSquare,
    /// Dimension mismatch.
    DimensionMismatch,
}

impl core::fmt::Display for TrmmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix A is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for TrmmError {}

/// Performs the triangular matrix-matrix multiply and returns a new matrix.
///
/// # Arguments
///
/// * `side` - Left (B = alpha * op(A) * B) or Right (B = alpha * B * op(A))
/// * `uplo` - Lower or Upper triangular
/// * `trans` - Whether to transpose A
/// * `diag` - Whether A has unit diagonal
/// * `alpha` - Scalar multiplier
/// * `a` - The triangular matrix A (k×k where k = m for Left, k = n for Right)
/// * `b` - The matrix B (m×n)
///
/// # Returns
///
/// The result matrix B = alpha * op(A) * B or B = alpha * B * op(A).
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::trmm::{trmm, TrmmSide, TrmmUplo, TrmmTrans, TrmmDiag};
/// use oxiblas_matrix::Mat;
///
/// // Compute B = L * B where L is lower triangular
/// let l = Mat::from_rows(&[
///     &[2.0f64, 0.0, 0.0],
///     &[1.0, 3.0, 0.0],
///     &[2.0, 1.0, 4.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[1.0, 2.0],
///     &[1.0, 2.0],
/// ]);
///
/// let result = trmm(TrmmSide::Left, TrmmUplo::Lower, TrmmTrans::NoTrans, TrmmDiag::NonUnit,
///                   1.0, l.as_ref(), b.as_ref()).unwrap();
///
/// // result[0,:] = 2*[1,2] = [2, 4]
/// // result[1,:] = 1*[1,2] + 3*[1,2] = [4, 8]
/// // result[2,:] = 2*[1,2] + 1*[1,2] + 4*[1,2] = [7, 14]
/// assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
/// assert!((result[(1, 0)] - 4.0).abs() < 1e-10);
/// assert!((result[(2, 0)] - 7.0).abs() < 1e-10);
/// ```
pub fn trmm<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: TrmmSide,
    uplo: TrmmUplo,
    trans: TrmmTrans,
    diag: TrmmDiag,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, TrmmError> {
    let m = b.nrows();
    let n = b.ncols();

    let mut result = Mat::zeros(m, n);
    // Copy B to result
    for j in 0..n {
        for i in 0..m {
            result[(i, j)] = b[(i, j)];
        }
    }

    trmm_in_place(side, uplo, trans, diag, alpha, a, result.as_mut())?;
    Ok(result)
}

/// Performs the triangular matrix-matrix multiply in-place.
///
/// B is overwritten with alpha * op(A) * B (left) or alpha * B * op(A) (right).
///
/// For Left: A is m×m, B is m×n
/// For Right: A is n×n, B is m×n
///
/// Only the `uplo` triangle of A is read, and its diagonal only for
/// `TrmmDiag::NonUnit`. `alpha == 0` sets B to zero without reading A or B.
/// `alpha == 1` skips the scaling in the diagonal-block kernels; for complex
/// types the off-diagonal GEMM updates still multiply by `1 + 0i`, which can
/// turn an infinite component into NaN. Sequential.
///
/// # Arguments
///
/// * `side` - Left or Right
/// * `uplo` - Lower or Upper triangular
/// * `trans` - Operation on A
/// * `diag` - Unit or `NonUnit` diagonal
/// * `alpha` - Scalar multiplier
/// * `a` - The triangular matrix A
/// * `b` - The matrix B (overwritten with result)
pub fn trmm_in_place<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: TrmmSide,
    uplo: TrmmUplo,
    trans: TrmmTrans,
    diag: TrmmDiag,
    alpha: T,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
) -> Result<(), TrmmError> {
    let m = b.nrows();
    let n = b.ncols();

    // Check A is square
    let k_a = a.nrows();
    if k_a != a.ncols() {
        return Err(TrmmError::NotSquare);
    }

    // Check dimension compatibility
    match side {
        TrmmSide::Left => {
            if k_a != m {
                return Err(TrmmError::DimensionMismatch);
            }
        }
        TrmmSide::Right => {
            if k_a != n {
                return Err(TrmmError::DimensionMismatch);
            }
        }
    }

    if m == 0 || n == 0 {
        return Ok(());
    }

    // Reference ?TRMM: alpha == 0 sets B := 0 without reading A or B.
    if alpha == T::zero() {
        for j in 0..n {
            for i in 0..m {
                b[(i, j)] = T::zero();
            }
        }
        return Ok(());
    }

    let side = match side {
        TrmmSide::Left => Side::Left,
        TrmmSide::Right => Side::Right,
    };
    let uplo = match uplo {
        TrmmUplo::Lower => Uplo::Lower,
        TrmmUplo::Upper => Uplo::Upper,
    };
    let trans = match trans {
        TrmmTrans::NoTrans => Trans::NoTrans,
        TrmmTrans::Trans => Trans::Trans,
        TrmmTrans::ConjTrans => Trans::ConjTrans,
    };
    trmm_apply(
        side,
        uplo,
        trans,
        diag == TrmmDiag::Unit,
        alpha,
        a,
        b,
        Par::Seq,
    );
    Ok(())
}

// =============================================================================
// Previous (textbook) implementation, kept as a test oracle.
// =============================================================================

/// Left multiply: B = alpha * op(A) * B where A is m×m (naive implementation)
#[cfg(test)]
pub(crate) fn trmm_left_naive<T: Field>(
    uplo: TrmmUplo,
    trans: TrmmTrans,
    diag: TrmmDiag,
    alpha: T,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    m: usize,
    n: usize,
) {
    let conj = matches!(trans, TrmmTrans::ConjTrans);
    let do_trans = matches!(trans, TrmmTrans::Trans | TrmmTrans::ConjTrans);

    // The effective triangle after transpose
    let effective_uplo = if do_trans {
        match uplo {
            TrmmUplo::Lower => TrmmUplo::Upper,
            TrmmUplo::Upper => TrmmUplo::Lower,
        }
    } else {
        uplo
    };

    match effective_uplo {
        TrmmUplo::Lower => {
            // Process rows from top to bottom, but compute in reverse for in-place
            // To avoid overwriting values we need, process from bottom to top
            for j in 0..n {
                for i in (0..m).rev() {
                    let mut sum = T::zero();

                    // Elements below and on diagonal (for lower triangle)
                    for k in 0..=i {
                        let a_val = if do_trans {
                            // A^T[i,k] = A[k,i]
                            let val = a[(k, i)];
                            if conj { val.conj() } else { val }
                        } else {
                            a[(i, k)]
                        };

                        if k == i {
                            // Diagonal element
                            let diag_val = if diag == TrmmDiag::Unit {
                                T::one()
                            } else {
                                a_val
                            };
                            sum += diag_val * b[(k, j)];
                        } else if i > k {
                            // Only use if we're actually in the lower triangle
                            sum += a_val * b[(k, j)];
                        }
                    }

                    b[(i, j)] = alpha * sum;
                }
            }
        }
        TrmmUplo::Upper => {
            // Process rows from bottom to top for in-place
            for j in 0..n {
                for i in 0..m {
                    let mut sum = T::zero();

                    // Elements on and above diagonal (for upper triangle)
                    for k in i..m {
                        let a_val = if do_trans {
                            // A^T[i,k] = A[k,i]
                            let val = a[(k, i)];
                            if conj { val.conj() } else { val }
                        } else {
                            a[(i, k)]
                        };

                        if k == i {
                            // Diagonal element
                            let diag_val = if diag == TrmmDiag::Unit {
                                T::one()
                            } else {
                                a_val
                            };
                            sum += diag_val * b[(k, j)];
                        } else if k > i {
                            // Only use if we're actually in the upper triangle
                            sum += a_val * b[(k, j)];
                        }
                    }

                    b[(i, j)] = alpha * sum;
                }
            }
        }
    }
}

/// Right multiply: B = alpha * B * op(A) where A is n×n (naive implementation)
#[cfg(test)]
pub(crate) fn trmm_right_naive<T: Field>(
    uplo: TrmmUplo,
    trans: TrmmTrans,
    diag: TrmmDiag,
    alpha: T,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    m: usize,
    n: usize,
) {
    let conj = matches!(trans, TrmmTrans::ConjTrans);
    let do_trans = matches!(trans, TrmmTrans::Trans | TrmmTrans::ConjTrans);

    // The effective triangle after transpose
    let effective_uplo = if do_trans {
        match uplo {
            TrmmUplo::Lower => TrmmUplo::Upper,
            TrmmUplo::Upper => TrmmUplo::Lower,
        }
    } else {
        uplo
    };

    match effective_uplo {
        TrmmUplo::Lower => {
            // B * L where L is lower triangular
            // Result[i,j] = sum_{k>=j} B[i,k] * L[k,j]
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();

                    for k in j..n {
                        let a_val = if do_trans {
                            // A^T[k,j] = A[j,k]
                            let val = a[(j, k)];
                            if conj { val.conj() } else { val }
                        } else {
                            a[(k, j)]
                        };

                        if k == j {
                            let diag_val = if diag == TrmmDiag::Unit {
                                T::one()
                            } else {
                                a_val
                            };
                            sum += b[(i, k)] * diag_val;
                        } else if k > j {
                            sum += b[(i, k)] * a_val;
                        }
                    }

                    b[(i, j)] = alpha * sum;
                }
            }
        }
        TrmmUplo::Upper => {
            // B * U where U is upper triangular
            // Result[i,j] = sum_{k<=j} B[i,k] * U[k,j]
            for i in 0..m {
                for j in (0..n).rev() {
                    let mut sum = T::zero();

                    for k in 0..=j {
                        let a_val = if do_trans {
                            // A^T[k,j] = A[j,k]
                            let val = a[(j, k)];
                            if conj { val.conj() } else { val }
                        } else {
                            a[(k, j)]
                        };

                        if k == j {
                            let diag_val = if diag == TrmmDiag::Unit {
                                T::one()
                            } else {
                                a_val
                            };
                            sum += b[(i, k)] * diag_val;
                        } else if k < j {
                            sum += b[(i, k)] * a_val;
                        }
                    }

                    b[(i, j)] = alpha * sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trmm_left_lower_notrans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // B = [[1, 2], [1, 2], [1, 2]]
        // L * B = [[2, 4], [4, 8], [7, 14]]
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[1.0, 2.0], &[1.0, 2.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((result[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 4.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 8.0).abs() < 1e-10);
        assert!((result[(2, 0)] - 7.0).abs() < 1e-10);
        assert!((result[(2, 1)] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_left_upper_notrans() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // B = [[1], [1], [1]]
        // U * B = [[5], [4], [4]]
        let u = Mat::from_rows(&[&[2.0f64, 1.0, 2.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[1.0], &[1.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Upper,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            u.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 4.0).abs() < 1e-10);
        assert!((result[(2, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_left_lower_trans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // L^T = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // B = [[1], [1], [1]]
        // L^T * B = [[5], [4], [4]]
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[1.0], &[1.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::Trans,
            TrmmDiag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 4.0).abs() < 1e-10);
        assert!((result[(2, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_right_lower_notrans() {
        // B * L where L is lower triangular
        // L = [[2, 0], [1, 3]]
        // B = [[1, 1]]
        // B * L = [[2+1, 0+3]] = [[3, 3]]
        let l = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 1.0]]);

        let result = trmm(
            TrmmSide::Right,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((result[(0, 1)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_right_upper_notrans() {
        // B * U where U is upper triangular
        // U = [[2, 1], [0, 3]]
        // B = [[1, 1]]
        // B * U = [[2, 1+3]] = [[2, 4]]
        let u = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 1.0]]);

        let result = trmm(
            TrmmSide::Right,
            TrmmUplo::Upper,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            u.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((result[(0, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_unit_diagonal() {
        // L with unit diagonal (diagonal values are ignored)
        // L = [[*, 0, 0], [1, *, 0], [2, 1, *]]
        // B = [[1], [1], [1]]
        // L * B = [[1], [2], [4]]
        let l = Mat::from_rows(&[
            &[99.0f64, 0.0, 0.0], // diagonal ignored
            &[1.0, 88.0, 0.0],
            &[2.0, 1.0, 77.0],
        ]);
        let b = Mat::from_rows(&[&[1.0f64], &[1.0], &[1.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::Unit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((result[(2, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_with_alpha() {
        let l = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[1.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            2.0, // alpha = 2
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        // L * B = [[2], [4]], then * 2 = [[4], [8]]
        assert!((result[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_zero_alpha() {
        let l = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[5.0f64], &[6.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            0.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)]).abs() < 1e-10);
        assert!((result[(1, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_identity() {
        let eye = Mat::<f64>::eye(3);
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            eye.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert!((result[(i, j)] - b[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_trmm_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[1.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            a.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(result, Err(TrmmError::NotSquare)));
    }

    #[test]
    fn test_trmm_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[1.0, 2.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[1.0], &[1.0]]); // 3 rows, but A is 2x2

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            a.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(result, Err(TrmmError::DimensionMismatch)));
    }

    #[test]
    fn test_trmm_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let b: Mat<f64> = Mat::zeros(0, 0);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            a.as_ref(),
            b.as_ref(),
        )
        .unwrap();
        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 0);
    }

    #[test]
    fn test_trmm_f32() {
        let l = Mat::from_rows(&[&[2.0f32, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f32], &[1.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 2.0).abs() < 1e-5);
        assert!((result[(1, 0)] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_trmm_upper_trans() {
        // U = [[2, 1], [0, 3]]
        // U^T = [[2, 0], [1, 3]]
        // B = [[1], [1]]
        // U^T * B = [[2], [4]]
        let u = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[1.0]]);

        let result = trmm(
            TrmmSide::Left,
            TrmmUplo::Upper,
            TrmmTrans::Trans,
            TrmmDiag::NonUnit,
            1.0,
            u.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_in_place() {
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let mut b = Mat::from_rows(&[&[1.0f64, 2.0], &[1.0, 2.0], &[1.0, 2.0]]);

        trmm_in_place(
            TrmmSide::Left,
            TrmmUplo::Lower,
            TrmmTrans::NoTrans,
            TrmmDiag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_mut(),
        )
        .unwrap();

        assert!((b[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((b[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((b[(1, 0)] - 4.0).abs() < 1e-10);
        assert!((b[(1, 1)] - 8.0).abs() < 1e-10);
        assert!((b[(2, 0)] - 7.0).abs() < 1e-10);
        assert!((b[(2, 1)] - 14.0).abs() < 1e-10);
    }

    // Issue #9 class: alpha == 0 zero-fills B without reading A or B (reference ?TRMM).
    #[test]
    fn test_issue_9_trmm_alpha_zero_zero_fills_without_reading_a_or_b() {
        for side in [TrmmSide::Left, TrmmSide::Right] {
            for uplo in [TrmmUplo::Lower, TrmmUplo::Upper] {
                for trans in [TrmmTrans::NoTrans, TrmmTrans::Trans] {
                    // B is 3x2; A is 3x3 (Left) or 2x2 (Right).
                    let ka = if side == TrmmSide::Left { 3 } else { 2 };
                    let a: Mat<f64> = Mat::filled(ka, ka, f64::NAN);
                    let b: Mat<f64> = Mat::filled(3, 2, f64::INFINITY);
                    let result = trmm(
                        side,
                        uplo,
                        trans,
                        TrmmDiag::NonUnit,
                        0.0,
                        a.as_ref(),
                        b.as_ref(),
                    )
                    .expect("shapes are consistent");
                    for j in 0..2 {
                        for i in 0..3 {
                            assert!(result[(i, j)] == 0.0, "{side:?}/{uplo:?}/{trans:?}");
                        }
                    }
                }
            }
        }
    }
}
