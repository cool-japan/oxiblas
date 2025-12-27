//! TRMM: Triangular matrix-matrix multiply.
//!
//! Performs B = alpha * op(A) * B (left) or B = alpha * B * op(A) (right)
//! where A is a triangular matrix.
//!
//! For large matrices, uses optimized GEMM kernel by expanding the triangular
//! matrix to a full matrix with zeros in the non-triangular part.

use crate::level3::gemm::{GemmBlocking, gemm_with_blocking};
use crate::level3::gemm_kernel::GemmKernel;
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

    // Scale by alpha if needed
    if alpha == T::zero() {
        for j in 0..n {
            for i in 0..m {
                b[(i, j)] = T::zero();
            }
        }
        return Ok(());
    }

    // Use GEMM-based optimization for larger matrices
    const GEMM_THRESHOLD: usize = 32;
    let size = match side {
        TrmmSide::Left => m,
        TrmmSide::Right => n,
    };

    if size >= GEMM_THRESHOLD {
        trmm_via_gemm(side, uplo, trans, diag, alpha, a, b, m, n, k_a)
    } else {
        match side {
            TrmmSide::Left => trmm_left_naive(uplo, trans, diag, alpha, a, b, m, n),
            TrmmSide::Right => trmm_right_naive(uplo, trans, diag, alpha, a, b, m, n),
        }
        Ok(())
    }
}

/// GEMM-based TRMM optimization.
/// Expands triangular matrix to full matrix and uses optimized GEMM.
fn trmm_via_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: TrmmSide,
    uplo: TrmmUplo,
    trans: TrmmTrans,
    diag: TrmmDiag,
    alpha: T,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    m: usize,
    n: usize,
    k_a: usize,
) -> Result<(), TrmmError> {
    // Get kernel shape for blocking parameter calculation
    let shape = T::micro_kernel_shape();
    let mr = shape.mr;
    let nr = shape.nr;

    // Use moderate blocking parameters optimized for triangular operations
    // These are smaller than the default GEMM blocking to reduce overhead
    // for the intermediate-sized matrices typical in TRMM
    let elem_size = std::mem::size_of::<T>();
    let (mc, kc) = if elem_size >= 8 {
        // f64: smaller blocking to reduce overhead
        (256, 128)
    } else {
        // f32: moderate blocking
        (256, 256)
    };

    let blocking = GemmBlocking {
        mc: (mc / mr) * mr,
        kc,
        nc: (2048 / nr) * nr,
    };

    // Create expanded triangular matrix with zeros in non-triangular part
    let mut a_full: Mat<T> = Mat::zeros(k_a, k_a);

    let conj = matches!(trans, TrmmTrans::ConjTrans);

    match uplo {
        TrmmUplo::Lower => {
            for j in 0..k_a {
                for i in j..k_a {
                    if i == j {
                        // Diagonal element
                        a_full[(i, j)] = if diag == TrmmDiag::Unit {
                            T::one()
                        } else if conj {
                            a[(i, j)].conj()
                        } else {
                            a[(i, j)]
                        };
                    } else {
                        a_full[(i, j)] = if conj { a[(i, j)].conj() } else { a[(i, j)] };
                    }
                }
            }
        }
        TrmmUplo::Upper => {
            for j in 0..k_a {
                for i in 0..=j {
                    if i == j {
                        // Diagonal element
                        a_full[(i, j)] = if diag == TrmmDiag::Unit {
                            T::one()
                        } else if conj {
                            a[(i, j)].conj()
                        } else {
                            a[(i, j)]
                        };
                    } else {
                        a_full[(i, j)] = if conj { a[(i, j)].conj() } else { a[(i, j)] };
                    }
                }
            }
        }
    }

    // Prepare A matrix based on transpose
    let a_op: Mat<T> = match trans {
        TrmmTrans::NoTrans => a_full,
        TrmmTrans::Trans | TrmmTrans::ConjTrans => {
            // Transpose the full matrix
            let mut a_t: Mat<T> = Mat::zeros(k_a, k_a);
            for i in 0..k_a {
                for j in 0..k_a {
                    a_t[(j, i)] = a_full[(i, j)];
                }
            }
            a_t
        }
    };

    match side {
        TrmmSide::Left => {
            // B = alpha * op(A) * B
            // Create a copy of B to use as input
            let mut b_copy: Mat<T> = Mat::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    b_copy[(i, j)] = b[(i, j)];
                }
            }

            // Zero out B for GEMM
            for j in 0..n {
                for i in 0..m {
                    b[(i, j)] = T::zero();
                }
            }

            // Compute B = alpha * A * B_copy via GEMM with TRMM-optimized blocking
            gemm_with_blocking(
                alpha,
                a_op.as_ref(),
                b_copy.as_ref(),
                T::zero(),
                b.rb_mut(),
                Par::Seq,
                &blocking,
            );
        }
        TrmmSide::Right => {
            // B = alpha * B * op(A)
            // Create a copy of B to use as input
            let mut b_copy: Mat<T> = Mat::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    b_copy[(i, j)] = b[(i, j)];
                }
            }

            // Zero out B for GEMM
            for j in 0..n {
                for i in 0..m {
                    b[(i, j)] = T::zero();
                }
            }

            // Compute B = alpha * B_copy * A via GEMM with TRMM-optimized blocking
            gemm_with_blocking(
                alpha,
                b_copy.as_ref(),
                a_op.as_ref(),
                T::zero(),
                b.rb_mut(),
                Par::Seq,
                &blocking,
            );
        }
    }

    Ok(())
}

/// Left multiply: B = alpha * op(A) * B where A is m×m (naive implementation)
fn trmm_left_naive<T: Field>(
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
fn trmm_right_naive<T: Field>(
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
}
