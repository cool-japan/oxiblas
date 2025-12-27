//! Generic TRSM implementation for all Field types.

use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use crate::level3::gemm::{GemmBlocking, gemm_with_blocking};
use crate::level3::gemm_kernel::GemmKernel;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Solves the triangular system and returns the solution matrix.
///
/// # Arguments
///
/// * `side` - Left (A·X = α·B) or Right (X·A = α·B)
/// * `uplo` - Lower or Upper triangular
/// * `trans` - Whether to transpose A
/// * `diag` - Whether A has unit diagonal
/// * `alpha` - Scalar multiplier for B
/// * `a` - The triangular matrix A (n×n)
/// * `b` - The right-hand side matrix B
///
/// # Returns
///
/// The solution matrix X.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::trsm::{trsm, Side, Uplo, Trans, Diag};
/// use oxiblas_matrix::Mat;
///
/// // Solve L·X = B where L is lower triangular
/// let l = Mat::from_rows(&[
///     &[2.0f64, 0.0, 0.0],
///     &[1.0, 3.0, 0.0],
///     &[2.0, 1.0, 4.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[4.0f64, 8.0],
///     &[5.0, 11.0],
///     &[14.0, 30.0],
/// ]);
///
/// let x = trsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit,
///              1.0, l.as_ref(), b.as_ref()).unwrap();
///
/// // x should be [[2, 4], [1, 2], [2, 4]]
/// assert!((x[(0, 0)] - 2.0).abs() < 1e-10);
/// assert!((x[(0, 1)] - 4.0).abs() < 1e-10);
/// ```
pub fn trsm<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, TrsmError> {
    let m = b.nrows();
    let n = b.ncols();

    let mut x = Mat::zeros(m, n);
    // Copy B to X and scale by alpha
    for j in 0..n {
        for i in 0..m {
            x[(i, j)] = alpha * b[(i, j)];
        }
    }

    trsm_in_place(side, uplo, trans, diag, a, x.as_mut())?;
    Ok(x)
}

/// Solves the triangular system in-place.
///
/// B is overwritten with the solution X.
///
/// For Left solve: A·X = α·B, where A is m×m
/// For Right solve: X·A = α·B, where A is n×n
///
/// For large matrices, uses blocked algorithm with GEMM for off-diagonal updates.
pub fn trsm_in_place<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    b: MatMut<'_, T>,
) -> Result<(), TrsmError> {
    let m = b.nrows();
    let n = b.ncols();

    // Check dimensions
    match side {
        Side::Left => {
            if a.nrows() != a.ncols() {
                return Err(TrsmError::NotSquare);
            }
            if a.nrows() != m {
                return Err(TrsmError::DimensionMismatch);
            }
        }
        Side::Right => {
            if a.nrows() != a.ncols() {
                return Err(TrsmError::NotSquare);
            }
            if a.nrows() != n {
                return Err(TrsmError::DimensionMismatch);
            }
        }
    }

    if m == 0 || n == 0 {
        return Ok(());
    }

    // Use blocked algorithm for larger matrices
    const BLOCK_SIZE: usize = 64;
    // Use recursive algorithm for very large matrices (divide-and-conquer)
    const RECURSIVE_THRESHOLD: usize = 256;

    let size = match side {
        Side::Left => m,
        Side::Right => n,
    };

    if size >= RECURSIVE_THRESHOLD {
        // Recursive divide-and-conquer for large matrices
        trsm_recursive(side, uplo, trans, diag, a, b, m, n)
    } else if size >= BLOCK_SIZE {
        trsm_blocked(side, uplo, trans, diag, a, b, m, n, BLOCK_SIZE)
    } else {
        trsm_naive(side, uplo, trans, diag, a, b, m, n)
    }
}

/// Recursive TRSM using divide-and-conquer.
///
/// For very large matrices, this achieves better cache utilization by
/// recursively halving the problem size. GEMM updates are parallelized
/// for large matrices to utilize multi-core CPUs.
fn trsm_recursive<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    m: usize,
    n: usize,
) -> Result<(), TrsmError> {
    const BASE_CASE: usize = 64;

    // Get kernel shape for blocking parameter calculation
    let shape = T::micro_kernel_shape();
    let mr = shape.mr;
    let nr = shape.nr;

    let elem_size = std::mem::size_of::<T>();
    let (mc, kc) = if elem_size >= 8 {
        (256, 128)
    } else {
        (256, 256)
    };

    let blocking = GemmBlocking {
        mc: (mc / mr) * mr,
        kc,
        nc: (2048 / nr) * nr,
    };

    // Use parallel GEMM for large updates (threshold: 128x128)
    #[cfg(feature = "parallel")]
    let par = if m >= 128 && n >= 128 {
        Par::Rayon
    } else {
        Par::Seq
    };
    #[cfg(not(feature = "parallel"))]
    let par = Par::Seq;

    // Determine effective storage based on uplo and trans
    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    match side {
        Side::Left => {
            // Solve A·X = B, divide A and B by rows
            if m <= BASE_CASE {
                return trsm_blocked(side, uplo, trans, diag, a, b, m, n, BASE_CASE);
            }

            let mid = m / 2;

            if use_lower {
                // Forward substitution: solve lower block first
                // A = [A11  0  ]  B = [B1]
                //     [A21 A22]      [B2]

                // Step 1: Solve A11·X1 = B1
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(0, 0, mid, mid),
                    b.rb_mut().submatrix(0, 0, mid, n),
                    mid,
                    n,
                )?;

                // Step 2: Update B2 = B2 - A21·X1
                let rows_remaining = m - mid;
                let mut a21: Mat<T> = Mat::zeros(rows_remaining, mid);
                for j in 0..mid {
                    for i in 0..rows_remaining {
                        let a_val = if trans == Trans::NoTrans {
                            a[(mid + i, j)]
                        } else {
                            let val = a[(j, mid + i)];
                            if trans == Trans::ConjTrans {
                                val.conj()
                            } else {
                                val
                            }
                        };
                        a21[(i, j)] = a_val;
                    }
                }

                let mut x1: Mat<T> = Mat::zeros(mid, n);
                for j in 0..n {
                    for i in 0..mid {
                        x1[(i, j)] = b[(i, j)];
                    }
                }

                let mut update: Mat<T> = Mat::zeros(rows_remaining, n);
                gemm_with_blocking(
                    T::one(),
                    a21.as_ref(),
                    x1.as_ref(),
                    T::zero(),
                    update.as_mut(),
                    par,
                    &blocking,
                );

                for j in 0..n {
                    for i in 0..rows_remaining {
                        b[(mid + i, j)] -= update[(i, j)];
                    }
                }

                // Step 3: Solve A22·X2 = B2
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(mid, mid, rows_remaining, rows_remaining),
                    b.rb_mut().submatrix(mid, 0, rows_remaining, n),
                    rows_remaining,
                    n,
                )?;
            } else {
                // Backward substitution: solve upper block first
                let rows_first = m - mid;

                // Step 1: Solve A22·X2 = B2
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(mid, mid, rows_first, rows_first),
                    b.rb_mut().submatrix(mid, 0, rows_first, n),
                    rows_first,
                    n,
                )?;

                // Step 2: Update B1 = B1 - A12·X2
                let mut a12: Mat<T> = Mat::zeros(mid, rows_first);
                for j in 0..rows_first {
                    for i in 0..mid {
                        let a_val = if trans == Trans::NoTrans {
                            a[(i, mid + j)]
                        } else {
                            let val = a[(mid + j, i)];
                            if trans == Trans::ConjTrans {
                                val.conj()
                            } else {
                                val
                            }
                        };
                        a12[(i, j)] = a_val;
                    }
                }

                let mut x2: Mat<T> = Mat::zeros(rows_first, n);
                for j in 0..n {
                    for i in 0..rows_first {
                        x2[(i, j)] = b[(mid + i, j)];
                    }
                }

                let mut update: Mat<T> = Mat::zeros(mid, n);
                gemm_with_blocking(
                    T::one(),
                    a12.as_ref(),
                    x2.as_ref(),
                    T::zero(),
                    update.as_mut(),
                    par,
                    &blocking,
                );

                for j in 0..n {
                    for i in 0..mid {
                        b[(i, j)] -= update[(i, j)];
                    }
                }

                // Step 3: Solve A11·X1 = B1
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(0, 0, mid, mid),
                    b.rb_mut().submatrix(0, 0, mid, n),
                    mid,
                    n,
                )?;
            }
        }
        Side::Right => {
            // Solve X·A = B, divide A and B by columns
            if n <= BASE_CASE {
                return trsm_blocked(side, uplo, trans, diag, a, b, m, n, BASE_CASE);
            }

            let mid = n / 2;

            if use_lower {
                // Backward substitution for Right side with lower triangle
                let cols_first = n - mid;

                // Step 1: Solve X2·A22 = B2
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(mid, mid, cols_first, cols_first),
                    b.rb_mut().submatrix(0, mid, m, cols_first),
                    m,
                    cols_first,
                )?;

                // Step 2: Update B1 = B1 - X2·A21
                let mut a21: Mat<T> = Mat::zeros(cols_first, mid);
                for j in 0..mid {
                    for i in 0..cols_first {
                        let a_val = if trans == Trans::NoTrans {
                            a[(mid + i, j)]
                        } else {
                            let val = a[(j, mid + i)];
                            if trans == Trans::ConjTrans {
                                val.conj()
                            } else {
                                val
                            }
                        };
                        a21[(i, j)] = a_val;
                    }
                }

                let mut x2: Mat<T> = Mat::zeros(m, cols_first);
                for j in 0..cols_first {
                    for i in 0..m {
                        x2[(i, j)] = b[(i, mid + j)];
                    }
                }

                let mut update: Mat<T> = Mat::zeros(m, mid);
                gemm_with_blocking(
                    T::one(),
                    x2.as_ref(),
                    a21.as_ref(),
                    T::zero(),
                    update.as_mut(),
                    par,
                    &blocking,
                );

                for j in 0..mid {
                    for i in 0..m {
                        b[(i, j)] -= update[(i, j)];
                    }
                }

                // Step 3: Solve X1·A11 = B1
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(0, 0, mid, mid),
                    b.rb_mut().submatrix(0, 0, m, mid),
                    m,
                    mid,
                )?;
            } else {
                // Forward substitution for Right side with upper triangle
                // Step 1: Solve X1·A11 = B1
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(0, 0, mid, mid),
                    b.rb_mut().submatrix(0, 0, m, mid),
                    m,
                    mid,
                )?;

                // Step 2: Update B2 = B2 - X1·A12
                let cols_remaining = n - mid;
                let mut a12: Mat<T> = Mat::zeros(mid, cols_remaining);
                for j in 0..cols_remaining {
                    for i in 0..mid {
                        let a_val = if trans == Trans::NoTrans {
                            a[(i, mid + j)]
                        } else {
                            let val = a[(mid + j, i)];
                            if trans == Trans::ConjTrans {
                                val.conj()
                            } else {
                                val
                            }
                        };
                        a12[(i, j)] = a_val;
                    }
                }

                let mut x1: Mat<T> = Mat::zeros(m, mid);
                for j in 0..mid {
                    for i in 0..m {
                        x1[(i, j)] = b[(i, j)];
                    }
                }

                let mut update: Mat<T> = Mat::zeros(m, cols_remaining);
                gemm_with_blocking(
                    T::one(),
                    x1.as_ref(),
                    a12.as_ref(),
                    T::zero(),
                    update.as_mut(),
                    par,
                    &blocking,
                );

                for j in 0..cols_remaining {
                    for i in 0..m {
                        b[(i, mid + j)] -= update[(i, j)];
                    }
                }

                // Step 3: Solve X2·A22 = B2
                trsm_recursive(
                    side,
                    uplo,
                    trans,
                    diag,
                    a.submatrix(mid, mid, cols_remaining, cols_remaining),
                    b.rb_mut().submatrix(0, mid, m, cols_remaining),
                    m,
                    cols_remaining,
                )?;
            }
        }
    }

    Ok(())
}

/// Blocked TRSM algorithm using GEMM for off-diagonal updates.
fn trsm_blocked<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    m: usize,
    n: usize,
    nb: usize,
) -> Result<(), TrsmError> {
    // Get kernel shape for blocking parameter calculation
    let shape = T::micro_kernel_shape();
    let mr = shape.mr;
    let nr = shape.nr;

    // Use moderate blocking parameters optimized for triangular operations
    // TRSM uses smaller blocks than GEMM to reduce overhead from submatrix extraction
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

    // Determine effective storage based on uplo and trans
    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    match side {
        Side::Left => {
            // Solve A·X = B
            // Process in blocks
            if use_lower {
                // Forward substitution with blocks
                let mut k = 0;
                while k < m {
                    let kb = (nb).min(m - k);

                    // Solve diagonal block: A[k:k+kb, k:k+kb] · X[k:k+kb, :] = B[k:k+kb, :]
                    trsm_naive_submatrix(uplo, trans, diag, a, b.rb_mut(), m, n, k, k + kb, 0, n)?;

                    // Update remaining rows: B[k+kb:, :] -= A[k+kb:, k:k+kb] · X[k:k+kb, :]
                    if k + kb < m {
                        let rows_remaining = m - (k + kb);

                        // Extract submatrices for GEMM
                        // A_sub = A[k+kb:m, k:k+kb]
                        let mut a_sub: Mat<T> = Mat::zeros(rows_remaining, kb);
                        for j in 0..kb {
                            for i in 0..rows_remaining {
                                let a_val = if trans == Trans::NoTrans {
                                    a[(k + kb + i, k + j)]
                                } else {
                                    let val = a[(k + j, k + kb + i)];
                                    if trans == Trans::ConjTrans {
                                        val.conj()
                                    } else {
                                        val
                                    }
                                };
                                a_sub[(i, j)] = a_val;
                            }
                        }

                        // X_sub = B[k:k+kb, :]
                        let mut x_sub: Mat<T> = Mat::zeros(kb, n);
                        for j in 0..n {
                            for i in 0..kb {
                                x_sub[(i, j)] = b[(k + i, j)];
                            }
                        }

                        // Compute update = A_sub * X_sub (using custom blocking)
                        let mut update: Mat<T> = Mat::zeros(rows_remaining, n);
                        gemm_with_blocking(
                            T::one(),
                            a_sub.as_ref(),
                            x_sub.as_ref(),
                            T::zero(),
                            update.as_mut(),
                            Par::Seq,
                            &blocking,
                        );

                        // B[k+kb:, :] -= update
                        for j in 0..n {
                            for i in 0..rows_remaining {
                                b[(k + kb + i, j)] -= update[(i, j)];
                            }
                        }
                    }

                    k += kb;
                }
            } else {
                // Backward substitution with blocks
                let mut k = m;
                while k > 0 {
                    let kb = (nb).min(k);
                    let kstart = k - kb;

                    // Solve diagonal block: A[kstart:k, kstart:k] · X[kstart:k, :] = B[kstart:k, :]
                    trsm_naive_submatrix(uplo, trans, diag, a, b.rb_mut(), m, n, kstart, k, 0, n)?;

                    // Update previous rows: B[0:kstart, :] -= A[0:kstart, kstart:k] · X[kstart:k, :]
                    if kstart > 0 {
                        let rows_remaining = kstart;

                        // Extract submatrices for GEMM
                        // A_sub = A[0:kstart, kstart:k]
                        let mut a_sub: Mat<T> = Mat::zeros(rows_remaining, kb);
                        for j in 0..kb {
                            for i in 0..rows_remaining {
                                let a_val = if trans == Trans::NoTrans {
                                    a[(i, kstart + j)]
                                } else {
                                    let val = a[(kstart + j, i)];
                                    if trans == Trans::ConjTrans {
                                        val.conj()
                                    } else {
                                        val
                                    }
                                };
                                a_sub[(i, j)] = a_val;
                            }
                        }

                        // X_sub = B[kstart:k, :]
                        let mut x_sub: Mat<T> = Mat::zeros(kb, n);
                        for j in 0..n {
                            for i in 0..kb {
                                x_sub[(i, j)] = b[(kstart + i, j)];
                            }
                        }

                        // Compute update = A_sub * X_sub (using custom blocking)
                        let mut update: Mat<T> = Mat::zeros(rows_remaining, n);
                        gemm_with_blocking(
                            T::one(),
                            a_sub.as_ref(),
                            x_sub.as_ref(),
                            T::zero(),
                            update.as_mut(),
                            Par::Seq,
                            &blocking,
                        );

                        // B[0:kstart, :] -= update
                        for j in 0..n {
                            for i in 0..rows_remaining {
                                b[(i, j)] -= update[(i, j)];
                            }
                        }
                    }

                    k = kstart;
                }
            }
        }
        Side::Right => {
            // Solve X·A = B
            if use_lower {
                // Process columns in reverse order
                let mut k = n;
                while k > 0 {
                    let kb = (nb).min(k);
                    let kstart = k - kb;

                    // Solve diagonal block: X[:, kstart:k] · A[kstart:k, kstart:k] = B[:, kstart:k]
                    trsm_naive_submatrix_right(uplo, trans, diag, a, b.rb_mut(), m, n, kstart, k)?;

                    // Update previous columns: B[:, 0:kstart] -= X[:, kstart:k] · A[kstart:k, 0:kstart]
                    if kstart > 0 {
                        let cols_remaining = kstart;

                        // Extract A_sub = A[kstart:k, 0:kstart]
                        let mut a_sub: Mat<T> = Mat::zeros(kb, cols_remaining);
                        for j in 0..cols_remaining {
                            for i in 0..kb {
                                let a_val = if trans == Trans::NoTrans {
                                    a[(kstart + i, j)]
                                } else {
                                    let val = a[(j, kstart + i)];
                                    if trans == Trans::ConjTrans {
                                        val.conj()
                                    } else {
                                        val
                                    }
                                };
                                a_sub[(i, j)] = a_val;
                            }
                        }

                        // X_sub = B[:, kstart:k]
                        let mut x_sub: Mat<T> = Mat::zeros(m, kb);
                        for j in 0..kb {
                            for i in 0..m {
                                x_sub[(i, j)] = b[(i, kstart + j)];
                            }
                        }

                        // Compute update = X_sub * A_sub (using custom blocking)
                        let mut update: Mat<T> = Mat::zeros(m, cols_remaining);
                        gemm_with_blocking(
                            T::one(),
                            x_sub.as_ref(),
                            a_sub.as_ref(),
                            T::zero(),
                            update.as_mut(),
                            Par::Seq,
                            &blocking,
                        );

                        // B[:, 0:kstart] -= update
                        for j in 0..cols_remaining {
                            for i in 0..m {
                                b[(i, j)] -= update[(i, j)];
                            }
                        }
                    }

                    k = kstart;
                }
            } else {
                // Process columns in forward order
                let mut k = 0;
                while k < n {
                    let kb = (nb).min(n - k);

                    // Solve diagonal block: X[:, k:k+kb] · A[k:k+kb, k:k+kb] = B[:, k:k+kb]
                    trsm_naive_submatrix_right(uplo, trans, diag, a, b.rb_mut(), m, n, k, k + kb)?;

                    // Update remaining columns: B[:, k+kb:] -= X[:, k:k+kb] · A[k:k+kb, k+kb:]
                    if k + kb < n {
                        let cols_remaining = n - (k + kb);

                        // Extract A_sub = A[k:k+kb, k+kb:n]
                        let mut a_sub: Mat<T> = Mat::zeros(kb, cols_remaining);
                        for j in 0..cols_remaining {
                            for i in 0..kb {
                                let a_val = if trans == Trans::NoTrans {
                                    a[(k + i, k + kb + j)]
                                } else {
                                    let val = a[(k + kb + j, k + i)];
                                    if trans == Trans::ConjTrans {
                                        val.conj()
                                    } else {
                                        val
                                    }
                                };
                                a_sub[(i, j)] = a_val;
                            }
                        }

                        // X_sub = B[:, k:k+kb]
                        let mut x_sub: Mat<T> = Mat::zeros(m, kb);
                        for j in 0..kb {
                            for i in 0..m {
                                x_sub[(i, j)] = b[(i, k + j)];
                            }
                        }

                        // Compute update = X_sub * A_sub (using custom blocking)
                        let mut update: Mat<T> = Mat::zeros(m, cols_remaining);
                        gemm_with_blocking(
                            T::one(),
                            x_sub.as_ref(),
                            a_sub.as_ref(),
                            T::zero(),
                            update.as_mut(),
                            Par::Seq,
                            &blocking,
                        );

                        // B[:, k+kb:] -= update
                        for j in 0..cols_remaining {
                            for i in 0..m {
                                b[(i, k + kb + j)] -= update[(i, j)];
                            }
                        }
                    }

                    k += kb;
                }
            }
        }
    }

    Ok(())
}

/// Naive TRSM on a submatrix (for Left side).
fn trsm_naive_submatrix<T: Field>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    _m: usize,
    _n: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> Result<(), TrsmError> {
    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    if use_lower {
        for j in col_start..col_end {
            for i in row_start..row_end {
                let diag_val = if diag == Diag::Unit {
                    T::one()
                } else {
                    let d = a[(i, i)];
                    if d == T::zero() {
                        return Err(TrsmError::Singular);
                    }
                    d
                };

                let mut sum = b[(i, j)];
                for k in row_start..i {
                    let a_val = if trans == Trans::NoTrans {
                        a[(i, k)]
                    } else {
                        let val = a[(k, i)];
                        if trans == Trans::ConjTrans {
                            val.conj()
                        } else {
                            val
                        }
                    };
                    sum -= a_val * b[(k, j)];
                }
                b[(i, j)] = sum / diag_val;
            }
        }
    } else {
        for j in col_start..col_end {
            for i in (row_start..row_end).rev() {
                let diag_val = if diag == Diag::Unit {
                    T::one()
                } else {
                    let d = a[(i, i)];
                    if d == T::zero() {
                        return Err(TrsmError::Singular);
                    }
                    d
                };

                let mut sum = b[(i, j)];
                for k in (i + 1)..row_end {
                    let a_val = if trans == Trans::NoTrans {
                        a[(i, k)]
                    } else {
                        let val = a[(k, i)];
                        if trans == Trans::ConjTrans {
                            val.conj()
                        } else {
                            val
                        }
                    };
                    sum -= a_val * b[(k, j)];
                }
                b[(i, j)] = sum / diag_val;
            }
        }
    }

    Ok(())
}

/// Naive TRSM on a submatrix (for Right side).
fn trsm_naive_submatrix_right<T: Field>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    m: usize,
    _n: usize,
    col_start: usize,
    col_end: usize,
) -> Result<(), TrsmError> {
    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    if use_lower {
        for j in (col_start..col_end).rev() {
            let diag_val = if diag == Diag::Unit {
                T::one()
            } else {
                let d = a[(j, j)];
                if d == T::zero() {
                    return Err(TrsmError::Singular);
                }
                d
            };

            for i in 0..m {
                b[(i, j)] /= diag_val;
            }

            for k in col_start..j {
                let a_val = if trans == Trans::NoTrans {
                    a[(j, k)]
                } else {
                    let val = a[(k, j)];
                    if trans == Trans::ConjTrans {
                        val.conj()
                    } else {
                        val
                    }
                };
                for i in 0..m {
                    b[(i, k)] = b[(i, k)] - b[(i, j)] * a_val;
                }
            }
        }
    } else {
        for j in col_start..col_end {
            let diag_val = if diag == Diag::Unit {
                T::one()
            } else {
                let d = a[(j, j)];
                if d == T::zero() {
                    return Err(TrsmError::Singular);
                }
                d
            };

            for i in 0..m {
                b[(i, j)] /= diag_val;
            }

            for k in (j + 1)..col_end {
                let a_val = if trans == Trans::NoTrans {
                    a[(j, k)]
                } else {
                    let val = a[(k, j)];
                    if trans == Trans::ConjTrans {
                        val.conj()
                    } else {
                        val
                    }
                };
                for i in 0..m {
                    b[(i, k)] = b[(i, k)] - b[(i, j)] * a_val;
                }
            }
        }
    }

    Ok(())
}

/// Naive TRSM implementation for small matrices.
fn trsm_naive<T: Field>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    mut b: MatMut<'_, T>,
    m: usize,
    n: usize,
) -> Result<(), TrsmError> {
    // Determine effective storage based on uplo and trans
    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    match side {
        Side::Left => {
            // Solve A·X = B (column by column)
            if use_lower {
                // Forward substitution
                for j in 0..n {
                    for i in 0..m {
                        let diag_val = if diag == Diag::Unit {
                            T::one()
                        } else {
                            let d = a[(i, i)];
                            if d == T::zero() {
                                return Err(TrsmError::Singular);
                            }
                            d
                        };

                        let mut sum = b[(i, j)];
                        for k in 0..i {
                            let a_val = if trans == Trans::NoTrans {
                                a[(i, k)]
                            } else {
                                a[(k, i)]
                            };
                            sum -= a_val * b[(k, j)];
                        }
                        b[(i, j)] = sum / diag_val;
                    }
                }
            } else {
                // Backward substitution
                for j in 0..n {
                    for i in (0..m).rev() {
                        let diag_val = if diag == Diag::Unit {
                            T::one()
                        } else {
                            let d = a[(i, i)];
                            if d == T::zero() {
                                return Err(TrsmError::Singular);
                            }
                            d
                        };

                        let mut sum = b[(i, j)];
                        for k in (i + 1)..m {
                            let a_val = if trans == Trans::NoTrans {
                                a[(i, k)]
                            } else {
                                a[(k, i)]
                            };
                            sum -= a_val * b[(k, j)];
                        }
                        b[(i, j)] = sum / diag_val;
                    }
                }
            }
        }
        Side::Right => {
            // Solve X·A = B (row by row)
            if use_lower {
                // X·L = B means we process columns of A in reverse
                for j in (0..n).rev() {
                    let diag_val = if diag == Diag::Unit {
                        T::one()
                    } else {
                        let d = a[(j, j)];
                        if d == T::zero() {
                            return Err(TrsmError::Singular);
                        }
                        d
                    };

                    for i in 0..m {
                        b[(i, j)] /= diag_val;
                    }

                    for k in 0..j {
                        let a_val = if trans == Trans::NoTrans {
                            a[(j, k)]
                        } else {
                            a[(k, j)]
                        };
                        for i in 0..m {
                            b[(i, k)] = b[(i, k)] - b[(i, j)] * a_val;
                        }
                    }
                }
            } else {
                // X·U = B means we process columns of A in forward order
                for j in 0..n {
                    let diag_val = if diag == Diag::Unit {
                        T::one()
                    } else {
                        let d = a[(j, j)];
                        if d == T::zero() {
                            return Err(TrsmError::Singular);
                        }
                        d
                    };

                    for i in 0..m {
                        b[(i, j)] /= diag_val;
                    }

                    for k in (j + 1)..n {
                        let a_val = if trans == Trans::NoTrans {
                            a[(j, k)]
                        } else {
                            a[(k, j)]
                        };
                        for i in 0..m {
                            b[(i, k)] = b[(i, k)] - b[(i, j)] * a_val;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// Complex-optimized TRSM using 3M method
// =============================================================================
