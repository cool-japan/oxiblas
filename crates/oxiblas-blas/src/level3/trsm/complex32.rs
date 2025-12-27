//! Complex32 TRSM specializations using 3M GEMM.

use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use crate::level3::complex_gemm::gemm3m_c32;
use num_complex::Complex32;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Complex32 TRSM using 3M GEMM optimization for off-diagonal updates.
///
/// Solves triangular systems for Complex32 matrices using the 3M method,
/// which reduces complex multiplications from 4 to 3 real operations.
pub fn trsm_c32(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: Complex32,
    a: MatRef<'_, Complex32>,
    b: MatRef<'_, Complex32>,
) -> Result<Mat<Complex32>, TrsmError> {
    let m = b.nrows();
    let n = b.ncols();

    let mut x: Mat<Complex32> = Mat::zeros(m, n);
    // Copy B to X and scale by alpha
    for j in 0..n {
        for i in 0..m {
            x[(i, j)] = alpha * b[(i, j)];
        }
    }

    trsm_c32_in_place(side, uplo, trans, diag, a, x.as_mut())?;
    Ok(x)
}

/// Complex32 TRSM in-place using 3M GEMM for off-diagonal updates.
pub fn trsm_c32_in_place(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex32>,
    b: MatMut<'_, Complex32>,
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
    let size = match side {
        Side::Left => m,
        Side::Right => n,
    };

    if size >= BLOCK_SIZE {
        trsm_c32_blocked(side, uplo, trans, diag, a, b, m, n, BLOCK_SIZE)
    } else {
        trsm_c32_naive(side, uplo, trans, diag, a, b, m, n)
    }
}

/// Blocked Complex32 TRSM using 3M GEMM for off-diagonal updates.
fn trsm_c32_blocked(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex32>,
    mut b: MatMut<'_, Complex32>,
    m: usize,
    n: usize,
    nb: usize,
) -> Result<(), TrsmError> {
    let zero = Complex32::new(0.0, 0.0);
    let one = Complex32::new(1.0, 0.0);

    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    match side {
        Side::Left => {
            if use_lower {
                let mut k = 0;
                while k < m {
                    let kb = nb.min(m - k);
                    trsm_c32_naive_submatrix(uplo, trans, diag, a, b.rb_mut(), k, k + kb, 0, n)?;

                    if k + kb < m {
                        let rows_remaining = m - (k + kb);
                        let mut a_sub: Mat<Complex32> = Mat::zeros(rows_remaining, kb);
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

                        let mut x_sub: Mat<Complex32> = Mat::zeros(kb, n);
                        for j in 0..n {
                            for i in 0..kb {
                                x_sub[(i, j)] = b[(k + i, j)];
                            }
                        }

                        let mut update: Mat<Complex32> = Mat::zeros(rows_remaining, n);
                        gemm3m_c32(one, a_sub.as_ref(), x_sub.as_ref(), zero, update.as_mut());

                        for j in 0..n {
                            for i in 0..rows_remaining {
                                b[(k + kb + i, j)] -= update[(i, j)];
                            }
                        }
                    }
                    k += kb;
                }
            } else {
                let mut k = m;
                while k > 0 {
                    let kb = nb.min(k);
                    let kstart = k - kb;
                    trsm_c32_naive_submatrix(uplo, trans, diag, a, b.rb_mut(), kstart, k, 0, n)?;

                    if kstart > 0 {
                        let rows_remaining = kstart;
                        let mut a_sub: Mat<Complex32> = Mat::zeros(rows_remaining, kb);
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

                        let mut x_sub: Mat<Complex32> = Mat::zeros(kb, n);
                        for j in 0..n {
                            for i in 0..kb {
                                x_sub[(i, j)] = b[(kstart + i, j)];
                            }
                        }

                        let mut update: Mat<Complex32> = Mat::zeros(rows_remaining, n);
                        gemm3m_c32(one, a_sub.as_ref(), x_sub.as_ref(), zero, update.as_mut());

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
            if use_lower {
                let mut k = n;
                while k > 0 {
                    let kb = nb.min(k);
                    let kstart = k - kb;
                    trsm_c32_naive_submatrix_right(uplo, trans, diag, a, b.rb_mut(), m, kstart, k)?;

                    if kstart > 0 {
                        let cols_remaining = kstart;
                        let mut a_sub: Mat<Complex32> = Mat::zeros(kb, cols_remaining);
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

                        let mut x_sub: Mat<Complex32> = Mat::zeros(m, kb);
                        for j in 0..kb {
                            for i in 0..m {
                                x_sub[(i, j)] = b[(i, kstart + j)];
                            }
                        }

                        let mut update: Mat<Complex32> = Mat::zeros(m, cols_remaining);
                        gemm3m_c32(one, x_sub.as_ref(), a_sub.as_ref(), zero, update.as_mut());

                        for j in 0..cols_remaining {
                            for i in 0..m {
                                b[(i, j)] -= update[(i, j)];
                            }
                        }
                    }
                    k = kstart;
                }
            } else {
                let mut k = 0;
                while k < n {
                    let kb = nb.min(n - k);
                    trsm_c32_naive_submatrix_right(uplo, trans, diag, a, b.rb_mut(), m, k, k + kb)?;

                    if k + kb < n {
                        let cols_remaining = n - (k + kb);
                        let mut a_sub: Mat<Complex32> = Mat::zeros(kb, cols_remaining);
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

                        let mut x_sub: Mat<Complex32> = Mat::zeros(m, kb);
                        for j in 0..kb {
                            for i in 0..m {
                                x_sub[(i, j)] = b[(i, k + j)];
                            }
                        }

                        let mut update: Mat<Complex32> = Mat::zeros(m, cols_remaining);
                        gemm3m_c32(one, x_sub.as_ref(), a_sub.as_ref(), zero, update.as_mut());

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

/// Naive Complex32 TRSM on a submatrix (for Left side).
fn trsm_c32_naive_submatrix(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex32>,
    mut b: MatMut<'_, Complex32>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> Result<(), TrsmError> {
    let one = Complex32::new(1.0, 0.0);
    let zero = Complex32::new(0.0, 0.0);

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
                    one
                } else {
                    let d = a[(i, i)];
                    if d == zero {
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
                    one
                } else {
                    let d = a[(i, i)];
                    if d == zero {
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

/// Naive Complex32 TRSM on a submatrix (for Right side).
fn trsm_c32_naive_submatrix_right(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex32>,
    mut b: MatMut<'_, Complex32>,
    m: usize,
    col_start: usize,
    col_end: usize,
) -> Result<(), TrsmError> {
    let one = Complex32::new(1.0, 0.0);
    let zero = Complex32::new(0.0, 0.0);

    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    if use_lower {
        for j in (col_start..col_end).rev() {
            let diag_val = if diag == Diag::Unit {
                one
            } else {
                let d = a[(j, j)];
                if d == zero {
                    return Err(TrsmError::Singular);
                }
                d
            };

            let inv_diag = one / diag_val;
            for i in 0..m {
                b[(i, j)] *= inv_diag;
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
                one
            } else {
                let d = a[(j, j)];
                if d == zero {
                    return Err(TrsmError::Singular);
                }
                d
            };

            let inv_diag = one / diag_val;
            for i in 0..m {
                b[(i, j)] *= inv_diag;
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

/// Naive Complex32 TRSM implementation for small matrices.
fn trsm_c32_naive(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex32>,
    b: MatMut<'_, Complex32>,
    m: usize,
    n: usize,
) -> Result<(), TrsmError> {
    match side {
        Side::Left => trsm_c32_naive_submatrix(uplo, trans, diag, a, b, 0, m, 0, n),
        Side::Right => trsm_c32_naive_submatrix_right(uplo, trans, diag, a, b, m, 0, n),
    }
}
