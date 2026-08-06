//! Complex64 TRSM specializations using 3M GEMM.

use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use crate::level3::complex_gemm::gemm3m_c64;
use num_complex::Complex64;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Complex64 TRSM using 3M GEMM optimization for off-diagonal updates.
///
/// Solves triangular systems for Complex64 matrices using the 3M method,
/// which reduces complex multiplications from 4 to 3 real operations.
pub fn trsm_c64(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: Complex64,
    a: MatRef<'_, Complex64>,
    b: MatRef<'_, Complex64>,
) -> Result<Mat<Complex64>, TrsmError> {
    let m = b.nrows();
    let n = b.ncols();

    let mut x: Mat<Complex64> = Mat::zeros(m, n);
    // Copy B to X and scale by alpha
    for j in 0..n {
        for i in 0..m {
            x[(i, j)] = alpha * b[(i, j)];
        }
    }

    trsm_c64_in_place(side, uplo, trans, diag, a, x.as_mut())?;
    Ok(x)
}

/// Complex64 TRSM in-place using 3M GEMM for off-diagonal updates.
pub fn trsm_c64_in_place(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex64>,
    b: MatMut<'_, Complex64>,
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
        trsm_c64_blocked(side, uplo, trans, diag, a, b, m, n, BLOCK_SIZE)
    } else {
        trsm_c64_naive(side, uplo, trans, diag, a, b, m, n)
    }
}

/// Blocked Complex64 TRSM using 3M GEMM for off-diagonal updates.
fn trsm_c64_blocked(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex64>,
    mut b: MatMut<'_, Complex64>,
    m: usize,
    n: usize,
    nb: usize,
) -> Result<(), TrsmError> {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);

    // Determine effective storage based on uplo and trans
    let use_lower = match (uplo, trans) {
        (Uplo::Lower, Trans::NoTrans) => true,
        (Uplo::Lower, Trans::Trans | Trans::ConjTrans) => false,
        (Uplo::Upper, Trans::NoTrans) => false,
        (Uplo::Upper, Trans::Trans | Trans::ConjTrans) => true,
    };

    match side {
        Side::Left => {
            if use_lower {
                // Forward substitution with blocks
                let mut k = 0;
                while k < m {
                    let kb = nb.min(m - k);

                    // Solve diagonal block
                    trsm_c64_naive_submatrix(uplo, trans, diag, a, b.rb_mut(), k, k + kb, 0, n)?;

                    // Update remaining rows using 3M GEMM
                    if k + kb < m {
                        let rows_remaining = m - (k + kb);

                        // Extract A_sub and X_sub
                        let mut a_sub: Mat<Complex64> = Mat::zeros(rows_remaining, kb);
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

                        let mut x_sub: Mat<Complex64> = Mat::zeros(kb, n);
                        for j in 0..n {
                            for i in 0..kb {
                                x_sub[(i, j)] = b[(k + i, j)];
                            }
                        }

                        // Use 3M GEMM for the update
                        let mut update: Mat<Complex64> = Mat::zeros(rows_remaining, n);
                        gemm3m_c64(one, a_sub.as_ref(), x_sub.as_ref(), zero, update.as_mut());

                        // B -= update
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
                    let kb = nb.min(k);
                    let kstart = k - kb;

                    // Solve diagonal block
                    trsm_c64_naive_submatrix(uplo, trans, diag, a, b.rb_mut(), kstart, k, 0, n)?;

                    // Update previous rows using 3M GEMM
                    if kstart > 0 {
                        let rows_remaining = kstart;

                        let mut a_sub: Mat<Complex64> = Mat::zeros(rows_remaining, kb);
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

                        let mut x_sub: Mat<Complex64> = Mat::zeros(kb, n);
                        for j in 0..n {
                            for i in 0..kb {
                                x_sub[(i, j)] = b[(kstart + i, j)];
                            }
                        }

                        let mut update: Mat<Complex64> = Mat::zeros(rows_remaining, n);
                        gemm3m_c64(one, a_sub.as_ref(), x_sub.as_ref(), zero, update.as_mut());

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
                // Process columns in reverse order
                let mut k = n;
                while k > 0 {
                    let kb = nb.min(k);
                    let kstart = k - kb;

                    // Solve diagonal block
                    trsm_c64_naive_submatrix_right(uplo, trans, diag, a, b.rb_mut(), m, kstart, k)?;

                    // Update previous columns using 3M GEMM
                    if kstart > 0 {
                        let cols_remaining = kstart;

                        let mut a_sub: Mat<Complex64> = Mat::zeros(kb, cols_remaining);
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

                        let mut x_sub: Mat<Complex64> = Mat::zeros(m, kb);
                        for j in 0..kb {
                            for i in 0..m {
                                x_sub[(i, j)] = b[(i, kstart + j)];
                            }
                        }

                        let mut update: Mat<Complex64> = Mat::zeros(m, cols_remaining);
                        gemm3m_c64(one, x_sub.as_ref(), a_sub.as_ref(), zero, update.as_mut());

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
                    let kb = nb.min(n - k);

                    // Solve diagonal block
                    trsm_c64_naive_submatrix_right(uplo, trans, diag, a, b.rb_mut(), m, k, k + kb)?;

                    // Update remaining columns using 3M GEMM
                    if k + kb < n {
                        let cols_remaining = n - (k + kb);

                        let mut a_sub: Mat<Complex64> = Mat::zeros(kb, cols_remaining);
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

                        let mut x_sub: Mat<Complex64> = Mat::zeros(m, kb);
                        for j in 0..kb {
                            for i in 0..m {
                                x_sub[(i, j)] = b[(i, k + j)];
                            }
                        }

                        let mut update: Mat<Complex64> = Mat::zeros(m, cols_remaining);
                        gemm3m_c64(one, x_sub.as_ref(), a_sub.as_ref(), zero, update.as_mut());

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

    let _ = neg_one; // Suppress unused warning
    Ok(())
}

/// Naive Complex64 TRSM on a submatrix (for Left side).
fn trsm_c64_naive_submatrix(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex64>,
    mut b: MatMut<'_, Complex64>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> Result<(), TrsmError> {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

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
                    let d = if trans == Trans::ConjTrans {
                        a[(i, i)].conj()
                    } else {
                        a[(i, i)]
                    };
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
                    let d = if trans == Trans::ConjTrans {
                        a[(i, i)].conj()
                    } else {
                        a[(i, i)]
                    };
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

/// Naive Complex64 TRSM on a submatrix (for Right side).
fn trsm_c64_naive_submatrix_right(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex64>,
    mut b: MatMut<'_, Complex64>,
    m: usize,
    col_start: usize,
    col_end: usize,
) -> Result<(), TrsmError> {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

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
                let d = if trans == Trans::ConjTrans {
                    a[(j, j)].conj()
                } else {
                    a[(j, j)]
                };
                if d == zero {
                    return Err(TrsmError::Singular);
                }
                d
            };

            // Precompute 1/diag for efficiency
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
                let d = if trans == Trans::ConjTrans {
                    a[(j, j)].conj()
                } else {
                    a[(j, j)]
                };
                if d == zero {
                    return Err(TrsmError::Singular);
                }
                d
            };

            // Precompute 1/diag for efficiency
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

/// Naive Complex64 TRSM implementation for small matrices.
fn trsm_c64_naive(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex64>,
    b: MatMut<'_, Complex64>,
    m: usize,
    n: usize,
) -> Result<(), TrsmError> {
    match side {
        Side::Left => trsm_c64_naive_submatrix(uplo, trans, diag, a, b, 0, m, 0, n),
        Side::Right => trsm_c64_naive_submatrix_right(uplo, trans, diag, a, b, m, 0, n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Computes `A^H * X` explicitly, used both to build right-hand sides
    /// and as an independent residual check for Side::Left ConjTrans solves.
    fn conj_transpose_mul(a: &Mat<Complex64>, x: &Mat<Complex64>) -> Mat<Complex64> {
        let n = a.nrows();
        let ncols = x.ncols();
        let mut result: Mat<Complex64> = Mat::zeros(n, ncols);
        for j in 0..ncols {
            for i in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    // (A^H)[i, k] = conj(A[k, i])
                    sum += a[(k, i)].conj() * x[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    /// Computes `X * A^H` explicitly, used both to build right-hand sides
    /// and as an independent residual check for Side::Right ConjTrans solves.
    fn mul_conj_transpose(x: &Mat<Complex64>, a: &Mat<Complex64>) -> Mat<Complex64> {
        let m = x.nrows();
        let n = a.nrows();
        let mut result: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    // (A^H)[k, j] = conj(A[j, k])
                    sum += x[(i, k)] * a[(j, k)].conj();
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    fn assert_mat_close(actual: &Mat<Complex64>, expected: &Mat<Complex64>, tol: f64, label: &str) {
        assert_eq!(actual.nrows(), expected.nrows());
        assert_eq!(actual.ncols(), expected.ncols());
        for i in 0..actual.nrows() {
            for j in 0..actual.ncols() {
                let diff = actual[(i, j)] - expected[(i, j)];
                assert!(
                    diff.norm() < tol,
                    "{label} mismatch at ({i}, {j}): got {:?}, expected {:?}",
                    actual[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    /// A lower-triangular matrix with genuinely complex entries both on and
    /// off the diagonal (Im != 0 everywhere non-zero), used across the
    /// ConjTrans regression tests below.
    fn complex_lower_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[
                Complex64::new(2.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(1.0, -2.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(0.0, 3.0),
                Complex64::new(-1.0, 2.0),
                Complex64::new(4.0, -1.0),
            ],
        ])
    }

    /// An upper-triangular counterpart with genuinely complex entries both
    /// on and off the diagonal.
    fn complex_upper_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[
                Complex64::new(2.0, 1.0),
                Complex64::new(1.0, -2.0),
                Complex64::new(0.0, 3.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(-1.0, 2.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, -1.0),
            ],
        ])
    }

    /// Arbitrary genuinely-complex 3x2 solution used for Side::Left tests.
    fn x_left_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(0.5, -0.5)],
            &[Complex64::new(2.0, -1.0), Complex64::new(-1.0, 0.0)],
            &[Complex64::new(0.0, 2.0), Complex64::new(3.0, 1.0)],
        ])
    }

    /// Arbitrary genuinely-complex 2x3 solution used for Side::Right tests.
    fn x_right_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(0.0, 2.0),
            ],
            &[
                Complex64::new(0.5, -0.5),
                Complex64::new(-1.0, 0.0),
                Complex64::new(3.0, 1.0),
            ],
        ])
    }

    // ------------------------------------------------------------------
    // Regression tests for Finding #1 (complex64.rs:352): every diagonal
    // read in this file divided by the UNCONJUGATED diagonal for
    // Trans::ConjTrans. All four Left/Right x Lower/Upper branches are
    // covered (they route through trsm_c64_naive_submatrix and
    // trsm_c64_naive_submatrix_right respectively), each with genuinely
    // complex on- and off-diagonal entries so a real-only or
    // diagonal-only-complex matrix could not have caught the bug.
    // ------------------------------------------------------------------

    #[test]
    fn test_trsm_c64_conjtrans_left_lower_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_left_fixture();
        // Solve A^H * X = B, so B = A^H * X_expected.
        let b = conj_transpose_mul(&a, &x_expected);

        let x = trsm_c64(
            Side::Left,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        // Independent residual check: reconstruct A^H * X_solved and
        // compare it against the original right-hand side B.
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-9, "A^H * X_solved vs B");
    }

    #[test]
    fn test_trsm_c64_conjtrans_left_upper_complex() {
        let a = complex_upper_fixture();
        let x_expected = x_left_fixture();
        let b = conj_transpose_mul(&a, &x_expected);

        let x = trsm_c64(
            Side::Left,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-9, "A^H * X_solved vs B");
    }

    #[test]
    fn test_trsm_c64_conjtrans_right_lower_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_right_fixture();
        // Solve X * A^H = B, so B = X_expected * A^H.
        let b = mul_conj_transpose(&x_expected, &a);

        let x = trsm_c64(
            Side::Right,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = mul_conj_transpose(&x, &a);
        assert_mat_close(&reconstructed, &b, 1e-9, "X_solved * A^H vs B");
    }

    #[test]
    fn test_trsm_c64_conjtrans_right_upper_complex() {
        let a = complex_upper_fixture();
        let x_expected = x_right_fixture();
        let b = mul_conj_transpose(&x_expected, &a);

        let x = trsm_c64(
            Side::Right,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = mul_conj_transpose(&x, &a);
        assert_mat_close(&reconstructed, &b, 1e-9, "X_solved * A^H vs B");
    }

    /// Exercises the blocked path (trsm_c64_blocked, size >= 64) with a
    /// genuinely complex ConjTrans lower-triangular solve, so the fix is
    /// verified through the diagonal-block solver AND the 3M-GEMM
    /// off-diagonal update working together.
    #[test]
    fn test_trsm_c64_conjtrans_blocked_large_complex() {
        let n = 80;
        let rhs_cols = 5;

        // Lower-triangular A with a genuinely complex diagonal and
        // genuinely complex sub-diagonal band (off-diagonal terms).
        let mut a: Mat<Complex64> = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = Complex64::new(3.0, 0.7);
            for j in 0..i {
                // Small-magnitude complex off-diagonal entries keep the
                // matrix well-conditioned while still exercising both the
                // real and imaginary parts of the ConjTrans update.
                a[(i, j)] = Complex64::new(0.01, -0.02);
            }
        }

        let mut x_expected: Mat<Complex64> = Mat::zeros(n, rhs_cols);
        for i in 0..n {
            for j in 0..rhs_cols {
                x_expected[(i, j)] = Complex64::new(1.0 + i as f64 * 0.01, 0.5 - j as f64 * 0.02);
            }
        }

        let b = conj_transpose_mul(&a, &x_expected);

        let x = trsm_c64(
            Side::Left,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-6, "solved X vs expected X");
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-6, "A^H * X_solved vs B");
    }
}
