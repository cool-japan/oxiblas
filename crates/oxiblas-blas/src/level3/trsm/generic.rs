//! Generic TRSM implementation for all Field types.
//!
//! The solve itself lives in the private `tri_driver` module: diagonal blocks are
//! solved by register-tiled kernels on packed copies of the block (with the
//! reciprocals of the diagonal), off-diagonal blocks are applied with one
//! in-place GEMM on views of A and B (no temporary matrices), and the
//! right-hand sides are cut into fixed panels that can run in parallel.
//!
//! # Threading
//!
//! [`trsm`] and [`trsm_in_place`] are sequential, like [`crate::level3::gemm()`].
//! [`trsm_with_par`] and [`trsm_in_place_with_par`] take a [`Par`]; they
//! process independent panels of right-hand sides (columns of B for
//! `Side::Left`, rows for `Side::Right`) concurrently. The panel boundaries do
//! not depend on the thread count, so every `Par` gives bitwise the same
//! result as the sequential call.
//!
//! # Rounding
//!
//! Diagonal entries are applied as reciprocals (`x · (1/d)`, the reciprocal
//! computed once per entry), and the blocked updates accumulate in a
//! different order than a textbook substitution, so results can differ from
//! a naive solve in the last bits. If a reciprocal overflows (a subnormal
//! diagonal entry), the call divides exactly instead.

use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use crate::level3::gemm_kernel::GemmKernel;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Solves the triangular system and returns the solution matrix.
///
/// Sequential; see [`trsm_with_par`] for the parallel variant.
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
/// // B = L·X for X = [[2, 4], [1, 2], [2, 4]]
/// let b = Mat::from_rows(&[
///     &[4.0f64, 8.0],
///     &[5.0, 10.0],
///     &[13.0, 26.0],
/// ]);
///
/// let x = trsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit,
///              1.0, l.as_ref(), b.as_ref()).unwrap();
///
/// let expected = [[2.0, 4.0], [1.0, 2.0], [2.0, 4.0]];
/// for i in 0..3 {
///     for j in 0..2 {
///         assert!((x[(i, j)] - expected[i][j]).abs() < 1e-10);
///     }
/// }
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
    trsm_with_par(side, uplo, trans, diag, alpha, a, b, Par::Seq)
}

/// Solves the triangular system with parallelization control and returns the
/// solution matrix.
///
/// Same as [`trsm`], with independent panels of right-hand sides processed
/// concurrently when `par` allows it and the problem is large enough. The
/// result is bitwise identical to [`trsm`] for every `par`.
#[allow(clippy::too_many_arguments)]
pub fn trsm_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    par: Par,
) -> Result<Mat<T>, TrsmError> {
    let m = b.nrows();
    let n = b.ncols();

    // Reference ?TRSM validates the shapes first and, for alpha == 0, sets X := 0 without
    // reading A or B (a singular A is accepted; NaN/Inf in A or B cannot leak into X).
    super::check_trsm_dims(side, a.nrows(), a.ncols(), m, n)?;
    if alpha == T::zero() {
        return Ok(Mat::zeros(m, n));
    }

    // X := alpha·B. As in reference ?TRSM, alpha == 1 skips the scaling (a
    // complex multiply by 1 + 0i is not an identity for infinite entries).
    let mut x = Mat::zeros(m, n);
    let scale = alpha != T::one();
    for j in 0..n {
        for i in 0..m {
            x[(i, j)] = if scale { alpha * b[(i, j)] } else { b[(i, j)] };
        }
    }

    trsm_in_place_with_par(side, uplo, trans, diag, a, x.as_mut(), par)?;
    Ok(x)
}

/// Solves the triangular system in-place.
///
/// B is overwritten with the solution X. This function has no `alpha`
/// parameter: B is not scaled (use [`trsm`] for the α-scaled solve).
///
/// For Left solve: op(A)·X = B, where A is m×m
/// For Right solve: X·op(A) = B, where A is n×n
///
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` as selected by `trans`. Only the `uplo`
/// triangle of A is read, and its diagonal only for `Diag::NonUnit`.
///
/// Sequential; see [`trsm_in_place_with_par`] for the parallel variant.
///
/// # Errors
///
/// [`TrsmError::NotSquare`] / [`TrsmError::DimensionMismatch`] for
/// inconsistent shapes, and [`TrsmError::Singular`] when `diag` is
/// `Diag::NonUnit`, B is not empty and a diagonal entry of A is exactly zero.
/// The diagonal is checked before B is modified, so B is left unchanged on
/// error.
pub fn trsm_in_place<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    b: MatMut<'_, T>,
) -> Result<(), TrsmError> {
    trsm_in_place_with_par(side, uplo, trans, diag, a, b, Par::Seq)
}

/// Solves the triangular system in-place with parallelization control.
///
/// Same as [`trsm_in_place`], with independent panels of right-hand sides
/// (columns of B for `Side::Left`, rows of B for `Side::Right`) processed
/// concurrently when `par` allows it and the problem is large enough. The
/// panel boundaries are fixed, so the result is bitwise identical to
/// [`trsm_in_place`] for every `par`.
///
/// # Errors
///
/// As [`trsm_in_place`].
pub fn trsm_in_place_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    b: MatMut<'_, T>,
    par: Par,
) -> Result<(), TrsmError> {
    super::check_trsm_dims(side, a.nrows(), a.ncols(), b.nrows(), b.ncols())?;
    if b.nrows() == 0 || b.ncols() == 0 {
        return Ok(());
    }
    super::tri_driver::trsm_solve(side, uplo, trans, diag, a, b, par)
}

// =============================================================================
// Previous (textbook substitution) implementation, kept as a test oracle for
// the behaviour the blocked solver must reproduce.
// =============================================================================

/// Naive TRSM on a submatrix (for Left side).
#[cfg(test)]
pub(super) fn trsm_naive_submatrix<T: Field>(
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
                    let d = if trans == Trans::ConjTrans {
                        a[(i, i)].conj()
                    } else {
                        a[(i, i)]
                    };
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
                    let d = if trans == Trans::ConjTrans {
                        a[(i, i)].conj()
                    } else {
                        a[(i, i)]
                    };
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
#[cfg(test)]
pub(super) fn trsm_naive_submatrix_right<T: Field>(
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
                let d = if trans == Trans::ConjTrans {
                    a[(j, j)].conj()
                } else {
                    a[(j, j)]
                };
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
                let d = if trans == Trans::ConjTrans {
                    a[(j, j)].conj()
                } else {
                    a[(j, j)]
                };
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
#[cfg(test)]
pub(super) fn trsm_naive<T: Field>(
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
                            let d = if trans == Trans::ConjTrans {
                                a[(i, i)].conj()
                            } else {
                                a[(i, i)]
                            };
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
                // Backward substitution
                for j in 0..n {
                    for i in (0..m).rev() {
                        let diag_val = if diag == Diag::Unit {
                            T::one()
                        } else {
                            let d = if trans == Trans::ConjTrans {
                                a[(i, i)].conj()
                            } else {
                                a[(i, i)]
                            };
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
        }
        Side::Right => {
            // Solve X·A = B (row by row)
            if use_lower {
                // X·L = B means we process columns of A in reverse
                for j in (0..n).rev() {
                    let diag_val = if diag == Diag::Unit {
                        T::one()
                    } else {
                        let d = if trans == Trans::ConjTrans {
                            a[(j, j)].conj()
                        } else {
                            a[(j, j)]
                        };
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
                // X·U = B means we process columns of A in forward order
                for j in 0..n {
                    let diag_val = if diag == Diag::Unit {
                        T::one()
                    } else {
                        let d = if trans == Trans::ConjTrans {
                            a[(j, j)].conj()
                        } else {
                            a[(j, j)]
                        };
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
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    /// Computes `A^H * X` explicitly (no shortcuts: literal conjugate-transpose
    /// times X), used both to build right-hand sides and as an independent
    /// residual check for Side::Left ConjTrans solves.
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
    // trsm_naive: regression tests for Finding #2 (generic.rs:1011). Prior
    // to the fix this function applied NO conjugation at all for
    // Trans::ConjTrans (neither diagonal nor off-diagonal terms). All four
    // Left/Right x Lower/Upper branches are covered so every occurrence of
    // the bug pattern is exercised.
    // ------------------------------------------------------------------

    #[test]
    fn test_trsm_naive_conjtrans_left_lower_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_left_fixture();
        // Solve A^H * X = B, so B = A^H * X_expected.
        let b = conj_transpose_mul(&a, &x_expected);

        let mut x = b.clone();
        trsm_naive(
            Side::Left,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            a.as_ref(),
            x.as_mut(),
            3,
            2,
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        // Independent residual check: reconstruct A^H * X_solved and
        // compare it against the original right-hand side B.
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-9, "A^H * X_solved vs B");
    }

    #[test]
    fn test_trsm_naive_conjtrans_left_upper_complex() {
        let a = complex_upper_fixture();
        let x_expected = x_left_fixture();
        let b = conj_transpose_mul(&a, &x_expected);

        let mut x = b.clone();
        trsm_naive(
            Side::Left,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            a.as_ref(),
            x.as_mut(),
            3,
            2,
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-9, "A^H * X_solved vs B");
    }

    #[test]
    fn test_trsm_naive_conjtrans_right_lower_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_right_fixture();
        // Solve X * A^H = B, so B = X_expected * A^H.
        let b = mul_conj_transpose(&x_expected, &a);
        let (m, n) = (x_expected.nrows(), a.nrows());

        let mut x = b.clone();
        trsm_naive(
            Side::Right,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            a.as_ref(),
            x.as_mut(),
            m,
            n,
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = mul_conj_transpose(&x, &a);
        assert_mat_close(&reconstructed, &b, 1e-9, "X_solved * A^H vs B");
    }

    #[test]
    fn test_trsm_naive_conjtrans_right_upper_complex() {
        let a = complex_upper_fixture();
        let x_expected = x_right_fixture();
        let b = mul_conj_transpose(&x_expected, &a);
        let (m, n) = (x_expected.nrows(), a.nrows());

        let mut x = b.clone();
        trsm_naive(
            Side::Right,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            a.as_ref(),
            x.as_mut(),
            m,
            n,
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = mul_conj_transpose(&x, &a);
        assert_mat_close(&reconstructed, &b, 1e-9, "X_solved * A^H vs B");
    }

    // ------------------------------------------------------------------
    // trsm_naive_submatrix / trsm_naive_submatrix_right: these already
    // conjugated off-diagonal terms correctly, but (like complex64.rs)
    // divided by the UNCONJUGATED diagonal for Trans::ConjTrans. These
    // cover the additional diagonal-conjugation fix applied to these
    // siblings (discovered while verifying Finding #2's claim that they
    // were already correct; they were correct for off-diagonal terms only).
    // ------------------------------------------------------------------

    #[test]
    fn test_trsm_naive_submatrix_conjtrans_diagonal_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_left_fixture();
        let b = conj_transpose_mul(&a, &x_expected);

        let mut x = b.clone();
        trsm_naive_submatrix(
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            a.as_ref(),
            x.as_mut(),
            3,
            2,
            0,
            3,
            0,
            2,
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-9, "A^H * X_solved vs B");
    }

    #[test]
    fn test_trsm_naive_submatrix_right_conjtrans_diagonal_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_right_fixture();
        let b = mul_conj_transpose(&x_expected, &a);
        let (m, n) = (x_expected.nrows(), a.nrows());

        let mut x = b.clone();
        trsm_naive_submatrix_right(
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            a.as_ref(),
            x.as_mut(),
            m,
            n,
            0,
            n,
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = mul_conj_transpose(&x, &a);
        assert_mat_close(&reconstructed, &b, 1e-9, "X_solved * A^H vs B");
    }
}
