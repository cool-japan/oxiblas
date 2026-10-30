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

    // Reference ?TRSM validates the shapes first and, for alpha == 0, sets X := 0 without
    // reading A or B (a singular A is accepted; NaN/Inf in A or B cannot leak into X).
    super::check_trsm_dims(side, a.nrows(), a.ncols(), m, n)?;
    if alpha == Complex32::new(0.0, 0.0) {
        return Ok(Mat::zeros(m, n));
    }

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
                    // op(A) = A^H: its diagonal is conj(A[i,i]).
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
                    // op(A) = A^H: its diagonal is conj(A[i,i]).
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
                // op(A) = A^H: its diagonal is conj(A[j,j]).
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
                // op(A) = A^H: its diagonal is conj(A[j,j]).
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

#[cfg(test)]
mod tests {
    use super::super::complex64::trsm_c64;
    use super::*;
    use num_complex::Complex64;

    fn c32(re: f32, im: f32) -> Complex32 {
        Complex32::new(re, im)
    }

    fn widen(z: Complex32) -> Complex64 {
        Complex64::new(f64::from(z.re), f64::from(z.im))
    }

    /// Small deterministic LCG so the fixtures are reproducible without a RNG crate.
    struct Lcg(u64);

    impl Lcg {
        /// Next value in `[-0.5, 0.5)`.
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }

        fn next_complex(&mut self) -> Complex32 {
            let re = self.next_unit();
            c32(re, self.next_unit())
        }
    }

    /// Triangular fixture of order `n`: a strongly dominant, genuinely complex
    /// diagonal; small complex entries inside the referenced triangle; and
    /// finite junk in the other triangle (it must never be read).
    fn tri_fixture(n: usize, uplo: Uplo, rng: &mut Lcg) -> Mat<Complex32> {
        let mut a: Mat<Complex32> = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                a[(i, j)] = if i == j {
                    c32(2.0 + rng.next_unit().abs(), 0.5 + rng.next_unit())
                } else if (uplo == Uplo::Lower) == (i > j) {
                    let z = rng.next_complex();
                    c32(z.re * 2.0 / n as f32, z.im * 2.0 / n as f32)
                } else {
                    c32(10.0, 10.0)
                };
            }
        }
        a
    }

    fn rhs_fixture(m: usize, n: usize, rng: &mut Lcg) -> Mat<Complex32> {
        let mut b: Mat<Complex32> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                b[(i, j)] = rng.next_complex();
            }
        }
        b
    }

    /// `op(T)[i][j]` in `f64` for the triangular matrix `T` stored in `a`.
    fn op_t(
        a: &Mat<Complex32>,
        uplo: Uplo,
        trans: Trans,
        diag: Diag,
        i: usize,
        j: usize,
    ) -> Complex64 {
        let (r, c) = if trans == Trans::NoTrans {
            (i, j)
        } else {
            (j, i)
        };
        let v = if r == c {
            if diag == Diag::Unit {
                Complex64::new(1.0, 0.0)
            } else {
                widen(a[(r, c)])
            }
        } else if (uplo == Uplo::Lower) == (r > c) {
            widen(a[(r, c)])
        } else {
            Complex64::new(0.0, 0.0)
        };
        if trans == Trans::ConjTrans {
            v.conj()
        } else {
            v
        }
    }

    /// Checks `op(T) X = alpha B` (Left) or `X op(T) = alpha B` (Right) in `f64`.
    fn assert_residual(
        (side, uplo, trans, diag): (Side, Uplo, Trans, Diag),
        alpha: Complex32,
        a: &Mat<Complex32>,
        b: &Mat<Complex32>,
        x: &Mat<Complex32>,
        tol: f64,
    ) {
        let (m, n) = (b.nrows(), b.ncols());
        for j in 0..n {
            for i in 0..m {
                let mut lhs = Complex64::new(0.0, 0.0);
                match side {
                    Side::Left => {
                        for l in 0..m {
                            lhs += op_t(a, uplo, trans, diag, i, l) * widen(x[(l, j)]);
                        }
                    }
                    Side::Right => {
                        for l in 0..n {
                            lhs += widen(x[(i, l)]) * op_t(a, uplo, trans, diag, l, j);
                        }
                    }
                }
                let rhs = widen(alpha) * widen(b[(i, j)]);
                assert!(
                    (lhs - rhs).norm() <= tol * (1.0 + rhs.norm()),
                    "{side:?}/{uplo:?}/{trans:?}/{diag:?}: residual at ({i},{j}): \
                     got {lhs:?}, expected {rhs:?}"
                );
            }
        }
    }

    /// Solves one configuration with `trsm_c32`, `trsm_c32_in_place` and `trsm_c64`
    /// and checks all three against the dense residual / each other.
    fn check_combo(flags: (Side, Uplo, Trans, Diag), n_a: usize, nrhs: usize, rng: &mut Lcg) {
        let (side, uplo, trans, diag) = flags;
        let (m, n) = match side {
            Side::Left => (n_a, nrhs),
            Side::Right => (nrhs, n_a),
        };
        let a = tri_fixture(n_a, uplo, rng);
        let b = rhs_fixture(m, n, rng);
        let alpha = c32(0.75, -0.25);

        let x = trsm_c32(side, uplo, trans, diag, alpha, a.as_ref(), b.as_ref())
            .expect("fixture diagonal is non-zero");
        // Primary oracle: dense residual, independent of the c64 code path.
        assert_residual(flags, alpha, &a, &b, &x, 1e-4);

        // The in-place entry point expects B pre-scaled by alpha.
        let mut scaled = b.clone();
        for j in 0..n {
            for i in 0..m {
                scaled[(i, j)] = alpha * b[(i, j)];
            }
        }
        trsm_c32_in_place(side, uplo, trans, diag, a.as_ref(), scaled.as_mut())
            .expect("fixture diagonal is non-zero");
        for j in 0..n {
            for i in 0..m {
                assert_eq!(
                    scaled[(i, j)],
                    x[(i, j)],
                    "{flags:?}: in-place vs out-of-place at ({i},{j})"
                );
            }
        }

        // Secondary oracle: the (already ConjTrans-aware) c64 implementation.
        let mut a64: Mat<Complex64> = Mat::zeros(n_a, n_a);
        for j in 0..n_a {
            for i in 0..n_a {
                a64[(i, j)] = widen(a[(i, j)]);
            }
        }
        let mut b64: Mat<Complex64> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                b64[(i, j)] = widen(b[(i, j)]);
            }
        }
        let x64 = trsm_c64(
            side,
            uplo,
            trans,
            diag,
            widen(alpha),
            a64.as_ref(),
            b64.as_ref(),
        )
        .expect("fixture diagonal is non-zero");
        for j in 0..n {
            for i in 0..m {
                let want = x64[(i, j)];
                assert!(
                    (widen(x[(i, j)]) - want).norm() <= 1e-4 * (1.0 + want.norm()),
                    "{flags:?} n_a={n_a}: c32 {:?} vs c64 {want:?} at ({i},{j})",
                    x[(i, j)]
                );
            }
        }
    }

    fn sweep(n_a: usize, nrhs: usize, seed: u64) {
        let mut rng = Lcg(seed);
        for side in [Side::Left, Side::Right] {
            for uplo in [Uplo::Lower, Uplo::Upper] {
                for trans in [Trans::NoTrans, Trans::Trans, Trans::ConjTrans] {
                    for diag in [Diag::NonUnit, Diag::Unit] {
                        check_combo((side, uplo, trans, diag), n_a, nrhs, &mut rng);
                    }
                }
            }
        }
    }

    /// Reporter's first repro: `conj(1+i) * x = 1` gives `x = (1+i)/2`; ignoring the
    /// conjugation of the diagonal gave `0.5-0.5i`.
    #[test]
    fn test_issue_8_trsm_c32_conjtrans_1x1_reporter_case() {
        let a = Mat::from_rows(&[&[c32(1.0, 1.0)]]);
        let b = Mat::from_rows(&[&[c32(1.0, 0.0)]]);
        let x = trsm_c32(
            Side::Left,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            c32(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal is non-zero");
        assert!((x[(0, 0)].re - 0.5).abs() < 1e-6, "got {:?}", x[(0, 0)]);
        assert!((x[(0, 0)].im - 0.5).abs() < 1e-6, "got {:?}", x[(0, 0)]);

        let mut bm = Mat::from_rows(&[&[c32(1.0, 0.0)]]);
        trsm_c32_in_place(
            Side::Left,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            a.as_ref(),
            bm.as_mut(),
        )
        .expect("diagonal is non-zero");
        assert!((bm[(0, 0)].re - 0.5).abs() < 1e-6, "got {:?}", bm[(0, 0)]);
        assert!((bm[(0, 0)].im - 0.5).abs() < 1e-6, "got {:?}", bm[(0, 0)]);
    }

    /// Reporter's second repro: A = [[2+i, i],[0, 1-i]], B = [1, i]. Here
    /// `A^H = [[2-i, 0],[-i, 1+i]]`, so `x0 = 1/(2-i) = 0.4+0.2i` and
    /// `x1 = (i + i*x0)/(1+i) = 0.6+0.8i` (x0 matches Netlib `ctrsm`).
    #[test]
    fn test_issue_8_trsm_c32_conjtrans_2x2_reporter_case() {
        let a = Mat::from_rows(&[
            &[c32(2.0, 1.0), c32(0.0, 1.0)],
            &[c32(0.0, 0.0), c32(1.0, -1.0)],
        ]);
        let b = Mat::from_rows(&[&[c32(1.0, 0.0)], &[c32(0.0, 1.0)]]);
        let x = trsm_c32(
            Side::Left,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            c32(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal is non-zero");
        let want = [c32(0.4, 0.2), c32(0.6, 0.8)];
        for (i, w) in want.iter().enumerate() {
            let got = x[(i, 0)];
            assert!(
                (got.re - w.re).abs() < 1e-6 && (got.im - w.im).abs() < 1e-6,
                "x[{i}] = {got:?}, expected {w:?}"
            );
        }
        assert_residual(
            (Side::Left, Uplo::Upper, Trans::ConjTrans, Diag::NonUnit),
            c32(1.0, 0.0),
            &a,
            &b,
            &x,
            1e-5,
        );
    }

    /// All 24 side / uplo / trans / diag combinations through the naive path.
    #[test]
    fn test_issue_8_trsm_c32_all_flag_combinations_small() {
        sweep(5, 3, 0x5EED_0001);
    }

    /// Order >= 64 routes through the blocked 3M-GEMM path (diagonal blocks are
    /// still solved by the naive helpers, so the ConjTrans fix is exercised there too).
    #[test]
    fn test_issue_8_trsm_c32_all_flag_combinations_blocked() {
        sweep(70, 3, 0x5EED_0002);
    }
}
