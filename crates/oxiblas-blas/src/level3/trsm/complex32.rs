//! Complex32 TRSM entry points.
//!
//! These solve through the same blocked driver as the generic
//! [`super::trsm_in_place`] (register-tiled diagonal blocks, in-place updates
//! on views); for complex element types the off-diagonal updates use the 3M
//! GEMM, which is markedly faster here than the generic complex GEMM kernel.

use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use num_complex::Complex32;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Complex32 TRSM: solves `op(A)·X = alpha·B` (Left) or `X·op(A) = alpha·B`
/// (Right) and returns X.
///
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ`. Sequential; the generic
/// [`super::trsm_with_par`] accepts `Complex32` as well and can run in parallel.
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

    // X := alpha·B; alpha == 1 skips the scaling as in reference ?TRSM (a
    // complex multiply by 1 + 0i is not an identity for infinite entries).
    let mut x: Mat<Complex32> = Mat::zeros(m, n);
    let scale = alpha != Complex32::new(1.0, 0.0);
    for j in 0..n {
        for i in 0..m {
            x[(i, j)] = if scale { alpha * b[(i, j)] } else { b[(i, j)] };
        }
    }

    trsm_c32_in_place(side, uplo, trans, diag, a, x.as_mut())?;
    Ok(x)
}

/// Complex32 TRSM in place: overwrites B with the solution of `op(A)·X = B`
/// (Left) or `X·op(A) = B` (Right).
///
/// Sequential. Errors and the reading of A are as for
/// [`super::trsm_in_place`] (B is unchanged when A is singular).
pub fn trsm_c32_in_place(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex32>,
    b: MatMut<'_, Complex32>,
) -> Result<(), TrsmError> {
    super::generic::trsm_in_place(side, uplo, trans, diag, a, b)
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

    /// Order 130 spans two 128-row diagonal blocks, so the in-place 3M-GEMM
    /// update runs between the two diagonal-block solves (the ConjTrans fix is
    /// exercised in both).
    #[test]
    fn test_issue_8_trsm_c32_all_flag_combinations_blocked() {
        sweep(130, 3, 0x5EED_0002);
    }
}
