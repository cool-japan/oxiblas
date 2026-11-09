//! Accuracy and robustness tests of [`HermitianEvdDc`] against independent
//! references: the implicit-QR [`HermitianEvd`], the real symmetric divide
//! and conquer ([`SymmetricEvdDc`]) on the real embedding
//! `[[Re A, -Im A], [Im A, Re A]]` (whose spectrum is that of `A` with every
//! eigenvalue twice), and prescribed spectra.
//!
//! Tolerances are `c·n·ε·‖A‖₂` with `c = 16`, as in the real symmetric
//! divide-and-conquer tests.

use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use oxiblas_matrix::{Mat, MatRef};

use super::{dc_pipeline, hermitian_from_upper, qr_pipeline};
use crate::evd::sym_kernels::{BlasMatMul, NaiveMatMul};
use crate::evd::{HermitianEvd, HermitianEvdDc, HermitianEvdDcError, SymmetricEvdDc};

/// The constant `c` of the `c·n·ε` tolerances.
const TOL_C: f64 = 16.0;

/// Deterministic split-mix PRNG producing values in `[-1, 1]`.
struct Prng(u64);

impl Prng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    fn next_c64(&mut self) -> Complex64 {
        Complex64::new(self.next_f64(), self.next_f64())
    }
}

/// Hermitian matrix with entries uniform in the unit square (real diagonal).
fn random_hermitian(n: usize, seed: u64) -> Mat<Complex64> {
    let mut rng = Prng(seed);
    let mut a: Mat<Complex64> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            let v = if i == j {
                Complex64::new(rng.next_f64(), 0.0)
            } else {
                rng.next_c64()
            };
            a[(i, j)] = v;
            a[(j, i)] = v.conj();
        }
    }
    a
}

/// `A := H·A·H` for the Householder reflector `H = I - 2·w·wᴴ` (`‖w‖ = 1`).
fn reflect_both_sides(a: &mut Mat<Complex64>, w: &[Complex64]) {
    let n = w.len();
    // p = A·w, s = wᴴ·A·w (real for Hermitian A)
    let mut p = vec![Complex64::zero(); n];
    for (i, pi) in p.iter_mut().enumerate() {
        for (j, &wj) in w.iter().enumerate() {
            *pi += a[(i, j)] * wj;
        }
    }
    let s: Complex64 = w.iter().zip(&p).map(|(wi, pi)| wi.conj() * pi).sum();
    // H·A·H = A - 2·w·pᴴ - 2·p·wᴴ + 4·s·w·wᴴ
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = a[(i, j)] - 2.0 * w[i] * p[j].conj() - 2.0 * p[i] * w[j].conj()
                + 4.0 * s * w[i] * w[j].conj();
        }
    }
}

/// `U·diag(eigs)·Uᴴ` with `U` a product of three random complex reflectors:
/// Hermitian with exactly the spectrum `eigs` up to the rounding of the
/// products (`O(ε·max|eigs|)`).
fn hermitian_with_spectrum(eigs: &[f64], seed: u64) -> Mat<Complex64> {
    let n = eigs.len();
    let mut rng = Prng(seed);
    let mut a: Mat<Complex64> = Mat::zeros(n, n);
    for (i, &e) in eigs.iter().enumerate() {
        a[(i, i)] = Complex64::new(e, 0.0);
    }
    for _ in 0..3 {
        let mut w: Vec<Complex64> = (0..n).map(|_| rng.next_c64()).collect();
        let norm = w.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for z in w.iter_mut() {
            *z /= norm;
        }
        reflect_both_sides(&mut a, &w);
    }
    // Exactly Hermitian (the solver reads the upper triangle only).
    for j in 0..n {
        a[(j, j)] = Complex64::new(a[(j, j)].re, 0.0);
        for i in 0..j {
            a[(j, i)] = a[(i, j)].conj();
        }
    }
    a
}

/// Spectral norm of a Hermitian matrix from its eigenvalues.
fn spectral_norm(eigs: &[f64]) -> f64 {
    eigs.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

/// `max_k ‖A·v_k - λ_k·v_k‖₂ / ‖A‖₂` and `max |VᴴV - I|`.
fn residual_and_orthogonality(a: &Mat<Complex64>, evd: &HermitianEvdDc<Complex64>) -> (f64, f64) {
    pair_residual_and_orthogonality(a, evd.eigenvalues(), evd.eigenvectors())
}

/// [`residual_and_orthogonality`] of explicit eigenpairs `(lam, v)`.
fn pair_residual_and_orthogonality(
    a: &Mat<Complex64>,
    lam: &[f64],
    v: MatRef<'_, Complex64>,
) -> (f64, f64) {
    let n = a.nrows();
    let anorm = spectral_norm(lam).max(f64::MIN_POSITIVE);
    let mut resid = 0.0f64;
    for k in 0..n {
        let mut ssq = 0.0f64;
        for i in 0..n {
            let mut av = Complex64::zero();
            for j in 0..n {
                av += a[(i, j)] * v[(j, k)];
            }
            ssq += (av - v[(i, k)] * lam[k]).norm_sqr();
        }
        resid = resid.max(ssq.sqrt() / anorm);
    }
    let mut ortho = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut dot = Complex64::zero();
            for r in 0..n {
                dot += v[(r, i)].conj() * v[(r, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            ortho = ortho.max((dot - expected).norm());
        }
    }
    (resid, ortho)
}

/// Eigenvalues of `A` from the real symmetric divide and conquer on its real
/// embedding: the doubled spectrum, every second value.
fn embedding_eigenvalues(a: &Mat<Complex64>) -> Vec<f64> {
    let n = a.nrows();
    let mut m: Mat<f64> = Mat::zeros(2 * n, 2 * n);
    for j in 0..n {
        for i in 0..n {
            let z = a[(i, j)];
            m[(i, j)] = z.re;
            m[(n + i, n + j)] = z.re;
            m[(i, n + j)] = -z.im;
            m[(n + i, j)] = z.im;
        }
    }
    let evd = SymmetricEvdDc::compute(m.as_ref()).expect("real embedding EVD");
    let mu = evd.eigenvalues();
    // Each eigenvalue appears twice; the pair is equal to ~ε‖A‖.
    (0..n).map(|k| 0.5 * (mu[2 * k] + mu[2 * k + 1])).collect()
}

/// The full set of checks of one random Hermitian matrix of order `n`.
fn check_random(n: usize, seed: u64) {
    let a = random_hermitian(n, seed);
    let dc = HermitianEvdDc::compute(a.as_ref()).expect("HermitianEvdDc");
    let qr = HermitianEvd::compute(a.as_ref()).expect("HermitianEvd");
    let lam = dc.eigenvalues();
    assert_eq!(lam.len(), n);
    assert!(
        lam.iter().all(|x| x.is_finite()),
        "n={n}: non-finite eigenvalue"
    );
    assert!(lam.windows(2).all(|w| w[0] <= w[1]), "n={n}: not ascending");

    let anorm = spectral_norm(qr.eigenvalues());
    let tol = TOL_C * n as f64 * f64::EPSILON * anorm;
    let reference = embedding_eigenvalues(&a);
    for i in 0..n {
        let d_qr = (lam[i] - qr.eigenvalues()[i]).abs();
        let d_emb = (lam[i] - reference[i]).abs();
        assert!(
            d_qr <= tol && d_emb <= tol,
            "n={n}: eigenvalue {i}: dc={} qr={} embedding={} (tol {tol:e})",
            lam[i],
            qr.eigenvalues()[i],
            reference[i]
        );
    }
    let (resid, ortho) = residual_and_orthogonality(&a, &dc);
    let n_eps = TOL_C * n as f64 * f64::EPSILON;
    assert!(resid <= n_eps, "n={n}: residual {resid:e} (tol {n_eps:e})");
    assert!(
        ortho <= n_eps,
        "n={n}: orthogonality {ortho:e} (tol {n_eps:e})"
    );
}

#[test]
fn random_hermitian_n50() {
    check_random(50, 0x0050);
}

#[test]
fn random_hermitian_n101() {
    check_random(101, 0x0101);
}

#[test]
fn random_hermitian_n150() {
    check_random(150, 0x0150);
}

#[test]
fn random_hermitian_n256() {
    check_random(256, 0x0256);
}

#[test]
fn random_hermitian_n400() {
    check_random(400, 0x0400);
}

/// Checks `evd` of `a` against the prescribed (sorted) spectrum `eigs`.
fn check_against_spectrum(a: &Mat<Complex64>, eigs_sorted: &[f64], label: &str) {
    let n = a.nrows();
    let dc = HermitianEvdDc::compute(a.as_ref()).expect("HermitianEvdDc");
    let lam = dc.eigenvalues();
    let anorm = spectral_norm(eigs_sorted);
    let tol = TOL_C * n as f64 * f64::EPSILON * anorm;
    for i in 0..n {
        assert!(
            (lam[i] - eigs_sorted[i]).abs() <= tol,
            "{label}: eigenvalue {i}: {} vs {} (tol {tol:e})",
            lam[i],
            eigs_sorted[i]
        );
    }
    let (resid, ortho) = residual_and_orthogonality(a, &dc);
    let n_eps = TOL_C * n as f64 * f64::EPSILON;
    assert!(resid <= n_eps, "{label}: residual {resid:e}");
    assert!(ortho <= n_eps, "{label}: orthogonality {ortho:e}");
}

#[test]
fn clustered_spectrum() {
    // Three tight clusters (spacing 1e-13 and 1e-9) between isolated values.
    let mut eigs = Vec::new();
    for j in 0..60 {
        eigs.push(1.0 + j as f64 * 1e-13);
    }
    for j in 0..50 {
        eigs.push(-2.0 + j as f64 * 1e-9);
    }
    for j in 0..30 {
        eigs.push(4.0 + j as f64 * 1e-15);
    }
    for j in 0..20 {
        eigs.push(-5.0 + j as f64 * 0.37);
    }
    let a = hermitian_with_spectrum(&eigs, 0xC1C1);
    eigs.sort_by(f64::total_cmp);
    check_against_spectrum(&a, &eigs, "clustered");
}

#[test]
fn repeated_eigenvalues() {
    // Three eigenvalues of multiplicity 50 each (heavy type-2 deflation).
    let mut eigs = Vec::new();
    for &v in &[-2.0, 0.5, 3.0] {
        eigs.extend(std::iter::repeat_n(v, 50));
    }
    let a = hermitian_with_spectrum(&eigs, 0x5EED);
    eigs.sort_by(f64::total_cmp);
    check_against_spectrum(&a, &eigs, "repeated");
}

#[test]
fn diagonal_input_returns_the_sorted_diagonal() {
    let n = 130;
    let mut rng = Prng(0xD1A6);
    let mut a: Mat<Complex64> = Mat::zeros(n, n);
    let mut diag = Vec::with_capacity(n);
    for i in 0..n {
        // Some exact repeats among distinct values.
        let v = if i % 7 == 3 {
            0.25
        } else {
            rng.next_f64() * 10.0
        };
        a[(i, i)] = Complex64::new(v, 0.0);
        diag.push(v);
    }
    let dc = HermitianEvdDc::compute(a.as_ref()).expect("diagonal");
    diag.sort_by(f64::total_cmp);
    assert_eq!(
        dc.eigenvalues(),
        &diag[..],
        "eigenvalues of a diagonal matrix"
    );
    let (resid, ortho) = residual_and_orthogonality(&a, &dc);
    assert!(resid <= f64::EPSILON && ortho <= f64::EPSILON);
}

#[test]
fn real_symmetric_as_hermitian() {
    let n = 120;
    let mut rng = Prng(0x5E41);
    let mut a: Mat<Complex64> = Mat::zeros(n, n);
    let mut s: Mat<f64> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            let v = rng.next_f64();
            a[(i, j)] = Complex64::new(v, 0.0);
            a[(j, i)] = Complex64::new(v, 0.0);
            s[(i, j)] = v;
            s[(j, i)] = v;
        }
    }
    let dc = HermitianEvdDc::compute(a.as_ref()).expect("HermitianEvdDc");
    let real = SymmetricEvdDc::compute(s.as_ref()).expect("SymmetricEvdDc");
    let anorm = spectral_norm(real.eigenvalues());
    let tol = TOL_C * n as f64 * f64::EPSILON * anorm;
    for (i, (&x, &y)) in dc.eigenvalues().iter().zip(real.eigenvalues()).enumerate() {
        assert!((x - y).abs() <= tol, "eigenvalue {i}: {x} vs {y}");
    }
    let (resid, ortho) = residual_and_orthogonality(&a, &dc);
    let n_eps = TOL_C * n as f64 * f64::EPSILON;
    assert!(resid <= n_eps && ortho <= n_eps, "{resid:e} {ortho:e}");
}

#[test]
fn tiny_and_huge_scales() {
    let n = 110;
    let a = random_hermitian(n, 0x5CA1);
    let base = HermitianEvdDc::compute(a.as_ref()).expect("unscaled");
    for k in [500, -500] {
        let s = 2f64.powi(k);
        let mut b = a.clone();
        for j in 0..n {
            for i in 0..n {
                b[(i, j)] *= s;
            }
        }
        let dc = HermitianEvdDc::compute(b.as_ref()).expect("scaled");
        let lam = dc.eigenvalues();
        assert!(
            lam.iter().all(|x| x.is_finite() && *x != 0.0),
            "2^{k}: {lam:?}"
        );
        let anorm = spectral_norm(lam);
        let tol = TOL_C * n as f64 * f64::EPSILON * anorm;
        for (i, (&x, &y)) in lam.iter().zip(base.eigenvalues()).enumerate() {
            assert!(
                (x - y * s).abs() <= tol,
                "2^{k}: eigenvalue {i}: {x:e} vs {:e}",
                y * s
            );
        }
        let (resid, ortho) = residual_and_orthogonality(&b, &dc);
        let n_eps = TOL_C * n as f64 * f64::EPSILON;
        assert!(
            resid <= n_eps && ortho <= n_eps,
            "2^{k}: {resid:e} {ortho:e}"
        );
    }
}

#[test]
fn complex32_matches_reference() {
    let n = 130;
    let a64 = random_hermitian(n, 0xF32);
    let mut a: Mat<Complex32> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let z = a64[(i, j)];
            a[(i, j)] = Complex32::new(z.re as f32, z.im as f32);
        }
    }
    let dc = HermitianEvdDc::compute(a.as_ref()).expect("HermitianEvdDc<Complex32>");
    let qr = HermitianEvd::compute(a.as_ref()).expect("HermitianEvd<Complex32>");
    let lam = dc.eigenvalues();
    assert!(lam.iter().all(|x| x.is_finite()));
    let anorm = qr.eigenvalues().iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let n_eps = TOL_C as f32 * n as f32 * f32::EPSILON;
    for i in 0..n {
        assert!(
            (lam[i] - qr.eigenvalues()[i]).abs() <= n_eps * anorm,
            "eigenvalue {i}: {} vs {}",
            lam[i],
            qr.eigenvalues()[i]
        );
    }
    let v = dc.eigenvectors();
    let (mut resid, mut ortho) = (0.0f32, 0.0f32);
    for k in 0..n {
        let mut ssq = 0.0f32;
        for i in 0..n {
            let mut av = Complex32::zero();
            for j in 0..n {
                av += a[(i, j)] * v[(j, k)];
            }
            ssq += (av - v[(i, k)] * lam[k]).norm_sqr();
        }
        resid = resid.max(ssq.sqrt() / anorm);
        for j in 0..n {
            let mut dot = Complex32::zero();
            for r in 0..n {
                dot += v[(r, k)].conj() * v[(r, j)];
            }
            let expected = if k == j { 1.0 } else { 0.0 };
            ortho = ortho.max((dot - expected).norm());
        }
    }
    assert!(resid <= n_eps, "Complex32 residual {resid:e}");
    assert!(ortho <= n_eps, "Complex32 orthogonality {ortho:e}");
}

#[test]
fn non_finite_input_is_an_error() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    for n in [1usize, 2, 7, 150] {
        let last = n - 1;
        let cases = [
            (0, last, Complex64::new(nan, 0.0)),
            (0, last, Complex64::new(0.5, nan)),
            (last / 2, last / 2, Complex64::new(inf, 0.0)),
            (0, last, Complex64::new(-inf, 1.0)),
            (last, last, Complex64::new(0.0, inf)),
        ];
        for (i, j, val) in cases {
            let mut a = random_hermitian(n, 0xBAD + n as u64);
            a[(i, j)] = val;
            let r = HermitianEvdDc::compute(a.as_ref());
            let err = r.as_ref().err().copied();
            assert_eq!(
                err,
                Some(HermitianEvdDcError::NotConverged),
                "n={n}: {val} at ({i},{j}) gave {:?}",
                r.as_ref().map(|e| e.eigenvalues().to_vec())
            );
        }
    }
}

#[test]
fn strict_lower_triangle_and_diagonal_imaginary_parts_are_not_read() {
    // Only the upper triangle is read, and the diagonal is taken as real (as
    // LAPACK ZHEEVD does).
    let n = 120;
    let a = random_hermitian(n, 0x1077);
    let mut b = a.clone();
    b[(n - 1, 0)] = Complex64::new(f64::NAN, f64::NAN);
    b[(5, 2)] = Complex64::new(1e30, -7.0);
    for i in 0..n {
        b[(i, i)] = Complex64::new(a[(i, i)].re, 0.25 * i as f64);
    }
    let ea = HermitianEvdDc::compute(a.as_ref()).expect("a");
    let eb = HermitianEvdDc::compute(b.as_ref()).expect("b");
    assert_eq!(ea.eigenvalues(), eb.eigenvalues());
}

#[test]
fn small_orders_match_reference() {
    // Orders around the leaf size of the real divide and conquer.
    for (idx, &n) in [2usize, 3, 5, 24, 25, 26, 51].iter().enumerate() {
        let a = random_hermitian(n, 0x5A11 + idx as u64);
        let dc = HermitianEvdDc::compute(a.as_ref()).expect("dc");
        let qr = HermitianEvd::compute(a.as_ref()).expect("qr");
        let anorm = spectral_norm(qr.eigenvalues());
        let tol = TOL_C * n as f64 * f64::EPSILON * anorm;
        for (x, y) in dc.eigenvalues().iter().zip(qr.eigenvalues()) {
            assert!((x - y).abs() <= tol, "n={n}: {x} vs {y}");
        }
        let (resid, ortho) = residual_and_orthogonality(&a, &dc);
        let n_eps = TOL_C * n as f64 * f64::EPSILON;
        assert!(
            resid <= n_eps && ortho <= n_eps,
            "n={n}: {resid:e} {ortho:e}"
        );
    }
}

// -----------------------------------------------------------------------------
// Internal paths not reachable through `compute` with Complex64/Complex32
// -----------------------------------------------------------------------------

#[test]
fn qr_fallback_pipeline_is_correct() {
    // The generic fallback (QR on the real tridiagonal, for complex types
    // without a GEMM back end) exercised with Complex64.
    for (idx, &n) in [2usize, 9, 60, 130].iter().enumerate() {
        let a = random_hermitian(n, 0xFA11 + idx as u64);
        let mut work = hermitian_from_upper(a.as_ref());
        let mut lam = Vec::new();
        let mut v: Mat<Complex64> = Mat::zeros(n, n);
        qr_pipeline(&mut work, &mut lam, &mut v).expect("qr_pipeline");
        assert!(lam.windows(2).all(|w| w[0] <= w[1]), "n={n}: not ascending");
        let dc = HermitianEvdDc::compute(a.as_ref()).expect("dc");
        let anorm = spectral_norm(dc.eigenvalues());
        let tol = TOL_C * n as f64 * f64::EPSILON * anorm;
        for (x, y) in lam.iter().zip(dc.eigenvalues()) {
            assert!((x - y).abs() <= tol, "n={n}: {x} vs {y}");
        }
        let (resid, ortho) = pair_residual_and_orthogonality(&a, &lam, v.as_ref());
        let n_eps = TOL_C * n as f64 * f64::EPSILON;
        assert!(
            resid <= n_eps && ortho <= n_eps,
            "n={n}: {resid:e} {ortho:e}"
        );
    }
}

#[test]
fn plain_loop_backend_matches_gemm_backend() {
    for (idx, &n) in [7usize, 60, 140].iter().enumerate() {
        let a = random_hermitian(n, 0xB1A5 + idx as u64);
        let run = |naive: bool| {
            let mut work = hermitian_from_upper(a.as_ref());
            let mut lam = Vec::new();
            let mut v: Mat<Complex64> = Mat::zeros(n, n);
            let r = if naive {
                dc_pipeline::<Complex64, f64, NaiveMatMul>(&mut work, &mut lam, &mut v)
            } else {
                dc_pipeline::<Complex64, f64, BlasMatMul>(&mut work, &mut lam, &mut v)
            };
            r.expect("dc_pipeline");
            (lam, v)
        };
        let (l_gemm, v_gemm) = run(false);
        let (l_loop, v_loop) = run(true);
        let scale = spectral_norm(&l_gemm);
        for k in 0..n {
            assert!(
                (l_gemm[k] - l_loop[k]).abs() <= 1e-12 * scale,
                "n={n}: eigenvalue {k}: {} vs {}",
                l_gemm[k],
                l_loop[k]
            );
            // Same eigenvector up to a unit phase (the spectrum is simple).
            let mut dot = Complex64::zero();
            for r in 0..n {
                dot += v_gemm[(r, k)].conj() * v_loop[(r, k)];
            }
            assert!(
                (dot.norm() - 1.0).abs() <= 1e-10,
                "n={n}: vector {k}: |dot| = {}",
                dot.norm()
            );
        }
    }
}
