//! Eigenpair residual tests for the non-symmetric eigensolvers.
//!
//! Every test checks each computed eigenpair individually, using the residual
//! `‖A·v − λ·v‖₂ / ‖v‖₂ ≤ c·n·ε·‖A‖_F` (and its left-eigenvector analogue
//! `‖uᴴ·A − λ·uᴴ‖₂ / ‖u‖₂`), so an eigenvector paired with the wrong eigenvalue
//! fails no matter how accurate the eigenvalues are. A decomposition that
//! returns `Err` counts as a failure as well.
//!
//! Conventions under test:
//!
//! - `GeneralEvd` (real input): the eigenvector of `eigenvalues()[k]` is
//!   `eigenvectors_real()[:, k] + i·eigenvectors_imag()[:, k]`. A complex
//!   conjugate pair occupies two consecutive positions, the eigenvalue with the
//!   positive imaginary part first, and the second eigenvector is the conjugate
//!   of the first. Left eigenvectors are stored the same way and satisfy
//!   `uᴴ·A = λ·uᴴ`.
//! - `ComplexGeneralEvd`: column `k` of `eigenvectors()` (`left_eigenvectors()`)
//!   is the right (left) eigenvector of `eigenvalues()[k]`.

use num_complex::{Complex32, Complex64};
use oxiblas_lapack::evd::{
    ComplexGeneralEvd, ComplexHessenberg, ComplexSchur, GeneralEvd, Schur, trevc_left, trevc_right,
};
use oxiblas_matrix::{Mat, MatRef};

/// Residual constant `c` in `residual ≤ c·n·ε·‖A‖_F`.
const C_RESIDUAL: f64 = 20.0;

// =============================================================================
// Helpers
// =============================================================================

/// Deterministic generator (64-bit LCG) producing values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_real(n: usize, seed: u64) -> Mat<f64> {
    let mut rng = Lcg(seed);
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = rng.next_f64();
        }
    }
    a
}

fn random_complex(n: usize, seed: u64) -> Mat<Complex64> {
    let mut rng = Lcg(seed);
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = Complex64::new(rng.next_f64(), rng.next_f64());
        }
    }
    a
}

/// Frobenius norm accumulated with `hypot` (no underflow/overflow of squares).
fn frobenius_real(a: &Mat<f64>) -> f64 {
    let mut s = 0.0f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            s = s.hypot(a[(i, j)]);
        }
    }
    s
}

/// Frobenius norm accumulated with `hypot` (no underflow/overflow of squares).
fn frobenius_complex(a: &Mat<Complex64>) -> f64 {
    let mut s = 0.0f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            s = s.hypot(a[(i, j)].norm());
        }
    }
    s
}

fn vec_norm(v: &[Complex64]) -> f64 {
    v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// `‖A·v − λ·v‖₂ / ‖v‖₂` for a real matrix and a complex vector.
fn right_residual_real(a: &Mat<f64>, lambda: Complex64, v: &[Complex64]) -> f64 {
    let n = a.nrows();
    let mut r = 0.0;
    for i in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for (j, vj) in v.iter().enumerate() {
            s += *vj * a[(i, j)];
        }
        r += (s - lambda * v[i]).norm_sqr();
    }
    r.sqrt() / vec_norm(v)
}

/// `‖uᴴ·A − λ·uᴴ‖₂ / ‖u‖₂` for a real matrix and a complex vector.
fn left_residual_real(a: &Mat<f64>, lambda: Complex64, u: &[Complex64]) -> f64 {
    let n = a.nrows();
    let mut r = 0.0;
    for j in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for (i, ui) in u.iter().enumerate() {
            s += ui.conj() * a[(i, j)];
        }
        r += (s - lambda * u[j].conj()).norm_sqr();
    }
    r.sqrt() / vec_norm(u)
}

fn right_residual_complex(a: &Mat<Complex64>, lambda: Complex64, v: &[Complex64]) -> f64 {
    let n = a.nrows();
    let mut r = 0.0;
    for i in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for (j, vj) in v.iter().enumerate() {
            s += a[(i, j)] * *vj;
        }
        r += (s - lambda * v[i]).norm_sqr();
    }
    r.sqrt() / vec_norm(v)
}

fn left_residual_complex(a: &Mat<Complex64>, lambda: Complex64, u: &[Complex64]) -> f64 {
    let n = a.nrows();
    let mut r = 0.0;
    for j in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for (i, ui) in u.iter().enumerate() {
            s += ui.conj() * a[(i, j)];
        }
        r += (s - lambda * u[j].conj()).norm_sqr();
    }
    r.sqrt() / vec_norm(u)
}

/// Column `k` of a split (real part, imaginary part) eigenvector store.
fn split_column(
    re: &oxiblas_matrix::MatRef<'_, f64>,
    im: &oxiblas_matrix::MatRef<'_, f64>,
    k: usize,
) -> Vec<Complex64> {
    (0..re.nrows())
        .map(|i| Complex64::new(re[(i, k)], im[(i, k)]))
        .collect()
}

/// Checks the layout invariants of a `GeneralEvd` eigenvalue list: real
/// eigenvalues have a zero imaginary part, and each complex conjugate pair is
/// adjacent with the positive imaginary part first.
fn assert_pair_layout(label: &str, evd: &GeneralEvd<f64>) {
    let ev = evd.eigenvalues();
    let mut k = 0;
    while k < ev.len() {
        if ev[k].imag == 0.0 {
            k += 1;
            continue;
        }
        assert!(
            k + 1 < ev.len(),
            "{label}: complex eigenvalue {k} has no partner"
        );
        assert!(
            ev[k].imag > 0.0,
            "{label}: pair at {k} must list +imag first, got {:?}",
            ev[k]
        );
        assert_eq!(
            ev[k + 1].real,
            ev[k].real,
            "{label}: pair at {k} real parts"
        );
        assert_eq!(
            ev[k + 1].imag,
            -ev[k].imag,
            "{label}: pair at {k} imag parts"
        );
        k += 2;
    }
}

/// Full right-eigenpair check of `GeneralEvd::compute` on a real matrix.
/// Returns the worst residual ratio `residual / (n·ε·‖A‖_F)`.
fn check_general_right(label: &str, a: &Mat<f64>) -> f64 {
    let n = a.nrows();
    let eps = f64::EPSILON;
    let anorm = frobenius_real(a).max(f64::MIN_POSITIVE);
    let tol = C_RESIDUAL * n as f64 * eps * anorm;
    let evd = match GeneralEvd::compute(a.as_ref()) {
        Ok(evd) => evd,
        Err(e) => panic!("{label}: GeneralEvd::compute failed: {e:?}"),
    };
    assert_eq!(evd.eigenvalues().len(), n, "{label}: eigenvalue count");
    assert_pair_layout(label, &evd);
    let (Some(vr), Some(vi)) = (evd.eigenvectors_real(), evd.eigenvectors_imag()) else {
        panic!("{label}: eigenvectors missing");
    };
    let mut worst = 0.0f64;
    for (k, lam) in evd.eigenvalues().iter().enumerate() {
        let lambda = Complex64::new(lam.real, lam.imag);
        let v = split_column(&vr, &vi, k);
        let norm = vec_norm(&v);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "{label}: eigenvector {k} must have unit norm, got {norm}"
        );
        if lam.imag == 0.0 {
            assert!(
                v.iter().all(|z| z.im == 0.0),
                "{label}: eigenvector {k} of a real eigenvalue must be real"
            );
        }
        if lam.imag < 0.0 {
            let prev = split_column(&vr, &vi, k - 1);
            for (x, y) in v.iter().zip(prev.iter()) {
                assert_eq!(
                    *x,
                    y.conj(),
                    "{label}: eigenvector {k} must be conj of {}",
                    k - 1
                );
            }
        }
        let res = right_residual_real(a, lambda, &v);
        assert!(
            res <= tol,
            "{label}: right eigenpair {k} (λ = {lambda}) residual {res:e} > {tol:e}"
        );
        worst = worst.max(res / (n as f64 * eps * anorm));
    }
    let verify = evd.verify(a.as_ref());
    assert!(verify <= tol, "{label}: verify() = {verify:e} > {tol:e}");
    worst
}

/// Full left-eigenpair check of `GeneralEvd::compute_full` on a real matrix.
fn check_general_left(label: &str, a: &Mat<f64>) -> f64 {
    let n = a.nrows();
    let eps = f64::EPSILON;
    let anorm = frobenius_real(a).max(f64::MIN_POSITIVE);
    let tol = C_RESIDUAL * n as f64 * eps * anorm;
    let evd = match GeneralEvd::compute_full(a.as_ref()) {
        Ok(evd) => evd,
        Err(e) => panic!("{label}: GeneralEvd::compute_full failed: {e:?}"),
    };
    assert_pair_layout(label, &evd);
    let (Some(ur), Some(ui)) = (evd.left_eigenvectors_real(), evd.left_eigenvectors_imag()) else {
        panic!("{label}: left eigenvectors missing");
    };
    let (Some(vr), Some(vi)) = (evd.eigenvectors_real(), evd.eigenvectors_imag()) else {
        panic!("{label}: right eigenvectors missing");
    };
    let mut worst = 0.0f64;
    for (k, lam) in evd.eigenvalues().iter().enumerate() {
        let lambda = Complex64::new(lam.real, lam.imag);
        let u = split_column(&ur, &ui, k);
        let norm = vec_norm(&u);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "{label}: left eigenvector {k} must have unit norm, got {norm}"
        );
        let res = left_residual_real(a, lambda, &u);
        assert!(
            res <= tol,
            "{label}: left eigenpair {k} (λ = {lambda}) residual {res:e} > {tol:e}"
        );
        worst = worst.max(res / (n as f64 * eps * anorm));
        // compute_full must return the same right eigenpairs as compute.
        let v = split_column(&vr, &vi, k);
        let res_r = right_residual_real(a, lambda, &v);
        assert!(
            res_r <= tol,
            "{label}: compute_full right eigenpair {k} residual {res_r:e} > {tol:e}"
        );
    }
    worst
}

fn check_general(label: &str, a: &Mat<f64>) {
    check_general_right(label, a);
    check_general_left(label, a);
}

/// Full check of `ComplexGeneralEvd::compute_full` (right and left pairs).
fn check_complex(label: &str, a: &Mat<Complex64>) -> f64 {
    let n = a.nrows();
    let eps = f64::EPSILON;
    let anorm = frobenius_complex(a).max(f64::MIN_POSITIVE);
    let tol = C_RESIDUAL * n as f64 * eps * anorm;
    let evd = match ComplexGeneralEvd::compute_full(a.as_ref()) {
        Ok(evd) => evd,
        Err(e) => panic!("{label}: ComplexGeneralEvd::compute_full failed: {e:?}"),
    };
    assert_eq!(evd.eigenvalues().len(), n, "{label}: eigenvalue count");
    let (Some(v), Some(u)) = (evd.eigenvectors(), evd.left_eigenvectors()) else {
        panic!("{label}: eigenvectors missing");
    };
    let mut worst = 0.0f64;
    for (k, &lambda) in evd.eigenvalues().iter().enumerate() {
        let vk: Vec<Complex64> = (0..n).map(|i| v[(i, k)]).collect();
        let uk: Vec<Complex64> = (0..n).map(|i| u[(i, k)]).collect();
        for (name, x) in [("right", &vk), ("left", &uk)] {
            let norm = vec_norm(x);
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "{label}: {name} eigenvector {k} must have unit norm, got {norm}"
            );
        }
        let res_r = right_residual_complex(a, lambda, &vk);
        let res_l = left_residual_complex(a, lambda, &uk);
        assert!(
            res_r <= tol,
            "{label}: right eigenpair {k} (λ = {lambda}) residual {res_r:e} > {tol:e}"
        );
        assert!(
            res_l <= tol,
            "{label}: left eigenpair {k} (λ = {lambda}) residual {res_l:e} > {tol:e}"
        );
        worst = worst.max(res_r.max(res_l) / (n as f64 * eps * anorm));
    }
    // `compute` must agree with `compute_full` on the right eigenpairs.
    let right_only = match ComplexGeneralEvd::compute(a.as_ref()) {
        Ok(evd) => evd,
        Err(e) => panic!("{label}: ComplexGeneralEvd::compute failed: {e:?}"),
    };
    let verify = right_only.verify(a.as_ref());
    assert!(verify <= tol, "{label}: verify() = {verify:e} > {tol:e}");
    worst
}

/// Matches two eigenvalue multisets greedily and returns the largest distance.
fn spectrum_distance(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut used = vec![false; b.len()];
    let mut worst = 0.0f64;
    for x in a {
        let mut best = f64::INFINITY;
        let mut best_j = 0;
        for (j, y) in b.iter().enumerate() {
            if !used[j] && (x - y).norm() < best {
                best = (x - y).norm();
                best_j = j;
            }
        }
        used[best_j] = true;
        worst = worst.max(best);
    }
    worst
}

/// Companion matrix of the monic polynomial `x^n + c[n-1]·x^(n-1) + … + c[0]`.
fn companion(c: &[f64]) -> Mat<f64> {
    let n = c.len();
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        a[(0, j)] = -c[n - 1 - j];
    }
    for i in 1..n {
        a[(i, i - 1)] = 1.0;
    }
    a
}

// =============================================================================
// GeneralEvd: right eigenpairs
// =============================================================================

#[test]
fn general_evd_small_examples_pair_correctly() {
    let rot = core::f64::consts::FRAC_PI_4;
    let cases: Vec<(&str, Mat<f64>)> = vec![
        ("[[2,1],[1,3]]", Mat::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]])),
        ("[[1,2],[3,4]]", Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]])),
        ("diag(2,3)", Mat::from_rows(&[&[2.0, 0.0], &[0.0, 3.0]])),
        ("diag(3,2)", Mat::from_rows(&[&[3.0, 0.0], &[0.0, 2.0]])),
        (
            "[[0,-1],[1,0]]",
            Mat::from_rows(&[&[0.0, -1.0], &[1.0, 0.0]]),
        ),
        (
            "[[1,2],[-3,1]]",
            Mat::from_rows(&[&[1.0, 2.0], &[-3.0, 1.0]]),
        ),
        (
            "rotation(pi/4)",
            Mat::from_rows(&[&[rot.cos(), -rot.sin()], &[rot.sin(), rot.cos()]]),
        ),
        ("[[0,1],[0,0]]", Mat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]])),
        ("[[0,0],[1,0]]", Mat::from_rows(&[&[0.0, 0.0], &[1.0, 0.0]])),
        ("zero 2x2", Mat::from_rows(&[&[0.0, 0.0], &[0.0, 0.0]])),
        ("[[7]]", Mat::from_rows(&[&[7.0]])),
        (
            "3x3 general",
            Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]),
        ),
        (
            "3x3 symmetric",
            Mat::from_rows(&[&[4.0, 1.0, 2.0], &[1.0, 3.0, 0.5], &[2.0, 0.5, 5.0]]),
        ),
        (
            "3x3 complex pair",
            Mat::from_rows(&[&[1.0, -2.0, 0.5], &[3.0, 1.0, 0.25], &[0.0, 0.1, 2.0]]),
        ),
        (
            "slow 5x5",
            Mat::from_rows(&[
                &[5.0, 1.0, 0.2, 0.0, 0.1],
                &[0.3, 5.0, 1.0, 0.15, 0.0],
                &[0.0, 0.25, 5.0, 1.0, 0.2],
                &[0.1, 0.0, 0.3, 5.0, 1.0],
                &[0.2, 0.1, 0.0, 0.35, 5.0],
            ]),
        ),
    ];
    for (label, a) in &cases {
        check_general(label, a);
    }
}

#[test]
fn general_evd_diag_2_3_pairs_e1_with_2() {
    // The reported symptom: λ = 3 used to be paired with e1.
    let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 3.0]]);
    let evd = GeneralEvd::compute(a.as_ref()).expect("evd");
    let Some(vr) = evd.eigenvectors_real() else {
        panic!("eigenvectors missing");
    };
    for (k, lam) in evd.eigenvalues().iter().enumerate() {
        let expected_axis = if lam.real == 2.0 { 0 } else { 1 };
        assert!(
            (vr[(expected_axis, k)].abs() - 1.0).abs() < 1e-15,
            "λ = {} must be paired with e{}",
            lam.real,
            expected_axis + 1
        );
    }
}

#[test]
fn general_evd_random_matrices() {
    let mut sizes: Vec<usize> = (2..=24).collect();
    sizes.extend_from_slice(&[25, 31, 32, 40, 47, 50, 64, 80, 100]);
    let mut worst = 0.0f64;
    for &n in &sizes {
        let seeds: &[u64] = if n <= 24 { &[1, 2, 3] } else { &[1] };
        for &seed in seeds {
            let a = random_real(n, 7919 * seed + n as u64);
            worst = worst.max(check_general_right(
                &format!("random n={n} seed={seed}"),
                &a,
            ));
        }
    }
    eprintln!("general_evd_random_matrices: worst residual / (n·ε·‖A‖) = {worst:.3}");
}

#[test]
fn general_evd_random_left_eigenvectors() {
    let mut sizes: Vec<usize> = (2..=16).collect();
    sizes.extend_from_slice(&[25, 40, 64]);
    for &n in &sizes {
        let a = random_real(n, 104_729 + n as u64);
        check_general_left(&format!("random left n={n}"), &a);
    }
}

#[test]
fn general_evd_complex_conjugate_pairs() {
    // Block diagonal of rotation-scaling blocks [[a, b], [-b, a]] (eigenvalues
    // a ± ib) and real 1×1 blocks, hidden by an orthogonal similarity, so the
    // spectrum is known exactly.
    for (n, seed) in [(4usize, 11u64), (5, 12), (8, 13), (11, 14), (20, 15)] {
        let mut d = Mat::zeros(n, n);
        let mut expected = Vec::new();
        let mut rng = Lcg(seed);
        let mut k = 0;
        while k < n {
            if k + 1 < n {
                let re = rng.next_f64() * 3.0;
                let im = 0.5 + rng.next_f64().abs() * 2.0;
                d[(k, k)] = re;
                d[(k, k + 1)] = im;
                d[(k + 1, k)] = -im;
                d[(k + 1, k + 1)] = re;
                expected.push(Complex64::new(re, im));
                expected.push(Complex64::new(re, -im));
                k += 2;
            } else {
                let re = rng.next_f64() * 3.0;
                d[(k, k)] = re;
                expected.push(Complex64::new(re, 0.0));
                k += 1;
            }
        }
        // Orthogonal Q from a Householder reflector product.
        let q = random_orthogonal(n, seed + 100);
        let a = similarity(&q, &d);
        let label = format!("complex pairs n={n}");
        check_general(&label, &a);
        let evd = GeneralEvd::eigenvalues_only(a.as_ref()).expect("evd");
        let got: Vec<Complex64> = evd
            .eigenvalues()
            .iter()
            .map(|e| Complex64::new(e.real, e.imag))
            .collect();
        let dist = spectrum_distance(&got, &expected);
        assert!(dist < 1e-12, "{label}: spectrum off by {dist:e}");
    }
}

/// Product of `n` random Householder reflectors.
fn random_orthogonal(n: usize, seed: u64) -> Mat<f64> {
    let mut q = Mat::zeros(n, n);
    for i in 0..n {
        q[(i, i)] = 1.0;
    }
    let mut rng = Lcg(seed);
    for _ in 0..n {
        let v: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let vv: f64 = v.iter().map(|x| x * x).sum();
        if vv == 0.0 {
            continue;
        }
        // q := q · (I − 2·v·vᵀ / vᵀv)
        for i in 0..n {
            let dot: f64 = (0..n).map(|j| q[(i, j)] * v[j]).sum();
            for (j, vj) in v.iter().enumerate() {
                q[(i, j)] -= 2.0 * dot * vj / vv;
            }
        }
    }
    q
}

/// `Q · D · Qᵀ`.
fn similarity(q: &Mat<f64>, d: &Mat<f64>) -> Mat<f64> {
    let n = q.nrows();
    let mut qd = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += q[(i, k)] * d[(k, j)];
            }
            qd[(i, j)] = s;
        }
    }
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += qd[(i, k)] * q[(j, k)];
            }
            a[(i, j)] = s;
        }
    }
    a
}

#[test]
fn general_evd_diagonal_and_triangular() {
    let mut cases: Vec<(String, Mat<f64>)> = vec![
        (
            "diag(2,5,3)".into(),
            Mat::from_rows(&[&[2.0, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]),
        ),
        (
            "diag with repeats".into(),
            Mat::from_rows(&[
                &[1.0, 0.0, 0.0, 0.0],
                &[0.0, 2.0, 0.0, 0.0],
                &[0.0, 0.0, 2.0, 0.0],
                &[0.0, 0.0, 0.0, -3.0],
            ]),
        ),
        (
            "upper triangular 3x3".into(),
            Mat::from_rows(&[&[1.0, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]),
        ),
        (
            "lower triangular 4x4".into(),
            Mat::from_rows(&[
                &[1.0, 0.0, 0.0, 0.0],
                &[2.0, 3.0, 0.0, 0.0],
                &[4.0, 5.0, 6.0, 0.0],
                &[7.0, 8.0, 9.0, 10.0],
            ]),
        ),
    ];
    for n in [6usize, 12, 30] {
        let mut rng = Lcg(n as u64 + 500);
        let mut up = Mat::zeros(n, n);
        let mut lo = Mat::zeros(n, n);
        let mut dg = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                up[(i, j)] = rng.next_f64();
                lo[(j, i)] = rng.next_f64();
            }
            dg[(i, i)] = rng.next_f64() * 10.0;
        }
        cases.push((format!("random upper triangular n={n}"), up));
        cases.push((format!("random lower triangular n={n}"), lo));
        cases.push((format!("random diagonal n={n}"), dg));
    }
    for (label, a) in &cases {
        check_general(label, a);
    }
}

#[test]
fn general_evd_companion_matrices() {
    let mut cases: Vec<(String, Mat<f64>)> = vec![
        ("(x-1)(x-2)(x-3)".into(), companion(&[-6.0, 11.0, -6.0])),
        ("x^2+1".into(), companion(&[1.0, 0.0])),
        ("x^4-1".into(), companion(&[-1.0, 0.0, 0.0, 0.0])),
        ("x^5+1".into(), companion(&[1.0, 0.0, 0.0, 0.0, 0.0])),
    ];
    for n in [3usize, 6, 7, 10, 16] {
        // x^n - 1: a cyclic permutation matrix, the classic stall case of
        // unshifted / standard-shift QR without exceptional shifts.
        let mut c = vec![0.0; n];
        c[0] = -1.0;
        cases.push((format!("x^{n}-1"), companion(&c)));
    }
    for (n, seed) in [(4usize, 1u64), (8, 2), (12, 3), (20, 4)] {
        let mut rng = Lcg(seed + 900);
        let c: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        cases.push((format!("random companion n={n}"), companion(&c)));
    }
    for (label, a) in &cases {
        check_general(label, a);
    }
}

#[test]
fn general_evd_defective_matrices() {
    let mut cases: Vec<(String, Mat<f64>)> = vec![
        (
            "jordan 2".into(),
            Mat::from_rows(&[&[2.0, 1.0], &[0.0, 2.0]]),
        ),
        (
            "jordan 3".into(),
            Mat::from_rows(&[&[2.0, 1.0, 0.0], &[0.0, 2.0, 1.0], &[0.0, 0.0, 2.0]]),
        ),
        (
            "near-defective 2x2".into(),
            Mat::from_rows(&[&[1.0, 1.0], &[1e-14, 1.0]]),
        ),
    ];
    // Constant diagonal with ones everywhere above it (n = 30): back-substitution
    // without rescaling overflows through the perturbed near-zero pivots.
    for n in [10usize, 30] {
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 3.0;
            for j in (i + 1)..n {
                a[(i, j)] = 1.0;
            }
        }
        cases.push((format!("jordan-like upper n={n}"), a));
        let mut j = Mat::zeros(n, n);
        for i in 0..n {
            j[(i, i)] = -1.5;
            if i + 1 < n {
                j[(i, i + 1)] = 1.0;
            }
        }
        cases.push((format!("jordan block n={n}"), j.clone()));
        let q = random_orthogonal(n, n as u64);
        cases.push((format!("hidden jordan block n={n}"), similarity(&q, &j)));
    }
    for (label, a) in &cases {
        check_general(label, a);
    }
}

#[test]
fn general_evd_scaled_matrices() {
    for scale in [1e-150, 1e-12, 1e12, 1e150] {
        for n in [2usize, 5, 9] {
            let mut a = random_real(n, 31 * n as u64);
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] *= scale;
                }
            }
            check_general(&format!("scaled {scale:e} n={n}"), &a);
        }
    }
}

#[test]
fn general_evd_f32_residuals() {
    let eps = f32::EPSILON;
    for n in [2usize, 3, 5, 8, 13, 20] {
        let a64 = random_real(n, 271 + n as u64);
        let mut a = Mat::<f32>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = a64[(i, j)] as f32;
            }
        }
        let mut anorm = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                anorm += a[(i, j)] * a[(i, j)];
            }
        }
        let anorm = anorm.sqrt();
        let evd = match GeneralEvd::compute(a.as_ref()) {
            Ok(evd) => evd,
            Err(e) => panic!("f32 n={n}: GeneralEvd::compute failed: {e:?}"),
        };
        let (Some(vr), Some(vi)) = (evd.eigenvectors_real(), evd.eigenvectors_imag()) else {
            panic!("f32 n={n}: eigenvectors missing");
        };
        let tol = C_RESIDUAL as f32 * n as f32 * eps * anorm;
        for (k, lam) in evd.eigenvalues().iter().enumerate() {
            let lambda = Complex32::new(lam.real, lam.imag);
            let v: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new(vr[(i, k)], vi[(i, k)]))
                .collect();
            let vnorm = v.iter().map(|z| z.norm_sqr()).sum::<f32>().sqrt();
            let mut r = 0.0f32;
            for i in 0..n {
                let mut s = Complex32::new(0.0, 0.0);
                for (j, vj) in v.iter().enumerate() {
                    s += *vj * a[(i, j)];
                }
                r += (s - lambda * v[i]).norm_sqr();
            }
            let res = r.sqrt() / vnorm;
            assert!(
                res <= tol,
                "f32 n={n}: eigenpair {k} residual {res:e} > {tol:e}"
            );
        }
        let verify = evd.verify(a.as_ref());
        assert!(verify <= tol, "f32 n={n}: verify() = {verify:e} > {tol:e}");
    }
}

// =============================================================================
// Real Schur form
// =============================================================================

#[test]
fn schur_is_canonical_and_ordered() {
    let mut cases: Vec<(String, Mat<f64>)> = vec![
        (
            "diag(2,3)".into(),
            Mat::from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]),
        ),
        (
            "[[2,1],[1,3]]".into(),
            Mat::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]),
        ),
        (
            "[[1,2],[3,4]]".into(),
            Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]),
        ),
        (
            "[[1,2],[-3,1]]".into(),
            Mat::from_rows(&[&[1.0, 2.0], &[-3.0, 1.0]]),
        ),
        ("x^4-1".into(), companion(&[-1.0, 0.0, 0.0, 0.0])),
    ];
    for n in [3usize, 5, 8, 16, 33, 64] {
        cases.push((format!("random n={n}"), random_real(n, 77 + n as u64)));
    }
    for (label, a) in &cases {
        let n = a.nrows();
        let eps = f64::EPSILON;
        let anorm = frobenius_real(a).max(f64::MIN_POSITIVE);
        let schur = match Schur::compute(a.as_ref()) {
            Ok(s) => s,
            Err(e) => panic!("{label}: Schur::compute failed: {e:?}"),
        };
        let (q, t) = (schur.q(), schur.t());
        // Reconstruction and orthogonality.
        let rec = schur.reconstruct();
        let mut diff = 0.0;
        let mut orth = 0.0;
        for i in 0..n {
            for j in 0..n {
                diff += (rec[(i, j)] - a[(i, j)]).powi(2);
                let mut qtq = 0.0;
                for k in 0..n {
                    qtq += q[(k, i)] * q[(k, j)];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                orth += (qtq - target).powi(2);
            }
        }
        assert!(
            diff.sqrt() <= C_RESIDUAL * n as f64 * eps * anorm,
            "{label}: ‖A − QTQᵀ‖ = {:e}",
            diff.sqrt()
        );
        assert!(
            orth.sqrt() <= C_RESIDUAL * n as f64 * eps,
            "{label}: ‖QᵀQ − I‖ = {:e}",
            orth.sqrt()
        );
        // Canonical quasi-triangular form, eigenvalues in diagonal order.
        let ev = schur.eigenvalues();
        assert_eq!(ev.len(), n, "{label}: eigenvalue count");
        for i in 0..n {
            for j in 0..i.saturating_sub(1) {
                assert_eq!(t[(i, j)], 0.0, "{label}: T[{i},{j}] below the subdiagonal");
            }
        }
        let mut k = 0;
        while k < n {
            if k + 1 < n && t[(k + 1, k)] != 0.0 {
                assert!(
                    k + 2 >= n || t[(k + 2, k + 1)] == 0.0,
                    "{label}: 2×2 blocks at {k} and {} overlap",
                    k + 1
                );
                assert_eq!(
                    t[(k, k)],
                    t[(k + 1, k + 1)],
                    "{label}: block {k} not standardized"
                );
                assert!(
                    t[(k, k + 1)] * t[(k + 1, k)] < 0.0,
                    "{label}: block {k} off-diagonals must have opposite signs"
                );
                let im = t[(k, k + 1)].abs().sqrt() * t[(k + 1, k)].abs().sqrt();
                assert_eq!(ev[k].real, t[(k, k)], "{label}: eigenvalue {k} real part");
                assert!(
                    (ev[k].imag - im).abs() <= 4.0 * eps * im,
                    "{label}: eigenvalue {k} imag part {} vs {im}",
                    ev[k].imag
                );
                assert_eq!(ev[k + 1].real, ev[k].real, "{label}: eigenvalue {}", k + 1);
                assert_eq!(ev[k + 1].imag, -ev[k].imag, "{label}: eigenvalue {}", k + 1);
                k += 2;
            } else {
                assert_eq!(
                    ev[k].real,
                    t[(k, k)],
                    "{label}: eigenvalue {k} vs T[{k},{k}]"
                );
                assert_eq!(ev[k].imag, 0.0, "{label}: eigenvalue {k} must be real");
                k += 1;
            }
        }
    }
}

// =============================================================================
// ComplexGeneralEvd
// =============================================================================

#[test]
fn complex_general_evd_small_examples() {
    let c = Complex64::new;
    let cases: Vec<(&str, Mat<Complex64>)> = vec![
        (
            "2x2 general",
            Mat::from_rows(&[&[c(1.0, 2.0), c(3.0, 1.0)], &[c(-1.0, 1.0), c(4.0, -1.0)]]),
        ),
        (
            "diag(2,3)",
            Mat::from_rows(&[&[c(2.0, 0.0), c(0.0, 0.0)], &[c(0.0, 0.0), c(3.0, 0.0)]]),
        ),
        (
            "jordan 2",
            Mat::from_rows(&[&[c(1.0, 1.0), c(1.0, 0.0)], &[c(0.0, 0.0), c(1.0, 1.0)]]),
        ),
        (
            "hermitian 2x2",
            Mat::from_rows(&[&[c(2.0, 0.0), c(1.0, -1.0)], &[c(1.0, 1.0), c(3.0, 0.0)]]),
        ),
        (
            "upper triangular 3x3",
            Mat::from_rows(&[
                &[c(1.0, 0.0), c(2.0, 1.0), c(3.0, 0.0)],
                &[c(0.0, 0.0), c(4.0, 0.0), c(5.0, -1.0)],
                &[c(0.0, 0.0), c(0.0, 0.0), c(6.0, 0.0)],
            ]),
        ),
        ("[[5+2i]]", Mat::from_rows(&[&[c(5.0, 2.0)]])),
    ];
    for (label, a) in &cases {
        check_complex(label, a);
    }
}

#[test]
fn complex_general_evd_random_matrices() {
    let mut sizes: Vec<usize> = (2..=16).collect();
    sizes.extend_from_slice(&[20, 25, 32, 50, 64]);
    let mut worst = 0.0f64;
    for &n in &sizes {
        let a = random_complex(n, 6007 + n as u64);
        worst = worst.max(check_complex(&format!("complex random n={n}"), &a));
    }
    eprintln!("complex_general_evd_random_matrices: worst ratio = {worst:.3}");
}

#[test]
fn complex_general_evd_eigenvalues_are_correct() {
    // Eigenvalues of every constructor must match the complex Schur form, and
    // for a real input also the real Schur form.
    for n in [3usize, 4, 6, 9, 17, 33] {
        let a = random_complex(n, 3001 + n as u64);
        let anorm = frobenius_complex(&a);
        let schur = ComplexSchur::compute(a.as_ref()).expect("schur");
        let reference = schur.eigenvalues().to_vec();
        let only = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).expect("evd");
        let full = ComplexGeneralEvd::compute(a.as_ref()).expect("evd");
        for (name, got) in [
            ("eigenvalues_only", only.eigenvalues()),
            ("compute", full.eigenvalues()),
        ] {
            let dist = spectrum_distance(got, &reference);
            assert!(
                dist <= 1e-10 * anorm,
                "n={n}: {name} eigenvalues differ from ComplexSchur by {dist:e}"
            );
        }
        let ar = random_real(n, 5003 + n as u64);
        let mut ac = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                ac[(i, j)] = Complex64::new(ar[(i, j)], 0.0);
            }
        }
        let real_evd = GeneralEvd::eigenvalues_only(ar.as_ref()).expect("real evd");
        let real_spec: Vec<Complex64> = real_evd
            .eigenvalues()
            .iter()
            .map(|e| Complex64::new(e.real, e.imag))
            .collect();
        let cplx = ComplexGeneralEvd::eigenvalues_only(ac.as_ref()).expect("evd");
        let dist = spectrum_distance(cplx.eigenvalues(), &real_spec);
        assert!(
            dist <= 1e-10 * frobenius_real(&ar),
            "n={n}: complex vs real spectrum differ by {dist:e}"
        );
    }
}

#[test]
fn complex_general_evd_structured_matrices() {
    let mut cases: Vec<(String, Mat<Complex64>)> = Vec::new();
    for n in [5usize, 12, 30] {
        let mut jordan = Mat::zeros(n, n);
        let mut upper = Mat::zeros(n, n);
        let mut diag = Mat::zeros(n, n);
        let mut rng = Lcg(n as u64 + 4242);
        for i in 0..n {
            jordan[(i, i)] = Complex64::new(0.5, -1.0);
            diag[(i, i)] = Complex64::new(rng.next_f64() * 4.0, rng.next_f64());
            for j in i..n {
                upper[(i, j)] = Complex64::new(rng.next_f64(), rng.next_f64());
                if j > i {
                    jordan[(i, j)] = Complex64::new(1.0, 0.0);
                }
            }
        }
        cases.push((format!("complex jordan-like n={n}"), jordan));
        cases.push((format!("complex upper triangular n={n}"), upper));
        cases.push((format!("complex diagonal n={n}"), diag));
        // Cyclic permutation (x^n − 1 companion).
        let mut cyc = Mat::zeros(n, n);
        for i in 0..n {
            cyc[((i + 1) % n, i)] = Complex64::new(1.0, 0.0);
        }
        cases.push((format!("complex cyclic permutation n={n}"), cyc));
    }
    for (label, a) in &cases {
        check_complex(label, a);
    }
}

#[test]
fn complex_general_evd_scaled_matrices() {
    for scale in [1e-150, 1e-12, 1e12, 1e150] {
        for n in [2usize, 4, 5, 8, 17] {
            let mut a = random_complex(n, 911 * n as u64);
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] *= scale;
                }
            }
            check_complex(&format!("complex scaled {scale:e} n={n}"), &a);
        }
    }
}

// =============================================================================
// Public trevc_right / trevc_left and Schur::eigenvectors
// =============================================================================

fn to_owned_real(m: MatRef<'_, f64>) -> Mat<f64> {
    let mut out = Mat::zeros(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out[(i, j)] = m[(i, j)];
        }
    }
    out
}

/// Expands trevc-layout columns (real column, or (Re, Im) column pair starting
/// at each index of `pair_starts`) into one complex vector per eigenvalue
/// position; the second member of a pair gets the conjugate vector.
fn trevc_vectors(v: &Mat<f64>, pair_starts: &[usize]) -> Vec<Vec<Complex64>> {
    let n = v.nrows();
    let mut out = Vec::with_capacity(n);
    let mut k = 0;
    while k < n {
        if pair_starts.contains(&k) {
            let x: Vec<Complex64> = (0..n)
                .map(|i| Complex64::new(v[(i, k)], v[(i, k + 1)]))
                .collect();
            let xc: Vec<Complex64> = x.iter().map(|z| z.conj()).collect();
            out.push(x);
            out.push(xc);
            k += 2;
        } else {
            out.push((0..n).map(|i| Complex64::new(v[(i, k)], 0.0)).collect());
            k += 1;
        }
    }
    out
}

/// `(xᴴ·T·x) / (xᴴ·x)`.
fn rayleigh(t: &Mat<f64>, x: &[Complex64]) -> Complex64 {
    let n = t.nrows();
    let mut num = Complex64::new(0.0, 0.0);
    for i in 0..n {
        let mut tx = Complex64::new(0.0, 0.0);
        for (j, xj) in x.iter().enumerate() {
            tx += *xj * t[(i, j)];
        }
        num += x[i].conj() * tx;
    }
    num / x.iter().map(|z| z.norm_sqr()).sum::<f64>()
}

/// Checks `trevc_right` / `trevc_left` on a quasi-triangular `t` whose complex
/// pairs start at `pair_starts`: every right column (pair) is an eigenvector
/// of unit norm, the left column (pair) of the same index is a left
/// eigenvector for the *same* eigenvalue, the first member of a pair has a
/// positive imaginary part, and the eigenvalues found match `spectrum` (and
/// `per_index` position by position, when given).
fn check_trevc(
    label: &str,
    t: &Mat<f64>,
    pair_starts: &[usize],
    spectrum: &[Complex64],
    per_index: Option<&[Complex64]>,
) {
    let n = t.nrows();
    let eps = f64::EPSILON;
    let tnorm = frobenius_real(t);
    let tol = C_RESIDUAL * n as f64 * eps * tnorm;
    let rights = trevc_vectors(&trevc_right(t), pair_starts);
    let lefts = trevc_vectors(&trevc_left(t), pair_starts);
    let mut found = Vec::with_capacity(n);
    for k in 0..n {
        let (x, y) = (&rights[k], &lefts[k]);
        for (name, v) in [("right", x), ("left", y)] {
            let norm = vec_norm(v);
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "{label}: {name} vector {k} must have unit norm, got {norm}"
            );
        }
        let lambda = rayleigh(t, x);
        if pair_starts.contains(&k) {
            assert!(
                lambda.im > 0.0,
                "{label}: pair at {k} must hold the +imag eigenvector, got λ = {lambda}"
            );
        }
        if let Some(expected) = per_index {
            assert!(
                (lambda - expected[k]).norm() <= 1e-10 * tnorm,
                "{label}: column {k} has λ = {lambda}, expected {}",
                expected[k]
            );
        }
        let res_r = right_residual_real(t, lambda, x);
        let res_l = left_residual_real(t, lambda, y);
        assert!(
            res_r <= tol,
            "{label}: right column {k} (λ = {lambda}) residual {res_r:e} > {tol:e}"
        );
        assert!(
            res_l <= tol,
            "{label}: left column {k} (λ = {lambda}) residual {res_l:e} > {tol:e}"
        );
        found.push(lambda);
    }
    let dist = spectrum_distance(&found, spectrum);
    assert!(dist <= 1e-10 * tnorm, "{label}: spectrum off by {dist:e}");
}

#[test]
fn trevc_on_schur_factors_pairs_each_column_with_its_eigenvalue() {
    for (n, seed) in [(5usize, 1u64), (8, 2), (13, 3), (30, 4)] {
        let a = random_real(n, 4243 * seed + n as u64);
        let schur = Schur::compute(a.as_ref()).expect("schur");
        let t = to_owned_real(schur.t());
        let pair_starts: Vec<usize> = (0..n.saturating_sub(1))
            .filter(|&k| t[(k + 1, k)] != 0.0)
            .collect();
        assert!(
            !pair_starts.is_empty(),
            "n={n}: test matrix should have complex eigenvalues"
        );
        let ev: Vec<Complex64> = schur
            .eigenvalues()
            .iter()
            .map(|e| Complex64::new(e.real, e.imag))
            .collect();
        check_trevc(&format!("schur T n={n}"), &t, &pair_starts, &ev, Some(&ev));
    }
}

#[test]
fn trevc_on_hand_built_quasi_triangular_factors() {
    let s6 = 6f64.sqrt();
    // Real 1×1 above a standardized complex block: the back-substitution of
    // row 0 must use λ = 1 + i·√6, not its conjugate.
    let t1 = Mat::from_rows(&[&[2.0, 1.0, 0.5], &[0.0, 1.0, 3.0], &[0.0, -2.0, 1.0]]);
    let ev1 = [
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, s6),
        Complex64::new(1.0, -s6),
    ];
    check_trevc("real above complex", &t1, &[1], &ev1, Some(&ev1));

    // Complex block above a real 1×1 (forward substitution for the left vectors).
    let t2 = Mat::from_rows(&[&[1.0, 3.0, 0.5], &[-2.0, 1.0, -1.0], &[0.0, 0.0, 4.0]]);
    let ev2 = [
        Complex64::new(1.0, s6),
        Complex64::new(1.0, -s6),
        Complex64::new(4.0, 0.0),
    ];
    check_trevc("complex above real", &t2, &[0], &ev2, Some(&ev2));

    // Non-standardized complex block (unequal diagonal): standardized
    // internally, eigenvectors rotated back.
    let s23 = 23f64.sqrt() / 2.0;
    let t3 = Mat::from_rows(&[&[2.0, 1.0, 0.5], &[0.0, 1.0, 3.0], &[0.0, -2.0, 2.0]]);
    let ev3 = [
        Complex64::new(2.0, 0.0),
        Complex64::new(1.5, s23),
        Complex64::new(1.5, -s23),
    ];
    check_trevc("non-standardized block", &t3, &[1], &ev3, Some(&ev3));

    // A 2×2 block with real eigenvalues (non-zero subdiagonal) is split into
    // two real eigenvectors.
    let s13 = 13f64.sqrt();
    let t4 = Mat::from_rows(&[&[2.0, 1.0, 0.5], &[0.0, 1.0, 3.0], &[0.0, 1.0, 2.0]]);
    let ev4 = [
        Complex64::new(2.0, 0.0),
        Complex64::new((3.0 + s13) / 2.0, 0.0),
        Complex64::new((3.0 - s13) / 2.0, 0.0),
    ];
    check_trevc("real-eigenvalue block", &t4, &[], &ev4, None);

    // Two complex blocks with real eigenvalues in between and dense coupling.
    let mut t5 = Mat::zeros(7, 7);
    let mut rng = Lcg(77);
    for i in 0..7 {
        for j in i..7 {
            t5[(i, j)] = rng.next_f64();
        }
    }
    // Blocks [[a, b], [c, a]] with b·c < 0 at (0, 1) and (4, 5).
    for &(k, a, b, c) in &[(0usize, 0.3, 2.0, -0.5), (4, -1.2, -0.7, 1.5)] {
        t5[(k, k)] = a;
        t5[(k + 1, k + 1)] = a;
        t5[(k, k + 1)] = b;
        t5[(k + 1, k)] = c;
    }
    let mut ev5 = Vec::new();
    let mut k = 0;
    while k < 7 {
        if k == 0 || k == 4 {
            let im = (t5[(k, k + 1)] * t5[(k + 1, k)]).abs().sqrt();
            ev5.push(Complex64::new(t5[(k, k)], im));
            ev5.push(Complex64::new(t5[(k, k)], -im));
            k += 2;
        } else {
            ev5.push(Complex64::new(t5[(k, k)], 0.0));
            k += 1;
        }
    }
    check_trevc("two blocks", &t5, &[0, 4], &ev5, Some(&ev5));
}

#[test]
fn schur_eigenvectors_are_eigenvectors_of_a() {
    for (n, seed) in [(4usize, 9u64), (9, 10), (17, 11)] {
        let a = random_real(n, 7777 * seed + n as u64);
        let anorm = frobenius_real(&a);
        let tol = C_RESIDUAL * n as f64 * f64::EPSILON * anorm;
        let schur = Schur::compute(a.as_ref()).expect("schur");
        let t = schur.t();
        let pair_starts: Vec<usize> = (0..n - 1).filter(|&k| t[(k + 1, k)] != 0.0).collect();
        let (vr, vl) = schur.eigenvectors();
        let rights = trevc_vectors(&vr, &pair_starts);
        let lefts = trevc_vectors(&vl, &pair_starts);
        for (k, e) in schur.eigenvalues().iter().enumerate() {
            let lambda = Complex64::new(e.real, e.imag);
            let res_r = right_residual_real(&a, lambda, &rights[k]);
            let res_l = left_residual_real(&a, lambda, &lefts[k]);
            assert!(
                res_r <= tol && res_l <= tol,
                "n={n}: eigenpair {k} (λ = {lambda}) residuals {res_r:e} / {res_l:e} > {tol:e}"
            );
        }
    }
}

// =============================================================================
// Scale invariance of the complex Hessenberg / Schur building blocks
// =============================================================================

#[test]
fn complex_hessenberg_is_a_similarity_at_small_scale() {
    for scale in [1.0, 1e-20, 1e-200] {
        let n = 12;
        let mut a = random_complex(n, 2027);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] *= scale;
            }
        }
        let anorm = frobenius_complex(&a);
        let hess = ComplexHessenberg::compute(a.as_ref()).expect("hessenberg");
        let (q, h) = (hess.q(), hess.h());
        let tol = C_RESIDUAL * n as f64 * f64::EPSILON;
        let mut diff = 0.0;
        let mut orth = 0.0;
        for i in 0..n {
            for j in 0..n {
                // (Q·H·Qᴴ)[i, j]
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    for l in 0..n {
                        s += q[(i, k)] * h[(k, l)] * q[(j, l)].conj();
                    }
                }
                diff += ((s - a[(i, j)]) / anorm).norm_sqr();
                let mut qhq = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    qhq += q[(k, i)].conj() * q[(k, j)];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                orth += (qhq - target).norm_sqr();
                if i > j + 1 {
                    assert!(
                        h[(i, j)].norm() <= f64::EPSILON * anorm,
                        "scale {scale:e}: H[{i},{j}] = {} is not negligible",
                        h[(i, j)]
                    );
                }
            }
        }
        assert!(
            diff.sqrt() <= tol,
            "scale {scale:e}: ‖A − QHQᴴ‖/‖A‖ = {:e}",
            diff.sqrt()
        );
        assert!(
            orth.sqrt() <= tol,
            "scale {scale:e}: ‖QᴴQ − I‖ = {:e}",
            orth.sqrt()
        );
    }
}

#[test]
fn complex_schur_residual_is_relative_at_every_scale() {
    for n in [3usize, 4, 8, 16] {
        let a = random_complex(n, 1009 + n as u64);
        let base = ComplexSchur::compute(a.as_ref()).expect("schur");
        let r0 = base.residual(a.as_ref());
        assert!(
            r0 > 0.0 && r0 <= C_RESIDUAL * n as f64 * f64::EPSILON,
            "n={n}: unscaled residual {r0:e}"
        );
        // Exact power-of-two scalings (≈ 1e-12, 1e-150, 1e12): the relative
        // residual must stay at the same ε level.
        for k in [-40i32, -500, 40] {
            let s = 2f64.powi(k);
            let mut b = Mat::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    b[(i, j)] = a[(i, j)] * s;
                }
            }
            let schur = match ComplexSchur::compute(b.as_ref()) {
                Ok(s) => s,
                Err(e) => panic!("n={n}, scale 2^{k}: ComplexSchur failed: {e:?}"),
            };
            let r = schur.residual(b.as_ref());
            assert!(
                r >= 0.25 * r0 && r <= 4.0 * r0,
                "n={n}, scale 2^{k}: relative residual {r:e} vs unscaled {r0:e}"
            );
        }
    }
}
