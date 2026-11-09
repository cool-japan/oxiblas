//! Accuracy tests of the divide-and-conquer SVD front-ends ([`SvdDc`],
//! [`ComplexSvdDc`]) on the cases found by review: extreme scales (B1), zero
//! diagonal entries in the bidiagonal (B2), graded and clustered spectra
//! (B3), plus special and non-finite inputs.
//!
//! Tolerances are `c·n·ε` (normwise: relative to `‖A‖`), the accuracy of a
//! backward-stable SVD. On a dense matrix no backward-stable algorithm can
//! do better than an absolute `O(ε·‖A‖)` per singular value (the matrix
//! itself carries rounding of that size), so per-σ comparisons with the
//! one-sided Jacobi `Svd` are normwise; high *relative* accuracy is checked
//! on the bidiagonal of the graded matrix, where it is achievable.

use num_complex::Complex64;
use oxiblas_matrix::{Mat, MatRef};

use super::bidiag_dc::bidiagonal_svd_dc;
use super::bidiag_qr::lasdq;
use super::blocked_bidiag::gebrd_tall;
use super::dc_kernels::BlasOps;
use crate::svd::{ComplexSvdDc, ComplexSvdDcError, Svd, SvdDc, SvdDcError};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

/// `A = H·diag(σ)·Gᵀ` (`m × n`, `σ` descending) with `H`, `G` products of
/// three random Householder reflections.
fn with_sv(m: usize, n: usize, sigma: &[f64], seed: u64) -> Mat<f64> {
    let mut a = Mat::<f64>::zeros(m, n);
    for (i, &s) in sigma.iter().enumerate() {
        a[(i, i)] = s;
    }
    let mut state = seed;
    let mut unit = |len: usize| {
        let mut v: Vec<f64> = (0..len).map(|_| lcg(&mut state)).collect();
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= nrm;
        }
        v
    };
    for _ in 0..3 {
        let v = unit(m);
        for j in 0..n {
            let dotp: f64 = (0..m).map(|i| v[i] * a[(i, j)]).sum();
            for i in 0..m {
                a[(i, j)] -= 2.0 * dotp * v[i];
            }
        }
        let w = unit(n);
        for i in 0..m {
            let dotp: f64 = (0..n).map(|j| a[(i, j)] * w[j]).sum();
            for j in 0..n {
                a[(i, j)] -= 2.0 * dotp * w[j];
            }
        }
    }
    a
}

fn random(m: usize, n: usize, seed: u64) -> Mat<f64> {
    let mut state = seed;
    let mut a = Mat::<f64>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            a[(i, j)] = lcg(&mut state);
        }
    }
    a
}

fn eps_tol(n: usize) -> f64 {
    16.0 * n as f64 * f64::EPSILON
}

/// `max |XᵀX − I|` for the columns of `x` (if `cols`) or its rows.
fn orth_error(x: MatRef<'_, f64>, cols: bool) -> f64 {
    let k = if cols { x.ncols() } else { x.nrows() };
    let len = if cols { x.nrows() } else { x.ncols() };
    let at = |r: usize, c: usize| if cols { x[(r, c)] } else { x[(c, r)] };
    let mut worst = 0.0f64;
    for a in 0..k {
        for b in a..k {
            let mut g = 0.0;
            for r in 0..len {
                g += at(r, a) * at(r, b);
            }
            worst = worst.max((g - if a == b { 1.0 } else { 0.0 }).abs());
        }
    }
    worst
}

/// Accuracy of an SVD of `a`, all relative to `max |a_ij|` so that it is
/// meaningful at any scale: `(‖A − UΣVᵀ‖_max / ‖A‖_max, UᵀU, VVᵀ errors)`.
fn check(a: &Mat<f64>, svd: &SvdDc<f64>) -> (f64, f64, f64) {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    let (u, s, vt) = (svd.u(), svd.singular_values(), svd.vt());
    assert_eq!(s.len(), k);
    assert_eq!((u.nrows(), u.ncols()), (m, m));
    assert_eq!((vt.nrows(), vt.ncols()), (n, n));
    for i in 1..k {
        assert!(
            s[i] <= s[i - 1] && s[i] >= 0.0,
            "not sorted/non-negative at {i}"
        );
    }
    let amax = (0..n)
        .flat_map(|j| (0..m).map(move |i| (i, j)))
        .fold(0.0f64, |acc, (i, j)| acc.max(a[(i, j)].abs()));
    let scale = if amax > 0.0 { 1.0 / amax } else { 1.0 };
    let mut resid = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut x = 0.0;
            for l in 0..k {
                x += u[(i, l)] * (s[l] * scale) * vt[(l, j)];
            }
            resid = resid.max((x - a[(i, j)] * scale).abs());
        }
    }
    (resid, orth_error(u, true), orth_error(vt, false))
}

fn assert_accurate(what: &str, a: &Mat<f64>, svd: &SvdDc<f64>) {
    let n = a.nrows().max(a.ncols());
    let (r, ou, ov) = check(a, svd);
    assert!(r < eps_tol(n), "{what}: residual {r:.3e}");
    assert!(ou < eps_tol(n), "{what}: UtU error {ou:.3e}");
    assert!(ov < eps_tol(n), "{what}: VtV error {ov:.3e}");
}

/// B1: `σ ∈ [1, 2]·f` at extreme scales, through the QR path (`n = 20`) and
/// the divide and conquer (`n = 40, 100`). Each σ is accurate to `c·n·ε`
/// relative, and the factorisation is exact and orthogonal.
#[test]
fn b1_extreme_scales() {
    for &n in &[20usize, 40, 100] {
        let base: Vec<f64> = (0..n).map(|i| 1.0 + (n - i) as f64 / n as f64).collect();
        for &f in &[1e-150f64, 1e-8, 1.0, 1e150] {
            let st: Vec<f64> = base.iter().map(|x| x * f).collect();
            let a = with_sv(n, n, &st, 99);
            let svd = SvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("n={n} f={f:e}: {e:?}"));
            let s = svd.singular_values();
            for i in 0..n {
                let rel = (s[i] - st[i]).abs() / st[i];
                assert!(rel < eps_tol(n), "n={n} f={f:e}: sigma[{i}] rel {rel:.3e}");
            }
            assert_accurate(&format!("n={n} f={f:e}"), &a, &svd);
        }
    }
}

/// B1 beyond the safe range (`1e±170`) and for tall / wide shapes.
#[test]
fn b1_scales_outside_safe_range_and_rectangular() {
    for &(m, n) in &[(30usize, 20usize), (20, 30), (60, 45), (45, 60)] {
        let k = m.min(n);
        let base: Vec<f64> = (0..k).map(|i| 1.0 + (k - i) as f64 / k as f64).collect();
        for &f in &[1e-300f64, 1e-170, 1e170, 1e300] {
            let st: Vec<f64> = base.iter().map(|x| x * f).collect();
            let a = with_sv(m, n, &st, 7);
            let svd =
                SvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("{m}x{n} f={f:e}: {e:?}"));
            let s = svd.singular_values();
            for i in 0..k {
                let rel = (s[i] - st[i]).abs() / st[i];
                assert!(
                    rel < eps_tol(m.max(n)),
                    "{m}x{n} f={f:e}: sigma[{i}] rel {rel:.3e}"
                );
            }
            assert_accurate(&format!("{m}x{n} f={f:e}"), &a, &svd);
        }
    }
}

/// Exact power-of-two scaling commutes with the SVD (bitwise).
#[test]
fn b1_power_of_two_scaling_is_exact() {
    let a = random(50, 40, 0x51);
    let base = SvdDc::compute(a.as_ref()).expect("svd");
    for &k in &[-900i32, -500, 500, 900] {
        let f = 2f64.powi(k);
        let mut b = a.clone();
        for j in 0..40 {
            for i in 0..50 {
                b[(i, j)] *= f;
            }
        }
        let svd = SvdDc::compute(b.as_ref()).expect("svd");
        for (x, y) in svd.singular_values().iter().zip(base.singular_values()) {
            assert_eq!(*x, *y * f, "k={k}");
        }
        for j in 0..50 {
            for i in 0..50 {
                assert_eq!(svd.u()[(i, j)], base.u()[(i, j)], "k={k} U");
            }
        }
    }
}

/// B2: matrices whose bidiagonal has exact zeros on the diagonal (zero
/// first column, zero rows or columns in the middle and at the end; square,
/// tall and wide; QR and divide-and-conquer paths).
#[test]
fn b2_zero_rows_and_columns() {
    // The reviewer's repros.
    let a = Mat::from_rows(&[&[0.0f64, 1.0], &[0.0, 1.0]]);
    let svd = SvdDc::compute(a.as_ref()).expect("2x2 zero column");
    assert!((svd.singular_values()[0] - 2f64.sqrt()).abs() < 4.0 * f64::EPSILON);
    assert_eq!(svd.singular_values()[1], 0.0);
    assert_accurate("2x2", &a, &svd);
    let a = Mat::from_rows(&[&[0.0f64, 1.0, 2.0], &[0.0, 3.0, 4.0], &[0.0, 5.0, 6.0]]);
    let svd = SvdDc::compute(a.as_ref()).expect("3x3 zero column");
    assert_eq!(svd.singular_values()[2], 0.0);
    assert_accurate("3x3", &a, &svd);

    for &n in &[3usize, 20, 25, 26, 40] {
        for &(dm, dn) in &[(0usize, 0usize), (7, 0), (0, 7)] {
            let (m, nc) = (n + dm, n + dn);
            let positions = [0, nc / 2, nc - 1];
            for &pos in &positions {
                // Zero column.
                let mut a = random(m, nc, 0x2b ^ (n * 31 + pos) as u64);
                for i in 0..m {
                    a[(i, pos)] = 0.0;
                }
                let svd = SvdDc::compute(a.as_ref())
                    .unwrap_or_else(|e| panic!("{m}x{nc} zero col {pos}: {e:?}"));
                assert_accurate(&format!("{m}x{nc} zero col {pos}"), &a, &svd);
                if m >= nc {
                    let s = svd.singular_values();
                    assert!(
                        s[nc - 1] < eps_tol(m) * s[0],
                        "{m}x{nc} col {pos}: smin {}",
                        s[nc - 1]
                    );
                }
                // Zero row.
                let rpos = (pos * m / nc).min(m - 1);
                let mut b = random(m, nc, 0x3c ^ (n * 17 + pos) as u64);
                for j in 0..nc {
                    b[(rpos, j)] = 0.0;
                }
                let svd = SvdDc::compute(b.as_ref())
                    .unwrap_or_else(|e| panic!("{m}x{nc} zero row {rpos}: {e:?}"));
                assert_accurate(&format!("{m}x{nc} zero row {rpos}"), &b, &svd);
            }
        }
    }
}

/// The singular values of the reviewer's 20×20 zero-first-column matrix
/// (QR path) include an exact zero and agree with Jacobi.
#[test]
fn b2_zero_column_matches_jacobi() {
    for &n in &[20usize, 40] {
        let mut a = random(n, n, 77);
        for i in 0..n {
            a[(i, 0)] = 0.0;
        }
        let dc = SvdDc::compute(a.as_ref()).expect("svd");
        let jac = Svd::compute(a.as_ref()).expect("jacobi");
        let (s, r) = (dc.singular_values(), jac.singular_values());
        for i in 0..n {
            assert!((s[i] - r[i]).abs() < eps_tol(n) * r[0], "n={n} sigma[{i}]");
        }
        if n <= 25 {
            assert_eq!(s[n - 1], 0.0, "QR path keeps the exact zero");
        }
        assert_accurate(&format!("zero column n={n}"), &a, &dc);
    }
}

/// Per-σ normwise agreement with the one-sided Jacobi SVD of the same
/// matrix: `|σ_dc − σ_jac| ≤ c·n·ε·σ_max` for every σ.
fn assert_matches_jacobi(what: &str, a: &Mat<f64>, dc: &SvdDc<f64>) {
    let n = a.nrows().max(a.ncols());
    let jac = Svd::compute(a.as_ref()).expect("jacobi");
    let (s, r) = (dc.singular_values(), jac.singular_values());
    for i in 0..s.len() {
        assert!(
            (s[i] - r[i]).abs() <= eps_tol(n) * r[0],
            "{what}: sigma[{i}] dc {:.17e} jacobi {:.17e}",
            s[i],
            r[i]
        );
    }
}

/// B3: graded spectrum `σᵢ = 10^(−12i/59)`, `n = 60`. On the dense matrix
/// every σ agrees with Jacobi to `c·n·ε·σ_max` (the previous `BᵀB` solver
/// was off by 19% at σ ≈ 5e-8) with an `ε`-level residual; on its
/// bidiagonal the divide and conquer has high relative accuracy (checked
/// against the relatively accurate QR iteration).
#[test]
fn b3_graded_spectrum() {
    let n = 60;
    let geo: Vec<f64> = (0..n)
        .map(|i| 10f64.powf(-12.0 * i as f64 / (n - 1) as f64))
        .collect();
    let a = with_sv(n, n, &geo, 11);
    let dc = SvdDc::compute(a.as_ref()).expect("svd");
    assert_accurate("graded", &a, &dc);
    assert_matches_jacobi("graded", &a, &dc);
    for i in 0..n {
        // The prescribed values themselves: normwise as well.
        assert!(
            (dc.singular_values()[i] - geo[i]).abs() < eps_tol(n),
            "graded sigma[{i}]"
        );
    }

    // Relative accuracy on the bidiagonal of the same matrix.
    let mut work = a.clone();
    let f = gebrd_tall::<f64, BlasOps>(BlasOps::SEQ, &mut work);
    let (_, s_dc, _) = bidiagonal_svd_dc(&f.d, &f.e).expect("bidiagonal d&c");
    let mut dq = f.d.clone();
    let mut eq = f.e.clone();
    let mut uq = Mat::<f64>::zeros(n, n);
    let mut vq = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        uq[(i, i)] = 1.0;
        vq[(i, i)] = 1.0;
    }
    lasdq(0, &mut dq, &mut eq, &mut uq, &mut vq).expect("bidiagonal qr");
    for k in 0..n {
        let reference = dq[n - 1 - k];
        let rel = (s_dc[k] - reference).abs() / reference;
        assert!(
            rel < 8.0 * n as f64 * f64::EPSILON,
            "bidiagonal sigma[{k}] rel {rel:.3e}"
        );
    }
}

/// B3: half the spectrum O(1), half clustered at `1e-8` (the previous
/// solver was off by 5% inside the cluster), and other cluster levels.
#[test]
fn b3_clusters() {
    let n = 60;
    for &lvl in &[1e-4f64, 1e-8, 1e-12] {
        let mut s: Vec<f64> = (0..n)
            .map(|i| {
                if i < n / 2 {
                    1.0 + 0.01 * (n / 2 - i) as f64
                } else {
                    lvl * (1.0 + (n - i) as f64 / n as f64)
                }
            })
            .collect();
        s.sort_by(|x, y| y.total_cmp(x));
        let a = with_sv(n, n, &s, 13);
        let dc = SvdDc::compute(a.as_ref()).expect("svd");
        assert_accurate(&format!("cluster {lvl:e}"), &a, &dc);
        assert_matches_jacobi(&format!("cluster {lvl:e}"), &a, &dc);
    }
}

/// B3 in `f32`: graded spectrum with `κ = 1e3` (the previous solver had a
/// residual of 3e-4). Everything at the `c·n·ε_f32` level.
#[test]
fn b3_f32_kappa_1e3() {
    let n = 60;
    for &kappa in &[1e1f64, 1e3, 1e4] {
        let sig: Vec<f64> = (0..n)
            .map(|i| kappa.powf(-(i as f64) / (n - 1) as f64))
            .collect();
        let a64 = with_sv(n, n, &sig, 21);
        let mut a = Mat::<f32>::zeros(n, n);
        let mut ar = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                a[(i, j)] = a64[(i, j)] as f32;
                ar[(i, j)] = a[(i, j)] as f64;
            }
        }
        let dc = SvdDc::compute(a.as_ref()).expect("f32 svd");
        let reference = Svd::compute(ar.as_ref()).expect("f64 jacobi");
        let tol = 16.0 * n as f64 * f32::EPSILON as f64;
        let (u, s, vt) = (dc.u(), dc.singular_values(), dc.vt());
        let r = reference.singular_values();
        for i in 0..n {
            assert!(
                (s[i] as f64 - r[i]).abs() < tol * r[0],
                "kappa={kappa:e}: sigma[{i}] {} vs {}",
                s[i],
                r[i]
            );
        }
        let mut resid = 0.0f64;
        let mut ou = 0.0f64;
        let mut ov = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut x = 0.0f64;
                let mut gu = 0.0f64;
                let mut gv = 0.0f64;
                for l in 0..n {
                    x += u[(i, l)] as f64 * s[l] as f64 * vt[(l, j)] as f64;
                    gu += u[(l, i)] as f64 * u[(l, j)] as f64;
                    gv += vt[(i, l)] as f64 * vt[(j, l)] as f64;
                }
                let id = if i == j { 1.0 } else { 0.0 };
                resid = resid.max((x - ar[(i, j)]).abs());
                ou = ou.max((gu - id).abs());
                ov = ov.max((gv - id).abs());
            }
        }
        assert!(resid < tol * r[0], "kappa={kappa:e}: residual {resid:.3e}");
        assert!(
            ou < tol && ov < tol,
            "kappa={kappa:e}: orthogonality {ou:.3e} {ov:.3e}"
        );
    }
}

/// Identity, zero, diagonal and rank-deficient matrices.
#[test]
fn special_matrices() {
    for &n in &[5usize, 30, 64] {
        let mut eye = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            eye[(i, i)] = 1.0;
        }
        let svd = SvdDc::compute(eye.as_ref()).expect("identity");
        assert!(
            svd.singular_values().iter().all(|&s| s == 1.0),
            "identity n={n}"
        );
        assert_accurate("identity", &eye, &svd);

        let z = Mat::<f64>::zeros(n + 3, n);
        let svd = SvdDc::compute(z.as_ref()).expect("zero");
        assert!(
            svd.singular_values().iter().all(|&s| s == 0.0),
            "zero n={n}"
        );
        let (_, ou, ov) = check(&z, &svd);
        assert!(ou < eps_tol(n) && ov < eps_tol(n), "zero n={n}: {ou} {ov}");

        // Diagonal with mixed signs and a zero: σ = sorted |d| exactly.
        let mut dg = Mat::<f64>::zeros(n, n);
        let mut expect: Vec<f64> = Vec::new();
        for i in 0..n {
            let v = (i as f64 - (n / 3) as f64) * 0.75;
            dg[(i, i)] = v;
            expect.push(v.abs());
        }
        expect.sort_by(|a, b| b.total_cmp(a));
        let svd = SvdDc::compute(dg.as_ref()).expect("diagonal");
        assert_eq!(svd.singular_values(), &expect[..], "diagonal n={n}");
        assert_accurate("diagonal", &dg, &svd);

        // Rank 3 (outer products): the trailing σ are at the ε·‖A‖ level.
        let x = random(n + 5, 3, 0x77);
        let y = random(3, n, 0x78);
        let mut lr = Mat::<f64>::zeros(n + 5, n);
        oxiblas_blas::level3::gemm::gemm(1.0, x.as_ref(), y.as_ref(), 0.0, lr.as_mut());
        let svd = SvdDc::compute(lr.as_ref()).expect("rank 3");
        let s = svd.singular_values();
        if n > 3 {
            assert!(s[2] > 1e-3 * s[0]);
            for (k, &sk) in s.iter().enumerate().skip(3) {
                assert!(sk < eps_tol(n) * s[0], "rank 3 n={n}: sigma[{k}] = {sk:e}");
            }
        }
        assert_accurate("rank 3", &lr, &svd);
    }
}

/// Non-finite input is rejected with `NotConverged` (documented), for both
/// paths and both shapes; no finite SVD exists.
#[test]
fn non_finite_input_is_an_error() {
    for &(m, n) in &[(3usize, 3usize), (40, 30), (30, 40)] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut a = random(m, n, 5);
            a[(m / 2, n / 3)] = bad;
            assert_eq!(
                SvdDc::compute(a.as_ref()).map(|_| ()),
                Err(SvdDcError::NotConverged),
                "{m}x{n} {bad}"
            );
            let mut c: Mat<Complex64> = Mat::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    c[(i, j)] = Complex64::new(a[(i, j)], 0.5);
                }
            }
            assert_eq!(
                ComplexSvdDc::compute(c.as_ref()).map(|_| ()),
                Err(ComplexSvdDcError::NotConverged),
                "complex {m}x{n} {bad}"
            );
        }
    }
    // Finite but huge entries are fine (scaled into range).
    let mut a = random(30, 30, 9);
    a[(3, 4)] = f64::MAX;
    let svd = SvdDc::compute(a.as_ref()).expect("f64::MAX entry");
    assert!(svd.singular_values()[0].is_finite());
}

/// Accuracy of a complex SVD relative to `max |a_ij|`.
fn complex_check(a: &Mat<Complex64>, svd: &ComplexSvdDc<Complex64>) -> (f64, f64, f64) {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    let (u, s, vh) = (svd.u(), svd.singular_values(), svd.vh());
    let amax = (0..n)
        .flat_map(|j| (0..m).map(move |i| (i, j)))
        .fold(0.0f64, |acc, (i, j)| acc.max(a[(i, j)].norm()));
    let scale = if amax > 0.0 { 1.0 / amax } else { 1.0 };
    let mut resid = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut x = Complex64::new(0.0, 0.0);
            for l in 0..k {
                x += u[(i, l)] * (s[l] * scale) * vh[(l, j)];
            }
            resid = resid.max((x - a[(i, j)] * scale).norm());
        }
    }
    let mut ou = 0.0f64;
    for p in 0..m {
        for q in 0..m {
            let mut g = Complex64::new(0.0, 0.0);
            for r in 0..m {
                g += u[(r, p)].conj() * u[(r, q)];
            }
            ou = ou.max((g - if p == q { 1.0 } else { 0.0 }).norm());
        }
    }
    let mut ov = 0.0f64;
    for p in 0..n {
        for q in 0..n {
            let mut g = Complex64::new(0.0, 0.0);
            for r in 0..n {
                g += vh[(p, r)] * vh[(q, r)].conj();
            }
            ov = ov.max((g - if p == q { 1.0 } else { 0.0 }).norm());
        }
    }
    (resid, ou, ov)
}

/// The complex front-end on the B1 scales and the zero-column repro.
#[test]
fn complex_scales_and_zero_column() {
    for &n in &[20usize, 40] {
        let mut state = 0xc0_u64 ^ n as u64;
        let mut base: Mat<Complex64> = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                base[(i, j)] = Complex64::new(lcg(&mut state), lcg(&mut state));
            }
        }
        let reference = ComplexSvdDc::compute(base.as_ref()).expect("complex svd");
        for &f in &[1e-150f64, 1e-8, 1.0, 1e150] {
            let mut a = base.clone();
            for j in 0..n {
                for i in 0..n {
                    a[(i, j)] *= f;
                }
            }
            let svd = ComplexSvdDc::compute(a.as_ref())
                .unwrap_or_else(|e| panic!("complex n={n} f={f:e}: {e:?}"));
            let (r, ou, ov) = complex_check(&a, &svd);
            assert!(r < eps_tol(n), "complex n={n} f={f:e}: residual {r:.3e}");
            assert!(
                ou < eps_tol(n) && ov < eps_tol(n),
                "complex n={n} f={f:e}: {ou} {ov}"
            );
            for (x, y) in svd
                .singular_values()
                .iter()
                .zip(reference.singular_values())
            {
                assert!((x / f - y).abs() < eps_tol(n) * reference.singular_values()[0]);
            }
        }
        let mut z = base.clone();
        for i in 0..n {
            z[(i, 0)] = Complex64::new(0.0, 0.0);
        }
        let svd = ComplexSvdDc::compute(z.as_ref()).expect("complex zero column");
        let (r, ou, ov) = complex_check(&z, &svd);
        assert!(
            r < eps_tol(n) && ou < eps_tol(n) && ov < eps_tol(n),
            "complex zero column n={n}"
        );
        assert!(svd.singular_values()[n - 1] < eps_tol(n) * svd.singular_values()[0]);
    }
    // The 2×2 repro.
    let z: Mat<Complex64> = Mat::from_rows(&[
        &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ]);
    let svd = ComplexSvdDc::compute(z.as_ref()).expect("complex 2x2 zero column");
    assert!((svd.singular_values()[0] - 2f64.sqrt()).abs() < 8.0 * f64::EPSILON);
    assert!(svd.singular_values()[1] < 8.0 * f64::EPSILON);
}

/// `compute_with_par` (any `Par`) is bitwise identical to `compute`, on
/// sizes where the parallel GEMM driver actually runs.
#[cfg(feature = "parallel")]
#[test]
fn compute_with_par_is_bitwise_identical() {
    use oxiblas_core::parallel::Par;
    for &(m, n, par) in &[
        (320usize, 320usize, Par::Rayon),
        (360, 260, Par::RayonWith(3)),
        (260, 360, Par::Rayon),
    ] {
        let a = random(m, n, 0xbeef ^ (m * n) as u64);
        let seq = SvdDc::compute(a.as_ref()).expect("seq");
        {
            let p = SvdDc::compute_with_par(a.as_ref(), par).expect("par");
            assert_eq!(
                p.singular_values(),
                seq.singular_values(),
                "{m}x{n} {par:?}"
            );
            for j in 0..m {
                for i in 0..m {
                    assert_eq!(
                        p.u()[(i, j)].to_bits(),
                        seq.u()[(i, j)].to_bits(),
                        "U {par:?}"
                    );
                }
            }
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(
                        p.vt()[(i, j)].to_bits(),
                        seq.vt()[(i, j)].to_bits(),
                        "Vt {par:?}"
                    );
                }
            }
        }
    }
    // f32 as well.
    let a64 = random(300, 300, 0xf32);
    let mut a = Mat::<f32>::zeros(300, 300);
    for j in 0..300 {
        for i in 0..300 {
            a[(i, j)] = a64[(i, j)] as f32;
        }
    }
    let seq = SvdDc::compute(a.as_ref()).expect("seq");
    let p = SvdDc::compute_with_par(a.as_ref(), Par::Rayon).expect("par");
    assert_eq!(p.singular_values(), seq.singular_values());
}

/// Release-only: larger graded / clustered / scaled problems against Jacobi.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "slow in debug (one-sided Jacobi reference); run with --release"
)]
fn large_graded_and_scaled_vs_jacobi() {
    for &n in &[200usize, 300] {
        let geo: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(-12.0 * i as f64 / (n - 1) as f64))
            .collect();
        let a = with_sv(n, n, &geo, 3);
        let dc = SvdDc::compute(a.as_ref()).expect("svd");
        assert_accurate(&format!("graded n={n}"), &a, &dc);
        assert_matches_jacobi(&format!("graded n={n}"), &a, &dc);
        for &f in &[1e-200f64, 1e200] {
            let mut b = a.clone();
            for j in 0..n {
                for i in 0..n {
                    b[(i, j)] *= f;
                }
            }
            let svd = SvdDc::compute(b.as_ref()).expect("scaled");
            assert_accurate(&format!("graded n={n} f={f:e}"), &b, &svd);
            for (x, y) in svd.singular_values().iter().zip(dc.singular_values()) {
                assert!((x / f - y).abs() <= eps_tol(n) * dc.singular_values()[0]);
            }
        }
    }
}
