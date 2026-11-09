//! Tests of the bidiagonal QR iteration ([`super::lasdq`]).

use super::*;

fn identity(n: usize) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = 1.0;
    }
    m
}

/// Dense `n × (n + sqre)` upper bidiagonal.
fn dense_bidiag(d: &[f64], e: &[f64], sqre: usize) -> Mat<f64> {
    let n = d.len();
    let mut b = Mat::<f64>::zeros(n, n + sqre);
    for i in 0..n {
        b[(i, i)] = d[i];
        if i + 1 < n + sqre {
            b[(i, i + 1)] = e[i];
        }
    }
    b
}

/// Runs [`lasdq`] from identity matrices and returns
/// `(σ ascending, max|UᵀU − I|, max|VᵀV − I|, max|B·V − U·[Σ 0]| / max|B|)`.
fn run(d: &[f64], e: &[f64], sqre: usize) -> (Vec<f64>, f64, f64, f64) {
    let n = d.len();
    let m = n + sqre;
    let b = dense_bidiag(d, e, sqre);
    let mut dd = d.to_vec();
    let mut ee = e.to_vec();
    let mut u = identity(n);
    let mut v = identity(m);
    lasdq(sqre, &mut dd, &mut ee, &mut u, &mut v).expect("lasdq converges");
    for i in 1..n {
        assert!(dd[i] >= dd[i - 1], "not ascending at {i}: {dd:?}");
    }
    assert!(dd.iter().all(|&s| s >= 0.0));
    let orth = |q: &Mat<f64>| {
        let k = q.ncols();
        let mut worst = 0.0f64;
        for a in 0..k {
            for c in 0..k {
                let g: f64 = (0..q.nrows()).map(|r| q[(r, a)] * q[(r, c)]).sum();
                worst = worst.max((g - if a == c { 1.0 } else { 0.0 }).abs());
            }
        }
        worst
    };
    let bmax = d
        .iter()
        .chain(e.iter())
        .fold(0.0f64, |acc, x| acc.max(x.abs()))
        .max(f64::MIN_POSITIVE);
    let mut resid = 0.0f64;
    for i in 0..n {
        for j in 0..m {
            let bv: f64 = (0..m).map(|t| b[(i, t)] * v[(t, j)]).sum();
            let us = if j < n { u[(i, j)] * dd[j] } else { 0.0 };
            resid = resid.max((bv - us).abs());
        }
    }
    (dd, orth(&u), orth(&v), resid / bmax)
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

#[test]
fn lartg_matches_reference_conventions() {
    for &(f, g) in &[
        (3.0f64, 4.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (0.0, -2.0),
        (5.0, 0.0),
        (1e-300, 3e-300),
        (1e300, -4e300),
        (2e-170, 1e160),
    ] {
        let (c, s, r) = lartg(f, g);
        assert!(c >= 0.0, "c = {c}");
        assert!((c * c + s * s - 1.0).abs() < 4e-16, "{f} {g}");
        let scale = f.abs().max(g.abs());
        assert!(((c * f + s * g) - r).abs() <= 4e-16 * scale, "{f} {g}");
        assert!((-s * f + c * g).abs() <= 4e-16 * scale, "{f} {g}");
        if f != 0.0 {
            assert_eq!(r.signum(), f.signum());
        }
    }
}

#[test]
fn two_by_two_kernels() {
    for &(f, g, h) in &[
        (2.0f64, 0.5, 1.0),
        (1.0, 3.0, -2.0),
        (-1e-5, 1.0, 1e-5),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1e-200, 1e200, 1e-200),
        (4.0, 0.0, -3.0),
    ] {
        let (smin2, smax2) = las2(f, g, h);
        let r = lasv2(f, g, h);
        assert!(
            (r.ssmax.abs() - smax2).abs() <= 1e-15 * smax2,
            "{f} {g} {h}"
        );
        assert!(
            (r.ssmin.abs() - smin2).abs() <= 1e-15 * smax2,
            "{f} {g} {h}"
        );
        // [csl snl; -snl csl] [f g; 0 h] [csr -snr; snr csr] = diag(ssmax, ssmin)
        let a = [[f, g], [0.0, h]];
        let l = [[r.csl, r.snl], [-r.snl, r.csl]];
        let rr = [[r.csr, -r.snr], [r.snr, r.csr]];
        let mut la = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                la[i][j] = l[i][0] * a[0][j] + l[i][1] * a[1][j];
            }
        }
        let mut out = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                out[i][j] = la[i][0] * rr[0][j] + la[i][1] * rr[1][j];
            }
        }
        let tol = 4e-16 * smax2.max(f64::MIN_POSITIVE);
        assert!((out[0][0] - r.ssmax).abs() <= tol, "{f} {g} {h}: {out:?}");
        assert!((out[1][1] - r.ssmin).abs() <= tol, "{f} {g} {h}: {out:?}");
        assert!(out[0][1].abs() <= tol && out[1][0].abs() <= tol, "{out:?}");
    }
}

#[test]
fn random_bidiagonals_square_and_rectangular() {
    let mut state = 0x51_u64;
    for &n in &[1usize, 2, 3, 7, 16, 25, 26, 40] {
        for sqre in 0..2 {
            let d: Vec<f64> = (0..n).map(|_| lcg(&mut state)).collect();
            let e: Vec<f64> = (0..n - 1 + sqre).map(|_| lcg(&mut state)).collect();
            let (s, ou, ov, res) = run(&d, &e, sqre);
            let tol = 8.0 * (n as f64) * f64::EPSILON;
            assert!(ou < tol && ov < tol, "n={n} sqre={sqre}: {ou} {ov}");
            assert!(res < tol, "n={n} sqre={sqre}: residual {res}");
            assert_eq!(s.len(), n);
        }
    }
}

/// `d = e = 1` (square): `σ_k = 2·cos(kπ/(2n+1)) = 2·sin((2n+1−2k)π/(4n+2))`,
/// `k = 1..n` (the sine form is accurate to a few ulps even for the smallest).
#[test]
fn ones_bidiagonal_has_known_singular_values() {
    for &n in &[5usize, 20] {
        let (s, ..) = run(&vec![1.0; n], &vec![1.0; n - 1], 0);
        let mut exact: Vec<f64> = (1..=n)
            .map(|k| {
                2.0 * ((2 * n + 1 - 2 * k) as f64 * core::f64::consts::PI / (4 * n + 2) as f64)
                    .sin()
            })
            .collect();
        exact.sort_by(|a, b| a.total_cmp(b));
        for k in 0..n {
            assert!(
                (s[k] - exact[k]).abs() <= 2.0 * n as f64 * f64::EPSILON * exact[k],
                "n={n} k={k}: {} vs {}",
                s[k],
                exact[k]
            );
        }
    }
}

/// Exact zeros on the diagonal (first, middle, last, several, all) are
/// removed by explicit chasing: the iteration converges, the matching
/// singular values are exactly zero and the factorisation stays exact.
#[test]
fn zero_diagonal_entries() {
    let mut state = 0x7a_u64;
    for &n in &[2usize, 3, 6, 25] {
        let patterns: Vec<Vec<usize>> = vec![
            vec![0],
            vec![n / 2],
            vec![n - 1],
            vec![0, n - 1],
            (0..n).step_by(2).collect(),
            (0..n).collect(),
        ];
        for zeros in patterns {
            for sqre in 0..2 {
                let mut d: Vec<f64> = (0..n).map(|_| 1.0 + lcg(&mut state).abs()).collect();
                let e: Vec<f64> = (0..n - 1 + sqre).map(|_| lcg(&mut state)).collect();
                for &k in &zeros {
                    d[k] = 0.0;
                }
                let (s, ou, ov, res) = run(&d, &e, sqre);
                let tol = 8.0 * (n as f64) * f64::EPSILON;
                assert!(ou < tol && ov < tol, "n={n} {zeros:?} sqre={sqre}");
                assert!(res < tol, "n={n} {zeros:?} sqre={sqre}: residual {res}");
                if sqre == 0 {
                    let nz = s.iter().filter(|&&x| x == 0.0).count();
                    // A zero on the diagonal of a square bidiagonal is at least
                    // one exact zero singular value.
                    assert!(nz >= 1, "n={n} {zeros:?}: {s:?}");
                }
            }
        }
    }
}

/// Graded bidiagonals: relative accuracy is kept (the singular values of a
/// diagonal-dominant graded matrix agree with its diagonal to O(ε) relative
/// error even though they span 200 orders of magnitude).
#[test]
fn graded_bidiagonal_relative_accuracy() {
    let n = 21;
    let d: Vec<f64> = (0..n).map(|i| 10f64.powi(-10 * i as i32)).collect();
    let e: Vec<f64> = (0..n - 1)
        .map(|i| 1e-3 * 10f64.powi(-10 * (i as i32 + 1)))
        .collect();
    let (s, ou, ov, _) = run(&d, &e, 0);
    assert!(ou < 1e-14 && ov < 1e-14);
    // B = D·(I + N) with ‖N‖ ≈ 1e-13 (e_i / d_i = 1e-13), so the singular
    // values are the diagonal entries to a relative 1e-13, across 200 decades.
    let mut dsorted = d.clone();
    dsorted.sort_by(|a, b| a.total_cmp(b));
    for k in 0..n {
        let rel = (s[k] - dsorted[k]).abs() / dsorted[k];
        assert!(rel < 1e-12, "k={k}: {} vs {} (rel {rel})", s[k], dsorted[k]);
    }
    // The same matrix scaled by 2^-300 (still normal numbers) gives the
    // scaled singular values.
    let sc = 2f64.powi(-300);
    let d2: Vec<f64> = d.iter().map(|x| x * sc).collect();
    let e2: Vec<f64> = e.iter().map(|x| x * sc).collect();
    let mut dd = d2.clone();
    let mut ee = e2.clone();
    let mut u = identity(n);
    let mut v = identity(n);
    lasdq(0, &mut dd, &mut ee, &mut u, &mut v).expect("converges");
    for k in 0..n {
        let rel = (dd[k] / sc - s[k]).abs() / s[k];
        assert!(rel < 64.0 * f64::EPSILON, "k={k}: rel {rel}");
    }
}

#[test]
fn non_finite_input_is_reported() {
    for bad in [f64::NAN, f64::INFINITY] {
        let mut d = vec![1.0, bad, 2.0];
        let mut e = vec![0.5, 0.5];
        let mut u = identity(3);
        let mut v = identity(3);
        assert_eq!(
            lasdq(0, &mut d, &mut e, &mut u, &mut v),
            Err(BidiagDcError::NotConverged)
        );
    }
}

#[test]
fn f32_bidiagonal() {
    let n = 20;
    let d: Vec<f32> = (0..n).map(|i| 1.0 + 0.1 * i as f32).collect();
    let e: Vec<f32> = (0..n - 1).map(|i| 0.3 - 0.01 * i as f32).collect();
    let mut dd = d.clone();
    let mut ee = e.clone();
    let mut u = Mat::<f32>::zeros(n, n);
    let mut v = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        u[(i, i)] = 1.0;
        v[(i, i)] = 1.0;
    }
    lasdq(0, &mut dd, &mut ee, &mut u, &mut v).expect("converges");
    let mut worst = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let mut bv = d[i] * v[(i, j)];
            if i + 1 < n {
                bv += e[i] * v[(i + 1, j)];
            }
            worst = worst.max((bv - u[(i, j)] * dd[j]).abs());
        }
    }
    assert!(worst < 16.0 * n as f32 * f32::EPSILON, "{worst}");
}
