//! Tests of the bidiagonal divide-and-conquer SVD kernel.

use super::*;

/// `(max |UᵀU − I|, max |VᵀV − I|, max |UΣVᵀ − B| / max |B|, σ)`.
fn svd_errors(d: &[f64], e: &[f64]) -> (f64, f64, f64, Vec<f64>) {
    let n = d.len();
    let (u, sigma, vt) = bidiagonal_svd_dc(d, e).expect("svd");
    let mut uu_err = 0.0f64;
    let mut vv_err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut uu = 0.0;
            let mut vv = 0.0;
            for k in 0..n {
                uu += u[(k, i)] * u[(k, j)];
                vv += vt[(i, k)] * vt[(j, k)];
            }
            let expect = if i == j { 1.0 } else { 0.0 };
            uu_err = uu_err.max((uu - expect).abs());
            vv_err = vv_err.max((vv - expect).abs());
        }
    }
    let bmax = d
        .iter()
        .chain(e.iter())
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(f64::MIN_POSITIVE);
    let mut rec_err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += u[(i, k)] * sigma[k] * vt[(k, j)];
            }
            let b = if j == i {
                d[i]
            } else if j == i + 1 {
                e[i]
            } else {
                0.0
            };
            rec_err = rec_err.max((acc - b).abs());
        }
    }
    for i in 1..n {
        assert!(sigma[i] <= sigma[i - 1], "not descending at {i}");
    }
    assert!(sigma.iter().all(|&s| s >= 0.0));
    (uu_err, vv_err, rec_err / bmax, sigma)
}

/// Tolerance `c·n·ε` used throughout.
fn tol(n: usize) -> f64 {
    8.0 * n as f64 * f64::EPSILON
}

/// Reference singular values (ascending) of the whole bidiagonal by the QR
/// iteration (`dbdsqr`, high relative accuracy), whatever its size.
fn qr_reference(d: &[f64], e: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut dd = d.to_vec();
    let mut ee = e.to_vec();
    let mut u = identity::<f64>(n);
    let mut v = identity::<f64>(n);
    lasdq(0, &mut dd, &mut ee, &mut u, &mut v).expect("qr reference");
    dd
}

#[test]
fn svd_small_qr_path() {
    let d = vec![2.0, 3.0, 1.5, 4.0, 0.7];
    let e = vec![0.5, -0.3, 0.8, 0.2];
    let (uu, vv, rec, _) = svd_errors(&d, &e);
    assert!(
        uu < tol(5) && vv < tol(5) && rec < tol(5),
        "{uu} {vv} {rec}"
    );
}

#[test]
fn svd_accuracy_across_sizes() {
    for &n in &[26usize, 50, 100, 200] {
        let d: Vec<f64> = (0..n)
            .map(|i| 2.0 + (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 1.7).cos())
            .collect();
        let e: Vec<f64> = (0..n - 1)
            .map(|i| 0.6 + 0.4 * (i as f64 * 0.9).cos() + 0.2 * (i as f64 * 0.3).sin())
            .collect();
        let (uu, vv, rec, s) = svd_errors(&d, &e);
        assert!(uu < tol(n), "n={n} UtU={uu}");
        assert!(vv < tol(n), "n={n} VtV={vv}");
        assert!(rec < tol(n), "n={n} recon={rec}");
        let r = qr_reference(&d, &e);
        for k in 0..n {
            let rel = (s[n - 1 - k] - r[k]).abs() / r[k];
            assert!(rel < tol(n), "n={n}: sigma {k} rel {rel}");
        }
    }
}

#[test]
fn svd_dc_path_with_zero_coupling() {
    let mut d = vec![0.0; 30];
    let mut e = vec![0.6; 29];
    for (i, di) in d.iter_mut().enumerate() {
        *di = 3.0 + (i as f64 * 0.5).sin();
    }
    e[15] = 0.0;
    let (uu, vv, rec, _) = svd_errors(&d, &e);
    assert!(
        uu < tol(30) && vv < tol(30) && rec < tol(30),
        "{uu} {vv} {rec}"
    );
}

/// Genuinely repeated singular values: `B₀ ⊕ B₀` (split at a zero coupling)
/// and the same with a tiny but non-negligible coupling, which keeps the
/// two copies in one subproblem so the merges must deflate the coincident
/// values with rotations.
#[test]
fn svd_repeated_singular_values_reconstructs() {
    let k = 16usize;
    let d0: Vec<f64> = (0..k)
        .map(|i| 2.5 + (i as f64 * 0.53).sin() + 0.4 * (i as f64 * 1.3).cos())
        .collect();
    let e0: Vec<f64> = (0..k - 1)
        .map(|i| 0.7 + 0.3 * (i as f64 * 0.8).cos())
        .collect();
    for coupling in [0.0, 1e-13] {
        let mut d = Vec::with_capacity(2 * k);
        d.extend_from_slice(&d0);
        d.extend_from_slice(&d0);
        let mut e = Vec::with_capacity(2 * k - 1);
        e.extend_from_slice(&e0);
        e.push(coupling);
        e.extend_from_slice(&e0);
        let (uu, vv, rec, sigma) = svd_errors(&d, &e);
        for (a, &sa) in sigma.iter().enumerate() {
            let has_partner = sigma
                .iter()
                .enumerate()
                .any(|(b, &sb)| b != a && (sa - sb).abs() < 1e-8);
            assert!(has_partner, "no coincident partner for sigma[{a}] = {sa}");
        }
        let n = 2 * k;
        assert!(uu < tol(n), "coupling {coupling}: UtU {uu}");
        assert!(vv < tol(n), "coupling {coupling}: VtV {vv}");
        assert!(rec < tol(n), "coupling {coupling}: reconstruction {rec}");
    }
}

/// Coupled clusters of nearly equal singular values exercise the
/// close-value rotations inside the merges.
#[test]
fn svd_clustered_spectrum() {
    let n = 120;
    let d: Vec<f64> = (0..n).map(|i| 1.0 + 1e-10 * (i % 7) as f64).collect();
    let e: Vec<f64> = (0..n - 1).map(|i| 1e-9 * ((i % 5) as f64 + 1.0)).collect();
    let (uu, vv, rec, _) = svd_errors(&d, &e);
    assert!(
        uu < tol(n) && vv < tol(n) && rec < tol(n),
        "{uu} {vv} {rec}"
    );
}

/// Zero diagonal entries in the divide-and-conquer path (first, middle,
/// last, periodic): orthogonal factors, exact reconstruction, and singular
/// values at the `ε·‖B‖` level for the zero ones.
#[test]
fn svd_rank_deficient_bidiagonal() {
    let n = 60;
    let base: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let e: Vec<f64> = (0..n - 1).map(|i| 0.3 + 0.001 * i as f64).collect();
    let patterns: Vec<Vec<usize>> = vec![
        vec![0],
        vec![n / 2],
        vec![n - 1],
        (0..n).step_by(3).collect(),
    ];
    for zeros in patterns {
        let mut d = base.clone();
        for &k in &zeros {
            d[k] = 0.0;
        }
        let (uu, vv, rec, sigma) = svd_errors(&d, &e);
        assert!(
            uu < tol(n) && vv < tol(n) && rec < tol(n),
            "{zeros:?}: {uu} {vv} {rec}"
        );
        // A square bidiagonal with k zero diagonal entries has at least one
        // zero singular value (rank n − 1 at most).
        assert!(
            sigma[n - 1] < tol(n) * sigma[0],
            "{zeros:?}: smallest {}",
            sigma[n - 1]
        );
    }
}

/// Scaling by a power of two is exact: the singular values scale exactly
/// (bitwise) and the vectors are unchanged, from 2^-1000 to 2^900.
#[test]
fn svd_scale_invariance_is_exact() {
    let n = 70;
    let d: Vec<f64> = (0..n).map(|i| 1.5 + (i as f64 * 0.41).sin()).collect();
    let e: Vec<f64> = (0..n - 1)
        .map(|i| 0.5 + 0.3 * (i as f64 * 0.7).cos())
        .collect();
    let (u0, s0, vt0) = bidiagonal_svd_dc(&d, &e).expect("svd");
    for &k in &[-1000i32, -300, 300, 900] {
        let f = 2f64.powi(k);
        let ds: Vec<f64> = d.iter().map(|x| x * f).collect();
        let es: Vec<f64> = e.iter().map(|x| x * f).collect();
        let (u1, s1, vt1) = bidiagonal_svd_dc(&ds, &es).expect("svd");
        for i in 0..n {
            assert_eq!(s1[i], s0[i] * f, "k={k} sigma[{i}]");
        }
        for i in 0..n {
            for j in 0..n {
                assert_eq!(u0[(i, j)], u1[(i, j)], "k={k}: U[{i},{j}]");
                assert_eq!(vt0[(i, j)], vt1[(i, j)], "k={k}: Vt[{i},{j}]");
            }
        }
    }
}

/// Graded bidiagonal (σ spanning twelve decades): the divide and conquer
/// agrees with the relatively accurate QR iteration and stays orthogonal.
#[test]
fn svd_graded_bidiagonal_matches_qr() {
    for &n in &[60usize, 150] {
        let d: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(-12.0 * i as f64 / (n - 1) as f64))
            .collect();
        let e: Vec<f64> = (0..n - 1).map(|i| 0.5 * d[i + 1]).collect();
        let (uu, vv, rec, s) = svd_errors(&d, &e);
        assert!(
            uu < tol(n) && vv < tol(n) && rec < tol(n),
            "n={n}: {uu} {vv} {rec}"
        );
        let r = qr_reference(&d, &e);
        let mut worst = 0.0f64;
        for k in 0..n {
            worst = worst.max((s[n - 1 - k] - r[k]).abs() / r[k]);
        }
        assert!(
            worst < 1e3 * tol(n),
            "n={n}: worst relative difference {worst}"
        );
    }
}

/// The portable (non-BLAS) backend gives the same decomposition quality.
#[test]
fn portable_backend_matches() {
    let n = 70;
    let d: Vec<f64> = (0..n).map(|i| 1.5 + (i as f64 * 0.41).sin()).collect();
    let e: Vec<f64> = (0..n - 1)
        .map(|i| 0.5 + 0.3 * (i as f64 * 0.7).cos())
        .collect();
    let mut u1 = Mat::zeros(n, n);
    let mut s1 = vec![0.0; n];
    let mut v1 = Mat::zeros(n, n);
    bidiag_svd_into::<f64, PortableOps>(PortableOps, &d, &e, &mut u1, &mut s1, &mut v1)
        .expect("portable");
    let mut u2 = Mat::zeros(n, n);
    let mut s2 = vec![0.0; n];
    let mut v2 = Mat::zeros(n, n);
    bidiag_svd_into::<f64, BlasOps>(BlasOps::SEQ, &d, &e, &mut u2, &mut s2, &mut v2).expect("blas");
    for i in 0..n {
        assert!((s1[i] - s2[i]).abs() < 1e-13 * s2[0], "sigma[{i}]");
    }
}

#[test]
fn non_finite_and_degenerate_inputs() {
    let n = 40;
    let mut d = vec![1.0; n];
    let e = vec![0.5; n - 1];
    d[7] = f64::NAN;
    assert_eq!(
        bidiagonal_svd_dc(&d, &e).map(|_| ()),
        Err(BidiagDcError::NotConverged)
    );
    d[7] = f64::INFINITY;
    assert_eq!(
        bidiagonal_svd_dc(&d, &e).map(|_| ()),
        Err(BidiagDcError::NotConverged)
    );
    // All zeros: σ = 0 with identity factors.
    let (u, s, vt) = bidiagonal_svd_dc(&vec![0.0; n], &vec![0.0; n - 1]).expect("zero");
    assert!(s.iter().all(|&x| x == 0.0));
    for i in 0..n {
        assert_eq!(u[(i, i)], 1.0);
        assert_eq!(vt[(i, i)], 1.0);
    }
    // Diagonal (every e zero): σ = sorted |d|.
    let dd: Vec<f64> = (0..n).map(|i| (i as f64 - 20.0) * 0.25).collect();
    let (uu, vv, rec, s) = svd_errors(&dd, &vec![0.0; n - 1]);
    assert!(uu < tol(n) && vv < tol(n) && rec < tol(n));
    let mut abs: Vec<f64> = dd.iter().map(|x| x.abs()).collect();
    abs.sort_by(|a, b| b.total_cmp(a));
    assert_eq!(s, abs);
}

#[test]
fn f32_kernel() {
    let n = 80;
    let d: Vec<f32> = (0..n).map(|i| 1.0 + 0.5 * (i as f32 * 0.3).sin()).collect();
    let e: Vec<f32> = (0..n - 1).map(|i| 0.4 * (i as f32 * 0.7).cos()).collect();
    let (u, s, vt) = bidiagonal_svd_dc(&d, &e).expect("svd");
    let mut worst = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += u[(i, k)] * s[k] * vt[(k, j)];
            }
            let b = if j == i {
                d[i]
            } else if j == i + 1 {
                e[i]
            } else {
                0.0
            };
            worst = worst.max((acc - b).abs());
        }
    }
    assert!(worst < 8.0 * n as f32 * f32::EPSILON * s[0], "{worst}");
}
