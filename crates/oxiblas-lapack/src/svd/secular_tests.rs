//! Tests of the singular-value secular equation solver ([`super::lasd4`]).

use super::*;
use crate::svd::Svd;
use oxiblas_matrix::Mat;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// All roots of the problem `(d, z, ρ)` with `‖z‖ = 1` normalised here.
fn roots(d: &[f64], z: &[f64], rho: f64) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = d.len();
    let nz = z.iter().map(|x| x * x).sum::<f64>().sqrt();
    let zn: Vec<f64> = z.iter().map(|x| x / nz).collect();
    let rho = rho * nz * nz;
    let mut sig = vec![0.0; n];
    let mut deltas = vec![vec![0.0; n]; n];
    let mut works = vec![vec![0.0; n]; n];
    for i in 0..n {
        sig[i] = lasd4(i, d, &zn, rho, &mut deltas[i], &mut works[i]).expect("root");
    }
    (sig, deltas, works)
}

/// Singular values (ascending) of `[√ρ·zᵀ; diag(d)]` by one-sided Jacobi.
fn reference(d: &[f64], z: &[f64], rho: f64) -> Vec<f64> {
    let n = d.len();
    let mut m = Mat::<f64>::zeros(n + 1, n);
    for j in 0..n {
        m[(0, j)] = rho.sqrt() * z[j];
        m[(j + 1, j)] = d[j];
    }
    let mut s = Svd::compute(m.as_ref())
        .expect("jacobi")
        .singular_values()
        .to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s
}

/// Checks interlacing, the δ/ω outputs and agreement with the reference.
fn check(d: &[f64], z: &[f64], rho: f64, tol: f64) {
    let n = d.len();
    let (sig, deltas, works) = roots(d, z, rho);
    let nz2 = z.iter().map(|x| x * x).sum::<f64>();
    for i in 0..n {
        assert!(
            sig[i] > d[i],
            "root {i} = {} not above d = {}",
            sig[i],
            d[i]
        );
        if i + 1 < n {
            assert!(
                sig[i] < d[i + 1],
                "root {i} = {} not below {}",
                sig[i],
                d[i + 1]
            );
        } else {
            let ub = (d[n - 1] * d[n - 1] + rho * nz2).sqrt();
            assert!(
                sig[i] <= ub * (1.0 + 4.0 * f64::EPSILON),
                "largest root above bound"
            );
        }
        if n > 1 {
            for j in 0..n {
                let dl = d[j] - sig[i];
                let wk = d[j] + sig[i];
                assert!(
                    (deltas[i][j] - dl).abs() <= 4.0 * f64::EPSILON * sig[i].max(d[j]),
                    "delta[{i}][{j}]"
                );
                assert!(
                    (works[i][j] - wk).abs() <= 4.0 * f64::EPSILON * wk,
                    "work[{i}][{j}]"
                );
            }
        }
    }
    let r = reference(d, z, rho);
    let smax = r[n - 1];
    for i in 0..n {
        assert!(
            (sig[i] - r[i]).abs() <= tol * smax,
            "root {i}: {} vs reference {} (n={n})",
            sig[i],
            r[i]
        );
    }
}

#[test]
fn one_and_two_roots() {
    let (s, ..) = roots(&[0.5], &[1.0], 2.0);
    assert!((s[0] - (0.25f64 + 2.0).sqrt()).abs() < 1e-15);
    check(&[0.0, 1.0], &[0.6, 0.8], 0.7, 1e-15);
    check(&[0.2, 0.3], &[1e-3, 1.0], 1e-2, 1e-15);
    check(&[1e-8, 1.0], &[1.0, 1e-4], 3.0, 1e-15);
}

#[test]
fn random_problems() {
    let mut state = 0x5ec_u64;
    for &n in &[3usize, 4, 10, 50, 200] {
        for _ in 0..3 {
            let mut d: Vec<f64> = (0..n).map(|_| lcg(&mut state)).collect();
            d.sort_by(|a, b| a.total_cmp(b));
            d[0] = 0.0;
            let z: Vec<f64> = (0..n).map(|_| lcg(&mut state) - 0.5).collect();
            let rho = 0.1 + 2.0 * lcg(&mut state);
            check(&d, &z, rho, 4e-15 * n as f64);
        }
    }
}

/// Clustered poles and tiny weights (the regime of the three-pole model and
/// of the geometric-mean bisection).
#[test]
fn clustered_and_graded_poles() {
    let n = 40;
    let d: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                0.0
            } else if i < 20 {
                1e-6 * i as f64
            } else {
                1.0 + 1e-9 * i as f64
            }
        })
        .collect();
    let z: Vec<f64> = (0..n).map(|i| 0.1 + ((i * 7) % 11) as f64 * 0.05).collect();
    check(&d, &z, 1.3, 1e-13);

    // Graded poles with tiny weights at the bottom (geomavg branch).
    let d: Vec<f64> = (0..12)
        .map(|i| if i == 0 { 0.0 } else { 10f64.powi(i - 12) })
        .collect();
    let mut z: Vec<f64> = (0..12).map(|i| 10f64.powi(-(12 - i) / 2)).collect();
    z[11] = 1.0;
    let (s, ..) = roots(&d, &z, 0.5);
    for i in 0..12 {
        assert!(
            s[i] > d[i] && (i == 11 || s[i] < d[i + 1]),
            "interlacing at {i}"
        );
    }
}

#[test]
fn f32_roots() {
    let d = [0.0f32, 0.25, 0.5, 1.0];
    let z = [0.5f32, 0.5, 0.5, 0.5];
    let mut delta = [0.0f32; 4];
    let mut work = [0.0f32; 4];
    let mut prev = 0.0f32;
    for i in 0..4 {
        let s = lasd4(i, &d, &z, 1.0, &mut delta, &mut work).expect("root");
        assert!(s > d[i] && s > prev);
        prev = s;
    }
}
