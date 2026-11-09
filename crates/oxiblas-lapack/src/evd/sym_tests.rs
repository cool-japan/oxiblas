//! Tests of the shared symmetric-eigensolver machinery (tridiagonal
//! reduction, back-transformation, tridiagonal QR and divide and conquer)
//! and of both drivers on large and structured matrices.

use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

use super::dc_merge::stedc;
use super::sym_kernels::{
    BlasMatMul, MatMul, NaiveMatMul, argsort, lower_from_upper, permute_columns,
};
use super::sym_steqr::{Deflation, RotTarget, RotationLog, steqr};
use super::sym_tridiag::{apply_q, form_q, reduce};
use super::{SymmetricEvd, SymmetricEvdDc};

/// Deterministic split-mix PRNG producing values in [-1, 1].
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
}

fn random_symmetric(n: usize, seed: u64) -> Mat<f64> {
    let mut rng = Prng(seed);
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            let v = rng.next_f64();
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    a
}

/// `H·diag(eigs)·H` with a random Householder `H`: exactly the spectrum `eigs`.
fn with_spectrum(eigs: &[f64], seed: u64) -> Mat<f64> {
    let n = eigs.len();
    let mut rng = Prng(seed);
    let mut w: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in w.iter_mut() {
        *x /= nrm;
    }
    // A = D - 2 w (Dw)ᵀ - 2 (Dw) wᵀ + 4 (wᵀDw) w wᵀ
    let dw: Vec<f64> = (0..n).map(|i| eigs[i] * w[i]).collect();
    let wdw: f64 = (0..n).map(|i| w[i] * dw[i]).sum();
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let diag = if i == j { eigs[i] } else { 0.0 };
            a[(i, j)] = diag - 2.0 * w[i] * dw[j] - 2.0 * dw[i] * w[j] + 4.0 * wdw * w[i] * w[j];
        }
    }
    a
}

/// The diagonally dominant matrix of the `lapack_evd` benchmark (clustered
/// eigenvalues).
fn bench_matrix(n: usize) -> Mat<f64> {
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let v = if i == j {
                n as f64 + ((i * 17) % 10) as f64
            } else {
                ((i + j) % 10) as f64 * 0.1
            };
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    a
}

fn tridiag_matrix(d: &[f64], e: &[f64]) -> Mat<f64> {
    let n = d.len();
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = d[i];
        if i + 1 < n {
            a[(i, i + 1)] = e[i];
            a[(i + 1, i)] = e[i];
        }
    }
    a
}

fn frob(a: &Mat<f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            s += a[(i, j)] * a[(i, j)];
        }
    }
    s.sqrt()
}

fn matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let mut c = Mat::zeros(a.nrows(), b.ncols());
    BlasMatMul::gemm(
        false,
        false,
        1.0,
        a.as_ref(),
        b.as_ref(),
        0.0,
        c.as_mut(),
        Par::Seq,
    );
    c
}

fn matvec(a: &Mat<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows()];
    for (j, &xj) in x.iter().enumerate() {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi += a[(i, j)] * xj;
        }
    }
    y
}

fn matvec_t(a: &Mat<f64>, x: &[f64]) -> Vec<f64> {
    (0..a.ncols())
        .map(|j| (0..a.nrows()).map(|i| a[(i, j)] * x[i]).sum())
        .collect()
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Copies `v` (a view) into an owned matrix.
fn owned(v: oxiblas_matrix::MatRef<'_, f64>) -> Mat<f64> {
    let mut m = Mat::zeros(v.nrows(), v.ncols());
    for j in 0..v.ncols() {
        for i in 0..v.nrows() {
            m[(i, j)] = v[(i, j)];
        }
    }
    m
}

/// Probe-vector accuracy of `(vals, vecs)` for `a` (`O(n²)` per probe):
/// `(max ‖(AV - VΛ)x‖ / (‖A‖_F‖x‖), max ‖VᵀVx - x‖ / ‖x‖)` over 3 probes.
fn probe_accuracy(a: &Mat<f64>, vals: &[f64], vecs: &Mat<f64>) -> (f64, f64) {
    let n = a.nrows();
    let anorm = frob(a).max(f64::MIN_POSITIVE);
    let mut rng = Prng(0xC0FF_EE00 + n as u64);
    let (mut resid, mut orth) = (0.0f64, 0.0f64);
    for _ in 0..3 {
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let xn = norm(&x);
        let vx = matvec(vecs, &x);
        let avx = matvec(a, &vx);
        let lx: Vec<f64> = x.iter().zip(vals).map(|(xi, l)| xi * l).collect();
        let vlx = matvec(vecs, &lx);
        let r: Vec<f64> = avx.iter().zip(&vlx).map(|(p, q)| p - q).collect();
        resid = resid.max(norm(&r) / (anorm * xn));
        let vtvx = matvec_t(vecs, &vx);
        let o: Vec<f64> = vtvx.iter().zip(&x).map(|(p, q)| p - q).collect();
        orth = orth.max(norm(&o) / xn);
    }
    (resid, orth)
}

/// Full accuracy: `(‖AV - VΛ‖_F / ‖A‖_F, ‖VᵀV - I‖_F)`.
fn full_accuracy(a: &Mat<f64>, vals: &[f64], vecs: &Mat<f64>) -> (f64, f64) {
    let n = a.nrows();
    let av = matmul(a, vecs);
    let mut r = 0.0;
    for j in 0..n {
        for i in 0..n {
            let d = av[(i, j)] - vecs[(i, j)] * vals[j];
            r += d * d;
        }
    }
    let mut vt = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            vt[(j, i)] = vecs[(i, j)];
        }
    }
    let mut g = matmul(&vt, vecs);
    for i in 0..n {
        g[(i, i)] -= 1.0;
    }
    (r.sqrt() / frob(a).max(f64::MIN_POSITIVE), frob(&g))
}

/// Runs both drivers and checks accuracy, ordering and agreement.
fn check_both(a: &Mat<f64>, tol: f64, label: &str) -> (Vec<f64>, Vec<f64>) {
    let n = a.nrows();
    let qr =
        SymmetricEvd::compute(a.as_ref()).unwrap_or_else(|e| panic!("{label}: QR failed: {e}"));
    let dc =
        SymmetricEvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("{label}: DC failed: {e}"));
    let scale = frob(a).max(f64::MIN_POSITIVE);
    for (name, vals, vecs) in [
        ("QR", qr.eigenvalues(), owned(qr.eigenvectors())),
        ("DC", dc.eigenvalues(), owned(dc.eigenvectors())),
    ] {
        assert!(
            vals.windows(2).all(|w| w[0] <= w[1]),
            "{label} {name}: eigenvalues not ascending"
        );
        let (resid, orth) = if n <= 300 {
            full_accuracy(a, vals, &vecs)
        } else {
            probe_accuracy(a, vals, &vecs)
        };
        let nf = n as f64;
        assert!(
            resid <= tol * nf.sqrt(),
            "{label} {name}: residual {resid:e} (n = {n})"
        );
        assert!(
            orth <= tol * nf,
            "{label} {name}: orthogonality {orth:e} (n = {n})"
        );
    }
    for (i, (x, y)) in qr.eigenvalues().iter().zip(dc.eigenvalues()).enumerate() {
        assert!(
            (x - y).abs() <= tol * scale,
            "{label}: eigenvalue {i} differs: QR {x} DC {y}"
        );
    }
    (qr.eigenvalues().to_vec(), dc.eigenvalues().to_vec())
}

// ---------------------------------------------------------------------------
// Tridiagonal reduction and back-transformation
// ---------------------------------------------------------------------------

fn check_reduction<G: MatMul<f64>>(n: usize, seed: u64) {
    let a = random_symmetric(n, seed);
    let mut work = lower_from_upper(a.as_ref());
    let tri = reduce::<f64, G>(&mut work, Par::Seq);
    let mut q = Mat::zeros(n, n);
    form_q::<f64, G>(&work, &tri.tau, &mut q, Par::Seq);
    // Q orthogonal and A = Q T Qᵀ.
    let t = tridiag_matrix(&tri.d, &tri.e);
    let qt = matmul(&q, &t);
    let mut qtr = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            qtr[(j, i)] = q[(i, j)];
        }
    }
    let rec = matmul(&qt, &qtr);
    let mut diff = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            diff = diff.max((rec[(i, j)] - a[(i, j)]).abs());
        }
    }
    assert!(diff <= 1e-13 * (n as f64), "n = {n}: ‖QTQᵀ - A‖ = {diff:e}");
    let qtq = matmul(&qtr, &q);
    let mut orth = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let target = if i == j { 1.0 } else { 0.0 };
            orth = orth.max((qtq[(i, j)] - target).abs());
        }
    }
    assert!(orth <= 1e-13 * (n as f64), "n = {n}: ‖QᵀQ - I‖ = {orth:e}");
}

#[test]
fn reduction_reconstructs_blocked_and_unblocked() {
    for (idx, &n) in [1usize, 2, 3, 4, 10, 33, 161, 200, 290].iter().enumerate() {
        check_reduction::<BlasMatMul>(n, 0x51 + idx as u64);
        check_reduction::<NaiveMatMul>(n, 0x51 + idx as u64);
    }
}

#[test]
fn blocked_and_unblocked_reductions_agree() {
    let n = 230;
    let a = random_symmetric(n, 0x7777);
    let mut w1 = lower_from_upper(a.as_ref());
    let mut w2 = lower_from_upper(a.as_ref());
    let t1 = reduce::<f64, BlasMatMul>(&mut w1, Par::Seq);
    let t2 = reduce::<f64, NaiveMatMul>(&mut w2, Par::Seq);
    // The two operation orders give the same T up to rounding-error growth
    // over the n reflector steps (a few hundred ulps of ‖A‖).
    let tol = 1e-12 * frob(&a);
    for i in 0..n {
        assert!((t1.d[i] - t2.d[i]).abs() <= tol, "d[{i}]");
    }
    for i in 0..n - 1 {
        assert!((t1.e[i] - t2.e[i]).abs() <= tol, "e[{i}]");
        assert!((t1.tau[i] - t2.tau[i]).abs() <= 1e-10, "tau[{i}]");
    }
}

#[test]
fn apply_q_matches_formed_q() {
    for &(n, m) in &[(5usize, 3usize), (70, 9), (250, 250), (250, 17)] {
        let a = random_symmetric(n, 0xABC + n as u64);
        let mut work = lower_from_upper(a.as_ref());
        let tri = reduce::<f64, BlasMatMul>(&mut work, Par::Seq);
        let mut q = Mat::zeros(n, n);
        form_q::<f64, BlasMatMul>(&work, &tri.tau, &mut q, Par::Seq);
        let mut rng = Prng(n as u64 * 31 + m as u64);
        let mut z = Mat::zeros(n, m);
        for j in 0..m {
            for i in 0..n {
                z[(i, j)] = rng.next_f64();
            }
        }
        let expected = matmul(&q, &z);
        for blocked in [true, false] {
            let mut zz = z.clone();
            if blocked {
                apply_q::<f64, BlasMatMul>(&work, &tri.tau, &mut zz, Par::Seq);
            } else {
                apply_q::<f64, NaiveMatMul>(&work, &tri.tau, &mut zz, Par::Seq);
            }
            for j in 0..m {
                for i in 0..n {
                    assert!(
                        (zz[(i, j)] - expected[(i, j)]).abs() <= 1e-12,
                        "n = {n}, m = {m}, blocked = {blocked}: ({i}, {j})"
                    );
                }
            }
        }
    }
}

#[test]
fn reduction_never_reads_the_strict_upper_triangle() {
    // n = 50 runs only the unblocked reduction; 161 and 290 also run blocked
    // panels with the trailing GEMM update, which writes (as workspace) but
    // never reads the strict upper triangle of its diagonal blocks. NaN there
    // must change nothing; the unblocked reduction must not write there.
    for &n in &[50usize, 161, 290] {
        let a = random_symmetric(n, 0x1A0 + n as u64);
        let mut clean = lower_from_upper(a.as_ref());
        let t0 = reduce::<f64, BlasMatMul>(&mut clean, Par::Seq);
        let mut work = lower_from_upper(a.as_ref());
        for j in 1..n {
            for i in 0..j {
                work[(i, j)] = f64::NAN;
            }
        }
        let t = reduce::<f64, BlasMatMul>(&mut work, Par::Seq);
        for j in 0..n {
            for i in j..n {
                assert_eq!(
                    work[(i, j)].to_bits(),
                    clean[(i, j)].to_bits(),
                    "n = {n}: lower entry ({i}, {j})"
                );
            }
            if n <= 160 {
                for i in 0..j {
                    assert!(
                        work[(i, j)].is_nan(),
                        "n = {n}: upper entry ({i}, {j}) written"
                    );
                }
            }
        }
        for (name, x, y) in [
            ("d", &t.d, &t0.d),
            ("e", &t.e, &t0.e),
            ("tau", &t.tau, &t0.tau),
        ] {
            for (idx, (u, v)) in x.iter().zip(y.iter()).enumerate() {
                assert_eq!(u.to_bits(), v.to_bits(), "n = {n}: {name}[{idx}]");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tridiagonal solvers
// ---------------------------------------------------------------------------

/// Reference: the same QR iteration with every rotation applied at once.
fn steqr_reference(d: &mut [f64], e: &mut [f64], z: &mut Mat<f64>, deflation: Deflation) {
    let n = d.len();
    let eps2 = (f64::EPSILON / 2.0).powi(2);
    let negligible = |e: f64, d1: f64, d2: f64| match deflation {
        Deflation::Relaxed => e.abs() <= 100.0 * f64::EPSILON * (d1.abs() + d2.abs()),
        Deflation::Strict => e * e <= eps2 * d1.abs() * d2.abs() + f64::MIN_POSITIVE,
    };
    let mut m = n - 1;
    let mut iter = 0;
    while m > 0 && iter < 100 * n {
        iter += 1;
        let mut l = m;
        while l > 0 {
            if negligible(e[l - 1], d[l - 1], d[l]) {
                e[l - 1] = 0.0;
                break;
            }
            l -= 1;
        }
        if l == m {
            m -= 1;
            continue;
        }
        let dd = (d[m - 1] - d[m]) / 2.0;
        let ee = e[m - 1];
        let mu = d[m] - ee * ee / (dd + dd.signum() * dd.hypot(ee));
        let mut x = d[l] - mu;
        let mut zz = e[l];
        for k in l..m {
            let (c, s) = if zz == 0.0 {
                (1.0, 0.0)
            } else if zz.abs() > x.abs() {
                let t = -x / zz;
                let s = 1.0 / (1.0 + t * t).sqrt();
                (s * t, s)
            } else {
                let t = -zz / x;
                let c = 1.0 / (1.0 + t * t).sqrt();
                (c, c * t)
            };
            if k > l {
                e[k - 1] = c * x - s * zz;
            }
            let (d1, d2, ek) = (d[k], d[k + 1], e[k]);
            d[k] = c * c * d1 + s * s * d2 - (c + c) * s * ek;
            d[k + 1] = s * s * d1 + c * c * d2 + (c + c) * s * ek;
            e[k] = c * s * (d1 - d2) + (c * c - s * s) * ek;
            if k < m - 1 {
                x = e[k];
                zz = -s * e[k + 1];
                e[k + 1] = c * e[k + 1];
            }
            for i in 0..z.nrows() {
                let (t1, t2) = (z[(i, k)], z[(i, k + 1)]);
                z[(i, k)] = c.mul_add(t1, -(s * t2));
                z[(i, k + 1)] = s.mul_add(t1, c * t2);
            }
        }
    }
}

#[test]
fn batched_rotations_match_immediate_application() {
    for deflation in [Deflation::Relaxed, Deflation::Strict] {
        batched_matches_immediate(deflation);
    }
}

fn batched_matches_immediate(deflation: Deflation) {
    for &n in &[2usize, 7, 40, 150, 333] {
        let mut rng = Prng(0xBEEF + n as u64);
        let d0: Vec<f64> = (0..n).map(|_| rng.next_f64() * 3.0).collect();
        let e0: Vec<f64> = (0..n - 1).map(|_| rng.next_f64()).collect();
        // A dense start matrix exercises every row block.
        let rows = n + 70;
        let mut z0 = Mat::zeros(rows, n);
        for j in 0..n {
            for i in 0..rows {
                z0[(i, j)] = rng.next_f64();
            }
        }
        let (mut d1, mut e1, mut z1) = (d0.clone(), e0.clone(), z0.clone());
        steqr_reference(&mut d1, &mut e1, &mut z1, deflation);
        let (mut d2, mut e2, mut z2) = (d0.clone(), e0.clone(), z0.clone());
        let ld = z2.row_stride();
        let mut log = RotationLog::new();
        let target = RotTarget {
            data: z2.raw_data_mut(),
            off: 0,
            ld,
            nrows: rows,
            identity: false,
        };
        steqr(&mut d2, &mut e2, Some(target), 100, deflation, &mut log)
            .unwrap_or_else(|_| panic!("n = {n}"));
        assert_eq!(d1, d2, "n = {n}: eigenvalues");
        for j in 0..n {
            for i in 0..rows {
                assert_eq!(z1[(i, j)], z2[(i, j)], "n = {n}: z({i}, {j})");
            }
        }
    }
}

#[test]
fn identity_start_skips_only_zero_rows() {
    // Starting from the identity, the support tracking skips rows that are
    // still zero; the result must match applying every rotation in full.
    for deflation in [Deflation::Relaxed, Deflation::Strict] {
        identity_start_matches(deflation);
    }
}

fn identity_start_matches(deflation: Deflation) {
    for &n in &[3usize, 30, 129, 260] {
        let mut rng = Prng(0xD00D + n as u64);
        let d0: Vec<f64> = (0..n).map(|_| rng.next_f64() * 3.0).collect();
        let mut e0: Vec<f64> = (0..n - 1).map(|_| rng.next_f64()).collect();
        if n > 100 {
            // A split in the middle exercises blocks that start below row 0.
            e0[n / 2] = 0.0;
        }
        let (mut d1, mut e1) = (d0.clone(), e0.clone());
        let mut z1: Mat<f64> = Mat::eye(n);
        steqr_reference(&mut d1, &mut e1, &mut z1, deflation);
        let (mut d2, mut e2) = (d0.clone(), e0.clone());
        let mut z2: Mat<f64> = Mat::eye(n);
        let ld = z2.row_stride();
        let mut log = RotationLog::new();
        let target = RotTarget {
            data: z2.raw_data_mut(),
            off: 0,
            ld,
            nrows: n,
            identity: true,
        };
        steqr(&mut d2, &mut e2, Some(target), 100, deflation, &mut log)
            .unwrap_or_else(|_| panic!("n = {n}"));
        assert_eq!(d1, d2, "n = {n}: eigenvalues");
        for j in 0..n {
            for i in 0..n {
                assert_eq!(z1[(i, j)], z2[(i, j)], "n = {n}: z({i}, {j})");
            }
        }
    }
}

#[test]
fn steqr_rejects_non_finite_input() {
    let mut d = vec![1.0, f64::NAN, 2.0];
    let mut e = vec![0.5, 0.5];
    let mut log = RotationLog::new();
    assert!(steqr(&mut d, &mut e, None, 100, Deflation::Strict, &mut log).is_err());
}

fn stedc_check(d0: &[f64], e0: &[f64], smlsiz: usize, label: &str) {
    let n = d0.len();
    let t = tridiag_matrix(d0, e0);
    let (mut d, mut e) = (d0.to_vec(), e0.to_vec());
    let mut z = Mat::zeros(n, n);
    stedc::<f64, BlasMatMul>(&mut d, &mut e, &mut z, smlsiz, Par::Seq)
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    assert!(d.windows(2).all(|w| w[0] <= w[1]), "{label}: not ascending");
    let (resid, orth) = full_accuracy(&t, &d, &z);
    let nf = n as f64;
    assert!(resid <= 1e-14 * nf, "{label}: residual {resid:e}");
    assert!(orth <= 1e-14 * nf, "{label}: orthogonality {orth:e}");
    // Eigenvalues agree with the QR iteration.
    let (mut dq, mut eq) = (d0.to_vec(), e0.to_vec());
    let mut log = RotationLog::new();
    steqr(&mut dq, &mut eq, None, 100, Deflation::Strict, &mut log)
        .unwrap_or_else(|_| panic!("{label}: steqr"));
    dq.sort_by(f64::total_cmp);
    let scale = dq.iter().fold(1e-300f64, |m, v| m.max(v.abs()));
    for i in 0..n {
        assert!(
            (d[i] - dq[i]).abs() <= 1e-13 * scale * nf.sqrt(),
            "{label}: eigenvalue {i}: {} vs {}",
            d[i],
            dq[i]
        );
    }
}

#[test]
fn stedc_random_and_forced_merges() {
    for &(n, smlsiz) in &[
        (2usize, 1usize),
        (3, 1),
        (17, 1),
        (64, 1),
        (100, 4),
        (257, 25),
        (300, 60),
    ] {
        let mut rng = Prng(0x1234 + n as u64);
        let d: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0).collect();
        let e: Vec<f64> = (0..n - 1).map(|_| rng.next_f64()).collect();
        stedc_check(&d, &e, smlsiz, &format!("random n={n} smlsiz={smlsiz}"));
    }
}

#[test]
fn stedc_structured_tridiagonals() {
    // Wilkinson W_n+: pairs of nearly equal eigenvalues (type-2 deflation).
    for &n in &[21usize, 101, 201] {
        let h = (n / 2) as f64;
        let d: Vec<f64> = (0..n).map(|i| (h - i as f64).abs()).collect();
        let e = vec![1.0; n - 1];
        stedc_check(&d, &e, 25, &format!("wilkinson {n}"));
        stedc_check(&d, &e, 4, &format!("wilkinson {n} small leaves"));
    }
    // (1, 2, 1) Toeplitz: known well-separated spectrum.
    let n = 150;
    stedc_check(&vec![2.0; n], &vec![1.0; n - 1], 10, "toeplitz");
    // Glued blocks: negligible couplings split the problem.
    let mut rng = Prng(99);
    let d: Vec<f64> = (0..120).map(|_| rng.next_f64()).collect();
    let mut e: Vec<f64> = (0..119).map(|_| rng.next_f64()).collect();
    for k in [10usize, 50, 51, 90] {
        e[k] = 1e-300;
    }
    stedc_check(&d, &e, 8, "glued");
    // All couplings zero (diagonal) and equal diagonal (all deflate).
    stedc_check(&d, &vec![0.0; 119], 8, "diagonal");
    stedc_check(&vec![3.0; 64], &vec![1e-18; 63], 4, "constant");
    // Graded diagonal over many orders of magnitude.
    let d: Vec<f64> = (0..80i32).map(|i| 10f64.powi(-i / 8)).collect();
    let e: Vec<f64> = (0..79i32).map(|i| 0.5 * 10f64.powi(-i / 8)).collect();
    stedc_check(&d, &e, 6, "graded");
    // Zero tridiagonal.
    stedc_check(&vec![0.0; 30], &vec![0.0; 29], 4, "zero");
}

/// Hard secular equations inside full divide-and-conquer runs.
#[test]
fn stedc_hard_secular_cases() {
    // Glued Wilkinson W21+: eight copies coupled by `glue` (above the split
    // threshold eps·10, so the matrix stays unreduced). With leaves of 25
    // rows every merge tears exactly at a glue (168 = 8·21), so the merges
    // have the tiny rho = glue·|z|² and pairs of (nearly) equal poles; with
    // leaves of 4 the tears fall inside the copies as well.
    let w21: Vec<f64> = (0..21).map(|i| (10.0 - i as f64).abs()).collect();
    for &glue in &[1e-14, 1e-12, 1e-10, 1e-6] {
        let mut d = Vec::new();
        let mut e = Vec::new();
        for copy in 0..8 {
            if copy > 0 {
                e.push(glue);
            }
            d.extend_from_slice(&w21);
            e.extend(std::iter::repeat_n(1.0, 20));
        }
        stedc_check(&d, &e, 25, &format!("glued wilkinson glue={glue:e}"));
        stedc_check(
            &d,
            &e,
            4,
            &format!("glued wilkinson glue={glue:e} small leaves"),
        );
    }
    // Tightly clustered spectrum: diagonal within 1e-10 of 1, couplings
    // about 1e-11 (clustered poles in every merge).
    let mut rng = Prng(0xC1A5);
    let n = 160;
    let d: Vec<f64> = (0..n).map(|_| 1.0 + 1e-10 * rng.next_f64()).collect();
    let e: Vec<f64> = (0..n - 1)
        .map(|_| 1e-11 * (0.5 + rng.next_f64().abs()))
        .collect();
    stedc_check(&d, &e, 8, "clustered");
    // Tiny couplings everywhere (tiny rho in every merge; most components
    // deflate, the rest have roots right next to their poles).
    let d: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let e: Vec<f64> = (0..n - 1)
        .map(|_| 1e-12 * (0.5 + rng.next_f64().abs()))
        .collect();
    stedc_check(&d, &e, 6, "tiny rho");
    // Graded over 18 orders of magnitude (and a mild grading with random
    // signs), huge dynamic range of the poles and weights.
    let d: Vec<f64> = (0..120i32).map(|i| 2f64.powi(-i / 2)).collect();
    let e: Vec<f64> = (0..119i32).map(|i| 0.5 * 2f64.powi(-i / 2)).collect();
    stedc_check(&d, &e, 6, "strongly graded");
    let d: Vec<f64> = (0..100i32)
        .map(|i| rng.next_f64() * 10f64.powi(-(i % 25) / 2))
        .collect();
    let e: Vec<f64> = (0..99i32)
        .map(|i| rng.next_f64() * 10f64.powi(-(i % 17) / 2))
        .collect();
    stedc_check(&d, &e, 5, "random graded");
}

#[test]
fn stedc_generic_backend_matches() {
    let n = 140;
    let mut rng = Prng(0xFACE);
    let d0: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let e0: Vec<f64> = (0..n - 1).map(|_| rng.next_f64()).collect();
    let (mut d1, mut e1) = (d0.clone(), e0.clone());
    let mut z1 = Mat::zeros(n, n);
    stedc::<f64, BlasMatMul>(&mut d1, &mut e1, &mut z1, 16, Par::Seq)
        .unwrap_or_else(|e| panic!("{e:?}"));
    let (mut d2, mut e2) = (d0.clone(), e0.clone());
    let mut z2 = Mat::zeros(n, n);
    stedc::<f64, NaiveMatMul>(&mut d2, &mut e2, &mut z2, 16, Par::Seq)
        .unwrap_or_else(|e| panic!("{e:?}"));
    for i in 0..n {
        assert!((d1[i] - d2[i]).abs() <= 1e-14, "eigenvalue {i}");
        // Same eigenvectors up to sign.
        let dotp: f64 = (0..n).map(|r| z1[(r, i)] * z2[(r, i)]).sum();
        assert!(
            (dotp.abs() - 1.0).abs() <= 1e-12,
            "eigenvector {i}: |dot| = {dotp}"
        );
    }
}

#[test]
fn argsort_and_permute_columns() {
    let vals = [3.0f64, -1.0, 2.0, -1.0, 0.0];
    let mut perm = Vec::new();
    argsort(&vals, &mut perm);
    assert_eq!(perm, vec![1, 3, 4, 2, 0]);
    let mut m = Mat::<f64>::zeros(2, 5);
    for j in 0..5 {
        m[(0, j)] = j as f64;
        m[(1, j)] = 10.0 * j as f64;
    }
    let ld = m.row_stride();
    let (mut saved, mut visited) = (Vec::new(), Vec::new());
    permute_columns(m.raw_data_mut(), 0, ld, 2, &perm, &mut saved, &mut visited);
    for (j, &p) in perm.iter().enumerate() {
        assert_eq!(m[(0, j)], p as f64);
        assert_eq!(m[(1, j)], 10.0 * p as f64);
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

#[test]
fn drivers_large_random_n500() {
    check_both(&random_symmetric(500, 0x500), 1e-14, "random n=500");
}

#[test]
fn drivers_large_random_n1000() {
    check_both(&random_symmetric(1000, 0x1000), 1e-14, "random n=1000");
}

#[test]
fn drivers_clustered_benchmark_matrix() {
    for &n in &[50usize, 200, 300] {
        check_both(&bench_matrix(n), 1e-14, &format!("bench matrix n={n}"));
    }
}

#[test]
fn drivers_known_clustered_spectrum() {
    // Tight clusters, exact repeats and well separated values.
    let n = 240;
    let eigs: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 => 1.0 + (i as f64) * 1e-13,
            1 => -2.0,
            2 => 5.0 + (i as f64) * 1e-9,
            _ => i as f64 / 7.0,
        })
        .collect();
    let a = with_spectrum(&eigs, 0x5EED);
    let (qr, dc) = check_both(&a, 1e-14, "clustered spectrum");
    let mut sorted = eigs.clone();
    sorted.sort_by(f64::total_cmp);
    for i in 0..n {
        assert!((qr[i] - sorted[i]).abs() <= 1e-12, "QR eigenvalue {i}");
        assert!((dc[i] - sorted[i]).abs() <= 1e-12, "DC eigenvalue {i}");
    }
}

#[test]
fn drivers_diagonal_identity_zero() {
    for &n in &[2usize, 3, 5, 50, 180] {
        let mut rng = Prng(n as u64);
        let diag: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0).collect();
        let mut sorted = diag.clone();
        sorted.sort_by(f64::total_cmp);
        let (qr, dc) = check_both(&Mat::diag(&diag), 1e-15, "diagonal");
        assert_eq!(qr, sorted, "QR diagonal n = {n}");
        assert_eq!(dc, sorted, "DC diagonal n = {n}");

        let eye: Mat<f64> = Mat::eye(n);
        let (qr, dc) = check_both(&eye, 1e-15, "identity");
        assert!(
            qr.iter().chain(dc.iter()).all(|&v| v == 1.0),
            "identity n = {n}"
        );

        let zero: Mat<f64> = Mat::zeros(n, n);
        let (qr, dc) = check_both(&zero, 1e-15, "zero");
        assert!(
            qr.iter().chain(dc.iter()).all(|&v| v == 0.0),
            "zero n = {n}"
        );
        let v = SymmetricEvdDc::compute(zero.as_ref()).unwrap_or_else(|e| panic!("{e}"));
        let v = v.eigenvectors();
        for j in 0..n {
            for i in 0..n {
                assert_eq!(v[(i, j)], if i == j { 1.0 } else { 0.0 });
            }
        }
    }
}

#[test]
fn drivers_rank_deficient() {
    // A = B·Bᵀ with B of rank 7: n - 7 zero eigenvalues.
    let (n, r) = (200usize, 7usize);
    let mut rng = Prng(0xDEF);
    let mut b = Mat::zeros(n, r);
    for j in 0..r {
        for i in 0..n {
            b[(i, j)] = rng.next_f64();
        }
    }
    let mut bt = Mat::zeros(r, n);
    for j in 0..r {
        for i in 0..n {
            bt[(j, i)] = b[(i, j)];
        }
    }
    let a = matmul(&b, &bt);
    let (qr, dc) = check_both(&a, 1e-14, "rank deficient");
    let scale = qr[n - 1];
    for vals in [&qr, &dc] {
        let zeros = vals.iter().filter(|v| v.abs() <= 1e-12 * scale).count();
        assert_eq!(zeros, n - r);
    }
}

#[test]
fn drivers_wilkinson_and_tridiagonal_input() {
    for &n in &[21usize, 201] {
        let h = (n / 2) as f64;
        let d: Vec<f64> = (0..n).map(|i| (h - i as f64).abs()).collect();
        check_both(&tridiag_matrix(&d, &vec![1.0; n - 1]), 1e-14, "wilkinson");
    }
}

#[test]
fn drivers_read_only_the_upper_triangle() {
    let n = 210;
    let a = random_symmetric(n, 0x0DD);
    let mut garbage = a.clone();
    for j in 0..n {
        for i in (j + 1)..n {
            garbage[(i, j)] = f64::NAN;
        }
    }
    let qr1 = SymmetricEvd::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    let qr2 = SymmetricEvd::compute(garbage.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(qr1.eigenvalues(), qr2.eigenvalues());
    let dc1 = SymmetricEvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    let dc2 = SymmetricEvdDc::compute(garbage.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(dc1.eigenvalues(), dc2.eigenvalues());
    for j in 0..n {
        for i in 0..n {
            assert_eq!(qr1.eigenvectors()[(i, j)], qr2.eigenvectors()[(i, j)]);
            assert_eq!(dc1.eigenvectors()[(i, j)], dc2.eigenvectors()[(i, j)]);
        }
    }
    // A strided (transposed) view of the same symmetric matrix.
    let at = a.as_ref().transpose();
    let qr3 = SymmetricEvd::compute(at).unwrap_or_else(|e| panic!("{e}"));
    let dc3 = SymmetricEvdDc::compute(at).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(qr1.eigenvalues(), qr3.eigenvalues());
    assert_eq!(dc1.eigenvalues(), dc3.eigenvalues());
}

#[test]
fn drivers_extreme_scales() {
    // Entries near the overflow / underflow thresholds: the drivers scale
    // into the safe range, so the spectrum scales exactly with the input.
    let n = 190;
    let base = random_symmetric(n, 0x5CA1E);
    let reference = SymmetricEvdDc::compute(base.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    for &factor in &[2f64.powi(1000), 2f64.powi(-1000), 1e300, 1e-300] {
        let mut a = base.clone();
        for j in 0..n {
            for i in 0..n {
                a[(i, j)] *= factor;
            }
        }
        let qr = SymmetricEvd::compute(a.as_ref()).unwrap_or_else(|e| panic!("{factor:e}: {e}"));
        let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("{factor:e}: {e}"));
        let scale = reference
            .eigenvalues()
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        for (name, vals, vecs) in [
            ("QR", qr.eigenvalues(), owned(qr.eigenvectors())),
            ("DC", dc.eigenvalues(), owned(dc.eigenvectors())),
        ] {
            for (i, (&x, &y)) in vals.iter().zip(reference.eigenvalues()).enumerate() {
                assert!(
                    (x / factor - y).abs() <= 1e-12 * scale,
                    "{factor:e} {name}: eigenvalue {i}: {} vs {y}",
                    x / factor
                );
            }
            let (resid, orth) = full_accuracy(&base, reference.eigenvalues(), &vecs);
            assert!(
                resid <= 1e-12 && orth <= 1e-11,
                "{factor:e} {name}: {resid:e} {orth:e}"
            );
        }
    }
}

#[test]
fn drivers_non_finite_input_is_an_error() {
    let mut a = random_symmetric(40, 3);
    a[(3, 10)] = f64::NAN;
    assert!(SymmetricEvd::compute(a.as_ref()).is_err());
    assert!(SymmetricEvdDc::compute(a.as_ref()).is_err());
}

#[test]
fn drivers_f32() {
    let n = 260;
    let a64 = random_symmetric(n, 0xF32);
    let mut a = Mat::<f32>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = a64[(i, j)] as f32;
        }
    }
    let reference = SymmetricEvdDc::compute(a64.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    let qr = SymmetricEvd::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    for (name, vals, vecs) in [
        ("QR", qr.eigenvalues(), qr.eigenvectors()),
        ("DC", dc.eigenvalues(), dc.eigenvectors()),
    ] {
        let mut vecs64 = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                vecs64[(i, j)] = f64::from(vecs[(i, j)]);
            }
        }
        let vals64: Vec<f64> = vals.iter().map(|&v| f64::from(v)).collect();
        let (resid, orth) = full_accuracy(&a64, &vals64, &vecs64);
        assert!(resid <= 1e-5, "f32 {name}: residual {resid:e}");
        assert!(orth <= 1e-4, "f32 {name}: orthogonality {orth:e}");
        // Single-precision backward error: about 100·eps₃₂ relative.
        let scale = reference
            .eigenvalues()
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (&x, &y)) in vals64.iter().zip(reference.eigenvalues()).enumerate() {
            assert!(
                (x - y).abs() <= 5e-5 * scale,
                "f32 {name}: eigenvalue {i}: {x} vs {y}"
            );
        }
    }
}

/// Bitwise equality of two decompositions (eigenvalues and eigenvectors).
#[cfg(feature = "parallel")]
fn assert_same_decomposition<T: Copy>(
    label: &str,
    bits: fn(T) -> u64,
    (vals, vecs): (&[T], oxiblas_matrix::MatRef<'_, T>),
    (want_vals, want_vecs): (&[T], oxiblas_matrix::MatRef<'_, T>),
) where
    T: oxiblas_core::scalar::Scalar,
{
    assert_eq!(vals.len(), want_vals.len(), "{label}: eigenvalue count");
    for (i, (&x, &y)) in vals.iter().zip(want_vals).enumerate() {
        assert_eq!(bits(x), bits(y), "{label}: eigenvalue {i}");
    }
    for j in 0..want_vecs.ncols() {
        for i in 0..want_vecs.nrows() {
            assert_eq!(
                bits(vecs[(i, j)]),
                bits(want_vecs[(i, j)]),
                "{label}: eigenvector entry ({i}, {j})"
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn drivers_with_par_match_compute_bitwise() {
    // Large enough for the parallel GEMM path in the reduction, the merges
    // and the back-transformation; that path only partitions its output, so
    // every `Par` must reproduce `compute` bit for bit.
    for &n in &[300usize, 420] {
        let a = random_symmetric(n, 0xA11 + n as u64);
        let qr = SymmetricEvd::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
        let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
        for par in [Par::Rayon, Par::RayonWith(3), Par::Seq] {
            let qr_par =
                SymmetricEvd::compute_with_par(a.as_ref(), par).unwrap_or_else(|e| panic!("{e}"));
            let dc_par =
                SymmetricEvdDc::compute_with_par(a.as_ref(), par).unwrap_or_else(|e| panic!("{e}"));
            assert_same_decomposition(
                &format!("SymmetricEvd n = {n}, {par:?}"),
                f64::to_bits,
                (qr_par.eigenvalues(), qr_par.eigenvectors()),
                (qr.eigenvalues(), qr.eigenvectors()),
            );
            assert_same_decomposition(
                &format!("SymmetricEvdDc n = {n}, {par:?}"),
                f64::to_bits,
                (dc_par.eigenvalues(), dc_par.eigenvectors()),
                (dc.eigenvalues(), dc.eigenvectors()),
            );
        }
    }
    let n = 330;
    let a64 = random_symmetric(n, 0xF33);
    let mut a = Mat::<f32>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = a64[(i, j)] as f32;
        }
    }
    let to_bits: fn(f32) -> u64 = |x| u64::from(x.to_bits());
    let qr = SymmetricEvd::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    let qr_par =
        SymmetricEvd::compute_with_par(a.as_ref(), Par::Rayon).unwrap_or_else(|e| panic!("{e}"));
    assert_same_decomposition(
        "f32 SymmetricEvd",
        to_bits,
        (qr_par.eigenvalues(), qr_par.eigenvectors()),
        (qr.eigenvalues(), qr.eigenvectors()),
    );
    let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap_or_else(|e| panic!("{e}"));
    let dc_par =
        SymmetricEvdDc::compute_with_par(a.as_ref(), Par::Rayon).unwrap_or_else(|e| panic!("{e}"));
    assert_same_decomposition(
        "f32 SymmetricEvdDc",
        to_bits,
        (dc_par.eigenvalues(), dc_par.eigenvectors()),
        (dc.eigenvalues(), dc.eigenvectors()),
    );
}
