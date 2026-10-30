//! Unit tests of the QZ decomposition and of its LAPACK kernels.

use oxiblas_matrix::Mat;

use super::balance::{ggbak_p, ggbal_p};
use super::hessenberg::{gghrd, qr_reduce_b};
use super::lag2::{dlag2, dlasv2};
use super::pipeline::{QzWork, run};
use super::scaling::{ScaleKind, lascl_mat, lascl_slice};
use super::test_util::{
    THRESH, lcg, lcg_matrix, mat_to_f64, matmul, matmul_bt, orthogonality_ratio, residual_ratio,
};
use super::util::{Consts, larfg, lartg};
use super::{Qz, QzError};

#[test]
fn test_qz_identity() {
    let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);
    let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

    let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();

    // Eigenvalues should be 1 and 2
    let mut eigs: Vec<f64> = qz.eigenvalues_real();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((eigs[0] - 1.0).abs() < 1e-6, "Expected 1, got {}", eigs[0]);
    assert!((eigs[1] - 2.0).abs() < 1e-6, "Expected 2, got {}", eigs[1]);
}

#[test]
fn test_qz_scaled() {
    let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);
    let b = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 2.0]]);

    let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();

    // Eigenvalues should be 1 (2/2) and 2 (4/2)
    let mut eigs: Vec<f64> = qz.eigenvalues_real();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((eigs[0] - 1.0).abs() < 1e-6);
    assert!((eigs[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_qz_verify() {
    // Simple diagonal case for verification
    let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);
    let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

    let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();
    let (error_a, error_b) = qz.verify(a.as_ref(), b.as_ref());

    // With diagonal A and B = I, errors should be small
    assert!(
        error_a < 1e-6,
        "Verification error for A: {} (should be < 1e-6)",
        error_a
    );
    assert!(
        error_b < 1e-6,
        "Verification error for B: {} (should be < 1e-6)",
        error_b
    );
}

#[test]
fn test_qz_1x1() {
    let a = Mat::from_rows(&[&[5.0f64]]);
    let b = Mat::from_rows(&[&[2.0f64]]);

    let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();
    let eigs = qz.eigenvalues();

    assert_eq!(eigs.len(), 1);
    assert!((eigs[0].value().unwrap() - 2.5).abs() < 1e-10);
}

#[test]
fn test_qz_error_display() {
    let msg = QzError::NotConverged { iterations: 90 }.to_string();
    assert_eq!(msg, "Did not converge after 90 iterations");
    assert_eq!(QzError::EmptyMatrix.to_string(), "Empty matrix");
}

#[test]
fn test_lartg_properties() {
    let k = Consts::<f64>::new();
    let (c, s, r) = lartg(3.0, 4.0, &k);
    assert!((c - 0.6).abs() < 1e-15 && (s - 0.8).abs() < 1e-15 && (r - 5.0).abs() < 1e-15);
    // r carries the sign of f and c >= 0 (LAPACK 3.10 semantics).
    let (c, s, r) = lartg(-3.0, 4.0, &k);
    assert!(c >= 0.0 && r < 0.0);
    assert!((c * -3.0 + s * 4.0 - r).abs() < 1e-14 && (-s * -3.0 + c * 4.0).abs() < 1e-14);
    assert_eq!(lartg(2.5, 0.0, &k), (1.0, 0.0, 2.5));
    assert_eq!(lartg(0.0, -2.0, &k), (0.0, -1.0, 2.0));
    // Safe scaling for huge and tiny arguments.
    for &(f, g) in &[(1e300, 3e300), (1e-300, -4e-300), (1e-170, 1e170)] {
        let (c, s, r) = lartg(f, g, &k);
        assert!(r.is_finite() && r != 0.0, "r = {r} for ({f}, {g})");
        assert!((c * c + s * s - 1.0).abs() < 1e-15);
        let resid = (-s * f + c * g).abs() / f64::max(f.abs(), g.abs());
        assert!(resid < 1e-15, "({f}, {g}): residual {resid}");
    }
}

#[test]
fn test_larfg_annihilates() {
    let k = Consts::<f64>::new();
    for &scale in &[1.0f64, 1e-300] {
        let x0 = [0.3 * scale, -1.2 * scale, 0.7 * scale];
        let mut alpha = 0.5 * scale;
        let mut v = x0;
        let tau = larfg(&mut alpha, &mut v, &k);
        // H·[0.5; x0] must equal [beta; 0] with H = I − tau·u·uᵀ, u = [1; v].
        let full = [0.5 * scale, x0[0], x0[1], x0[2]];
        let u = [1.0, v[0], v[1], v[2]];
        let dot: f64 = u.iter().zip(full.iter()).map(|(a, b)| a * b).sum();
        let hx: Vec<f64> = (0..4).map(|i| full[i] - tau * u[i] * dot).collect();
        // Scaled norm: the squares of 1e-300 entries would underflow.
        let norm = scale
            * full
                .iter()
                .map(|x| (x / scale) * (x / scale))
                .sum::<f64>()
                .sqrt();
        assert!(
            (hx[0] - alpha).abs() <= 1e-14 * norm,
            "beta mismatch at scale {scale}"
        );
        for (i, value) in hx.iter().enumerate().skip(1) {
            assert!(
                value.abs() <= 1e-14 * norm,
                "entry {i} not annihilated: {value}"
            );
        }
        assert!((alpha.abs() - norm).abs() <= 1e-14 * norm);
    }
    let mut alpha = 2.0f64;
    let mut zero = [0.0f64; 2];
    assert_eq!(larfg(&mut alpha, &mut zero, &k), 0.0);
}

#[test]
fn test_dlag2_matches_closed_form() {
    let k = Consts::<f64>::new();
    let safmin = k.safmin * k.safety;
    // Real pair: det(A − λB) = (1 − 2λ)(4 − 3λ) − (2 − λ)·3 = 6λ² − 8λ − 2.
    let g = dlag2(
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 1.0], [0.0, 3.0]],
        safmin,
        &k,
    );
    assert_eq!(g.wi, 0.0);
    let mut got = [g.wr1 / g.scale1, g.wr2 / g.scale2];
    got.sort_by(f64::total_cmp);
    let s7 = 7.0f64.sqrt();
    assert!((got[0] - (2.0 - s7) / 3.0).abs() < 1e-14);
    assert!((got[1] - (2.0 + s7) / 3.0).abs() < 1e-14);
    // Complex pair: 4λ² − 5λ + 7 = 0.
    let g = dlag2(
        [[1.0, -2.0], [3.0, 1.0]],
        [[1.0, 0.0], [0.0, 4.0]],
        safmin,
        &k,
    );
    assert!(g.wi > 0.0);
    assert!((g.wr1 / g.scale1 - 0.625).abs() < 1e-14);
    assert!((g.wi / g.scale1 - 87.0f64.sqrt() / 8.0).abs() < 1e-14);
}

#[test]
fn test_dlasv2_reconstructs() {
    let k = Consts::<f64>::new();
    let mut state = 7u64;
    for case in 0..50 {
        let (f, g, h) = if case == 0 {
            (2.0, 0.0, -3.0)
        } else if case == 1 {
            (1e-20, 1.0, 2e-20)
        } else {
            (lcg(&mut state), lcg(&mut state), lcg(&mut state))
        };
        let sv = dlasv2(f, g, h, &k);
        // [csl snl; −snl csl] · [f g; 0 h] · [csr −snr; snr csr].
        let l = [[sv.csl, sv.snl], [-sv.snl, sv.csl]];
        let r = [[sv.csr, -sv.snr], [sv.snr, sv.csr]];
        let m = [[f, g], [0.0, h]];
        let mut lm = [[0.0f64; 2]; 2];
        let mut out = [[0.0f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                lm[i][j] = l[i][0] * m[0][j] + l[i][1] * m[1][j];
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                out[i][j] = lm[i][0] * r[0][j] + lm[i][1] * r[1][j];
            }
        }
        let scale = f.abs().max(g.abs()).max(h.abs());
        let tol = 8.0 * f64::EPSILON * scale;
        assert!((out[0][0] - sv.ssmax).abs() <= tol, "case {case}: ssmax");
        assert!((out[1][1] - sv.ssmin).abs() <= tol, "case {case}: ssmin");
        assert!(
            out[0][1].abs() <= tol && out[1][0].abs() <= tol,
            "case {case}: off-diagonal"
        );
        assert!(sv.ssmax.abs() >= sv.ssmin.abs());
    }
}

#[test]
fn test_lascl_extreme_ratio() {
    let k = Consts::<f64>::new();
    // cto / cfrom = 1e600 is not representable; DLASCL must step.
    let mut v = [1e-300f64, -2e-300];
    lascl_slice(1e-300, 1e300, &mut v, &k);
    assert!((v[0] - 1e300).abs() <= 1e-14 * 1e300, "{v:?}");
    assert!((v[1] + 2e300).abs() <= 1e-14 * 2e300, "{v:?}");
    // And back down: cto / cfrom = 1e-600 steps through SMLNUM factors.
    lascl_slice(1e300, 1e-300, &mut v, &k);
    assert!((v[0] - 1e-300).abs() <= 1e-14 * 1e-300, "{v:?}");
    assert!((v[1] + 2e-300).abs() <= 1e-14 * 2e-300, "{v:?}");
    // Upper Hessenberg kind leaves entries below the subdiagonal untouched.
    let mut m = Mat::<f64>::filled(3, 3, 1.0);
    lascl_mat(ScaleKind::UpperHessenberg, 1.0, 4.0, &mut m, &k);
    assert_eq!(m[(2, 0)], 1.0);
    assert_eq!(m[(1, 0)], 4.0);
    assert_eq!(m[(0, 2)], 4.0);
    let mut m = Mat::<f64>::filled(3, 3, 1.0);
    lascl_mat(ScaleKind::UpperTriangular, 2.0, 1.0, &mut m, &k);
    assert_eq!(m[(1, 0)], 1.0);
    assert_eq!(m[(1, 1)], 0.5);
}

/// The block upper triangular test pencil of `test_issue_6_isolated_by_permutation`
/// in a fixed scrambled form, with its isolated eigenvalue positions.
fn permuted_block_pencil() -> (Mat<f64>, Mat<f64>) {
    let n = 5;
    let mut a0 = lcg_matrix(n, 91);
    let mut b0 = lcg_matrix(n, 92);
    for j in 0..n {
        for i in 0..n {
            // Row 0 / column 0 isolated at the top, row/column 4 at the bottom.
            if (i > 0 && j == 0) || (i == 4 && j < 4) {
                a0[(i, j)] = 0.0;
                b0[(i, j)] = 0.0;
            }
        }
    }
    let rows = [2usize, 4, 0, 3, 1];
    let cols = [3usize, 0, 4, 1, 2];
    let mut a = Mat::<f64>::zeros(n, n);
    let mut b = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = a0[(rows[i], cols[j])];
            b[(i, j)] = b0[(rows[i], cols[j])];
        }
    }
    (a, b)
}

#[test]
fn test_ggbal_isolates_and_ggbak_restores() {
    let (a, b) = permuted_block_pencil();
    let n = a.nrows();
    let (mut ab, mut bb) = (a.clone(), b.clone());
    let perm = ggbal_p(&mut ab, &mut bb);
    assert_eq!((perm.ilo, perm.ihi), (1, 3));
    // Block structure: zeros below the diagonal outside the middle block.
    for j in 0..n {
        for i in (j + 1)..n {
            if j < perm.ilo || i > perm.ihi {
                assert!(ab[(i, j)] == 0.0 && bb[(i, j)] == 0.0, "({i},{j}) not zero");
            }
        }
    }
    // DGGBAK applied to I yields the permutation matrices: Plᵀ·Ã·Prᵀ = A.
    let mut pl = Mat::<f64>::eye(n);
    let mut pr = Mat::<f64>::eye(n);
    ggbak_p(&mut pl, perm.ilo, perm.ihi, &perm.lperm);
    ggbak_p(&mut pr, perm.ilo, perm.ihi, &perm.rperm);
    let back = matmul_bt(&matmul(&pl, &ab), &pr);
    for i in 0..n {
        for j in 0..n {
            assert_eq!(back[(i, j)], a[(i, j)], "entry ({i},{j})");
        }
    }
}

#[test]
fn test_pipeline_reports_balanced_block() {
    let (a, b) = permuted_block_pencil();
    let work = run(a.as_ref(), b.as_ref()).unwrap();
    assert_eq!((work.ilo, work.ihi), (1, 3));
    assert!(work.ascale.is_none() && work.bscale.is_none());
    let qz = work.finish();
    assert!(residual_ratio(a.as_ref(), qz.q(), qz.s(), qz.z()) <= THRESH);
    assert!(residual_ratio(b.as_ref(), qz.q(), qz.t(), qz.z()) <= THRESH);
}

#[test]
fn test_pipeline_records_scaling() {
    let a = lcg_matrix(3, 5);
    let b = lcg_matrix(3, 6);
    let mut a_small = a.clone();
    for j in 0..3 {
        for i in 0..3 {
            a_small[(i, j)] *= 1e-200;
        }
    }
    let work = run(a_small.as_ref(), b.as_ref()).unwrap();
    let k = Consts::<f64>::new();
    let (anrm, anrmto) = work.ascale.unwrap();
    assert_eq!(anrmto, k.smlnum);
    assert!(anrm > 0.0 && anrm < k.smlnum);
    assert!(work.bscale.is_none());
}

#[test]
fn test_hessenberg_triangular_reduction() {
    let n = 7;
    let k = Consts::<f64>::new();
    let a = lcg_matrix(n, 11);
    let b = lcg_matrix(n, 12);
    let mut work = QzWork {
        h: a.clone(),
        t: b.clone(),
        q: Mat::eye(n),
        z: Mat::eye(n),
        alphar: vec![0.0; n],
        alphai: vec![0.0; n],
        beta: vec![0.0; n],
        ilo: 0,
        ihi: n - 1,
        lperm: (0..n).collect(),
        rperm: (0..n).collect(),
        ascale: None,
        bscale: None,
    };
    qr_reduce_b(&mut work, &k);
    gghrd(&mut work, &k);
    for j in 0..n {
        for i in (j + 1)..n {
            assert_eq!(work.t[(i, j)], 0.0, "T({i},{j})");
            if i > j + 1 {
                assert_eq!(work.h[(i, j)], 0.0, "H({i},{j})");
            }
        }
    }
    assert!(
        residual_ratio(
            a.as_ref(),
            work.q.as_ref(),
            work.h.as_ref(),
            work.z.as_ref()
        ) <= THRESH
    );
    assert!(
        residual_ratio(
            b.as_ref(),
            work.q.as_ref(),
            work.t.as_ref(),
            work.z.as_ref()
        ) <= THRESH
    );
    assert!(orthogonality_ratio(work.q.as_ref()) <= THRESH);
    assert!(orthogonality_ratio(work.z.as_ref()) <= THRESH);
    // Sanity: the factors are genuinely non-trivial.
    let q64 = mat_to_f64(work.q.as_ref());
    assert!((0..n).any(|i| (q64[(i, i)] - 1.0).abs() > 1e-3));
}
