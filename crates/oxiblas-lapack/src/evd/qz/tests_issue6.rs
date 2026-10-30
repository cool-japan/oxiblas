//! Regression tests for GitHub issue #6: `Qz::compute` must behave like
//! LAPACK `DGGES` (JOBVSL = JOBVSR = 'V', SORT = 'N') beyond diagonal
//! pencils.
//!
//! Every case checks the full `DGGES` contract with the LAPACK `ddrges`
//! threshold (`THRESH = 30`): `A = Q S Zᵀ`, `B = Q T Zᵀ`, orthogonal `Q`/`Z`,
//! generalized real Schur structure, and (where an independent oracle is
//! available) the eigenvalues by chordal distance.

use oxiblas_matrix::Mat;

use super::test_util::{
    Triple, assert_dgges_contract, eig_triples, lcg, lcg_matrix, mat_from_f64, matmul, matmul_bt,
    max_matched_chordal, random_orthogonal, residual_ratio, scaled, to_f64,
};
use super::{Qz, QzError};
use crate::evd::GeneralEvd;
use crate::lu::Lu;

/// Chordal tolerance for eigenvalues compared against an `f64` oracle.
const CHORDAL_TOL: f64 = 1e-8;

/// Eigenvalues of a standard problem `M x = λ x` (Schur-based oracle).
fn standard_oracle(m: &Mat<f64>) -> Vec<Triple> {
    let evd = GeneralEvd::eigenvalues_only(m.as_ref())
        .unwrap_or_else(|e| panic!("oracle GeneralEvd failed: {e:?}"));
    evd.eigenvalues()
        .iter()
        .map(|e| (e.real, e.imag, 1.0))
        .collect()
}

/// `B⁻¹ A` (oracle input for well-conditioned `B`).
fn binv_a(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let lu = Lu::compute(b.as_ref()).unwrap_or_else(|e| panic!("oracle LU failed: {e:?}"));
    lu.solve(a.as_ref())
        .unwrap_or_else(|e| panic!("oracle solve failed: {e:?}"))
}

/// Eigenvalues of the pencil through `B⁻¹ A`.
fn generalized_oracle(a: &Mat<f64>, b: &Mat<f64>) -> Vec<Triple> {
    standard_oracle(&binv_a(a, b))
}

/// Eigenvalues of the pencil through the Möbius-transformed standard
/// problem `M = (B − τA)⁻¹ A`, whose eigenvalues `κ = λ / (1 − τλ)` are all
/// finite (an infinite `λ` maps to `κ = −1/τ`), so a singular `B` does not
/// produce the exact zero eigenvalues the Schur oracle struggles with.
///
/// `λ = κ / d` with `d = 1 + τκ` is returned in homogeneous form
/// `(κ·conj(d), |d|²)`, which stays meaningful when `d ≈ 0`. Several `τ`
/// are tried until `B − τA` is invertible and the oracle converges.
fn mobius_oracle(a: &Mat<f64>, b: &Mat<f64>) -> Vec<Triple> {
    let n = a.nrows();
    for &tau in &[0.37f64, -0.61, 1.13, -1.7, 0.05] {
        let mut shifted = b.clone();
        for j in 0..n {
            for i in 0..n {
                shifted[(i, j)] -= tau * a[(i, j)];
            }
        }
        let Ok(lu) = Lu::compute(shifted.as_ref()) else {
            continue;
        };
        let Ok(m) = lu.solve(a.as_ref()) else {
            continue;
        };
        let Ok(evd) = GeneralEvd::eigenvalues_only(m.as_ref()) else {
            continue;
        };
        return evd
            .eigenvalues()
            .iter()
            .map(|e| {
                let (kr, ki) = (e.real, e.imag);
                let (dr, di) = (1.0 + tau * kr, tau * ki);
                if dr == 0.0 && di == 0.0 {
                    // d = 0 exactly: λ = κ / 0 is infinite.
                    (1.0, 0.0, 0.0)
                } else {
                    (kr * dr + ki * di, ki * dr - kr * di, dr * dr + di * di)
                }
            })
            .collect();
    }
    panic!("no Möbius shift gave a convergent oracle");
}

/// Asserts the eigenvalues of `qz` match `reference` within `tol`.
fn assert_eigs_close(label: &str, computed: &[Triple], reference: &[Triple], tol: f64) {
    let d = max_matched_chordal(computed, reference);
    assert!(
        d <= tol,
        "{label}: chordal eigenvalue distance {d:.3e} > {tol:.1e}\n computed  {computed:?}\n reference {reference:?}"
    );
}

/// Asserts `Σ λ = trace(B⁻¹ A)` for a pencil with only finite eigenvalues.
fn assert_trace(label: &str, qz: &Qz<f64>, m: &Mat<f64>) {
    let n = m.nrows();
    let tr: f64 = (0..n).map(|i| m[(i, i)]).sum();
    let (mut sr, mut si, mut mag) = (0.0f64, 0.0f64, 0.0f64);
    for e in qz.eigenvalues() {
        let (vr, vi) = e
            .value_complex()
            .unwrap_or_else(|| panic!("{label}: unexpected infinite eigenvalue"));
        sr += vr;
        si += vi;
        mag += vr.hypot(vi);
    }
    let tol = 1e-9 * mag.max(1.0);
    assert!(
        (sr - tr).abs() <= tol && si.abs() <= tol,
        "{label}: eigenvalue sum ({sr}, {si}) vs trace {tr}"
    );
}

/// Computes the QZ decomposition, panicking with context on failure.
fn qz_of(label: &str, a: &Mat<f64>, b: &Mat<f64>) -> Qz<f64> {
    Qz::compute(a.as_ref(), b.as_ref()).unwrap_or_else(|e| panic!("{label}: Qz failed: {e:?}"))
}

/// Random pencil test: contract, `B⁻¹ A` oracle and trace identity.
fn check_random(n: usize, b_shift: f64) {
    let label = format!("random n={n}");
    let a = lcg_matrix(n, 100 + n as u64);
    let mut b = lcg_matrix(n, 200 + n as u64);
    for i in 0..n {
        b[(i, i)] += b_shift;
    }
    let qz = qz_of(&label, &a, &b);
    assert_dgges_contract(&label, a.as_ref(), b.as_ref(), &qz);
    assert_eq!(
        qz.num_infinite(),
        0,
        "{label}: spurious infinite eigenvalue"
    );
    let m = binv_a(&a, &b);
    assert_eigs_close(&label, &eig_triples(&qz), &standard_oracle(&m), CHORDAL_TOL);
    assert_trace(&label, &qz, &m);
}

/// Pencil `(Q0 A0 Z0ᵀ, Q0 B0 Z0ᵀ)` with a prescribed spectrum: `A0` is block
/// upper triangular with 1×1 and `[[x, y], [−y, x]]` diagonal blocks, `B0`
/// is upper triangular with `d·I` on each diagonal block, so the eigenvalues
/// are exactly `(x ± i y)/d` and `x/d`.
fn known_spectrum_pencil(n: usize, seed: u64, offdiag: f64) -> (Mat<f64>, Mat<f64>, Vec<Triple>) {
    let mut a0 = Mat::<f64>::zeros(n, n);
    let mut b0 = Mat::<f64>::zeros(n, n);
    let mut spectrum = Vec::with_capacity(n);
    let mut block_end = vec![0usize; n];
    let (mut i, mut blk) = (0usize, 0usize);
    while i < n {
        let d = 0.5 + 0.25 * (blk % 3) as f64;
        if blk % 2 == 0 && i + 1 < n {
            let x = -3.0 + 0.4 * blk as f64;
            let y = 0.3 + 0.15 * blk as f64;
            a0[(i, i)] = x;
            a0[(i, i + 1)] = y;
            a0[(i + 1, i)] = -y;
            a0[(i + 1, i + 1)] = x;
            b0[(i, i)] = d;
            b0[(i + 1, i + 1)] = d;
            spectrum.push((x / d, y / d, 1.0));
            spectrum.push((x / d, -y / d, 1.0));
            block_end[i] = i + 2;
            block_end[i + 1] = i + 2;
            i += 2;
        } else {
            let x = 1.0 + 0.5 * blk as f64;
            a0[(i, i)] = x;
            b0[(i, i)] = d;
            spectrum.push((x / d, 0.0, 1.0));
            block_end[i] = i + 1;
            i += 1;
        }
        blk += 1;
    }
    let mut state = seed ^ 0x51_5151;
    for r in 0..n {
        for c in block_end[r]..n {
            a0[(r, c)] = offdiag * lcg(&mut state);
            b0[(r, c)] = offdiag * lcg(&mut state);
        }
    }
    let q0 = random_orthogonal(n, seed);
    let z0 = random_orthogonal(n, seed + 7);
    let a = matmul_bt(&matmul(&q0, &a0), &z0);
    let b = matmul_bt(&matmul(&q0, &b0), &z0);
    (a, b, spectrum)
}

/// Upper Hessenberg `A` (dense, nonzero subdiagonal) and upper triangular
/// `B`: such a pencil reaches `DHGEQZ` unchanged (no permutation, trivial
/// reflectors, identity `DGGHRD` rotations), which lets the tests drive the
/// zero-diagonal-of-`T` branches directly.
fn hessenberg_triangular(n: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    let mut state = seed ^ 0xabcdef;
    let mut a = Mat::<f64>::zeros(n, n);
    let mut b = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            if i <= j + 1 {
                a[(i, j)] = lcg(&mut state);
            }
            if i < j {
                b[(i, j)] = lcg(&mut state);
            }
        }
        // Keep subdiagonal entries and the diagonal of B away from zero.
        if j + 1 < n {
            a[(j + 1, j)] = 0.5 + 0.5 * lcg(&mut state).abs();
        }
        b[(j, j)] = 1.0 + 0.5 * lcg(&mut state).abs();
    }
    (a, b)
}

#[test]
fn test_issue_6_diagonal() {
    let a = Mat::diag(&[1.0f64, 2.0, 3.0]);
    let b = Mat::<f64>::eye(3);
    let qz = qz_of("diagonal", &a, &b);
    assert_dgges_contract("diagonal", a.as_ref(), b.as_ref(), &qz);
    let mut eigs = qz.eigenvalues_real();
    eigs.sort_by(f64::total_cmp);
    assert_eq!(eigs, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_issue_6_upper_tri_b_2x2() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let b = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
    let qz = qz_of("upper_tri_b", &a, &b);
    assert_dgges_contract("upper_tri_b", a.as_ref(), b.as_ref(), &qz);
    let s7 = 7.0f64.sqrt();
    let exact = [((2.0 + s7) / 3.0, 0.0, 1.0), ((2.0 - s7) / 3.0, 0.0, 1.0)];
    assert_eigs_close("upper_tri_b", &eig_triples(&qz), &exact, 1e-14);
}

#[test]
fn test_issue_6_complex_pair_b_diag_1_4() {
    // det(A − λ B) = 4λ² − 5λ + 7  ⇒  λ = (5 ± i√87)/8.
    let a = Mat::from_rows(&[&[1.0f64, -2.0], &[3.0, 1.0]]);
    let b = Mat::diag(&[1.0f64, 4.0]);
    let qz = qz_of("complex_pair", &a, &b);
    assert_dgges_contract("complex_pair", a.as_ref(), b.as_ref(), &qz);
    let e = qz.eigenvalues();
    assert!(e[0].alpha_imag > 0.0 && e[1].alpha_imag < 0.0);
    let im = 87.0f64.sqrt() / 8.0;
    let exact = [(0.625, im, 1.0), (0.625, -im, 1.0)];
    assert_eigs_close("complex_pair", &eig_triples(&qz), &exact, 1e-14);
}

#[test]
fn test_issue_6_reporter_3x3_b_identity() {
    // The reporter's minimal repro: ‖B − Q T Zᵀ‖ was 1.2 and T was not triangular.
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);
    let b = Mat::<f64>::eye(3);
    let qz = qz_of("reporter_3x3", &a, &b);
    let (ea, eb) = qz.verify(a.as_ref(), b.as_ref());
    assert!(ea < 1e-13 && eb < 1e-13, "verify() = ({ea:e}, {eb:e})");
    let t = qz.t();
    assert!(t[(1, 0)] == 0.0 && t[(2, 0)] == 0.0 && t[(2, 1)] == 0.0);
    assert_dgges_contract("reporter_3x3", a.as_ref(), b.as_ref(), &qz);
    assert_eigs_close(
        "reporter_3x3",
        &eig_triples(&qz),
        &standard_oracle(&a),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_random_n3() {
    check_random(3, 0.0);
}

#[test]
fn test_issue_6_random_n4() {
    check_random(4, 0.0);
}

#[test]
fn test_issue_6_random_n5() {
    check_random(5, 0.0);
}

#[test]
fn test_issue_6_random_n8() {
    check_random(8, 0.0);
}

#[test]
fn test_issue_6_random_n12() {
    check_random(12, 0.0);
}

#[test]
fn test_issue_6_random_n30() {
    // Diagonal shift keeps B well conditioned for the B⁻¹A oracle.
    check_random(30, 3.0);
}

#[test]
fn test_issue_6_known_spectrum_n30() {
    let (a, b, spectrum) = known_spectrum_pencil(30, 31, 0.3);
    let qz = qz_of("known_spectrum_n30", &a, &b);
    assert_dgges_contract("known_spectrum_n30", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 0);
    let complex = qz
        .eigenvalues()
        .iter()
        .filter(|e| e.alpha_imag != 0.0)
        .count();
    assert_eq!(complex, 20, "expected 10 complex conjugate pairs");
    assert_eigs_close(
        "known_spectrum_n30",
        &eig_triples(&qz),
        &spectrum,
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_known_spectrum_n9_f32() {
    let (a, b, spectrum) = known_spectrum_pencil(9, 9, 0.3);
    let (a32, b32) = (mat_from_f64::<f32>(&a), mat_from_f64::<f32>(&b));
    let qz = Qz::compute(a32.as_ref(), b32.as_ref())
        .unwrap_or_else(|e| panic!("known_spectrum_n9_f32: {e:?}"));
    assert_dgges_contract("known_spectrum_n9_f32", a32.as_ref(), b32.as_ref(), &qz);
    assert_eigs_close("known_spectrum_n9_f32", &eig_triples(&qz), &spectrum, 1e-4);
}

#[test]
fn test_issue_6_f32_random_n8() {
    let a = mat_from_f64::<f32>(&lcg_matrix(8, 108));
    let b = mat_from_f64::<f32>(&lcg_matrix(8, 208));
    let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap_or_else(|e| panic!("f32 n=8: {e:?}"));
    assert_dgges_contract("f32 n=8", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 0);
}

#[test]
fn test_issue_6_singular_b_diag() {
    // Replaces the tautological `test_qz_infinite_eigenvalue`.
    let a = Mat::<f64>::eye(2);
    let b = Mat::diag(&[1.0f64, 0.0]);
    let qz = qz_of("singular_b_diag", &a, &b);
    assert_dgges_contract("singular_b_diag", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 1);
    let finite: Vec<f64> = qz
        .eigenvalues()
        .iter()
        .filter(|e| e.is_finite())
        .filter_map(|e| e.value())
        .collect();
    assert_eq!(finite, vec![1.0]);
    assert_eq!(qz.eigenvalues_real(), vec![1.0]);
}

#[test]
fn test_issue_6_singular_b_two_zero_cols() {
    let n = 6;
    let a = lcg_matrix(n, 61);
    let mut b = lcg_matrix(n, 62);
    for i in 0..n {
        b[(i, 1)] = 0.0;
        b[(i, 4)] = 0.0;
    }
    let qz = qz_of("two_zero_cols", &a, &b);
    assert_dgges_contract("two_zero_cols", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 2);
    assert_eigs_close(
        "two_zero_cols",
        &eig_triples(&qz),
        &mobius_oracle(&a, &b),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_b_zero() {
    let n = 4;
    let a = lcg_matrix(n, 41);
    let b = Mat::<f64>::zeros(n, n);
    let qz = qz_of("b_zero", &a, &b);
    assert_dgges_contract("b_zero", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), n);
    assert!(qz.eigenvalues().iter().all(|e| e.beta == 0.0));
}

#[test]
fn test_issue_6_singular_pencil() {
    // α = β = 0 (singular pencil) counts as "not finite".
    let a = Mat::diag(&[1.0f64, 0.0, 2.0]);
    let b = Mat::diag(&[1.0f64, 0.0, 1.0]);
    let qz = qz_of("singular_pencil", &a, &b);
    assert_dgges_contract("singular_pencil", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 1);
    let mut eigs = qz.eigenvalues_real();
    eigs.sort_by(f64::total_cmp);
    assert_eq!(eigs, vec![1.0, 2.0]);
}

#[test]
fn test_issue_6_scaled_1e_minus_20() {
    let n = 5;
    let (a, b) = (lcg_matrix(n, 51), lcg_matrix(n, 52));
    let (a_s, b_s) = (scaled(&a, 1e-20), scaled(&b, 1e-20));
    let qz = qz_of("scaled_1e-20", &a_s, &b_s);
    assert_dgges_contract("scaled_1e-20", a_s.as_ref(), b_s.as_ref(), &qz);
    assert_eigs_close(
        "scaled_1e-20",
        &eig_triples(&qz),
        &generalized_oracle(&a, &b),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_scaled_1e300() {
    let n = 5;
    let (a, b) = (lcg_matrix(n, 53), lcg_matrix(n, 54));
    let reference = generalized_oracle(&a, &b);

    // A only: the eigenvalues scale by 1e300 (DGGES scales A down to BIGNUM).
    let a_s = scaled(&a, 1e300);
    let qz = qz_of("scaled_1e300 (A)", &a_s, &b);
    assert_dgges_contract("scaled_1e300 (A)", a_s.as_ref(), b.as_ref(), &qz);
    let back: Vec<Triple> = eig_triples(&qz)
        .into_iter()
        .map(|(ar, ai, be)| (ar * 1e-300, ai * 1e-300, be))
        .collect();
    assert_eigs_close("scaled_1e300 (A)", &back, &reference, CHORDAL_TOL);

    // A and B: the eigenvalues are unchanged.
    let b_s = scaled(&b, 1e300);
    let qz = qz_of("scaled_1e300 (A, B)", &a_s, &b_s);
    assert_dgges_contract("scaled_1e300 (A, B)", a_s.as_ref(), b_s.as_ref(), &qz);
    assert_eigs_close(
        "scaled_1e300 (A, B)",
        &eig_triples(&qz),
        &reference,
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_a_times_1e_minus_12() {
    let n = 5;
    let (a, b) = (lcg_matrix(n, 55), lcg_matrix(n, 56));
    let a_s = scaled(&a, 1e-12);
    let qz = qz_of("a_times_1e-12", &a_s, &b);
    assert_dgges_contract("a_times_1e-12", a_s.as_ref(), b.as_ref(), &qz);
    let back: Vec<Triple> = eig_triples(&qz)
        .into_iter()
        .map(|(ar, ai, be)| (ar * 1e12, ai * 1e12, be))
        .collect();
    assert_eigs_close(
        "a_times_1e-12",
        &back,
        &generalized_oracle(&a, &b),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_unscale_guard_purely_imaginary() {
    // ‖A‖ < SMLNUM triggers DGGES scaling; the eigenvalues ±1e-200·i have
    // ALPHAR exactly 0, where the reference pre-unscale test divides by zero.
    let a = Mat::from_rows(&[&[0.0f64, 1e-200], &[-1e-200, 0.0]]);
    let b = Mat::<f64>::eye(2);
    let qz = qz_of("unscale_guard", &a, &b);
    assert_dgges_contract("unscale_guard", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 0);
    let mut parts: Vec<(f64, f64)> = qz
        .eigenvalues()
        .iter()
        .map(|e| {
            e.value_complex()
                .unwrap_or_else(|| panic!("unscale_guard: infinite eigenvalue {e:?}"))
        })
        .collect();
    parts.sort_by(|x, y| x.1.total_cmp(&y.1));
    for (k, &(re, im)) in parts.iter().enumerate() {
        let want = if k == 0 { -1e-200 } else { 1e-200 };
        assert!(
            re.is_finite() && im.is_finite(),
            "non-finite eigenvalue {parts:?}"
        );
        assert!(re.abs() <= 1e-212, "real part {re:e} should vanish");
        assert!(
            (im - want).abs() <= 1e-12 * 1e-200,
            "imag part {im:e} vs {want:e}"
        );
    }
}

#[test]
fn test_issue_6_n1_negative_b() {
    let a = Mat::from_rows(&[&[3.0f64]]);
    let b = Mat::from_rows(&[&[-2.0f64]]);
    let qz = qz_of("n1_negative_b", &a, &b);
    assert_dgges_contract("n1_negative_b", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.s()[(0, 0)], -3.0);
    assert_eq!(qz.t()[(0, 0)], 2.0);
    assert_eq!(qz.q()[(0, 0)], 1.0);
    assert_eq!(qz.z()[(0, 0)], -1.0);
    assert_eq!(qz.eigenvalues()[0].value(), Some(-1.5));
}

#[test]
fn test_issue_6_isolated_by_permutation() {
    // Block upper triangular pencil (1×1 | dense 4×4 | 1×1) hidden by row and
    // column permutations; DGGBAL('P') must isolate both 1×1 blocks and
    // DGGBAK must undo the permutations in Q and Z.
    let n = 6;
    let dense_a = lcg_matrix(4, 71);
    let dense_b = lcg_matrix(4, 72);
    let mut a0 = Mat::<f64>::zeros(n, n);
    let mut b0 = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        a0[(0, j)] = 1.0 + 0.1 * j as f64;
        b0[(0, j)] = 0.5 - 0.1 * j as f64;
    }
    a0[(0, 0)] = 2.0;
    b0[(0, 0)] = 3.0;
    for i in 1..5 {
        for j in 1..n {
            if j < 5 {
                a0[(i, j)] = dense_a[(i - 1, j - 1)];
                b0[(i, j)] = dense_b[(i - 1, j - 1)];
            } else {
                a0[(i, j)] = 0.25 * i as f64;
                b0[(i, j)] = -0.25 * i as f64;
            }
        }
    }
    a0[(5, 5)] = 7.0;
    b0[(5, 5)] = -4.0;
    let rows = [3usize, 5, 0, 4, 1, 2];
    let cols = [4usize, 0, 2, 5, 3, 1];
    let mut a = Mat::<f64>::zeros(n, n);
    let mut b = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = a0[(rows[i], cols[j])];
            b[(i, j)] = b0[(rows[i], cols[j])];
        }
    }
    let qz = qz_of("isolated", &a, &b);
    assert_dgges_contract("isolated", a.as_ref(), b.as_ref(), &qz);
    let ratios: Vec<f64> = qz.eigenvalues_real();
    assert!(
        ratios.contains(&(2.0 / 3.0)),
        "missing isolated 2/3: {ratios:?}"
    );
    assert!(ratios.contains(&-1.75), "missing isolated -7/4: {ratios:?}");
    assert_eigs_close(
        "isolated",
        &eig_triples(&qz),
        &mobius_oracle(&a, &b),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_zero_t_leading_ilazro() {
    // T(0,0) = 0 at the top of the active block: ILAZRO with j = ILO, the
    // zero is chased off the top (DHGEQZ label 40).
    let (a, mut b) = hessenberg_triangular(5, 1);
    b[(0, 0)] = 0.0;
    let qz = qz_of("zero_t_leading", &a, &b);
    assert_dgges_contract("zero_t_leading", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 1);
    assert_eigs_close(
        "zero_t_leading",
        &eig_triples(&qz),
        &mobius_oracle(&a, &b),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_zero_t_interior_chase() {
    // Interior T(2,2) = 0 with a non-negligible H(2,1): the zero is chased
    // down to T(ILAST,ILAST) (DHGEQZ label 50) and deflated (label 70).
    let (a, mut b) = hessenberg_triangular(5, 2);
    b[(2, 2)] = 0.0;
    let qz = qz_of("zero_t_interior", &a, &b);
    assert_dgges_contract("zero_t_interior", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 1);
    assert_eigs_close(
        "zero_t_interior",
        &eig_triples(&qz),
        &mobius_oracle(&a, &b),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_zero_t_ilazr2() {
    // T(2,2) = 0 next to two consecutive small subdiagonals of H: the ILAZR2
    // test splits off the top part (DHGEQZ label 40 with ILAZR2).
    let (mut a, mut b) = hessenberg_triangular(5, 3);
    a[(2, 1)] = 1e-15;
    a[(3, 2)] = 1e-3;
    b[(2, 2)] = 0.0;
    let qz = qz_of("zero_t_ilazr2", &a, &b);
    assert_dgges_contract("zero_t_ilazr2", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 1);
}

#[test]
fn test_issue_6_zero_t_two_zeros() {
    let (a, mut b) = hessenberg_triangular(7, 4);
    b[(1, 1)] = 0.0;
    b[(4, 4)] = 0.0;
    let qz = qz_of("zero_t_two", &a, &b);
    assert_dgges_contract("zero_t_two", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 2);
    assert_eigs_close(
        "zero_t_two",
        &eig_triples(&qz),
        &mobius_oracle(&a, &b),
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_dims_errors() {
    let empty = Mat::<f64>::zeros(0, 0);
    assert_eq!(
        Qz::compute(empty.as_ref(), empty.as_ref()).err(),
        Some(QzError::EmptyMatrix)
    );
    let rect = Mat::<f64>::zeros(2, 3);
    let sq2 = Mat::<f64>::eye(2);
    let sq3 = Mat::<f64>::eye(3);
    assert_eq!(
        Qz::compute(rect.as_ref(), sq2.as_ref()).err(),
        Some(QzError::NotSquare { nrows: 2, ncols: 3 })
    );
    assert_eq!(
        Qz::compute(sq2.as_ref(), sq3.as_ref()).err(),
        Some(QzError::DimensionMismatch {
            nrows_a: 2,
            ncols_a: 2,
            nrows_b: 3,
            ncols_b: 3
        })
    );
    assert_eq!(
        Qz::compute(sq2.as_ref(), rect.as_ref()).err(),
        Some(QzError::DimensionMismatch {
            nrows_a: 2,
            ncols_a: 2,
            nrows_b: 2,
            ncols_b: 3
        })
    );
}

#[test]
fn test_issue_6_nan_input() {
    let mut a = lcg_matrix(3, 81);
    let b = lcg_matrix(3, 82);
    a[(1, 2)] = f64::NAN;
    assert_eq!(
        Qz::compute(a.as_ref(), b.as_ref()).err(),
        Some(QzError::NotConverged { iterations: 0 })
    );
    let a = lcg_matrix(3, 81);
    let mut b = lcg_matrix(3, 82);
    b[(0, 0)] = f64::INFINITY;
    assert_eq!(
        Qz::compute(a.as_ref(), b.as_ref()).err(),
        Some(QzError::NotConverged { iterations: 0 })
    );
}

#[test]
fn test_issue_6_random_sweep() {
    // Many small random pencils (n = 1..=10): the reporter saw
    // NotConverged and wrong factors for every non-trivial size.
    for seed in 0..240u64 {
        let n = 1 + (seed % 10) as usize;
        let label = format!("sweep seed={seed} n={n}");
        let a = lcg_matrix(n, 1000 + seed);
        let b = lcg_matrix(n, 5000 + seed);
        let qz = qz_of(&label, &a, &b);
        assert_dgges_contract(&label, a.as_ref(), b.as_ref(), &qz);
        let rb = residual_ratio(b.as_ref(), qz.q(), qz.t(), qz.z());
        assert!(rb.is_finite(), "{label}: rB = {rb}");
        assert!(qz.eigenvalues().iter().all(|e| to_f64(e.beta) >= 0.0));
    }
}

#[test]
fn test_issue_6_zero_t_2x2_top() {
    // 2×2 with T(0,0) = 0: ILAZRO at j = ILO, one chase step absorbs the
    // zero and the bottom 1×1 block deflates directly (label 40 → 80).
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
    let b = Mat::from_rows(&[&[0.0f64, 1.0], &[0.0, 2.0]]);
    let qz = qz_of("zero_t_2x2_top", &a, &b);
    assert_dgges_contract("zero_t_2x2_top", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 1);
    // det(A − λB) = (4 − 2λ) − 3(2 − λ) = λ − 2  ⇒  λ = 2 (plus λ = ∞).
    assert_eigs_close(
        "zero_t_2x2_top",
        &eig_triples(&qz),
        &[(2.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
        1e-14,
    );
}

#[test]
fn test_issue_6_zero_t_double_infinite_2x2() {
    // det(A − λB) ≡ 1: both eigenvalues are infinite. The chase rotation
    // (c = s) also annihilates T(1,1), so loop 40 ends at label 70.
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[1.0, 3.0]]);
    let b = Mat::from_rows(&[&[0.0f64, 1.0], &[0.0, 1.0]]);
    let qz = qz_of("double_infinite", &a, &b);
    assert_dgges_contract("double_infinite", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 2);
    assert!(qz.eigenvalues().iter().all(|e| e.beta == 0.0));
}

#[test]
fn test_issue_6_defective_2x2_reiterates() {
    // B⁻¹A is a rotated Jordan block (double eigenvalue 1): the first DLAG2
    // sees a complex pair, the DLASV2 standardization moves the shift onto
    // the real line, and DHGEQZ continues with real single shifts.
    let a = Mat::from_rows(&[
        &[1.2130944067898453f64, 0.17183370027139766],
        &[-0.010643286799973969, 1.166177625597872],
    ]);
    let b = Mat::from_rows(&[
        &[1.3357374096797803f64, -1.0462136296871376],
        &[0.0, 1.0604722305634264],
    ]);
    let qz = qz_of("defective_2x2", &a, &b);
    assert_dgges_contract("defective_2x2", a.as_ref(), b.as_ref(), &qz);
    assert_eq!(qz.num_infinite(), 0);
    // A defective eigenvalue is only determined to O(sqrt(eps)).
    assert_eigs_close(
        "defective_2x2",
        &eig_triples(&qz),
        &[(1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
        1e-6,
    );
}

#[test]
fn test_issue_6_unscale_guard_tiny_real_part() {
    // ‖A‖ < SMLNUM (DGGES scales A up); the pair 1e-310 ± 1e-290·i has a
    // real part whose unscaled ALPHAR underflows below SAFMIN, the case the
    // DGGES pre-unscale guard (loop 20) is written for. The real part is only
    // determined to O(eps·‖A‖) by any backward-stable method.
    let a = Mat::from_rows(&[&[1e-310f64, 1e-290], &[-1e-290, 1e-310]]);
    let b = Mat::<f64>::eye(2);
    let qz = qz_of("unscale_tiny_real", &a, &b);
    assert_dgges_contract("unscale_tiny_real", a.as_ref(), b.as_ref(), &qz);
    let mut parts: Vec<(f64, f64)> = qz
        .eigenvalues()
        .iter()
        .map(|e| {
            e.value_complex()
                .unwrap_or_else(|| panic!("unscale_tiny_real: infinite eigenvalue {e:?}"))
        })
        .collect();
    parts.sort_by(|x, y| x.1.total_cmp(&y.1));
    for (k, &(re, im)) in parts.iter().enumerate() {
        let want_im = if k == 0 { -1e-290 } else { 1e-290 };
        assert!((im - want_im).abs() <= 1e-12 * 1e-290, "imag part {im:e}");
        assert!(re.is_finite(), "real part {re:e}");
        assert!(
            (re - 1e-310).abs() <= 10.0 * f64::EPSILON * 1e-290,
            "real part {re:e} vs 1e-310"
        );
    }
}
