//! Regression tests for GitHub issue #6, sorted part: LAPACK `DGGES` with
//! `SORT = 'S'` ([`Qz::compute_sorted`]), `DTGSEN`-style reordering of an
//! existing factorization ([`Qz::reorder`]) and the LAPACK 3.12 kernels
//! behind them: `DLAGV2`, `DGETC2`/`DGESC2`, `DTGSY2`, `DTGEX2`, `DTGEXC`,
//! `DTGSEN` (`IJOB = 0`) and the `DGGES` LASTSL/LST2SL final check.
//!
//! Every sorted factorization is checked with the `ddrges` threshold
//! (`THRESH = 30`) on both backward errors and both orthogonality ratios,
//! for the exact generalized real Schur structure, for the selected
//! eigenvalues occupying exactly the leading `sdim` positions, and for the
//! spectrum agreeing with the unsorted run under greedy chordal matching.

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{Mat, MatRef};

use super::lagv2::dlagv2;
use super::pipeline::run;
use super::reorder::{check_ordering, tgexc, tgsen0};
use super::swap::{SchurPencil, SwapRejection, tgex2};
use super::sylvester::{Mat8, Small, SylvesterPencils, gesc2, getc2, tgsy2};
use super::test_util::{
    THRESH, Triple, assert_dgges_contract, eig_triples, lcg, lcg_matrix, mat_from_f64, matmul,
    matmul_bt, max_matched_chordal, orthogonality_ratio, random_orthogonal, residual_ratio, scaled,
};
use super::util::Consts;
use super::{GeneralizedEigenvalue, Qz, QzError, QzOrdering, QzReorderError};

/// Chordal tolerance between a sorted and the unsorted `f64` spectrum.
const CHORDAL_TOL: f64 = 1e-8;

/// Chordal tolerance for the eigenvalues of individual diagonal blocks,
/// compared in closed form (well-separated test spectra).
const BLOCK_TOL: f64 = 1e-11;

/// Chordal tolerance between a sorted and the unsorted `f32` spectrum
/// (both computed in `f32`; `1e4 · f32::EPSILON`).
const F32_TOL: f64 = 1e4 * f32::EPSILON as f64;

/// Selection predicate as taken by the helpers.
type Pred<'a, T> = &'a dyn Fn(&GeneralizedEigenvalue<T>) -> bool;

// ---------------------------------------------------------------------------
// Selection helpers
// ---------------------------------------------------------------------------

/// Pair-aware selection flags (the `DTGSEN`/`DGGES` rule): both members of a
/// complex pair are selected when either member satisfies `pred`.
fn selection_flags<T: Field + Real>(
    eigs: &[GeneralizedEigenvalue<T>],
    pred: Pred<'_, T>,
) -> Vec<bool> {
    let n = eigs.len();
    let mut flags = vec![false; n];
    let mut j = 0;
    while j < n {
        if eigs[j].alpha_imag != T::zero() && j + 1 < n {
            let (first, second) = (pred(&eigs[j]), pred(&eigs[j + 1]));
            flags[j] = first || second;
            flags[j + 1] = first || second;
            j += 2;
        } else {
            flags[j] = pred(&eigs[j]);
            j += 1;
        }
    }
    flags
}

/// True when the `true` flags form a prefix.
fn is_prefix(flags: &[bool]) -> bool {
    let k = flags.iter().take_while(|&&f| f).count();
    flags[k..].iter().all(|&f| !f)
}

/// Number of `true` flags.
fn count(flags: &[bool]) -> usize {
    flags.iter().filter(|&&f| f).count()
}

/// Asserts the `DGGES` `SORT = 'S'` contract of a sorted factorization:
/// Schur-form contract (`THRESH = 30`, exact structure), selected
/// eigenvalues exactly in the leading `sdim` positions, `sdim` equal to the
/// pair-aware count, no selection change (`INFO ≠ N+2`) and the spectrum of
/// the unsorted run (`reference`) within `tol` (greedy chordal matching).
fn assert_sorted<T: Field + Real + bytemuck::Zeroable>(
    label: &str,
    (a, b): (MatRef<'_, T>, MatRef<'_, T>),
    qz: &Qz<T>,
    ordering: QzOrdering,
    pred: Pred<'_, T>,
    reference: &[Triple],
    tol: f64,
) {
    assert_dgges_contract(label, a, b, qz);
    let flags = selection_flags(qz.eigenvalues(), pred);
    assert_eq!(
        ordering.sdim,
        count(&flags),
        "{label}: sdim {} vs pair-aware count of {flags:?}",
        ordering.sdim
    );
    assert!(
        is_prefix(&flags),
        "{label}: selected eigenvalues are not leading: {flags:?}"
    );
    assert!(
        !ordering.selection_changed,
        "{label}: selection changed (DGGES INFO = N+2)"
    );
    let d = max_matched_chordal(&eig_triples(qz), reference);
    assert!(
        d <= tol,
        "{label}: sorted spectrum differs from the unsorted one by {d:.3e} > {tol:.1e}"
    );
}

/// `Re λ < c` (finite eigenvalues only).
fn re_below(c: f64) -> impl Fn(&GeneralizedEigenvalue<f64>) -> bool {
    move |e| e.value_complex().is_some_and(|(re, _)| re < c)
}

/// `|λ| < r` (finite eigenvalues only).
fn abs_below(r: f64) -> impl Fn(&GeneralizedEigenvalue<f64>) -> bool {
    move |e| e.value_complex().is_some_and(|(re, im)| re.hypot(im) < r)
}

/// Finite in the relative sense, `β > 1e-10 · (|Re α| + |Im α|)`.
///
/// After a swap an infinite eigenvalue carries a rounding-level `β`, so
/// `β == 0` is not a usable test on reordered factorizations.
fn numerically_finite(e: &GeneralizedEigenvalue<f64>) -> bool {
    e.beta > 1e-10 * (e.alpha_real.abs() + e.alpha_imag.abs())
}

/// One of four predicate families for the random sweeps, parameterized by
/// `seed`: `Re λ < c`, `|λ| < r`, `Im λ > 0` (selects a pair through one
/// member only) and a value-hash selection.
fn sweep_predicate(seed: u64) -> Box<dyn Fn(&GeneralizedEigenvalue<f64>) -> bool> {
    let mut state = seed.wrapping_mul(0x9e37_79b9) ^ 0x0abc_def1;
    let c = lcg(&mut state);
    match seed % 4 {
        0 => Box::new(re_below(0.8 * c)),
        1 => Box::new(abs_below(0.6 + 0.8 * c.abs())),
        2 => Box::new(|e: &GeneralizedEigenvalue<f64>| e.alpha_imag > 0.0),
        _ => Box::new(|e: &GeneralizedEigenvalue<f64>| {
            e.value_complex()
                .is_some_and(|(re, im)| (re.abs() * 997.0 + im.abs() * 113.0).fract() < 0.5)
        }),
    }
}

/// Radius between the two magnitudes closest to the median of `|λ|`
/// (geometric mean across the first clear gap, formed without over- or
/// underflow for 1e±300-scaled spectra), for half/half selections.
fn split_radius(eigs: &[GeneralizedEigenvalue<f64>]) -> f64 {
    let mut mags: Vec<f64> = eigs
        .iter()
        .filter_map(|e| e.value_complex())
        .map(|(re, im)| re.hypot(im))
        .collect();
    mags.sort_by(f64::total_cmp);
    let mid = mags.len() / 2;
    for offset in 0..mags.len() {
        for i in [mid + offset, mid.wrapping_sub(offset)] {
            if i >= 1 && i < mags.len() && mags[i - 1] < mags[i] * (1.0 - 1e-6) {
                return mags[i - 1].sqrt() * mags[i].sqrt();
            }
        }
    }
    panic!("no gap between eigenvalue magnitudes: {mags:?}");
}

// ---------------------------------------------------------------------------
// Test pencils
// ---------------------------------------------------------------------------

/// One diagonal block of a test Schur form: the real eigenvalue `x / d`
/// (`Real(x, d)`) or the complex pair `(x ± i·y) / d` (`Pair(x, y, d)`,
/// `S` block `[[x, y], [−y, x]]`, `T` block `d·I`).
#[derive(Clone, Copy)]
enum Block {
    Real(f64, f64),
    Pair(f64, f64, f64),
}

/// Upper quasi-triangular `S0` and upper triangular `T0` with the given
/// diagonal blocks and pseudo-random strictly upper parts scaled by
/// `offdiag`.
fn schur_from_blocks(blocks: &[Block], seed: u64, offdiag: f64) -> (Mat<f64>, Mat<f64>) {
    let n: usize = blocks
        .iter()
        .map(|b| match b {
            Block::Real(..) => 1,
            Block::Pair(..) => 2,
        })
        .sum();
    let mut s = Mat::<f64>::zeros(n, n);
    let mut t = Mat::<f64>::zeros(n, n);
    let mut end = vec![0usize; n];
    let mut i = 0;
    for blk in blocks {
        match *blk {
            Block::Real(x, d) => {
                s[(i, i)] = x;
                t[(i, i)] = d;
                end[i] = i + 1;
                i += 1;
            }
            Block::Pair(x, y, d) => {
                s[(i, i)] = x;
                s[(i, i + 1)] = y;
                s[(i + 1, i)] = -y;
                s[(i + 1, i + 1)] = x;
                t[(i, i)] = d;
                t[(i + 1, i + 1)] = d;
                end[i] = i + 2;
                end[i + 1] = i + 2;
                i += 2;
            }
        }
    }
    let mut state = seed ^ 0x5eed_0123;
    for r in 0..n {
        for c in end[r]..n {
            s[(r, c)] = offdiag * lcg(&mut state);
            t[(r, c)] = offdiag * lcg(&mut state);
        }
    }
    (s, t)
}

/// `(Q0·S·Z0ᵀ, Q0·T·Z0ᵀ)`.
fn hide_with(s: &Mat<f64>, t: &Mat<f64>, q0: &Mat<f64>, z0: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    (matmul_bt(&matmul(q0, s), z0), matmul_bt(&matmul(q0, t), z0))
}

/// The block pencil hidden by random orthogonal factors.
fn block_pencil(blocks: &[Block], seed: u64, offdiag: f64) -> (Mat<f64>, Mat<f64>) {
    let (s, t) = schur_from_blocks(blocks, seed, offdiag);
    let n = s.nrows();
    hide_with(
        &s,
        &t,
        &random_orthogonal(n, seed),
        &random_orthogonal(n, seed + 11),
    )
}

/// Pencil with the real spectrum `alpha[i] / beta[i]`.
fn real_spectrum_pencil(alpha: &[f64], beta: &[f64], seed: u64) -> (Mat<f64>, Mat<f64>) {
    let blocks: Vec<Block> = alpha
        .iter()
        .zip(beta)
        .map(|(&x, &d)| Block::Real(x, d))
        .collect();
    block_pencil(&blocks, seed, 0.5)
}

/// Block upper triangular pencil (`1×1 | dense 5×5 | 1×1`) hidden by row and
/// column permutations, so `DGGBAL('P')` isolates eigenvalues and `DGGBAK`
/// permutes `Q` and `Z`.
fn permuted_pencil(seed: u64) -> (Mat<f64>, Mat<f64>) {
    let n = 7;
    let mut state = seed ^ 0x9e37;
    let mut a0 = Mat::<f64>::zeros(n, n);
    let mut b0 = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        a0[(0, j)] = lcg(&mut state);
        b0[(0, j)] = lcg(&mut state);
    }
    a0[(0, 0)] = 1.75;
    b0[(0, 0)] = 0.5;
    for i in 1..n - 1 {
        for j in 1..n {
            a0[(i, j)] = lcg(&mut state);
            b0[(i, j)] = lcg(&mut state);
        }
    }
    a0[(n - 1, n - 1)] = -2.5;
    b0[(n - 1, n - 1)] = 1.25;
    let mut a = Mat::<f64>::zeros(n, n);
    let mut b = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let (r, c) = ((3 * i + 1) % n, (5 * j + 2) % n);
            a[(i, j)] = a0[(r, c)];
            b[(i, j)] = b0[(r, c)];
        }
    }
    (a, b)
}

/// `blockdiag(R, R)` with `R = [[0, 1], [−1, 0]]` and `B = I`: two 2×2
/// blocks with identical eigenvalues `±i`, so swapping them makes the
/// generalized Sylvester operator of `DTGEX2` exactly singular.
fn identical_pairs_pencil() -> (Mat<f64>, Mat<f64>) {
    let mut a = Mat::<f64>::zeros(4, 4);
    for blk in 0..2 {
        let i = 2 * blk;
        a[(i, i + 1)] = 1.0;
        a[(i + 1, i)] = -1.0;
    }
    (a, Mat::<f64>::eye(4))
}

// ---------------------------------------------------------------------------
// Closed-form block spectra and structure checks for the kernels
// ---------------------------------------------------------------------------

/// `(α, β)` with `β ≥ 0`.
fn homogeneous(re: f64, im: f64, beta: f64) -> Triple {
    if beta < 0.0 {
        (-re, -im, -beta)
    } else {
        (re, im, beta)
    }
}

/// Generalized eigenvalues of the 2×2 pencil `(s, t)` (`t[1][0]` ignored)
/// from `det(s − λ·t) = 0`, as homogeneous triples.
fn block2_eigs(s: [[f64; 2]; 2], t: [[f64; 2]; 2]) -> [Triple; 2] {
    let a2 = t[0][0] * t[1][1];
    let a1 = -(s[0][0] * t[1][1] + s[1][1] * t[0][0] - s[1][0] * t[0][1]);
    let a0 = s[0][0] * s[1][1] - s[0][1] * s[1][0];
    let disc = a1 * a1 - 4.0 * a2 * a0;
    if disc < 0.0 {
        let im = (-disc).sqrt();
        [
            homogeneous(-a1, im, 2.0 * a2),
            homogeneous(-a1, -im, 2.0 * a2),
        ]
    } else {
        // q = −(a1 + sign(a1)·√disc)/2: λ1 = q / a2, λ2 = a0 / q.
        let q = -0.5 * (a1 + a1.signum() * disc.sqrt());
        [homogeneous(q, 0.0, a2), homogeneous(a0, 0.0, q)]
    }
}

/// The diagonal blocks of a Schur pencil (`S(j+1, j) ≠ 0` starts a 2×2
/// block) with their eigenvalues in closed form.
fn schur_blocks(s: &Mat<f64>, t: &Mat<f64>) -> Vec<Vec<Triple>> {
    let n = s.nrows();
    let mut out = Vec::new();
    let mut j = 0;
    while j < n {
        if j + 1 < n && s[(j + 1, j)] != 0.0 {
            let sb = [
                [s[(j, j)], s[(j, j + 1)]],
                [s[(j + 1, j)], s[(j + 1, j + 1)]],
            ];
            let tb = [
                [t[(j, j)], t[(j, j + 1)]],
                [t[(j + 1, j)], t[(j + 1, j + 1)]],
            ];
            out.push(block2_eigs(sb, tb).to_vec());
            j += 2;
        } else {
            out.push(vec![homogeneous(s[(j, j)], 0.0, t[(j, j)])]);
            j += 1;
        }
    }
    out
}

/// Asserts that the blocks of `(s, t)` carry the expected block spectra, in
/// order.
fn assert_block_order(label: &str, s: &Mat<f64>, t: &Mat<f64>, expected: &[&Vec<Triple>]) {
    let got = schur_blocks(s, t);
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: block count {got:?} vs {expected:?}"
    );
    for (k, (g, e)) in got.iter().zip(expected).enumerate() {
        let d = max_matched_chordal(g, e);
        assert!(
            d <= BLOCK_TOL,
            "{label}: block {k} eigenvalues {g:?} vs {e:?} (chordal {d:.3e})"
        );
    }
}

/// Structure violations of a generalized real Schur pencil produced by the
/// kernels (signs of the diagonal of `T` are not normalized yet): exact
/// zeros below the diagonal of `T` and below the subdiagonal of `S`, no two
/// consecutive nonzero subdiagonals, a diagonal `T` part in 2×2 blocks.
fn kernel_structure(s: &Mat<f64>, t: &Mat<f64>) -> Vec<String> {
    let n = s.nrows();
    let mut bad = Vec::new();
    for j in 0..n {
        for i in (j + 1)..n {
            if t[(i, j)] != 0.0 {
                bad.push(format!("T({i},{j}) = {}", t[(i, j)]));
            }
            if i > j + 1 && s[(i, j)] != 0.0 {
                bad.push(format!("S({i},{j}) = {}", s[(i, j)]));
            }
        }
    }
    for i in 1..n.saturating_sub(1) {
        if s[(i, i - 1)] != 0.0 && s[(i + 1, i)] != 0.0 {
            bad.push(format!("consecutive subdiagonals at {i}"));
        }
    }
    for j in 0..n.saturating_sub(1) {
        if s[(j + 1, j)] != 0.0 && t[(j, j + 1)] != 0.0 {
            bad.push(format!("2x2 block at {j}: T({j},{}) != 0", j + 1));
        }
    }
    bad
}

/// Asserts `A = Q·S·Zᵀ`, `B = Q·T·Zᵀ` (ratios ≤ `THRESH`), orthogonal `Q`,
/// `Z` and the kernel structure of `(S, T)`.
fn assert_kernel_contract(
    label: &str,
    (a, b): (&Mat<f64>, &Mat<f64>),
    (s, t, q, z): (&Mat<f64>, &Mat<f64>, &Mat<f64>, &Mat<f64>),
) {
    let ra = residual_ratio(a.as_ref(), q.as_ref(), s.as_ref(), z.as_ref());
    let rb = residual_ratio(b.as_ref(), q.as_ref(), t.as_ref(), z.as_ref());
    let oq = orthogonality_ratio(q.as_ref());
    let oz = orthogonality_ratio(z.as_ref());
    assert!(
        ra <= THRESH && rb <= THRESH && oq <= THRESH && oz <= THRESH,
        "{label}: rA={ra:.3e} rB={rb:.3e} Q={oq:.3e} Z={oz:.3e}"
    );
    let bad = kernel_structure(s, t);
    assert!(bad.is_empty(), "{label}: structure {bad:?}");
}

/// A Schur pencil `(S0, T0)` with blocks `[1×1, n1, n2, 1×1]` (so the swap
/// at `j1 = 1` has rows above and columns to the right), random orthogonal
/// `Q0`, `Z0` and the hidden pencil `A = Q0·S0·Z0ᵀ`, `B = Q0·T0·Z0ᵀ`.
struct KernelPencil {
    s: Mat<f64>,
    t: Mat<f64>,
    q: Mat<f64>,
    z: Mat<f64>,
    a: Mat<f64>,
    b: Mat<f64>,
}

fn kernel_pencil(n1: usize, n2: usize, seed: u64) -> KernelPencil {
    let mut state = seed ^ 0x00c0_ffee;
    let mut block = |size: usize, idx: usize| {
        let r = lcg(&mut state);
        if size == 1 {
            Block::Real(-1.5 + 1.1 * idx as f64 + 0.2 * r, 0.8 + 0.1 * idx as f64)
        } else {
            Block::Pair(
                0.4 - 0.7 * idx as f64 + 0.2 * r,
                0.9 + 0.2 * idx as f64,
                0.6 + 0.15 * idx as f64,
            )
        }
    };
    let blocks = [
        Block::Real(2.5, 1.3),
        block(n1, 1),
        block(n2, 2),
        Block::Real(-0.9, 0.7),
    ];
    let (s, t) = schur_from_blocks(&blocks, seed, 0.6);
    let n = s.nrows();
    let (q, z) = (random_orthogonal(n, seed), random_orthogonal(n, seed + 11));
    let (a, b) = hide_with(&s, &t, &q, &z);
    KernelPencil { s, t, q, z, a, b }
}

/// Swaps the blocks of sizes `n1` (at `j1 = 1`) and `n2` with `DTGEX2` and
/// checks the backward errors, the structure, and that the two blocks
/// exchanged their eigenvalues while the others kept theirs.
fn check_tgex2(n1: usize, n2: usize, seed: u64) {
    let label = format!("dtgex2 n1={n1} n2={n2} seed={seed}");
    let k = Consts::<f64>::new();
    let p = kernel_pencil(n1, n2, seed);
    let before = schur_blocks(&p.s, &p.t);
    let (mut s, mut t, mut q, mut z) = (p.s.clone(), p.t.clone(), p.q.clone(), p.z.clone());
    let res = tgex2(
        &mut SchurPencil {
            s: &mut s,
            t: &mut t,
            q: &mut q,
            z: &mut z,
        },
        1,
        n1,
        n2,
        &k,
    );
    assert_eq!(res, Ok(()), "{label}: swap rejected");
    assert_kernel_contract(&label, (&p.a, &p.b), (&s, &t, &q, &z));
    assert_block_order(
        &label,
        &s,
        &t,
        &[&before[0], &before[2], &before[1], &before[3]],
    );
    // The boundary between the exchanged blocks is an exact zero.
    assert_eq!(s[(1 + n2, n2)], 0.0, "{label}: S21 block not zeroed");
}

// ---------------------------------------------------------------------------
// DGETC2 / DGESC2
// ---------------------------------------------------------------------------

/// Pseudo-random `n × n` matrix (entries in `[-1, 1)`) in a `Mat8`.
fn mat8(n: usize, seed: u64) -> Mat8<f64> {
    let mut state = seed ^ 0x00dd_ba11;
    let mut a = [[0.0f64; 8]; 8];
    for row in a.iter_mut().take(n) {
        for v in row.iter_mut().take(n) {
            *v = lcg(&mut state);
        }
    }
    a
}

#[test]
fn test_issue_6_reorder_dgetc2_dgesc2_solve() {
    let k = Consts::<f64>::new();
    for n in 1..=8usize {
        for seed in 0..6u64 {
            let a0 = mat8(n, 31 * n as u64 + seed);
            let mut state = seed + 17;
            let mut x = [0.0f64; 8];
            for v in x.iter_mut().take(n) {
                *v = lcg(&mut state);
            }
            let b0 = x;
            let mut lu = a0;
            let piv = getc2(n, &mut lu, &k);
            assert_eq!(piv.info, 0, "n={n} seed={seed}: perturbed pivot");
            let scale = gesc2(n, &lu, &mut x, &piv, &k);
            assert_eq!(scale, 1.0, "n={n} seed={seed}");
            let anorm = (0..n)
                .map(|i| (0..n).map(|j| a0[i][j].abs()).sum::<f64>())
                .fold(0.0, f64::max);
            let xnorm = x[..n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for i in 0..n {
                let ax: f64 = (0..n).map(|j| a0[i][j] * x[j]).sum();
                let r = (ax - scale * b0[i]).abs();
                assert!(
                    r <= 100.0 * n as f64 * f64::EPSILON * anorm * xnorm,
                    "n={n} seed={seed}: residual {r:e} in row {i}"
                );
            }
        }
    }
}

#[test]
fn test_issue_6_reorder_dgetc2_pivot_order_lapack_3_12() {
    // LAPACK 3.12.0 scans the rows in the outer loop with `.GE.`, so among
    // equal maxima the last one in row-major order wins: (1, 0). (Later
    // LAPACK versions scan columns first and would pick (0, 1).)
    let k = Consts::<f64>::new();
    let mut a = [[0.0f64; 8]; 8];
    a[0][1] = 2.0;
    a[1][0] = 2.0;
    let piv = getc2(2, &mut a, &k);
    assert_eq!((piv.ipiv[0], piv.jpiv[0]), (1, 0));
    assert_eq!(piv.info, 0);
    assert_eq!((a[0][0], a[1][0], a[1][1]), (2.0, 0.0, 2.0));
}

#[test]
fn test_issue_6_reorder_dgetc2_singular_and_scaling() {
    let k = Consts::<f64>::new();
    // Rank one: the second pivot is exactly zero and is raised to
    // SMIN = max(eps · 4, SMLNUM) with INFO = 2.
    let mut a = [[0.0f64; 8]; 8];
    a[0][0] = 1.0;
    a[0][1] = 2.0;
    a[1][0] = 2.0;
    a[1][1] = 4.0;
    let piv = getc2(2, &mut a, &k);
    assert_eq!(piv.info, 2);
    assert_eq!(a[1][1], 4.0 * f64::EPSILON);
    // N = 1 below SMLNUM: perturbed to SMLNUM; DGESC2 then scales the
    // right-hand side to avoid overflow (SCALE < 1).
    let smlnum = k.safmin / k.eps;
    let mut a = [[0.0f64; 8]; 8];
    a[0][0] = 1e-300;
    let piv = getc2(1, &mut a, &k);
    assert_eq!(piv.info, 1);
    assert_eq!(a[0][0], smlnum);
    let mut rhs = [0.0f64; 8];
    rhs[0] = 1e10;
    let scale = gesc2(1, &a, &mut rhs, &piv, &k);
    assert!(scale > 0.0 && scale < 1.0, "scale {scale}");
    let r = (smlnum * rhs[0] - scale * 1e10).abs();
    assert!(r <= 4.0 * f64::EPSILON * scale * 1e10, "residual {r:e}");
}

// ---------------------------------------------------------------------------
// DTGSY2
// ---------------------------------------------------------------------------

/// Quasi-triangular `A` (blocks of `sizes`, eigenvalues near `shift`) and
/// upper triangular `D` (positive diagonal, nonzero in-block upper entry),
/// with random strictly upper parts; returns `(A, D, order)`.
fn sylvester_pair(sizes: &[usize], shift: f64, seed: u64) -> (Small<f64>, Small<f64>, usize) {
    let mut a = [[0.0f64; 4]; 4];
    let mut d = [[0.0f64; 4]; 4];
    let mut state = seed ^ 0x0051_17e5;
    let mut end = [0usize; 4];
    let mut i = 0;
    for (idx, &size) in sizes.iter().enumerate() {
        let x = shift + 0.35 * idx as f64;
        if size == 1 {
            a[i][i] = x;
            d[i][i] = 1.0 + 0.2 * idx as f64;
            end[i] = i + 1;
            i += 1;
        } else {
            a[i][i] = x;
            a[i][i + 1] = 0.9;
            a[i + 1][i] = -0.7;
            a[i + 1][i + 1] = x + 0.1;
            d[i][i] = 1.1 + 0.1 * idx as f64;
            d[i][i + 1] = 0.3 * lcg(&mut state);
            d[i + 1][i + 1] = 0.9;
            end[i] = i + 2;
            end[i + 1] = i + 2;
            i += 2;
        }
    }
    for r in 0..i {
        for c in end[r]..i {
            a[r][c] = lcg(&mut state);
            d[r][c] = lcg(&mut state);
        }
    }
    (a, d, i)
}

/// Solves `A·R − L·B = scale·C`, `D·R − L·E = scale·F` with `DTGSY2` and
/// checks both residuals.
fn check_tgsy2(sizes_a: &[usize], sizes_b: &[usize], seed: u64) {
    let label = format!("dtgsy2 {sizes_a:?} x {sizes_b:?}");
    let k = Consts::<f64>::new();
    let (a, d, m) = sylvester_pair(sizes_a, 2.0, seed);
    let (b, e, n) = sylvester_pair(sizes_b, -2.0, seed + 1);
    let mut state = seed ^ 0x00c0_ffee;
    let mut c0 = [[0.0f64; 4]; 4];
    let mut f0 = [[0.0f64; 4]; 4];
    for i in 0..m {
        for j in 0..n {
            c0[i][j] = lcg(&mut state);
            f0[i][j] = lcg(&mut state);
        }
    }
    let (mut r, mut l) = (c0, f0);
    let sol = tgsy2(&SylvesterPencils { m, n, a, b, d, e }, &mut r, &mut l, &k);
    assert_eq!(sol.info, 0, "{label}: common eigenvalues reported");
    assert_eq!(sol.scale, 1.0, "{label}");
    let norm = |x: &Small<f64>, rows: usize, cols: usize| {
        (0..rows)
            .map(|i| (0..cols).map(|j| x[i][j].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    };
    let size = norm(&a, m, m) * norm(&r, m, n)
        + norm(&l, m, n) * norm(&b, n, n)
        + norm(&d, m, m) * norm(&r, m, n)
        + norm(&l, m, n) * norm(&e, n, n)
        + norm(&c0, m, n)
        + norm(&f0, m, n);
    for i in 0..m {
        for j in 0..n {
            let ar: f64 = (0..m).map(|p| a[i][p] * r[p][j]).sum();
            let lb: f64 = (0..n).map(|p| l[i][p] * b[p][j]).sum();
            let dr: f64 = (0..m).map(|p| d[i][p] * r[p][j]).sum();
            let le: f64 = (0..n).map(|p| l[i][p] * e[p][j]).sum();
            let r1 = (ar - lb - sol.scale * c0[i][j]).abs();
            let r2 = (dr - le - sol.scale * f0[i][j]).abs();
            assert!(
                r1.max(r2) <= 64.0 * f64::EPSILON * size,
                "{label}: residual ({r1:e}, {r2:e}) at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_issue_6_reorder_dtgsy2_residual() {
    let cases: [(&[usize], &[usize]); 9] = [
        (&[1], &[1]),
        (&[1], &[2]),
        (&[2], &[1]),
        (&[2], &[2]),
        (&[1, 1], &[1, 1]),
        (&[1, 2], &[2, 1]),
        (&[2, 2], &[1, 2, 1]),
        (&[2, 1, 1], &[1, 2]),
        (&[1, 1, 1, 1], &[2, 2]),
    ];
    for (seed, (sa, sb)) in cases.iter().enumerate() {
        check_tgsy2(sa, sb, 400 + seed as u64);
    }
}

#[test]
fn test_issue_6_reorder_dtgsy2_common_eigenvalues() {
    // (A, D) and (B, E) share the pair ±i: the Kronecker operator is
    // singular, DGETC2 perturbs a pivot and DTGSY2 reports INFO > 0.
    let k = Consts::<f64>::new();
    let mut a = [[0.0f64; 4]; 4];
    a[0][1] = 1.0;
    a[1][0] = -1.0;
    let mut d = [[0.0f64; 4]; 4];
    d[0][0] = 1.0;
    d[1][1] = 1.0;
    let mut c = [[1.0f64; 4]; 4];
    let mut f = [[1.0f64; 4]; 4];
    let sol = tgsy2(
        &SylvesterPencils {
            m: 2,
            n: 2,
            a,
            b: a,
            d,
            e: d,
        },
        &mut c,
        &mut f,
        &k,
    );
    assert!(sol.info > 0, "singular Sylvester operator not detected");
}

// ---------------------------------------------------------------------------
// DLAGV2
// ---------------------------------------------------------------------------

/// `[csl snl; −snl csl] · m · [csr −snr; snr csr]`.
fn rotate2(m: [[f64; 2]; 2], (csl, snl, csr, snr): (f64, f64, f64, f64)) -> [[f64; 2]; 2] {
    let l = [[csl, snl], [-snl, csl]];
    let r = [[csr, -snr], [snr, csr]];
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
    out
}

/// Checks the `DLAGV2` reconstruction `(A', B') = Qᵀ (A, B) Z` and the
/// rotation normalization; returns the output pencil.
fn check_dlagv2(label: &str, a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> super::lagv2::Lagv2<f64> {
    let k = Consts::<f64>::new();
    let g = dlagv2(a, b, &k);
    let rot = (g.csl, g.snl, g.csr, g.snr);
    assert!((g.csl * g.csl + g.snl * g.snl - 1.0).abs() <= 4.0 * f64::EPSILON);
    assert!((g.csr * g.csr + g.snr * g.snr - 1.0).abs() <= 4.0 * f64::EPSILON);
    for (name, input, output) in [("A", a, g.a), ("B", b, g.b)] {
        let want = rotate2(input, rot);
        let scale = input
            .iter()
            .flatten()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        for i in 0..2 {
            for j in 0..2 {
                // Entries DLAGV2 sets to exact zeros are compared against
                // rounding-level values of the rotated input.
                assert!(
                    (want[i][j] - output[i][j]).abs() <= 16.0 * f64::EPSILON * scale,
                    "{label}: {name}({i},{j}) = {} vs rotated {}",
                    output[i][j],
                    want[i][j]
                );
            }
        }
    }
    assert_eq!(g.b[1][0], 0.0, "{label}: B(2,1) not zeroed");
    g
}

#[test]
fn test_issue_6_reorder_dlagv2_real_pair() {
    // det(A − λB) = 6λ² − 8λ − 2 ⇒ λ = (2 ± √7)/3: upper triangular output.
    let g = check_dlagv2(
        "real pair",
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 1.0], [0.0, 3.0]],
    );
    assert_eq!(g.a[1][0], 0.0);
    assert_eq!(g.alphai, [0.0, 0.0]);
    assert_eq!(g.alphar, [g.a[0][0], g.a[1][1]]);
    assert_eq!(g.beta, [g.b[0][0], g.b[1][1]]);
    let s7 = 7.0f64.sqrt();
    let got = [(g.alphar[0], 0.0, g.beta[0]), (g.alphar[1], 0.0, g.beta[1])];
    let want = [((2.0 + s7) / 3.0, 0.0, 1.0), ((2.0 - s7) / 3.0, 0.0, 1.0)];
    assert!(max_matched_chordal(&got, &want) <= 1e-15);
}

#[test]
fn test_issue_6_reorder_dlagv2_complex_pair() {
    // det(A − λB) = 4λ² − 5λ + 7 ⇒ λ = (5 ± i√87)/8, B already diagonal.
    let g = check_dlagv2(
        "complex pair",
        [[1.0, -2.0], [3.0, 1.0]],
        [[1.0, 0.0], [0.0, 4.0]],
    );
    assert_eq!((g.b[0][1], g.b[1][0]), (0.0, 0.0));
    assert_eq!(g.beta, [1.0, 1.0]);
    assert!(g.alphai[0] > 0.0 && g.alphai[1] == -g.alphai[0]);
    assert_eq!(g.alphar[0], g.alphar[1]);
    assert!((g.alphar[0] - 0.625).abs() <= 1e-15);
    assert!((g.alphai[0] - 87.0f64.sqrt() / 8.0).abs() <= 1e-15);
    // Non-diagonal B: DLASV2 diagonalizes it; the block spectrum is kept.
    let (a, b) = ([[1.0, -2.0], [3.0, 1.0]], [[1.0, 0.5], [0.0, 4.0]]);
    let g = check_dlagv2("complex pair, upper B", a, b);
    assert_eq!((g.b[0][1], g.b[1][0]), (0.0, 0.0));
    let got = [
        (g.alphar[0], g.alphai[0], g.beta[0]),
        (g.alphar[1], g.alphai[1], g.beta[1]),
    ];
    assert!(max_matched_chordal(&got, &block2_eigs(a, b)) <= 1e-14);
    assert!(max_matched_chordal(&block2_eigs(g.a, g.b), &block2_eigs(a, b)) <= 1e-14);
}

#[test]
fn test_issue_6_reorder_dlagv2_degenerate_branches() {
    // |A(2,1)| ≤ ulp after scaling: identity rotations, A(2,1) := 0.
    let g = check_dlagv2(
        "A21 negligible",
        [[1.0, 2.0], [1e-20, 3.0]],
        [[1.0, 1.0], [0.0, 2.0]],
    );
    assert_eq!((g.csl, g.snl, g.csr, g.snr), (1.0, 0.0, 1.0, 0.0));
    assert_eq!(g.a[1][0], 0.0);
    // |B(1,1)| ≤ ulp: left rotation zeroes A(2,1); infinite eigenvalue first.
    let g = check_dlagv2(
        "B11 negligible",
        [[1.0, 2.0], [3.0, 4.0]],
        [[1e-20, 1.0], [0.0, 2.0]],
    );
    assert_eq!((g.csr, g.snr), (1.0, 0.0));
    assert_eq!((g.a[1][0], g.b[0][0]), (0.0, 0.0));
    assert_eq!(g.beta[0], 0.0);
    // |B(2,2)| ≤ ulp: right rotation zeroes A(2,1); infinite eigenvalue last.
    let g = check_dlagv2(
        "B22 negligible",
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 1.0], [0.0, 1e-20]],
    );
    assert_eq!((g.csl, g.snl), (1.0, 0.0));
    assert_eq!((g.a[1][0], g.b[1][1]), (0.0, 0.0));
    assert_eq!(g.beta[1], 0.0);
}

#[test]
fn test_issue_6_reorder_dlagv2_negative_b_reference_behaviour() {
    // Reference DLAGV2 does not enforce its documented `b11 ≥ b22 > 0` in
    // the complex branch (DLASV2 returns signed singular values): with
    // B = diag(−2, 1) the rotations are the identity and B stays negative.
    // DTGSEN's normalization (see `reorder`) makes such T blocks positive.
    let g = check_dlagv2(
        "negative B",
        [[1.0, 2.0], [3.0, 1.0]],
        [[-2.0, 0.0], [0.0, 1.0]],
    );
    assert!(g.alphai[0] > 0.0, "expected a complex pair");
    assert!(g.b[0][0] < 0.0, "B(1,1) = {}", g.b[0][0]);
    assert_eq!((g.b[0][1], g.b[1][0]), (0.0, 0.0));
}

// ---------------------------------------------------------------------------
// DTGEX2 / DTGEXC / DTGSEN kernels
// ---------------------------------------------------------------------------

#[test]
fn test_issue_6_reorder_dtgex2_1x1_1x1() {
    for seed in 0..4 {
        check_tgex2(1, 1, 500 + seed);
    }
}

#[test]
fn test_issue_6_reorder_dtgex2_1x1_2x2() {
    for seed in 0..4 {
        check_tgex2(1, 2, 510 + seed);
    }
}

#[test]
fn test_issue_6_reorder_dtgex2_2x2_1x1() {
    for seed in 0..4 {
        check_tgex2(2, 1, 520 + seed);
    }
}

#[test]
fn test_issue_6_reorder_dtgex2_2x2_2x2() {
    for seed in 0..4 {
        check_tgex2(2, 2, 530 + seed);
    }
}

#[test]
fn test_issue_6_reorder_dtgex2_rejects_identical_blocks() {
    let k = Consts::<f64>::new();
    let (a, b) = identical_pairs_pencil();
    let (mut s, mut t) = (a.clone(), b.clone());
    let (mut q, mut z) = (Mat::<f64>::eye(4), Mat::<f64>::eye(4));
    let res = tgex2(
        &mut SchurPencil {
            s: &mut s,
            t: &mut t,
            q: &mut q,
            z: &mut z,
        },
        0,
        2,
        2,
        &k,
    );
    assert_eq!(res, Err(SwapRejection::Sylvester));
    // A rejected swap leaves the pencil untouched.
    for j in 0..4 {
        for i in 0..4 {
            assert_eq!(s[(i, j)].to_bits(), a[(i, j)].to_bits());
            assert_eq!(t[(i, j)].to_bits(), b[(i, j)].to_bits());
            let id = if i == j { 1.0f64 } else { 0.0 };
            assert_eq!(q[(i, j)].to_bits(), id.to_bits());
            assert_eq!(z[(i, j)].to_bits(), id.to_bits());
        }
    }
}

#[test]
fn test_issue_6_reorder_dtgexc_moves_blocks() {
    let k = Consts::<f64>::new();
    let blocks = [
        Block::Real(1.5, 1.0),
        Block::Pair(-0.3, 1.2, 0.9),
        Block::Real(-2.0, 1.1),
        Block::Pair(0.8, 0.5, 1.3),
        Block::Real(0.25, 0.7),
    ];
    let (s0, t0) = schur_from_blocks(&blocks, 41, 0.5);
    let n = s0.nrows();
    let (q0, z0) = (random_orthogonal(n, 41), random_orthogonal(n, 52));
    let (a, b) = hide_with(&s0, &t0, &q0, &z0);
    let orig = schur_blocks(&s0, &t0);
    let [r0, p1, r2, p3, r4] = [&orig[0], &orig[1], &orig[2], &orig[3], &orig[4]];
    let (mut s, mut t, mut q, mut z) = (s0, t0, q0, z0);
    let mut step = |label: &str, ifst: usize, ilst: usize, want: usize, order: &[&Vec<Triple>]| {
        let pos = tgexc(
            &mut SchurPencil {
                s: &mut s,
                t: &mut t,
                q: &mut q,
                z: &mut z,
            },
            ifst,
            ilst,
            &k,
        );
        assert_eq!(pos, Ok(want), "{label}: final position");
        assert_kernel_contract(label, (&a, &b), (&s, &t, &q, &z));
        assert_block_order(label, &s, &t, order);
    };
    // IFST on the second row of a pair: moved as a whole to the top.
    step("pair up", 5, 0, 0, &[p3, r0, p1, r2, r4]);
    // Pair down onto a trailing 1×1: ILST is adjusted to n − 2.
    step("pair down", 0, 6, 5, &[r0, p1, r2, r4, p3]);
    // 1×1 up over a 1×1, a pair and a 1×1.
    step("1x1 up", 4, 0, 0, &[r4, r0, p1, r2, p3]);
    // 1×1 down over a pair and a 1×1.
    step("1x1 down", 1, 4, 4, &[r4, p1, r2, r0, p3]);
}

/// Moves the block at `ifst` to `ilst` with `DTGEXC` in a Schur pencil where
/// the `Block::Pair`s starting at the rows `fake` are turned into 2×2
/// blocks with *real* eigenvalues (`[[x, y], [y, x]]`, `T = d·I`,
/// `λ = (x ± y)/d`). `DTGEXC` treats them as 2×2 blocks, and the `DLAGV2`
/// re-standardization after their first swap splits them into two 1×1
/// blocks, which drives the split branches of `DTGEXC` (`NBF = 3`, "2-by-2
/// block did split").
fn check_split_move(
    label: &str,
    blocks: &[Block],
    fake: &[usize],
    (ifst, ilst, want): (usize, usize, usize),
    seed: u64,
) {
    let k = Consts::<f64>::new();
    let (mut s0, t0) = schur_from_blocks(blocks, seed, 0.5);
    for &i in fake {
        s0[(i + 1, i)] = -s0[(i + 1, i)];
    }
    let n = s0.nrows();
    let (q0, z0) = (random_orthogonal(n, seed), random_orthogonal(n, seed + 11));
    let (a, b) = hide_with(&s0, &t0, &q0, &z0);
    let before = schur_blocks(&s0, &t0);
    let moved = before
        .iter()
        .scan(0usize, |row, blk| {
            let start = *row;
            *row += blk.len();
            Some((start, blk))
        })
        .find(|&(start, _)| start == ifst)
        .map(|(_, blk)| blk.clone())
        .unwrap_or_else(|| panic!("{label}: no block starts at {ifst}"));
    assert!(
        moved.len() == 2 && moved.iter().all(|e| e.1 == 0.0),
        "{label}: the moving block must be a 2x2 with real eigenvalues"
    );
    let (mut s, mut t, mut q, mut z) = (s0, t0, q0, z0);
    let pos = tgexc(
        &mut SchurPencil {
            s: &mut s,
            t: &mut t,
            q: &mut q,
            z: &mut z,
        },
        ifst,
        ilst,
        &k,
    );
    assert_eq!(pos, Ok(want), "{label}: final position");
    assert_kernel_contract(label, (&a, &b), (&s, &t, &q, &z));
    let after = schur_blocks(&s, &t);
    assert_eq!(
        after.len(),
        before.len() + fake.len(),
        "{label}: every real-eigenvalue 2x2 block must have split: {after:?}"
    );
    let flat = |blks: &[Vec<Triple>]| blks.iter().flatten().copied().collect::<Vec<Triple>>();
    let d = max_matched_chordal(&flat(&after), &flat(&before));
    assert!(d <= BLOCK_TOL, "{label}: spectrum changed by {d:.3e}");
    // The moved eigenvalues occupy rows [pos, pos+1] as two 1×1 blocks.
    assert_eq!(
        s[(want + 1, want)],
        0.0,
        "{label}: moved block did not split"
    );
    let landed = [
        homogeneous(s[(want, want)], 0.0, t[(want, want)]),
        homogeneous(s[(want + 1, want + 1)], 0.0, t[(want + 1, want + 1)]),
    ];
    let d = max_matched_chordal(&landed, &moved);
    assert!(
        d <= BLOCK_TOL,
        "{label}: rows {want}..={} hold {landed:?}, expected {moved:?}",
        want + 1
    );
}

#[test]
fn test_issue_6_reorder_dtgexc_split_blocks() {
    // Down: the moving real 2×2 splits on its first swap (NBF := 3), then
    // passes a genuine pair ("did not split"), a second real 2×2 that splits
    // when swapped ("did split") and a trailing 1×1 (two 1×1 swaps); ILST is
    // adjusted from n − 1 to n − 2.
    check_split_move(
        "split down",
        &[
            Block::Pair(1.0, 0.5, 1.0),
            Block::Real(-1.0, 1.0),
            Block::Pair(0.3, 1.1, 0.9),
            Block::Pair(-2.0, 0.7, 1.2),
            Block::Real(2.5, 1.0),
        ],
        &[0, 5],
        (0, 7, 6),
        71,
    );
    // Up: the moving real 2×2 splits on its first swap with a 1×1, then
    // passes a genuine pair ("did not split"), a real 2×2 that splits
    // ("did split") and the leading 1×1 (two 1×1 swaps).
    check_split_move(
        "split up",
        &[
            Block::Real(0.75, 1.0),
            Block::Pair(-1.5, 0.6, 1.1),
            Block::Pair(0.2, 0.9, 1.3),
            Block::Real(-3.0, 1.0),
            Block::Pair(2.0, 0.8, 0.9),
        ],
        &[1, 6],
        (6, 0, 0),
        73,
    );
}

#[test]
fn test_issue_6_reorder_dtgsen_normalizes_2x2_t_signs() {
    // The reference DTGEX2 path (QR/RQ triangularization + DLAGV2) can leave
    // negative diagonal entries in T. DTGSEN normalizes 1×1 blocks; this
    // implementation also normalizes the diagonal T part of 2×2 blocks (so
    // that T has a non-negative diagonal as documented for DGGES).
    let k = Consts::<f64>::new();
    let mut negative_pairs = 0usize;
    for seed in 0..36u64 {
        let (n1, n2) = [(1usize, 2usize), (2, 1), (2, 2)][(seed % 3) as usize];
        let label = format!("normalize n1={n1} n2={n2} seed={seed}");
        let p = kernel_pencil(n1, n2, 600 + seed);
        let n = p.s.nrows();
        let (mut s, mut t, mut q, mut z) = (p.s.clone(), p.t.clone(), p.q.clone(), p.z.clone());
        let mut pencil = SchurPencil {
            s: &mut s,
            t: &mut t,
            q: &mut q,
            z: &mut z,
        };
        assert_eq!(tgex2(&mut pencil, 1, n1, n2, &k), Ok(()), "{label}");
        for j in 0..n - 1 {
            if pencil.s[(j + 1, j)] != 0.0
                && (pencil.t[(j, j)] < 0.0 || pencil.t[(j + 1, j + 1)] < 0.0)
            {
                negative_pairs += 1;
            }
        }
        // An empty selection runs only the label-60 normalization.
        let (mut ar, mut ai, mut be) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let status = tgsen0(&vec![false; n], &mut pencil, &mut ar, &mut ai, &mut be, &k);
        assert_eq!(status, Ok(()), "{label}");
        assert_kernel_contract(&label, (&p.a, &p.b), (&s, &t, &q, &z));
        let mut j = 0;
        while j < n {
            assert!(!t[(j, j)].is_sign_negative(), "{label}: T({j},{j}) < 0");
            if j + 1 < n && s[(j + 1, j)] != 0.0 {
                assert!(!t[(j + 1, j + 1)].is_sign_negative(), "{label}");
                assert!(ai[j] > 0.0 && ai[j + 1] == -ai[j], "{label}: pair signs");
                assert!(be[j] > 0.0 && be[j + 1] == be[j], "{label}: pair betas");
                j += 2;
            } else {
                assert_eq!(ar[j].to_bits(), s[(j, j)].to_bits(), "{label}");
                assert_eq!(be[j].to_bits(), t[(j, j)].to_bits(), "{label}");
                assert_eq!(ai[j], 0.0, "{label}");
                j += 1;
            }
        }
        // The triples describe the returned blocks.
        let triples: Vec<Triple> = (0..n).map(|i| (ar[i], ai[i], be[i])).collect();
        let blocks: Vec<Triple> = schur_blocks(&s, &t).into_iter().flatten().collect();
        assert!(
            max_matched_chordal(&triples, &blocks) <= BLOCK_TOL,
            "{label}: triples {triples:?} vs blocks {blocks:?}"
        );
    }
    assert!(
        negative_pairs > 0,
        "the DTGEX2 path never produced a negative 2x2 T diagonal; the normalization is untested"
    );
}

// ---------------------------------------------------------------------------
// DGGES final check (LASTSL / LST2SL)
// ---------------------------------------------------------------------------

/// Crafted eigenvalue: `sel` encodes the selection in the sign of
/// `alpha_real` (the predicate is `alpha_real > 0`), `im` is `alpha_imag`.
fn crafted(sel: bool, im: f64) -> GeneralizedEigenvalue<f64> {
    GeneralizedEigenvalue::complex(if sel { 1.0 } else { -1.0 }, im, 1.0)
}

#[test]
fn test_issue_6_reorder_check_ordering_crafted() {
    let r = |sel| crafted(sel, 0.0);
    let pair = |first, second| [crafted(first, 1.0), crafted(second, -1.0)];
    let cases: Vec<(Vec<GeneralizedEigenvalue<f64>>, usize, bool)> = vec![
        (vec![r(true), r(true), r(false)], 2, false),
        (vec![r(true), r(false), r(true)], 2, true),
        (vec![r(false), r(true)], 1, true),
        (vec![r(false), r(false)], 0, false),
        // A pair counts 2 when either member is selected.
        (
            [pair(true, false).to_vec(), vec![r(true)]].concat(),
            3,
            false,
        ),
        (
            [pair(false, true).to_vec(), vec![r(false)]].concat(),
            2,
            false,
        ),
        // A pair selected through its second member after an unselected
        // eigenvalue: LST2SL (status before the pair) triggers INFO = N+2.
        (
            [vec![r(false)], pair(false, true).to_vec()].concat(),
            2,
            true,
        ),
        ([pair(false, false), pair(true, false)].concat(), 2, true),
        (
            [pair(true, true).to_vec(), vec![r(false), r(false)]].concat(),
            2,
            false,
        ),
        (
            [vec![r(true)], pair(false, false).to_vec(), vec![r(true)]].concat(),
            2,
            true,
        ),
    ];
    for (k, (eigs, sdim, changed)) in cases.iter().enumerate() {
        let mut calls = 0usize;
        let mut pred = |e: &GeneralizedEigenvalue<f64>| {
            calls += 1;
            e.alpha_real > 0.0
        };
        let got = check_ordering(eigs, &mut pred);
        assert_eq!(
            got,
            QzOrdering {
                sdim: *sdim,
                selection_changed: *changed
            },
            "case {k}"
        );
        assert_eq!(calls, eigs.len(), "case {k}: one call per eigenvalue");
    }
}

// ---------------------------------------------------------------------------
// Qz::compute_sorted (DGGES, SORT = 'S')
// ---------------------------------------------------------------------------

/// Unsorted factorization and its eigenvalue triples.
fn unsorted<T: Field + Real + bytemuck::Zeroable>(
    label: &str,
    a: &Mat<T>,
    b: &Mat<T>,
) -> (Qz<T>, Vec<Triple>) {
    let qz = Qz::compute(a.as_ref(), b.as_ref())
        .unwrap_or_else(|e| panic!("{label}: unsorted Qz failed: {e:?}"));
    let triples = eig_triples(&qz);
    (qz, triples)
}

/// `compute_sorted`, panicking with context on failure.
fn sorted<T: Field + Real + bytemuck::Zeroable>(
    label: &str,
    a: &Mat<T>,
    b: &Mat<T>,
    pred: Pred<'_, T>,
) -> (Qz<T>, QzOrdering) {
    Qz::compute_sorted(a.as_ref(), b.as_ref(), pred)
        .unwrap_or_else(|e| panic!("{label}: compute_sorted failed: {e:?}"))
}

#[test]
fn test_issue_6_sorted_real_by_value() {
    let label = "real by value";
    let (a, b) = real_spectrum_pencil(
        &[3.0, -1.0, 2.0, -4.0, 0.5, -2.0],
        &[1.0, 0.5, 2.0, 1.0, 1.5, 0.8],
        5,
    );
    let pred = re_below(0.0);
    let (qz0, reference) = unsorted(label, &a, &b);
    assert!(
        !is_prefix(&selection_flags(qz0.eigenvalues(), &pred)),
        "{label}: precondition: the unsorted form must need reordering"
    );
    let (qz, ordering) = sorted(label, &a, &b, &pred);
    assert_eq!(ordering.sdim, 3);
    assert_sorted(
        label,
        (a.as_ref(), b.as_ref()),
        &qz,
        ordering,
        &pred,
        &reference,
        CHORDAL_TOL,
    );
    let mut lead: Vec<f64> = qz.eigenvalues()[..3]
        .iter()
        .filter_map(|e| e.value())
        .collect();
    lead.sort_by(f64::total_cmp);
    let want = [-4.0, -2.5, -2.0];
    for (g, w) in lead.iter().zip(want) {
        assert!((g - w).abs() <= 1e-12, "{label}: leading {lead:?}");
    }
}

#[test]
fn test_issue_6_sorted_complex_pair_one_member() {
    let label = "pair through one member";
    let (a, b) = block_pencil(
        &[
            Block::Real(1.0, 1.0),
            Block::Pair(-0.5, 2.0, 1.0),
            Block::Real(2.0, 1.0),
            Block::Pair(0.3, 0.4, 1.0),
            Block::Real(-3.0, 1.0),
        ],
        21,
        0.5,
    );
    // Only the member with Im λ = +2 satisfies the predicate.
    let pred = |e: &GeneralizedEigenvalue<f64>| e.value_complex().is_some_and(|(_, im)| im > 1.0);
    let (qz0, reference) = unsorted(label, &a, &b);
    assert!(
        !is_prefix(&selection_flags(qz0.eigenvalues(), &pred)),
        "{label}: precondition: the unsorted form must need reordering"
    );
    let (qz, ordering) = sorted(label, &a, &b, &pred);
    assert_eq!(ordering.sdim, 2, "{label}: the whole pair counts");
    assert_sorted(
        label,
        (a.as_ref(), b.as_ref()),
        &qz,
        ordering,
        &pred,
        &reference,
        CHORDAL_TOL,
    );
    let lead = &eig_triples(&qz)[..2];
    assert!(max_matched_chordal(lead, &[(-0.5, 2.0, 1.0), (-0.5, -2.0, 1.0)]) <= 1e-10);
    assert!(qz.s()[(1, 0)] != 0.0, "{label}: leading block is not 2x2");
}

/// All/none selections: `DTGSEN` takes its quick return, so `S`, `T`, `Q`,
/// `Z` are bitwise those of the unsorted run.
fn check_all_or_none(select_all: bool) {
    let label = if select_all {
        "all selected"
    } else {
        "none selected"
    };
    let (a, b) = (lcg_matrix(8, 801), lcg_matrix(8, 802));
    let pred = move |_: &GeneralizedEigenvalue<f64>| select_all;
    let (qz0, reference) = unsorted(label, &a, &b);
    let (qz, ordering) = sorted(label, &a, &b, &pred);
    assert_eq!(ordering.sdim, if select_all { 8 } else { 0 });
    assert_sorted(
        label,
        (a.as_ref(), b.as_ref()),
        &qz,
        ordering,
        &pred,
        &reference,
        1e-10,
    );
    for (x, y) in [
        (qz.s(), qz0.s()),
        (qz.t(), qz0.t()),
        (qz.q(), qz0.q()),
        (qz.z(), qz0.z()),
    ] {
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(x[(i, j)].to_bits(), y[(i, j)].to_bits(), "{label}");
            }
        }
    }
}

#[test]
fn test_issue_6_sorted_all_selected() {
    check_all_or_none(true);
}

#[test]
fn test_issue_6_sorted_none_selected() {
    check_all_or_none(false);
}

#[test]
fn test_issue_6_sorted_finite_before_infinite() {
    // B has zero columns 0 and 4 (two infinite eigenvalues). Column 0 of A
    // has a single nonzero, so DGGBAL('P') isolates the infinite eigenvalue
    // (A(0,0), 0) in the leading position of the unsorted Schur form; the
    // other one comes out of the QZ iteration.
    let label = "finite first";
    let n = 6;
    let mut a = lcg_matrix(n, 61);
    let mut b = lcg_matrix(n, 62);
    for i in 0..n {
        if i > 0 {
            a[(i, 0)] = 0.0;
        }
        b[(i, 0)] = 0.0;
        b[(i, 4)] = 0.0;
    }
    let pred = numerically_finite;
    let (qz0, reference) = unsorted(label, &a, &b);
    assert_eq!(qz0.num_infinite(), 2);
    assert!(
        !is_prefix(&selection_flags(qz0.eigenvalues(), &pred)),
        "{label}: precondition: an infinite eigenvalue must lead"
    );
    let (qz, ordering) = sorted(label, &a, &b, &pred);
    assert_eq!(ordering.sdim, n - 2);
    assert_sorted(
        label,
        (a.as_ref(), b.as_ref()),
        &qz,
        ordering,
        &pred,
        &reference,
        CHORDAL_TOL,
    );
    assert!(
        qz.eigenvalues()[n - 2..]
            .iter()
            .all(|e| !numerically_finite(e))
    );
}

#[test]
fn test_issue_6_sorted_random_sweep() {
    for seed in 0..200u64 {
        let n = 2 + (seed % 11) as usize;
        let label = format!("sorted sweep seed={seed} n={n}");
        let a = lcg_matrix(n, 7000 + seed);
        let b = lcg_matrix(n, 9000 + seed);
        let pred = sweep_predicate(seed);
        let (_, reference) = unsorted(&label, &a, &b);
        let (qz, ordering) = sorted(&label, &a, &b, &*pred);
        assert_sorted(
            &label,
            (a.as_ref(), b.as_ref()),
            &qz,
            ordering,
            &*pred,
            &reference,
            CHORDAL_TOL,
        );
    }
}

#[test]
fn test_issue_6_sorted_f32() {
    let pred = |e: &GeneralizedEigenvalue<f32>| e.value_complex().is_some_and(|(re, _)| re < 0.0);
    let upper = |e: &GeneralizedEigenvalue<f32>| e.alpha_imag > 0.0;
    let mut cases: Vec<(String, Mat<f32>, Mat<f32>)> = [(6usize, 61u64), (8, 81), (10, 101)]
        .iter()
        .map(|&(n, seed)| {
            (
                format!("f32 random n={n}"),
                mat_from_f64::<f32>(&lcg_matrix(n, seed)),
                mat_from_f64::<f32>(&lcg_matrix(n, seed + 1)),
            )
        })
        .collect();
    let (a, b) = block_pencil(
        &[
            Block::Real(1.0, 1.0),
            Block::Pair(-0.5, 2.0, 1.0),
            Block::Real(2.0, 1.0),
            Block::Pair(0.3, 0.4, 0.8),
            Block::Pair(1.5, 0.7, 1.2),
        ],
        23,
        0.4,
    );
    cases.push((
        "f32 pairs".into(),
        mat_from_f64::<f32>(&a),
        mat_from_f64::<f32>(&b),
    ));
    for (label, a, b) in &cases {
        for (name, p) in [("Re<0", &pred as Pred<'_, f32>), ("Im>0", &upper)] {
            let label = format!("{label} {name}");
            let (qz0, reference) = unsorted(&label, a, b);
            let (qz, ordering) = sorted(&label, a, b, p);
            assert_sorted(
                &label,
                (a.as_ref(), b.as_ref()),
                &qz,
                ordering,
                p,
                &reference,
                F32_TOL,
            );
            let mut re = qz0.clone();
            let ordering = re
                .reorder(p)
                .unwrap_or_else(|e| panic!("{label}: reorder failed: {e:?}"));
            assert_sorted(
                &format!("{label} (reorder)"),
                (a.as_ref(), b.as_ref()),
                &re,
                ordering,
                p,
                &reference,
                F32_TOL,
            );
        }
    }
}

#[test]
fn test_issue_6_sorted_scaled_1e_minus_20() {
    let label = "scaled 1e-20";
    let (a, b) = (lcg_matrix(7, 151), lcg_matrix(7, 152));
    let (a_s, b_s) = (scaled(&a, 1e-20), scaled(&b, 1e-20));
    let pred = re_below(0.0);
    let (_, reference) = unsorted(label, &a_s, &b_s);
    let (qz, ordering) = sorted(label, &a_s, &b_s, &pred);
    assert_sorted(
        label,
        (a_s.as_ref(), b_s.as_ref()),
        &qz,
        ordering,
        &pred,
        &reference,
        CHORDAL_TOL,
    );
    // The selection is scale invariant: same count as the unscaled pencil.
    let (_, unscaled) = sorted(label, &a, &b, &pred);
    assert_eq!(ordering.sdim, unscaled.sdim);
}

/// `A` and `B` scaled by `fa` and `fb` beyond `[SMLNUM, BIGNUM]`, so that
/// `DGGES` scales both internally (to the same bound): the unscaled
/// eigenvalues are `(fa/fb)·μ` while the internally scaled ones are `O(μ)`.
/// The selector must see the unscaled values; given the internal ones, the
/// half/half split by magnitude would select everything or nothing.
fn check_scaled_pencil(label: &str, (fa, fb): (f64, f64), seed: u64) {
    let (a, b) = (lcg_matrix(7, seed), lcg_matrix(7, seed + 1));
    let (a_s, b_s) = (scaled(&a, fa), scaled(&b, fb));
    let work = run(a_s.as_ref(), b_s.as_ref()).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    assert!(
        work.ascale.is_some() && work.bscale.is_some(),
        "{label}: DGGES scaling not triggered"
    );
    let (qz0, reference) = unsorted(label, &a_s, &b_s);
    let pred = abs_below(split_radius(qz0.eigenvalues()));
    let expected = count(&selection_flags(qz0.eigenvalues(), &pred));
    assert!(expected > 0 && expected < 7, "{label}: split {expected}");
    let (qz, ordering) = sorted(label, &a_s, &b_s, &pred);
    assert_eq!(
        ordering.sdim, expected,
        "{label}: selector saw scaled values"
    );
    assert_sorted(
        label,
        (a_s.as_ref(), b_s.as_ref()),
        &qz,
        ordering,
        &pred,
        &reference,
        CHORDAL_TOL,
    );
    let mut re = qz0.clone();
    let ordering = re
        .reorder(&pred)
        .unwrap_or_else(|e| panic!("{label}: reorder failed: {e:?}"));
    assert_sorted(
        &format!("{label} (reorder)"),
        (a_s.as_ref(), b_s.as_ref()),
        &re,
        ordering,
        &pred,
        &reference,
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_sorted_scaled_1e300() {
    check_scaled_pencil("scaled 1e300 (A), 1e290 (B)", (1e300, 1e290), 153);
    // A and B by the same factor: the eigenvalues are unchanged.
    let label = "scaled 1e300 (A, B)";
    let (a, b) = (lcg_matrix(7, 155), lcg_matrix(7, 156));
    let (a_s, b_s) = (scaled(&a, 1e300), scaled(&b, 1e300));
    let pred = re_below(0.0);
    let (_, reference) = unsorted(label, &a_s, &b_s);
    let (qz, ordering) = sorted(label, &a_s, &b_s, &pred);
    assert_sorted(
        label,
        (a_s.as_ref(), b_s.as_ref()),
        &qz,
        ordering,
        &pred,
        &reference,
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_sorted_scaled_1e_minus_200() {
    check_scaled_pencil("scaled 1e-200 (A), 1e-190 (B)", (1e-200, 1e-190), 157);
}

#[test]
fn test_issue_6_sorted_unbalanced_scaling_rejected() {
    // DGGES scales A and B independently. With ‖A‖ = 1e300·‖B‖ the scaled
    // pencil keeps ‖Ã‖/‖B‖ ≈ BIGNUM ≈ 1.5e138, and the Kronecker system of
    // DTGSY2 mixes S rows of size 1e138 with T rows of size 1: DGETC2's
    // absolute pivot threshold SMIN = eps·max|Z| flags the T equations, so
    // every swap involving a 2×2 block is rejected, exactly as in reference
    // LAPACK (DGGES INFO = N+3), even though the eigenvalues are well
    // separated.
    let (a, b) = (lcg_matrix(7, 153), lcg_matrix(7, 154));
    let a_s = scaled(&a, 1e300);
    let (qz0, _) = unsorted("unbalanced", &a_s, &b);
    let radius = split_radius(qz0.eigenvalues());
    let pred = abs_below(radius);
    let err = Qz::compute_sorted(a_s.as_ref(), b.as_ref(), &pred).err();
    assert!(
        matches!(err, Some(QzReorderError::SwapFailed { .. })),
        "unbalanced: {err:?}"
    );
    // reorder() on the unscaled factorization (‖S‖/‖T‖ ≈ 1e300) hits the
    // same limit and leaves the factorization untouched.
    let mut qz = qz0.clone();
    assert!(matches!(
        qz.reorder(&pred),
        Err(QzReorderError::SwapFailed { .. })
    ));
    assert_bitwise_equal("unbalanced reorder", &qz0, &qz);
    // Scaling only changes the eigenvalues by the factor: the same selection
    // on (A, B) in units of 1e300 is computed without trouble.
    let unit = abs_below(radius * 1e-300);
    let (_, reference) = unsorted("balanced", &a, &b);
    let (qz, ordering) = sorted("balanced", &a, &b, &unit);
    assert_eq!(
        ordering.sdim,
        count(&selection_flags(qz0.eigenvalues(), &pred))
    );
    assert_sorted(
        "balanced",
        (a.as_ref(), b.as_ref()),
        &qz,
        ordering,
        &unit,
        &reference,
        CHORDAL_TOL,
    );
}

#[test]
fn test_issue_6_sorted_selection_changed_end_to_end() {
    // A stateful selector: the DGGES final check sees a different selection
    // than DTGSEN did, which is exactly what INFO = N+2 reports.
    let label = "selection changed";
    let n = 5;
    let (a, b) = real_spectrum_pencil(&[-1.0, 2.0, -3.0, 4.0, 5.0], &[1.0; 5], 77);
    let mut calls = 0usize;
    let pred = |e: &GeneralizedEigenvalue<f64>| {
        calls += 1;
        let v = e.value().unwrap_or(f64::NAN);
        if calls <= n { v < 0.0 } else { v > 0.0 }
    };
    let (qz, ordering) = Qz::compute_sorted(a.as_ref(), b.as_ref(), pred)
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    assert_eq!(
        calls,
        2 * n,
        "{label}: one selection pass and one check pass"
    );
    assert_eq!(
        ordering,
        QzOrdering {
            sdim: 3,
            selection_changed: true
        }
    );
    assert_dgges_contract(label, a.as_ref(), b.as_ref(), &qz);
    assert!(
        qz.eigenvalues()[..2]
            .iter()
            .all(|e| e.value().is_some_and(|v| v < 0.0))
    );
}

#[test]
fn test_issue_6_sorted_swap_failed() {
    let (a, b) = identical_pairs_pencil();
    // Precondition: the unsorted factorization is the input itself.
    let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap_or_else(|e| panic!("{e:?}"));
    for j in 0..4 {
        for i in 0..4 {
            let id = if i == j { 1.0 } else { 0.0 };
            assert_eq!(qz.s()[(i, j)], a[(i, j)]);
            assert_eq!(
                (qz.t()[(i, j)], qz.q()[(i, j)], qz.z()[(i, j)]),
                (id, id, id)
            );
        }
    }
    // Select the second pair (the 3rd and 4th eigenvalue): its swap with the
    // identical leading pair is rejected (DTGSEN INFO = 1, DGGES N+3).
    let mut calls = 0usize;
    let second_pair = |_: &GeneralizedEigenvalue<f64>| {
        calls += 1;
        calls == 3 || calls == 4
    };
    let err = Qz::compute_sorted(a.as_ref(), b.as_ref(), second_pair).err();
    assert_eq!(err, Some(QzReorderError::SwapFailed { position: 2 }));
    assert_eq!(calls, 4, "no final check after a rejected swap");
}

#[test]
fn test_issue_6_sorted_errors() {
    use std::error::Error;
    let mut calls = 0usize;
    let empty = Mat::<f64>::zeros(0, 0);
    let res = Qz::compute_sorted(empty.as_ref(), empty.as_ref(), |_| {
        calls += 1;
        true
    });
    assert_eq!(res.err(), Some(QzReorderError::Qz(QzError::EmptyMatrix)));
    let mut a = lcg_matrix(3, 81);
    a[(0, 2)] = f64::NAN;
    let b = lcg_matrix(3, 82);
    let res = Qz::compute_sorted(a.as_ref(), b.as_ref(), |_| {
        calls += 1;
        true
    });
    assert_eq!(
        res.err(),
        Some(QzReorderError::Qz(QzError::NotConverged { iterations: 0 }))
    );
    assert_eq!(calls, 0, "the selector must not run on invalid input");
    let from: QzReorderError = QzError::EmptyMatrix.into();
    assert_eq!(from, QzReorderError::Qz(QzError::EmptyMatrix));
    assert_eq!(from.to_string(), "QZ decomposition failed: Empty matrix");
    assert_eq!(
        from.source().map(|s| s.to_string()).as_deref(),
        Some("Empty matrix")
    );
    let swap = QzReorderError::SwapFailed { position: 2 };
    assert!(swap.source().is_none());
    assert_eq!(
        swap.to_string(),
        "Eigenvalue reordering failed: swapping the diagonal block at position 2 was rejected \
         by the stability tests (eigenvalues too close, or the pencil is badly balanced)"
    );
}

// ---------------------------------------------------------------------------
// Qz::reorder (DTGSEN, IJOB = 0, on an existing factorization)
// ---------------------------------------------------------------------------

/// Bitwise equality of two factorizations.
fn assert_bitwise_equal(label: &str, x: &Qz<f64>, y: &Qz<f64>) {
    let n = x.n();
    assert_eq!(n, y.n(), "{label}");
    for (name, p, r) in [
        ("S", x.s(), y.s()),
        ("T", x.t(), y.t()),
        ("Q", x.q(), y.q()),
        ("Z", x.z(), y.z()),
    ] {
        for j in 0..n {
            for i in 0..n {
                assert_eq!(
                    p[(i, j)].to_bits(),
                    r[(i, j)].to_bits(),
                    "{label}: {name}({i},{j}) {} vs {}",
                    p[(i, j)],
                    r[(i, j)]
                );
            }
        }
    }
    for (k, (e, f)) in x.eigenvalues().iter().zip(y.eigenvalues()).enumerate() {
        let bits = |g: &GeneralizedEigenvalue<f64>| {
            (
                g.alpha_real.to_bits(),
                g.alpha_imag.to_bits(),
                g.beta.to_bits(),
            )
        };
        assert_eq!(bits(e), bits(f), "{label}: eigenvalue {k}");
    }
}

#[test]
fn test_issue_6_reorder_matches_compute_sorted() {
    // Without DGGES scaling the only difference between the two paths is
    // the DGGBAK row permutation of Q and Z, which commutes with the column
    // updates of the reordering: the results agree bitwise.
    let (pa, pb) = permuted_pencil(3);
    let cases = [
        ("random n=9", lcg_matrix(9, 901), lcg_matrix(9, 902)),
        ("permuted n=7", pa, pb),
    ];
    for (label, a, b) in &cases {
        let n = a.nrows();
        let work = run(a.as_ref(), b.as_ref()).unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert!(work.ascale.is_none() && work.bscale.is_none(), "{label}");
        if label.starts_with("permuted") {
            assert!(
                work.ilo > 0 || work.ihi + 1 < n,
                "{label}: no DGGBAL permutation"
            );
        }
        for (name, pred) in [
            (
                "Re<0",
                Box::new(re_below(0.0)) as Box<dyn Fn(&GeneralizedEigenvalue<f64>) -> bool>,
            ),
            ("|l|<1", Box::new(abs_below(1.0))),
            (
                "Im>0",
                Box::new(|e: &GeneralizedEigenvalue<f64>| e.alpha_imag > 0.0),
            ),
        ] {
            let label = format!("{label} {name}");
            let (sorted_qz, o1) = sorted(&label, a, b, &*pred);
            let mut qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap_or_else(|e| panic!("{e:?}"));
            let o2 = qz
                .reorder(&*pred)
                .unwrap_or_else(|e| panic!("{label}: reorder failed: {e:?}"));
            assert_eq!(o1, o2, "{label}");
            assert_bitwise_equal(&label, &sorted_qz, &qz);
        }
    }
}

#[test]
fn test_issue_6_reorder_random_sweep() {
    for seed in 0..120u64 {
        let n = 2 + (seed % 11) as usize;
        let label = format!("reorder sweep seed={seed} n={n}");
        let a = lcg_matrix(n, 3000 + seed);
        let b = lcg_matrix(n, 5000 + seed);
        let pred = sweep_predicate(seed + 1);
        let (qz0, reference) = unsorted(&label, &a, &b);
        let mut qz = qz0.clone();
        let ordering = qz
            .reorder(&*pred)
            .unwrap_or_else(|e| panic!("{label}: reorder failed: {e:?}"));
        assert_sorted(
            &label,
            (a.as_ref(), b.as_ref()),
            &qz,
            ordering,
            &*pred,
            &reference,
            CHORDAL_TOL,
        );
    }
}

#[test]
fn test_issue_6_reorder_transactional_swap_failed() {
    let (a, b) = identical_pairs_pencil();
    let mut qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap_or_else(|e| panic!("{e:?}"));
    let before = qz.clone();
    let mut calls = 0usize;
    let res = qz.reorder(|_| {
        calls += 1;
        calls == 3 || calls == 4
    });
    assert_eq!(res, Err(QzReorderError::SwapFailed { position: 2 }));
    assert_eq!(calls, 4);
    assert_bitwise_equal("transactional", &before, &qz);
    // The factorization stays usable: a selection that needs no swap
    // succeeds.
    let ordering = qz
        .reorder(|e| e.alpha_imag > 0.0)
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ordering.sdim, 4);
}

#[test]
fn test_issue_6_reorder_twice() {
    let label = "reorder twice";
    let (a, b) = (lcg_matrix(10, 1001), lcg_matrix(10, 1002));
    let (mut qz, reference) = unsorted(label, &a, &b);
    let first = re_below(0.0);
    let o1 = qz.reorder(&first).unwrap_or_else(|e| panic!("{e:?}"));
    assert_sorted(
        label,
        (a.as_ref(), b.as_ref()),
        &qz,
        o1,
        &first,
        &reference,
        CHORDAL_TOL,
    );
    let second = |e: &GeneralizedEigenvalue<f64>| !first(e);
    let o2 = qz.reorder(second).unwrap_or_else(|e| panic!("{e:?}"));
    assert_sorted(
        label,
        (a.as_ref(), b.as_ref()),
        &qz,
        o2,
        &second,
        &reference,
        CHORDAL_TOL,
    );
    assert_eq!(o1.sdim + o2.sdim, 10);
}

#[test]
fn test_issue_6_reorder_n1() {
    let a = Mat::from_rows(&[&[3.0f64]]);
    let b = Mat::from_rows(&[&[-2.0f64]]);
    for select in [true, false] {
        let pred = move |_: &GeneralizedEigenvalue<f64>| select;
        let (qz, ordering) = sorted("n=1", &a, &b, &pred);
        assert_eq!(ordering.sdim, usize::from(select));
        assert_dgges_contract("n=1", a.as_ref(), b.as_ref(), &qz);
        assert_eq!((qz.s()[(0, 0)], qz.t()[(0, 0)]), (-3.0, 2.0));
        let mut qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap_or_else(|e| panic!("{e:?}"));
        let ordering = qz.reorder(pred).unwrap_or_else(|e| panic!("{e:?}"));
        assert_eq!(ordering.sdim, usize::from(select));
        assert_eq!(qz.eigenvalues()[0].value(), Some(-1.5));
    }
}
