//! Shared helpers for the QZ unit tests.
//!
//! Provides deterministic pseudo-random pencils (same LCG as the SVD tests),
//! LAPACK `ddrges`-style backward-error and orthogonality ratios checked
//! against `THRESH = 30`, a structure check of the generalized real Schur
//! form, and chordal-distance eigenvalue matching against independent
//! oracles. Everything is evaluated in `f64`, whatever the scalar type of the
//! decomposition, so `f32` factorizations are judged against `f32::EPSILON`
//! without additional rounding noise from the checks themselves.

use num_traits::ToPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::Qz;

/// LAPACK `ddrges` pass/fail threshold for the scaled test ratios.
pub(super) const THRESH: f64 = 30.0;

/// Generalized eigenvalue as `(Re α, Im α, β)` in `f64`.
pub(super) type Triple = (f64, f64, f64);

/// Deterministic pseudo-random value in `[-1, 1)` (no external rng); same
/// construction as the SVD tests (`svd/qr_based.rs`).
pub(super) fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

/// Deterministic pseudo-random `n × n` matrix with entries in `[-1, 1)`.
pub(super) fn lcg_matrix(n: usize, seed: u64) -> Mat<f64> {
    let mut state: u64 =
        0x2545_f491_4f6c_dd1d ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (n as u64);
    let mut m = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = lcg(&mut state);
        }
    }
    m
}

/// Converts a real scalar to `f64` (NaN if it is not representable).
pub(super) fn to_f64<T: Real>(x: T) -> f64 {
    ToPrimitive::to_f64(&x).unwrap_or(f64::NAN)
}

/// Copies a matrix view into an `f64` matrix.
pub(super) fn mat_to_f64<T: Real>(m: MatRef<'_, T>) -> Mat<f64> {
    let mut out = Mat::<f64>::zeros(m.nrows(), m.ncols());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out[(i, j)] = to_f64(m[(i, j)]);
        }
    }
    out
}

/// Rounds an `f64` matrix to another real scalar type.
pub(super) fn mat_from_f64<T: Field + Real + bytemuck::Zeroable>(m: &Mat<f64>) -> Mat<T> {
    let mut out = Mat::<T>::zeros(m.nrows(), m.ncols());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out[(i, j)] = T::from_f64(m[(i, j)]).unwrap_or_else(T::nan);
        }
    }
    out
}

/// Returns `alpha · m`.
pub(super) fn scaled(m: &Mat<f64>, alpha: f64) -> Mat<f64> {
    let mut out = m.clone();
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out[(i, j)] = alpha * m[(i, j)];
        }
    }
    out
}

/// Frobenius norm with `dlassq`-style scaling (no overflow near `f64::MAX`).
pub(super) fn frob(m: &Mat<f64>) -> f64 {
    let mut scale = 0.0f64;
    let mut sumsq = 1.0f64;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let x = m[(i, j)].abs();
            if x != 0.0 {
                if scale < x {
                    sumsq = 1.0 + sumsq * (scale / x) * (scale / x);
                    scale = x;
                } else {
                    sumsq += (x / scale) * (x / scale);
                }
            }
        }
    }
    scale * sumsq.sqrt()
}

/// `a · b`.
pub(super) fn matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let mut c = Mat::<f64>::zeros(a.nrows(), b.ncols());
    for j in 0..b.ncols() {
        for k in 0..a.ncols() {
            let bkj = b[(k, j)];
            if bkj != 0.0 {
                for i in 0..a.nrows() {
                    c[(i, j)] += a[(i, k)] * bkj;
                }
            }
        }
    }
    c
}

/// `a · bᵀ`.
pub(super) fn matmul_bt(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    matmul(a, &b.transpose())
}

/// `‖M − Q·X·Zᵀ‖_F / (n · eps · max(‖M‖_F, safmin))` (LAPACK `dget51` mode 1).
pub(super) fn residual_ratio<T: Field + Real>(
    m: MatRef<'_, T>,
    q: MatRef<'_, T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
) -> f64 {
    let n = m.nrows();
    let m64 = mat_to_f64(m);
    let recon = matmul_bt(&matmul(&mat_to_f64(q), &mat_to_f64(x)), &mat_to_f64(z));
    let mut diff = m64.clone();
    for j in 0..n {
        for i in 0..n {
            diff[(i, j)] = m64[(i, j)] - recon[(i, j)];
        }
    }
    let eps = to_f64(<T as Scalar>::epsilon());
    let safmin = to_f64(<T as Scalar>::min_positive());
    frob(&diff) / (n as f64 * eps * frob(&m64).max(safmin))
}

/// `‖QᵀQ − I‖_F / (n · eps)` (LAPACK `dget51` mode 3).
pub(super) fn orthogonality_ratio<T: Field + Real>(q: MatRef<'_, T>) -> f64 {
    let n = q.nrows();
    let q64 = mat_to_f64(q);
    let mut qtq = matmul(&q64.transpose(), &q64);
    for i in 0..n {
        qtq[(i, i)] -= 1.0;
    }
    frob(&qtq) / (n as f64 * to_f64(<T as Scalar>::epsilon()))
}

/// Complex product `(a.0 + i a.1)(b.0 + i b.1)`.
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Rescales `(Re α, Im α, β)` by its largest component, so products and
/// norms of eigenvalue triples of 1e300-scaled pencils cannot overflow.
fn normalize(t: Triple) -> Triple {
    let m = t.0.abs().max(t.1.abs()).max(t.2.abs());
    if m == 0.0 || !m.is_finite() {
        t
    } else {
        (t.0 / m, t.1 / m, t.2 / m)
    }
}

/// `dget53`-style test of one generalized eigenvalue `(α, β)` of a 2×2
/// block pencil `(S_b, T_b)`: `σ_min(β S_b − α T_b) / (ulp · max(|β|‖S_b‖, |α|‖T_b‖))`.
fn block_eigen_ratio(s: [[f64; 2]; 2], t: [[f64; 2]; 2], e: Triple, ulp: f64) -> f64 {
    let norm1 =
        |m: [[f64; 2]; 2]| (m[0][0].abs() + m[1][0].abs()).max(m[0][1].abs() + m[1][1].abs());
    // Scale the block pencil and the eigenvalue pair to unit size.
    let sc = 1.0 / norm1(s).max(norm1(t)).max(f64::MIN_POSITIVE);
    let e = normalize(e);
    let wn = (e.0 * e.0 + e.1 * e.1 + e.2 * e.2)
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let (ar, ai, be) = (e.0 / wn, e.1 / wn, e.2 / wn);
    // Scale the block entries first: `(be · sc) · s` would form `be · sc`,
    // which is subnormal for unbalanced pencils at extreme scale (e.g.
    // ‖S‖ ≈ 1e300, ‖T‖ ≈ 1e290) and loses about ten bits.
    let c = |i: usize, j: usize| {
        (
            be * (sc * s[i][j]) - ar * (sc * t[i][j]),
            -ai * (sc * t[i][j]),
        )
    };
    let (c11, c12, c21, c22) = (c(0, 0), c(0, 1), c(1, 0), c(1, 1));
    let p = cmul(c11, c22);
    let r = cmul(c12, c21);
    let det = ((p.0 - r.0).powi(2) + (p.1 - r.1).powi(2)).sqrt();
    let cabs = |z: (f64, f64)| z.0.abs() + z.1.abs();
    let cnorm = (cabs(c11) + cabs(c21))
        .max(cabs(c12) + cabs(c22))
        .max(f64::MIN_POSITIVE);
    let sigmin = det / cnorm;
    let alpha_abs = (ar * ar + ai * ai).sqrt();
    let denom = ulp
        * (be.abs() * (sc * norm1(s)))
            .max(alpha_abs * (sc * norm1(t)))
            .max(1e-300);
    sigmin / denom
}

/// Structure violations of the generalized real Schur form returned by `Qz`.
///
/// Checks (bitwise where the `DGGES` contract is exact): `T` upper
/// triangular, `S` upper Hessenberg without two consecutive nonzero
/// subdiagonals, 2×2 blocks with diagonal positive `T` block, positive and
/// negative `alpha_imag` members, conjugate eigenvalue ratios and a small
/// `dget53` ratio, 1×1 blocks with `alpha_real == S(j,j)`,
/// `beta == T(j,j) >= 0` and `alpha_imag == 0`.
pub(super) fn structure_violations<T: Field + Real + bytemuck::Zeroable>(
    qz: &Qz<T>,
) -> Vec<String> {
    let n = qz.n();
    let s = qz.s();
    let t = qz.t();
    let e = qz.eigenvalues();
    let mut bad = Vec::new();
    if e.len() != n {
        bad.push(format!("{} eigenvalues for n = {n}", e.len()));
        return bad;
    }
    let zero = T::zero();
    for j in 0..n {
        for i in (j + 1)..n {
            if t[(i, j)] != zero {
                bad.push(format!("T({i},{j}) = {} below the diagonal", t[(i, j)]));
            }
            if i > j + 1 && s[(i, j)] != zero {
                bad.push(format!("S({i},{j}) = {} below the subdiagonal", s[(i, j)]));
            }
        }
    }
    for i in 1..n.saturating_sub(1) {
        if s[(i, i - 1)] != zero && s[(i + 1, i)] != zero {
            bad.push(format!("two consecutive nonzero subdiagonals at {i}"));
        }
    }
    let ulp = to_f64(<T as Scalar>::epsilon());
    let mut i = 0;
    while i < n {
        if i + 1 < n && s[(i + 1, i)] != zero {
            let (e1, e2) = (e[i], e[i + 1]);
            if t[(i, i + 1)] != zero {
                bad.push(format!("2x2 block at {i}: T({i},{}) != 0", i + 1));
            }
            if !(t[(i, i)] > zero && t[(i + 1, i + 1)] > zero) {
                bad.push(format!("2x2 block at {i}: T diagonal not positive"));
            }
            if !(e1.alpha_imag > zero && e2.alpha_imag < zero) {
                bad.push(format!("2x2 block at {i}: alpha_imag signs wrong"));
            }
            if !(e1.beta > zero && e2.beta > zero) {
                bad.push(format!("2x2 block at {i}: beta not positive"));
            }
            let t1 = (
                to_f64(e1.alpha_real),
                to_f64(e1.alpha_imag),
                to_f64(e1.beta),
            );
            let t2 = (
                to_f64(e2.alpha_real),
                to_f64(e2.alpha_imag),
                to_f64(e2.beta),
            );
            // Conjugate ratios: α1/β1 = conj(α2/β2) up to rounding (not an
            // exact `alpha_imag` negation: β1 ≠ β2 in general).
            let (n1, n2) = (normalize(t1), normalize(t2));
            let scale = (n1.0.abs() + n1.1.abs()) * n2.2 + (n2.0.abs() + n2.1.abs()) * n1.2;
            let re_gap = (n1.0 * n2.2 - n2.0 * n1.2).abs();
            let im_gap = (n1.1 * n2.2 + n2.1 * n1.2).abs();
            if !(re_gap.max(im_gap) <= 64.0 * ulp * scale) {
                bad.push(format!("2x2 block at {i}: pair is not conjugate"));
            }
            let sb = [
                [to_f64(s[(i, i)]), to_f64(s[(i, i + 1)])],
                [to_f64(s[(i + 1, i)]), to_f64(s[(i + 1, i + 1)])],
            ];
            let tb = [
                [to_f64(t[(i, i)]), to_f64(t[(i, i + 1)])],
                [to_f64(t[(i + 1, i)]), to_f64(t[(i + 1, i + 1)])],
            ];
            for (k, tr) in [t1, t2].into_iter().enumerate() {
                let r = block_eigen_ratio(sb, tb, tr, ulp);
                if !(r <= THRESH) {
                    bad.push(format!("2x2 block at {i}: member {k} dget53 ratio {r}"));
                }
            }
            i += 2;
        } else {
            let ev = e[i];
            if ev.alpha_imag != zero {
                bad.push(format!("1x1 block at {i}: alpha_imag != 0"));
            }
            if ev.alpha_real != s[(i, i)] {
                bad.push(format!("1x1 block at {i}: alpha_real != S({i},{i})"));
            }
            if ev.beta != t[(i, i)] {
                bad.push(format!("1x1 block at {i}: beta != T({i},{i})"));
            }
            if !(t[(i, i)] >= zero) {
                bad.push(format!("1x1 block at {i}: T({i},{i}) < 0"));
            }
            i += 1;
        }
    }
    bad
}

/// Asserts the full `DGGES` contract: both backward errors and both
/// orthogonality ratios at most `THRESH`, plus the Schur-form structure.
pub(super) fn assert_dgges_contract<T: Field + Real + bytemuck::Zeroable>(
    label: &str,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    qz: &Qz<T>,
) {
    let ra = residual_ratio(a, qz.q(), qz.s(), qz.z());
    let rb = residual_ratio(b, qz.q(), qz.t(), qz.z());
    let oq = orthogonality_ratio(qz.q());
    let oz = orthogonality_ratio(qz.z());
    assert!(
        ra <= THRESH && rb <= THRESH && oq <= THRESH && oz <= THRESH,
        "{label}: residual/orthogonality ratios rA={ra:.3e} rB={rb:.3e} \
         Q={oq:.3e} Z={oz:.3e} exceed THRESH={THRESH}"
    );
    let bad = structure_violations(qz);
    assert!(
        bad.is_empty(),
        "{label}: Schur structure violations: {bad:?}"
    );
}

/// QZ eigenvalues as `(Re α, Im α, β)` triples.
pub(super) fn eig_triples<T: Field + Real + bytemuck::Zeroable>(qz: &Qz<T>) -> Vec<Triple> {
    qz.eigenvalues()
        .iter()
        .map(|e| (to_f64(e.alpha_real), to_f64(e.alpha_imag), to_f64(e.beta)))
        .collect()
}

/// Chordal distance `|α1 β2 − α2 β1| / (‖(α1, β1)‖ ‖(α2, β2)‖)` between two
/// generalized eigenvalues with complex `α` and real `β` (well defined for
/// infinite eigenvalues).
pub(super) fn chordal(x: Triple, y: Triple) -> f64 {
    let (x, y) = (normalize(x), normalize(y));
    let num_re = x.0 * y.2 - y.0 * x.2;
    let num_im = x.1 * y.2 - y.1 * x.2;
    let nx = (x.0 * x.0 + x.1 * x.1 + x.2 * x.2).sqrt();
    let ny = (y.0 * y.0 + y.1 * y.1 + y.2 * y.2).sqrt();
    (num_re * num_re + num_im * num_im).sqrt() / (nx * ny).max(f64::MIN_POSITIVE)
}

/// Greedy chordal matching of `computed` against `reference`; returns the
/// largest matched distance (`f64::INFINITY` on a length mismatch).
pub(super) fn max_matched_chordal(computed: &[Triple], reference: &[Triple]) -> f64 {
    if computed.len() != reference.len() {
        return f64::INFINITY;
    }
    let mut used = vec![false; reference.len()];
    let mut worst = 0.0f64;
    for &c in computed {
        let mut best = f64::INFINITY;
        let mut best_k = None;
        for (k, &r) in reference.iter().enumerate() {
            if !used[k] {
                let d = chordal(c, r);
                if d < best || best_k.is_none() {
                    best = d;
                    best_k = Some(k);
                }
            }
        }
        if let Some(k) = best_k {
            used[k] = true;
        }
        worst = worst.max(best);
    }
    worst
}

/// Random orthogonal matrix: the `Q` factor of a plain Householder QR of an
/// LCG matrix (independent of the code under test).
pub(super) fn random_orthogonal(n: usize, seed: u64) -> Mat<f64> {
    let mut a = lcg_matrix(n, seed);
    let mut q = Mat::<f64>::eye(n);
    for k in 0..n.saturating_sub(1) {
        let norm = (k..n).map(|i| a[(i, k)] * a[(i, k)]).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[(k, k)] >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0f64; n - k];
        v[0] = a[(k, k)] - alpha;
        for i in (k + 1)..n {
            v[i - k] = a[(i, k)];
        }
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        for j in k..n {
            let f = 2.0 * (k..n).map(|i| v[i - k] * a[(i, j)]).sum::<f64>() / vnorm2;
            for i in k..n {
                a[(i, j)] -= f * v[i - k];
            }
        }
        for i in 0..n {
            let f = 2.0 * (k..n).map(|j| q[(i, j)] * v[j - k]).sum::<f64>() / vnorm2;
            for j in k..n {
                q[(i, j)] -= f * v[j - k];
            }
        }
    }
    q
}
