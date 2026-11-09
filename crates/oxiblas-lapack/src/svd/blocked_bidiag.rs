//! Blocked Householder bidiagonalisation and compact-WY back-transformation.
//!
//! [`gebrd_tall`] reduces a tall-or-square matrix (`m ≥ n`) to upper
//! bidiagonal form `A = Q·B·Pᵀ` in the LAPACK `dgebrd` layout:
//!
//! - the left reflector `H_j = I − τq_j·v_j·v_jᵀ` has `v_j[j] = 1` (implied) and
//!   `v_j[j+1..m]` stored in column `j` below the diagonal;
//! - the right reflector `G_j = I − τp_j·g_j·g_jᵀ` has `g_j[j+1] = 1` (implied)
//!   and `g_j[j+2..n]` stored in row `j` right of the super-diagonal;
//! - `Q = H_0·H_1⋯H_{n−1}` and `P = G_0·G_1⋯G_{n−2}`.
//!
//! Panels of `NB` columns are reduced with the `dlabrd` recurrences (two
//! column-oriented matrix–vector passes over the trailing matrix per column)
//! and the trailing matrix is then updated with two GEMMs,
//! `A₂₂ −= V·Yᵀ + X·Uᵀ`. The last `CROSSOVER` columns use the unblocked
//! `dgebd2` sweep, whose updates are column-oriented as well.
//!
//! [`apply_reflectors_left`] multiplies a matrix by `Q` or `P` without forming
//! them: blocks of reflectors are turned into compact-WY form
//! `I − Y·T·Yᵀ` (`dlarft`) and applied with three GEMMs each (`dlarfb`).

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::dc_kernels::{DenseOps, axpy, axpy4, dot, dot4, fabs, nrm2, scal};

/// Panel width of the blocked reduction.
const NB: usize = 32;
/// Trailing size below which the unblocked reduction is used.
const CROSSOVER: usize = 128;
/// Number of reflectors per compact-WY block in the back-transformation.
const WY_BLOCK: usize = 64;

/// Bidiagonal entries and Householder scalars of a tall reduction.
#[derive(Debug, Clone)]
pub(crate) struct TallBidiag<T> {
    /// Diagonal of `B` (`n` entries).
    pub d: Vec<T>,
    /// Super-diagonal of `B` (`n − 1` entries).
    pub e: Vec<T>,
    /// Scalars of the left reflectors (`n` entries).
    pub tauq: Vec<T>,
    /// Scalars of the right reflectors (`n − 1` entries).
    pub taup: Vec<T>,
}

/// Where the Householder vectors of a reflector sequence are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectorLayout {
    /// Left reflectors (`Q`): vector `j` lives in column `j`, rows `j+1..m`.
    Columns,
    /// Right reflectors (`P`): vector `j` lives in row `j`, columns `j+2..n`.
    Rows,
}

/// Generates an elementary reflector `H = I − τ·v·vᵀ` with
/// `H·x = (β, 0, …, 0)ᵀ` (LAPACK `dlarfg`).
///
/// On entry `x[0]` is `α`; on return `x[0] = β` and `x[1..]` holds `v[1..]`
/// (`v[0] = 1` is implied). Returns `τ` (zero when `x[1..]` is already zero).
pub(crate) fn larfg<T: Field + Real>(x: &mut [T]) -> T {
    if x.len() <= 1 {
        return T::zero();
    }
    let (head, tail) = x.split_at_mut(1);
    let mut alpha = head[0];
    let mut xnorm = nrm2(tail);
    if xnorm == T::zero() {
        return T::zero();
    }
    let signed = |mag: T, s: T| if s < T::zero() { -fabs(mag) } else { fabs(mag) };
    let mut beta = -signed(<T as Real>::hypot(alpha, xnorm), alpha);

    // Rescale when β would be subnormal (dlarfg's SAFMIN loop).
    let safmin = <T as Scalar>::min_positive() / <T as Scalar>::epsilon();
    let mut knt = 0usize;
    if fabs(beta) < safmin {
        let rsafmn = T::one() / safmin;
        loop {
            knt += 1;
            scal(rsafmn, tail);
            beta = beta * rsafmn;
            alpha = alpha * rsafmn;
            if !(fabs(beta) < safmin && knt < 20) {
                break;
            }
        }
        xnorm = nrm2(tail);
        beta = -signed(<T as Real>::hypot(alpha, xnorm), alpha);
    }

    let tau = (beta - alpha) / beta;
    scal(T::one() / (alpha - beta), tail);
    for _ in 0..knt {
        beta = beta * safmin;
    }
    head[0] = beta;
    tau
}

/// Reduces `a` (`m × n`, `m ≥ n`) to upper bidiagonal form in place.
///
/// On return `a` holds the Householder vectors in the layout described in the
/// module documentation, and the returned [`TallBidiag`] holds `d`, `e`,
/// `τq` and `τp`.
pub(crate) fn gebrd_tall<T, B>(ops: B, a: &mut Mat<T>) -> TallBidiag<T>
where
    T: Field + Real + bytemuck::Zeroable,
    B: DenseOps<T>,
{
    let m = a.nrows();
    let n = a.ncols();
    debug_assert!(m >= n);
    let mut out = TallBidiag {
        d: vec![T::zero(); n],
        e: vec![T::zero(); n.saturating_sub(1)],
        tauq: vec![T::zero(); n],
        taup: vec![T::zero(); n.saturating_sub(1)],
    };
    if n == 0 {
        return out;
    }

    let nb = NB;
    let nx = CROSSOVER.max(nb);
    let mut p = 0;
    if n > nx {
        let mut x = Mat::<T>::zeros(m, nb);
        let mut y = Mat::<T>::zeros(n, nb);
        while n - p > nx {
            labrd(a, p, nb, &mut x, &mut y, &mut out);

            // Trailing update A₂₂ −= V·Y₂ᵀ + X₂·U.
            let r0 = p + nb;
            let mr = m - r0;
            let nr = n - r0;
            {
                let (left, right) = a.as_mut().split_cols(r0);
                let (top_right, mut a22) = right.split_rows(r0);
                let v = left.rb().submatrix(r0, p, mr, nb);
                let u = top_right.rb().submatrix(p, 0, nb, nr);
                let y2 = y.as_ref().submatrix(nb, 0, nr, nb);
                let x2 = x.as_ref().submatrix(nb, 0, mr, nb);
                ops.gemm(false, true, -T::one(), v, y2, T::one(), a22.rb_mut());
                ops.gemm(false, false, -T::one(), x2, u, T::one(), a22);
            }
            for j in p..r0 {
                a[(j, j)] = out.d[j];
                a[(j, j + 1)] = out.e[j];
            }
            p = r0;
        }
    }
    gebd2_tall(a, p, &mut out);
    out
}

/// Reduces the first `nb` rows and columns of `a[p.., p..]` (LAPACK `dlabrd`,
/// `m ≥ n` branch) and returns the panel matrices `X` (rows `p..m`) and `Y`
/// (rows ↔ columns `p..n`) needed for the trailing update.
///
/// On return the diagonal and super-diagonal entries of the panel rows hold
/// the implied unit entries of the reflectors (restored by the caller).
fn labrd<T: Field + Real>(
    a: &mut Mat<T>,
    p: usize,
    nb: usize,
    x: &mut Mat<T>,
    y: &mut Mat<T>,
    out: &mut TallBidiag<T>,
) {
    let m = a.nrows();
    let n = a.ncols();
    let lda = a.row_stride();
    let ldx = x.row_stride();
    let ldy = y.row_stride();
    let ml = m - p;
    let nl = n - p;

    let mut tmp = vec![T::zero(); nb + 1];
    let mut tmp2 = vec![T::zero(); nb + 1];
    let mut wvec = vec![T::zero(); ml];
    let mut zvec = vec![T::zero(); ml];
    let mut g = vec![T::zero(); nl];
    let mut rowacc = vec![T::zero(); nl];

    for ii in 0..nb {
        let j = p + ii;
        let ng = n - j - 1; // length of the right reflector

        // ---- Column update and left reflector ----------------------------
        // a[j..m, j] −= V[j..m, 0..ii]·Y[ii, 0..ii]ᵀ + X[ii.., 0..ii]·a[p..j, j]
        {
            let ad = a.raw_data_mut();
            let (prev, cur) = ad.split_at_mut(j * lda);
            let (above, colj) = cur[..m].split_at_mut(j);
            let yd = y.raw_data();
            let xd = x.raw_data();
            for t in 0..ii {
                let vcol = &prev[(p + t) * lda + j..(p + t) * lda + m];
                axpy(-yd[ii + t * ldy], vcol, colj);
                let xcol = &xd[t * ldx + ii..t * ldx + ml];
                axpy(-above[p + t], xcol, colj);
            }
            let tau = larfg(colj);
            out.d[j] = colj[0];
            out.tauq[j] = tau;
            if ng > 0 {
                colj[0] = T::one();
            }
        }
        if ng == 0 {
            continue;
        }
        let tauq = out.tauq[j];

        // ---- Y[ii+1..nl, ii] ----------------------------------------------
        {
            let ad = a.raw_data();
            let u = &ad[j * lda + j..j * lda + m];
            let xd = x.raw_data();
            for t in 0..ii {
                tmp2[t] = dot(&xd[t * ldx + ii..t * ldx + ml], u);
                tmp[t] = dot(&ad[(p + t) * lda + j..(p + t) * lda + m], u);
            }
            // Fused vector over rows p..m: [−X[ii.., 0..ii]ᵀu ; u].
            for t in 0..ii {
                wvec[t] = -tmp2[t];
            }
            wvec[ii..ml].copy_from_slice(u);

            let yd = y.raw_data_mut();
            let (yprev, ycur) = yd.split_at_mut(ii * ldy);
            let ycol = &mut ycur[ii + 1..nl];
            let colp = |c: usize| &ad[c * lda + p..c * lda + m];
            let wv = &wvec[..ml];
            let mut c = j + 1;
            while c + 4 <= n {
                let r = dot4([colp(c), colp(c + 1), colp(c + 2), colp(c + 3)], wv);
                let q = c - j - 1;
                ycol[q..q + 4].copy_from_slice(&r);
                c += 4;
            }
            while c < n {
                ycol[c - j - 1] = dot(colp(c), wv);
                c += 1;
            }
            for t in 0..ii {
                axpy(-tmp[t], &yprev[t * ldy + ii + 1..t * ldy + nl], ycol);
            }
            scal(tauq, ycol);
        }

        // ---- Row update and right reflector -------------------------------
        // a[j, j+1..n] −= Y[ii+1.., 0..=ii]·a[j, p..=j]ᵀ + a[p..j, j+1..n]ᵀ·X[ii, 0..ii]ᵀ
        {
            for t in 0..=ii {
                tmp[t] = a[(j, p + t)];
            }
            for t in 0..ii {
                tmp2[t] = x[(ii, t)];
            }
            let racc = &mut rowacc[..ng];
            racc.fill(T::zero());
            let yd = y.raw_data();
            for t in 0..=ii {
                axpy(tmp[t], &yd[t * ldy + ii + 1..t * ldy + nl], racc);
            }
            let ad = a.raw_data();
            for (q, c) in (j + 1..n).enumerate() {
                let corr = dot(&ad[c * lda + p..c * lda + j], &tmp2[..ii]);
                g[q] = ad[c * lda + j] - racc[q] - corr;
            }
        }
        let taup = larfg(&mut g[..ng]);
        out.e[j] = g[0];
        out.taup[j] = taup;
        g[0] = T::one();
        for (q, c) in (j + 1..n).enumerate() {
            a[(j, c)] = g[q];
        }

        // ---- X[ii+1..ml, ii] ----------------------------------------------
        {
            let ad = a.raw_data();
            let z = &mut zvec[..ml];
            z.fill(T::zero());
            let col = |c: usize| &ad[c * lda + p..c * lda + m];
            let mut c = j + 1;
            while c + 4 <= n {
                let q = c - j - 1;
                axpy4(
                    [g[q], g[q + 1], g[q + 2], g[q + 3]],
                    [col(c), col(c + 1), col(c + 2), col(c + 3)],
                    z,
                );
                c += 4;
            }
            while c < n {
                axpy(g[c - j - 1], col(c), z);
                c += 1;
            }
            // z[0..ii] = a[p..j, j+1..n]·g ; z[ii+1..] = a[j+1..m, j+1..n]·g
            let yd = y.raw_data();
            for t in 0..=ii {
                tmp[t] = dot(&yd[t * ldy + ii + 1..t * ldy + nl], &g[..ng]);
            }
            let xd = x.raw_data_mut();
            let (xprev, xcur) = xd.split_at_mut(ii * ldx);
            let xcol = &mut xcur[ii + 1..ml];
            xcol.copy_from_slice(&z[ii + 1..ml]);
            for t in 0..=ii {
                axpy(-tmp[t], &ad[(p + t) * lda + j + 1..(p + t) * lda + m], xcol);
            }
            for t in 0..ii {
                axpy(-z[t], &xprev[t * ldx + ii + 1..t * ldx + ml], xcol);
            }
            scal(taup, xcol);
        }
    }
}

/// Unblocked reduction (LAPACK `dgebd2`, `m ≥ n`) of `a[p.., p..]`.
///
/// Both reflector applications are column-oriented: the left one is a
/// transposed matrix–vector product followed by column axpys, the right one a
/// matrix–vector product followed by column axpys.
fn gebd2_tall<T: Field + Real>(a: &mut Mat<T>, p: usize, out: &mut TallBidiag<T>) {
    let m = a.nrows();
    let n = a.ncols();
    let lda = a.row_stride();
    let mut g = vec![T::zero(); n];
    let mut w = vec![T::zero(); m];

    for j in p..n {
        // Left reflector H_j on a[j..m, j], applied to a[j..m, j+1..n].
        {
            let ad = a.raw_data_mut();
            let (left, right) = ad.split_at_mut((j + 1) * lda);
            let colj = &mut left[j * lda + j..j * lda + m];
            let tau = larfg(colj);
            out.d[j] = colj[0];
            out.tauq[j] = tau;
            if tau != T::zero() && j + 1 < n {
                colj[0] = T::one();
                let v = &*colj;
                for c in j + 1..n {
                    let off = (c - j - 1) * lda;
                    let col = &mut right[off + j..off + m];
                    let s = dot(v, col);
                    axpy(-(tau * s), v, col);
                }
                colj[0] = out.d[j];
            }
        }
        if j + 1 >= n {
            continue;
        }

        // Right reflector G_j on a[j, j+1..n], applied to a[j+1..m, j+1..n].
        let ng = n - j - 1;
        for (q, c) in (j + 1..n).enumerate() {
            g[q] = a[(j, c)];
        }
        let tau = larfg(&mut g[..ng]);
        out.e[j] = g[0];
        out.taup[j] = tau;
        for (q, c) in (j + 1..n).enumerate() {
            a[(j, c)] = g[q];
        }
        g[0] = T::one();
        if tau == T::zero() || j + 1 >= m {
            continue;
        }
        let rows = m - j - 1;
        let wv = &mut w[..rows];
        wv.fill(T::zero());
        {
            let ad = a.raw_data();
            for (q, c) in (j + 1..n).enumerate() {
                axpy(g[q], &ad[c * lda + j + 1..c * lda + m], wv);
            }
        }
        let ad = a.raw_data_mut();
        for (q, c) in (j + 1..n).enumerate() {
            let coef = -(tau * g[q]);
            axpy(coef, wv, &mut ad[c * lda + j + 1..c * lda + m]);
        }
    }
}

/// Multiplies `c` from the left by `Q = H_0⋯H_{k−1}` (layout
/// [`ReflectorLayout::Columns`]) or `P = G_0⋯G_{k−1}` (layout
/// [`ReflectorLayout::Rows`]), where `k = tau.len()`.
///
/// For `Columns`, `c.nrows()` must equal `refl.nrows()`; for `Rows`,
/// `c.nrows()` must equal `refl.ncols()`. Reflectors are applied in blocks of
/// [`WY_BLOCK`] as `C ← (I − Y·T·Yᵀ)·C` (`dlarft` + `dlarfb`), last block
/// first, so the product is formed exactly as `H_0·(H_1·(⋯·C))`.
pub(crate) fn apply_reflectors_left<T, B>(
    ops: B,
    refl: &Mat<T>,
    tau: &[T],
    layout: ReflectorLayout,
    c: &mut Mat<T>,
) where
    T: Field + Real + bytemuck::Zeroable,
    B: DenseOps<T>,
{
    let k = tau.len();
    let rows = c.nrows();
    let ncols = c.ncols();
    if k == 0 || ncols == 0 {
        return;
    }
    let offset = match layout {
        ReflectorLayout::Columns => 0,
        ReflectorLayout::Rows => 1,
    };

    let wy = WY_BLOCK;
    let nblocks = k.div_ceil(wy);
    for blk in (0..nblocks).rev() {
        let j0 = blk * wy;
        let b = wy.min(k - j0);
        let r0 = j0 + offset;
        if r0 >= rows {
            continue;
        }
        let len = rows - r0;
        let bw = b.min(len);

        // Y (len × bw): unit lower trapezoidal block of Householder vectors.
        let mut yb = Mat::<T>::zeros(len, bw);
        for t in 0..bw {
            yb[(t, t)] = T::one();
            let refl_idx = j0 + t;
            for r in (t + 1)..len {
                yb[(r, t)] = match layout {
                    ReflectorLayout::Columns => refl[(r0 + r, refl_idx)],
                    ReflectorLayout::Rows => refl[(refl_idx, r0 + r)],
                };
            }
        }

        // T (bw × bw), forward column-wise (dlarft), from the Gram matrix YᵀY.
        let mut gram = Mat::<T>::zeros(bw, bw);
        ops.gemm(
            true,
            false,
            T::one(),
            yb.as_ref(),
            yb.as_ref(),
            T::zero(),
            gram.as_mut(),
        );
        let mut tmat = Mat::<T>::zeros(bw, bw);
        let mut zbuf = vec![T::zero(); bw];
        for t in 0..bw {
            let tau_t = tau[j0 + t];
            tmat[(t, t)] = tau_t;
            if t == 0 || tau_t == T::zero() {
                continue;
            }
            for s in 0..t {
                zbuf[s] = -tau_t * gram[(s, t)];
            }
            for s in 0..t {
                let mut acc = T::zero();
                for q in s..t {
                    acc = acc + tmat[(s, q)] * zbuf[q];
                }
                tmat[(s, t)] = acc;
            }
        }

        // C[r0.., :] −= Y·(T·(Yᵀ·C[r0.., :])).
        let mut w1 = Mat::<T>::zeros(bw, ncols);
        let mut w2 = Mat::<T>::zeros(bw, ncols);
        let mut csub = c.submatrix_mut(r0, 0, len, ncols);
        ops.gemm(
            true,
            false,
            T::one(),
            yb.as_ref(),
            csub.rb(),
            T::zero(),
            w1.as_mut(),
        );
        ops.gemm(
            false,
            false,
            T::one(),
            tmat.as_ref(),
            w1.as_ref(),
            T::zero(),
            w2.as_mut(),
        );
        ops.gemm(
            false,
            false,
            -T::one(),
            yb.as_ref(),
            w2.as_ref(),
            T::one(),
            csub.rb_mut(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svd::dc_kernels::{BlasOps, PortableOps};

    fn lcg_matrix(m: usize, n: usize, seed: u64) -> Mat<f64> {
        let mut state = seed;
        let mut a = Mat::<f64>::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                a[(i, j)] = ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
            }
        }
        a
    }

    fn identity(n: usize) -> Mat<f64> {
        let mut eye = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            eye[(i, i)] = 1.0;
        }
        eye
    }

    fn check_reduction<B: DenseOps<f64>>(ops: B, m: usize, n: usize) {
        let a = lcg_matrix(m, n, 0x9e37_79b9 ^ (m * 31 + n) as u64);
        let mut w = a.clone();
        let f = gebrd_tall::<f64, B>(ops, &mut w);

        let mut q = identity(m);
        apply_reflectors_left::<f64, B>(ops, &w, &f.tauq, ReflectorLayout::Columns, &mut q);
        let mut pm = identity(n);
        apply_reflectors_left::<f64, B>(ops, &w, &f.taup, ReflectorLayout::Rows, &mut pm);

        // Q and P orthogonal.
        for (mat, dim) in [(&q, m), (&pm, n)] {
            for i in 0..dim {
                for j in 0..dim {
                    let mut s = 0.0;
                    for r in 0..dim {
                        s += mat[(r, i)] * mat[(r, j)];
                    }
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((s - expect).abs() < 1e-12, "m={m} n={n} orth {s}");
                }
            }
        }

        // Q·B·Pᵀ = A, with B·Pᵀ formed from the two diagonals (O(mn)).
        let mut bp = Mat::<f64>::zeros(m, n);
        for r in 0..n {
            for j in 0..n {
                let mut x = f.d[r] * pm[(j, r)];
                if r + 1 < n {
                    x += f.e[r] * pm[(j, r + 1)];
                }
                bp[(r, j)] = x;
            }
        }
        let mut max_err = 0.0f64;
        for j in 0..n {
            for i in 0..m {
                let mut s = 0.0;
                for r in 0..n {
                    s += q[(i, r)] * bp[(r, j)];
                }
                max_err = max_err.max((s - a[(i, j)]).abs());
            }
        }
        assert!(max_err < 1e-12, "m={m} n={n} reconstruction {max_err}");
    }

    #[test]
    fn unblocked_reduction_reconstructs() {
        check_reduction(BlasOps::SEQ, 7, 5);
        check_reduction(BlasOps::SEQ, 40, 40);
        check_reduction(PortableOps, 12, 9);
    }

    #[test]
    fn blocked_reduction_reconstructs() {
        // n > CROSSOVER exercises the dlabrd panels and the GEMM trailing update.
        check_reduction(BlasOps::SEQ, 190, 170);
        check_reduction(BlasOps::SEQ, 165, 165);
    }

    #[test]
    fn blocked_matches_unblocked_bidiagonal() {
        let a = lcg_matrix(180, 170, 42);
        let mut w1 = a.clone();
        let f_blocked = gebrd_tall::<f64, BlasOps>(BlasOps::SEQ, &mut w1);
        let mut w2 = a.clone();
        let mut f_plain = TallBidiag {
            d: vec![0.0; 170],
            e: vec![0.0; 169],
            tauq: vec![0.0; 170],
            taup: vec![0.0; 169],
        };
        gebd2_tall(&mut w2, 0, &mut f_plain);
        for i in 0..170 {
            assert!((f_blocked.d[i] - f_plain.d[i]).abs() < 1e-10 * (1.0 + f_plain.d[i].abs()));
        }
        for i in 0..169 {
            assert!((f_blocked.e[i] - f_plain.e[i]).abs() < 1e-10 * (1.0 + f_plain.e[i].abs()));
        }
    }

    #[test]
    fn larfg_annihilates_tail() {
        let mut x = vec![3.0f64, 4.0, 0.0, 12.0];
        let orig = x.clone();
        let tau = larfg(&mut x);
        let beta = x[0];
        assert!((beta.abs() - 13.0).abs() < 1e-13);
        // H·orig = (β, 0, 0, 0) with v = (1, x[1..]).
        let v = [1.0, x[1], x[2], x[3]];
        let vx: f64 = v.iter().zip(&orig).map(|(a, b)| a * b).sum();
        for i in 0..4 {
            let hx = orig[i] - tau * v[i] * vx;
            let expect = if i == 0 { beta } else { 0.0 };
            assert!((hx - expect).abs() < 1e-12);
        }
        let mut zero_tail = vec![-2.0f64, 0.0, 0.0];
        assert_eq!(larfg(&mut zero_tail), 0.0);
        assert_eq!(zero_tail[0], -2.0);
    }
}
