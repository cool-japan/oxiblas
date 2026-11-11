//! Householder tridiagonal reduction shared by the dense symmetric
//! eigensolvers, and the matching back-transformation.
//!
//! [`reduce`] is LAPACK `DSYTRD` with `UPLO = 'L'`: only the lower triangle
//! of the work matrix is read, the reflectors are kept below the subdiagonal
//! and `Q` is never formed. With a GEMM back end ([`MatMul::BLOCKED`]) panels
//! of [`NB`] columns are reduced as in `DLATRD` and the trailing matrix gets
//! one rank-`2·NB` update `A₂₂ -= V·Wᵀ + W·Vᵀ` per panel, done by GEMM on
//! lower-trapezoidal column blocks; those blocks include whole diagonal
//! blocks, so the update also writes (but never reads) the strict upper
//! triangle of each diagonal block. Otherwise (and for the last [`NX`]
//! columns) the unblocked, column-oriented `DSYTD2` is used, which writes
//! only the lower triangle.
//!
//! [`apply_q`] computes `Q·Z` (`DORMTR`, left, no transpose) by applying the
//! stored reflectors in blocks with the compact-WY representation
//! `H(k)⋯H(k+nb-1) = I - V·T·Vᵀ` (`DLARFT` + `DLARFB`). The test-only
//! `form_q` does the same on the identity, skipping the columns a block
//! cannot touch (`DORGTR`).

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::Mat;

use super::sym_kernels::{Consts, MatMul, axpy, dot, larfg, scal, symv_lower, syr2_lower};

/// Panel width of the blocked reduction.
const NB: usize = 32;
/// Columns left to the unblocked reduction (blocked only while more remain).
const NX: usize = 128;
/// Column-block width of the trailing GEMM update (bounds the triangle
/// overhead of computing full diagonal blocks).
const UPDATE_BLOCK: usize = 128;
/// Reflectors per compact-WY block in [`apply_q`] (and the test-only `form_q`).
const WY_BLOCK: usize = 64;

/// Result of [`reduce`]: `A = Q·T·Qᵀ` with `T = tridiag(e, d, e)`.
///
/// Reflector `H(j) = I - tau[j]·v·vᵀ` (`j < n - 1`) has `v[0..=j] = 0`,
/// `v[j + 1] = 1` and `v[j + 2..n]` stored in column `j` of the work matrix
/// below the subdiagonal; `Q = H(0)·H(1)⋯H(n-2)`.
#[derive(Debug, Clone)]
pub(crate) struct Tridiagonal<T> {
    /// Diagonal of `T` (length `n`).
    pub(crate) d: Vec<T>,
    /// Off-diagonal of `T` (length `n - 1`).
    pub(crate) e: Vec<T>,
    /// Reflector scalars (length `n - 1`).
    pub(crate) tau: Vec<T>,
}

/// Reduces the symmetric matrix whose lower triangle is stored in `a` to
/// tridiagonal form (LAPACK `DSYTRD`, lower).
///
/// On return the reflectors are stored below the subdiagonal of `a`; the
/// rest of `a` is workspace. `par` is passed to every GEMM (bitwise the same
/// result for every `Par`).
pub(crate) fn reduce<T, G>(a: &mut Mat<T>, par: Par) -> Tridiagonal<T>
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
{
    let n = a.nrows();
    let mut d = vec![T::zero(); n];
    let mut e = vec![T::zero(); n.saturating_sub(1)];
    let mut tau = vec![T::zero(); n.saturating_sub(1)];
    if n == 0 {
        return Tridiagonal { d, e, tau };
    }

    let mut i = 0;
    if G::BLOCKED && n > NX + NB {
        let mut w = Mat::<T>::zeros(n, NB);
        let mut x = Mat::<T>::zeros(n, 2 * NB);
        let mut yt = Mat::<T>::zeros(2 * NB, n);
        let mut tmp = vec![T::zero(); NB];
        while n - i > NX {
            {
                let ld = a.row_stride();
                let ldw = w.row_stride();
                let panel = Panel {
                    n,
                    i,
                    nb: NB,
                    ld,
                    ldw,
                };
                latrd_lower(
                    &panel,
                    a.raw_data_mut(),
                    w.raw_data_mut(),
                    &mut tmp,
                    &mut d,
                    &mut e,
                    &mut tau,
                );
            }
            trailing_update::<T, G>(a, &w, &mut x, &mut yt, i, NB, par);
            i += NB;
        }
    }

    let mut wbuf = vec![T::zero(); n];
    let ld = a.row_stride();
    sytd2_lower(
        a.raw_data_mut(),
        ld,
        n,
        i,
        &mut wbuf,
        &mut d,
        &mut e,
        &mut tau,
    );
    Tridiagonal { d, e, tau }
}

/// Geometry of one `DLATRD` panel.
struct Panel {
    /// Matrix order.
    n: usize,
    /// First (global) column of the panel.
    i: usize,
    /// Panel width.
    nb: usize,
    /// Leading dimension of the matrix.
    ld: usize,
    /// Leading dimension of `W` (panel-local rows `0..n - i`).
    ldw: usize,
}

/// Reduces the `nb` panel columns starting at column `p.i` (LAPACK
/// `DLATRD`, lower) and returns `W` so that the trailing update is
/// `A₂₂ -= V·Wᵀ + W·Vᵀ`.
///
/// The trailing matrix is not modified; the panel's reflectors are stored in
/// `a` with an explicit unit on the subdiagonal (needed by the update).
fn latrd_lower<T: Field + Real>(
    p: &Panel,
    a: &mut [T],
    w: &mut [T],
    tmp: &mut [T],
    d: &mut [T],
    e: &mut [T],
    tau: &mut [T],
) {
    let Panel { n, i, nb, ld, ldw } = *p;
    let m = n - i;
    let c = Consts::<T>::new();
    for j in 0..nb {
        let col = i + j;
        // Bring column `col` (rows col..n) up to date with the previous
        // panel columns: A(:,col) -= V(:,0..j)·W(col,0..j)ᵀ + W(:,0..j)·V(col,0..j)ᵀ.
        if j > 0 {
            let (left, right) = a.split_at_mut(col * ld);
            let target = &mut right[col..n];
            for q in 0..j {
                let vbase = (i + q) * ld;
                let vcol = &left[vbase + col..vbase + n];
                let v_row = left[vbase + col];
                let wcol = &w[q * ldw + j..q * ldw + m];
                let w_row = w[q * ldw + j];
                for ((t, &vv), &ww) in target.iter_mut().zip(vcol).zip(wcol) {
                    *t = Real::mul_add(-w_row, vv, Real::mul_add(-v_row, ww, *t));
                }
            }
        }
        d[col] = a[col * ld + col];
        if col + 1 >= n {
            continue;
        }

        // Reflector annihilating A(col+2..n, col).
        let base = col * ld;
        let alpha = a[base + col + 1];
        let (beta, t) = larfg(alpha, &mut a[base + col + 2..base + n]);
        e[col] = beta;
        tau[col] = t;
        a[base + col + 1] = T::one();

        let a_ro: &[T] = a;
        let v = &a_ro[base + col + 1..base + n];
        let len = n - col - 1;
        let (w_left, w_right) = w.split_at_mut(j * ldw);
        let wj = &mut w_right[j + 1..m];

        // W(:,j) = A₂₂·v with the (not yet updated) trailing matrix ...
        symv_lower(len, a_ro, (col + 1) * ld + col + 1, ld, v, wj);
        // ... corrected for the earlier panel reflectors.
        if j > 0 {
            for (q, tq) in tmp.iter_mut().enumerate().take(j) {
                *tq = dot(&w_left[q * ldw + j + 1..q * ldw + m], v);
            }
            for (q, &tq) in tmp.iter().enumerate().take(j) {
                let vbase = (i + q) * ld;
                axpy(-tq, &a_ro[vbase + col + 1..vbase + n], wj);
            }
            for (q, tq) in tmp.iter_mut().enumerate().take(j) {
                let vbase = (i + q) * ld;
                *tq = dot(&a_ro[vbase + col + 1..vbase + n], v);
            }
            for (q, &tq) in tmp.iter().enumerate().take(j) {
                axpy(-tq, &w_left[q * ldw + j + 1..q * ldw + m], wj);
            }
        }
        scal(t, wj);
        let shift = -c.half * t * dot(wj, v);
        axpy(shift, v, wj);
    }
}

/// `A₂₂ -= V·Wᵀ + W·Vᵀ = [V W]·[W V]ᵀ` on the lower triangle of the trailing
/// matrix after the panel at column `i` (lower-trapezoidal GEMM blocks, which
/// also overwrite the strict upper triangle of their diagonal blocks).
///
/// Each column block is one pass with `k = 2·nb`. The triangle-only SYR2K of
/// `oxiblas_blas` (which runs its two `k = nb` segments as two passes over
/// the triangle) made the whole eigensolver only 1-3 % faster (n = 500..2000,
/// one thread, Apple M3), so it is not used here.
fn trailing_update<T, G>(
    a: &mut Mat<T>,
    w: &Mat<T>,
    x: &mut Mat<T>,
    yt: &mut Mat<T>,
    i: usize,
    nb: usize,
    par: Par,
) where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
{
    let n = a.nrows();
    let i2 = i + nb;
    if i2 >= n {
        return;
    }
    let m2 = n - i2;
    // X = [V W] and Yᵀ = [W V]ᵀ on rows i2..n.
    {
        let lda = a.row_stride();
        let ldw = w.row_stride();
        let ldx = x.row_stride();
        let ldy = yt.row_stride();
        let av = a.raw_data();
        let wv = w.raw_data();
        let xv = x.raw_data_mut();
        for q in 0..nb {
            let vcol = &av[(i + q) * lda + i2..(i + q) * lda + n];
            let wcol = &wv[q * ldw + nb..q * ldw + nb + m2];
            xv[q * ldx..q * ldx + m2].copy_from_slice(vcol);
            xv[(nb + q) * ldx..(nb + q) * ldx + m2].copy_from_slice(wcol);
        }
        // Transposed copy in row blocks that stay in cache.
        let yv = yt.raw_data_mut();
        let mut r0 = 0;
        while r0 < m2 {
            let r1 = (r0 + 64).min(m2);
            for q in 0..nb {
                let vcol = &av[(i + q) * lda + i2..(i + q) * lda + n];
                let wcol = &wv[q * ldw + nb..q * ldw + nb + m2];
                for r in r0..r1 {
                    yv[q + r * ldy] = wcol[r];
                    yv[nb + q + r * ldy] = vcol[r];
                }
            }
            r0 = r1;
        }
    }
    let mut j0 = 0;
    while j0 < m2 {
        let j1 = (j0 + UPDATE_BLOCK).min(m2);
        let xs = x.submatrix(j0, 0, m2 - j0, 2 * nb);
        let ys = yt.submatrix(0, j0, 2 * nb, j1 - j0);
        let cs = a.submatrix_mut(i2 + j0, i2 + j0, m2 - j0, j1 - j0);
        G::gemm(false, false, -T::one(), xs, ys, T::one(), cs, par);
        j0 = j1;
    }
}

/// Unblocked reduction of columns `i0..n` (LAPACK `DSYTD2`, lower):
/// column-oriented `symv` and rank-2 update on the lower triangle.
#[allow(clippy::too_many_arguments)]
fn sytd2_lower<T: Field + Real>(
    a: &mut [T],
    ld: usize,
    n: usize,
    i0: usize,
    wbuf: &mut [T],
    d: &mut [T],
    e: &mut [T],
    tau: &mut [T],
) {
    let c = Consts::<T>::new();
    for col in i0..n {
        let base = col * ld;
        d[col] = a[base + col];
        if col + 1 >= n {
            break;
        }
        let alpha = a[base + col + 1];
        let (beta, t) = larfg(alpha, &mut a[base + col + 2..base + n]);
        e[col] = beta;
        tau[col] = t;
        if t == T::zero() {
            continue;
        }
        let len = n - col - 1;
        a[base + col + 1] = T::one();
        let (left, right) = a.split_at_mut((col + 1) * ld);
        let v = &left[base + col + 1..base + n];
        let wv = &mut wbuf[..len];
        // w = tau·A₂₂·v - (tau²/2)(vᵀA₂₂v)·v
        symv_lower(len, right, col + 1, ld, v, wv);
        scal(t, wv);
        let shift = -c.half * t * dot(wv, v);
        axpy(shift, v, wv);
        // A₂₂ -= v·wᵀ + w·vᵀ
        syr2_lower(len, right, col + 1, ld, -T::one(), v, wv);
        left[base + col + 1] = beta;
    }
}

/// Number of reflectors that can differ from the identity (the last one,
/// `H(n-2)`, acts on a single entry and always has `tau = 0`).
#[inline]
fn active_reflectors(n: usize) -> usize {
    n.saturating_sub(2)
}

/// `Z := Q·Z` for `Q = H(0)⋯H(n-2)` from [`reduce`] (LAPACK `DORMTR`,
/// `SIDE = 'L'`, `TRANS = 'N'`). `z` must have `n` rows. `par` is passed to
/// every GEMM.
pub(crate) fn apply_q<T, G>(a: &Mat<T>, tau: &[T], z: &mut Mat<T>, par: Par)
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
{
    let ncols = z.ncols();
    apply_reflectors::<T, G, _>(a, tau, z, |_| 0..ncols, par);
}

/// Overwrites `q` (`n × n`) with `Q = H(0)⋯H(n-2)` from [`reduce`]
/// (LAPACK `DORGTR`).
#[cfg(test)]
pub(crate) fn form_q<T, G>(a: &Mat<T>, tau: &[T], q: &mut Mat<T>, par: Par)
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
{
    let n = a.nrows();
    for j in 0..n {
        for i in 0..n {
            q[(i, j)] = if i == j { T::one() } else { T::zero() };
        }
    }
    // Before block [k0, k1) is applied, columns 0..=k0 of the partial product
    // are still unit vectors that the block cannot touch.
    apply_reflectors::<T, G, _>(a, tau, q, |k0| (k0 + 1)..n, par);
}

/// Applies `H(0)⋯H(n-2)` from the left to `z`, last reflector first,
/// restricting the block starting at reflector `k0` to the columns
/// `cols(k0)`.
fn apply_reflectors<T, G, F>(a: &Mat<T>, tau: &[T], z: &mut Mat<T>, cols: F, par: Par)
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
    F: Fn(usize) -> core::ops::Range<usize>,
{
    let n = a.nrows();
    let r = active_reflectors(n);
    if r == 0 || z.ncols() == 0 {
        return;
    }
    if G::BLOCKED && r > WY_BLOCK {
        apply_reflectors_blocked::<T, G, F>(a, tau, z, cols, par);
    } else {
        apply_reflectors_unblocked(a, tau, z, cols);
    }
}

/// One reflector at a time (generic path).
fn apply_reflectors_unblocked<T, F>(a: &Mat<T>, tau: &[T], z: &mut Mat<T>, cols: F)
where
    T: Field + Real + bytemuck::Zeroable,
    F: Fn(usize) -> core::ops::Range<usize>,
{
    let n = a.nrows();
    let lda = a.row_stride();
    let ldz = z.row_stride();
    let av = a.raw_data();
    let zv = z.raw_data_mut();
    for j in (0..active_reflectors(n)).rev() {
        let t = tau[j];
        if t == T::zero() {
            continue;
        }
        // v = [1; a(j+2..n, j)] acting on rows j+1..n.
        let v_tail = &av[j * lda + j + 2..j * lda + n];
        for col in cols(j) {
            let zc = &mut zv[col * ldz + j + 1..col * ldz + n];
            let (head, tail) = zc.split_at_mut(1);
            let s = head[0] + dot(v_tail, tail);
            let ts = t * s;
            head[0] -= ts;
            axpy(-ts, v_tail, tail);
        }
    }
}

/// Compact-WY blocks of [`WY_BLOCK`] reflectors, applied with three GEMMs.
fn apply_reflectors_blocked<T, G, F>(a: &Mat<T>, tau: &[T], z: &mut Mat<T>, cols: F, par: Par)
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
    F: Fn(usize) -> core::ops::Range<usize>,
{
    let n = a.nrows();
    let r = active_reflectors(n);
    let nb = WY_BLOCK;
    let ncols_max = z.ncols();
    let mut vb = Mat::<T>::zeros(n, nb);
    let mut gram = Mat::<T>::zeros(nb, nb);
    let mut tm = Mat::<T>::zeros(nb, nb);
    let mut wm = Mat::<T>::zeros(nb, ncols_max);
    let mut w2 = Mat::<T>::zeros(nb, ncols_max);

    let nblocks = r.div_ceil(nb);
    for blk in (0..nblocks).rev() {
        let k0 = blk * nb;
        let k1 = (k0 + nb).min(r);
        let kb = k1 - k0;
        let mb = n - k0 - 1;
        let range = cols(k0);
        if range.is_empty() {
            continue;
        }
        let c0 = range.start;
        let nc = range.end - range.start;

        // Explicit unit lower-trapezoidal V (mb × kb), rows k0+1..n.
        {
            let lda = a.row_stride();
            let ldv = vb.row_stride();
            let av = a.raw_data();
            let vv = vb.raw_data_mut();
            for q in 0..kb {
                let col = &mut vv[q * ldv..q * ldv + mb];
                let (head, tail) = col.split_at_mut(q + 1);
                for h in head[..q].iter_mut() {
                    *h = T::zero();
                }
                head[q] = T::one();
                let src = (k0 + q) * lda + k0 + 2 + q;
                tail.copy_from_slice(&av[src..src + (mb - q - 1)]);
            }
        }
        let vref = vb.submatrix(0, 0, mb, kb);

        // T factor (DLARFT, forward, columnwise) from the Gram matrix VᵀV.
        G::gemm(
            true,
            false,
            T::one(),
            vref,
            vref,
            T::zero(),
            gram.submatrix_mut(0, 0, kb, kb),
            par,
        );
        for q in 0..kb {
            let tq = tau[k0 + q];
            for p in 0..kb {
                tm[(p, q)] = T::zero();
            }
            tm[(q, q)] = tq;
            if tq == T::zero() {
                continue;
            }
            // T(0..q, q) = -tau_q · T(0..q, 0..q) · G(0..q, q)
            for p in 0..q {
                let mut acc = T::zero();
                for l in p..q {
                    acc += tm[(p, l)] * gram[(l, q)];
                }
                tm[(p, q)] = -tq * acc;
            }
        }

        // W = Vᵀ·Z(k0+1..n, cols);  W = T·W;  Z(k0+1..n, cols) -= V·W.
        G::gemm(
            true,
            false,
            T::one(),
            vref,
            z.submatrix(k0 + 1, c0, mb, nc),
            T::zero(),
            wm.submatrix_mut(0, 0, kb, nc),
            par,
        );
        G::gemm(
            false,
            false,
            T::one(),
            tm.submatrix(0, 0, kb, kb),
            wm.submatrix(0, 0, kb, nc),
            T::zero(),
            w2.submatrix_mut(0, 0, kb, nc),
            par,
        );
        G::gemm(
            false,
            false,
            -T::one(),
            vref,
            w2.submatrix(0, 0, kb, nc),
            T::one(),
            z.submatrix_mut(k0 + 1, c0, mb, nc),
            par,
        );
    }
}
