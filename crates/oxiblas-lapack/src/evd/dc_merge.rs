//! Cuppen divide and conquer for the symmetric tridiagonal eigenproblem
//! (LAPACK `DSTEDC` / `DLAED0` with `DLAED1`-`DLAED4`).
//!
//! The tridiagonal is split at negligible off-diagonals; blocks of at most
//! `smlsiz` rows go to the implicit QR iteration, larger ones are scaled by a
//! power of two so that their max-norm lies in `[1, 2)` and torn recursively
//! in halves by a rank-one modification down to leaves of at most `smlsiz`
//! rows. Two solved halves are merged by
//!
//! 1. deflating negligible components of the coupling vector `z` and
//!    (near-)equal poles (Gu-Eisenstat / `DLAED2`),
//! 2. solving the secular equation `1 + ρ Σ z_j²/(d_j - λ) = 0` for the
//!    remaining poles with a safeguarded rational-interpolation root finder
//!    that returns the accurate differences `d_j - λ` (`DLAED4`-style),
//! 3. recomputing `z` from the Löwner formula so the eigenvectors of the
//!    rank-one problem are numerically orthogonal (`DLAED3`), and
//! 4. updating the eigenvectors with two GEMMs that only touch the nonzero
//!    blocks of `diag(Q₁, Q₂)` (`DLAED3`).
//!
//! The deflation and sort permutations are applied as column gathers into a
//! workspace allocated once per call; within the recursion the eigenpairs are
//! kept unsorted together with a sorting permutation (`INDXQ`), and only the
//! final result is put in ascending order.

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::sym_kernels::{MatMul, argsort, nrm2, permute_columns, pow2_below};
use super::sym_steqr::{Deflation, RotTarget, RotationLog, steqr};

/// Maximum number of iterations for a single secular-equation root.
const MAX_SECULAR_ITER: usize = 100;
/// Iteration limit (per row) of the QR solver used on the leaves.
const LEAF_MAX_ITER: usize = 100;

/// Failure of the divide-and-conquer solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DcError {
    /// The QR iteration on a leaf (or the input) did not converge.
    NotConverged,
    /// The secular equation solver did not converge within its iteration
    /// limit, or produced a non-finite root.
    SecularEquationFailed,
}

/// Workspace of one divide-and-conquer run, sized for the largest merge.
struct Workspace<T: Field> {
    /// Gathered (deflated and grouped) columns of `diag(Q₁, Q₂)`.
    q2: Mat<T>,
    /// Secular differences `d_i - λ_j`, then the rank-one eigenvectors.
    s: Mat<T>,
    /// Coupling vector `z`.
    z: Vec<T>,
    /// Poles of the secular equation (ascending).
    dlamda: Vec<T>,
    /// `ρ · z_i²` of the poles.
    zeta: Vec<T>,
    /// Secular roots.
    lambda: Vec<T>,
    /// Löwner products / recomputed `z`.
    zhat: Vec<T>,
    /// One eigenvector of the rank-one problem.
    u: Vec<T>,
    /// Deflated eigenvalues (ascending).
    dvals: Vec<T>,
    /// Columns in ascending order of `d`.
    indx: Vec<usize>,
    /// Non-deflated columns in ascending order of their pole.
    active: Vec<usize>,
    /// Deflated columns.
    deflated: Vec<usize>,
    /// Column types: 1 = top half only, 2 = both halves, 3 = bottom half
    /// only, 4 = deflated.
    coltyp: Vec<u8>,
    /// Column of `Q` gathered into packed position `p`.
    packed_col: Vec<usize>,
    /// Secular index of packed position `p`.
    packed_to_sorted: Vec<usize>,
    /// Scratch permutation.
    scratch: Vec<usize>,
    /// Rotation log of the leaf solver.
    log: RotationLog<T>,
    /// Parallelism of the eigenvector-update GEMMs.
    par: Par,
}

impl<T: Field + Real + bytemuck::Zeroable> Workspace<T> {
    fn new(n: usize, par: Par) -> Self {
        Self {
            q2: Mat::zeros(n, n),
            s: Mat::zeros(n, n),
            z: vec![T::zero(); n],
            dlamda: vec![T::zero(); n],
            zeta: vec![T::zero(); n],
            lambda: vec![T::zero(); n],
            zhat: vec![T::zero(); n],
            u: vec![T::zero(); n],
            dvals: Vec::with_capacity(n),
            indx: Vec::with_capacity(n),
            active: Vec::with_capacity(n),
            deflated: Vec::with_capacity(n),
            coltyp: vec![0; n],
            packed_col: vec![0; n],
            packed_to_sorted: vec![0; n],
            scratch: Vec::with_capacity(n),
            log: RotationLog::new(),
            par,
        }
    }
}

/// Eigendecomposition of the symmetric tridiagonal `tridiag(e, d, e)` by
/// divide and conquer.
///
/// `z` must be `n × n` and zero on entry. On return `d` holds the
/// eigenvalues in ascending order and the columns of `z` the corresponding
/// orthonormal eigenvectors; `e` is destroyed. Blocks of at most `smlsiz`
/// rows (at least 1) are solved by the QR iteration. `par` is passed to the
/// eigenvector-update GEMMs of the merges (bitwise the same result for every
/// `Par`).
pub(crate) fn stedc<T, G>(
    d: &mut [T],
    e: &mut [T],
    z: &mut Mat<T>,
    smlsiz: usize,
    par: Par,
) -> Result<(), DcError>
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
{
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    let smlsiz = smlsiz.max(1);
    if d.iter().chain(e.iter()).any(|v| !v.is_finite()) {
        return Err(DcError::NotConverged);
    }

    // Split at negligible off-diagonals (DSTEDC; the test is invariant
    // under scaling).
    let eps = <T as Scalar>::epsilon();
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start;
        while end + 1 < n {
            let tiny = eps * Real::sqrt(Scalar::abs(d[end])) * Real::sqrt(Scalar::abs(d[end + 1]));
            if Scalar::abs(e[end]) <= tiny {
                e[end] = T::zero();
                break;
            }
            end += 1;
        }
        blocks.push((start, end + 1));
        start = end + 1;
    }
    let largest_dc = blocks
        .iter()
        .map(|&(lo, hi)| hi - lo)
        .filter(|&size| size > smlsiz)
        .max()
        .unwrap_or(0);
    let mut ws = Workspace::<T>::new(largest_dc, par);
    let mut indxq = vec![0usize; n];

    for &(lo, hi) in &blocks {
        let size = hi - lo;
        if size == 1 {
            z[(lo, lo)] = T::one();
            continue;
        }
        if size <= smlsiz {
            // Small blocks go straight to the QR iteration (scale-invariant).
            solve_block::<T, G>(d, e, z, &mut indxq, lo, hi, smlsiz, &mut ws)?;
            continue;
        }
        // Scale the block by a power of two (exact both ways) so that its
        // max-norm lies in [1, 2) (unless it is zero), solve it, scale back.
        let blk_nrm = d[lo..hi]
            .iter()
            .chain(e[lo..hi - 1].iter())
            .fold(T::zero(), |acc, &v| acc.max(Scalar::abs(v)));
        let scale = pow2_below(blk_nrm);
        for v in d[lo..hi].iter_mut().chain(e[lo..hi - 1].iter_mut()) {
            *v /= scale;
        }
        solve_block::<T, G>(d, e, z, &mut indxq, lo, hi, smlsiz, &mut ws)?;
        for v in d[lo..hi].iter_mut() {
            *v *= scale;
        }
    }

    // Final ascending order: one column gather.
    let mut perm = Vec::with_capacity(n);
    argsort(d, &mut perm);
    let sorted: Vec<T> = perm.iter().map(|&p| d[p]).collect();
    d.copy_from_slice(&sorted);
    let ld = z.row_stride();
    let mut saved = Vec::with_capacity(n);
    let mut visited = Vec::with_capacity(n);
    permute_columns(z.raw_data_mut(), 0, ld, n, &perm, &mut saved, &mut visited);
    Ok(())
}

/// Divide and conquer on the unreduced block `[lo, hi)`; leaves the block's
/// eigenvalues in `d[lo..hi]` (in the order of the columns of
/// `z[lo..hi, lo..hi]`) and a sorting permutation in `indxq[lo..hi]`.
#[allow(clippy::too_many_arguments)]
fn solve_block<T, G>(
    d: &mut [T],
    e: &mut [T],
    z: &mut Mat<T>,
    indxq: &mut [usize],
    lo: usize,
    hi: usize,
    smlsiz: usize,
    ws: &mut Workspace<T>,
) -> Result<(), DcError>
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
{
    let size = hi - lo;
    if size <= smlsiz {
        for i in lo..hi {
            z[(i, i)] = T::one();
        }
        if size > 1 {
            let ld = z.row_stride();
            let target = RotTarget {
                data: z.raw_data_mut(),
                off: lo + lo * ld,
                ld,
                nrows: size,
                identity: true,
            };
            steqr(
                &mut d[lo..hi],
                &mut e[lo..hi - 1],
                Some(target),
                LEAF_MAX_ITER,
                Deflation::Strict,
                &mut ws.log,
            )
            .map_err(|_| DcError::NotConverged)?;
        }
        argsort(&d[lo..hi], &mut ws.scratch);
        indxq[lo..hi].copy_from_slice(&ws.scratch);
        return Ok(());
    }

    // Rank-one tear: T = diag(T₁, T₂) + |β|·v·vᵀ, v = e_{m-1} + sign(β)·e_m.
    let mid = lo + size / 2;
    let beta = e[mid - 1];
    let abs_beta = Scalar::abs(beta);
    d[mid - 1] -= abs_beta;
    d[mid] -= abs_beta;

    solve_block::<T, G>(d, e, z, indxq, lo, mid, smlsiz, ws)?;
    solve_block::<T, G>(d, e, z, indxq, mid, hi, smlsiz, ws)?;
    merge::<T, G>(d, z, indxq, lo, mid, hi, beta, ws)
}

/// Merges the solved halves `[lo, mid)` and `[mid, hi)` coupled by the
/// off-diagonal `beta` (LAPACK `DLAED1`/`DLAED2`/`DLAED3`).
#[allow(clippy::too_many_arguments)]
fn merge<T, G>(
    d_all: &mut [T],
    zmat: &mut Mat<T>,
    indxq_all: &mut [usize],
    lo: usize,
    mid: usize,
    hi: usize,
    beta: T,
    ws: &mut Workspace<T>,
) -> Result<(), DcError>
where
    T: Field + Real + bytemuck::Zeroable,
    G: MatMul<T>,
{
    let n = hi - lo;
    let n1 = mid - lo;
    let n2 = n - n1;
    let d = &mut d_all[lo..hi];
    let indxq = &mut indxq_all[lo..hi];
    let ldz = zmat.row_stride();
    let qoff = lo + lo * ldz;

    // Coupling vector: last row of Q₁ and (sign(β)·) first row of Q₂.
    {
        let zd = zmat.raw_data();
        for j in 0..n1 {
            ws.z[j] = zd[qoff + j * ldz + n1 - 1];
        }
        let sign = if beta < T::zero() {
            -T::one()
        } else {
            T::one()
        };
        for j in 0..n2 {
            ws.z[n1 + j] = sign * zd[qoff + (n1 + j) * ldz + n1];
        }
    }
    let z = &mut ws.z[..n];

    // Ascending order of all poles from the halves' sorting permutations.
    merge_sorted(d, &indxq[..n1], 0, &indxq[n1..], n1, &mut ws.indx);

    let mut rho = Scalar::abs(beta);
    let znorm = nrm2(z);
    if !(rho > T::zero()) || !(znorm > T::zero()) {
        // No coupling: the merged problem is already solved.
        indxq.copy_from_slice(&ws.indx);
        return Ok(());
    }
    for zi in z.iter_mut() {
        *zi /= znorm;
    }
    rho = rho * znorm * znorm;

    // Deflation tolerance (DLAED2).
    let eps = <T as Scalar>::epsilon();
    let dmax = d.iter().fold(T::zero(), |acc, &v| acc.max(Scalar::abs(v)));
    let zmax = z.iter().fold(T::zero(), |acc, &v| acc.max(Scalar::abs(v)));
    let eight = T::from_f64(8.0).unwrap_or(T::one());
    let tol = eight * eps * dmax.max(zmax);
    if rho * zmax <= tol {
        // Every component deflates: only the order changes.
        indxq.copy_from_slice(&ws.indx);
        return Ok(());
    }

    // Deflation pass over the poles in ascending order.
    let coltyp = &mut ws.coltyp[..n];
    for (j, t) in coltyp.iter_mut().enumerate() {
        *t = if j < n1 { 1 } else { 3 };
    }
    ws.active.clear();
    ws.deflated.clear();
    let mut pj: Option<usize> = None;
    for jj in 0..n {
        let nj = ws.indx[jj];
        if rho * Scalar::abs(z[nj]) <= tol {
            // Type-1 deflation: negligible z component.
            coltyp[nj] = 4;
            ws.deflated.push(nj);
            continue;
        }
        let Some(p) = pj else {
            pj = Some(nj);
            continue;
        };
        let tau = Real::hypot(z[nj], z[p]);
        let c = z[nj] / tau;
        let s = -z[p] / tau;
        let t = d[nj] - d[p];
        if Scalar::abs(t * c * s) <= tol {
            // Type-2 deflation: (near-)equal poles. Rotate columns p and nj
            // of Q so that z[p] vanishes; column p deflates.
            z[nj] = tau;
            z[p] = T::zero();
            if coltyp[nj] != coltyp[p] {
                coltyp[nj] = 2;
            }
            coltyp[p] = 4;
            rotate_columns(zmat.raw_data_mut(), qoff, ldz, n, p, nj, c, s);
            let dp = d[p] * c * c + d[nj] * s * s;
            d[nj] = d[p] * s * s + d[nj] * c * c;
            d[p] = dp;
            ws.deflated.push(p);
        } else {
            ws.active.push(p);
        }
        pj = Some(nj);
    }
    if let Some(p) = pj {
        ws.active.push(p);
    }
    // Rotations move poles only between their neighbours; keep the active
    // poles strictly ordered regardless.
    {
        let dref: &[T] = d;
        ws.active.sort_by(|&x, &y| total_order(dref[x], dref[y]));
        ws.deflated.sort_by(|&x, &y| total_order(dref[x], dref[y]));
    }
    let k = ws.active.len();

    // Group the active columns by type (1, 2, 3), keeping pole order within
    // each group, so the eigenvector update can skip the zero blocks.
    let mut ctot = [0usize; 4];
    for &col in ws.active.iter() {
        ctot[usize::from(coltyp[col]) - 1] += 1;
    }
    let mut psm = [0usize, ctot[0], ctot[0] + ctot[1]];
    for (si, &col) in ws.active.iter().enumerate() {
        let g = usize::from(coltyp[col]) - 1;
        let p = psm[g];
        psm[g] += 1;
        ws.packed_col[p] = col;
        ws.packed_to_sorted[p] = si;
    }
    let ctot1 = ctot[0];
    let n12 = ctot[0] + ctot[1];
    let n23 = k - ctot1;

    // Gather: packed active columns (only their nonzero halves), then the
    // deflated columns in ascending order.
    {
        let ldq2 = ws.q2.row_stride();
        let zd = zmat.raw_data();
        let q2d = ws.q2.raw_data_mut();
        for p in 0..k {
            let col = ws.packed_col[p];
            let (r0, r1) = match coltyp[col] {
                1 => (0, n1),
                3 => (n1, n),
                _ => (0, n),
            };
            let src = qoff + col * ldz;
            q2d[p * ldq2 + r0..p * ldq2 + r1].copy_from_slice(&zd[src + r0..src + r1]);
        }
        ws.dvals.clear();
        for (t, &col) in ws.deflated.iter().enumerate() {
            let src = qoff + col * ldz;
            let dst = (k + t) * ldq2;
            q2d[dst..dst + n].copy_from_slice(&zd[src..src + n]);
            ws.dvals.push(d[col]);
        }
    }

    // Secular equation on the active poles.
    for (si, &col) in ws.active.iter().enumerate() {
        ws.dlamda[si] = d[col];
        ws.zeta[si] = rho * z[col] * z[col];
        ws.zhat[si] = z[col];
    }
    secular_eigenvectors(ws, k, rho)?;

    // Eigenvector update: Q(0..n1, 0..k) = Q2(0..n1, 0..n12)·S(0..n12, :),
    // Q(n1..n, 0..k) = Q2(n1..n, ctot1..k)·S(ctot1..k, :).
    if n12 > 0 {
        G::gemm(
            false,
            false,
            T::one(),
            ws.q2.submatrix(0, 0, n1, n12),
            ws.s.submatrix(0, 0, n12, k),
            T::zero(),
            zmat.submatrix_mut(lo, lo, n1, k),
            ws.par,
        );
    } else {
        fill_zero(zmat, lo, lo, n1, k);
    }
    if n23 > 0 {
        G::gemm(
            false,
            false,
            T::one(),
            ws.q2.submatrix(n1, ctot1, n2, n23),
            ws.s.submatrix(ctot1, 0, n23, k),
            T::zero(),
            zmat.submatrix_mut(lo + n1, lo, n2, k),
            ws.par,
        );
    } else {
        fill_zero(zmat, lo + n1, lo, n2, k);
    }
    // Deflated columns go behind the k new ones.
    {
        let ldq2 = ws.q2.row_stride();
        let q2d = ws.q2.raw_data();
        let zd = zmat.raw_data_mut();
        for t in 0..(n - k) {
            let src = (k + t) * ldq2;
            let dst = qoff + (k + t) * ldz;
            zd[dst..dst + n].copy_from_slice(&q2d[src..src + n]);
        }
    }
    d[..k].copy_from_slice(&ws.lambda[..k]);
    d[k..].copy_from_slice(&ws.dvals);

    // Sorting permutation: merge the two ascending runs.
    let (dl, dr) = (&d[..k], &d[k..]);
    ws.scratch.clear();
    let (mut a, mut b) = (0, 0);
    while a < k || b < dr.len() {
        let take_left = b >= dr.len() || (a < k && dl[a] <= dr[b]);
        if take_left {
            ws.scratch.push(a);
            a += 1;
        } else {
            ws.scratch.push(k + b);
            b += 1;
        }
    }
    indxq.copy_from_slice(&ws.scratch);
    Ok(())
}

/// The total order of `f64::total_cmp` for any [`Real`] type, up to the
/// payloads of NaNs: negative NaN < -Inf < ... < -0 < +0 < ... < +Inf <
/// positive NaN. (The poles are finite here; a total order keeps the sorts
/// well defined regardless.)
fn total_order<T: Real>(a: T, b: T) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    let nan_rank = |x: T| -> i8 {
        if !x.is_nan() {
            0
        } else if x.is_sign_negative() {
            -1
        } else {
            1
        }
    };
    match nan_rank(a).cmp(&nan_rank(b)) {
        Ordering::Equal => match a.partial_cmp(&b) {
            // Equal numbers differ at most in the sign of zero.
            Some(Ordering::Equal) => b.is_sign_negative().cmp(&a.is_sign_negative()),
            Some(order) => order,
            // Two NaNs of the same sign.
            None => Ordering::Equal,
        },
        order => order,
    }
}

/// Merges the column indices `left` (+`off_l`) and `right` (+`off_r`), each
/// sorted by `d`, into `out` sorted by `d`.
fn merge_sorted<T: Field + Real>(
    d: &[T],
    left: &[usize],
    off_l: usize,
    right: &[usize],
    off_r: usize,
    out: &mut Vec<usize>,
) {
    out.clear();
    let (mut a, mut b) = (0, 0);
    while a < left.len() || b < right.len() {
        let take_left =
            b >= right.len() || (a < left.len() && d[left[a] + off_l] <= d[right[b] + off_r]);
        if take_left {
            out.push(left[a] + off_l);
            a += 1;
        } else {
            out.push(right[b] + off_r);
            b += 1;
        }
    }
}

/// Zeroes the `nrows × ncols` block of `m` at `(r0, c0)`.
fn fill_zero<T: Field + bytemuck::Zeroable>(
    m: &mut Mat<T>,
    r0: usize,
    c0: usize,
    nrows: usize,
    ncols: usize,
) {
    let ld = m.row_stride();
    let data = m.raw_data_mut();
    for j in 0..ncols {
        let base = r0 + (c0 + j) * ld;
        for v in data[base..base + nrows].iter_mut() {
            *v = T::zero();
        }
    }
}

/// Plane rotation of columns `p` and `q` (rows `0..nrows`) of the block at
/// `data[off..]`: `(x_p, x_q) <- (c·x_p + s·x_q, c·x_q - s·x_p)` (`DROT`).
#[allow(clippy::too_many_arguments)]
fn rotate_columns<T: Field + Real>(
    data: &mut [T],
    off: usize,
    ld: usize,
    nrows: usize,
    p: usize,
    q: usize,
    c: T,
    s: T,
) {
    let (first, second, swap) = if p < q { (p, q, false) } else { (q, p, true) };
    let (lo, hi) = data.split_at_mut(off + second * ld);
    let a = &mut lo[off + first * ld..off + first * ld + nrows];
    let b = &mut hi[..nrows];
    let (xp, xq) = if swap { (b, a) } else { (a, b) };
    for (u, v) in xp.iter_mut().zip(xq.iter_mut()) {
        let pu = *u;
        let qv = *v;
        *u = c * pu + s * qv;
        *v = c * qv - s * pu;
    }
}

/// Solves the `k` secular equations with poles `ws.dlamda[..k]` (strictly
/// ascending) and weights `ws.zeta[..k] = ρ·z²` (signs of `z` in
/// `ws.zhat[..k]`), and leaves in `ws.s` the eigenvectors of the rank-one
/// problem with rows in packed order (`ws.packed_to_sorted`) and in
/// `ws.lambda[..k]` the eigenvalues.
fn secular_eigenvectors<T: Field + Real + bytemuck::Zeroable>(
    ws: &mut Workspace<T>,
    k: usize,
    rho: T,
) -> Result<(), DcError> {
    let lds = ws.s.row_stride();
    let dl = &ws.dlamda[..k];
    let zeta = &ws.zeta[..k];
    if k == 1 {
        ws.lambda[0] = dl[0] + zeta[0];
        ws.s[(0, 0)] = T::one();
        return Ok(());
    }
    let sum_zeta = zeta.iter().fold(T::zero(), |acc, &v| acc + v);
    {
        let sd = ws.s.raw_data_mut();
        for j in 0..k {
            let delta = &mut sd[j * lds..j * lds + k];
            ws.lambda[j] = secular_root(dl, zeta, sum_zeta, j, delta)?;
        }
    }

    // Gu-Eisenstat: recompute |z| from the Löwner formula,
    //   ρ·ẑ_a² = (λ_a - d_a) · Π_{l≠a} (λ_l - d_a)/(d_l - d_a),
    // with column l of S holding d_i - λ_l.
    let sd = ws.s.raw_data_mut();
    let prod = &mut ws.u[..k];
    for (a, pa) in prod.iter_mut().enumerate() {
        *pa = -sd[a * lds + a];
    }
    for l in 0..k {
        let col = &sd[l * lds..l * lds + k];
        let dlv = dl[l];
        for (a, (pa, &delta)) in prod.iter_mut().zip(col).enumerate() {
            if a != l {
                *pa *= -delta / (dlv - dl[a]);
            }
        }
    }
    let zhat = &mut ws.zhat[..k];
    for (zh, &pa) in zhat.iter_mut().zip(prod.iter()) {
        let val = pa / rho;
        let mag = if val > T::zero() {
            Real::sqrt(val)
        } else {
            T::zero()
        };
        *zh = if *zh >= T::zero() { mag } else { -mag };
    }

    // Eigenvectors u_j(a) = ẑ_a / (d_a - λ_j), normalised, rows permuted to
    // the packed column order of Q2.
    let u = &mut ws.u[..k];
    for j in 0..k {
        let col = &mut sd[j * lds..j * lds + k];
        let mut nrm_sq = T::zero();
        for ((ua, &zh), &delta) in u.iter_mut().zip(zhat.iter()).zip(col.iter()) {
            let v = zh / delta;
            *ua = v;
            nrm_sq += v * v;
        }
        let nrm = Real::sqrt(nrm_sq);
        let inv = if nrm > T::zero() {
            T::one() / nrm
        } else {
            T::one()
        };
        for (p, slot) in col.iter_mut().enumerate() {
            *slot = u[ws.packed_to_sorted[p]] * inv;
        }
    }
    Ok(())
}

/// Iteration limit of [`secular_root`].
#[cfg(not(test))]
#[inline]
fn secular_iter_limit() -> usize {
    MAX_SECULAR_ITER
}

#[cfg(test)]
thread_local! {
    /// Test hook: the iteration limit of [`secular_root`] on this thread
    /// (lowered to reach the non-convergence error).
    pub(super) static SECULAR_ITER_LIMIT: core::cell::Cell<usize> =
        const { core::cell::Cell::new(MAX_SECULAR_ITER) };
}

/// Iteration limit of [`secular_root`] (the test hook's value).
#[cfg(test)]
fn secular_iter_limit() -> usize {
    SECULAR_ITER_LIMIT.with(core::cell::Cell::get)
}

/// Solves `1 + Σ_j zeta_j / (d_j - λ) = 0` for its `i`-th root (0-indexed)
/// and writes `delta[j] = d_j - λ` (accurate even for a root next to a pole).
///
/// The root lies in `(d_i, d_{i+1})` for `i < k-1` and in
/// `(d_{k-1}, d_{k-1} + sum_zeta)` for the last one; `zeta_j > 0`,
/// `sum_zeta = Σ zeta_j` and the poles are strictly ascending. As in LAPACK
/// `DLAED4`, the origin is moved to the nearer pole `d_o` and the iteration
/// runs on `η = λ - d_o`:
///
/// - the first iterate is the root of a model that keeps the two poles
///   around the root (the last two poles for the last root) exact and
///   freezes the other terms at the middle of the search interval, so it is
///   accurate even for a root next to its pole (a tiny weight);
/// - every step takes the root of the two-pole rational ("middle way")
///   model of the secular function at the current iterate, from the
///   quadratic formula without cancellation; a step of the wrong sign
///   becomes a Newton step and one leaving the sign bracket a bisection;
/// - the iteration stops once `|f| <= ε·(8·(1 + Σ|t_j|) + Σ|s_m| +
///   |η|·f′)`, the bound of `DLAED4` on the rounding error of `f`: the terms
///   `t_j = zeta_j / (d_j - λ)`, the running error of their summation
///   (partial sums `s_m`, summed from the farthest pole inwards) and the
///   effect of rounding `η` itself; or once the bracket has no
///   floating-point number left inside.
///
/// # Errors
///
/// [`DcError::SecularEquationFailed`] if the iteration has not converged
/// after [`MAX_SECULAR_ITER`] steps, or the root is not finite.
fn secular_root<T: Field + Real>(
    d: &[T],
    zeta: &[T],
    sum_zeta: T,
    i: usize,
    delta: &mut [T],
) -> Result<T, DcError> {
    let k = d.len();
    let zero = T::zero();
    let eps = <T as Scalar>::epsilon();
    let two = T::one() + T::one();
    let four = two * two;
    let eight = four * two;
    let delta = &mut delta[..k];

    if k == 1 {
        // 1 + zeta_0 / (d_0 - λ) = 0.
        delta[0] = -zeta[0];
        let lambda = d[0] + zeta[0];
        return if lambda.is_finite() {
            Ok(lambda)
        } else {
            Err(DcError::SecularEquationFailed)
        };
    }

    // The poles of the model: the two around the root, or the last two.
    let last = i + 1 == k;
    let (p1, p2) = if last { (k - 2, k - 1) } else { (i, i + 1) };
    let start = if last {
        start_last(d, zeta, sum_zeta)
    } else {
        start_middle(d, zeta, i)
    };
    let (orig, mut elo, mut ehi) = (start.orig, start.lo, start.hi);
    // η = 0 is the pole at the origin; the other bracket ends are regular.
    let start_ok =
        start.eta.is_finite() && start.eta >= elo && start.eta <= ehi && start.eta != zero;
    let mut eta = if start_ok {
        start.eta
    } else {
        (elo + ehi) / two
    };

    let d_orig = d[orig];
    // delta holds d_j - d_orig until the end.
    for (dj, &dv) in delta.iter_mut().zip(d.iter()) {
        *dj = dv - d_orig;
    }

    let mut converged = false;
    for _ in 0..secular_iter_limit() {
        // psi: the poles up to p1, phi: the poles from p2 on, each summed
        // from the farthest pole inwards, with the running error bound.
        let (mut psi, mut dpsi, mut phi, mut dphi) = (zero, zero, zero, zero);
        let (mut abs_sum, mut erretm) = (zero, zero);
        for (&zj, &dd) in zeta[..=p1].iter().zip(delta[..=p1].iter()) {
            let inv = T::one() / (dd - eta);
            let t = zj * inv;
            psi += t;
            dpsi += t * inv;
            abs_sum += Scalar::abs(t);
            erretm += Scalar::abs(psi);
        }
        for (&zj, &dd) in zeta[p2..].iter().zip(delta[p2..].iter()).rev() {
            let inv = T::one() / (dd - eta);
            let t = zj * inv;
            phi += t;
            dphi += t * inv;
            abs_sum += Scalar::abs(t);
            erretm += Scalar::abs(phi);
        }
        let f = T::one() + psi + phi;
        let fprime = dpsi + dphi;
        if f.is_nan() {
            // Only reachable from non-finite input: report, do not iterate.
            break;
        }

        // f is increasing in eta; keep the sign bracket.
        if f >= zero {
            ehi = eta;
        } else {
            elo = eta;
        }
        let bound = eight * (T::one() + abs_sum) + erretm + Scalar::abs(eta) * fprime;
        if f.is_finite() && Scalar::abs(f) <= eps * bound {
            converged = true;
            break;
        }

        // Two-pole model c0 + b1/(dl - s) + b2/(dr - s) at the poles p1, p2
        // (`b1 = dpsi·dl²`, `b2 = dphi·dr²`, `c0` matching f): its root
        // solves c0·s² - bq·s + cq = 0.
        let dl = delta[p1] - eta;
        let dr = delta[p2] - eta;
        let mut c0 = f - dl * dpsi - dr * dphi;
        let bq = (dl + dr) * f - dl * dr * fprime;
        let cq = dl * dr * f;
        let newton = -f / fprime;
        let mut step = if last {
            // Both poles left of the root: the root of the model to their
            // right. It exists for c0 > 0; otherwise take |c0| (`DLAED4`).
            if c0 < zero {
                c0 = -c0;
            }
            if c0 == zero {
                newton
            } else {
                let sq = Real::sqrt(Scalar::abs(bq * bq - four * c0 * cq));
                if bq >= zero {
                    (bq + sq) / (two * c0)
                } else {
                    two * cq / (bq - sq)
                }
            }
        } else if c0 == zero {
            if bq == zero { newton } else { cq / bq }
        } else {
            // The model's root between its poles, `2cq/(bq + √disc)`.
            let sq = Real::sqrt(Scalar::abs(bq * bq - four * c0 * cq));
            if bq <= zero {
                (bq - sq) / (two * c0)
            } else {
                two * cq / (bq + sq)
            }
        };
        // f < 0 needs a step s > 0 and f > 0 one s < 0; rounding (or the
        // |c0| model) can violate that, a Newton step cannot.
        if !step.is_finite() || step * f > zero {
            step = newton;
        }
        let mut next = eta + step;
        if !(next > elo && next < ehi) {
            next = (elo + ehi) / two;
        }
        if next == eta {
            // No progress: bisect, or stop once no floating-point number is
            // left strictly inside the bracket (the root is resolved).
            let bis = (elo + ehi) / two;
            if bis == elo || bis == ehi {
                converged = true;
                break;
            }
            next = bis;
        }
        eta = next;
    }
    if !converged {
        return Err(DcError::SecularEquationFailed);
    }

    for dj in delta.iter_mut() {
        *dj -= eta;
    }
    let lambda = d_orig + eta;
    if !lambda.is_finite() {
        return Err(DcError::SecularEquationFailed);
    }
    Ok(lambda)
}

/// Origin, sign bracket and first iterate of [`secular_root`].
struct SecularStart<T> {
    /// Index of the origin pole `d_o`.
    orig: usize,
    /// Lower end of the bracket for `η = λ - d_o`.
    lo: T,
    /// Upper end of the bracket for `η`.
    hi: T,
    /// First iterate (validated by the caller).
    eta: T,
}

/// [`SecularStart`] of the root in `(d_i, d_{i+1})` (`DLAED4`, `I < N`).
///
/// The sign of `f` at the middle of the interval picks the nearer pole as
/// the origin. The first iterate is the root of `c + zeta_i/(d_i - λ) +
/// zeta_{i+1}/(d_{i+1} - λ)` with the other terms frozen at the middle in
/// `c`, a quadratic in `η` solved without cancellation.
fn start_middle<T: Field + Real>(d: &[T], zeta: &[T], i: usize) -> SecularStart<T> {
    let two = T::one() + T::one();
    let four = two * two;
    let gap = d[i + 1] - d[i];
    let mid = gap / two;
    let mut c = T::one();
    for (j, (&dj, &zj)) in d.iter().zip(zeta.iter()).enumerate() {
        if j != i && j != i + 1 {
            c += zj / ((dj - d[i]) - mid);
        }
    }
    let fmid = c - zeta[i] / mid + zeta[i + 1] / (gap - mid);
    if fmid >= T::zero() {
        // Root in (d_i, d_i + mid]: origin d_i and the model
        // c·η² - a·η + b = 0, a = c·gap + zeta_i + zeta_{i+1}, b = zeta_i·gap.
        let a = c * gap + zeta[i] + zeta[i + 1];
        let b = zeta[i] * gap;
        let sq = Real::sqrt(Scalar::abs(a * a - four * b * c));
        let eta = if a > T::zero() {
            two * b / (a + sq)
        } else {
            (a - sq) / (two * c)
        };
        SecularStart {
            orig: i,
            lo: T::zero(),
            hi: mid,
            eta,
        }
    } else {
        // Root in (d_{i+1} - mid, d_{i+1}): origin d_{i+1} and the model
        // c·η² + a·η - b = 0, a = c·gap - zeta_i - zeta_{i+1}, b = zeta_{i+1}·gap.
        let a = c * gap - zeta[i] - zeta[i + 1];
        let b = zeta[i + 1] * gap;
        let sq = Real::sqrt(Scalar::abs(a * a + four * b * c));
        let eta = if a < T::zero() {
            two * b / (a - sq)
        } else {
            -(a + sq) / (two * c)
        };
        SecularStart {
            orig: i + 1,
            lo: -(gap - mid),
            hi: T::zero(),
            eta,
        }
    }
}

/// [`SecularStart`] of the last root, in `(d_{k-1}, d_{k-1} + sum_zeta)`
/// (`DLAED4`, `I = N`): origin `d_{k-1}`; the sign of `f` at
/// `η = sum_zeta / 2` halves the bracket, and the first iterate is the root
/// of the model keeping the last two poles exact (or the upper end of the
/// bracket when that model puts the root beyond it).
fn start_last<T: Field + Real>(d: &[T], zeta: &[T], sum_zeta: T) -> SecularStart<T> {
    let k = d.len();
    let o = k - 1;
    let two = T::one() + T::one();
    let four = two * two;
    let mid = sum_zeta / two;
    let mut c = T::one();
    for (&dj, &zj) in d[..k - 2].iter().zip(zeta[..k - 2].iter()) {
        c += zj / ((dj - d[o]) - mid);
    }
    let del = d[o] - d[k - 2];
    let fmid = c + zeta[k - 2] / (-del - mid) - zeta[o] / mid;
    // Model c - zeta_{k-2}/(del + η) - zeta_{k-1}/η = 0, that is
    // c·η² - a·η - b = 0, a = -c·del + zeta_{k-2} + zeta_{k-1}, b = zeta_{k-1}·del.
    let a = -c * del + zeta[k - 2] + zeta[o];
    let b = zeta[o] * del;
    let model_root = || {
        let sq = Real::sqrt(Scalar::abs(a * a + four * b * c));
        if a < T::zero() {
            two * b / (sq - a)
        } else {
            (a + sq) / (two * c)
        }
    };
    if fmid <= T::zero() {
        // Root in [mid, sum_zeta).
        let at_end = zeta[k - 2] / (del + sum_zeta) + zeta[o] / sum_zeta;
        let eta = if c <= at_end { sum_zeta } else { model_root() };
        SecularStart {
            orig: o,
            lo: mid,
            hi: sum_zeta,
            eta,
        }
    } else {
        SecularStart {
            orig: o,
            lo: T::zero(),
            hi: mid,
            eta: model_root(),
        }
    }
}

#[cfg(test)]
#[path = "dc_merge_tests.rs"]
mod tests;
