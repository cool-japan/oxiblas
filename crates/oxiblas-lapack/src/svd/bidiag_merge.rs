//! Merge step of the bidiagonal divide-and-conquer SVD (LAPACK `dlasd1`,
//! `dlasd2` and `dlasd3`).
//!
//! A node of the divide-and-conquer tree is the `n × m` upper bidiagonal
//! (`n = nl + nr + 1`, `m = n + sqre`) split at row `nl`:
//!
//! ```text
//!     ⎡ B₁          0   ⎤        B₁: nl × (nl + 1),  B₂: nr × (nr + sqre)
//! B = ⎢ α·e_nlᵀ  β·e₁ᵀ  ⎥        α = d[nl], β = e[nl]
//!     ⎣ 0           B₂  ⎦
//! ```
//!
//! With the SVDs of the children in the diagonal blocks of `U` and `V`, `B`
//! becomes `diag(U₁, 1, U₂)·M·diag(V₁, V₂)ᵀ` where `M` has the children's
//! singular values on its diagonal and the coupling row
//! `zᵀ = (α·last row of V₁, β·first row of V₂)` in row `nl`. [`merge`]:
//!
//! 1. scales the node to unit max-norm (by a power of two, so exactly);
//! 2. **deflates** (`dlasd2`): sorts the diagonal, drops components with
//!    `|zⱼ| ≤ 8ε·max(|α|, |β|, |d|)` and rotates pairs of singular values
//!    closer than that tolerance into one (Givens rotations on the columns of
//!    `U` and `V`); the surviving columns are grouped by their row structure
//!    (upper block only, lower block only, dense) so that the products below
//!    skip the zero blocks;
//! 3. solves the secular equation for the `K` remaining singular values
//!    ([`crate::svd::secular::lasd4`]) and recomputes `z` from the computed
//!    roots (Gu–Eisenstat), so that the singular vectors of `M`,
//!    `vᵢ ∝ (ẑⱼ/(dⱼ² − σᵢ²))ⱼ` and `uᵢ ∝ (−1, dⱼẑⱼ/(dⱼ² − σᵢ²))`, are
//!    numerically orthogonal without any re-orthogonalisation (`dlasd3`);
//! 4. back-transforms them with GEMMs over the diagonal blocks of `U` and `V`
//!    (two per factor), through the [`DenseOps`] backend.

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::Mat;

use super::bidiag_dc::BidiagDcError;
use super::dc_kernels::{
    DenseOps, cmp_asc_nan_last, fabs, lapack_eps, mul_pow2, nrm2, pow2_exponent, real_const,
    rot_cols, scal_div, scale_pow2, sign_of,
};
use super::secular::lasd4;

/// Column type: non-zero only in the upper block (rows `0..nl`).
const UPPER: u8 = 1;
/// Column type: non-zero only in the lower block (rows `nl+1..`).
const LOWER: u8 = 2;
/// Column type: dense (a deflation rotation mixed an upper and a lower one).
const DENSE: u8 = 3;
/// Column type: deflated.
const DEFLATED: u8 = 4;

/// A node of the divide-and-conquer tree.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    /// Rows of the upper child.
    pub nl: usize,
    /// Rows of the lower child.
    pub nr: usize,
    /// `1` if the node has one more column than rows.
    pub sqre: usize,
    /// Diagonal offset of the node's blocks in the global `U` and `V`.
    pub off: usize,
}

/// Output of the deflation step.
struct Deflation<R: Field> {
    /// Order of the secular equation (including the special first entry).
    k: usize,
    /// Secular weights (`z[0..k]`).
    z: Vec<R>,
    /// Poles (`dsigma[0] = 0`, then the non-deflated values ascending).
    dsigma: Vec<R>,
    /// Columns of `U` in grouped order (`n × n`; column 0 is `e_nl`).
    u2: Mat<R>,
    /// Columns of `V` in grouped order (`m × n`; column 0 couples `z[0]`).
    v2: Mat<R>,
    /// Grouped position → position in `dsigma` order.
    idxc: Vec<usize>,
    /// Number of columns of each type.
    ctot: [usize; 4],
}

/// Merges the solved children of `node` (LAPACK `dlasd1`).
///
/// On entry `d[0..nl]` and `d[nl+1..n]` hold the children's singular values,
/// `idxq[0..nl]` / `idxq[nl+1..n]` the permutations (relative to each child)
/// that sort them ascending, and the children's singular vectors fill the
/// diagonal blocks of the node's blocks of `u` (`n × n`) and `v` (`m × m`,
/// by columns) at offset `node.off`, which are zero elsewhere. On exit `d`
/// holds the node's singular values, `idxq` sorts them ascending and the
/// node's blocks of `u` and `v` hold its singular vectors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn merge<R, B>(
    ops: &B,
    node: Node,
    d: &mut [R],
    alpha: R,
    beta: R,
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    idxq: &mut [usize],
) -> Result<(), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
    B: DenseOps<R>,
{
    let n = node.nl + node.nr + 1;
    // Scale to unit max-norm by a power of two (exactly reversible).
    d[node.nl] = R::zero();
    let mut orgnrm = fabs(alpha).max(fabs(beta));
    for &x in d.iter() {
        orgnrm = orgnrm.max(fabs(x));
    }
    let ks = pow2_exponent(orgnrm);
    scale_pow2(d, ks);
    let alpha = mul_pow2(alpha, ks);
    let beta = mul_pow2(beta, ks);

    let defl = deflate(node, d, alpha, beta, u, v, idxq);
    secular_update(ops, node, d, defl, u, v)?;

    scale_pow2(d, -ks);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| cmp_asc_nan_last(d[a], d[b]));
    idxq.copy_from_slice(&order);
    Ok(())
}

/// Deflation step (LAPACK `dlasd2`) on the scaled node.
#[allow(clippy::too_many_arguments)]
fn deflate<R: Field + Real + bytemuck::Zeroable>(
    node: Node,
    d: &mut [R],
    alpha: R,
    beta: R,
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    idxq: &mut [usize],
) -> Deflation<R> {
    let Node { nl, nr, sqre, off } = node;
    let n = nl + nr + 1;
    let m = n + sqre;
    let zero = R::zero();
    let one = R::one();
    let ldu = u.row_stride();
    let ldv = v.row_stride();

    // Coupling row z (in the children's bases); the left part of d (and of
    // its permutation) moves one position back to make room for z[0].
    let mut z = vec![zero; m];
    let z1;
    {
        let vd = v.raw_data();
        let vat = |i: usize, j: usize| vd[(off + j) * ldv + off + i];
        z1 = alpha * vat(nl, nl);
        z[0] = z1;
        for i in (1..=nl).rev() {
            z[i] = alpha * vat(nl, i - 1);
            d[i] = d[i - 1];
            idxq[i] = idxq[i - 1] + 1;
        }
        for (i, zi) in z.iter_mut().enumerate().skip(nl + 1) {
            *zi = beta * vat(nl + 1, i);
        }
    }
    let mut coltyp = vec![0u8; n];
    for (i, ct) in coltyp.iter_mut().enumerate().skip(1) {
        *ct = if i <= nl { UPPER } else { LOWER };
    }
    for q in idxq.iter_mut().skip(nl + 1) {
        *q += nl + 1;
    }

    // Merge the two ascending halves (`dlamrg`, ties from the upper half).
    let mut dtmp = vec![zero; n];
    let mut ztmp = vec![zero; n];
    let mut cttmp = vec![0u8; n];
    for i in 1..n {
        dtmp[i] = d[idxq[i]];
        ztmp[i] = z[idxq[i]];
        cttmp[i] = coltyp[idxq[i]];
    }
    let mut idx = vec![0usize; n];
    {
        let (mut a, mut b) = (1usize, nl + 1);
        for slot in idx.iter_mut().skip(1) {
            let take_a = if a > nl {
                false
            } else if b >= n {
                true
            } else {
                dtmp[a] <= dtmp[b]
            };
            if take_a {
                *slot = a;
                a += 1;
            } else {
                *slot = b;
                b += 1;
            }
        }
    }
    for i in 1..n {
        d[i] = dtmp[idx[i]];
        z[i] = ztmp[idx[i]];
        coltyp[i] = cttmp[idx[i]];
    }
    // Column of the node's blocks holding sorted position `i`.
    let col_of = |i: usize| -> usize {
        let jd = idxq[idx[i]];
        if jd <= nl { jd - 1 } else { jd }
    };

    let eps = lapack_eps::<R>();
    let tol = real_const::<R>(8.0) * eps * fabs(d[n - 1]).max(fabs(alpha).max(fabs(beta)));

    // Deflate small z components and (nearly) equal singular values.
    let mut k = 1usize;
    let mut k2 = n;
    let mut idxp = vec![0usize; n];
    let mut zsave = vec![zero; n];
    let mut dsigma = vec![zero; n];
    let mut j = 1usize;
    let mut jprev = None;
    while j < n {
        if fabs(z[j]) <= tol {
            k2 -= 1;
            idxp[k2] = j;
            coltyp[j] = DEFLATED;
            j += 1;
        } else {
            jprev = Some(j);
            j += 1;
            break;
        }
    }
    if let Some(mut jp) = jprev {
        while j < n {
            if fabs(z[j]) <= tol {
                k2 -= 1;
                idxp[k2] = j;
                coltyp[j] = DEFLATED;
            } else if fabs(d[j] - d[jp]) <= tol {
                // Rotate jp into j so that z[jp] vanishes.
                let tau = <R as Real>::hypot(z[j], z[jp]);
                let c = z[j] / tau;
                let s = -z[jp] / tau;
                z[j] = tau;
                z[jp] = zero;
                let (cjp, cj) = (col_of(jp), col_of(j));
                rot_cols(u, off + cjp, off + cj, off, n, c, s);
                rot_cols(v, off + cjp, off + cj, off, m, c, s);
                if coltyp[j] != coltyp[jp] {
                    coltyp[j] = DENSE;
                }
                coltyp[jp] = DEFLATED;
                k2 -= 1;
                idxp[k2] = jp;
                jp = j;
            } else {
                zsave[k] = z[jp];
                dsigma[k] = d[jp];
                idxp[k] = jp;
                k += 1;
                jp = j;
            }
            j += 1;
        }
        zsave[k] = z[jp];
        dsigma[k] = d[jp];
        idxp[k] = jp;
        k += 1;
    }

    // Group the columns by type: [special | upper | lower | dense | deflated].
    let mut ctot = [0usize; 4];
    for &ct in coltyp.iter().skip(1) {
        ctot[usize::from(ct - 1)] += 1;
    }
    let mut psm = [
        1,
        1 + ctot[0],
        1 + ctot[0] + ctot[1],
        1 + ctot[0] + ctot[1] + ctot[2],
    ];
    let mut idxc = vec![0usize; n];
    for jj in 1..n {
        let ct = usize::from(coltyp[idxp[jj]] - 1);
        idxc[psm[ct]] = jj;
        psm[ct] += 1;
    }
    let mut u2 = Mat::<R>::zeros(n, n);
    let mut v2 = Mat::<R>::zeros(m, n);
    {
        let ldu2 = u2.row_stride();
        let ldv2 = v2.row_stride();
        let ud = u.raw_data();
        let vd = v.raw_data();
        let u2d = u2.raw_data_mut();
        let v2d = v2.raw_data_mut();
        for jj in 1..n {
            dsigma[jj] = d[idxp[jj]];
            let src = off + col_of(idxp[idxc[jj]]);
            u2d[jj * ldu2..jj * ldu2 + n]
                .copy_from_slice(&ud[src * ldu + off..src * ldu + off + n]);
            v2d[jj * ldv2..jj * ldv2 + m]
                .copy_from_slice(&vd[src * ldv + off..src * ldv + off + m]);
        }
    }

    // The special first entry.
    dsigma[0] = zero;
    let hlftol = tol / real_const::<R>(2.0);
    if k > 1 && fabs(dsigma[1]) <= hlftol {
        dsigma[1] = hlftol;
    }
    let (mut c, mut s) = (one, zero);
    if sqre == 1 {
        let z0 = <R as Real>::hypot(z1, z[m - 1]);
        if z0 <= tol {
            z[0] = tol;
        } else {
            c = z1 / z0;
            s = z[m - 1] / z0;
            z[0] = z0;
        }
    } else {
        z[0] = if fabs(z1) <= tol { tol } else { z1 };
    }
    z[1..k].copy_from_slice(&zsave[1..k]);
    z.truncate(k);

    // U2[:, 0] = e_nl; V2[:, 0] couples z[0]; for sqre = 1 the rotation of
    // the two null vectors also gives the node's null vector V[:, m − 1].
    u2[(nl, 0)] = one;
    {
        let vd = v.raw_data_mut();
        let cnl = (off + nl) * ldv + off;
        if sqre == 1 {
            let cm = (off + m - 1) * ldv + off;
            for i in 0..=nl {
                let vnl = vd[cnl + i];
                vd[cm + i] = -s * vnl;
                v2[(i, 0)] = c * vnl;
            }
            for i in nl + 1..m {
                let vm = vd[cm + i];
                v2[(i, 0)] = s * vm;
                vd[cm + i] = c * vm;
            }
        } else {
            for i in 0..m {
                v2[(i, 0)] = vd[cnl + i];
            }
        }
    }

    // Deflated singular values and vectors go to the back of d, U and V.
    if n > k {
        d[k..n].copy_from_slice(&dsigma[k..n]);
        let ldu2 = u2.row_stride();
        let ldv2 = v2.row_stride();
        let ud = u.raw_data_mut();
        for jj in k..n {
            let dst = (off + jj) * ldu + off;
            ud[dst..dst + n].copy_from_slice(&u2.raw_data()[jj * ldu2..jj * ldu2 + n]);
        }
        let vd = v.raw_data_mut();
        for jj in k..n {
            let dst = (off + jj) * ldv + off;
            vd[dst..dst + m].copy_from_slice(&v2.raw_data()[jj * ldv2..jj * ldv2 + m]);
        }
    }
    Deflation {
        k,
        z,
        dsigma,
        u2,
        v2,
        idxc,
        ctot,
    }
}

/// Secular equation and back-transformation (LAPACK `dlasd3`).
fn secular_update<R, B>(
    ops: &B,
    node: Node,
    d: &mut [R],
    defl: Deflation<R>,
    u: &mut Mat<R>,
    v: &mut Mat<R>,
) -> Result<(), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
    B: DenseOps<R>,
{
    let Node { nl, nr, sqre, off } = node;
    let n = nl + nr + 1;
    let m = n + sqre;
    let zero = R::zero();
    let one = R::one();
    let Deflation {
        k,
        mut z,
        dsigma,
        u2,
        mut v2,
        idxc,
        ctot,
    } = defl;
    let ldu = u.row_stride();
    let ldv = v.row_stride();

    if k == 1 {
        // Everything deflated: σ₀ = |z₀| with the special vectors.
        d[0] = fabs(z[0]);
        let vd = v.raw_data_mut();
        vd[off * ldv + off..off * ldv + off + m].copy_from_slice(&v2.raw_data()[..m]);
        let flip = !(z[0] > zero);
        let ud = u.raw_data_mut();
        let dst = &mut ud[off * ldu + off..off * ldu + off + n];
        for (x, &y) in dst.iter_mut().zip(&u2.raw_data()[..n]) {
            *x = if flip { -y } else { y };
        }
        return Ok(());
    }

    // Normalise z (ρ absorbs its norm) and solve for the K roots.
    let qsave = z.clone();
    let rho = nrm2(&z);
    if !(rho > zero && rho.is_finite()) {
        return Err(BidiagDcError::SecularEquationFailed);
    }
    scal_div(rho, &mut z);
    let rho = rho * rho;
    let dsig = &dsigma[..k];
    let mut dlt = Mat::<R>::zeros(k, k);
    let mut wrk = Mat::<R>::zeros(k, k);
    let lds = dlt.row_stride();
    let ldw = wrk.row_stride();
    {
        let dd = dlt.raw_data_mut();
        let wd = wrk.raw_data_mut();
        for j in 0..k {
            d[j] = lasd4(
                j,
                dsig,
                &z,
                rho,
                &mut dd[j * lds..j * lds + k],
                &mut wd[j * ldw..j * ldw + k],
            )
            .ok_or(BidiagDcError::SecularEquationFailed)?;
        }
    }

    // Gu–Eisenstat: the weights for which the computed roots are exact,
    // ẑᵢ² = ∏ⱼ (dᵢ² − σⱼ²) / ∏_{j≠i} (dᵢ² − dⱼ²), accumulated in the order
    // of `dlasd3` (factor k − 1 first, then j = 0, 1, …, k − 2).
    {
        let dd = dlt.raw_data();
        let wd = wrk.raw_data();
        let last = k - 1;
        let mut zprod: Vec<R> = (0..k)
            .map(|i| dd[last * lds + i] * wd[last * ldw + i])
            .collect();
        for j in 0..last {
            let dcol = &dd[j * lds..j * lds + k];
            let wcol = &wd[j * ldw..j * ldw + k];
            for i in 0..k {
                let p = if j < i { j } else { j + 1 };
                zprod[i] =
                    zprod[i] * (dcol[i] * wcol[i] / (dsig[i] - dsig[p]) / (dsig[i] + dsig[p]));
            }
        }
        for i in 0..k {
            z[i] = sign_of(<R as Real>::sqrt(fabs(zprod[i])), qsave[i]);
        }
    }

    // Singular vectors of M, normalised and permuted to the grouped order:
    // column i of `qu` / `qv` belongs to root i.
    let mut qu = Mat::<R>::zeros(k, k);
    let mut qv = Mat::<R>::zeros(k, k);
    {
        let ldq = qu.row_stride();
        let dd = dlt.raw_data_mut();
        let wd = wrk.raw_data_mut();
        let qud = qu.raw_data_mut();
        let qvd = qv.raw_data_mut();
        for i in 0..k {
            let dcol = &mut dd[i * lds..i * lds + k];
            let wcol = &mut wd[i * ldw..i * ldw + k];
            // wcol ← ẑⱼ / (dⱼ² − σᵢ²); dcol ← dⱼ·wcol (dcol[0] = −1).
            wcol[0] = z[0] / dcol[0] / wcol[0];
            dcol[0] = -one;
            for jj in 1..k {
                wcol[jj] = z[jj] / dcol[jj] / wcol[jj];
                dcol[jj] = dsig[jj] * wcol[jj];
            }
            let un = nrm2(dcol);
            let vn = nrm2(wcol);
            if !(un > zero && un.is_finite() && vn > zero && vn.is_finite()) {
                return Err(BidiagDcError::SecularEquationFailed);
            }
            let qucol = &mut qud[i * ldq..i * ldq + k];
            let qvcol = &mut qvd[i * ldq..i * ldq + k];
            qucol[0] = dcol[0] / un;
            qvcol[0] = wcol[0] / vn;
            for jj in 1..k {
                let jc = idxc[jj];
                qucol[jj] = dcol[jc] / un;
                qvcol[jj] = wcol[jc] / vn;
            }
        }
    }

    let (c1, c2, c3) = (ctot[0], ctot[1], ctot[2]);
    // U[:, 0..k] = U2 · Qu, skipping the structurally zero blocks.
    if k == 2 {
        ops.gemm(
            false,
            false,
            one,
            u2.as_ref().submatrix(0, 0, n, 2),
            qu.as_ref(),
            zero,
            u.submatrix_mut(off, off, n, 2),
        );
    } else {
        {
            let mut top = u.submatrix_mut(off, off, nl, k);
            let k3 = 1 + c1 + c2;
            if c1 > 0 {
                ops.gemm(
                    false,
                    false,
                    one,
                    u2.as_ref().submatrix(0, 1, nl, c1),
                    qu.as_ref().submatrix(1, 0, c1, k),
                    zero,
                    top.rb_mut(),
                );
                if c3 > 0 {
                    ops.gemm(
                        false,
                        false,
                        one,
                        u2.as_ref().submatrix(0, k3, nl, c3),
                        qu.as_ref().submatrix(k3, 0, c3, k),
                        one,
                        top.rb_mut(),
                    );
                }
            } else if c3 > 0 {
                ops.gemm(
                    false,
                    false,
                    one,
                    u2.as_ref().submatrix(0, k3, nl, c3),
                    qu.as_ref().submatrix(k3, 0, c3, k),
                    zero,
                    top.rb_mut(),
                );
            } else {
                top.fill_zero();
            }
        }
        for i in 0..k {
            u[(off + nl, off + i)] = qu[(0, i)];
        }
        ops.gemm(
            false,
            false,
            one,
            u2.as_ref().submatrix(nl + 1, 1 + c1, nr, c2 + c3),
            qu.as_ref().submatrix(1 + c1, 0, c2 + c3, k),
            zero,
            u.submatrix_mut(off + nl + 1, off, nr, k),
        );
    }

    // V[:, 0..k] = V2 · Qv, skipping the structurally zero blocks.
    if k == 2 {
        ops.gemm(
            false,
            false,
            one,
            v2.as_ref().submatrix(0, 0, m, 2),
            qv.as_ref(),
            zero,
            v.submatrix_mut(off, off, m, 2),
        );
        return Ok(());
    }
    {
        let mut top = v.submatrix_mut(off, off, nl + 1, k);
        ops.gemm(
            false,
            false,
            one,
            v2.as_ref().submatrix(0, 0, nl + 1, 1 + c1),
            qv.as_ref().submatrix(0, 0, 1 + c1, k),
            zero,
            top.rb_mut(),
        );
        if c3 > 0 {
            let k3 = 1 + c1 + c2;
            ops.gemm(
                false,
                false,
                one,
                v2.as_ref().submatrix(0, k3, nl + 1, c3),
                qv.as_ref().submatrix(k3, 0, c3, k),
                one,
                top,
            );
        }
    }
    // The special column also feeds the lower block: move it next to the
    // lower and dense columns (over the last upper one, whose lower part is
    // zero and which the upper product has already consumed).
    let kt = c1;
    if kt > 0 {
        for i in 0..k {
            qv[(kt, i)] = qv[(0, i)];
        }
        for r in nl + 1..m {
            v2[(r, kt)] = v2[(r, 0)];
        }
    }
    let ct = 1 + c2 + c3;
    ops.gemm(
        false,
        false,
        one,
        v2.as_ref().submatrix(nl + 1, kt, m - nl - 1, ct),
        qv.as_ref().submatrix(kt, 0, ct, k),
        zero,
        v.submatrix_mut(off + nl + 1, off, m - nl - 1, k),
    );
    Ok(())
}
