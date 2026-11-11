//! Reduction of the balanced pencil to Hessenberg–triangular form:
//! QR factorization of `B` (`DGEQRF`/`DORMQR`/`DORGQR` as used by `DGGES`)
//! followed by `DGGHRD` (COMPQ = COMPZ = 'V').

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::Mat;

use super::pipeline::QzWork;
use super::util::{Consts, larfg, lartg, rot_cols, rot_rows};

/// Applies `H = I − tau·u·uᵀ`, `u = [1; v]`, from the left to the rows
/// `row0..=row0+v.len()` of `m`, over the columns `cols` (`DLARF('L')`).
fn reflect_rows<T: Field + Real>(
    m: &mut Mat<T>,
    row0: usize,
    v: &[T],
    tau: T,
    cols: core::ops::Range<usize>,
) {
    for j in cols {
        let mut w = m[(row0, j)];
        for (l, &vl) in v.iter().enumerate() {
            w = w + vl * m[(row0 + 1 + l, j)];
        }
        let tw = tau * w;
        m[(row0, j)] = m[(row0, j)] - tw;
        for (l, &vl) in v.iter().enumerate() {
            m[(row0 + 1 + l, j)] = m[(row0 + 1 + l, j)] - vl * tw;
        }
    }
}

/// Applies `H = I − tau·u·uᵀ`, `u = [1; v]`, from the right to the columns
/// `col0..=col0+v.len()` of `m`, over the rows `rows` (`DLARF('R')`).
fn reflect_cols<T: Field + Real>(
    m: &mut Mat<T>,
    col0: usize,
    v: &[T],
    tau: T,
    rows: core::ops::Range<usize>,
) {
    for i in rows {
        let mut w = m[(i, col0)];
        for (l, &vl) in v.iter().enumerate() {
            w = w + vl * m[(i, col0 + 1 + l)];
        }
        let tw = tau * w;
        m[(i, col0)] = m[(i, col0)] - tw;
        for (l, &vl) in v.iter().enumerate() {
            m[(i, col0 + 1 + l)] = m[(i, col0 + 1 + l)] - vl * tw;
        }
    }
}

/// Step 4 of `DGGES`: `B(ilo..=ihi, ilo..n) = Q·R` with Householder
/// reflectors generated column by column (`DGEQR2`), each applied at once to
/// the trailing columns of `B`, to `A(ilo..=ihi, ilo..n)` (`DORMQR('L','T')`)
/// and accumulated into `Q := Q·H` (`DORGQR`, with `Q = I` on entry).
///
/// The reflector storage below the diagonal of `B` is replaced by exact
/// zeros, which is what `DGGHRD` writes there before its own reduction.
pub(super) fn qr_reduce_b<T: Field + Real + bytemuck::Zeroable>(w: &mut QzWork<T>, k: &Consts<T>) {
    let n = w.h.nrows();
    let (ilo, ihi) = (w.ilo, w.ihi);
    if ihi < ilo {
        return;
    }
    for col in ilo..=ihi {
        let mut alpha = w.t[(col, col)];
        let mut v: Vec<T> = ((col + 1)..=ihi).map(|i| w.t[(i, col)]).collect();
        let tau = larfg(&mut alpha, &mut v, k);
        w.t[(col, col)] = alpha;
        for i in (col + 1)..=ihi {
            w.t[(i, col)] = T::zero();
        }
        if tau == T::zero() {
            continue;
        }
        reflect_rows(&mut w.t, col, &v, tau, (col + 1)..n);
        reflect_rows(&mut w.h, col, &v, tau, ilo..n);
        reflect_cols(&mut w.q, col, &v, tau, 0..n);
    }
}

/// `DGGHRD` (COMPQ = COMPZ = 'V'): reduces `A` to upper Hessenberg form by
/// Givens rotations while keeping `B` upper triangular, updating `Q` and `Z`.
///
/// For each column `jcol` of the unreduced block, `A(jrow, jcol)` is
/// annihilated bottom-up by a rotation of rows `jrow−1, jrow` (applied to
/// `A`, `B` and the columns of `Q`); the fill-in `B(jrow, jrow−1)` is then
/// removed by a rotation of columns `jrow, jrow−1` (applied to `A`, `B` and
/// `Z`).
pub(super) fn gghrd<T: Field + Real + bytemuck::Zeroable>(w: &mut QzWork<T>, k: &Consts<T>) {
    let n = w.h.nrows();
    let (ilo, ihi) = (w.ilo, w.ihi);
    if n <= 1 {
        return;
    }
    // Zero out the lower triangle of B (loops 10/20).
    for jcol in 0..n - 1 {
        for jrow in (jcol + 1)..n {
            w.t[(jrow, jcol)] = T::zero();
        }
    }
    // Reduce A and B (loops 30/40): jcol = ilo..=ihi−2, jrow = ihi down to jcol+2.
    if ihi < ilo + 2 {
        return;
    }
    for jcol in ilo..=(ihi - 2) {
        for jrow in ((jcol + 2)..=ihi).rev() {
            // Step 1: rotate rows jrow−1, jrow to kill A(jrow, jcol).
            let temp = w.h[(jrow - 1, jcol)];
            let (c, s, r) = lartg(temp, w.h[(jrow, jcol)], k);
            w.h[(jrow - 1, jcol)] = r;
            w.h[(jrow, jcol)] = T::zero();
            // DROT(N−JCOL, A(JROW−1,JCOL+1), LDA, A(JROW,JCOL+1), LDA, C, S)
            rot_rows(&mut w.h, jrow - 1, jrow, (jcol + 1)..n, c, s);
            // DROT(N+2−JROW, B(JROW−1,JROW−1), LDB, B(JROW,JROW−1), LDB, C, S)
            rot_rows(&mut w.t, jrow - 1, jrow, (jrow - 1)..n, c, s);
            // DROT(N, Q(1,JROW−1), 1, Q(1,JROW), 1, C, S)
            rot_cols(&mut w.q, jrow - 1, jrow, 0..n, c, s);

            // Step 2: rotate columns jrow, jrow−1 to kill B(jrow, jrow−1).
            let temp = w.t[(jrow, jrow)];
            let (c, s, r) = lartg(temp, w.t[(jrow, jrow - 1)], k);
            w.t[(jrow, jrow)] = r;
            w.t[(jrow, jrow - 1)] = T::zero();
            // DROT(IHI, A(1,JROW), 1, A(1,JROW−1), 1, C, S)
            rot_cols(&mut w.h, jrow, jrow - 1, 0..(ihi + 1), c, s);
            // DROT(JROW−1, B(1,JROW), 1, B(1,JROW−1), 1, C, S)
            rot_cols(&mut w.t, jrow, jrow - 1, 0..jrow, c, s);
            // DROT(N, Z(1,JROW), 1, Z(1,JROW−1), 1, C, S)
            rot_cols(&mut w.z, jrow, jrow - 1, 0..n, c, s);
        }
    }
}
