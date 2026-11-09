//! Copy-free recursive drivers for TRSM and TRMM.
//!
//! # Canonical problem
//!
//! Every call is described as `T · C` with an `s x s` triangular `T` and
//! right-hand sides `C`:
//!
//! - Left (`op(A)·X = B`, `B := alpha·op(A)·B`): `T = op(A)`, `C = B`
//!   (the right-hand sides are the columns of B);
//! - Right (`X·op(A) = B`, `B := alpha·B·op(A)`): transposing gives
//!   `T = op(A)ᵀ`, `C = Bᵀ` (the right-hand sides are the rows of B).
//!
//! B is never transposed or copied as a whole: the packing routines read it
//! through either orientation.
//!
//! # Structure
//!
//! 1. The diagonal of `T` is checked once (TRSM: a zero on a non-unit
//!    diagonal is [`TrsmError::Singular`], reported before B is touched) and
//!    every diagonal block of order [`DIAG_BLOCK`] is packed once, with the
//!    reciprocals of the diagonal for TRSM (or the diagonal itself, divided
//!    robustly, when a reciprocal is unusable: see `nonunit_diag`).
//! 2. The right-hand sides are cut into fixed panels of [`rhs_panel`]
//!    columns (Left) or rows (Right): 128, or 256 once there are enough
//!    right-hand sides for four wide panels. The width depends only on the
//!    shape of the call, never on the thread count. The panels are
//!    independent; each is processed by the same sequential code whether
//!    the call runs on one thread or several, so the parallel result is
//!    bitwise identical to the sequential one by construction.
//! 3. Within a panel the triangular dimension is split recursively along
//!    block boundaries. Off-diagonal blocks are applied with one in-place
//!    GEMM on views (`C₂ ∓= T₂₁·C₁`, through the transpose-aware GEMM, so
//!    `op(A)` blocks are never copied), and diagonal blocks with the
//!    register-tiled kernels of [`super::tri_kernel`].

use super::tri_kernel::{
    SolveDiag, packed_tri_len, tile_shape, trmm_micro_panel, trsm_micro_panel,
};
use super::tri_pack::{TriView, pack_rhs, pack_tri_block, unpack_rhs};
use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use crate::level3::complex_gemm::{gemm3m_c32, gemm3m_c64};
use crate::level3::gemm::{gemm_transposed_c32, gemm_transposed_c64, gemm_transposed_with_par};
use crate::level3::gemm_kernel::GemmKernel;
use core::any::{Any, TypeId};
use num_complex::{Complex32, Complex64};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{MatMut, MatRef};

/// Order of the diagonal blocks solved by the register-tiled kernels. A
/// multiple of every tile height `MR`.
pub(crate) const DIAG_BLOCK: usize = 128;

/// Width of the right-hand-side panels (columns of B for Left, rows for
/// Right). A multiple of every tile width `NR`.
pub(crate) const RHS_PANEL: usize = 128;

/// Wider panels for calls with many right-hand sides: wider off-diagonal
/// GEMMs (and fewer per-call costs of the complex 3M GEMM) pay off once at
/// least four of them are available to run in parallel.
pub(crate) const RHS_PANEL_WIDE: usize = 256;

/// Panel width for a call with `nrhs` right-hand sides. It depends only on
/// the shape of the call, never on the thread count, which keeps parallel
/// and sequential results identical.
#[inline]
pub(crate) fn rhs_panel(nrhs: usize) -> usize {
    if nrhs >= 4 * RHS_PANEL_WIDE {
        RHS_PANEL_WIDE
    } else {
        RHS_PANEL
    }
}

/// Work gate (`s² · nrhs` multiply-adds, and per thread) below which the
/// panels are processed on the calling thread.
#[cfg(feature = "parallel")]
const PAR_GATE: oxiblas_core::parallel::ParThreshold =
    oxiblas_core::parallel::ParThreshold::new(128 * 128 * 128, 64 * 64 * 64);

/// Number of calls that took the parallel panel path (lets the tests check
/// that their parallel cases really ran in parallel).
#[cfg(all(test, feature = "parallel"))]
pub(crate) static PARALLEL_RUNS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// The canonical description of one TRSM/TRMM call.
pub(crate) struct Canon<'a, T: Field> {
    side: Side,
    trans: Trans,
    a: MatRef<'a, T>,
    view: TriView<'a, T>,
    /// `T` is lower triangular (forward order); otherwise upper (reversed).
    lower: bool,
    /// Order of `T`.
    size: usize,
}

impl<'a, T: Field> Canon<'a, T> {
    pub(crate) fn new(side: Side, uplo: Uplo, trans: Trans, a: MatRef<'a, T>) -> Self {
        // op(A) is lower triangular iff (uplo == Lower) == (trans == NoTrans).
        let op_lower = (uplo == Uplo::Lower) == (trans == Trans::NoTrans);
        Self {
            side,
            trans,
            a,
            view: TriView::new(a, side, trans),
            lower: if side == Side::Left {
                op_lower
            } else {
                !op_lower
            },
            size: a.nrows(),
        }
    }

    fn nblocks(&self) -> usize {
        self.size.div_ceil(DIAG_BLOCK)
    }

    /// `(offset, order)` of diagonal block `blk`.
    fn block(&self, blk: usize) -> (usize, usize) {
        let o = blk * DIAG_BLOCK;
        (o, DIAG_BLOCK.min(self.size - o))
    }

    /// Index range of the blocks `blo .. bhi`.
    fn range(&self, blo: usize, bhi: usize) -> (usize, usize) {
        (blo * DIAG_BLOCK, (bhi * DIAG_BLOCK).min(self.size))
    }

    /// Global index of local row `i` of the block `(o, kb)`.
    fn global(&self, o: usize, kb: usize, i: usize) -> usize {
        if self.lower { o + i } else { o + kb - 1 - i }
    }

    /// Number of right-hand sides held by `b`.
    fn nrhs(&self, b: &MatMut<'_, T>) -> usize {
        match self.side {
            Side::Left => b.ncols(),
            Side::Right => b.nrows(),
        }
    }

    /// Packs every diagonal block; `diag` holds one value per index of `T`
    /// (in global order) for the diagonal slots, `None` for a unit diagonal.
    fn pack_blocks(&self, diag: Option<&[T]>, mr: usize) -> Vec<Vec<T>> {
        (0..self.nblocks())
            .map(|blk| {
                let (o, kb) = self.block(blk);
                let local: Option<Vec<T>> =
                    diag.map(|d| (0..kb).map(|i| d[self.global(o, kb, i)]).collect());
                let mut packed = Vec::with_capacity(packed_tri_len(kb, mr));
                pack_tri_block(
                    &self.view,
                    o,
                    kb,
                    self.lower,
                    local.as_deref(),
                    mr,
                    &mut packed,
                );
                packed
            })
            .collect()
    }
}

/// `true` when both components of `v` are finite.
#[inline]
fn is_finite<T: Field>(v: T) -> bool {
    num_traits::Float::is_finite(v.real()) && num_traits::Float::is_finite(v.imag())
}

/// How the non-zero, non-unit diagonal `d` of a TRSM is applied, with the
/// values to pack in its diagonal slots: the reciprocals of `d` (multiplied,
/// [`SolveDiag::Reciprocal`]) unless one of them is unusable, and then `d`
/// itself (divided, [`SolveDiag::Divide`]).
pub(super) fn nonunit_diag<T: Field>(d: Vec<T>) -> (SolveDiag, Vec<T>) {
    let inv: Vec<T> = d.iter().map(|&v| T::one() / v).collect();
    if d.iter()
        .zip(inv.iter())
        .any(|(&v, &r)| reciprocal_unusable(v, r))
    {
        (SolveDiag::Divide, d)
    } else {
        (SolveDiag::Reciprocal, inv)
    }
}

/// Whether the computed reciprocal `r = 1 / v` of a diagonal entry would
/// turn a representable quotient into a wrong one.
///
/// That is the case for a finite `v` whose reciprocal overflows (a subnormal
/// entry, or a complex one whose squared modulus underflows to zero) or
/// underflows to zero, and for a complex `v` whose squared modulus is
/// subnormal: `num_complex` computes `1 / v` as `conj(v) / |v|²`, so `|v|²`
/// overflowing to infinity (`|v|` above about `1.3e154` for `Complex64`,
/// `1.8e19` for `Complex32`) gives a zero reciprocal, and a subnormal `|v|²`
/// keeps only a few correct digits. A real reciprocal is correctly rounded
/// and never zero, so for real types only the first case applies, as before.
/// Non-finite entries keep the reciprocal, so NaN and Inf propagate as in
/// the multiply mode.
#[inline]
fn reciprocal_unusable<T: Field>(v: T, r: T) -> bool {
    if !is_finite(v) {
        return false;
    }
    if !is_finite(r) || r == T::zero() {
        return true;
    }
    !T::is_real() && !num_traits::Float::is_normal(v.abs_sq())
}

/// Maps the block of `op(A)` at `(r0, c0)` with `nr x nc` entries to a view of
/// the stored `A` plus the transpose flag that recovers it in GEMM.
#[inline]
fn op_block<'a, T: Scalar>(
    a: MatRef<'a, T>,
    trans: Trans,
    r0: usize,
    c0: usize,
    nr: usize,
    nc: usize,
) -> (MatRef<'a, T>, Trans) {
    match trans {
        Trans::NoTrans => (a.submatrix(r0, c0, nr, nc), Trans::NoTrans),
        Trans::Trans | Trans::ConjTrans => (a.submatrix(c0, r0, nc, nr), trans),
    }
}

/// Reinterprets a view of `T` as a view of `U` when both are the same type.
#[inline]
fn cast_ref<'a, T: Scalar, U: Scalar>(m: MatRef<'a, T>) -> Option<MatRef<'a, U>> {
    if TypeId::of::<T>() != TypeId::of::<U>() {
        return None;
    }
    // SAFETY: `T` and `U` are the same type (identical `TypeId`s), so the
    // pointer cast is an identity. Shape and both strides are carried over,
    // so the new view addresses exactly the elements of `m`, for the same
    // lifetime `'a`, with shared access only.
    Some(unsafe {
        MatRef::new_strided(
            m.as_ptr().cast::<U>(),
            m.nrows(),
            m.ncols(),
            m.row_stride(),
            m.col_stride(),
        )
    })
}

/// Reinterprets a mutable view of `T` as one of `U` when both are the same
/// type; gives the view back unchanged otherwise.
#[inline]
fn cast_mut<'a, T: Scalar, U: Scalar>(
    mut m: MatMut<'a, T>,
) -> Result<MatMut<'a, U>, MatMut<'a, T>> {
    if TypeId::of::<T>() != TypeId::of::<U>() {
        return Err(m);
    }
    let (nrows, ncols, ld) = (m.nrows(), m.ncols(), m.row_stride());
    let ptr = m.as_mut_ptr().cast::<U>();
    // SAFETY: `T` and `U` are the same type (identical `TypeId`s), so the
    // pointer cast is an identity. `m` is consumed, so the new view is the
    // only handle to these elements for `'a`; shape and leading dimension are
    // unchanged (a `MatMut` is always column-major with a unit row step).
    Ok(unsafe { MatMut::new(ptr, nrows, ncols, ld) })
}

/// `C += alpha · op(A) · op(B)` in place.
///
/// Real types use the transpose-aware GEMM, which reads transposed operands
/// while packing. The complex types use the 3M GEMM (three real GEMMs, about
/// 2.5x faster here than the generic complex kernel): `Trans` operands are
/// passed as transposed views, `ConjTrans` ones through
/// [`gemm_transposed_c64`]/[`gemm_transposed_c32`], which materialize that
/// one operand. B is never copied.
fn gemm_acc<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: MatMut<'_, T>,
) {
    let Some(c) = try_gemm3m::<T, Complex64>(trans_a, trans_b, alpha, a, b, c) else {
        return;
    };
    let Some(c) = try_gemm3m::<T, Complex32>(trans_a, trans_b, alpha, a, b, c) else {
        return;
    };
    gemm_transposed_with_par(trans_a, trans_b, alpha, a, b, T::one(), c, Par::Seq);
}

/// The 3M complex GEMM, `C += alpha · op(A) · op(B)`.
trait Gemm3m: Scalar {
    fn gemm3m_acc(
        trans_a: Trans,
        trans_b: Trans,
        alpha: Self,
        a: MatRef<'_, Self>,
        b: MatRef<'_, Self>,
        c: MatMut<'_, Self>,
    );
}

macro_rules! impl_gemm3m {
    ($t:ty, $gemm3m:ident, $gemm_t:ident) => {
        impl Gemm3m for $t {
            fn gemm3m_acc(
                trans_a: Trans,
                trans_b: Trans,
                alpha: Self,
                a: MatRef<'_, Self>,
                b: MatRef<'_, Self>,
                c: MatMut<'_, Self>,
            ) {
                let one = <$t>::new(1.0, 0.0);
                if trans_a == Trans::ConjTrans || trans_b == Trans::ConjTrans {
                    $gemm_t(trans_a, trans_b, alpha, a, b, one, c);
                } else {
                    let a = if trans_a == Trans::NoTrans {
                        a
                    } else {
                        a.transpose()
                    };
                    let b = if trans_b == Trans::NoTrans {
                        b
                    } else {
                        b.transpose()
                    };
                    $gemm3m(alpha, a, b, one, c);
                }
            }
        }
    };
}

impl_gemm3m!(Complex64, gemm3m_c64, gemm_transposed_c64);
impl_gemm3m!(Complex32, gemm3m_c32, gemm_transposed_c32);

/// Runs the 3M update when `T` is the complex type `U` (returning `None`);
/// otherwise hands `c` back untouched.
fn try_gemm3m<'c, T: Scalar, U: Gemm3m>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: MatMut<'c, T>,
) -> Option<MatMut<'c, T>> {
    let (Some(a), Some(b), Some(&alpha)) = (
        cast_ref::<T, U>(a),
        cast_ref::<T, U>(b),
        (&alpha as &dyn Any).downcast_ref::<U>(),
    ) else {
        return Some(c);
    };
    match cast_mut::<T, U>(c) {
        Ok(c) => {
            U::gemm3m_acc(trans_a, trans_b, alpha, a, b, c);
            None
        }
        // Not reachable once the casts above succeeded; the caller then
        // falls back to the generic GEMM, which is equally correct.
        Err(c) => Some(c),
    }
}

/// `C[target] += alpha · T[target, source] · C[source]` in place, written in
/// terms of B: Left `B[t, :] += alpha·op(A)[t, s]·B[s, :]`, Right
/// `B[:, t] += alpha·B[:, s]·op(A)[s, t]`. `target` and `source` are disjoint
/// index ranges of `T`.
fn update<T: Field + GemmKernel + bytemuck::Zeroable>(
    c: &Canon<'_, T>,
    b: MatMut<'_, T>,
    target: (usize, usize),
    source: (usize, usize),
    alpha: T,
) {
    let (t0, t1) = target;
    let (s0, s1) = source;
    if t0 == t1 || s0 == s1 {
        return;
    }
    match c.side {
        Side::Left => {
            let ncols = b.ncols();
            let (blk, flag) = op_block(c.a, c.trans, t0, s0, t1 - t0, s1 - s0);
            if t0 >= s1 {
                let (top, bottom) = b.split_rows(t0);
                let src = top.rb().submatrix(s0, 0, s1 - s0, ncols);
                let dst = bottom.submatrix(0, 0, t1 - t0, ncols);
                gemm_acc(flag, Trans::NoTrans, alpha, blk, src, dst);
            } else {
                let (top, bottom) = b.split_rows(s0);
                let src = bottom.rb().submatrix(0, 0, s1 - s0, ncols);
                let dst = top.submatrix(t0, 0, t1 - t0, ncols);
                gemm_acc(flag, Trans::NoTrans, alpha, blk, src, dst);
            }
        }
        Side::Right => {
            let nrows = b.nrows();
            let (blk, flag) = op_block(c.a, c.trans, s0, t0, s1 - s0, t1 - t0);
            if t0 >= s1 {
                let (left, right) = b.split_cols(t0);
                let src = left.rb().submatrix(0, s0, nrows, s1 - s0);
                let dst = right.submatrix(0, 0, nrows, t1 - t0);
                gemm_acc(Trans::NoTrans, flag, alpha, src, blk, dst);
            } else {
                let (left, right) = b.split_cols(s0);
                let src = right.rb().submatrix(0, 0, nrows, s1 - s0);
                let dst = left.submatrix(0, t0, nrows, t1 - t0);
                gemm_acc(Trans::NoTrans, flag, alpha, src, blk, dst);
            }
        }
    }
}

/// Applies a diagonal-block kernel to every `NR`-wide micro-panel of the
/// right-hand sides in `b`.
fn for_each_micro_panel<T: Field, K>(
    c: &Canon<'_, T>,
    blk: usize,
    b: &mut MatMut<'_, T>,
    xp: &mut Vec<T>,
    kernel: K,
) where
    K: Fn(&mut Vec<T>, usize),
{
    let (mr, nr) = tile_shape::<T>();
    let (o, kb) = c.block(blk);
    let nrhs = c.nrhs(b);
    let mut j0 = 0;
    while j0 < nrhs {
        let nc = nr.min(nrhs - j0);
        pack_rhs(b, c.side, o, kb, c.lower, j0, nc, mr, nr, xp);
        kernel(xp, kb);
        unpack_rhs(b, c.side, o, kb, c.lower, j0, nc, nr, xp);
        j0 += nc;
    }
}

/// Recursive forward/backward substitution over the blocks `blo .. bhi`.
#[allow(clippy::too_many_arguments)]
fn trsm_rec<T: Field + GemmKernel + bytemuck::Zeroable>(
    c: &Canon<'_, T>,
    packs: &[Vec<T>],
    diag: SolveDiag,
    mut b: MatMut<'_, T>,
    blo: usize,
    bhi: usize,
    xp: &mut Vec<T>,
) {
    if bhi - blo == 1 {
        let tri = &packs[blo];
        for_each_micro_panel(c, blo, &mut b, xp, |x, kb| {
            trsm_micro_panel(tri, kb, x, diag)
        });
        return;
    }
    let h = (bhi - blo).div_ceil(2);
    // Lower: the leading blocks are solved first; upper: the trailing ones.
    let (first, second) = if c.lower {
        ((blo, blo + h), (blo + h, bhi))
    } else {
        ((bhi - h, bhi), (blo, bhi - h))
    };
    trsm_rec(c, packs, diag, b.rb_mut(), first.0, first.1, xp);
    update(
        c,
        b.rb_mut(),
        c.range(second.0, second.1),
        c.range(first.0, first.1),
        -T::one(),
    );
    trsm_rec(c, packs, diag, b, second.0, second.1, xp);
}

/// Recursive triangular multiply over the blocks `blo .. bhi`, in place: the
/// half whose result depends on the other half is finished first, while the
/// other half still holds its original values.
#[allow(clippy::too_many_arguments)]
fn trmm_rec<T: Field + GemmKernel + bytemuck::Zeroable>(
    c: &Canon<'_, T>,
    packs: &[Vec<T>],
    unit: bool,
    alpha: T,
    mut b: MatMut<'_, T>,
    blo: usize,
    bhi: usize,
    xp: &mut Vec<T>,
) {
    if bhi - blo == 1 {
        let tri = &packs[blo];
        let scale = if alpha == T::one() { None } else { Some(alpha) };
        for_each_micro_panel(c, blo, &mut b, xp, |x, kb| {
            trmm_micro_panel(tri, kb, x, unit, scale)
        });
        return;
    }
    let h = (bhi - blo).div_ceil(2);
    let lo = (blo, blo + h);
    let hi = (blo + h, bhi);
    // Lower: rows of the high half depend on the low half; upper: vice versa.
    let (late, early) = if c.lower { (hi, lo) } else { (lo, hi) };
    trmm_rec(c, packs, unit, alpha, b.rb_mut(), late.0, late.1, xp);
    update(
        c,
        b.rb_mut(),
        c.range(late.0, late.1),
        c.range(early.0, early.1),
        alpha,
    );
    trmm_rec(c, packs, unit, alpha, b, early.0, early.1, xp);
}

/// Splits `b` into the fixed right-hand-side panels and runs `f` on each,
/// in parallel when `par` allows it and the work is large enough.
fn for_each_panel<T: Field, F>(b: MatMut<'_, T>, side: Side, par: Par, work: usize, f: &F)
where
    F: Fn(MatMut<'_, T>) + Sync,
{
    let len = match side {
        Side::Left => b.ncols(),
        Side::Right => b.nrows(),
    };
    // Fixed once from the whole call, so every split below lands on the same
    // panel boundaries as the sequential loop.
    let width = rhs_panel(len);
    #[cfg(feature = "parallel")]
    {
        if len > width && PAR_GATE.should_parallelize(work, par) {
            #[cfg(test)]
            PARALLEL_RUNS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            par_panels(b, side, width, f);
            return;
        }
    }
    let _ = (par, work);
    seq_panels(b, side, width, f);
}

/// Runs `f` on the panels of `b` in order.
fn seq_panels<T: Field, F>(mut b: MatMut<'_, T>, side: Side, width: usize, f: &F)
where
    F: Fn(MatMut<'_, T>),
{
    let (nrows, ncols) = (b.nrows(), b.ncols());
    let len = match side {
        Side::Left => ncols,
        Side::Right => nrows,
    };
    let mut j0 = 0;
    while j0 < len {
        let w = width.min(len - j0);
        let panel = match side {
            Side::Left => b.rb_mut().submatrix(0, j0, nrows, w),
            Side::Right => b.rb_mut().submatrix(j0, 0, w, ncols),
        };
        f(panel);
        j0 += w;
    }
}

/// Runs `f` on the panels of `b` with recursive `rayon::join`, splitting only
/// at panel boundaries (multiples of `width` from the first column/row).
#[cfg(feature = "parallel")]
fn par_panels<T: Field, F>(b: MatMut<'_, T>, side: Side, width: usize, f: &F)
where
    F: Fn(MatMut<'_, T>) + Sync,
{
    let len = match side {
        Side::Left => b.ncols(),
        Side::Right => b.nrows(),
    };
    let np = len.div_ceil(width);
    if np <= 1 {
        f(b);
        return;
    }
    let mid = (np / 2) * width;
    let (x, y) = match side {
        Side::Left => b.split_cols(mid),
        Side::Right => b.split_rows(mid),
    };
    rayon::join(
        || par_panels(x, side, width, f),
        || par_panels(y, side, width, f),
    );
}

/// In-place TRSM on a non-empty problem whose shapes have been validated:
/// solves `op(A)·X = B` (Left) or `X·op(A) = B` (Right), overwriting B.
pub(crate) fn trsm_solve<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, T>,
    b: MatMut<'_, T>,
    par: Par,
) -> Result<(), TrsmError> {
    let c = Canon::new(side, uplo, trans, a);
    let (mr, _) = tile_shape::<T>();

    let (mode, values) = match diag {
        Diag::Unit => (SolveDiag::Unit, None),
        Diag::NonUnit => {
            let mut d = Vec::with_capacity(c.size);
            for k in 0..c.size {
                let v = c.view.get(k, k);
                if v == T::zero() {
                    return Err(TrsmError::Singular);
                }
                d.push(v);
            }
            let (mode, values) = nonunit_diag(d);
            (mode, Some(values))
        }
    };

    let packs = c.pack_blocks(values.as_deref(), mr);
    let nblocks = c.nblocks();
    let work = c.size.saturating_mul(c.size).saturating_mul(c.nrhs(&b));
    let run = |panel: MatMut<'_, T>| {
        let mut xp = Vec::new();
        trsm_rec(&c, &packs, mode, panel, 0, nblocks, &mut xp);
    };
    for_each_panel(b, side, par, work, &run);
    Ok(())
}

/// In-place TRMM on a non-empty problem whose shapes have been validated
/// and with `alpha != 0`: `B := alpha·op(A)·B` (Left) or `alpha·B·op(A)`
/// (Right).
#[allow(clippy::too_many_arguments)]
pub(crate) fn trmm_apply<T: Field + GemmKernel + bytemuck::Zeroable>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    unit: bool,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatMut<'_, T>,
    par: Par,
) {
    let c = Canon::new(side, uplo, trans, a);
    let (mr, _) = tile_shape::<T>();
    let values: Option<Vec<T>> = if unit {
        None
    } else {
        Some((0..c.size).map(|k| c.view.get(k, k)).collect())
    };
    let packs = c.pack_blocks(values.as_deref(), mr);
    let nblocks = c.nblocks();
    let work = c.size.saturating_mul(c.size).saturating_mul(c.nrhs(&b));
    let run = |panel: MatMut<'_, T>| {
        let mut xp = Vec::new();
        trmm_rec(&c, &packs, unit, alpha, panel, 0, nblocks, &mut xp);
    };
    for_each_panel(b, side, par, work, &run);
}
