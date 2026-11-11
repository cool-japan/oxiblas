//! Packing of diagonal blocks and right-hand-side micro-panels for the
//! triangular kernels in [`super::tri_kernel`].
//!
//! The drivers describe every TRSM/TRMM call as a *canonical* problem
//! `T · C` on an `s x s` triangular matrix `T` and right-hand sides `C`
//! (see [`super::tri_driver`]). A diagonal block of `T` is turned into the
//! lower-triangular `L` the kernels expect: `L[i, k] = T[o + i, o + k]` when
//! `T` is lower (forward order), and `L[i, k] = T[o + kb - 1 - i, o + kb - 1
//! - k]` when `T` is upper (reversed order, which makes it lower). The rows
//! of `C` are reversed the same way, so one forward kernel serves all cases.

use super::types::{Side, Trans};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

/// Read access to the canonical triangular matrix `T` of a call.
///
/// `T[p, q]` is `A[p, q]` or `A[q, p]` (`swap`), conjugated for `conj`.
/// Only entries of the referenced triangle (and the diagonal for a
/// non-unit matrix) are ever requested.
#[derive(Clone, Copy)]
pub(crate) struct TriView<'a, T: Field> {
    pub(crate) a: MatRef<'a, T>,
    pub(crate) swap: bool,
    pub(crate) conj: bool,
}

impl<'a, T: Field> TriView<'a, T> {
    /// The canonical matrix of a call: TRSM `op(A)·X = B` / TRMM
    /// `op(A)·B` (Left) use `T = op(A)`; the Right forms are transposed to
    /// `op(A)ᵀ · Xᵀ = Bᵀ`, so `T = op(A)ᵀ`.
    pub(crate) fn new(a: MatRef<'a, T>, side: Side, trans: Trans) -> Self {
        // Left: T[p,q] = op(A)[p,q]  -> A[p,q] (NoTrans) or A[q,p] (Trans/ConjTrans).
        // Right: T[p,q] = op(A)[q,p] -> A[q,p] (NoTrans) or A[p,q] (Trans/ConjTrans).
        let swap = (side == Side::Left) != (trans == Trans::NoTrans);
        Self {
            a,
            swap,
            conj: trans == Trans::ConjTrans,
        }
    }

    /// Returns `T[p, q]`.
    #[inline]
    pub(crate) fn get(&self, p: usize, q: usize) -> T {
        let v = if self.swap {
            self.a[(q, p)]
        } else {
            self.a[(p, q)]
        };
        if self.conj { v.conj() } else { v }
    }
}

/// Global index (in the triangular dimension) of local row `i` of the block
/// `[o, o + kb)`.
#[inline(always)]
fn global(o: usize, kb: usize, forward: bool, i: usize) -> usize {
    if forward { o + i } else { o + kb - 1 - i }
}

/// Packs the diagonal block `[o, o + kb)` of `T` for the kernels (layout in
/// the [`super::tri_kernel`] module docs) into `out`, which is cleared first.
///
/// `diag[i]` is stored in the diagonal slot of local row `i` (already in
/// local order); with `diag == None` (unit diagonal) the slot is zero and the
/// diagonal of `A` is never read.
pub(crate) fn pack_tri_block<T: Field>(
    view: &TriView<'_, T>,
    o: usize,
    kb: usize,
    forward: bool,
    diag: Option<&[T]>,
    mr: usize,
    out: &mut Vec<T>,
) {
    // Panel p is L[p·mr .. p·mr + mr, 0 .. (p + 1)·mr] stored column-major
    // with leading dimension mr (its rectangular part followed by its
    // diagonal tile), so L[i, k] (k <= i) lives at
    // base(p) + k·mr + (i - p·mr) with p = i / mr and base(p) = mr²·p(p+1)/2.
    // Everything not written below (the strict upper part of the tiles and
    // the padding rows) stays zero.
    out.clear();
    out.resize(super::tri_kernel::packed_tri_len(kb, mr), T::zero());
    let idx = |i: usize, k: usize| {
        let p = i / mr;
        mr * mr * p * (p + 1) / 2 + k * mr + (i - p * mr)
    };
    let g = |i: usize| global(o, kb, forward, i);
    let a = view.a;
    let conj = |v: T| if view.conj { v.conj() } else { v };
    // Read whole columns of the stored A (contiguous for a column-major
    // view): column g(k) of T when T = A, and row g(i) of T (column g(i) of
    // A) when T = Aᵀ. Other views are read element by element.
    for outer in 0..kb {
        let col = if a.col_stride() == 1 {
            a.col_as_slice(g(outer))
        } else {
            None
        };
        match (view.swap, col) {
            // T[g(i), g(k)] = A[g(k), g(i)]: row i of L from column g(i).
            (true, Some(col)) => {
                let i = outer;
                for k in 0..i {
                    out[idx(i, k)] = conj(col[g(k)]);
                }
            }
            // T[g(i), g(k)] = A[g(i), g(k)]: column k of L from column g(k).
            (false, Some(col)) => {
                let k = outer;
                for i in (k + 1)..kb {
                    out[idx(i, k)] = conj(col[g(i)]);
                }
            }
            (_, None) => {
                let k = outer;
                for i in (k + 1)..kb {
                    out[idx(i, k)] = view.get(g(i), g(k));
                }
            }
        }
    }
    if let Some(d) = diag {
        for (i, &v) in d.iter().enumerate().take(kb) {
            out[idx(i, i)] = v;
        }
    }
}

/// Copies the right-hand-side micro-panel (rows `[o, o + kb)` of the
/// canonical `C`, columns `j0 .. j0 + nc`) of `b` into `xp` (row-major, `nr`
/// columns, `kb` rounded up to `mr` rows; padding is zero-filled).
///
/// Left: `C = B`, so canonical row `g` is row `g` of B and column `c` is
/// column `j0 + c`. Right: `C = Bᵀ`, so canonical row `g` is column `g` of B
/// and column `c` is row `j0 + c` (contiguous in B).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_rhs<T: Field>(
    b: &MatMut<'_, T>,
    side: Side,
    o: usize,
    kb: usize,
    forward: bool,
    j0: usize,
    nc: usize,
    mr: usize,
    nr: usize,
    xp: &mut Vec<T>,
) {
    let rows = kb.div_ceil(mr) * mr;
    if xp.len() != rows * nr {
        xp.clear();
        xp.resize(rows * nr, T::zero());
    }
    let src = b.rb();
    match side {
        Side::Left => {
            for c in 0..nc {
                // Rows o .. o + kb of column j0 + c (reversed for backward)
                // into column c of the micro-panel.
                let col = &column(&src, j0 + c)[o..o + kb];
                let dst = xp[c..].iter_mut().step_by(nr);
                if forward {
                    for (d, &v) in dst.zip(col.iter()) {
                        *d = v;
                    }
                } else {
                    for (d, &v) in dst.zip(col.iter().rev()) {
                        *d = v;
                    }
                }
            }
        }
        Side::Right => {
            for (i, row) in xp[..kb * nr].chunks_exact_mut(nr).enumerate() {
                let col = &column(&src, global(o, kb, forward, i))[j0..j0 + nc];
                // Element loop: `nc` is at most a few elements, too short for
                // `copy_from_slice` (a `memcpy` call) to pay off.
                for (dst, &v) in row.iter_mut().zip(col.iter()) {
                    *dst = v;
                }
            }
        }
    }
    // Padding (left over from a previous micro-panel otherwise): the unused
    // columns of the valid rows and every padding row.
    if nc < nr {
        for row in xp[..kb * nr].chunks_exact_mut(nr) {
            row[nc..].fill(T::zero());
        }
    }
    xp[kb * nr..].fill(T::zero());
}

/// Writes the first `nc` columns of the packed micro-panel back into `b`
/// (the inverse of [`pack_rhs`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn unpack_rhs<T: Field>(
    b: &mut MatMut<'_, T>,
    side: Side,
    o: usize,
    kb: usize,
    forward: bool,
    j0: usize,
    nc: usize,
    nr: usize,
    xp: &[T],
) {
    match side {
        Side::Left => {
            for c in 0..nc {
                let col = &mut column_mut(b, j0 + c)[o..o + kb];
                let src = xp[c..].iter().step_by(nr);
                if forward {
                    for (d, &v) in col.iter_mut().zip(src) {
                        *d = v;
                    }
                } else {
                    for (d, &v) in col.iter_mut().rev().zip(src) {
                        *d = v;
                    }
                }
            }
        }
        Side::Right => {
            for (i, row) in xp[..kb * nr].chunks_exact(nr).enumerate() {
                let col = &mut column_mut(b, global(o, kb, forward, i))[j0..j0 + nc];
                for (dst, &v) in col.iter_mut().zip(row.iter()) {
                    *dst = v;
                }
            }
        }
    }
}

/// Column `j` of a column-major view as a slice.
#[inline]
fn column<'b, T: Field>(m: &MatRef<'b, T>, j: usize) -> &'b [T] {
    // A `MatRef` obtained from a `MatMut` always has unit row-to-row stride,
    // so every column is contiguous.
    m.col_as_slice(j).unwrap_or(&[])
}

/// Column `j` of a column-major mutable view as a slice.
#[inline]
fn column_mut<'b, T: Field>(m: &'b mut MatMut<'_, T>, j: usize) -> &'b mut [T] {
    m.col_as_slice_mut(j).unwrap_or(&mut [])
}
