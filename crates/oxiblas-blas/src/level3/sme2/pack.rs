//! Operand normalisation and the plain packers of the SME2 path (milestone 1
//! of the packers; the ZA-based packers come with Wave 4 in `pack_za.rs`).
//!
//! # What this module gives a driver
//!
//! The SME2 macro-kernels read A and B in the packed layouts of `args.rs`
//! (A in 64-row panels, B in `NR`-column panels, both zero-padded) and write a
//! column-major C. This module turns the GEMM operands into those layouts:
//!
//! 1. [`ColMajor::normalize`] rewrites an operand view `op(X)` into a stored
//!    matrix with unit column stride (column-major, leading dimension
//!    `row_stride`) plus `NoTrans` or `Trans`. A view whose rows are adjacent
//!    (`row_stride == 1`) is transposed and its `Trans` flipped. `ConjTrans` is
//!    `Trans`: the SME2 path serves real types only, and `ColMajor` can only
//!    be built for the real SME element types (`f32`, `f64`: the impl is
//!    bounded by `SmeElement`), so a complex operand cannot lose its
//!    conjugation here. A view in which neither
//!    stride is 1 is declined (`None`), and so is an empty view. A dimension of
//!    length 1 makes its stride irrelevant, so a single row or column with any
//!    strides is accepted. After normalisation the packers of `gemm.rs` always
//!    take their contiguous fast paths, never the element-wise fallback.
//! 2. [`pack_a`] packs a block `op(A)[rows, depth]` with
//!    `pack_a_trans_dispatch(.., mr = 64)`, [`pack_b`] a block
//!    `op(B)[depth, cols]` with `pack_b_trans_dispatch(.., nr = T::NR)`. They
//!    return [`PackedA`] / [`PackedB`], a borrow of the buffer that remembers
//!    the block's size; [`a_pack_len`] / [`b_pack_len`] give the buffer
//!    lengths to request from `PackBuffer`.
//! 3. [`block_args`] combines one packed A block, one packed B block and the
//!    matching block of C into the `MacroArgs` of one kernel call
//!    (`MacroArgs::packed`, so the epilogue mode follows alpha and the
//!    effective beta).
//!
//! Every function checks its inputs and returns `None` instead of panicking
//! (a region outside the operand, a buffer that is too short, an empty block,
//! an overflowing size), so the asserts inside the `gemm.rs` packers never
//! fire on this path.
//!
//! # The loop of a sequential driver (`gemm_blocked_sequential` order)
//!
//! ```text
//! a  = ColMajor::normalize(a_view, trans_a)?          once, on the calling thread
//! b  = ColMajor::normalize(b_view, trans_b)?          (None: decline, NEON runs)
//! (m, k) = (a.rows(), a.cols());  n = b.cols()        (b.rows() == k)
//! buf_a = PackBuffer::take(PackSlot::A, a_pack_len(min(mc, m), min(kc, k))?)
//! buf_b = PackBuffer::take(PackSlot::B, b_pack_len::<T>(min(kc, k), min(nc, n))?)
//! for depth in spans(k, kc):                          pc loop (kc from the caller, R1)
//!     beta_eff = block_beta(depth, beta)              beta first, 1 afterwards
//!     for cols in spans(n, nc):                       jc loop
//!         pb = pack_b(&b, depth, cols, &mut buf_b)?
//!         for rows in spans(m, mc):                   ic loop
//!             pa = pack_a(&a, rows, depth, &mut buf_a)?
//!             args = block_args(&pa, &pb, &mut c, rows.start, cols.start, alpha, beta_eff)?
//!             unsafe { kernel(&args) }                one streaming region per block
//! ```
//!
//! A `None` inside the loops cannot happen when the sizes come from the
//! normalised operands and the buffers from the two length helpers; a driver
//! that still gets one must not fall back half-way (C may already hold some
//! blocks), so it should treat it as a bug of its own, before the first
//! kernel call if it validates up front.

// Nothing in the library packs for the SME2 kernels before the GEMM hook
// (Wave 6) lands; only tests use this module until then. `expect` turns into
// a warning once that changes, so the attribute cannot outlive its reason.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the SME2 drivers once the GEMM hook lands"
    )
)]

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

use super::args::{MR, MacroArgs, SmeElement};
use crate::level3::gemm::{pack_a_trans_dispatch, pack_b_trans_dispatch};
use crate::level3::trsm::Trans;

/// A GEMM operand `op(X)` whose stored matrix has unit column stride
/// (column-major: element `(r, c)` of the stored matrix is at
/// `r + c * ld` with `ld = row_stride`), and whose transpose is `NoTrans`
/// or `Trans`.
#[derive(Debug, Clone, Copy)]
pub(super) struct ColMajor<'a, T: Field> {
    /// The stored matrix; `col_stride() == 1`.
    mat: MatRef<'a, T>,
    /// `NoTrans` or `Trans` (never `ConjTrans`).
    trans: Trans,
}

// Bounded by `SmeElement` (implemented by `f32` and `f64` only) and not just
// `Field`: `normalize` maps `ConjTrans` to `Trans`, which is right only for a
// real type. With this bound a complex `ColMajor` cannot be constructed, so
// the mapping cannot silently drop a conjugation (a compile error instead of
// a runtime check that could be forgotten).
impl<'a, T: SmeElement + Field> ColMajor<'a, T> {
    /// Rewrites `op(view)` (with `op` given by `trans`) into a column-major
    /// stored matrix and a transpose, or declines.
    ///
    /// - `ConjTrans` is treated as `Trans`: `T` is real (`f32` or `f64`, by
    ///   the `SmeElement` bound), so `conj(x) = x`.
    /// - `col_stride == 1` (or a single row): the view is kept (a single row
    ///   gets unit column stride).
    /// - `row_stride == 1` (or a single column): the view is transposed and
    ///   the transpose flipped, so `op(view)` is unchanged.
    /// - Otherwise, and for an empty view: `None`.
    ///
    /// The result reads exactly the elements of `op(view)`: element `(i, j)`
    /// of [`rows`](Self::rows) × [`cols`](Self::cols) is element `(i, j)` of
    /// `op(view)`.
    pub(super) fn normalize(view: MatRef<'a, T>, trans: Trans) -> Option<Self> {
        let trans = match trans {
            Trans::NoTrans => Trans::NoTrans,
            Trans::Trans | Trans::ConjTrans => Trans::Trans,
        };
        let (rows, cols) = (view.nrows(), view.ncols());
        if rows == 0 || cols == 0 {
            return None;
        }
        if view.col_stride() == 1 {
            return Some(Self { mat: view, trans });
        }
        if rows == 1 {
            // SAFETY: a 1 × cols view addresses only the offsets
            // `j * row_stride` (`j < cols`); the rebuilt view has the same
            // pointer, shape and row stride, so it addresses exactly those
            // offsets, which `view` guarantees are valid for `'a`.
            let mat = unsafe { MatRef::new_strided(view.as_ptr(), 1, cols, view.row_stride(), 1) };
            return Some(Self { mat, trans });
        }
        if view.row_stride() == 1 || cols == 1 {
            // A view with adjacent rows is the transpose of a column-major
            // matrix; a single column is one whatever its row stride. In both
            // cases element `(i, 0..)` of the stored transpose is
            // `view(.., i)`: rebuild it with unit column stride.
            // SAFETY: the rebuilt `cols × rows` view addresses the offsets
            // `c + r * col_stride` for `c < cols`, `r < rows` with a column
            // stride of 1 — when `row_stride == 1` these are exactly the
            // offsets `r * col_stride + c * row_stride` of `view`; when
            // `cols == 1` only `c = 0` exists, so they are `r * col_stride`,
            // again offsets of `view`. All are valid for `'a`.
            let mat =
                unsafe { MatRef::new_strided(view.as_ptr(), cols, rows, view.col_stride(), 1) };
            let trans = match trans {
                Trans::NoTrans => Trans::Trans,
                _ => Trans::NoTrans,
            };
            return Some(Self { mat, trans });
        }
        None
    }

    /// Rows of `op(X)`.
    pub(super) fn rows(&self) -> usize {
        match self.trans {
            Trans::NoTrans => self.mat.nrows(),
            _ => self.mat.ncols(),
        }
    }

    /// Columns of `op(X)`.
    pub(super) fn cols(&self) -> usize {
        match self.trans {
            Trans::NoTrans => self.mat.ncols(),
            _ => self.mat.nrows(),
        }
    }

    /// The stored, column-major matrix (`col_stride() == 1`; its leading
    /// dimension is `row_stride()`), for in-place reads (Wave 4).
    pub(super) fn stored(&self) -> &MatRef<'a, T> {
        &self.mat
    }

    /// `NoTrans` or `Trans`: whether `op(X)` is the stored matrix transposed.
    pub(super) fn trans(&self) -> Trans {
        self.trans
    }
}

/// A half-open range `start..start + len` of rows, columns or depth steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Span {
    /// First index.
    pub(super) start: usize,
    /// Number of indices.
    pub(super) len: usize,
}

/// The blocks of `0..total` in steps of `step`: `start = 0, step, 2·step, …`,
/// each `len = min(step, total - start)`. A `step` of 0 counts as 1; a
/// `total` of 0 gives no block.
pub(super) fn spans(total: usize, step: usize) -> impl Iterator<Item = Span> {
    let step = step.max(1);
    (0..total).step_by(step).map(move |start| Span {
        start,
        len: step.min(total - start),
    })
}

/// The effective beta of the depth block `depth`: `beta` for the first block
/// (`start == 0`), `1` for every later one, as the NEON driver does.
pub(super) fn block_beta<T: SmeElement>(depth: Span, beta: T) -> T {
    if depth.start == 0 { beta } else { T::ONE }
}

/// Elements of the A buffer for blocks of up to `rows × depth`
/// (`ceil(rows / 64) · 64 · depth`), or `None` on overflow.
pub(super) fn a_pack_len(rows: usize, depth: usize) -> Option<usize> {
    rows.div_ceil(MR).checked_mul(MR)?.checked_mul(depth)
}

/// Elements of the B buffer for blocks of up to `depth × cols`
/// (`ceil(cols / NR) · NR · depth`), or `None` on overflow.
pub(super) fn b_pack_len<T: SmeElement>(depth: usize, cols: usize) -> Option<usize> {
    cols.div_ceil(T::NR).checked_mul(T::NR)?.checked_mul(depth)
}

/// Whether `span` is non-empty and lies inside `0..total`.
fn inside(span: Span, total: usize) -> bool {
    span.len > 0
        && span
            .start
            .checked_add(span.len)
            .is_some_and(|end| end <= total)
}

/// One packed A block: `rows × depth` of `op(A)` in 64-row panels
/// (`A(i, p) = data[(i / 64) · 64 · depth + p · 64 + i % 64]`, rows
/// `rows..ceil(rows / 64) · 64` zero).
#[derive(Debug, Clone, Copy)]
pub(super) struct PackedA<'p, T> {
    /// The packed prefix of the buffer.
    data: &'p [T],
    /// Rows of the block (`≥ 1`).
    rows: usize,
    /// Depth of the block (`≥ 1`).
    depth: usize,
}

impl<T> PackedA<'_, T> {
    /// The packed elements (`ceil(rows / 64) · 64 · depth`).
    pub(super) fn data(&self) -> &[T] {
        self.data
    }

    /// Rows of the block.
    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// Depth of the block.
    pub(super) fn depth(&self) -> usize {
        self.depth
    }
}

/// One packed B block: `depth × cols` of `op(B)` in `NR`-column panels
/// (`B(p, j) = data[(j / NR) · depth · NR + p · NR + j % NR]`, columns
/// `cols..ceil(cols / NR) · NR` zero).
#[derive(Debug, Clone, Copy)]
pub(super) struct PackedB<'p, T> {
    /// The packed prefix of the buffer.
    data: &'p [T],
    /// Depth of the block (`≥ 1`).
    depth: usize,
    /// Columns of the block (`≥ 1`).
    cols: usize,
}

impl<T> PackedB<'_, T> {
    /// The packed elements (`ceil(cols / NR) · NR · depth`).
    pub(super) fn data(&self) -> &[T] {
        self.data
    }

    /// Depth of the block.
    pub(super) fn depth(&self) -> usize {
        self.depth
    }

    /// Columns of the block.
    pub(super) fn cols(&self) -> usize {
        self.cols
    }
}

/// Packs `op(A)[rows, depth]` into `buf` in the plain layout of `args.rs`
/// (`a_depth = 64`, `a_panel = 64 · depth`, zero-padded to whole panels),
/// through `pack_a_trans_dispatch(.., mr = 64)`.
///
/// `None` (and `buf` untouched) when a span is empty or outside `op(A)`, or
/// when `buf` is shorter than [`a_pack_len`]`(rows.len, depth.len)`.
pub(super) fn pack_a<'p, T: SmeElement + Field>(
    a: &ColMajor<'_, T>,
    rows: Span,
    depth: Span,
    buf: &'p mut AlignedVec<T>,
) -> Option<PackedA<'p, T>> {
    if !inside(rows, a.rows()) || !inside(depth, a.cols()) {
        return None;
    }
    let len = a_pack_len(rows.len, depth.len)?;
    if buf.len() < len {
        return None;
    }
    pack_a_trans_dispatch(
        &a.mat,
        a.trans,
        rows.start,
        depth.start,
        rows.len,
        depth.len,
        buf,
        MR,
    );
    Some(PackedA {
        data: buf.as_slice().get(..len)?,
        rows: rows.len,
        depth: depth.len,
    })
}

/// Packs `op(B)[depth, cols]` into `buf` in the plain layout of `args.rs`
/// (`b_panel = depth · NR`, zero-padded to whole panels), through
/// `pack_b_trans_dispatch(.., nr = T::NR)`.
///
/// `None` (and `buf` untouched) when a span is empty or outside `op(B)`, or
/// when `buf` is shorter than [`b_pack_len`]`(depth.len, cols.len)`.
pub(super) fn pack_b<'p, T: SmeElement + Field>(
    b: &ColMajor<'_, T>,
    depth: Span,
    cols: Span,
    buf: &'p mut AlignedVec<T>,
) -> Option<PackedB<'p, T>> {
    if !inside(depth, b.rows()) || !inside(cols, b.cols()) {
        return None;
    }
    let len = b_pack_len::<T>(depth.len, cols.len)?;
    if buf.len() < len {
        return None;
    }
    pack_b_trans_dispatch(
        &b.mat,
        b.trans,
        depth.start,
        cols.start,
        depth.len,
        cols.len,
        buf,
        T::NR,
    );
    Some(PackedB {
        data: buf.as_slice().get(..len)?,
        depth: depth.len,
        cols: cols.len,
    })
}

/// The argument block of one kernel call: `C[row0.., col0..]` (of size
/// `a.rows() × b.cols()`) from the packed blocks `a` and `b`, with `alpha`
/// and the effective beta `beta` ([`block_beta`]); built by
/// `MacroArgs::packed`, so the mode follows alpha and beta and the Wave-4
/// fields are off. `ldc` is `c.row_stride()` (at least the block's rows).
///
/// `None` when the depths differ, when the block leaves `c`, or when C's
/// columns overlap (`row_stride` below the block's rows with more than one
/// column).
///
/// The result holds raw pointers into `a`, `b` and `c` without their
/// lifetimes: use it for one kernel call while all three borrows are alive,
/// and do not touch `c` through another path meanwhile. Under those terms
/// every address the kernel contract names is valid: the packed blocks are
/// whole panels, and the C block is inside `c`.
pub(super) fn block_args<T: SmeElement + Field>(
    a: &PackedA<'_, T>,
    b: &PackedB<'_, T>,
    c: &mut MatMut<'_, T>,
    row0: usize,
    col0: usize,
    alpha: T,
    beta: T,
) -> Option<MacroArgs<T>> {
    let (m, n, kc) = (a.rows, b.cols, a.depth);
    if b.depth != kc
        || !inside(
            Span {
                start: row0,
                len: m,
            },
            c.nrows(),
        )
        || !inside(
            Span {
                start: col0,
                len: n,
            },
            c.ncols(),
        )
    {
        return None;
    }
    let ldc = if n == 1 {
        c.row_stride().max(m)
    } else if c.row_stride() >= m {
        c.row_stride()
    } else {
        return None;
    };
    Some(MacroArgs::packed(
        a.data.as_ptr(),
        b.data.as_ptr(),
        c.ptr_at_mut(row0, col0),
        ldc,
        m,
        n,
        kc,
        alpha,
        beta,
    ))
}
