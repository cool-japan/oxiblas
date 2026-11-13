//! Triangle-only blocked GEMM: the shared core of SYRK, SYR2K and GEMMT.
//!
//! [`tri_update`] computes
//!
//! ```text
//! C := beta·C + alpha·Σₛ op(Aₛ)·op(Bₛ)      (only the `uplo` triangle of C)
//! ```
//!
//! for an `n × n` matrix C and a short list of *k-segments*: segment `s`
//! contributes an `n × kₛ` operand `op(Aₛ)` and a `kₛ × n` operand `op(Bₛ)`.
//!
//! - SYRK is one segment, `(A, Aᵀ)` (`NoTrans`) or `(Aᵀ, A)` (`Trans`).
//! - SYR2K is `alpha·[A B]·[B A]ᵀ`: the two segments `(A, Bᵀ)` and `(B, Aᵀ)`
//!   (or `(Aᵀ, B)` and `(Bᵀ, A)`) of one call with `k' = 2k`. The
//!   concatenations are never materialized: every k-block lies inside one
//!   segment and is packed straight from the matrices of that segment (so
//!   each segment takes at least one sweep over the triangle of C).
//! - GEMMT is one segment, `(op(A), op(B))`.
//!
//! # Algorithm
//!
//! The loops, blocking, packers and micro-kernel are those of the blocked GEMM
//! (`gemm_blocked_sequential`): for every KC block of a segment, every NC block
//! of columns `[j0, j0 + jb)` and every MC block of rows, `op(B)` and `op(A)`
//! are packed with the GEMM packers (transposed operands are packed in place
//! through their transposed fast paths) and the micro-kernel runs on `MR × NR`
//! tiles. The triangle restricts every level:
//!
//! - per NC block only the rows that meet the triangle are packed and visited
//!   (`Lower`: rows `>= j0`; `Upper`: rows `< j0 + jb`);
//! - per MC block only the column micro-panels that meet the triangle;
//! - per column micro-panel only the row micro-tiles that meet the triangle.
//!
//! A micro-tile entirely inside the triangle is computed in place in C. A tile
//! that straddles the diagonal, or is cut by the edge of C, is computed in an
//! `MR × NR` scratch tile, and only its entries inside the triangle are copied
//! back: the other triangle of C is never written (it stays bitwise
//! untouched) and never read.
//!
//! # Scalars
//!
//! `beta` is applied to the triangle once, before any product accumulates:
//! `beta == 1` leaves it alone and any other non-zero value scales it; every
//! k-block then accumulates into C with the micro-kernel's `beta = 1`.
//! `beta == 0` is folded into the first k-block instead, which stores its
//! products into every entry of the triangle with the micro-kernel's
//! `beta = 0`: C is never read (NaN or Inf there cannot propagate) and no
//! separate zero-fill pass is made. `alpha == 0` (or `k == 0`) only applies
//! beta (zero-filling the triangle for `beta == 0`) and never reads A or B, as
//! in reference BLAS.
//!
//! Complex element types run every tile through the scratch tile with the
//! micro-kernel's `beta = 0` and then add the tile to C (or store it, in the
//! first k-block when `beta == 0`). Their micro-kernels form `beta·C` for
//! `beta = 1` as the complex product `(1, 0)·C`, whose `0·∞` makes NaN of
//! the other component of an infinite entry of C; reference C/ZSYRK only ever
//! adds to C.
//! For real types `1·x` is exact, so full tiles use the in-place kernel.
//!
//! # Rounding
//!
//! The blocking is chosen exactly as GEMM chooses it for an `n × k` by
//! `k × n` product, so every entry is accumulated in the same order as by
//! `gemm_transposed`. With `beta == 0` or `beta == 1` (and `n²k` above GEMM's
//! small-matrix threshold) the triangle is bitwise equal to the corresponding
//! triangle of `gemm_transposed(…, beta, …)`; for complex types with
//! `beta == 1` this holds for finite C up to the sign of zero components,
//! which GEMM's `(1, 0)·C` product can flip. With any other
//! beta, `beta·C` is rounded once on its own and the k-blocks are added to it,
//! whereas GEMM fuses `beta·C` into the first k-block's store, so results can
//! differ in the last bit.

use crate::level3::autotune::{KC_BLOCK_UNIT, balanced_block};
use crate::level3::gemm::{
    GemmBlocking, default_blocking, op_dims, pack_a_trans_dispatch, pack_b_trans_dispatch,
};
use crate::level3::gemm_kernel::{GemmKernel, MicroKernel, MicroKernelShape};
use crate::level3::gemm_packing::{PackBuffer, PackSlot};
use crate::level3::trsm::{Trans, Uplo};
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{MatMut, MatRef};

/// Largest `MR × NR` of the micro-kernels in this crate (AVX-512 f32 is
/// 16 × 16); larger shapes would use a heap scratch tile.
const MAX_TILE: usize = 16 * 16;

/// One k-segment of a triangle update: `op(A)` is `n × kₛ` and `op(B)` is
/// `kₛ × n`.
#[derive(Clone, Copy)]
pub(crate) struct KSegment<'a, T: Scalar> {
    /// Stored left operand.
    pub(crate) a: MatRef<'a, T>,
    /// How `a` enters the product.
    pub(crate) trans_a: Trans,
    /// Stored right operand.
    pub(crate) b: MatRef<'a, T>,
    /// How `b` enters the product.
    pub(crate) trans_b: Trans,
}

impl<'a, T: Scalar> KSegment<'a, T> {
    /// The segment `op(a)·op(b)`.
    pub(crate) fn new(a: MatRef<'a, T>, trans_a: Trans, b: MatRef<'a, T>, trans_b: Trans) -> Self {
        Self {
            a,
            trans_a,
            b,
            trans_b,
        }
    }

    /// Inner dimension `kₛ` of the segment.
    fn k(&self) -> usize {
        op_dims(&self.a, self.trans_a).1
    }
}

/// `C := beta·C + alpha·Σₛ op(Aₛ)·op(Bₛ)` on the `uplo` triangle of C, with
/// GEMM's blocking (see the [module documentation](self)).
///
/// # Panics
///
/// Panics if C is not square or a segment's shapes do not match it (callers
/// validate their arguments first).
pub(crate) fn tri_update<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    alpha: T,
    segments: &[KSegment<'_, T>],
    beta: T,
    c: MatMut<'_, T>,
) {
    tri_update_impl(uplo, alpha, segments, beta, c, None);
}

/// [`tri_update`] with explicit blocking targets (fitted to the problem as
/// GEMM fits them); lets tests force many MC / NC / KC blocks on small
/// matrices.
#[cfg(test)]
pub(crate) fn tri_update_with_blocking<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    alpha: T,
    segments: &[KSegment<'_, T>],
    beta: T,
    c: MatMut<'_, T>,
    blocking: &GemmBlocking,
) {
    tri_update_impl(uplo, alpha, segments, beta, c, Some(blocking));
}

fn tri_update_impl<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    alpha: T,
    segments: &[KSegment<'_, T>],
    beta: T,
    mut c: MatMut<'_, T>,
    blocking: Option<&GemmBlocking>,
) {
    let n = c.nrows();
    assert_eq!(c.ncols(), n, "tri_update: C must be square");
    for seg in segments {
        let (a_rows, a_k) = op_dims(&seg.a, seg.trans_a);
        let (b_k, b_cols) = op_dims(&seg.b, seg.trans_b);
        assert!(
            a_rows == n && b_cols == n && a_k == b_k,
            "tri_update: segment shapes do not match C"
        );
    }
    if n == 0 {
        return;
    }

    let k_max = segments.iter().map(KSegment::k).max().unwrap_or(0);
    let forms_product = alpha != T::zero() && k_max > 0;
    // With a product to form, `beta == 0` is folded into the first k-block,
    // which overwrites every entry of the triangle without reading C; any
    // other beta is applied here, once, before the products accumulate.
    let overwrite_first = forms_product && beta == T::zero();
    if !overwrite_first {
        scale_triangle(uplo, beta, &mut c);
    }
    if !forms_product {
        // C := beta·C only; A and B are never read.
        return;
    }

    // One dispatch decision for the whole call: the packed layout below uses
    // the shape of exactly the kernel that consumes it. It is the canonical
    // kernel, and the blocking is GEMM's default blocking for it, so the KC
    // partition (and with it the accumulation order) is the one of GEMM
    // whatever micro-kernel a GEMM plan carries.
    let kernel = T::micro_kernel_dispatch();
    let targets = match blocking {
        Some(b) => *b,
        None => default_blocking::<T>(n, k_max, n, &kernel.shape),
    };
    tri_blocked(
        uplo,
        alpha,
        segments,
        &mut c,
        &targets,
        &kernel,
        overwrite_first,
    );
}

/// Rows `[lo, hi)` of column `j` that lie in the `uplo` triangle of an
/// `n × n` matrix.
#[inline]
fn tri_rows(uplo: Uplo, j: usize, n: usize) -> (usize, usize) {
    match uplo {
        Uplo::Lower => (j.min(n), n),
        Uplo::Upper => (0, (j + 1).min(n)),
    }
}

/// `C := beta·C` on the `uplo` triangle of the square matrix C.
///
/// `beta == 1` leaves C untouched (bitwise) and `beta == 0` zero-fills the
/// triangle without reading it, so NaN or Inf there cannot propagate. The
/// other triangle is neither read nor written.
pub(crate) fn scale_triangle<T: Field>(uplo: Uplo, beta: T, c: &mut MatMut<'_, T>) {
    if beta == T::one() {
        return;
    }
    let n = c.nrows();
    let beta_is_zero = beta == T::zero();
    for j in 0..c.ncols() {
        let (lo, hi) = tri_rows(uplo, j, n);
        if let Some(col) = c.col_as_slice_mut(j) {
            let part = &mut col[lo..hi];
            if beta_is_zero {
                part.fill(T::zero());
            } else {
                for x in part.iter_mut() {
                    *x = beta * *x;
                }
            }
        }
    }
}

/// The blocked loops: KC blocks of every segment, NC blocks of columns, MC
/// blocks of the rows that meet the triangle.
///
/// With `overwrite_first` the first k-block overwrites the triangle (the
/// micro-kernel's `beta = 0`, C is not read); every other k-block adds to it.
#[allow(clippy::too_many_arguments)]
fn tri_blocked<T: Field + GemmKernel + bytemuck::Zeroable>(
    uplo: Uplo,
    alpha: T,
    segments: &[KSegment<'_, T>],
    c: &mut MatMut<'_, T>,
    targets: &GemmBlocking,
    kernel: &MicroKernel<T>,
    overwrite_first: bool,
) {
    let n = c.nrows();
    let MicroKernelShape { mr, nr } = kernel.shape;

    // Block sizes fitted to the problem as GEMM fits them (`fit_to_dims`).
    let mc = balanced_block(n, targets.mc, mr);
    let nc = balanced_block(n, targets.nc, nr);
    let kc_of = |k: usize| balanced_block(k, targets.kc, KC_BLOCK_UNIT);
    let kc_max = segments.iter().map(|s| kc_of(s.k())).max().unwrap_or(1);

    // Packing buffers padded to whole micro-panels, on loan from this
    // thread's GEMM cache (no allocation or zero-fill on reuse).
    let padded_mc = mc.div_ceil(mr) * mr;
    let padded_nc = nc.div_ceil(nr) * nr;
    let mut pack_a = PackBuffer::<T>::take(PackSlot::A, padded_mc * kc_max);
    let mut pack_b = PackBuffer::<T>::take(PackSlot::B, kc_max * padded_nc);
    let mut scratch = Scratch::<T>::new(mr * nr);
    let mut overwrite = overwrite_first;

    for seg in segments {
        let k = seg.k();
        if k == 0 {
            continue;
        }
        let kc = kc_of(k);
        for p in (0..k).step_by(kc) {
            let pb = kc.min(k - p);
            for j0 in (0..n).step_by(nc) {
                let jb = nc.min(n - j0);
                // Rows of C that meet the triangle in columns [j0, j0 + jb).
                let (row_lo, row_hi) = match uplo {
                    Uplo::Lower => (j0, n),
                    Uplo::Upper => (0, j0 + jb),
                };

                // op(B)[p..p+pb, j0..j0+jb] -> pack_b (transpose-aware).
                pack_b_trans_dispatch(&seg.b, seg.trans_b, p, j0, pb, jb, &mut pack_b, nr);

                for i0 in (row_lo..row_hi).step_by(mc) {
                    let ib = mc.min(row_hi - i0);
                    // op(A)[i0..i0+ib, p..p+pb] -> pack_a (transpose-aware).
                    pack_a_trans_dispatch(&seg.a, seg.trans_a, i0, p, ib, pb, &mut pack_a, mr);
                    tri_macro_kernel(
                        uplo,
                        alpha,
                        overwrite,
                        Panels {
                            a: &pack_a,
                            b: &pack_b,
                            i0,
                            ib,
                            j0,
                            jb,
                            kb: pb,
                        },
                        c,
                        kernel,
                        &mut scratch,
                    );
                }
            }
            // Every entry of the triangle was written by this k-block.
            overwrite = false;
        }
    }
}

/// The packed `op(A)` rows `[i0, i0 + ib)` and `op(B)` columns
/// `[j0, j0 + jb)` of one KC block of depth `kb`.
struct Panels<'p, T> {
    a: &'p [T],
    b: &'p [T],
    i0: usize,
    ib: usize,
    j0: usize,
    jb: usize,
    kb: usize,
}

/// One `MR × NR` scratch tile (leading dimension `MR`), zero-initialized on
/// first use. Entries outside the live part of a tile may hold values of an
/// earlier tile; the micro-kernel may read them, but they are never copied
/// to C.
struct Scratch<T> {
    stack: Option<[T; MAX_TILE]>,
    heap: Vec<T>,
    len: usize,
}

impl<T: Field> Scratch<T> {
    fn new(len: usize) -> Self {
        Self {
            stack: None,
            heap: Vec::new(),
            len,
        }
    }

    /// The tile: exactly `MR · NR` initialized elements.
    #[inline]
    fn tile(&mut self) -> &mut [T] {
        let len = self.len;
        if len <= MAX_TILE {
            &mut self.stack.get_or_insert([T::zero(); MAX_TILE])[..len]
        } else {
            if self.heap.len() < len {
                self.heap = vec![T::zero(); len];
            }
            &mut self.heap[..len]
        }
    }
}

/// Rows `[lo, hi)` of the tile with top row `gi` (and `im` live rows) whose
/// entries in global column `col` lie in the `uplo` triangle.
#[inline]
fn live_rows(uplo: Uplo, gi: usize, col: usize, im: usize) -> (usize, usize) {
    match uplo {
        // row >= col
        Uplo::Lower => (col.saturating_sub(gi).min(im), im),
        // row <= col
        Uplo::Upper => {
            if col < gi {
                (0, 0)
            } else {
                (0, (col - gi + 1).min(im))
            }
        }
    }
}

/// Runs the micro-kernel over the tiles of `C[i0..i0+ib, j0..j0+jb]` that
/// meet the `uplo` triangle, adding `alpha·op(A)·op(B)` of one KC block to C
/// (or, with `overwrite`, storing it without reading C).
///
/// # Panics
///
/// Panics if the packed panels are shorter than the block needs or the block
/// does not lie inside C.
fn tri_macro_kernel<T: Field + GemmKernel>(
    uplo: Uplo,
    alpha: T,
    overwrite: bool,
    panels: Panels<'_, T>,
    c: &mut MatMut<'_, T>,
    kernel: &MicroKernel<T>,
    scratch: &mut Scratch<T>,
) {
    let Panels {
        a: pack_a,
        b: pack_b,
        i0,
        ib,
        j0,
        jb,
        kb,
    } = panels;
    if ib == 0 || jb == 0 {
        return;
    }
    let MicroKernelShape { mr, nr } = kernel.shape;
    let micro = kernel.kernel;
    let m_blocks = ib.div_ceil(mr);
    let n_blocks = jb.div_ceil(nr);
    let a_panel = mr * kb;
    let b_panel = nr * kb;

    // Every micro-panel the kernel reads lies inside the packed buffers, and
    // every entry written lies inside C.
    assert!(
        pack_a.len() >= m_blocks * a_panel && pack_b.len() >= n_blocks * b_panel,
        "tri_macro_kernel: packed panels shorter than the {ib} x {jb} x {kb} block"
    );
    assert!(
        i0 + ib <= c.nrows() && j0 + jb <= c.ncols(),
        "tri_macro_kernel: block outside C"
    );

    let ldc = c.row_stride();
    let c_base = c.as_mut_ptr();
    // Real kernels add C for beta == 1 without multiplying it, so full tiles
    // inside the triangle can run in place; complex ones always use scratch.
    let in_place = T::is_real();
    // The micro-kernel's beta for tiles computed in place (or, for real
    // types, in scratch): 0 overwrites, 1 adds the existing C.
    let beta_k = if overwrite { T::zero() } else { T::one() };
    let i_last = i0 + ib - 1;

    // Column micro-panels with at least one column meeting the triangle in
    // rows [i0, i_last].
    let (jr_lo, jr_hi) = match uplo {
        // Columns <= i_last.
        Uplo::Lower => {
            if i_last < j0 {
                (0, 0)
            } else {
                (0, ((i_last - j0) / nr + 1).min(n_blocks))
            }
        }
        // Columns >= i0.
        Uplo::Upper => {
            let lo = if i0 > j0 { (i0 - j0) / nr } else { 0 };
            (lo.min(n_blocks), n_blocks)
        }
    };

    for jr in jr_lo..jr_hi {
        let gj = j0 + jr * nr;
        let jn = nr.min(jb - jr * nr);
        let g_last = gj + jn - 1;
        // B micro-panel for this column block (inside `pack_b` by the assert).
        let b_ptr = pack_b.as_ptr().wrapping_add(jr * b_panel);

        // Row micro-tiles meeting the triangle in columns [gj, g_last].
        let (ir_lo, ir_hi) = match uplo {
            // The first tile is the one holding row gj.
            Uplo::Lower => {
                let lo = if gj > i0 { (gj - i0) / mr } else { 0 };
                (lo.min(m_blocks), m_blocks)
            }
            // The last tile is the one holding row g_last.
            Uplo::Upper => {
                if g_last < i0 {
                    (0, 0)
                } else {
                    (0, ((g_last - i0) / mr + 1).min(m_blocks))
                }
            }
        };

        for ir in ir_lo..ir_hi {
            let gi = i0 + ir * mr;
            let im = mr.min(ib - ir * mr);
            // A micro-panel for this row block (inside `pack_a` by the assert).
            let a_ptr = pack_a.as_ptr().wrapping_add(ir * a_panel);
            let inside = match uplo {
                Uplo::Lower => gi >= g_last,
                Uplo::Upper => gi + im - 1 <= gj,
            };

            if in_place && inside && im == mr && jn == nr {
                // SAFETY: the full tile `C[gi..gi+MR, gj..gj+NR]` lies inside
                // the block, which lies inside C (assert above), so every
                // `c_base + (gi + ii) + (gj + jj) * ldc` the kernel touches is
                // an element of C, whose exclusive borrow we hold; all of it
                // is in the triangle (`inside`). The packed panels hold
                // `MR * kb` / `kb * NR` elements at `a_ptr` / `b_ptr` (assert
                // above) in the layout of `kernel`, whose shape they were
                // packed with.
                unsafe {
                    micro(
                        kb,
                        alpha,
                        a_ptr,
                        b_ptr,
                        beta_k,
                        c_base.add(gi + gj * ldc),
                        ldc,
                    );
                }
                continue;
            }

            // Straddling, edge or complex tile: compute in scratch, then copy
            // back (real) or add (complex) only the live entries.
            let tile = scratch.tile();
            // SAFETY: (all raw C accesses below) `ii < im <= ib - ir * mr` and
            // `jj < jn <= jb - jr * nr`, so `(gi + ii, gj + jj)` lies inside
            // the block and therefore inside C (assert above); `live_rows`
            // keeps it inside the triangle. `tile` holds `MR * NR` initialized
            // elements with leading dimension `mr`, which is what the kernel
            // reads and writes; the packed panels are as for a full tile
            // (zero-padded past `im` / `jn`).
            unsafe {
                let c_tile = c_base.add(gi + gj * ldc);
                if in_place {
                    if !overwrite {
                        for jj in 0..jn {
                            let (lo, hi) = live_rows(uplo, gi, gj + jj, im);
                            for ii in lo..hi {
                                tile[ii + jj * mr] = *c_tile.add(ii + jj * ldc);
                            }
                        }
                    }
                    micro(kb, alpha, a_ptr, b_ptr, beta_k, tile.as_mut_ptr(), mr);
                    for jj in 0..jn {
                        let (lo, hi) = live_rows(uplo, gi, gj + jj, im);
                        for ii in lo..hi {
                            *c_tile.add(ii + jj * ldc) = tile[ii + jj * mr];
                        }
                    }
                } else {
                    micro(kb, alpha, a_ptr, b_ptr, T::zero(), tile.as_mut_ptr(), mr);
                    if overwrite {
                        for jj in 0..jn {
                            let (lo, hi) = live_rows(uplo, gi, gj + jj, im);
                            for ii in lo..hi {
                                *c_tile.add(ii + jj * ldc) = tile[ii + jj * mr];
                            }
                        }
                    } else {
                        for jj in 0..jn {
                            let (lo, hi) = live_rows(uplo, gi, gj + jj, im);
                            for ii in lo..hi {
                                let dst = c_tile.add(ii + jj * ldc);
                                *dst = *dst + tile[ii + jj * mr];
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "tri_gemm_tests.rs"]
pub(crate) mod tests;
