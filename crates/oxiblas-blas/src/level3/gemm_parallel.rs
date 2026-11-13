//! Two-dimensional (M × N) parallel driver for the blocked GEMM.
//!
//! The previous parallel driver split C over NC column blocks only. NC is
//! sized for the last-level cache (thousands of columns), so every problem
//! with `n <= NC` — including all square problems up to that size — formed a
//! single work item and ran on one thread.
//!
//! This driver keeps the BLIS loop order and splits each NC block of C into a
//! grid of tiles aligned to the micro-kernel grid:
//!
//! ```text
//! for jc in NC blocks of n                      (sequential)
//!     split C[:, jc..jc+w] into rows × cols tiles   (once per NC block)
//!     for pc in KC blocks of k                  (sequential, in order)
//!         phase 1: pack B~ = op(B)[pc.., jc..]  (parallel over NR panels, shared)
//!         phase 2: every tile, in parallel      (one task per tile)
//!             for ic in MC blocks of the tile
//!                 pack A~ into the worker's buffer
//!                 C_tile = alpha * A~ * B~ + beta_eff * C_tile
//! ```
//!
//! Tile boundaries are multiples of MR / NR and the KC partition is the
//! sequential one, so every MR × NR micro-tile sees exactly the same packed
//! panels and the same accumulation order as the sequential driver: results
//! are bitwise identical for any thread count. Each tile is a disjoint
//! [`MatMut`] split off C with the consuming `split_cols` / `split_rows`, so C
//! needs neither raw pointers nor synchronization.
//!
//! That is the [`PackScope::Shared`] scope. With [`PackScope::PerTile`] every
//! tile packs its own part of B as well, and the KC loop moves inside the
//! tile task (one fork/join per NC block instead of two per KC step):
//!
//! ```text
//! for jc in NC blocks of n                      (sequential)
//!     split C[:, jc..jc+w] into rows × cols tiles   (about 4 per thread)
//!     every tile, in parallel                   (one task per tile)
//!         for pc in KC blocks of k              (sequential, in order)
//!             pack B~ = op(B)[pc.., tile columns]   into the worker's buffer
//!             for ic in MC blocks of the tile
//!                 pack A~ into the worker's buffer
//!                 C_tile = alpha * A~ * B~ + beta_eff * C_tile
//! ```

use super::autotune::balanced_block;
use super::gemm::{
    GemmBlocking, macro_panel_multiply_with, op_dims, pack_a_trans_dispatch, pack_b_trans_dispatch,
    read_op,
};
use super::gemm_kernel::{GemmKernel, MicroKernel, MicroKernelShape};
use super::gemm_packing::{PackBuffer, PackSlot};
use super::gemm_plan::PackScope;
use super::trsm::Trans;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};
use rayon::prelude::*;

/// Target number of C tiles per worker thread.
///
/// Over-decomposition lets rayon's work stealing absorb uneven tile sizes and
/// heterogeneous cores (e.g. performance + efficiency clusters).
const TILES_PER_THREAD: usize = 3;

/// Target number of packed-B chunks per worker thread in phase 1.
const B_CHUNKS_PER_THREAD: usize = 2;

/// Target number of C tiles per worker thread under [`PackScope::PerTile`].
///
/// A per-tile task runs the whole KC loop of its tile, so a task is longer
/// than a shared-scope task and there is one fork/join per NC block instead
/// of two per KC step; one more tile per thread than the shared scope keeps
/// the work stealing able to balance the tail.
const PER_TILE_TILES_PER_THREAD: usize = 4;

/// Number of tile rows and tile columns for one NC block of C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TileGrid {
    rows: usize,
    cols: usize,
}

impl TileGrid {
    /// Chooses the tile grid for an `m × w` block of C on `threads` threads.
    ///
    /// With `mu = ⌈m / MR⌉` micro-rows, `nu = ⌈w / NR⌉` micro-columns and
    /// `G = max(1, min(3 · threads, mu · nu / 4))` target tiles, picks
    /// `rows ∈ 1..=min(G, mu)` with `cols = min(⌈G / rows⌉, nu)` minimising
    /// the largest tile (in micro-tiles) and then the number of tile columns
    /// (every extra tile column re-packs its rows of A).
    fn choose(m: usize, w: usize, shape: &MicroKernelShape, threads: usize) -> Self {
        let mu = m.div_ceil(shape.mr);
        let nu = w.div_ceil(shape.nr);
        let target = (TILES_PER_THREAD * threads).min(mu * nu / 4).max(1);
        let mut best = Self { rows: 1, cols: 1 };
        let mut best_key = (usize::MAX, usize::MAX);
        for rows in 1..=target.min(mu) {
            let cols = target.div_ceil(rows).min(nu);
            let key = (mu.div_ceil(rows) * nu.div_ceil(cols), cols);
            if key < best_key {
                best_key = key;
                best = Self { rows, cols };
            }
        }
        best
    }

    /// Chooses the tile grid of [`PackScope::PerTile`] for an `m × w` block
    /// of C on `threads` threads.
    ///
    /// With `mu = ⌈m / MR⌉`, `nu = ⌈w / NR⌉` and
    /// `G = max(1, min(4 · threads, mu · nu / 4))` target tiles, the
    /// candidates are `rows ∈ 1..=min(G, mu)` with `cols = min(⌈G / rows⌉,
    /// nu)`. Each has an estimated span: `⌈rows · cols / threads⌉` rounds of
    /// its largest tile (in micro-tiles). Among the candidates whose span is
    /// within 1/8 of the shortest, it picks, in this order:
    /// 1. the most nearly square tile (largest tile height over width, or
    ///    width over height, in elements);
    /// 2. the fewest elements packed per KC row, `rows · w + cols · m` (each
    ///    tile row packs all `w` columns of B, each tile column all `m` rows
    ///    of A);
    /// 3. the fewest tile columns.
    ///
    /// Ranking by the largest tile alone (as the shared grid does) prefers
    /// thin, over-decomposed grids, because `cols` rounds `G / rows` up: for
    /// f64 2048² on 8 threads (8 × 6 kernel) it picks 29 × 2 tiles of
    /// 72 × 1026; this rule picks 6 × 6 tiles of 344 × 342.
    fn choose_per_tile(m: usize, w: usize, shape: &MicroKernelShape, threads: usize) -> Self {
        let threads = threads.max(1);
        let mu = m.div_ceil(shape.mr);
        let nu = w.div_ceil(shape.nr);
        let target = (PER_TILE_TILES_PER_THREAD * threads)
            .min(mu * nu / 4)
            .max(1);
        let candidates = (1..=target.min(mu)).map(|rows| {
            let cols = target.div_ceil(rows).min(nu);
            let (tile_mu, tile_nu) = (mu.div_ceil(rows), nu.div_ceil(cols));
            let span = (rows * cols).div_ceil(threads) * tile_mu * tile_nu;
            (
                Self { rows, cols },
                span,
                tile_mu * shape.mr,
                tile_nu * shape.nr,
            )
        });
        let shortest = candidates
            .clone()
            .map(|(_, span, _, _)| span)
            .min()
            .unwrap_or(0);
        let mut best = Self { rows: 1, cols: 1 };
        let mut best_key = (usize::MAX, usize::MAX, usize::MAX);
        for (grid, span, height, width) in candidates {
            if span.saturating_mul(8) > shortest.saturating_mul(9) {
                continue;
            }
            let aspect = height.max(width).saturating_mul(1000) / height.min(width).max(1);
            let packed = grid
                .rows
                .saturating_mul(w)
                .saturating_add(grid.cols.saturating_mul(m));
            let key = (aspect, packed, grid.cols);
            if key < best_key {
                best_key = key;
                best = grid;
            }
        }
        best
    }

    /// Total number of tiles.
    fn count(self) -> usize {
        self.rows * self.cols
    }

    /// Height of the tallest tile row of an `m`-row block.
    fn tallest_rows(self, m: usize, mr: usize) -> usize {
        (m.div_ceil(mr).div_ceil(self.rows) * mr).min(m)
    }

    /// Width of the widest tile column of a `w`-column block.
    fn widest_cols(self, w: usize, nr: usize) -> usize {
        (w.div_ceil(nr).div_ceil(self.cols) * nr).min(w)
    }
}

/// Start of part `index` (`0..=parts`) when `len` elements — `⌈len / unit⌉`
/// grid units — are split into `parts` near-equal runs of whole units.
///
/// Every part is non-empty when `parts <= ⌈len / unit⌉`, and every start
/// except `len` itself is a multiple of `unit`.
fn split_point(index: usize, parts: usize, len: usize, unit: usize) -> usize {
    (index * len.div_ceil(unit) / parts * unit).min(len)
}

/// One tile of C.
struct Tile<'a, T: Field> {
    /// First row of the tile in C (and in `op(A)`).
    row0: usize,
    /// First column of the tile relative to its NC block (a multiple of NR).
    col0: usize,
    /// Exclusive view of the tile.
    view: MatMut<'a, T>,
}

/// Splits `block` (an `m × w` view of C) into the tiles of `grid`.
///
/// The consuming `split_cols` / `split_rows` make the tiles disjoint by
/// construction; a split point never equals the remaining length because no
/// tile is empty (`grid.rows <= ⌈m / MR⌉`, `grid.cols <= ⌈w / NR⌉`), the
/// remainder simply becomes the last tile.
fn split_tiles<'a, T: Field>(
    block: MatMut<'a, T>,
    grid: TileGrid,
    shape: &MicroKernelShape,
) -> Vec<Tile<'a, T>> {
    let (m, w) = (block.nrows(), block.ncols());
    let mut tiles = Vec::with_capacity(grid.count());
    let mut rest_cols = Some(block);
    for tc in 0..grid.cols {
        let Some(cols_view) = rest_cols.take() else {
            break;
        };
        let col0 = split_point(tc, grid.cols, w, shape.nr);
        let column = if tc + 1 == grid.cols {
            cols_view
        } else {
            let col1 = split_point(tc + 1, grid.cols, w, shape.nr);
            let (head, tail) = cols_view.split_cols(col1 - col0);
            rest_cols = Some(tail);
            head
        };

        let mut rest_rows = Some(column);
        for tr in 0..grid.rows {
            let Some(rows_view) = rest_rows.take() else {
                break;
            };
            let row0 = split_point(tr, grid.rows, m, shape.mr);
            let view = if tr + 1 == grid.rows {
                rows_view
            } else {
                let row1 = split_point(tr + 1, grid.rows, m, shape.mr);
                let (head, tail) = rows_view.split_rows(row1 - row0);
                rest_rows = Some(tail);
                head
            };
            tiles.push(Tile { row0, col0, view });
        }
    }
    tiles
}

/// Packs `op(B)[row0..row0 + pb, col0..col0 + w]` into `dst` as `⌈w / NR⌉`
/// panels of `pb × NR` (row-major inside a panel, panel stride `pb · NR`,
/// zero-padded past `w`) — the layout the micro-kernels consume — with the
/// panels split across tasks.
#[allow(clippy::too_many_arguments)]
fn pack_b_shared<T: Field>(
    b: &MatRef<'_, T>,
    trans_b: Trans,
    row0: usize,
    col0: usize,
    pb: usize,
    w: usize,
    dst: &mut [T],
    nr: usize,
    threads: usize,
) {
    let panels = w.div_ceil(nr);
    let panel_len = pb * nr;
    let per_chunk = panels.div_ceil(B_CHUNKS_PER_THREAD * threads).max(1);
    dst[..panels * panel_len]
        .par_chunks_mut(per_chunk * panel_len)
        .enumerate()
        .for_each(|(chunk, out)| {
            for (local, panel) in out.chunks_mut(panel_len).enumerate() {
                let j0 = (chunk * per_chunk + local) * nr;
                pack_b_panel(b, trans_b, row0, col0 + j0, pb, nr.min(w - j0), panel, nr);
            }
        });
}

/// Packs one `pb × NR` panel: columns `col0..col0 + cols` of
/// `op(B)[row0..row0 + pb, ..]`, zero-padding columns `cols..NR`.
///
/// Column-major `NoTrans` sources stream each column once; every other
/// operand (transposed modes, strided views) is read through [`read_op`],
/// which honours both strides and conjugates for `ConjTrans`.
#[allow(clippy::too_many_arguments)]
fn pack_b_panel<T: Field>(
    b: &MatRef<'_, T>,
    trans_b: Trans,
    row0: usize,
    col0: usize,
    pb: usize,
    cols: usize,
    dst: &mut [T],
    nr: usize,
) {
    if trans_b == Trans::NoTrans && b.col_stride() == 1 {
        for jj in 0..cols {
            let col = col0 + jj;
            match b.col_as_slice(col) {
                Some(column) => {
                    for (p, &value) in column[row0..row0 + pb].iter().enumerate() {
                        dst[p * nr + jj] = value;
                    }
                }
                None => {
                    for p in 0..pb {
                        dst[p * nr + jj] = read_op(b, trans_b, row0 + p, col);
                    }
                }
            }
        }
    } else {
        for p in 0..pb {
            for jj in 0..cols {
                dst[p * nr + jj] = read_op(b, trans_b, row0 + p, col0 + jj);
            }
        }
    }
    if cols < nr {
        for p in 0..pb {
            dst[p * nr + cols..(p + 1) * nr].fill(T::zero());
        }
    }
}

/// Runs the blocked GEMM `C = alpha * op(A) * op(B) + beta * C` on the 2D tile
/// grid described in the module docs.
///
/// `blocking` must already be fitted to the problem (the GEMM funnel does
/// that), and `alpha == 0` must have been handled by the caller. `kernel` is
/// the micro-kernel of the call's plan: A and B are packed with its shape and
/// every tile runs it (never a re-dispatched one). `scope` is the plan's pack
/// scope: [`run_shared`] or [`run_per_tile`]; both give the bits of the
/// sequential driver. Returns `false` without touching C when the widest NC
/// block cannot be split into at least two tiles; the caller then runs the
/// sequential driver.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_gemm_blocked_parallel<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    kernel: &MicroKernel<T>,
    scope: PackScope,
    par: Par,
) -> bool {
    match scope {
        PackScope::Shared => run_shared(
            trans_a, trans_b, alpha, a, b, beta, c, blocking, kernel, par,
        ),
        PackScope::PerTile => run_per_tile(
            trans_a, trans_b, alpha, a, b, beta, c, blocking, kernel, par,
        ),
    }
}

#[cfg(test)]
thread_local! {
    /// Calls the 2D driver ran on this thread, per runner: `(Shared,
    /// PerTile)` (test hook; each runner counts itself once it has run, on
    /// the calling thread).
    static SCOPE_RUNS: core::cell::Cell<(usize, usize)> = const { core::cell::Cell::new((0, 0)) };
}

/// Counts one engaged run of `scope` on this thread.
#[cfg(test)]
fn count_scope_run(scope: PackScope) {
    let _ = SCOPE_RUNS.try_with(|runs| {
        let (shared, per_tile) = runs.get();
        runs.set(match scope {
            PackScope::Shared => (shared + 1, per_tile),
            PackScope::PerTile => (shared, per_tile + 1),
        });
    });
}

/// `(Shared, PerTile)`: how many calls the 2D driver ran on this thread with
/// each scope (test hook).
#[cfg(test)]
pub(crate) fn scope_runs() -> (usize, usize) {
    SCOPE_RUNS.try_with(core::cell::Cell::get).unwrap_or((0, 0))
}

/// [`try_gemm_blocked_parallel`] with [`PackScope::Shared`]: one packed B~
/// per `(jc, pc)` step, packed in parallel and shared by every tile.
#[allow(clippy::too_many_arguments)]
fn run_shared<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    kernel: &MicroKernel<T>,
    par: Par,
) -> bool {
    // Logical (post-transpose) dimensions.
    let (m, k) = op_dims(a, trans_a);
    let (_, n) = op_dims(b, trans_b);
    if m == 0 || n == 0 || k == 0 {
        return false;
    }

    let shape = &kernel.shape;
    let mr = shape.mr;
    let nr = shape.nr;
    let mc = blocking.mc.min(m);
    let kc = blocking.kc.min(k);
    let nc = blocking.nc.min(n);
    let threads = par.num_threads();

    // Grids of the full-width NC blocks and of the (possibly narrower) last.
    let full_grid = TileGrid::choose(m, nc, shape, threads);
    if full_grid.count() < 2 {
        return false;
    }
    let last_w = n - (n - 1) / nc * nc;
    let last_grid = TileGrid::choose(m, last_w, shape, threads);

    // Each tile packs at most `round_up(min(mc, tallest tile), MR) x KC` of A.
    let max_tile_rows = full_grid
        .tallest_rows(m, mr)
        .max(last_grid.tallest_rows(m, mr));
    let a_len = mc.min(max_tile_rows).div_ceil(mr) * mr * kc;

    // Shared B~ of one (jc, pc) step, reused for every step, on loan from the
    // calling thread's packing cache. Each tile task borrows its A~ buffer from
    // the cache of the worker that runs it (moved out of the thread-local slot
    // for the task, so a task stolen by a worker that is itself waiting inside
    // another GEMM just finds the slot empty and allocates). Workers keep their
    // A~ buffer between tasks and calls instead of every call zero-filling one
    // buffer per worker (policy: `PACK_CACHE_MAX_BYTES`).
    let mut pack_b = PackBuffer::<T>::take(PackSlot::B, nc.div_ceil(nr) * nr * kc);

    for jc in (0..n).step_by(nc) {
        let w = nc.min(n - jc);
        let grid = if w == nc { full_grid } else { last_grid };
        let mut tiles = split_tiles(c.rb_mut().submatrix(0, jc, m, w), grid, shape);

        for pc in (0..k).step_by(kc) {
            let pb = kc.min(k - pc);
            // beta applies on the first KC step only; later steps accumulate.
            let beta_eff = if pc == 0 { beta } else { T::one() };

            // Phase 1: pack B~ = op(B)[pc..pc+pb, jc..jc+w], shared by all tiles.
            pack_b_shared(b, trans_b, pc, jc, pb, w, &mut pack_b, nr, threads);
            let packed_b: &[T] = &pack_b;

            // Phase 2: every tile updates its own part of C.
            tiles.par_iter_mut().with_max_len(1).for_each(|tile| {
                let mut a_buf = PackBuffer::<T>::take(PackSlot::A, a_len);
                let rows = tile.view.nrows();
                let cols = tile.view.ncols();
                let b_panels = &packed_b[tile.col0 / nr * pb * nr..];
                let mc_tile = balanced_block(rows, mc, mr);
                for ic in (0..rows).step_by(mc_tile) {
                    let ib = mc_tile.min(rows - ic);
                    pack_a_trans_dispatch(a, trans_a, tile.row0 + ic, pc, ib, pb, &mut a_buf, mr);
                    macro_panel_multiply_with(
                        alpha,
                        &a_buf,
                        ib,
                        pb,
                        b_panels,
                        cols,
                        beta_eff,
                        tile.view.rb_mut().submatrix(ic, 0, ib, cols),
                        kernel,
                    );
                }
            });
        }
    }
    #[cfg(test)]
    count_scope_run(PackScope::Shared);
    true
}

/// [`try_gemm_blocked_parallel`] with [`PackScope::PerTile`]: every tile
/// packs its own A~ and B~ into its worker's buffers and runs the whole KC
/// loop of the tile, with one fork/join per NC block.
///
/// # Why the result is bitwise the sequential driver's
///
/// An element of C is computed by one MR × NR micro-tile per KC block, and
/// its bits depend only on the values the micro-kernel reads (the packed A
/// row panel and B column panel of that KC block), the kernel and its
/// `beta_eff`, and the order of the KC blocks:
/// - Tile origins are multiples of MR and NR ([`split_point`] over the
///   micro-grid), and the MC sub-blocks inside a tile
///   (`balanced_block(rows, mc, mr)`) are multiples of MR or the whole tile,
///   so every micro-tile starts at the same `(row, column)` as in the
///   sequential driver and is padded only where C ends, as there. The
///   packers copy values (`ConjTrans` conjugates, as in the sequential
///   driver, which uses the same [`pack_a_trans_dispatch`] /
///   [`pack_b_trans_dispatch`]), so the panels hold the same values.
/// - The KC partition is the sequential one (`(0..k).step_by(kc)` with the
///   same `kc`), walked in order inside each tile.
/// - `beta` is applied exactly once, at the first KC block (`pc == 0`); every
///   later block accumulates with `beta_eff = 1`, as in the sequential
///   driver. Each tile writes only its own disjoint view of C, so no element
///   is touched by two tasks.
///
/// Neither MC nor NC changes a result bit (they only regroup micro-tiles),
/// so the tile grid and the thread count do not either.
#[allow(clippy::too_many_arguments)]
fn run_per_tile<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    kernel: &MicroKernel<T>,
    par: Par,
) -> bool {
    // Logical (post-transpose) dimensions.
    let (m, k) = op_dims(a, trans_a);
    let (_, n) = op_dims(b, trans_b);
    if m == 0 || n == 0 || k == 0 {
        return false;
    }

    let shape = &kernel.shape;
    let mr = shape.mr;
    let nr = shape.nr;
    let mc = blocking.mc.min(m);
    let kc = blocking.kc.min(k);
    let nc = blocking.nc.min(n);
    let threads = par.num_threads();

    // Grids of the full-width NC blocks and of the (possibly narrower) last.
    let full_grid = TileGrid::choose_per_tile(m, nc, shape, threads);
    if full_grid.count() < 2 {
        return false;
    }
    let last_w = n - (n - 1) / nc * nc;
    let last_grid = TileGrid::choose_per_tile(m, last_w, shape, threads);

    // Each tile packs at most `round_up(min(mc, tallest tile), MR) x KC` of A
    // and `round_up(widest tile, NR) x KC` of B.
    let max_tile_rows = full_grid
        .tallest_rows(m, mr)
        .max(last_grid.tallest_rows(m, mr));
    let max_tile_cols = full_grid
        .widest_cols(nc, nr)
        .max(last_grid.widest_cols(last_w, nr));
    let a_len = mc.min(max_tile_rows).div_ceil(mr) * mr * kc;
    let b_len = max_tile_cols.div_ceil(nr) * nr * kc;

    for jc in (0..n).step_by(nc) {
        let w = nc.min(n - jc);
        let grid = if w == nc { full_grid } else { last_grid };
        let mut tiles = split_tiles(c.rb_mut().submatrix(0, jc, m, w), grid, shape);

        tiles.par_iter_mut().with_max_len(1).for_each(|tile| {
            // Both buffers come from the cache of the worker that runs the
            // task (moved out of its thread-local slots for the task, so a
            // task stolen by a worker that waits inside another GEMM finds the
            // slots empty and allocates). Workers keep them between tasks and
            // calls (policy: `PACK_CACHE_MAX_BYTES`).
            let mut a_buf = PackBuffer::<T>::take(PackSlot::A, a_len);
            let mut b_buf = PackBuffer::<T>::take(PackSlot::B, b_len);
            let rows = tile.view.nrows();
            let cols = tile.view.ncols();
            let mc_tile = balanced_block(rows, mc, mr);
            for pc in (0..k).step_by(kc) {
                let pb = kc.min(k - pc);
                // beta applies on the first KC step only; later steps accumulate.
                let beta_eff = if pc == 0 { beta } else { T::one() };
                pack_b_trans_dispatch(b, trans_b, pc, jc + tile.col0, pb, cols, &mut b_buf, nr);
                for ic in (0..rows).step_by(mc_tile) {
                    let ib = mc_tile.min(rows - ic);
                    pack_a_trans_dispatch(a, trans_a, tile.row0 + ic, pc, ib, pb, &mut a_buf, mr);
                    macro_panel_multiply_with(
                        alpha,
                        &a_buf,
                        ib,
                        pb,
                        &b_buf,
                        cols,
                        beta_eff,
                        tile.view.rb_mut().submatrix(ic, 0, ib, cols),
                        kernel,
                    );
                }
            }
        });
    }
    #[cfg(test)]
    count_scope_run(PackScope::PerTile);
    true
}

#[cfg(test)]
#[path = "gemm_parallel_tests.rs"]
mod tests;
