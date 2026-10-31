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

use super::autotune::balanced_block;
use super::gemm::{
    GemmBlocking, macro_panel_multiply, op_dims, pack_a_trans_dispatch, pack_b_trans_dispatch,
    read_op,
};
use super::gemm_kernel::{GemmKernel, MicroKernelShape};
use super::gemm_pack_pool;
use super::trsm::Trans;
use oxiblas_core::memory::AlignedVec;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};
use rayon::prelude::*;
use std::sync::{Mutex, PoisonError};

/// Target number of C tiles per worker thread.
///
/// Over-decomposition lets rayon's work stealing absorb uneven tile sizes and
/// heterogeneous cores (e.g. performance + efficiency clusters).
const TILES_PER_THREAD: usize = 3;

/// Target number of packed-B chunks per worker thread in phase 1.
const B_CHUNKS_PER_THREAD: usize = 2;

/// Target number of C tiles per worker thread with [`PackScope::PerTile`].
///
/// Each tile packs its own panels, so tiles are kept near-square (that
/// minimises the re-packed `(rows + cols) * k` per tile) and a little more
/// numerous than in the shared mode, for work stealing. Measured on a
/// Threadripper PRO 3975WX (f64, n = 4096, 32 threads, prototype of this
/// driver): 1, 2, 3 and 4 tiles per thread gave 837, 873, 938 and 975
/// GFLOP/s.
const PER_TILE_TILES_PER_THREAD: usize = 4;

/// Who packs `op(B)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PackScope {
    /// One B~ per KC step, packed cooperatively and read by every tile; two
    /// fork/joins per KC step.
    Shared,
    /// Every tile runs the whole KC loop on its own, packing its own slices
    /// of `op(A)` and `op(B)` into buffers from the worker's own pool; one
    /// fork/join per NC block.
    ///
    /// On multi-die x86 parts (AMD Zen: one L3 per 4- or 8-core CCX) a shared
    /// B~ is packed by threads on every die and then read by every die
    /// across the fabric. On a Threadripper PRO 3975WX (f64, n = 4096, 32
    /// threads) the shared driver reached 582 GFLOP/s and this one 979, with
    /// OpenBLAS 0.3.29 at 1318.
    PerTile,
}

impl PackScope {
    /// Threads up to which `PerTile` is not used on `x86_64`.
    ///
    /// Up to a few threads the shared B~ stays within one L3 (an AMD Zen 2
    /// CCX has 4 cores) and packing it once wins. Threadripper PRO 3975WX,
    /// median GFLOP/s, PerTile / Shared: 2 threads 0.78-0.93 (f64 n = 2048:
    /// 96 vs 110), 4 threads 0.84-1.29, 8 threads 0.97-2.25, 16 threads
    /// 1.00-1.69.
    const SHARED_MAX_THREADS: usize = 4;

    /// The scope used by [`try_gemm_blocked_parallel`] for `threads` threads.
    ///
    /// `PerTile` on `x86_64` above [`Self::SHARED_MAX_THREADS`], where it was
    /// measured; `Shared` otherwise and on other targets, where the shared
    /// driver was tuned (Apple M-series).
    pub(super) fn for_threads(threads: usize) -> Self {
        if cfg!(target_arch = "x86_64") && threads > Self::SHARED_MAX_THREADS {
            Self::PerTile
        } else {
            Self::Shared
        }
    }
}

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

    /// Chooses a near-square tile grid for an `m × w` block of C, with about
    /// [`PER_TILE_TILES_PER_THREAD`] tiles per thread (see
    /// [`PackScope::PerTile`]).
    fn choose_square(m: usize, w: usize, shape: &MicroKernelShape, threads: usize) -> Self {
        let mu = m.div_ceil(shape.mr);
        let nu = w.div_ceil(shape.nr);
        let target = (PER_TILE_TILES_PER_THREAD * threads).min(mu * nu).max(1);
        // rows / cols ≈ m / w makes the tiles near-square in elements.
        let ideal = (target as f64 * m as f64 / w as f64).sqrt().round() as usize;
        let rows = ideal.clamp(1, mu.min(target));
        let cols = target.div_ceil(rows).clamp(1, nu);
        Self { rows, cols }
    }

    /// Chooses the grid for `scope`.
    fn for_scope(
        scope: PackScope,
        m: usize,
        w: usize,
        shape: &MicroKernelShape,
        threads: usize,
    ) -> Self {
        match scope {
            PackScope::Shared => Self::choose(m, w, shape, threads),
            PackScope::PerTile => Self::choose_square(m, w, shape, threads),
        }
    }

    /// Widest tile column of a `w`-column block.
    fn widest_cols(self, w: usize, nr: usize) -> usize {
        (w.div_ceil(nr).div_ceil(self.cols) * nr).min(w)
    }

    /// Total number of tiles.
    fn count(self) -> usize {
        self.rows * self.cols
    }

    /// Height of the tallest tile row of an `m`-row block.
    fn tallest_rows(self, m: usize, mr: usize) -> usize {
        (m.div_ceil(mr).div_ceil(self.rows) * mr).min(m)
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
/// that), and `alpha == 0` must have been handled by the caller. Returns
/// `false` without touching C when the widest NC block cannot be split into
/// at least two tiles; the caller then runs the sequential driver.
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
    shape: &MicroKernelShape,
    par: Par,
) -> bool {
    try_gemm_blocked_parallel_scoped(
        trans_a,
        trans_b,
        alpha,
        a,
        b,
        beta,
        c,
        blocking,
        shape,
        par,
        PackScope::for_threads(par.num_threads()),
    )
}

/// [`try_gemm_blocked_parallel`] with an explicit [`PackScope`].
#[allow(clippy::too_many_arguments)]
pub(super) fn try_gemm_blocked_parallel_scoped<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    shape: &MicroKernelShape,
    par: Par,
    scope: PackScope,
) -> bool {
    // Logical (post-transpose) dimensions.
    let (m, k) = op_dims(a, trans_a);
    let (_, n) = op_dims(b, trans_b);
    if m == 0 || n == 0 || k == 0 {
        return false;
    }

    let mr = shape.mr;
    let nr = shape.nr;
    let mc = blocking.mc.min(m);
    let kc = blocking.kc.min(k);
    let nc = blocking.nc.min(n);
    let threads = par.num_threads();

    // Grids of the full-width NC blocks and of the (possibly narrower) last.
    let full_grid = TileGrid::for_scope(scope, m, nc, shape, threads);
    if full_grid.count() < 2 {
        return false;
    }
    let last_w = n - (n - 1) / nc * nc;
    let last_grid = TileGrid::for_scope(scope, m, last_w, shape, threads);

    // Each tile packs at most `round_up(min(mc, tallest tile), MR) x KC` of A.
    let max_tile_rows = full_grid
        .tallest_rows(m, mr)
        .max(last_grid.tallest_rows(m, mr));
    let a_len = mc.min(max_tile_rows).div_ceil(mr) * mr * kc;

    if scope == PackScope::PerTile {
        // ... and at most `round_up(widest tile, NR) x KC` of B.
        let max_tile_cols = full_grid
            .widest_cols(nc, nr)
            .max(last_grid.widest_cols(last_w, nr));
        let b_len = max_tile_cols.div_ceil(nr) * nr * kc;
        for jc in (0..n).step_by(nc) {
            let w = nc.min(n - jc);
            let grid = if w == nc { full_grid } else { last_grid };
            let mut tiles = split_tiles(c.rb_mut().submatrix(0, jc, m, w), grid, shape);
            tiles.par_iter_mut().with_max_len(1).for_each(|tile| {
                // Buffers from the worker's own pool: allocated (and first
                // touched) by the thread that uses them.
                let mut a_buf: AlignedVec<T> = gemm_pack_pool::take(a_len);
                let mut b_buf: AlignedVec<T> = gemm_pack_pool::take(b_len);
                let rows = tile.view.nrows();
                let cols = tile.view.ncols();
                let mc_tile = balanced_block(rows, mc, mr);
                for pc in (0..k).step_by(kc) {
                    let pb = kc.min(k - pc);
                    let beta_eff = if pc == 0 { beta } else { T::one() };
                    pack_b_trans_dispatch(b, trans_b, pc, jc + tile.col0, pb, cols, &mut b_buf, nr);
                    for ic in (0..rows).step_by(mc_tile) {
                        let ib = mc_tile.min(rows - ic);
                        pack_a_trans_dispatch(
                            a,
                            trans_a,
                            tile.row0 + ic,
                            pc,
                            ib,
                            pb,
                            &mut a_buf,
                            mr,
                        );
                        macro_panel_multiply(
                            alpha,
                            &a_buf,
                            ib,
                            pb,
                            &b_buf,
                            pb,
                            cols,
                            beta_eff,
                            tile.view.rb_mut().submatrix(ic, 0, ib, cols),
                            shape,
                        );
                    }
                }
                gemm_pack_pool::give(a_buf);
                gemm_pack_pool::give(b_buf);
            });
        }
        return true;
    }

    // One A~ buffer per worker of the current pool, allocated once per call.
    // Tile tasks never re-enter rayon while holding a buffer, so each slot is
    // only ever used by its own worker; the mutex makes even a stray thread
    // (`current_thread_index() == None`) safe, merely serialised.
    let workers = rayon::current_num_threads().max(1);
    let a_bufs: Vec<Mutex<AlignedVec<T>>> = (0..workers)
        .map(|_| Mutex::new(AlignedVec::zeros(a_len)))
        .collect();
    // Shared B~ of one (jc, pc) step, reused for every step.
    let mut pack_b: AlignedVec<T> = AlignedVec::zeros(nc.div_ceil(nr) * nr * kc);

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
                let slot = rayon::current_thread_index().unwrap_or(0) % a_bufs.len();
                let mut a_buf = a_bufs[slot].lock().unwrap_or_else(PoisonError::into_inner);
                let rows = tile.view.nrows();
                let cols = tile.view.ncols();
                let b_panels = &packed_b[tile.col0 / nr * pb * nr..];
                let mc_tile = balanced_block(rows, mc, mr);
                for ic in (0..rows).step_by(mc_tile) {
                    let ib = mc_tile.min(rows - ic);
                    pack_a_trans_dispatch(a, trans_a, tile.row0 + ic, pc, ib, pb, &mut a_buf, mr);
                    macro_panel_multiply(
                        alpha,
                        &a_buf,
                        ib,
                        pb,
                        b_panels,
                        pb,
                        cols,
                        beta_eff,
                        tile.view.rb_mut().submatrix(ic, 0, ib, cols),
                        shape,
                    );
                }
            });
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level3::autotune::KC_BLOCK_UNIT;
    use crate::level3::gemm::{
        PARALLEL_GATE, gemm_transposed_with_blocking, gemm_transposed_with_par,
    };
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    /// Element helpers for the bitwise comparisons.
    trait Probe: Field + GemmKernel + bytemuck::Zeroable {
        /// Exact bit patterns (real, imaginary).
        fn bits(self) -> (u64, u64);
        /// True when every component is finite.
        fn finite(self) -> bool;
        /// Deterministic, non-symmetric sample value.
        fn sample(seed: usize) -> Self;
        /// A NaN (NaN in every component).
        fn nan() -> Self;
    }

    fn unit_sample(seed: usize) -> f64 {
        ((seed.wrapping_mul(2_654_435_761) >> 7) % 1009) as f64 / 504.5 - 1.0
    }

    impl Probe for f64 {
        fn bits(self) -> (u64, u64) {
            (self.to_bits(), 0)
        }
        fn finite(self) -> bool {
            self.is_finite()
        }
        fn sample(seed: usize) -> Self {
            unit_sample(seed)
        }
        fn nan() -> Self {
            f64::NAN
        }
    }

    impl Probe for f32 {
        fn bits(self) -> (u64, u64) {
            (u64::from(self.to_bits()), 0)
        }
        fn finite(self) -> bool {
            self.is_finite()
        }
        fn sample(seed: usize) -> Self {
            unit_sample(seed) as f32
        }
        fn nan() -> Self {
            f32::NAN
        }
    }

    impl Probe for Complex64 {
        fn bits(self) -> (u64, u64) {
            (self.re.to_bits(), self.im.to_bits())
        }
        fn finite(self) -> bool {
            self.re.is_finite() && self.im.is_finite()
        }
        fn sample(seed: usize) -> Self {
            Complex64::new(unit_sample(seed), unit_sample(seed.wrapping_add(7_919)))
        }
        fn nan() -> Self {
            Complex64::new(f64::NAN, f64::NAN)
        }
    }

    fn fill<T: Probe>(rows: usize, cols: usize, seed: usize) -> Mat<T> {
        let mut out: Mat<T> = Mat::zeros(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                out[(i, j)] = T::sample(i * 7_919 + j * 104_729 + seed * 1_299_709);
            }
        }
        out
    }

    fn assert_bits_eq<T: Probe>(got: &Mat<T>, expected: &Mat<T>, ctx: &str) {
        for j in 0..got.ncols() {
            for i in 0..got.nrows() {
                assert_eq!(
                    got[(i, j)].bits(),
                    expected[(i, j)].bits(),
                    "{ctx} at ({i},{j}): {} vs {}",
                    got[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    /// The blocking `gemm_transposed_with_par` selects, fitted to the problem.
    fn fitted_blocking<T: Probe>(m: usize, k: usize, n: usize) -> GemmBlocking {
        let shape = T::micro_kernel_shape();
        let blk = if m >= 512 || k >= 512 || n >= 512 {
            GemmBlocking::auto_tuned::<T>(m, k, n, &shape)
        } else {
            GemmBlocking::for_kernel::<T>(&shape)
        };
        fit(&blk, m, k, n, &shape)
    }

    fn fit(
        blk: &GemmBlocking,
        m: usize,
        k: usize,
        n: usize,
        shape: &MicroKernelShape,
    ) -> GemmBlocking {
        GemmBlocking {
            mc: balanced_block(m, blk.mc, shape.mr),
            kc: balanced_block(k, blk.kc, KC_BLOCK_UNIT),
            nc: balanced_block(n, blk.nc, shape.nr),
        }
    }

    /// Runs `C = alpha * op(A) * op(B) + beta * C` sequentially, then on the 2D
    /// driver (directly and through the public entry point) for every thread
    /// count, asserting bitwise-identical results. With `poison_c`, C starts
    /// as NaN and `beta == 0`, so C must not be read.
    fn check<T: Probe>(
        (m, k, n): (usize, usize, usize),
        (ta, tb): (Trans, Trans),
        blocking: Option<GemmBlocking>,
        threads: &[usize],
        poison_c: bool,
    ) {
        let shape = T::micro_kernel_shape();
        let a = if ta == Trans::NoTrans {
            fill::<T>(m, k, 1)
        } else {
            fill::<T>(k, m, 1)
        };
        let b = if tb == Trans::NoTrans {
            fill::<T>(k, n, 2)
        } else {
            fill::<T>(n, k, 2)
        };
        let alpha = T::sample(5);
        let (beta, c_init) = if poison_c {
            (T::zero(), Mat::filled(m, n, T::nan()))
        } else {
            (T::sample(9), fill::<T>(m, n, 3))
        };
        let fitted = match blocking {
            Some(blk) => fit(&blk, m, k, n, &shape),
            None => fitted_blocking::<T>(m, k, n),
        };
        let ctx = format!("{m}x{k}x{n} {ta:?}/{tb:?} {fitted:?}");

        let mut c_seq = c_init.clone();
        gemm_transposed_with_blocking(
            ta,
            tb,
            alpha,
            a.as_ref(),
            b.as_ref(),
            beta,
            c_seq.as_mut(),
            Par::Seq,
            &fitted,
        );
        if poison_c {
            for j in 0..n {
                for i in 0..m {
                    assert!(c_seq[(i, j)].finite(), "{ctx}: beta = 0 read C");
                }
            }
        }

        for &t in threads {
            let par = Par::RayonWith(t);
            assert!(
                PARALLEL_GATE.should_parallelize(m * n * k, par),
                "{ctx}: gate, t={t}"
            );

            for scope in [PackScope::Shared, PackScope::PerTile] {
                let mut c_par = c_init.clone();
                let engaged = try_gemm_blocked_parallel_scoped(
                    ta,
                    tb,
                    alpha,
                    &a.as_ref(),
                    &b.as_ref(),
                    beta,
                    &mut c_par.as_mut(),
                    &fitted,
                    &shape,
                    par,
                    scope,
                );
                assert!(engaged, "{ctx}: 2D driver declined, t={t}, {scope:?}");
                assert_bits_eq(&c_par, &c_seq, &format!("{ctx} direct t={t} {scope:?}"));
            }

            let mut c_api = c_init.clone();
            gemm_transposed_with_blocking(
                ta,
                tb,
                alpha,
                a.as_ref(),
                b.as_ref(),
                beta,
                c_api.as_mut(),
                par,
                &fitted,
            );
            assert_bits_eq(&c_api, &c_seq, &format!("{ctx} api t={t}"));
        }
    }

    /// Shapes (m, k, n): square-ish with two KC blocks, tall, wide, and a deep
    /// k with five KC blocks.
    const SHAPES: [(usize, usize, usize); 4] = [
        (513, 517, 300),
        (2048, 64, 64),
        (64, 64, 4096),
        (37, 4099, 129),
    ];

    #[test]
    fn test_issue_16_parallel_matches_sequential_bitwise_f64() {
        let combos = [
            (Trans::NoTrans, Trans::NoTrans),
            (Trans::Trans, Trans::NoTrans),
            (Trans::NoTrans, Trans::ConjTrans),
            (Trans::Trans, Trans::Trans),
        ];
        for &shape in &SHAPES {
            for &combo in &combos {
                check::<f64>(shape, combo, None, &[2, 3, 8], false);
            }
            check::<f64>(shape, (Trans::NoTrans, Trans::NoTrans), None, &[5], true);
        }
    }

    #[test]
    fn test_issue_16_parallel_matches_sequential_bitwise_f32() {
        for &shape in &SHAPES {
            check::<f32>(
                shape,
                (Trans::NoTrans, Trans::NoTrans),
                None,
                &[2, 8],
                false,
            );
            check::<f32>(shape, (Trans::Trans, Trans::ConjTrans), None, &[3], false);
            check::<f32>(shape, (Trans::NoTrans, Trans::Trans), None, &[4], true);
        }
    }

    #[test]
    fn test_issue_16_parallel_matches_sequential_bitwise_c64() {
        // 213 * 157 * 137 = 4.58M multiply-adds clears the 8-thread gate
        // (PARALLEL_GATE: 80^3 per thread, i.e. 160^3 on 8 threads).
        let all = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
        for &ta in &all {
            for &tb in &all {
                check::<Complex64>((213, 157, 137), (ta, tb), None, &[2, 3, 8], false);
            }
        }
        for &shape in &[(2048usize, 64usize, 64usize), (64, 64, 4096)] {
            check::<Complex64>(
                shape,
                (Trans::ConjTrans, Trans::ConjTrans),
                None,
                &[3],
                false,
            );
            check::<Complex64>(shape, (Trans::NoTrans, Trans::NoTrans), None, &[5], true);
        }
    }

    #[test]
    fn test_issue_16_parallel_bitwise_with_many_blocks() {
        // Small explicit targets: four NC blocks, four KC blocks and several
        // MC sub-blocks per tile exercise every loop of the 2D driver.
        let blocking = GemmBlocking {
            mc: 64,
            kc: 48,
            nc: 96,
        };
        let combos = [
            (Trans::NoTrans, Trans::NoTrans),
            (Trans::Trans, Trans::NoTrans),
            (Trans::NoTrans, Trans::Trans),
            (Trans::ConjTrans, Trans::ConjTrans),
        ];
        for &combo in &combos {
            check::<f64>((200, 150, 300), combo, Some(blocking), &[2, 3, 5, 8], false);
        }
        check::<f64>((200, 150, 300), combos[0], Some(blocking), &[4], true);
        check::<f32>((200, 150, 300), combos[1], Some(blocking), &[3], false);
        check::<Complex64>((200, 150, 300), combos[3], Some(blocking), &[3], false);
    }

    #[test]
    fn test_issue_16_parallel_bitwise_strided_views() {
        // Transposed views (col_stride != 1) for A and B, and a C view with a
        // leading dimension larger than its row count.
        let (m, k, n) = (160usize, 150usize, 120usize);
        let a_store = fill::<f64>(k, m, 1);
        let b_store = fill::<f64>(n, k, 2);
        let a_view = a_store.as_ref().transpose();
        let b_view = b_store.as_ref().transpose();
        assert_ne!(a_view.col_stride(), 1);
        let shape = f64::micro_kernel_shape();
        let fitted = fitted_blocking::<f64>(m, k, n);

        let mut c_seq_big = fill::<f64>(m + 5, n + 3, 3);
        let mut c_par_big = c_seq_big.clone();
        gemm_transposed_with_blocking(
            Trans::NoTrans,
            Trans::NoTrans,
            0.75,
            a_view,
            b_view,
            -1.25,
            c_seq_big.as_mut().submatrix(2, 1, m, n),
            Par::Seq,
            &fitted,
        );
        let engaged = try_gemm_blocked_parallel(
            Trans::NoTrans,
            Trans::NoTrans,
            0.75,
            &a_view,
            &b_view,
            -1.25,
            &mut c_par_big.as_mut().submatrix(2, 1, m, n),
            &fitted,
            &shape,
            Par::RayonWith(4),
        );
        assert!(engaged);
        // The whole backing store must agree, including the untouched border.
        assert_bits_eq(&c_par_big, &c_seq_big, "strided views");
    }

    #[test]
    fn test_issue_16_tile_grid_valid_and_balanced() {
        for &(mr, nr) in &[(8usize, 6usize), (8, 8), (16, 6), (4, 4), (2, 2)] {
            let shape = MicroKernelShape { mr, nr };
            for &m in &[1usize, 7, 64, 100, 513, 2048, 4096] {
                for &w in &[1usize, 6, 64, 517, 2048, 5712] {
                    for &threads in &[2usize, 3, 4, 8, 12, 64] {
                        let grid = TileGrid::choose(m, w, &shape, threads);
                        let (mu, nu) = (m.div_ceil(mr), w.div_ceil(nr));
                        let ctx = format!("{mr}x{nr} m={m} w={w} t={threads}: {grid:?}");
                        assert!((1..=mu).contains(&grid.rows), "{ctx}");
                        assert!((1..=nu).contains(&grid.cols), "{ctx}");
                        assert!(grid.count() < 2 * TILES_PER_THREAD * threads, "{ctx}");
                        if mu * nu >= 8 {
                            assert!(grid.count() >= 2, "{ctx}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_pack_scope_for_threads() {
        for t in 0..=PackScope::SHARED_MAX_THREADS {
            assert_eq!(PackScope::for_threads(t), PackScope::Shared, "t={t}");
        }
        let many = PackScope::for_threads(PackScope::SHARED_MAX_THREADS + 1);
        if cfg!(target_arch = "x86_64") {
            assert_eq!(many, PackScope::PerTile);
        } else {
            assert_eq!(many, PackScope::Shared);
        }
    }

    #[test]
    fn test_per_tile_grid_valid_and_near_square() {
        for &(mr, nr) in &[(8usize, 6usize), (16, 6), (4, 4), (2, 2)] {
            let shape = MicroKernelShape { mr, nr };
            for &m in &[1usize, 7, 64, 100, 513, 2048, 4096] {
                for &w in &[1usize, 6, 64, 517, 2048, 5712] {
                    for &threads in &[2usize, 3, 4, 8, 12, 32, 64] {
                        let grid = TileGrid::choose_square(m, w, &shape, threads);
                        let (mu, nu) = (m.div_ceil(mr), w.div_ceil(nr));
                        let ctx = format!("{mr}x{nr} m={m} w={w} t={threads}: {grid:?}");
                        assert!((1..=mu).contains(&grid.rows), "{ctx}");
                        assert!((1..=nu).contains(&grid.cols), "{ctx}");
                        let target = PER_TILE_TILES_PER_THREAD * threads;
                        assert!(grid.count() < 2 * target, "{ctx}");
                        if mu * nu >= target {
                            assert!(grid.count() >= target, "{ctx}");
                        }
                        // Square problems with room for it get near-square
                        // tiles (in elements), not thin strips.
                        if m == w && mu >= target && nu >= target {
                            let tile_h = m.div_ceil(grid.rows);
                            let tile_w = w.div_ceil(grid.cols);
                            assert!(tile_h * 2 >= tile_w && tile_w * 2 >= tile_h, "{ctx}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_issue_16_split_tiles_cover_c_exactly_once() {
        let shape = MicroKernelShape { mr: 8, nr: 6 };
        for &(m, w, threads) in &[
            (37usize, 50usize, 3usize),
            (513, 300, 8),
            (8, 6, 2),
            (64, 4096, 12),
        ] {
            let grid = TileGrid::choose(m, w, &shape, threads);
            let mut c: Mat<f64> = Mat::zeros(m, w);
            let tiles = split_tiles(c.as_mut(), grid, &shape);
            assert_eq!(tiles.len(), grid.count());
            let mut covered_rows = 0;
            for mut tile in tiles {
                let (rows, cols) = (tile.view.nrows(), tile.view.ncols());
                assert!(rows > 0 && cols > 0, "empty tile");
                assert_eq!(tile.row0 % shape.mr, 0, "row start off the micro grid");
                assert_eq!(tile.col0 % shape.nr, 0, "column start off the micro grid");
                assert!(tile.row0 + rows <= m && tile.col0 + cols <= w);
                // Tag each element with its global row so a misplaced view is
                // caught, not just a double-counted one.
                for j in 0..cols {
                    for i in 0..rows {
                        tile.view[(i, j)] += 1.0 + (tile.row0 + i) as f64 * 1e3;
                    }
                }
                covered_rows += rows * cols;
            }
            assert_eq!(covered_rows, m * w);
            for j in 0..w {
                for i in 0..m {
                    let expected = 1.0 + i as f64 * 1e3;
                    assert_eq!(c[(i, j)], expected, "({i},{j}) misplaced or double-covered");
                }
            }
        }
    }

    #[test]
    fn test_issue_16_square_problems_split_across_threads() {
        // With the NC-only split, a square problem with n <= NC was a single
        // work item. Every NC block must now offer at least one tile per
        // thread.
        let threads = 12;
        for &n in &[512usize, 1024, 2048, 4096] {
            for (label, shape, blk) in [
                (
                    "f64",
                    f64::micro_kernel_shape(),
                    GemmBlocking::auto_tuned::<f64>(n, n, n, &f64::micro_kernel_shape()),
                ),
                (
                    "f32",
                    f32::micro_kernel_shape(),
                    GemmBlocking::auto_tuned::<f32>(n, n, n, &f32::micro_kernel_shape()),
                ),
            ] {
                let nc = balanced_block(n, blk.nc, shape.nr);
                let last_w = n - (n - 1) / nc * nc;
                for w in [nc, last_w] {
                    let grid = TileGrid::choose(n, w, &shape, threads);
                    assert!(
                        grid.count() >= threads,
                        "{label} n={n} w={w}: only {} tiles ({grid:?})",
                        grid.count()
                    );
                }
            }
        }
    }

    #[test]
    fn test_issue_16_parallel_alpha_zero_and_api_paths() {
        // The funnel handles alpha == 0 before the 2D driver: NaN operands
        // must not leak, whatever the thread count.
        let n = 200;
        let a: Mat<f64> = Mat::filled(n, n, f64::NAN);
        let b: Mat<f64> = Mat::filled(n, n, f64::INFINITY);
        for &t in &[2usize, 7] {
            let mut c: Mat<f64> = Mat::filled(n, n, 2.0);
            gemm_transposed_with_par(
                Trans::Trans,
                Trans::NoTrans,
                0.0,
                a.as_ref(),
                b.as_ref(),
                0.5,
                c.as_mut(),
                Par::RayonWith(t),
            );
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(c[(i, j)].to_bits(), 1.0f64.to_bits(), "t={t} ({i},{j})");
                }
            }
        }
    }
}
