//! The SME2 GEMM driver: the decision, the policy and the sequential block
//! loop.
//!
//! [`try_gemm`] is the one entry point of the backend for real `f32` / `f64`
//! products `C = alpha · op(A) · op(B) + beta · C`. It takes the SME2
//! decision once, on the calling thread, and then either computes the whole
//! product (`true`: C holds the complete result) or declines without having
//! touched C (`false`: C is bit for bit what it was, and the caller runs its
//! usual NEON path).
//!
//! # Where it is called
//!
//! The GEMM hook in `gemm_core` (`gemm.rs`) calls it inside the blocked
//! route, after the route policy chose that route, after
//! `plan.micro_kernel()` and after `call_blocking_for` computed the blocks
//! of the call, and only for a plan of the public entry points
//! (`GemmPlan::is_production`: no given micro-kernel, no forced route, no
//! forced pack scope). Every public entry point that reaches `gemm_core`
//! builds such a plan (`gemm`, `gemm_with_par`, `gemm_transposed`,
//! `gemm_transposed_with_par` with the default blocking;
//! `gemm_with_blocking` and `gemm_transposed_with_blocking` with the
//! caller's blocking; `gemm_auto*`, `gemm_asymmetric*`, the Strassen leaves
//! and the runtime-tuned GEMM with this crate's heuristic blocking). So the
//! driver sees exactly the calls that the NEON build runs on the blocked
//! route, with the same blocks; a call it declines runs that route with the
//! same blocks.
//!
//! # The bitwise promise
//!
//! Every call the driver accepts gives the bits of the NEON blocked route of
//! the same call (the KC partition of `gemm::call_blocking_for`, ZA zeroed
//! per tile, the NEON epilogue order), for every part count, except NaN sign
//! and payload (the ZA instructions return the default NaN; NaN positions
//! are equal). Nothing the policy below chooses (in place, online packing,
//! the ZA transposers, few rows, the B caps, prefetch) changes a result bit:
//! each choice only changes where the kernels read A and B from, how C is
//! cut into blocks (MC, NC, parts) or which hints they issue, and every C
//! element is still the same chain of fused multiply-adds over its own row of
//! A and column of B, depth block by depth block in ascending order.
//!
//! # The decision (nothing in it writes C)
//!
//! 1. **Type and hardware.** `T` is `f32` or `f64` (the `TypeId` hand-off of
//!    `level1/typed.rs`), and `detect::enabled::<T>()`. Calls that the GEMM
//!    core never sends (an empty product, `k == 0`, `alpha == 0`, mismatched
//!    dimensions) decline as well. `enabled` is the first question that
//!    reaches the detection, on the calling thread, and it never waits:
//!    while the verdict of the first-use checks is open it either runs the
//!    checks on this thread (which then settle the verdict) or, when another
//!    thread is running them or this thread is not ready, answers `false` and
//!    the call declines. So a decision that gets past this step has seen the
//!    verdict settled to "passed", and no part of the call (each part asks
//!    `detect::thread_ready()` on its own thread, in `parallel.rs`) can be
//!    the thread that runs the checks.
//! 2. **KC (rule R1).** `kc = blocking.sequential().kc.min(k)`, and the depth
//!    partition is `(0..k).step_by(kc)`, exactly the one of the NEON
//!    sequential driver (`gemm_blocked_sequential`) and of the 2D parallel
//!    driver for the same call. A degenerate step (`kc <` [`MIN_KC`] while
//!    `kc < k`, possible only with an explicit blocking) declines: NEON then
//!    gives the same bits, faster.
//! 3. **Operands.** `ColMajor::normalize` for A and B (a view with adjacent
//!    rows is read through its transpose); a fully strided view declines.
//! 4. **Class** ([`classify`]): thin (`m == 1 || n == 1`) declines (where
//!    the route policy turns the thin route on, `gemm_thin` gives such a
//!    product to the SME2 GEMV); small (`m · n · k <=` [`SMALL_MACS`]) with
//!    `m >= 24 && n >= 8 && k >= 8` or `m >= 16 && n >= 16 && k >= 64` is
//!    the small class, accepted only when [`SMALL_CLASS_ON`] is set (it is
//!    not: see the constant); every other small product declines;
//!    everything else is blocked. A C whose columns overlap
//!    (`row_stride < m`) declines.
//! 5. **Thread.** `detect::thread_ready()` on the calling thread, exactly
//!    once per call.
//! 6. **Parts** ([`part_count`]): 1, unless `par` is parallel, the build
//!    reports at least two SME units (`detect::units`) and the work `m·n·k`
//!    reaches [`PAR_MIN_MACS_F32`] / [`PAR_MIN_MACS_F64`]; then
//!    `min(units, par.num_threads())`. The thread count is asked for only
//!    after the threshold passed, because for `Par::Rayon` it builds rayon's
//!    global pool.
//! 7. **Route** ([`route_for`]): few rows (`m <=` `few_rows::max_rows::<T>()`:
//!    48 for `f32`, 32 for `f64`) or blocked; see "The policy".
//!
//! Then every value of the plan is computed and checked (the split of C,
//! the route, the block sizes, the A source of every block and the buffer
//! lengths of every part, with checked arithmetic, against the packing-cache
//! limit) before the first kernel call; a plan that cannot be built
//! declines. From there on the call returns `true`.
//!
//! # The policy
//!
//! The thresholds are the constants below; [`Tuning`] carries them into a
//! call (test builds can override them per thread). All sizes are in bytes
//! of the element type.
//!
//! **Few rows** (`m <= max_rows`, [`Route::FewRows`]): the whole piece of a
//! part is one `few_rows_gemm` call (one streaming region; A read in place,
//! B transposed on the fly in ZA, never packed). The parts split the
//! columns (whole `NR`-column panels). The depth step is the blocked path's
//! KC (`kc` above), as `few_rows.rs` requires. A must be a column-major
//! `m × k` view:
//! - `op(A) = A` (`NoTrans` after normalisation, unit row step): the stored
//!   matrix itself;
//! - `op(A) = Aᵀ`: a column-major copy, made once on the calling thread in
//!   the A slot of its packing cache and shared read-only by the parts, when
//!   `m · k` elements fit the packing-cache limit; otherwise the call takes
//!   the blocked route.
//!
//! The arguments are validated once for the whole call (`few_rows_args` on
//! the full C; the parts' sub-views share its strides); a call they do not
//! fit takes the blocked route.
//!
//! **Blocked** ([`Route::Blocked`], [`run_part`]): the loop pc → jc → ic of
//! `gemm_blocked_sequential` with the caller's KC partition, MC =
//! `balanced_block(part rows, `[`MC_TARGET`]`, 64)`, and per part:
//! - **B** is packed once per `(pc, jc)` block: `pack_b_za` (the ZA
//!   transposer) for `op(B) = B`, the plain `pack_b` for `op(B) = Bᵀ` (a
//!   contiguous copy). NC is `balanced_block(part columns, nc_target, NR)`
//!   with `nc_target` the widest multiple of `NR` whose packed B block fits
//!   the B cap (at least `NR`). The cap ([`b_cap`]) is [`B_BUFFER_BYTES`]
//!   (4 MiB), except when all rows of the part are one MC block and the
//!   packed B of a depth block (`kc ×` all columns of the part) exceeds it:
//!   then B is used by one row block only, and [`B_ONE_BLOCK_BYTES`] (2 MiB)
//!   keeps it in a smaller cache.
//! - **A**, per `(pc, ic)` block, decided from the block's shape only
//!   ([`a_source`]; never per tile):
//!   - `op(A) = Aᵀ`: `pack_a_za` (the ZA transposer) into the A buffer;
//!   - `op(A) = A` (column-contiguous): **in place** when the block's span
//!     of A, `((depth - 1) · lda + rows) · elem` with the block's own
//!     `rows` and `depth`, is at most [`IN_PLACE_BYTES`] (2 MiB);
//!     otherwise **online packing**: the kernel reads A in place for column
//!     panel 0 and packs it into the A buffer as it goes, then reads the
//!     buffer for the later panels. The packed copy is **reused across jc**
//!     (the kernel calls of the later column blocks read it as a plain
//!     packed block) when the copy of all row blocks of the part,
//!     `round_up(part rows, 64) · kc · elem`, is at most [`A_REUSE_BYTES`]
//!     (4 MiB) and fits the packing cache next to B; otherwise it is
//!     **rebuilt** per `(jc, ic)` in an MC-row buffer. A reused region is
//!     read only after the kernel filled it in the same depth block.
//! - **Prefetch**: every kernel call of the blocked class carries
//!   `PREFETCH_C` (hints for the tile's live C columns); the small class
//!   (when enabled) does not.
//! - **Buffers** come from the calling part's packing cache (`PackBuffer`,
//!   A and B slots). Their sum must fit the thread's packing-cache limit
//!   (`gemm_packing::pack_cache_limit`, read once on the calling thread),
//!   so that they can be parked for the next call: a reused online copy that
//!   does not fit falls back to the rebuilt one; when even the plain buffers
//!   (MC × kc for A, unless every block reads A in place, and the B block)
//!   do not fit, the call declines before anything touches C.
//!
//! # The sequential driver ([`run_part`], one blocked part of C)
//!
//! ```text
//! buf_a = PackBuffer::take(A, part.a_len)                 once per part
//! buf_b = PackBuffer::take(B, part.b_len)
//! for depth in spans(k, kc):                              pc (R1: the caller's partition)
//!     beta_eff = block_beta(depth, beta)                  beta first, 1 afterwards
//!     for cols in spans(part columns, NC):                jc
//!         pb = pack_b_za / pack_b(B, depth, cols, buf_b)  NR-column plain layout
//!         for rows in spans(part rows, MC):               ic
//!             A source (in place / online fill / reused copy / pack_a_za)
//!             kernel(args with PREFETCH_C)                one streaming region
//! ```
//!
//! The plan makes every packer and argument-block call succeed. Should one
//! still fail, the block is computed by the NEON macro-panel
//! (`macro_panel_multiply_with` with the canonical micro-kernel, the same
//! depth block and the same effective beta), which gives the same bits, so
//! the call never has to give up half-way; a few-rows part that the kernel
//! refuses runs the NEON sequential driver on its piece.
//!
//! The split over SME units and the NEON fallback of a part whose thread is
//! not ready live in `parallel.rs`. Every SME2 entry of a part (macro
//! kernels, ZA transposers, few rows) runs only on the thread that ran the
//! part's own readiness check.

#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use std::marker::PhantomData;

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

use super::args::{MacroArgs, PREFETCH_C, PREFETCH_OFF, SmeElement};
use super::detect;
use super::few_rows::{SmeFewRows, few_rows_args, few_rows_gemm, max_rows};
use super::kernel_f32::sme2_gemm_f32;
use super::kernel_f64::sme2_gemm_f64;
use super::pack::{
    ColMajor, PackedA, PackedB, Span, a_pack_len, block_args, block_beta, pack_b, spans,
};
use super::pack_za::{SmePack, pack_a_za, pack_b_za};
use super::parallel;
use crate::level1::typed::{same_matmut, same_matref, same_value};
use crate::level3::gemm::{
    CallBlocking, GemmBlocking, macro_panel_multiply_with, op_dims, pack_a_trans_dispatch,
    pack_b_trans_dispatch,
};
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::gemm_packing::{PackBuffer, PackSlot, pack_cache_limit};
use crate::level3::trsm::Trans;

/// Smallest KC step the backend accepts when the step splits the depth
/// (`kc < k`). Each depth block costs one streaming region per C block, so a
/// shorter explicit step would spend more time entering and leaving
/// streaming mode than multiplying; NEON gives the same bits.
pub(super) const MIN_KC: usize = 32;

/// Work (`m · n · k`) up to which a product counts as small: `32³`.
pub(super) const SMALL_MACS: usize = 32 * 32 * 32;

/// Whether the driver accepts the small class (products of at most
/// [`SMALL_MACS`] multiply-adds that are tall and wide enough to fill ZA
/// tiles, [`Class::Small`]). Off.
///
/// The GEMM hook sits inside the blocked route of `gemm_core`, so the driver
/// only ever sees calls that the NEON build also runs on the blocked route,
/// with the same blocks. With the default route policy every product of at
/// most 32³ multiply-adds takes a small route and never reaches the hook;
/// so this constant concerns only small products that a policy (or a later
/// route change) sends to the blocked route, and nobody has measured whether
/// SME2 is faster there: the streaming-mode entry and exit and the ZA
/// zeroing of one block may cost more than the few thousand multiply-adds.
/// Switching the constant on changes speed only, never a result bit (the
/// driver gives the NEON blocked route's bits for every call it accepts).
/// Taking the products that NEON computes on its small routes would need a
/// hook ahead of the route and would change their bits: not built.
// TODO(M4 tuning): measure the small class on the M4 (transposed and
// strided products up to 32³ that the NEON route blocks) and switch this on
// where SME2 wins.
pub(super) const SMALL_CLASS_ON: bool = false;

/// Row-block target of the sequential driver (MC): 512 rows, eight 64-row
/// panels. The A buffer holds `MC × kc` elements (2 MiB of `f64` at
/// `kc = 512`).
pub(super) const MC_TARGET: usize = 512;

/// The B cap (bytes): NC is the widest multiple of NR with `kc · NC`
/// elements within it (4 MiB).
pub(super) const B_BUFFER_BYTES: usize = 4 << 20;

/// The B cap of a part whose rows are one MC block and whose packed B of a
/// depth block exceeds [`B_BUFFER_BYTES`] (2 MiB): each B block is then used
/// by one row block only, so a smaller block stays in a nearer cache.
// TODO(M4 tuning): the PR text's 8 MiB cap for shallow products (small kc)
// is an experiment for the M4; nothing implements it.
pub(super) const B_ONE_BLOCK_BYTES: usize = 2 << 20;

/// Largest span of A (bytes) a block reads in place: a block of `rows ×
/// depth` with leading dimension `lda` touches `((depth - 1) · lda + rows)`
/// elements of A (2 MiB).
pub(super) const IN_PLACE_BYTES: usize = 2 << 20;

/// Largest online-packed copy of A (bytes) that is reused across the column
/// blocks: `round_up(part rows, 64) · kc` elements (4 MiB).
pub(super) const A_REUSE_BYTES: usize = 4 << 20;

/// Work (`m · n · k`) from which an `f32` product is split over several SME
/// units: `2²⁷`. Below it a part runs for too short a time to repay rayon's
/// wake-up (tens of microseconds).
pub(super) const PAR_MIN_MACS_F32: usize = 1 << 27;

/// Work (`m · n · k`) from which an `f64` product is split over several SME
/// units: `2²⁵` (an `f64` FMOPA does a quarter of the `f32` work).
pub(super) const PAR_MIN_MACS_F64: usize = 1 << 25;

/// The element types the driver serves, with their macro-kernel (and, by
/// the supertraits, their ZA transposer and few-rows kernel).
pub(super) trait SmeGemm:
    SmeElement + Field + GemmKernel + bytemuck::Zeroable + Send + Sync + SmePack + SmeFewRows
{
    /// [`PAR_MIN_MACS_F32`] or [`PAR_MIN_MACS_F64`].
    const PAR_MIN_MACS: usize;

    /// Runs the macro-kernel of the type on one block.
    ///
    /// # Safety
    ///
    /// The `# Safety` section of `sme2_gemm_f32` / `sme2_gemm_f64`:
    /// `detect::thread_ready()` returned `true` on this thread and
    /// `detect::enabled::<Self>()` holds; `m, n, kc >= 1`, `ldc >= m`, the
    /// mode from `Epilogue::select`; every address of the layout in
    /// `args.rs` is valid, C overlaps neither A nor B nor the `MacroArgs`
    /// block, and an online-packing buffer (`a_pack`) holds
    /// `ceil(m / 64) · 64 · kc` writable elements that overlap none of them.
    unsafe fn kernel(args: &MacroArgs<Self>);
}

impl SmeGemm for f32 {
    const PAR_MIN_MACS: usize = PAR_MIN_MACS_F32;

    unsafe fn kernel(args: &MacroArgs<f32>) {
        // SAFETY: the caller upholds this function's contract, which is the
        // kernel's.
        unsafe { sme2_gemm_f32(args) }
    }
}

impl SmeGemm for f64 {
    const PAR_MIN_MACS: usize = PAR_MIN_MACS_F64;

    unsafe fn kernel(args: &MacroArgs<f64>) {
        // SAFETY: the caller upholds this function's contract, which is the
        // kernel's.
        unsafe { sme2_gemm_f64(args) }
    }
}

/// The size class of an `m × k` by `k × n` product (decision step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Class {
    /// `m == 1 || n == 1`: declined (the thin route gives it to the SME2
    /// GEMV where the route policy turns that route on).
    Thin,
    /// At most [`SMALL_MACS`] and too narrow for SME2: declined.
    SmallDeclined,
    /// At most [`SMALL_MACS`], but tall and wide enough to fill ZA tiles:
    /// accepted only when the small class is on ([`SMALL_CLASS_ON`]).
    Small,
    /// Everything else.
    Blocked,
}

impl Class {
    /// Whether the class runs on SME2 when the small class is `small_on`.
    pub(super) fn accepted(self, small_on: bool) -> bool {
        match self {
            Self::Blocked => true,
            Self::Small => small_on,
            Self::Thin | Self::SmallDeclined => false,
        }
    }

    /// The prefetch switches of the class's kernel calls: `PREFETCH_C` for
    /// the blocked class, none for the small class (when it runs).
    pub(super) fn prefetch(self) -> u64 {
        match self {
            Self::Blocked => PREFETCH_C,
            Self::Thin | Self::SmallDeclined | Self::Small => PREFETCH_OFF,
        }
    }
}

/// The class of an `m × k` by `k × n` product (`m, n, k >= 1`).
pub(super) fn classify(m: usize, n: usize, k: usize) -> Class {
    if m == 1 || n == 1 {
        return Class::Thin;
    }
    let work = m.saturating_mul(n).saturating_mul(k);
    if work > SMALL_MACS {
        Class::Blocked
    } else if (m >= 24 && n >= 8 && k >= 8) || (m >= 16 && n >= 16 && k >= 64) {
        Class::Small
    } else {
        Class::SmallDeclined
    }
}

/// Whether a KC step `kc` (already `min(blocking.kc, k)`) is usable for depth
/// `k`: at least 1, and at least [`MIN_KC`] unless it covers the whole depth.
pub(super) fn kc_accepted(kc: usize, k: usize) -> bool {
    kc >= 1 && (kc >= MIN_KC || kc >= k)
}

/// The number of parts of a product of work `work` (decision step 6): 1
/// unless `par` is parallel, `units >= 2` and `work >= par_min_macs`; then
/// `min(units, par.num_threads())`, at least 1.
///
/// `par.num_threads()` is asked for last: for `Par::Rayon` it builds rayon's
/// global pool, which a call that stays on one part must not do.
pub(super) fn part_count(work: usize, par: Par, units: usize, par_min_macs: usize) -> usize {
    if units < 2 || par.is_sequential() || work < par_min_macs {
        return 1;
    }
    units.min(par.num_threads()).max(1)
}

/// How a call computes its parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Route {
    /// One `few_rows_gemm` call per part; the parts split the columns.
    /// `copied`: `op(A) = Aᵀ` was copied to a column-major `m × k`
    /// temporary (otherwise the stored A is read in place).
    FewRows {
        /// A was copied to a column-major temporary.
        copied: bool,
    },
    /// The blocked loop of [`run_part`].
    Blocked,
}

/// The route of a product with `m` rows of type `T` (decision step 7)
/// before the few-rows arguments are validated: few rows when
/// `m <= few_rows::max_rows::<T>()`.
pub(super) fn route_for<T: SmeFewRows>(m: usize) -> Route {
    if m <= max_rows::<T>() {
        Route::FewRows { copied: false }
    } else {
        Route::Blocked
    }
}

/// Where one blocked `(pc, ic)` block reads `op(A)` from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ASource {
    /// `op(A) = A`, read in place by the kernel.
    InPlace,
    /// `op(A) = A`, read in place for column panel 0 and packed online.
    Online,
    /// `op(A) = Aᵀ`, packed by the ZA transposer.
    Za,
}

/// Whether a block of `rows × depth` of a column-major A with leading
/// dimension `lda` spans at most `limit` bytes:
/// `((depth - 1) · lda + rows) · elem <= limit` (`false` on overflow).
pub(super) fn in_place_fits(
    lda: usize,
    rows: usize,
    depth: usize,
    elem: usize,
    limit: usize,
) -> bool {
    depth
        .checked_sub(1)
        .and_then(|steps| steps.checked_mul(lda))
        .and_then(|span| span.checked_add(rows))
        .and_then(|span| span.checked_mul(elem))
        .is_some_and(|bytes| bytes <= limit)
}

/// Whether the online-packed copy of `rows` rows (all row blocks of a part)
/// at depth `kc` is reused across the column blocks:
/// `round_up(rows, 64) · kc · elem <= limit` (`false` on overflow).
pub(super) fn a_reuse_fits(rows: usize, kc: usize, elem: usize, limit: usize) -> bool {
    a_pack_len(rows, kc)
        .and_then(|len| len.checked_mul(elem))
        .is_some_and(|bytes| bytes <= limit)
}

/// The A source of one `(pc, ic)` block of `rows × depth` (the block's own
/// sizes) for `op(A)` with transpose `trans` and leading dimension `lda`.
pub(super) fn a_source(
    trans: Trans,
    lda: usize,
    rows: usize,
    depth: usize,
    elem: usize,
    in_place_bytes: usize,
) -> ASource {
    match trans {
        Trans::NoTrans if in_place_fits(lda, rows, depth, elem, in_place_bytes) => ASource::InPlace,
        Trans::NoTrans => ASource::Online,
        Trans::Trans | Trans::ConjTrans => ASource::Za,
    }
}

/// The B cap (bytes) of a part of `rows × cols` with MC `mc` and depth
/// step `kc`: `tuning.b_one_block_bytes` (capped by `tuning.b_bytes`) when
/// the rows are one MC block (`rows <= mc`) and the packed B of a depth
/// block (`kc ×` all `cols`, padded to `NR`) exceeds `tuning.b_bytes`;
/// `tuning.b_bytes` otherwise. `None` on overflow.
pub(super) fn b_cap<T: SmeElement>(
    rows: usize,
    mc: usize,
    cols: usize,
    kc: usize,
    tuning: &Tuning,
) -> Option<usize> {
    let whole = super::pack::b_pack_len::<T>(kc, cols)?.checked_mul(core::mem::size_of::<T>())?;
    if rows <= mc && whole > tuning.b_bytes {
        Some(tuning.b_one_block_bytes.min(tuning.b_bytes))
    } else {
        Some(tuning.b_bytes)
    }
}

/// Block-size targets and thresholds of one call (the constants of this
/// module; test builds can override them per thread with
/// `override_tuning`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Tuning {
    /// MC target ([`MC_TARGET`]).
    pub(super) mc_target: usize,
    /// The B cap in bytes ([`B_BUFFER_BYTES`]).
    pub(super) b_bytes: usize,
    /// The B cap of a one-row-block part with a large B
    /// ([`B_ONE_BLOCK_BYTES`]).
    pub(super) b_one_block_bytes: usize,
    /// Largest span of A read in place ([`IN_PLACE_BYTES`]).
    pub(super) in_place_bytes: usize,
    /// Largest online copy of A reused across jc ([`A_REUSE_BYTES`]).
    pub(super) a_reuse_bytes: usize,
    /// Whether the small class runs ([`SMALL_CLASS_ON`]).
    pub(super) small_class: bool,
    /// Work from which the product is split over SME units
    /// ([`SmeGemm::PAR_MIN_MACS`]).
    pub(super) par_min_macs: usize,
    /// The packing-cache limit of the calling thread in bytes
    /// (`gemm_packing::pack_cache_limit()`, read when the tuning is made):
    /// the buffers of every part must fit it together.
    pub(super) pack_limit: usize,
    /// Depth blocks (bit `pc / kc`, the first 64 blocks) that run on the
    /// NEON macro-panel instead of the kernel, to exercise that path (test
    /// builds only).
    #[cfg(test)]
    pub(super) neon_depths: u64,
    /// Every part waits, before anything else, until all parts of the call
    /// have arrived (`parallel.rs`), so that the parts run on distinct
    /// threads (test builds only).
    #[cfg(test)]
    pub(super) rendezvous: bool,
}

impl Tuning {
    /// The tuning of a `T` call made on the calling thread.
    pub(super) fn of<T: SmeGemm>() -> Self {
        let tuning = Self {
            mc_target: MC_TARGET,
            b_bytes: B_BUFFER_BYTES,
            b_one_block_bytes: B_ONE_BLOCK_BYTES,
            in_place_bytes: IN_PLACE_BYTES,
            a_reuse_bytes: A_REUSE_BYTES,
            small_class: SMALL_CLASS_ON,
            par_min_macs: T::PAR_MIN_MACS,
            pack_limit: pack_cache_limit(),
            #[cfg(test)]
            neon_depths: 0,
            #[cfg(test)]
            rendezvous: false,
        };
        #[cfg(test)]
        let tuning = TUNING_OVERRIDE.with(Cell::get).apply(tuning);
        tuning
    }
}

/// Per-thread replacements for the [`Tuning`] of a call (test builds only;
/// `None` keeps the constant). MC, NC, the B caps, the A-source thresholds
/// and the small class never change a result bit, so tests may change them
/// to reach every branch of the policy with small matrices, and lower the
/// parallel threshold to split small products. The packing-cache limit is
/// not overridden here: tests set the real one
/// (`gemm_packing::set_pack_cache_limit`).
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TuningOverride {
    /// Replaces [`Tuning::mc_target`].
    pub(super) mc_target: Option<usize>,
    /// Replaces [`Tuning::b_bytes`].
    pub(super) b_bytes: Option<usize>,
    /// Replaces [`Tuning::b_one_block_bytes`].
    pub(super) b_one_block_bytes: Option<usize>,
    /// Replaces [`Tuning::in_place_bytes`].
    pub(super) in_place_bytes: Option<usize>,
    /// Replaces [`Tuning::a_reuse_bytes`].
    pub(super) a_reuse_bytes: Option<usize>,
    /// Replaces [`Tuning::small_class`].
    pub(super) small_class: Option<bool>,
    /// Replaces [`Tuning::par_min_macs`].
    pub(super) par_min_macs: Option<usize>,
    /// [`Tuning::neon_depths`].
    pub(super) neon_depths: u64,
    /// [`Tuning::rendezvous`]: every part of a call waits for all the others
    /// before it does anything else. The rayon pool that runs the call needs
    /// at least as many threads as the call has parts; otherwise the parts
    /// time out (after 60 s, recorded in the run report) and carry on.
    pub(super) rendezvous: bool,
}

#[cfg(test)]
impl TuningOverride {
    /// `tuning` with these replacements.
    fn apply(self, tuning: Tuning) -> Tuning {
        Tuning {
            mc_target: self.mc_target.unwrap_or(tuning.mc_target),
            b_bytes: self.b_bytes.unwrap_or(tuning.b_bytes),
            b_one_block_bytes: self.b_one_block_bytes.unwrap_or(tuning.b_one_block_bytes),
            in_place_bytes: self.in_place_bytes.unwrap_or(tuning.in_place_bytes),
            a_reuse_bytes: self.a_reuse_bytes.unwrap_or(tuning.a_reuse_bytes),
            small_class: self.small_class.unwrap_or(tuning.small_class),
            par_min_macs: self.par_min_macs.unwrap_or(tuning.par_min_macs),
            pack_limit: tuning.pack_limit,
            neon_depths: self.neon_depths,
            rendezvous: self.rendezvous,
        }
    }
}

#[cfg(test)]
thread_local! {
    /// The tuning override of this thread.
    static TUNING_OVERRIDE: Cell<TuningOverride> = const {
        Cell::new(TuningOverride {
            mc_target: None,
            b_bytes: None,
            b_one_block_bytes: None,
            in_place_bytes: None,
            a_reuse_bytes: None,
            small_class: None,
            par_min_macs: None,
            neon_depths: 0,
            rendezvous: false,
        })
    };
}

/// Restores the previous tuning override of its thread when dropped.
///
/// The guard is neither `Send` nor `Sync`: it restores the override of the
/// thread that drops it, so it must be dropped on the thread that created it.
#[cfg(test)]
#[derive(Debug)]
#[must_use = "the tuning is overridden only while the guard lives"]
pub(super) struct TuningGuard {
    /// The override to restore.
    previous: TuningOverride,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

/// Overrides the tuning of the calls made on this thread while the guard
/// lives (test builds only). Read once per call, on the calling thread.
#[cfg(test)]
pub(super) fn override_tuning(tuning: TuningOverride) -> TuningGuard {
    let previous = TUNING_OVERRIDE.with(|cell| cell.replace(tuning));
    TuningGuard {
        previous,
        _not_send: PhantomData,
    }
}

#[cfg(test)]
impl Drop for TuningGuard {
    fn drop(&mut self) {
        TUNING_OVERRIDE.with(|cell| cell.set(self.previous));
    }
}

/// Everything a part needs to compute its piece of C, decided once on the
/// calling thread.
#[derive(Debug, Clone, Copy)]
pub(super) struct Plan<T> {
    /// alpha.
    pub(super) alpha: T,
    /// beta (the effective beta of the first depth block).
    pub(super) beta: T,
    /// Depth.
    pub(super) k: usize,
    /// The KC step (`blocking.kc.min(k)`, rule R1).
    pub(super) kc: usize,
    /// The caller's sequential blocking, for the NEON fallback of a part
    /// (`gemm_blocked_sequential` takes its KC from it: the same partition).
    pub(super) blocking: GemmBlocking,
    /// The route of the call (test builds: the run record reports it).
    #[cfg(test)]
    pub(super) route: Route,
    /// The prefetch switches of every kernel call ([`Class::prefetch`]).
    pub(super) prefetch: u64,
    /// The tuning of the call.
    pub(super) tuning: Tuning,
}

/// The SME2 GEMM: `C = alpha · op(A) · op(B) + beta · C` on SME2, or a
/// decline.
///
/// - `true`: C holds the complete result, bit for bit the NEON blocked
///   result of the same call (`gemm_planned` with the same blocking source)
///   except NaN sign and payload, for every `par` and every part count.
/// - `false`: C is untouched, bit for bit, and the caller computes the
///   product on its usual path.
///
/// `blocking` is the fitted blocking of the call (`gemm::call_blocking_for`); its
/// sequential KC is the depth step. The caller has done the dimension checks
/// and the quick returns (`m == 0 || n == 0`, `k == 0`, `alpha == 0`); such a
/// call that reaches this function anyway declines. See the module docs for
/// the decision and the policy.
#[allow(clippy::too_many_arguments)]
pub(in crate::level3) fn try_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &CallBlocking,
    par: Par,
) -> bool {
    if let (Some(a), Some(b), Some(alpha), Some(beta)) = (
        same_matref::<T, f32>(*a),
        same_matref::<T, f32>(*b),
        same_value::<T, f32>(alpha),
        same_value::<T, f32>(beta),
    ) {
        return same_matmut::<T, f32>(c.rb_mut()).is_some_and(|c| {
            try_gemm_typed(trans_a, trans_b, alpha, a, b, beta, c, blocking, par)
        });
    }
    if let (Some(a), Some(b), Some(alpha), Some(beta)) = (
        same_matref::<T, f64>(*a),
        same_matref::<T, f64>(*b),
        same_value::<T, f64>(alpha),
        same_value::<T, f64>(beta),
    ) {
        return same_matmut::<T, f64>(c.rb_mut()).is_some_and(|c| {
            try_gemm_typed(trans_a, trans_b, alpha, a, b, beta, c, blocking, par)
        });
    }
    false
}

/// [`try_gemm`] for an SME2 element type: the decision of the module docs,
/// then the run.
#[allow(clippy::too_many_arguments)]
fn try_gemm_typed<T: SmeGemm>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    blocking: &CallBlocking,
    par: Par,
) -> bool {
    // 1. Hardware (and the build's caps, and a test's force-off) for `T`.
    if !detect::enabled::<T>() {
        return false;
    }
    let (m, k) = op_dims(&a, trans_a);
    let (kb, n) = op_dims(&b, trans_b);
    if m == 0 || n == 0 || k == 0 || k != kb || c.nrows() != m || c.ncols() != n || alpha == T::ZERO
    {
        return false;
    }

    // 2. The caller's KC partition (R1).
    let blocking = *blocking.sequential();
    let kc = blocking.kc.min(k);
    if !kc_accepted(kc, k) {
        return false;
    }

    // 3. Operands with unit column stride.
    let (Some(a), Some(b)) = (
        ColMajor::normalize(a, trans_a),
        ColMajor::normalize(b, trans_b),
    ) else {
        return false;
    };
    if a.rows() != m || a.cols() != k || b.rows() != k || b.cols() != n {
        return false;
    }

    // 4. Size class (the small class only when it is on), and a C whose
    //    columns do not overlap (thin products are declined, so C has at
    //    least two columns here).
    let tuning = Tuning::of::<T>();
    let class = classify(m, n, k);
    if !class.accepted(tuning.small_class) || c.row_stride() < m {
        return false;
    }

    // 5. The calling thread can enter a streaming region (the only
    //    readiness check of the decision).
    if !detect::thread_ready() {
        return false;
    }

    // 6. Parts.
    let work = m.saturating_mul(n).saturating_mul(k);
    let parts = part_count(work, par, detect::units(), tuning.par_min_macs);

    // 7. Route, then every value of the plan, before the first kernel call.
    let elem = core::mem::size_of::<T>();
    let few_rows_ok = |a_rows: &MatRef<'_, T>, c: &mut MatMut<'_, T>| {
        few_rows_args(a_rows, &b, Span { start: 0, len: n }, alpha, beta, kc, c).is_some()
    };
    let mut route = route_for::<T>(m);
    if route != Route::Blocked {
        route = match a.trans() {
            Trans::NoTrans if few_rows_ok(a.stored(), &mut c) => Route::FewRows { copied: false },
            Trans::NoTrans => Route::Blocked,
            _ if m
                .checked_mul(k)
                .and_then(|len| len.checked_mul(elem))
                .is_some_and(|bytes| bytes <= tuning.pack_limit) =>
            {
                Route::FewRows { copied: true }
            }
            _ => Route::Blocked,
        };
    }
    // `op(A) = Aᵀ` on the few-rows route: the column-major copy, on this
    // thread, in the A slot of its packing cache (plain Rust: no SME2 code).
    let mut a_copy = None;
    if route == (Route::FewRows { copied: true }) {
        let len = m.saturating_mul(k);
        let mut buf = PackBuffer::<T>::take(PackSlot::A, len);
        let fits = copy_transposed(&a, m, k, &mut buf)
            && MatRef::from_column_major(&buf, m, k).is_some_and(|view| few_rows_ok(&view, &mut c));
        if fits {
            a_copy = Some(buf);
        } else {
            route = Route::Blocked;
        }
    }
    let a_rows = match (&route, &a_copy) {
        (Route::FewRows { copied: true }, Some(buf)) => MatRef::from_column_major(buf, m, k),
        (Route::FewRows { copied: false }, _) => Some(*a.stored()),
        _ => None,
    };
    let shape = Shape {
        m,
        n,
        kc,
        a_trans: a.trans(),
        lda: a.stored().row_stride(),
        route,
    };
    let Some(layout) = parallel::Layout::new::<T>(&shape, parts, &tuning) else {
        return false;
    };
    let plan = Plan {
        alpha,
        beta,
        k,
        kc,
        blocking,
        #[cfg(test)]
        route,
        prefetch: class.prefetch(),
        tuning,
    };

    #[cfg(test)]
    super::test_support::note_decision();
    parallel::run(&plan, &a, &b, a_rows.as_ref(), &layout, c);
    drop(a_copy);
    true
}

/// Writes `op(A) = Aᵀ` (stored column-major `k × m`) into `buf` as a
/// column-major `m × k` matrix (`buf[i + p · m] = op(A)(i, p)`); `false`
/// (nothing promised about `buf`) when `buf` is too short or the stored
/// columns are not contiguous.
fn copy_transposed<T: SmeGemm>(a: &ColMajor<'_, T>, m: usize, k: usize, buf: &mut [T]) -> bool {
    if a.trans() == Trans::NoTrans || m.checked_mul(k).is_none_or(|len| buf.len() < len) {
        return false;
    }
    for i in 0..m {
        // Row `i` of op(A) is stored column `i`: `k` contiguous elements.
        let Some(row) = a.stored().col_as_slice(i) else {
            return false;
        };
        for (p, value) in row.iter().take(k).enumerate() {
            buf[i + p * m] = *value;
        }
    }
    true
}

/// The shape of a call as the layout needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Shape {
    /// Rows of C.
    pub(super) m: usize,
    /// Columns of C.
    pub(super) n: usize,
    /// The KC step.
    pub(super) kc: usize,
    /// The transpose of the normalised A (`NoTrans` or `Trans`).
    pub(super) a_trans: Trans,
    /// The leading dimension of the stored A.
    pub(super) lda: usize,
    /// The route.
    pub(super) route: Route,
}

impl Shape {
    /// A blocked call of `m × n` with step `kc` whose A is transposed (every
    /// block packed by the ZA transposer, with an MC-row A buffer).
    #[cfg(test)]
    pub(super) fn blocked(m: usize, n: usize, kc: usize) -> Self {
        Self {
            m,
            n,
            kc,
            a_trans: Trans::Trans,
            lda: kc,
            route: Route::Blocked,
        }
    }
}

/// How a blocked part uses its A buffer (decided once per part from the
/// largest block, `MC × kc`; the source of every block is still decided per
/// block by [`a_source`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AMode {
    /// Every block reads A in place: no A buffer (also the mode of a
    /// few-rows part).
    InPlace,
    /// Blocks that do not read A in place pack it online into a copy of all
    /// row blocks of the part, reused across jc.
    OnlineReuse,
    /// Blocks that do not read A in place pack it online into an MC-row
    /// buffer, rebuilt per `(jc, ic)`.
    OnlineRebuild,
    /// Every block packs `op(A) = Aᵀ` with the ZA transposer into an MC-row
    /// buffer.
    Za,
}

/// The sizes of one part: its rows and columns of C and its block sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Part {
    /// Rows of C (and of `op(A)`) of the part.
    pub(super) rows: Span,
    /// Columns of C (and of `op(B)`) of the part.
    pub(super) cols: Span,
    /// The route of the part (the call's).
    pub(super) route: Route,
    /// MC: `balanced_block(rows.len, mc_target, 64)` (`rows.len` for a
    /// few-rows part).
    pub(super) mc: usize,
    /// NC: `balanced_block(cols.len, nc_target, NR)` with the cap `b_cap`
    /// (`cols.len` for a few-rows part).
    pub(super) nc: usize,
    /// The B cap in bytes ([`b_cap`]; 0 for a few-rows part).
    pub(super) b_cap: usize,
    /// How the part uses its A buffer.
    pub(super) a_mode: AMode,
    /// Elements of the A buffer: 0 ([`AMode::InPlace`]),
    /// `a_pack_len(rows.len, kc)` ([`AMode::OnlineReuse`]) or
    /// `a_pack_len(mc, kc)`.
    pub(super) a_len: usize,
    /// Elements of the B buffer (`b_pack_len(kc, nc)`; 0 for few rows).
    pub(super) b_len: usize,
}

/// What the policy did in one part (test builds only): every count is
/// taken from what was actually passed to the kernels and packers (the
/// argument block of each macro-kernel call, the kernel entries counted
/// around each transposer and few-rows call), never from a flag the driver
/// computes beside the call.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PolicyCount {
    /// Macro-kernel calls that read A in place (no online buffer, A outside
    /// the A buffer).
    pub(super) a_in_place: usize,
    /// Macro-kernel calls that packed A online into the copy reused across
    /// jc ([`AMode::OnlineReuse`]).
    pub(super) a_online_fill: usize,
    /// Macro-kernel calls that packed A online into the rebuilt MC-row
    /// buffer ([`AMode::OnlineRebuild`]).
    pub(super) a_online_rebuilt: usize,
    /// Macro-kernel calls that read a reused online copy of `op(A) = A`
    /// from the A buffer.
    pub(super) a_reused: usize,
    /// Macro-kernel calls that read `op(A) = Aᵀ` packed by the ZA
    /// transposer from the A buffer.
    pub(super) a_za_reads: usize,
    /// ZA-transposer kernel entries of `pack_a_za`.
    pub(super) a_za: usize,
    /// ZA-transposer kernel entries of `pack_b_za`.
    pub(super) b_za: usize,
    /// Calls of the plain `pack_b` (`op(B) = Bᵀ`).
    pub(super) b_plain: usize,
    /// Macro-kernel calls with `PREFETCH_C`.
    pub(super) prefetch: usize,
    /// Few-rows regions (`few_rows::regions_on_this_thread` delta).
    pub(super) few_rows: usize,
}

/// How the blocks of one part ran: SME2 calls that compute C and blocks
/// computed by the NEON macro-panel instead. A part of a validated layout
/// runs every block on SME2 (`neon == 0`); the count lets the tests prove it,
/// because a NEON block gives the same bits and would otherwise go unnoticed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct BlockCount {
    /// SME2 calls that compute C: macro-kernel blocks (one call each), or
    /// the one `few_rows_gemm` call of a few-rows part.
    pub(super) sme: usize,
    /// Blocks computed on NEON (a macro-panel block, or the whole piece of a
    /// few-rows part the kernel refused).
    pub(super) neon: usize,
    /// Kernel entries counted by the kernels themselves on the thread that
    /// ran the part, over the part's run (test builds only; recorded by
    /// `parallel::run`). Unlike `sme`, which is the driver's own number, it
    /// moves only when a kernel really ran: `sme + policy.a_za +
    /// policy.b_za` when everything ran as recorded.
    #[cfg(test)]
    pub(super) kernel_calls: u64,
    /// What the policy did (test builds only).
    #[cfg(test)]
    pub(super) policy: PolicyCount,
}

/// The A buffer of a part and what it holds in the current depth block.
struct ABuffer<T: GemmKernel> {
    /// The buffer (from the part's packing cache).
    buf: PackBuffer<T>,
    /// The A mode of the part.
    mode: AMode,
    /// For [`AMode::OnlineReuse`]: whether the region of row block `ic`
    /// holds the online copy of the current depth block (empty otherwise).
    filled: Vec<bool>,
}

/// The sequential driver: computes the piece `rows × cols` of C (the whole
/// of `c`, a view of exactly that piece) with one kernel call per
/// `(depth, column, row)` block, in the order of the module docs, and
/// returns how the blocks ran.
///
/// The caller has checked `detect::thread_ready()` on this thread, and the
/// part comes from a validated [`parallel::Layout`] of `plan` (blocked
/// route).
pub(super) fn run_part<T: SmeGemm>(
    plan: &Plan<T>,
    a: &ColMajor<'_, T>,
    b: &ColMajor<'_, T>,
    part: &Part,
    mut c: MatMut<'_, T>,
) -> BlockCount {
    let mut count = BlockCount::default();
    let row_blocks = part.rows.len.div_ceil(part.mc.max(1));
    let mut buf_a = ABuffer {
        buf: PackBuffer::<T>::take(PackSlot::A, part.a_len),
        mode: part.a_mode,
        filled: if part.a_mode == AMode::OnlineReuse {
            vec![false; row_blocks]
        } else {
            Vec::new()
        },
    };
    let mut buf_b = PackBuffer::<T>::take(PackSlot::B, part.b_len);
    for depth in spans(plan.k, plan.kc) {
        let beta_eff = block_beta(depth, plan.beta);
        buf_a.filled.iter_mut().for_each(|filled| *filled = false);
        for cols in spans(part.cols.len, part.nc) {
            let global_cols = Span {
                start: part.cols.start + cols.start,
                len: cols.len,
            };
            // SAFETY: the caller checked `detect::thread_ready()` on this
            // thread, and the decision checked `detect::enabled::<T>()`.
            let packed_b = unsafe { pack_b_block(b, depth, global_cols, &mut buf_b, &mut count) };
            for (row_block, rows) in spans(part.rows.len, part.mc).enumerate() {
                let global_rows = Span {
                    start: part.rows.start + rows.start,
                    len: rows.len,
                };
                let block = Block {
                    rows: global_rows,
                    local_rows: rows,
                    row_block,
                    depth,
                    at: (rows.start, cols.start),
                    beta_eff,
                };
                let done = packed_b.as_ref().is_some_and(|packed_b| {
                    // SAFETY: as for `pack_b_block` above.
                    unsafe { sme_block(plan, a, packed_b, &block, &mut buf_a, &mut c, &mut count) }
                });
                if done {
                    count.sme += 1;
                } else {
                    count.neon += 1;
                    neon_block(
                        plan,
                        a,
                        b,
                        (global_rows, depth, global_cols),
                        c.rb_mut()
                            .submatrix(rows.start, cols.start, rows.len, cols.len),
                        beta_eff,
                    );
                }
            }
        }
    }
    count
}

/// One few-rows part: `few_rows_gemm` on the part's columns with the call's
/// A view `a_rows` (column-major `m × k`), whole depth, the call's KC
/// step. A piece the kernel refuses (it validates its arguments; the
/// decision validated the same strides) is computed by the NEON sequential
/// driver instead.
///
/// The caller has checked `detect::thread_ready()` on this thread, and the
/// part comes from a validated few-rows [`parallel::Layout`] of `plan`.
pub(super) fn run_few_rows<T: SmeGemm>(
    plan: &Plan<T>,
    a: &ColMajor<'_, T>,
    a_rows: Option<&MatRef<'_, T>>,
    b: &ColMajor<'_, T>,
    part: &Part,
    mut c: MatMut<'_, T>,
) -> BlockCount {
    let mut count = BlockCount::default();
    #[cfg(test)]
    let regions = super::few_rows::regions_on_this_thread();
    let done = match a_rows {
        // SAFETY: `few_rows_gemm`'s contract holds: the caller checked
        // `detect::thread_ready()` on this thread and the decision checked
        // `detect::enabled::<T>()`; C (an exclusive view) overlaps neither A
        // (the shared stored A, or the decision's own copy) nor B (shared).
        Some(a_rows) => unsafe {
            few_rows_gemm(
                *a_rows,
                b,
                part.cols,
                plan.alpha,
                plan.beta,
                plan.kc,
                c.rb_mut(),
            )
        },
        None => false,
    };
    #[cfg(test)]
    {
        let ran = super::few_rows::regions_on_this_thread().wrapping_sub(regions);
        count.policy.few_rows = usize::try_from(ran).unwrap_or(usize::MAX);
    }
    if done {
        count.sme = 1;
    } else {
        count.neon = 1;
        parallel::neon_part(plan, a, b, part, c);
    }
    count
}

/// Packs the B block `op(B)[depth, cols]` into `buf`: with the ZA
/// transposer for `op(B) = B`, with the plain packer for `op(B) = Bᵀ`.
///
/// # Safety
///
/// `detect::thread_ready()` returned `true` on this thread and
/// `detect::enabled::<T>()` holds (the transposer's contract).
unsafe fn pack_b_block<'p, T: SmeGemm>(
    b: &ColMajor<'_, T>,
    depth: Span,
    cols: Span,
    buf: &'p mut AlignedVec<T>,
    count: &mut BlockCount,
) -> Option<PackedB<'p, T>> {
    match b.trans() {
        Trans::NoTrans => {
            #[cfg(test)]
            let entries = super::test_support::kernel_entries_here();
            // SAFETY: this function's contract is the transposer's.
            let packed = unsafe { pack_b_za(b, depth, cols, buf) };
            #[cfg(test)]
            {
                let ran = super::test_support::kernel_entries_here().wrapping_sub(entries);
                count.policy.b_za += usize::try_from(ran).unwrap_or(usize::MAX);
            }
            #[cfg(not(test))]
            let _ = count;
            packed
        }
        Trans::Trans | Trans::ConjTrans => {
            let packed = pack_b(b, depth, cols, buf);
            #[cfg(test)]
            {
                count.policy.b_plain += usize::from(packed.is_some());
            }
            #[cfg(not(test))]
            let _ = count;
            packed
        }
    }
}

/// One `(pc, jc, ic)` block of a blocked part.
#[derive(Debug, Clone, Copy)]
struct Block<T> {
    /// Rows of `op(A)` and C (global).
    rows: Span,
    /// Rows within the part.
    local_rows: Span,
    /// Index of the row block within the part.
    row_block: usize,
    /// The depth block.
    depth: Span,
    /// The block's position (row, column) within the part's view of C.
    at: (usize, usize),
    /// The effective beta of the depth block.
    beta_eff: T,
}

/// One block on SME2: takes `op(A)[rows, depth]` from the source the policy
/// gives the block and runs the kernel on the block of `c`. `false`, before
/// any write to C, when a packer or an argument check refuses (the caller
/// then computes the block on NEON).
///
/// # Safety
///
/// `detect::thread_ready()` returned `true` on this thread and
/// `detect::enabled::<T>()` holds.
unsafe fn sme_block<T: SmeGemm>(
    plan: &Plan<T>,
    a: &ColMajor<'_, T>,
    packed_b: &PackedB<'_, T>,
    block: &Block<T>,
    buf_a: &mut ABuffer<T>,
    c: &mut MatMut<'_, T>,
    count: &mut BlockCount,
) -> bool {
    #[cfg(test)]
    {
        let index = block.depth.start / plan.kc;
        if index < 64 && plan.tuning.neon_depths & (1 << index) != 0 {
            return false;
        }
    }
    let (rows, depth) = (block.rows, block.depth);
    let elem = core::mem::size_of::<T>();
    let lda = a.stored().row_stride();
    let source = a_source(
        a.trans(),
        lda,
        rows.len,
        depth.len,
        elem,
        plan.tuning.in_place_bytes,
    );
    let args = match source {
        ASource::Za => {
            #[cfg(test)]
            let entries = super::test_support::kernel_entries_here();
            // SAFETY: this function's contract is the transposer's.
            let packed = unsafe { pack_a_za(a, rows, depth, &mut buf_a.buf) };
            #[cfg(test)]
            {
                let ran = super::test_support::kernel_entries_here().wrapping_sub(entries);
                count.policy.a_za += usize::try_from(ran).unwrap_or(usize::MAX);
            }
            let Some(packed_a) = packed else {
                return false;
            };
            block_args(
                &packed_a,
                packed_b,
                c,
                block.at.0,
                block.at.1,
                plan.alpha,
                block.beta_eff,
            )
        }
        ASource::InPlace => in_place_args(plan, a, packed_b, block, c),
        ASource::Online => {
            let len = a_pack_len(rows.len, depth.len).unwrap_or(usize::MAX);
            let (offset, reuse) = match buf_a.mode {
                // The region of row block `ic` in the copy of the depth
                // block: row blocks start at multiples of 64 (MC is a
                // multiple of 64 when the part has several row blocks), so
                // the copy is the plain packed block of all rows.
                AMode::OnlineReuse => (block.local_rows.start.saturating_mul(depth.len), true),
                AMode::OnlineRebuild => (0, false),
                AMode::InPlace | AMode::Za => return false,
            };
            let Some(region) = offset
                .checked_add(len)
                .and_then(|end| buf_a.buf.get_mut(offset..end))
            else {
                return false;
            };
            let filled = reuse && buf_a.filled.get(block.row_block).copied().unwrap_or(false);
            if filled {
                // A later column block: the kernel reads the copy that the
                // first one packed in this depth block.
                PackedA::from_packed(region, rows.len, depth.len).and_then(|packed_a| {
                    block_args(
                        &packed_a,
                        packed_b,
                        c,
                        block.at.0,
                        block.at.1,
                        plan.alpha,
                        block.beta_eff,
                    )
                })
            } else {
                in_place_args(plan, a, packed_b, block, c)
                    .map(|args| args.with_online_pack(region.as_mut_ptr()))
            }
        }
    };
    let Some(args) = args else {
        return false;
    };
    let args = args.with_prefetch(plan.prefetch);
    #[cfg(test)]
    count.policy.note_call(&args, buf_a, a.trans());
    #[cfg(not(test))]
    let _ = count;
    // SAFETY: the kernel's contract holds:
    // - `detect::enabled::<T>()` and `detect::thread_ready()` on this thread
    //   (this function's contract);
    // - the block was built by `block_args` (from packed blocks of the same
    //   depth) or `in_place_args`, both with `MacroArgs::packed` (the mode is
    //   `Epilogue::select(alpha, beta_eff)`), `m, n, kc >= 1`, the C block
    //   inside `c` with `ldc >= m` (or a single column);
    // - A: a packed block holds whole zero-padded panels in the A buffer; an
    //   in-place A is the stored matrix at `(rows.start, depth.start)`, and
    //   `in_place_args` checked that the block lies inside it, so every
    //   `a[i + p · lda]` with `i < m`, `p < kc` is an element of the view;
    //   B is a whole packed block in the B buffer;
    // - an online buffer (`a_pack`) is a region of exactly
    //   `a_pack_len(m, kc)` elements of the A buffer, which is an
    //   allocation of its own (it overlaps neither A, B, C nor the block);
    // - C is borrowed exclusively, so it overlaps neither A nor B (shared
    //   views) nor the buffers;
    // - `args` is used while the buffers, A, B and `c` are borrowed.
    unsafe { <T as SmeGemm>::kernel(&args) };
    if source == ASource::Online
        && let Some(filled) = buf_a.filled.get_mut(block.row_block)
    {
        *filled = true;
    }
    true
}

/// The argument block of a kernel call that reads `op(A)[rows, depth]` in
/// place from the stored matrix (`op(A) = A`), with the packed B block
/// `packed_b` and the block of `c` at `block.at`; `None` when the block
/// leaves A or C, the depths differ, or C's columns overlap.
fn in_place_args<T: SmeGemm>(
    plan: &Plan<T>,
    a: &ColMajor<'_, T>,
    packed_b: &PackedB<'_, T>,
    block: &Block<T>,
    c: &mut MatMut<'_, T>,
) -> Option<MacroArgs<T>> {
    let (rows, depth) = (block.rows, block.depth);
    let stored = a.stored();
    let inside = |span: Span, total: usize| {
        span.len > 0
            && span
                .start
                .checked_add(span.len)
                .is_some_and(|end| end <= total)
    };
    if a.trans() != Trans::NoTrans
        || !inside(rows, stored.nrows())
        || !inside(depth, stored.ncols())
        || packed_b.depth() != depth.len
    {
        return None;
    }
    let (m, n) = (rows.len, packed_b.cols());
    let (row0, col0) = block.at;
    if !inside(
        Span {
            start: row0,
            len: m,
        },
        c.nrows(),
    ) || !inside(
        Span {
            start: col0,
            len: n,
        },
        c.ncols(),
    ) {
        return None;
    }
    let ldc = if n == 1 {
        c.row_stride().max(m)
    } else if c.row_stride() >= m {
        c.row_stride()
    } else {
        return None;
    };
    let a_ptr = stored.ptr_at(rows.start, depth.start);
    Some(
        MacroArgs::packed(
            a_ptr,
            packed_b.data().as_ptr(),
            c.ptr_at_mut(row0, col0),
            ldc,
            m,
            n,
            depth.len,
            plan.alpha,
            block.beta_eff,
        )
        .with_a_in_place(a_ptr, stored.row_stride()),
    )
}

#[cfg(test)]
impl PolicyCount {
    /// Records one macro-kernel call from its argument block: where A came
    /// from (the online buffer pointer, and whether A points into the A
    /// buffer) and the prefetch switches.
    fn note_call<T: GemmKernel>(&mut self, args: &MacroArgs<T>, buf_a: &ABuffer<T>, trans: Trans) {
        let range = buf_a.buf.as_ptr_range();
        let from_buffer = range.contains(&args.a);
        if !args.a_pack.is_null() {
            match buf_a.mode {
                AMode::OnlineReuse => self.a_online_fill += 1,
                _ => self.a_online_rebuilt += 1,
            }
        } else if from_buffer {
            match trans {
                Trans::NoTrans => self.a_reused += 1,
                Trans::Trans | Trans::ConjTrans => self.a_za_reads += 1,
            }
        } else {
            self.a_in_place += 1;
        }
        self.prefetch += usize::from(args.prefetch & PREFETCH_C != 0);
    }
}

/// One block on NEON: `C[rows, cols] = alpha · op(A)[rows, depth] ·
/// op(B)[depth, cols] + beta_eff · C[rows, cols]` with the canonical
/// micro-kernel and the NEON packers, the arithmetic of one block of the
/// NEON blocked driver (the same bits as the kernel's). `c` is the view of
/// that block.
fn neon_block<T: SmeGemm>(
    plan: &Plan<T>,
    a: &ColMajor<'_, T>,
    b: &ColMajor<'_, T>,
    (rows, depth, cols): (Span, Span, Span),
    c: MatMut<'_, T>,
    beta_eff: T,
) {
    let kernel = T::micro_kernel_dispatch();
    let (mr, nr) = (kernel.shape.mr, kernel.shape.nr);
    let a_len = rows
        .len
        .div_ceil(mr)
        .saturating_mul(mr)
        .saturating_mul(depth.len);
    let b_len = cols
        .len
        .div_ceil(nr)
        .saturating_mul(nr)
        .saturating_mul(depth.len);
    let mut buf_a = PackBuffer::<T>::take(PackSlot::A, a_len);
    let mut buf_b = PackBuffer::<T>::take(PackSlot::B, b_len);
    pack_a_trans_dispatch(
        a.stored(),
        a.trans(),
        rows.start,
        depth.start,
        rows.len,
        depth.len,
        &mut buf_a,
        mr,
    );
    pack_b_trans_dispatch(
        b.stored(),
        b.trans(),
        depth.start,
        cols.start,
        depth.len,
        cols.len,
        &mut buf_b,
        nr,
    );
    macro_panel_multiply_with(
        plan.alpha, &buf_a, rows.len, depth.len, &buf_b, cols.len, beta_eff, c, &kernel,
    );
}
