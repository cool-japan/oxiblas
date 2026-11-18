//! The split of an SME2 GEMM over SME units, and the NEON fallback of a part.
//!
//! [`Layout::new`] cuts C into `parts` pieces, at 64-row or `NR`-column
//! boundaries, into near-equal runs of whole panels (no piece is empty: the
//! part count is clamped to the number of panels on that side): along the
//! longer side (rows when `m >= n`, columns otherwise) on the blocked route,
//! along the columns on the few-rows route. It sizes the blocks, the B cap,
//! the A mode and the buffers of every blocked piece with checked
//! arithmetic, against the packing-cache limit (the policy of `driver.rs`).
//! It runs during the decision, before anything touches C, so a layout that
//! cannot be built declines the call.
//!
//! [`run`] splits the exclusive view of C into one disjoint `MatMut` per
//! piece with the consuming `split_rows` / `split_cols` (no raw pointers, no
//! aliasing), and computes the pieces: one piece on the calling thread, more
//! on rayon (one task per piece, on the ambient pool). Every piece checks
//! `detect::thread_ready()` on the thread that runs it and only then enters
//! SME2 code: the sequential driver (`driver::run_part`, with its own packing
//! buffers from that thread's cache: macro-kernels, ZA transposers, online
//! packing) or one few-rows call (`driver::run_few_rows`). A piece whose
//! thread is not ready (a rayon worker
//! that is itself inside SME code, say) is computed by the NEON sequential
//! driver, `gemm_blocked_sequential` with the caller's sequential blocking
//! and the canonical micro-kernel, on the matching sub-views of `op(A)` and
//! `op(B)`: the same KC partition, so the same bits.
//!
//! The SME2 decision and `kc` are taken once, on the calling thread, and
//! passed down in the plan: a piece never asks `detect::enabled` again (the
//! test force-off switch is thread-local, rayon workers would not see it).
//!
//! Test builds can mark chosen piece indices as "not ready"
//! (`mark_not_ready`, read on the calling thread), so that the NEON
//! fallback of a piece runs under QEMU, and they can read back what the
//! last call on the thread did (`take_last_run`): for every piece whether
//! it ran on NEON, its block counts and the kernel entries counted on its
//! thread over its run, the `ThreadId` and the rayon worker index of the
//! thread that ran it, and whether it gave up waiting at the rendezvous.
//! With the rendezvous of the test tuning (`TuningOverride::rendezvous`)
//! every piece waits, before it does anything else (before its
//! `thread_ready()` check), until all pieces of the call have arrived; that
//! puts the pieces on distinct threads without relying on the scheduler. The
//! pool needs at least as many threads as the call has pieces; a piece that
//! waits longer than `RENDEZVOUS_TIMEOUT` (60 s) records the timeout and carries
//! on, and the pieces that arrive after it do not wait, so a broken
//! rendezvous costs one timeout and never hangs a test run.

#[cfg(test)]
use std::cell::{Cell, RefCell};
#[cfg(test)]
use std::marker::PhantomData;
#[cfg(test)]
use std::sync::{Condvar, Mutex};
#[cfg(test)]
use std::thread::ThreadId;
#[cfg(test)]
use std::time::{Duration, Instant};

use oxiblas_matrix::{MatMut, MatRef};

use super::args::{MR, SmeElement};
use super::detect;
use super::driver::{
    AMode, BlockCount, Part, Plan, Route, Shape, SmeGemm, Tuning, a_reuse_fits, b_cap,
    in_place_fits, run_few_rows, run_part,
};
use super::pack::{ColMajor, Span, a_pack_len, b_pack_len};
use crate::level3::autotune::balanced_block;
use crate::level3::gemm::gemm_blocked_sequential;
use crate::level3::trsm::Trans;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// The side of C the parts are cut along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Split {
    /// Pieces of whole 64-row panels, each with all columns (`m >= n`).
    Rows,
    /// Pieces of whole `NR`-column panels, each with all rows (`m < n`).
    Cols,
}

/// The pieces of C of one call and their block sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Layout {
    /// The side the pieces are cut along.
    pub(super) split: Split,
    /// The pieces, in order; they tile C exactly.
    pub(super) parts: Vec<Part>,
}

impl Layout {
    /// The layout of the call `shape` in (at most) `parts` pieces, or `None`
    /// when a size overflows or the buffers of a piece do not fit the
    /// packing-cache limit (`tuning.pack_limit`). `m, n, kc >= 1`.
    ///
    /// - Blocked route: the pieces are cut along the longer side of C (rows
    ///   when `m >= n`); every piece gets its MC, B cap, NC and A mode (see
    ///   `driver.rs`, "The policy").
    /// - Few-rows route: the pieces are cut along the columns, whatever the
    ///   shape; a piece is one few-rows call and has no buffer.
    pub(super) fn new<T: SmeElement>(shape: &Shape, parts: usize, tuning: &Tuning) -> Option<Self> {
        let (m, n, kc) = (shape.m, shape.n, shape.kc);
        if m == 0 || n == 0 || kc == 0 {
            return None;
        }
        let split = match shape.route {
            Route::FewRows { .. } => Split::Cols,
            Route::Blocked if m >= n => Split::Rows,
            Route::Blocked => Split::Cols,
        };
        let (len, unit) = match split {
            Split::Rows => (m, MR),
            Split::Cols => (n, T::NR),
        };
        let count = parts.clamp(1, len.div_ceil(unit));
        let piece = |index: usize| -> Option<Part> {
            let start = split_point(index, count, len, unit)?;
            let end = split_point(index + 1, count, len, unit)?;
            if end <= start {
                return None;
            }
            let piece = Span {
                start,
                len: end - start,
            };
            let (rows, cols) = match split {
                Split::Rows => (piece, Span { start: 0, len: n }),
                Split::Cols => (Span { start: 0, len: m }, piece),
            };
            match shape.route {
                Route::FewRows { .. } => Some(Part {
                    rows,
                    cols,
                    route: shape.route,
                    mc: rows.len,
                    nc: cols.len,
                    b_cap: 0,
                    a_mode: AMode::InPlace,
                    a_len: 0,
                    b_len: 0,
                }),
                Route::Blocked => part_sizes::<T>(shape, rows, cols, tuning),
            }
        };
        // Every checked split and size first, without allocating, so that a
        // decline (`None`) happens before anything is allocated; then the
        // pieces again (the same pure arithmetic, which cannot fail now).
        for index in 0..count {
            piece(index)?;
        }
        let mut pieces = Vec::with_capacity(count);
        for index in 0..count {
            pieces.push(piece(index)?);
        }
        Some(Self {
            split,
            parts: pieces,
        })
    }
}

/// Start of piece `index` (`0..=parts`) when `len` elements, `⌈len / unit⌉`
/// panels, are cut into `parts` near-equal runs of whole panels; `len` for
/// `index == parts`. Every start below `len` is a multiple of `unit`, and
/// every piece is non-empty when `1 <= parts <= ⌈len / unit⌉`.
fn split_point(index: usize, parts: usize, len: usize, unit: usize) -> Option<usize> {
    let panels = u128::try_from(len.div_ceil(unit)).ok()?;
    let index = u128::try_from(index).ok()?;
    let parts = u128::try_from(parts).ok()?;
    let panel = usize::try_from(index.checked_mul(panels)?.checked_div(parts)?).ok()?;
    Some(panel.saturating_mul(unit).min(len))
}

/// The block sizes, A mode and buffer lengths of the blocked piece
/// `rows × cols` of `shape`, or `None` when one overflows or the buffers do
/// not fit the packing-cache limit.
///
/// - MC = `balanced_block(rows, mc_target, 64)`; the B cap from
///   `driver::b_cap`; NC the balanced block of the widest multiple of NR
///   whose packed B block (`kc × NC`) fits the cap (at least NR).
/// - The A mode from the largest block (`MC × kc`): `Aᵀ` → ZA (an MC-row
///   buffer); `A` read in place by every block → no buffer; otherwise online
///   packing, with a copy of all rows reused across jc when it is at most
///   `a_reuse_bytes` and fits the packing cache next to B, else rebuilt in an
///   MC-row buffer.
/// - The A and B buffers together must fit `tuning.pack_limit`.
fn part_sizes<T: SmeElement>(
    shape: &Shape,
    rows: Span,
    cols: Span,
    tuning: &Tuning,
) -> Option<Part> {
    let elem = core::mem::size_of::<T>();
    let kc = shape.kc;
    let mc = balanced_block(rows.len, tuning.mc_target, MR);
    let cap = b_cap::<T>(rows.len, mc, cols.len, kc, tuning)?;
    // The widest multiple of NR whose packed B block (kc × NC) fits the cap.
    let nc_fit = cap / kc.checked_mul(elem)?;
    let nc_target = (nc_fit / T::NR * T::NR).max(T::NR);
    let nc = balanced_block(cols.len, nc_target, T::NR);
    let b_len = b_pack_len::<T>(kc, nc)?;
    let b_bytes = b_len.checked_mul(elem)?;
    // Both buffers together within the packing-cache limit (and so
    // representable allocations).
    let limit = tuning
        .pack_limit
        .min(usize::try_from(isize::MAX).unwrap_or(usize::MAX));
    let fits = |a_len: usize| -> Option<bool> {
        Some(a_len.checked_mul(elem)?.checked_add(b_bytes)? <= limit)
    };
    let (a_mode, a_len) = match shape.a_trans {
        Trans::NoTrans if in_place_fits(shape.lda, mc, kc, elem, tuning.in_place_bytes) => {
            (AMode::InPlace, 0)
        }
        Trans::NoTrans => {
            let whole = a_pack_len(rows.len, kc)?;
            if a_reuse_fits(rows.len, kc, elem, tuning.a_reuse_bytes) && fits(whole)? {
                (AMode::OnlineReuse, whole)
            } else {
                (AMode::OnlineRebuild, a_pack_len(mc, kc)?)
            }
        }
        Trans::Trans | Trans::ConjTrans => (AMode::Za, a_pack_len(mc, kc)?),
    };
    if !fits(a_len)? {
        return None;
    }
    Some(Part {
        rows,
        cols,
        route: Route::Blocked,
        mc,
        nc,
        b_cap: cap,
        a_mode,
        a_len,
        b_len,
    })
}

/// Computes the product of `plan` into `c` (all of C), piece by piece as
/// `layout` says. `c` has `row_stride >= m` (checked in the decision), so
/// the splits never panic.
///
/// `a_rows` is `op(A)` as a column-major `m × k` view for the few-rows
/// route (the stored A, or the decision's copy), `None` otherwise.
pub(super) fn run<T: SmeGemm>(
    plan: &Plan<T>,
    a: &ColMajor<'_, T>,
    b: &ColMajor<'_, T>,
    a_rows: Option<&MatRef<'_, T>>,
    layout: &Layout,
    c: MatMut<'_, T>,
) {
    let not_ready = not_ready_mask();
    #[cfg(test)]
    let rendezvous = plan
        .tuning
        .rendezvous
        .then(|| Rendezvous::new(layout.parts.len()));
    let views = split_views(c, layout);
    let tasks: Vec<(usize, &Part, MatMut<'_, T>)> = layout
        .parts
        .iter()
        .zip(views)
        .enumerate()
        .map(|(index, (part, view))| (index, part, view))
        .collect();
    let run_one = |(index, part, view): (usize, &Part, MatMut<'_, T>)| {
        // Test builds: the rendezvous comes before anything else, and the
        // kernel entries of this thread are counted over the whole piece.
        #[cfg(test)]
        let (entries, timed_out) = (
            super::test_support::kernel_entries_here(),
            rendezvous.as_ref().is_some_and(Rendezvous::arrive),
        );
        let forced = index < 64 && not_ready & (1 << index) != 0;
        // (Fields are evaluated in order: `place` after the piece ran.)
        // Every SME2 entry of the piece (macro-kernels, ZA transposers, few
        // rows) runs after this thread's own readiness check.
        if !forced && detect::thread_ready() {
            let blocks = match part.route {
                Route::Blocked => run_part(plan, a, b, part, view),
                Route::FewRows { .. } => run_few_rows(plan, a, a_rows, b, part, view),
            };
            PartRun {
                neon: false,
                blocks,
                #[cfg(test)]
                place: Place::here(entries, timed_out),
            }
        } else {
            neon_part(plan, a, b, part, view);
            PartRun {
                neon: true,
                blocks: BlockCount::default(),
                #[cfg(test)]
                place: Place::here(entries, timed_out),
            }
        }
    };
    #[cfg(feature = "parallel")]
    let runs: Vec<PartRun> = if tasks.len() > 1 {
        tasks.into_par_iter().with_max_len(1).map(run_one).collect()
    } else {
        tasks.into_iter().map(run_one).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let runs: Vec<PartRun> = tasks.into_iter().map(run_one).collect();
    #[cfg(test)]
    LAST_RUN.with(|last| {
        *last.borrow_mut() = Some(RunReport {
            route: plan.route,
            prefetch: plan.prefetch,
            layout: layout.clone(),
            neon: runs.iter().map(|run| run.neon).collect(),
            blocks: runs
                .iter()
                .map(|run| BlockCount {
                    kernel_calls: run.place.kernel_calls,
                    ..run.blocks
                })
                .collect(),
            threads: runs.iter().map(|run| run.place.thread).collect(),
            workers: runs.iter().map(|run| run.place.worker).collect(),
            timed_out: runs.iter().map(|run| run.place.timed_out).collect(),
        });
    });
    // A validated layout runs every block of an SME2 piece on SME2 (the
    // block fallback is only a safety net); test builds force NEON blocks
    // on purpose and check the counts instead.
    #[cfg(not(test))]
    debug_assert!(
        runs.iter()
            .all(|run| run.neon || (run.blocks.neon == 0 && run.blocks.sme > 0)),
        "an SME2 piece of a validated layout ran a block on NEON"
    );
}

/// How one piece ran.
#[derive(Debug, Clone, Copy)]
struct PartRun {
    /// The piece ran on the NEON sequential driver (its thread was not
    /// ready).
    neon: bool,
    /// The blocks of the SME2 sequential driver (zero for a NEON piece).
    blocks: BlockCount,
    /// Where the piece ran (test builds only).
    #[cfg(test)]
    place: Place,
}

/// The thread that ran a piece, and how its rendezvous went (test builds
/// only).
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct Place {
    /// The thread.
    thread: ThreadId,
    /// `rayon::current_thread_index()` of the thread (`None` outside rayon).
    worker: Option<usize>,
    /// The piece gave up waiting at the rendezvous.
    timed_out: bool,
    /// Kernel entries on the thread over the piece's run.
    kernel_calls: u64,
}

#[cfg(test)]
impl Place {
    /// The calling thread, which has just run the piece: `entries` was its
    /// kernel-entry count when the piece started.
    fn here(entries: u64, timed_out: bool) -> Self {
        #[cfg(feature = "parallel")]
        let worker = rayon::current_thread_index();
        #[cfg(not(feature = "parallel"))]
        let worker = None;
        Self {
            thread: std::thread::current().id(),
            worker,
            timed_out,
            kernel_calls: super::test_support::kernel_entries_here().wrapping_sub(entries),
        }
    }
}

/// How long a piece waits at the rendezvous before it gives up (test builds
/// only).
#[cfg(test)]
pub(super) const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(60);

/// The meeting point of the pieces of one call (test builds only; see the
/// module docs).
#[cfg(test)]
#[derive(Debug)]
struct Rendezvous {
    /// The number of pieces.
    parts: usize,
    /// Pieces arrived so far, and whether a piece gave up waiting.
    state: Mutex<(usize, bool)>,
    /// Signalled when the last piece arrives or a piece gives up.
    arrived: Condvar,
}

#[cfg(test)]
impl Rendezvous {
    /// A rendezvous of `parts` pieces.
    fn new(parts: usize) -> Self {
        Self {
            parts,
            state: Mutex::new((0, false)),
            arrived: Condvar::new(),
        }
    }

    /// Arrives and waits until every piece has arrived. `true` when the
    /// piece stopped waiting before that: it waited [`RENDEZVOUS_TIMEOUT`],
    /// or an earlier piece had already given up (then it does not wait).
    fn arrive(&self) -> bool {
        let deadline = Instant::now() + RENDEZVOUS_TIMEOUT;
        let Ok(mut state) = self.state.lock() else {
            return true;
        };
        state.0 += 1;
        if state.0 >= self.parts {
            self.arrived.notify_all();
            return false;
        }
        while state.0 < self.parts && !state.1 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                state.1 = true;
                self.arrived.notify_all();
                break;
            }
            state = match self.arrived.wait_timeout(state, left) {
                Ok((state, _)) => state,
                Err(_) => return true,
            };
        }
        state.0 < self.parts
    }
}

/// One disjoint view of `c` per piece of `layout`, in order.
fn split_views<'c, T: SmeGemm>(c: MatMut<'c, T>, layout: &Layout) -> Vec<MatMut<'c, T>> {
    let mut views = Vec::with_capacity(layout.parts.len());
    let mut rest = Some(c);
    let last = layout.parts.len().saturating_sub(1);
    for (index, part) in layout.parts.iter().enumerate() {
        let Some(view) = rest.take() else {
            break;
        };
        if index == last {
            views.push(view);
            break;
        }
        let (head, tail) = match layout.split {
            Split::Rows => view.split_rows(part.rows.len),
            Split::Cols => view.split_cols(part.cols.len),
        };
        views.push(head);
        rest = Some(tail);
    }
    views
}

/// A piece on NEON: `gemm_blocked_sequential` on `op(A)[rows, :]`,
/// `op(B)[:, cols]` and the piece's view of C, with the caller's sequential
/// blocking (the KC partition of the plan) and the canonical micro-kernel.
pub(super) fn neon_part<T: SmeGemm>(
    plan: &Plan<T>,
    a: &ColMajor<'_, T>,
    b: &ColMajor<'_, T>,
    part: &Part,
    mut view: MatMut<'_, T>,
) {
    // op(A)[rows, 0..k] and op(B)[0..k, cols] as sub-views of the stored
    // (column-major) matrices: rows of op(X) are columns of the stored
    // matrix when op is a transpose.
    let a_sub = match a.trans() {
        Trans::NoTrans => a
            .stored()
            .submatrix(part.rows.start, 0, part.rows.len, plan.k),
        _ => a
            .stored()
            .submatrix(0, part.rows.start, plan.k, part.rows.len),
    };
    let b_sub = match b.trans() {
        Trans::NoTrans => b
            .stored()
            .submatrix(0, part.cols.start, plan.k, part.cols.len),
        _ => b
            .stored()
            .submatrix(part.cols.start, 0, part.cols.len, plan.k),
    };
    gemm_blocked_sequential(
        a.trans(),
        b.trans(),
        plan.alpha,
        &a_sub,
        &b_sub,
        plan.beta,
        &mut view,
        &plan.blocking,
        &T::micro_kernel_dispatch(),
    );
}

#[cfg(test)]
thread_local! {
    /// Piece indices (bit `i` = piece `i`) that run as "not ready".
    static NOT_READY: Cell<u64> = const { Cell::new(0) };
    /// What the last call on this thread did.
    static LAST_RUN: RefCell<Option<RunReport>> = const { RefCell::new(None) };
}

/// The "not ready" mask of the calling thread (always 0 outside tests).
fn not_ready_mask() -> u64 {
    #[cfg(test)]
    {
        NOT_READY.with(Cell::get)
    }
    #[cfg(not(test))]
    {
        0
    }
}

/// Restores the previous "not ready" mask of its thread when dropped.
///
/// The guard is neither `Send` nor `Sync`: it restores the mask of the
/// thread that drops it, so it must be dropped on the thread that created it.
#[cfg(test)]
#[derive(Debug)]
#[must_use = "the pieces are marked only while the guard lives"]
pub(super) struct NotReadyGuard {
    /// The mask to restore.
    previous: u64,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

/// Makes the pieces whose bit is set in `mask` (bit `i` = piece `i`) run as
/// if their thread were not ready, for the calls made on this thread while
/// the guard lives (test builds only). The mask is read on the calling
/// thread and handed to the pieces.
#[cfg(test)]
pub(super) fn mark_not_ready(mask: u64) -> NotReadyGuard {
    let previous = NOT_READY.with(|cell| cell.replace(mask));
    NotReadyGuard {
        previous,
        _not_send: PhantomData,
    }
}

#[cfg(test)]
impl Drop for NotReadyGuard {
    fn drop(&mut self) {
        NOT_READY.with(|cell| cell.set(self.previous));
    }
}

/// What one call did (test builds only).
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RunReport {
    /// The route of the call.
    pub(super) route: Route,
    /// The prefetch switches of the call's kernel calls.
    pub(super) prefetch: u64,
    /// The layout it ran.
    pub(super) layout: Layout,
    /// For each piece, whether it ran on the NEON fallback.
    pub(super) neon: Vec<bool>,
    /// For each piece, how its blocks ran (zero for a NEON piece), with the
    /// kernel entries counted on its thread over its run and what the
    /// policy did (`BlockCount::policy`).
    pub(super) blocks: Vec<BlockCount>,
    /// For each piece, the thread that ran it.
    pub(super) threads: Vec<ThreadId>,
    /// For each piece, `rayon::current_thread_index()` of the thread that
    /// ran it (`None`: not a rayon worker).
    pub(super) workers: Vec<Option<usize>>,
    /// For each piece, whether it gave up waiting at the rendezvous.
    pub(super) timed_out: Vec<bool>,
}

/// The report of the last call that ran on this thread, taken out (test
/// builds only).
#[cfg(test)]
pub(super) fn take_last_run() -> Option<RunReport> {
    LAST_RUN.with(|last| last.borrow_mut().take())
}
