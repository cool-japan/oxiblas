//! Run-time detection of SME2 hardware, in Pure Rust (no FFI, no libc).
//!
//! The detection has two stages. [`hw`] reports SME2 hardware only when
//! both have succeeded. Each stage is settled once per process, and each
//! is read in two forms: the waiting form ([`hw`]: behind `sme2_active`,
//! the test gates and the detection report) and the
//! non-waiting form ([`hw_now`]: behind [`enabled`], [`thread_ready`] and
//! [`units`], which are what the public GEMM and GEMV paths call). The
//! section "Waiting, and fork" below says exactly which call waits for what.
//!
//! # Stage 1: the candidate
//!
//! The candidate is computed once per process. Each step runs only when the
//! previous one succeeded:
//!
//! 1. **Environment.** `OXIBLAS_DISABLE_SME2` set to anything other than an
//!    empty string or `0` turns the backend off ([`env_disables`]).
//! 2. **Operating-system report**, obtained without executing any SME
//!    instruction. On macOS this is one call of `/usr/sbin/sysctl` (an
//!    absolute path, so a different `sysctl` earlier in `PATH` is never
//!    used) for the `hw.optional.arm.*` feature keys and
//!    `hw.optional.arm.sme_max_svl_b`, which must be 64. On Linux and Android
//!    it is the `AT_HWCAP2` word of `/proc/self/auxv`, which qemu-user
//!    emulates as well. Other operating systems give no report, so SME2 stays
//!    off there. FEAT_SME and FEAT_SME2 are required; `f64` additionally
//!    needs FEAT_SME_F64F64.
//! 3. **Register probe.** Only after the operating system reported SME2 does
//!    [`hw_from_report`] execute `RDSVL`, which must return 64 bytes (a
//!    512-bit streaming vector length). On a CPU without SME the instruction
//!    is undefined and raises SIGILL, which is why the probe comes last.
//!
//! A present candidate says "the operating system reports usable SME2
//! hardware"; no kernel has run yet. The tests that call a kernel directly
//! and the register reads of the test support ask the candidate.
//!
//! - **Who computes it.** The first detection call of the process, in
//!   either form, on any thread: stage 1 has no readiness condition (its
//!   register probe only reads registers), so a thread that is not ready
//!   computes it too. That thread takes a process-wide lock, computes the
//!   value and stores it ([`CandidateCell`]); the value never changes
//!   afterwards, and it is computed at most once per process.
//! - **A call that finds another thread computing it.** The waiting form
//!   waits for that thread and returns its value. The non-waiting form gets
//!   `None` for this call at once and stores nothing: that BLAS call runs on
//!   NEON (which gives the same bits), and a later call reads the value. When
//!   nobody computes it, both forms compute it themselves.
//! - **A panic while it is computed** (none of the steps is known to panic)
//!   poisons the lock and stores nothing; the panic leaves the call that
//!   computed. The next call of either form that finds the lock poisoned
//!   settles "no candidate" for the life of the process without computing
//!   again (the backend stays off), so no later call waits for it and it is
//!   never computed twice.
//! - **Once it is known** both forms read it with one atomic load and no
//!   lock.
//!
//! # Stage 2: the first-use check
//!
//! Before [`hw`] reports the candidate, the kernels must have passed their
//! checks against plain Rust on this CPU, once per process: a kernel that
//! is wrong on real hardware turns the backend off instead of returning
//! wrong numbers. The verdict has three states: open, passed and failed.
//!
//! - **Who runs it.** The first detection call on a thread whose SME state
//!   is ready while the verdict is open, through either form below. "Ready"
//!   is the pure rule [`ThreadState::ready`] on the raw
//!   [`probe_thread_state`], not [`thread_ready`] (whose test seams must not
//!   be consumed by the detection). That thread takes a process-wide lock,
//!   runs the check and settles the verdict.
//! - **Two forms, for a ready thread that finds another thread running the
//!   check.** The waiting form, [`hw`] (behind `sme2_active`, the test gates
//!   and the detection report), waits for the lock and then reads the
//!   settled verdict, so every waiting call that returns after the check
//!   began gives the same answer. The non-waiting form, [`hw_now`] (behind
//!   [`enabled`], [`thread_ready`] and [`units`], which are what the public
//!   GEMM and GEMV paths call), never waits: it answers `None` for this call
//!   and leaves the verdict alone, so that call is computed on NEON (which
//!   gives the same bits) instead of blocking a BLAS call on another
//!   thread's lock. Without such a concurrent check both forms behave the
//!   same: a ready thread whose call finds the verdict open and the lock
//!   free runs the check itself.
//! - **Fork**: see "Waiting, and fork" below.
//! - **A thread that is not ready** while the verdict is open never runs the
//!   check: a thread that is not ready says nothing about the kernels. In
//!   the non-waiting form it gets `None` for that call at once and never
//!   touches the lock. In the waiting form it takes the lock and releases it
//!   at once, running nothing, so it waits for a check that another thread
//!   is running, and then answers what is stored: the candidate when that
//!   check passed, `None` when it failed (a check that panicked leaves the
//!   lock poisoned, and the waiting call settles that failure, as every call
//!   that takes a poisoned lock does), and `None` with the verdict left open
//!   and the lock free when nobody has run the check. A later call from a
//!   ready thread settles it.
//! - **The check** ([`run_checks`]) runs, with the candidate's `f64` flag
//!   and in this order, stopping at the first failure:
//!   `selftest::run_self_test_detailed` (the macro-kernels: the largest
//!   check, and the one whose mismatch says most), `pack_za::self_check`
//!   (the ZA transposers that feed them), `few_rows::self_check` and
//!   `gemv::sme2::self_check`. None of them calls [`hw`], [`enabled`],
//!   [`thread_ready`] or `sme2_active`, creates a thread or panics; should a
//!   call from inside the check ever reach [`hw`] on the checking thread, it
//!   returns `None` instead of waiting for the lock it holds.
//! - **Failure** is final for the life of the process: the backend stays
//!   off, and one line goes to standard error, once ([`failure_line`]: which
//!   check failed and, for the macro-kernels, the mismatch that the
//!   self-test reports). A poisoned lock (a check that panicked) counts as a
//!   failure too.
//! - **Floating-point state.** The check runs with FPCR = 0 (round to
//!   nearest, no flush-to-zero, default NaN off, no trap enabled), and
//!   afterwards FPCR and FPSR are bit for bit what the caller had
//!   (`fpstate::with_default_fp_state`): the verdict does not depend on the
//!   caller's modes, a trap the caller enabled cannot fire on the check's
//!   NaN and overflow cases, and the caller's cumulative flags do not move.
//! - **Cost**, paid once by the thread that runs the check: 2 004 kernel
//!   calls of the macro-kernel self-test (714 `f32`, 1 290 `f64`, pinned by
//!   `test_sme2_selftest_cost_matches_the_docs`; about 91 MB of allocation
//!   traffic, at most about 207 kB live), plus per type 12 calls of the ZA
//!   transposers (one per check block), 16 of the few-rows GEMM and 12 of
//!   the GEMV (small buffers of a few kilobytes each): 2 084 kernel calls
//!   with `f64` (754 with `f32` alone), pinned by
//!   `test_sme2_detect_first_use_check_is_quiet_and_costs_the_docs`. Under
//!   QEMU (a release build, while other tests ran) that test measured
//!   0.65 s for the whole check (0.42 s with `f32` alone); it has not been
//!   measured on hardware yet.
//! - **After the verdict** both forms cost two atomic loads (the
//!   candidate's and the verdict's) and take no lock.
//!
//! The readers of test builds, `candidate` (stage 1 alone, in the waiting
//! form) and `verdict`, never start the check.
//!
//! # Waiting, and fork
//!
//! - **The waiting form** ([`hw`]; `sme2_active`, the test gates, the
//!   detection report) waits for a thread that computes the candidate
//!   (stage 1) and for a thread that runs the first-use check (stage 2),
//!   whether the calling thread is ready or not; it runs the check itself
//!   only on a ready thread. The one exception is a call from inside the
//!   check, on the checking thread: it returns `None` at once instead of
//!   waiting for the lock it holds. The test-build reader `candidate` waits
//!   only for stage 1.
//! - **The non-waiting form** ([`hw_now`]; [`enabled`], [`thread_ready`],
//!   [`units`], so every SME2 decision of the public GEMM and GEMV paths)
//!   waits for neither: a call that finds another thread inside either
//!   stage gets `None` at once and runs on NEON. The only work such a call
//!   does itself is a stage that nobody else is in: computing the candidate
//!   (on macOS it waits for its own `sysctl` subprocess) and, on a ready
//!   thread, running the first-use check.
//! - **What a waiting call guarantees.** When [`hw`] returns (from any call
//!   but one made inside the check), no stage that was in progress when it
//!   was called is still in progress, and the candidate is settled. When the
//!   candidate is present, the verdict is settled too, unless the check has
//!   not run yet: then the call came from a thread that is not ready, the
//!   verdict is open and the call left its lock free (another thread may
//!   take it right afterwards). So a call from a ready thread always returns
//!   with both stages settled. (Without a candidate the check never runs and
//!   its lock is never taken.)
//! - **Fork.** Both stages store their result before they release their
//!   lock, and a settled stage is read without the lock. A process forked
//!   while another thread holds a stage's lock, before that stage stored its
//!   result, inherits the lock in its locked state but not the thread that
//!   would release it, so in the child that stage stays unsettled for good.
//!   There the non-waiting form answers `None` for good: the child's BLAS
//!   calls never use SME2 and never wait. `sme2_active` (the waiting form)
//!   would wait forever in such a child. (A child forked after the stage
//!   stored its result sees the settled value without the lock: its BLAS
//!   calls may use SME2, and nothing in it waits.) A program that forks
//!   should therefore call `sme2_active` once before it forks:
//!   - from a ready thread ([`ThreadState::ready`]: outside SME code and
//!     with a 512-bit streaming vector length; being outside SME code alone
//!     is not enough), the call returns with both stages settled (without
//!     a candidate there is no check to settle), so the child inherits them
//!     and nothing in it ever waits for an inherited lock of the detection,
//!     whatever other threads do;
//!   - from a thread that is not ready, the verdict can be still open when
//!     it returns (the check has not run). Its lock is then free at the fork
//!     only when no other thread is in a BLAS call or a detection query, or
//!     starts one, between that call and the fork (the usual rule for
//!     forking a multi-threaded process); the child's first BLAS call from a
//!     ready thread then runs the check itself.
//!
//!   A build that caps SIMD (`SME2_ALLOWED`) never runs either stage through
//!   `sme2_active` or the BLAS paths.
//!
//! # Readiness and the unit count
//!
//! [`thread_ready`] repeats the register probe on every call, on the thread
//! that is about to run SME code, and applies the rule of
//! [`ThreadState::ready`] to what it reads. Test builds can make it answer
//! `false` on chosen threads (`set_pretend_not_ready`) or after a number of
//! calls (`set_ready_countdown`), so that the tests can prove the check runs
//! on the thread that executes the code.
//!
//! The detection also counts the SME units ([`Sme2Hw::units`], read through
//! [`units`]): the GEMM driver splits a large product into at most that many
//! parts. On macOS it is the number of performance clusters,
//! `max(1, hw.perflevel0.physicalcpu / hw.perflevel0.cpusperl2)` (an Apple
//! M4 has one SME unit per performance cluster); Linux and Android report no
//! topology the backend could trust, so there it is 1. Test builds can
//! override the count per thread (`override_units`).

use std::any::TypeId;
use std::cell::Cell;
use std::ffi::OsStr;
#[cfg(test)]
use std::marker::PhantomData;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock, TryLockError};

use super::SME2_ALLOWED;
use super::selftest::Mismatch;
use super::{few_rows, fpstate, pack_za, selftest};
use crate::level2::gemv::sme2 as gemv_sme2;

#[cfg(test)]
#[path = "tests_detect.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_detect_first_use.rs"]
mod tests_first_use;

#[cfg(test)]
#[path = "tests_detect_forms.rs"]
mod tests_forms;

#[cfg(test)]
#[path = "tests_detect_wait.rs"]
mod tests_wait;

/// Environment variable that turns the SME2 backend off for the whole process.
const DISABLE_ENV: &str = "OXIBLAS_DISABLE_SME2";

/// Streaming vector length, in bytes, that the kernels are written for (512
/// bits, the SVL of the Apple M4).
const REQUIRED_SVL_BYTES: u64 = 64;

/// Kernel calls of the first-use check with `f32` only: 714 of the
/// macro-kernel self-test, 12 of the ZA transposers, 16 of the few-rows
/// GEMM and 12 of the GEMV.
#[cfg(test)]
pub(super) const FIRST_USE_KERNEL_CALLS_F32: u64 = 714 + 12 + 16 + 12;

/// Kernel calls of the first-use check with `f32` and `f64`: for `f64`
/// another 1 290 of the macro-kernel self-test, 12 of the ZA transposers,
/// 16 of the few-rows GEMM and 12 of the GEMV.
#[cfg(test)]
pub(super) const FIRST_USE_KERNEL_CALLS: u64 = FIRST_USE_KERNEL_CALLS_F32 + 1290 + 12 + 16 + 12;

/// SME2 hardware available to this process, as found by [`hw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sme2Hw {
    /// FEAT_SME_F64F64 is present, so `f64` products may use SME2 as well.
    pub(crate) f64: bool,
    /// SME units available to this process (at least 1): the number of
    /// parts the GEMM driver may split a large product into.
    pub(crate) units: usize,
}

/// SME features as reported by the operating system, read without executing
/// any SME instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OsSme {
    /// FEAT_SME.
    sme: bool,
    /// FEAT_SME2.
    sme2: bool,
    /// FEAT_SME_F64F64.
    f64f64: bool,
    /// Maximum streaming vector length in bytes, when the operating system
    /// reports one (macOS `hw.optional.arm.sme_max_svl_b`). Linux reports
    /// none, so there only `RDSVL` tells.
    max_svl_bytes: Option<u64>,
    /// SME units (at least 1): performance clusters on macOS, 1 elsewhere.
    units: usize,
}

impl OsSme {
    /// Whether the report allows the backend: SME and SME2, and a 512-bit
    /// maximum streaming vector length when the operating system reports one.
    fn offers_sme2(&self) -> bool {
        self.sme
            && self.sme2
            && self
                .max_svl_bytes
                .is_none_or(|svl| svl == REQUIRED_SVL_BYTES)
    }
}

/// The calling thread's SME state, as read by [`probe_thread_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ThreadState {
    /// `SVCR`: bit 0 is PSTATE.SM (streaming mode), bit 1 is PSTATE.ZA (ZA
    /// storage on).
    pub(super) svcr: u64,
    /// `TPIDR2_EL0`: non-zero while a lazy save of a caller's ZA contents is
    /// pending (SME procedure-call standard).
    pub(super) tpidr2: u64,
    /// `RDSVL #1`: the streaming vector length in bytes.
    pub(super) svl_bytes: u64,
}

impl ThreadState {
    /// The readiness rule of [`thread_ready`] and of the first-use check:
    /// SVCR is zero (neither streaming mode nor ZA storage is on),
    /// TPIDR2_EL0 is zero (no lazy save of a caller's ZA contents is
    /// pending) and the streaming vector length is the one the kernels are
    /// written for.
    pub(super) fn ready(&self) -> bool {
        self.svcr == 0 && self.tpidr2 == 0 && self.svl_bytes == REQUIRED_SVL_BYTES
    }
}

/// Stage 1, computed once per process by the first [`hw`] or [`hw_now`] call
/// (or, in test builds, `candidate` call) that finds it unknown and nobody
/// computing it.
static CANDIDATE: CandidateCell = CandidateCell::new();

/// Computations of the process's candidate so far (test builds only).
#[cfg(test)]
static CANDIDATE_RUNS: AtomicUsize = AtomicUsize::new(0);

/// Stage 2: the process-wide verdict of the first-use check.
static FIRST_USE: FirstUse = FirstUse::new();

/// First-use checks of the real kernels started so far, on any thread
/// (test builds only).
#[cfg(test)]
static CHECKS_STARTED: AtomicUsize = AtomicUsize::new(0);

/// How many first-use checks of the real kernels this process has started
/// so far: 0 before the check, 1 from its start on (more only if a test
/// ran the real check on a stand-in verdict, which none does; test builds
/// only). A test that finds this count 0 and then the verdict open knows
/// that no check was in progress.
#[cfg(test)]
pub(crate) fn checks_started() -> usize {
    CHECKS_STARTED.load(Ordering::SeqCst)
}

/// The candidate (stage 1 of the module docs): what the environment, the
/// operating system and the register probe say, computed once per process.
/// No kernel has run for it, and reading it never starts the first-use
/// check. `None` means: disabled through the environment, no SME2 in the
/// operating system's report, a streaming vector length other than 512
/// bits, or a computation that panicked.
///
/// Stage 1 alone, in the waiting form: when the candidate is unknown, the
/// call computes it, or waits for the thread that computes it and returns
/// that thread's value. Once it is known, a call costs one atomic load. Test
/// builds only: the tests that call a kernel directly, the hardware gate and
/// the detection report read it; the library reaches stage 1 through
/// [`hw`] and [`hw_now`].
#[cfg(test)]
pub(crate) fn candidate() -> Option<&'static Sme2Hw> {
    CANDIDATE.get_in(Form::Wait, compute_candidate)
}

/// [`detect_candidate`] for the process's [`CANDIDATE`], counted in test
/// builds (`candidate_runs`).
fn compute_candidate() -> Option<Sme2Hw> {
    #[cfg(test)]
    CANDIDATE_RUNS.fetch_add(1, Ordering::SeqCst);
    detect_candidate()
}

/// How many times this process computed its candidate so far: 0 before the
/// first detection call, 1 afterwards (test builds only).
#[cfg(test)]
pub(crate) fn candidate_runs() -> usize {
    CANDIDATE_RUNS.load(Ordering::SeqCst)
}

/// Whether the process's candidate is known, without computing it or
/// waiting for it (test builds only).
#[cfg(test)]
pub(crate) fn candidate_known() -> bool {
    CANDIDATE.known().is_some()
}

/// The verdict of the first-use check, without starting it (test builds:
/// the tests and the detection report read it).
#[cfg(test)]
pub(crate) fn verdict() -> Verdict<'static> {
    FIRST_USE.verdict()
}

/// The SME2 hardware of this process: the candidate, once its kernels
/// have passed the first-use check (see the module docs). This is the
/// waiting form.
///
/// `None` means the SME2 backend must not run: there is no candidate, the
/// first-use check failed, or the verdict is still open and the calling
/// thread is not ready (that call only; a later call from a ready thread
/// settles the verdict). While the candidate is unknown, a call computes it
/// or waits for the thread that computes it. While the verdict is open, a
/// call waits for a thread that runs the check and then gives that thread's
/// answer, from a ready thread and from one that is not ready alike; when
/// nobody runs it, a call from a ready thread runs it, and a call from a
/// thread that is not ready returns `None` and leaves the verdict open and
/// its lock free. Only a call from inside the check, on the checking
/// thread, returns `None` at once instead of waiting for its own lock. So
/// when the call returns, no stage that was in progress when it was made is
/// still in progress (module docs, "Waiting, and fork"). Once both are
/// settled, a call costs two atomic loads, on any thread.
///
/// Only callers that need the settled answer use this form: the public
/// `sme2_active`, the test gates and the detection report. The public BLAS
/// paths ask [`enabled`], [`thread_ready`] and [`units`], which use
/// [`hw_now`] and never wait.
pub(crate) fn hw() -> Option<&'static Sme2Hw> {
    hw_in(Form::Wait)
}

/// The non-waiting form of [`hw`]: the same answer, except that a call that
/// finds another thread computing the candidate, or (on a ready thread)
/// another thread running the first-use check, gets `None` for this call at
/// once, without waiting for that thread and without storing anything.
///
/// It is what the public GEMM and GEMV paths reach (through [`enabled`],
/// [`thread_ready`] and [`units`]): a BLAS call never blocks on a lock that
/// another thread holds, and in a process forked while another thread held
/// the lock of a stage that had not stored its result a BLAS call never
/// hangs (it gets `None` there for good, so its BLAS calls run on NEON). On
/// a thread that is not ready it never touches the verdict's lock. A call
/// that gets `None` this way is computed by the NEON path, which gives the
/// same bits. When the candidate
/// is unknown and nobody computes it, the call computes it (on any thread);
/// when the verdict is open and no other thread is running the check, a
/// ready thread runs it itself, exactly as with [`hw`]. Once both are
/// settled, a call costs two atomic loads.
pub(crate) fn hw_now() -> Option<&'static Sme2Hw> {
    hw_in(Form::NoWait)
}

/// [`hw`] (`Form::Wait`) or [`hw_now`] (`Form::NoWait`).
#[inline]
fn hw_in(form: Form) -> Option<&'static Sme2Hw> {
    #[cfg(test)]
    note_query();
    let candidate = candidate_cell().get_in(form, compute_candidate)?;
    // A test's stand-in stage-1 cell only decides whether this call waits:
    // its answer never reaches the register probe below.
    #[cfg(test)]
    if CANDIDATE_CELL_HERE.with(Cell::get).is_some() {
        return None;
    }
    let passed = first_use_cell().passed_in(
        form,
        || {
            // SAFETY: the candidate is present and is the process's (a
            // test's stand-in stage-1 cell returned above), so the operating
            // system reported FEAT_SME for this process and the detection
            // already executed this probe.
            unsafe { probe_thread_state() }.ready()
        },
        || {
            #[cfg(test)]
            CHECKS_STARTED.fetch_add(1, Ordering::SeqCst);
            // SAFETY: `FirstUse::passed_in` runs this closure only on the
            // thread whose readiness closure (above) just returned `true`,
            // and the candidate is present: SME2 with a 512-bit streaming
            // vector length, and FEAT_SME_F64F64 when `candidate.f64`.
            unsafe { run_checks(&REAL_CHECKS, candidate.f64) }
        },
        write_failure_line,
    );
    passed.then_some(candidate)
}

/// The stage-1 cell that [`hw_in`] reads: [`CANDIDATE`], or in test builds
/// the stand-in of a live `StagesHere` guard on the calling thread.
#[inline]
fn candidate_cell() -> &'static CandidateCell {
    #[cfg(test)]
    if let Some(cell) = CANDIDATE_CELL_HERE.with(Cell::get) {
        return cell;
    }
    &CANDIDATE
}

/// The verdict that [`hw_in`] reads: [`FIRST_USE`], or in test builds the
/// stand-in of a live `StagesHere` guard on the calling thread.
#[inline]
fn first_use_cell() -> &'static FirstUse {
    #[cfg(test)]
    if let Some(first_use) = FIRST_USE_HERE.with(Cell::get) {
        return first_use;
    }
    &FIRST_USE
}

#[cfg(test)]
thread_local! {
    /// The stage-1 cell that [`hw_in`] reads on this thread instead of
    /// [`CANDIDATE`] (test builds only; `None` = the process's).
    static CANDIDATE_CELL_HERE: Cell<Option<&'static CandidateCell>> = const { Cell::new(None) };
    /// The verdict that [`hw_in`] reads on this thread instead of
    /// [`FIRST_USE`] (test builds only; `None` = the process's).
    static FIRST_USE_HERE: Cell<Option<&'static FirstUse>> = const { Cell::new(None) };
}

/// Makes [`hw_in`] read a stand-in for one of the process's stages on the
/// calling thread while the guard lives; guards nest (test builds only). It
/// lets a test drive the real [`hw`] and [`hw_now`] (and the questions of
/// the BLAS paths built on them) against a stage whose lock it holds or
/// whose computation it parks, without touching that stage of the process.
///
/// - A stand-in stage-1 cell decides only whether the call waits and what
///   is stored there: a call that reads one answers `None` after it, so it
///   never reaches stage 2 (whose register probe needs the process's
///   candidate).
/// - A stand-in verdict is read after the process's candidate, with the
///   real closures of stage 2 (the register probe and the real checks).
#[cfg(test)]
#[must_use = "the stand-in is used only while the guard lives"]
pub(super) struct StagesHere {
    /// The stage-1 cell to restore.
    previous_candidate: Option<&'static CandidateCell>,
    /// The verdict to restore.
    previous_first_use: Option<&'static FirstUse>,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

#[cfg(test)]
impl StagesHere {
    /// Puts `cell` on the calling thread as its stage-1 cell.
    pub(super) fn candidate(cell: &'static CandidateCell) -> Self {
        Self {
            previous_candidate: CANDIDATE_CELL_HERE.with(|here| here.replace(Some(cell))),
            previous_first_use: FIRST_USE_HERE.with(Cell::get),
            _not_send: PhantomData,
        }
    }

    /// Puts `first_use` on the calling thread as its verdict.
    pub(super) fn first_use(first_use: &'static FirstUse) -> Self {
        Self {
            previous_candidate: CANDIDATE_CELL_HERE.with(Cell::get),
            previous_first_use: FIRST_USE_HERE.with(|here| here.replace(Some(first_use))),
            _not_send: PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for StagesHere {
    fn drop(&mut self) {
        CANDIDATE_CELL_HERE.with(|here| here.set(self.previous_candidate));
        FIRST_USE_HERE.with(|here| here.set(self.previous_first_use));
    }
}

/// How a detection call behaves when it finds another thread computing the
/// candidate or running the first-use check (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Form {
    /// Wait for that thread, then read what it settled ([`hw`], and
    /// `candidate` in test builds).
    Wait,
    /// Answer "no candidate" or "not passed" for this call at once and store
    /// nothing ([`hw_now`]).
    NoWait,
}

/// Stage 1 of the detection: the candidate, computed at most once, and a
/// lock for the thread that computes it. [`CANDIDATE`] is the process's;
/// tests make their own with stand-in computations.
pub(super) struct CandidateCell {
    /// The candidate once it is known. Set only by the holder of `lock`, so
    /// setting it never waits; "no candidate" (`Some(None)`) also after a
    /// computation that panicked.
    value: OnceLock<Option<Sme2Hw>>,
    /// Held by the thread that computes the candidate; poisoned when that
    /// computation panicked.
    lock: Mutex<()>,
}

impl CandidateCell {
    /// A cell whose candidate is unknown.
    pub(super) const fn new() -> Self {
        Self {
            value: OnceLock::new(),
            lock: Mutex::new(()),
        }
    }

    /// The candidate when it is known (`Some(candidate)`), `None` while it
    /// is not; never computes it and never waits (test builds).
    #[cfg(test)]
    pub(super) fn known(&self) -> Option<Option<&Sme2Hw>> {
        self.value.get().map(Option::as_ref)
    }

    /// The candidate, in the form `form`; computes it with `compute` first
    /// when it is unknown and no other thread computes it.
    ///
    /// A known candidate is read with one atomic load. While it is unknown
    /// the thread asks for the lock:
    ///
    /// - [`Form::Wait`] takes it (waiting for a thread that computes the
    ///   candidate);
    /// - [`Form::NoWait`] takes it only when it is free. When another thread
    ///   holds it (that thread computes the candidate, or has just stored it
    ///   and is about to release the lock), the call reads the value once
    ///   more and returns it when it is known by then, `None` otherwise: it
    ///   computes nothing and stores nothing.
    ///
    /// With the lock it reads the value again, and if it is still unknown
    /// runs `compute` and stores its result. A poisoned lock (a computation
    /// that panicked) stores "no candidate" without running `compute`, in
    /// both forms. `compute` runs only on the calling thread, only while it
    /// holds the lock, and at most once per cell.
    #[inline]
    pub(super) fn get_in(
        &self,
        form: Form,
        compute: impl FnOnce() -> Option<Sme2Hw>,
    ) -> Option<&Sme2Hw> {
        match self.value.get() {
            Some(known) => known.as_ref(),
            None => self.settle(form, compute),
        }
    }

    /// The unknown-candidate path of [`CandidateCell::get_in`].
    #[cold]
    #[inline(never)]
    fn settle(&self, form: Form, compute: impl FnOnce() -> Option<Sme2Hw>) -> Option<&Sme2Hw> {
        let (guard, poisoned) = match form {
            Form::Wait => match self.lock.lock() {
                Ok(guard) => (guard, false),
                Err(poison) => (poison.into_inner(), true),
            },
            Form::NoWait => match self.lock.try_lock() {
                Ok(guard) => (guard, false),
                Err(TryLockError::Poisoned(poison)) => (poison.into_inner(), true),
                // Another thread holds the lock: it computes the candidate
                // (or has just stored it). This call does not wait and
                // leaves the value to that thread.
                Err(TryLockError::WouldBlock) => {
                    return self.value.get().and_then(Option::as_ref);
                }
            },
        };
        let known = match self.value.get() {
            Some(known) => known,
            None => {
                let computed = if poisoned { None } else { compute() };
                // Only the holder of the lock sets the value, so this never
                // waits.
                self.value.get_or_init(|| computed)
            }
        };
        drop(guard);
        known.as_ref()
    }
}

#[cfg(test)]
thread_local! {
    /// Calls of [`hw`] and [`hw_now`] on this thread (test builds only).
    static QUERIES: Cell<u64> = const { Cell::new(0) };
}

/// Counts one detection call on this thread (test builds only).
#[cfg(test)]
fn note_query() {
    let _ = QUERIES.try_with(|count| count.set(count.get() + 1));
}

/// Calls of [`hw`] and [`hw_now`] (directly, or through [`enabled`],
/// [`thread_ready`], [`units`] or `sme2_active`) made on the calling thread
/// so far. Test builds only: the tests prove with it that a call below a
/// crossover, or on a route that SME2 does not serve, never reaches the
/// detection.
#[cfg(test)]
pub(crate) fn queries_here() -> u64 {
    QUERIES.try_with(Cell::get).unwrap_or(0)
}

/// Whether the SME2 backend may run for element type `T` in this process.
///
/// `f32` needs the hardware of [`hw`], and `f64` additionally needs
/// FEAT_SME_F64F64; every other type is served by the other backends. Always
/// `false`, without running the detection, when the build caps SIMD
/// (`SME2_ALLOWED` in the parent module). In test builds it is also `false`
/// on a thread that holds a `test_support::force_off` guard.
///
/// It asks [`hw_now`], so while another thread computes the candidate or
/// runs the first-use check it is `false` for this call, at once, without
/// waiting (module docs, "Waiting, and fork"). The SME2 decisions of
/// the public GEMM and GEMV paths call it first, on the calling thread: a
/// decision that gets past it has seen the verdict settled to "passed"
/// (this thread ran the check, or found it passed), so no part of the call
/// can be the thread that runs the check.
pub(crate) fn enabled<T: 'static>() -> bool {
    if !SME2_ALLOWED {
        return false;
    }
    #[cfg(test)]
    if super::test_support::forced_off() {
        return false;
    }
    let Some(hw) = hw_now() else {
        return false;
    };
    let ty = TypeId::of::<T>();
    ty == TypeId::of::<f32>() || (ty == TypeId::of::<f64>() && hw.f64)
}

/// Whether the calling thread may enter a streaming region now.
///
/// `true` only when [`hw_now`] reports SME2 hardware (so the first-use check
/// has passed; while another thread computes the candidate or runs the
/// check, `false` at once, without waiting) and the thread is in the state
/// that the SME procedure-call
/// standard calls "off": SVCR is zero (neither streaming mode nor ZA storage
/// is on), TPIDR2_EL0 is zero (no lazy save of a caller's ZA contents is
/// pending), and the streaming vector length is still 512 bits (Linux lets a
/// thread change it). In any other state the backend declines rather than
/// commit or clobber someone else's ZA contents.
///
/// The check reads three registers. It must run on the thread that executes
/// the SME code, before any output is written. Without SME2 hardware it
/// returns `false` without executing any SME instruction.
///
/// In test builds it returns `false` first, before reading any register, on
/// a thread that pretends not to be ready (`set_pretend_not_ready`) and on a
/// thread whose readiness countdown has run out (`set_ready_countdown`);
/// every call counts down a running countdown, also one that answers
/// `false` for the pretend flag.
pub(crate) fn thread_ready() -> bool {
    #[cfg(test)]
    if countdown_ran_out() || pretends_not_ready() {
        return false;
    }
    if hw_now().is_none() {
        return false;
    }
    // SAFETY: `hw_now()` is `Some` only for a present candidate, so the operating
    // system reported FEAT_SME for this process and the detection already
    // executed this probe: the SME register reads and RDSVL are implemented
    // and enabled.
    unsafe { probe_thread_state() }.ready()
}

/// Runs the steps of stage 1 (module docs); called at most once per process,
/// through [`CandidateCell::get_in`] of [`CANDIDATE`].
fn detect_candidate() -> Option<Sme2Hw> {
    if env_disables(std::env::var_os(DISABLE_ENV).as_deref()) {
        return None;
    }
    let os = os_report()?;
    hw_from_report(os, || {
        // SAFETY: `hw_from_report` calls this closure only after the operating
        // system reported FEAT_SME and FEAT_SME2 for this process, so the
        // probe's SME instructions are implemented and enabled.
        unsafe { probe_thread_state() }.svl_bytes
    })
}

/// Whether a value of `OXIBLAS_DISABLE_SME2` turns the backend off.
///
/// Any value other than an empty string or `0` does, including values that
/// are not valid Unicode. Taking the value as an argument keeps this testable
/// without changing the process environment.
fn env_disables(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

/// Combines the operating-system report with the register probe.
///
/// `read_svl_bytes` executes `RDSVL`. It is called only when the report
/// offers SME2, because on a CPU without SME the instruction raises SIGILL.
fn hw_from_report(os: OsSme, read_svl_bytes: impl FnOnce() -> u64) -> Option<Sme2Hw> {
    if !os.offers_sme2() || read_svl_bytes() != REQUIRED_SVL_BYTES {
        return None;
    }
    Some(Sme2Hw {
        f64: os.f64f64,
        units: os.units.max(1),
    })
}

// ---------------------------------------------------------------------------
// Stage 2: the first-use check
// ---------------------------------------------------------------------------

/// The verdict of a first-use check (see the module docs), as the tests
/// and the detection report read it.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict<'a> {
    /// No ready thread has run the check yet.
    Open,
    /// Every check passed: [`hw`] reports the candidate.
    Passed,
    /// A check failed (or its lock was poisoned), with the line of
    /// [`failure_line`]: the backend stays off for the life of the process.
    Failed(&'a str),
}

/// The open verdict, as stored in [`FirstUse`].
const OPEN: u8 = 0;
/// The verdict "passed", as stored in [`FirstUse`].
const PASSED: u8 = 1;
/// The verdict "failed", as stored in [`FirstUse`].
const FAILED: u8 = 2;

/// Which first-use check failed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CheckFailure {
    /// The macro-kernel self-test, with its first mismatch.
    MacroKernels(Mismatch),
    /// The ZA transposers (`pack_za::self_check`).
    ZaPackers,
    /// The few-rows GEMM (`few_rows::self_check`).
    FewRows,
    /// The SME2 GEMV (`gemv::sme2::self_check`).
    Gemv,
    /// The lock of the check was poisoned: a check panicked.
    Poisoned,
}

/// The one line that a failed first-use check writes to standard error (no
/// line break inside it): what failed, and that the backend is off.
pub(super) fn failure_line(failure: &CheckFailure) -> String {
    let what = match failure {
        CheckFailure::MacroKernels(mismatch) => {
            format!("the self-test of the SME2 macro-kernels failed ({mismatch})")
        }
        CheckFailure::ZaPackers => "the check of the SME2 ZA transposers failed".to_owned(),
        CheckFailure::FewRows => "the check of the SME2 few-rows GEMM failed".to_owned(),
        CheckFailure::Gemv => "the check of the SME2 GEMV failed".to_owned(),
        CheckFailure::Poisoned => {
            "a first-use check of SME2 panicked (its lock is poisoned)".to_owned()
        }
    };
    let line = format!(
        "oxiblas: the SME2 backend is off for this process: {what}; please report this line"
    );
    line.replace(['\n', '\r'], " ")
}

/// Writes the failure line, with a line break, to the standard error handle
/// in one `write_all`. A failed write is ignored (`eprintln!` would panic).
fn write_failure_line(line: &str) {
    use std::io::Write;
    let text = format!("{line}\n");
    let _ = std::io::stderr().lock().write_all(text.as_bytes());
}

thread_local! {
    /// Whether this thread is running a first-use check now.
    static IN_CHECK: Cell<bool> = const { Cell::new(false) };
}

/// Marks the calling thread as running a first-use check until it drops
/// (also while a panic unwinds).
struct InCheck {
    /// The previous mark, restored on drop.
    previous: bool,
}

impl InCheck {
    /// Marks the calling thread.
    fn enter() -> Self {
        Self {
            previous: IN_CHECK.with(|flag| flag.replace(true)),
        }
    }
}

impl Drop for InCheck {
    fn drop(&mut self) {
        IN_CHECK.with(|flag| flag.set(self.previous));
    }
}

/// A first-use verdict: three states, a lock for the thread that runs the
/// check, and the failure line. [`FIRST_USE`] is the process's; tests make
/// their own with stand-in checks.
pub(super) struct FirstUse {
    /// [`OPEN`], [`PASSED`] or [`FAILED`].
    state: AtomicU8,
    /// Held by the thread that runs the check, and briefly by every call
    /// that takes it only to read the verdict again (a waiting call from a
    /// thread that is not ready never runs the check under it); poisoned
    /// when a check panicked.
    lock: Mutex<()>,
    /// The failure line, set before the state becomes [`FAILED`].
    line: OnceLock<String>,
}

impl FirstUse {
    /// An open verdict.
    pub(super) const fn new() -> Self {
        Self {
            state: AtomicU8::new(OPEN),
            lock: Mutex::new(()),
            line: OnceLock::new(),
        }
    }

    /// The verdict, without running the check (test builds).
    #[cfg(test)]
    pub(super) fn verdict(&self) -> Verdict<'_> {
        match self.state.load(Ordering::Acquire) {
            PASSED => Verdict::Passed,
            FAILED => Verdict::Failed(self.line.get().map_or("", String::as_str)),
            _ => Verdict::Open,
        }
    }

    /// [`FirstUse::passed_in`] in the waiting form ([`Form::Wait`]): a thread
    /// that finds another thread running the check, ready or not, waits for
    /// it and gives its answer (test builds: the library calls `passed_in`).
    #[cfg(test)]
    pub(super) fn passed(
        &self,
        ready: impl FnOnce() -> bool,
        check: impl FnOnce() -> Result<(), CheckFailure>,
        report: impl FnOnce(&str),
    ) -> bool {
        self.passed_in(Form::Wait, ready, check, report)
    }

    /// [`FirstUse::passed_in`] in the non-waiting form ([`Form::NoWait`]): a
    /// thread that finds another thread running the check answers without
    /// waiting and leaves the verdict alone (test builds).
    #[cfg(test)]
    pub(super) fn passed_now(
        &self,
        ready: impl FnOnce() -> bool,
        check: impl FnOnce() -> Result<(), CheckFailure>,
        report: impl FnOnce(&str),
    ) -> bool {
        self.passed_in(Form::NoWait, ready, check, report)
    }

    /// Whether the check has passed, in the form `form`; settles an open
    /// verdict first when the calling thread is ready.
    ///
    /// A settled verdict is read with one atomic load, on any thread, and no
    /// lock is taken. While it is open:
    ///
    /// - A call from inside the check (the calling thread runs a check
    ///   itself) answers `false` at once, in both forms, without touching the
    ///   lock: it holds that lock, so taking it would never return.
    /// - Otherwise `ready` is asked once, and then:
    ///   - [`Form::Wait`] takes the lock, whether the thread is ready or not
    ///     (waiting for a thread that runs the check);
    ///   - [`Form::NoWait`] on a thread that is not ready answers `false` at
    ///     once and never touches the lock;
    ///   - [`Form::NoWait`] on a ready thread takes the lock only when it is
    ///     free. When another thread holds it (that thread runs the check,
    ///     has just settled it and is about to release the lock, or is a
    ///     waiting call from a thread that is not ready), the call reads the
    ///     verdict once more and answers `true` only when it is "passed" by
    ///     then, `false` otherwise: it runs no check, stores nothing and
    ///     reports nothing.
    ///
    /// With the lock it reads the verdict again and answers a settled one.
    /// If it is still open, a ready thread runs `check`, stores the verdict
    /// and, for a failure, writes [`failure_line`] through `report` (once per
    /// verdict); a waiting call from a thread that is not ready releases the
    /// lock and answers `false`, leaving the verdict open (nobody has run the
    /// check). A poisoned lock (a check that panicked on the ready thread
    /// that ran it) settles a failure without running `check`, for every
    /// caller that takes it, ready or not.
    ///
    /// So a waiting call never returns while a check that was in progress
    /// when it was made is still in progress, and a waiting call from a
    /// thread that is not ready answers what that check settled.
    ///
    /// `check` runs only on the calling thread, only after `ready` returned
    /// `true` there, and at most once per verdict. In both forms `true`
    /// means the verdict is settled to "passed".
    #[inline]
    pub(super) fn passed_in(
        &self,
        form: Form,
        ready: impl FnOnce() -> bool,
        check: impl FnOnce() -> Result<(), CheckFailure>,
        report: impl FnOnce(&str),
    ) -> bool {
        match self.state.load(Ordering::Acquire) {
            PASSED => true,
            FAILED => false,
            _ => self.settle(form, ready, check, report),
        }
    }

    /// The open-verdict path of [`FirstUse::passed_in`].
    #[cold]
    #[inline(never)]
    fn settle(
        &self,
        form: Form,
        ready: impl FnOnce() -> bool,
        check: impl FnOnce() -> Result<(), CheckFailure>,
        report: impl FnOnce(&str),
    ) -> bool {
        // A call from inside the check, on the checking thread: it holds the
        // lock, so taking it here would never return.
        if IN_CHECK.with(Cell::get) {
            return false;
        }
        let ready = ready();
        let (guard, poisoned) = match form {
            Form::Wait => match self.lock.lock() {
                Ok(guard) => (guard, false),
                Err(poison) => (poison.into_inner(), true),
            },
            // A thread that is not ready never runs the check, so the
            // non-waiting form has nothing to do with the lock.
            Form::NoWait if !ready => return false,
            Form::NoWait => match self.lock.try_lock() {
                Ok(guard) => (guard, false),
                Err(TryLockError::Poisoned(poison)) => (poison.into_inner(), true),
                // Another thread holds the lock: it runs the check (or has
                // just settled it). This call does not wait and leaves the
                // verdict to that thread.
                Err(TryLockError::WouldBlock) => {
                    return self.state.load(Ordering::Acquire) == PASSED;
                }
            },
        };
        match self.state.load(Ordering::Acquire) {
            PASSED => return true,
            FAILED => return false,
            _ => {}
        }
        // A waiting call from a thread that is not ready has waited for any
        // check in progress and found the verdict still open: nobody has run
        // the check. It leaves the verdict open and the lock free.
        if !ready && !poisoned {
            drop(guard);
            return false;
        }
        let outcome = if poisoned {
            Err(CheckFailure::Poisoned)
        } else {
            let _in_check = InCheck::enter();
            check()
        };
        let passed = match outcome {
            Ok(()) => {
                self.state.store(PASSED, Ordering::Release);
                true
            }
            Err(failure) => {
                let line = self.line.get_or_init(|| failure_line(&failure));
                self.state.store(FAILED, Ordering::Release);
                report(line);
                false
            }
        };
        drop(guard);
        passed
    }
}

/// The four first-use checks (see the module docs); [`REAL_CHECKS`] binds
/// the real kernels, tests pass stand-ins.
#[derive(Clone, Copy)]
pub(super) struct Checks {
    /// The macro-kernel self-test.
    pub(super) macro_kernels: unsafe fn(bool) -> Result<(), Mismatch>,
    /// The ZA transposers.
    pub(super) za_packers: unsafe fn(bool) -> bool,
    /// The few-rows GEMM.
    pub(super) few_rows: unsafe fn(bool) -> bool,
    /// The SME2 GEMV.
    pub(super) gemv: unsafe fn(bool) -> bool,
}

/// The checks of the real kernels.
pub(super) const REAL_CHECKS: Checks = Checks {
    macro_kernels: selftest::run_self_test_detailed,
    za_packers: pack_za::self_check,
    few_rows: few_rows::self_check,
    gemv: gemv_sme2::self_check,
};

/// Runs the first-use check: the four checks of `checks` in the order of
/// the module docs, stopping at the first failure, with FPCR = 0 and the
/// caller's FPCR and FPSR restored afterwards
/// (`fpstate::with_default_fp_state`). In test builds the kernel calls of
/// the check are not counted by `test_support`'s kernel-entry counters
/// (`test_support::quiet_counters`).
///
/// # Safety
///
/// As for the four checks with `f64_present`: for the real ones, the CPU
/// has SME2 with a 512-bit streaming vector length (and FEAT_SME_F64F64
/// when `f64_present`), and the calling thread was just found in the state
/// of [`ThreadState::ready`].
pub(super) unsafe fn run_checks(checks: &Checks, f64_present: bool) -> Result<(), CheckFailure> {
    #[cfg(test)]
    let _quiet = super::test_support::quiet_counters();
    fpstate::with_default_fp_state(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_checks_in_order(checks, f64_present) }
    })
}

/// The four checks of `checks` in order, stopping at the first failure. Not
/// inlined, so that no floating-point operation moves across the FPCR
/// writes around it.
///
/// # Safety
///
/// As for [`run_checks`].
#[inline(never)]
unsafe fn run_checks_in_order(checks: &Checks, f64_present: bool) -> Result<(), CheckFailure> {
    // SAFETY: forwarded from this function's contract.
    unsafe { (checks.macro_kernels)(f64_present) }.map_err(CheckFailure::MacroKernels)?;
    // SAFETY: forwarded from this function's contract.
    if !unsafe { (checks.za_packers)(f64_present) } {
        return Err(CheckFailure::ZaPackers);
    }
    // SAFETY: forwarded from this function's contract.
    if !unsafe { (checks.few_rows)(f64_present) } {
        return Err(CheckFailure::FewRows);
    }
    // SAFETY: forwarded from this function's contract.
    if !unsafe { (checks.gemv)(f64_present) } {
        return Err(CheckFailure::Gemv);
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// The SME unit count that [`units`] reports on this thread instead of
    /// the detected one (test builds only; 0 = no override).
    static UNITS_OVERRIDE: Cell<usize> = const { Cell::new(0) };
    /// Whether [`thread_ready`] answers `false` on this thread (test builds
    /// only).
    static PRETEND_NOT_READY: Cell<bool> = const { Cell::new(false) };
    /// Calls of [`thread_ready`] on this thread that still answer as usual
    /// before every later call answers `false` (test builds only; `None` =
    /// no countdown).
    static READY_COUNTDOWN: Cell<Option<u32>> = const { Cell::new(None) };
}

/// Starts a readiness countdown on the calling thread (`Some(calls)`): the
/// next `calls` calls of [`thread_ready`] answer as usual, every later call
/// answers `false`. `None` stops the countdown. Returns the previous setting
/// (the calls a running countdown still had left). Test builds only.
///
/// The countdown never makes a thread look ready that is not: before it runs
/// out, [`thread_ready`] reads the registers as always. It lets a test put a
/// thread that passes one check and fails the next one under a single call,
/// such as the calling worker of a split product that passes the driver's
/// decision (one call) and then fails the own check of the part it runs
/// itself.
#[cfg(test)]
pub(crate) fn set_ready_countdown(calls: Option<u32>) -> Option<u32> {
    READY_COUNTDOWN.with(|left| left.replace(calls))
}

/// Counts one call of [`thread_ready`] down; `true` when the countdown of
/// the calling thread has run out (test builds only).
#[cfg(test)]
fn countdown_ran_out() -> bool {
    READY_COUNTDOWN.with(|left| match left.get() {
        None => false,
        Some(0) => true,
        Some(calls) => {
            left.set(Some(calls - 1));
            false
        }
    })
}

/// Makes [`thread_ready`] answer `false` on the calling thread from now on
/// (`on`), or stops that (`!on`); returns the previous setting. Test builds
/// only.
///
/// A plain setter rather than a guard, because tests set it on the workers
/// of a rayon pool from inside `ThreadPool::broadcast`, and it must stay set
/// after that closure returns; such a test clears it again with a second
/// `broadcast` (from a drop guard, so that a failing assertion clears it
/// too).
#[cfg(test)]
pub(crate) fn set_pretend_not_ready(on: bool) -> bool {
    PRETEND_NOT_READY.with(|flag| flag.replace(on))
}

/// Whether the calling thread pretends not to be ready (test builds only).
#[cfg(test)]
pub(crate) fn pretends_not_ready() -> bool {
    PRETEND_NOT_READY.with(Cell::get)
}

/// SME units the GEMM driver may split a product over: [`Sme2Hw::units`]
/// when [`hw_now`] found SME2 hardware, otherwise 1 (also while another
/// thread computes the candidate or runs the first-use check: the call does
/// not wait). Always at least 1.
///
/// In test builds a live `override_units` guard on the calling thread
/// replaces the detected count. The override is thread-local, so a driver
/// reads it once, on the thread that takes the SME2 decision.
pub(crate) fn units() -> usize {
    #[cfg(test)]
    {
        let forced = UNITS_OVERRIDE.with(Cell::get);
        if forced != 0 {
            return forced;
        }
    }
    hw_now().map_or(1, |hw| hw.units.max(1))
}

/// Restores the previous unit override of its thread when dropped (test
/// builds only).
///
/// The guard is neither `Send` nor `Sync`: it restores the override of the
/// thread that drops it, so it must be dropped on the thread that created it.
#[cfg(test)]
#[derive(Debug)]
#[must_use = "the unit count is overridden only while the guard lives"]
pub(crate) struct UnitsOverride {
    /// The override to restore (0 = none).
    previous: usize,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

/// Makes [`units`] report `units` (at least 1) on the calling thread while
/// the returned guard lives; guards nest. Test builds only: it lets the
/// tests run the parallel split with 2 and 3 parts on an emulator that
/// reports a single unit.
#[cfg(test)]
pub(crate) fn override_units(units: usize) -> UnitsOverride {
    let previous = UNITS_OVERRIDE.with(|cell| cell.replace(units.max(1)));
    UnitsOverride {
        previous,
        _not_send: PhantomData,
    }
}

#[cfg(test)]
impl Drop for UnitsOverride {
    fn drop(&mut self) {
        UNITS_OVERRIDE.with(|cell| cell.set(self.previous));
    }
}

/// The register probe: SVCR, TPIDR2_EL0 and `RDSVL #1`.
///
/// All three are legal inside and outside streaming mode, and none touches
/// memory, the stack or the condition flags. The template is also checked by
/// the static template lint (`tests_lint`), which allows exactly these reads.
macro_rules! probe_template {
    () => {
        concat!(
            ".arch_extension sme\n",
            "mrs {svcr}, svcr\n",
            "mrs {tpidr2}, tpidr2_el0\n",
            "rdsvl {svl}, #1",
        )
    };
}

/// The probe template as text, for the static template lint.
#[cfg(test)]
pub(super) const PROBE_TEMPLATE: &str = probe_template!();

/// Reads the calling thread's SME state.
///
/// # Safety
///
/// The operating system must have reported FEAT_SME for this process (see
/// [`hw_from_report`]). On a CPU without SME these instructions are undefined
/// and the process receives SIGILL.
#[inline(never)]
pub(super) unsafe fn probe_thread_state() -> ThreadState {
    let svcr: u64;
    let tpidr2: u64;
    let svl_bytes: u64;
    // SAFETY: the caller guarantees that SME is implemented and enabled for
    // this process. MRS SVCR, MRS TPIDR2_EL0 and RDSVL only read registers
    // (no memory, no stack, no flags) and are legal whether or not the thread
    // is in streaming mode. The block is deliberately not `pure`: SVCR and
    // TPIDR2_EL0 change between calls, so it must not be merged or hoisted.
    unsafe {
        core::arch::asm!(
            probe_template!(),
            svcr = out(reg) svcr,
            tpidr2 = out(reg) tpidr2,
            svl = out(reg) svl_bytes,
            options(nomem, nostack, preserves_flags),
        );
    }
    ThreadState {
        svcr,
        tpidr2,
        svl_bytes,
    }
}

// ---------------------------------------------------------------------------
// macOS: `sysctl hw.optional.arm.*`
// ---------------------------------------------------------------------------

/// Absolute path of the `sysctl` tool.
#[cfg(target_os = "macos")]
const SYSCTL_PATH: &str = "/usr/sbin/sysctl";

/// FEAT_SME (`1` when present).
#[cfg(any(target_os = "macos", test))]
const KEY_FEAT_SME: &str = "hw.optional.arm.FEAT_SME";
/// FEAT_SME2 (`1` when present).
#[cfg(any(target_os = "macos", test))]
const KEY_FEAT_SME2: &str = "hw.optional.arm.FEAT_SME2";
/// FEAT_SME_F64F64 (`1` when present).
#[cfg(any(target_os = "macos", test))]
const KEY_FEAT_SME_F64F64: &str = "hw.optional.arm.FEAT_SME_F64F64";
/// Maximum streaming vector length in bytes (`0` without SME).
#[cfg(any(target_os = "macos", test))]
const KEY_SME_MAX_SVL_B: &str = "hw.optional.arm.sme_max_svl_b";
/// Number of performance cores.
#[cfg(any(target_os = "macos", test))]
const KEY_PERF0_PHYSICALCPU: &str = "hw.perflevel0.physicalcpu";
/// Number of performance cores sharing one L2, i.e. one cluster.
#[cfg(any(target_os = "macos", test))]
const KEY_PERF0_CPUSPERL2: &str = "hw.perflevel0.cpusperl2";

/// The keys read in the single `sysctl` call.
///
/// The two `hw.perflevel0.*` keys describe the performance cores:
/// `physicalcpu / cpusperl2` is the number of performance clusters, and an
/// Apple M4 has one SME unit per performance cluster.
#[cfg(any(target_os = "macos", test))]
const SYSCTL_KEYS: [&str; 6] = [
    KEY_FEAT_SME,
    KEY_FEAT_SME2,
    KEY_FEAT_SME_F64F64,
    KEY_SME_MAX_SVL_B,
    KEY_PERF0_PHYSICALCPU,
    KEY_PERF0_CPUSPERL2,
];

/// Runs `sysctl` once for [`SYSCTL_KEYS`] and returns its standard output.
///
/// The exit status is ignored on purpose: a key unknown to the running macOS
/// version makes `sysctl` report it on standard error and exit with status
/// 1, but it still prints the known keys. `None` when the tool cannot be
/// spawned (for example inside a sandbox); the backend then stays off.
#[cfg(target_os = "macos")]
fn sysctl_text() -> Option<String> {
    let output = std::process::Command::new(SYSCTL_PATH)
        .args(SYSCTL_KEYS)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The operating-system report on macOS.
#[cfg(target_os = "macos")]
fn os_report() -> Option<OsSme> {
    sysctl_text().map(|text| macos_report(&text))
}

/// The value of `key` in `sysctl` output made of `key: value` lines, parsed
/// as an unsigned integer.
///
/// Keys must match exactly, so `hw.optional.arm.FEAT_SME2p1` is not
/// `hw.optional.arm.FEAT_SME2`. Lines without a colon and values that are not
/// integers are skipped; the first line with the key and an integer value
/// wins.
#[cfg(any(target_os = "macos", test))]
fn sysctl_value(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.trim() == key {
            value.trim().parse().ok()
        } else {
            None
        }
    })
}

/// Decodes the macOS report.
///
/// A feature counts as present only when its key reads exactly `1`; a
/// missing `sme_max_svl_b` reads as 0, which fails the 512-bit requirement.
/// The unit count is [`macos_units`] of the two `hw.perflevel0.*` keys.
#[cfg(any(target_os = "macos", test))]
fn macos_report(text: &str) -> OsSme {
    let present = |key: &str| sysctl_value(text, key) == Some(1);
    OsSme {
        sme: present(KEY_FEAT_SME),
        sme2: present(KEY_FEAT_SME2),
        f64f64: present(KEY_FEAT_SME_F64F64),
        max_svl_bytes: Some(sysctl_value(text, KEY_SME_MAX_SVL_B).unwrap_or(0)),
        units: macos_units(
            sysctl_value(text, KEY_PERF0_PHYSICALCPU),
            sysctl_value(text, KEY_PERF0_CPUSPERL2),
        ),
    }
}

/// SME units from `hw.perflevel0.physicalcpu` and `hw.perflevel0.cpusperl2`:
/// `max(1, physicalcpu / cpusperl2)`, the number of performance clusters
/// (one SME unit each on an Apple M4). A missing key, a zero `cpusperl2` or
/// a count that does not fit `usize` gives 1: an unexpected report costs
/// parallelism, never correctness (every part count gives the same bits).
#[cfg(any(target_os = "macos", test))]
fn macos_units(physicalcpu: Option<u64>, cpusperl2: Option<u64>) -> usize {
    let clusters = match (physicalcpu, cpusperl2) {
        (Some(cores), Some(per_cluster)) => cores.checked_div(per_cluster).unwrap_or(1),
        _ => 1,
    };
    usize::try_from(clusters).map_or(1, |clusters| clusters.max(1))
}

// ---------------------------------------------------------------------------
// Linux and Android: `AT_HWCAP2` from `/proc/self/auxv`
// ---------------------------------------------------------------------------
//
// The constants are the Linux arm64 uapi values. They match the toolchain's
// `std_detect` (`detect/os/linux/auxvec.rs`: `AT_HWCAP2 = 26`;
// `detect/os/linux/aarch64.rs`: `sme` is bit 23, `smef64f64` bit 25 and
// `sme2` bit 37 of `AT_HWCAP2`) and the `libc` crate (`AT_HWCAP2 = 26`;
// aarch64 Android `HWCAP2_SME = 1 << 23`, `HWCAP2_SME_F64F64 = 1 << 25`).

/// `AT_NULL`: the key that terminates the auxiliary vector.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const AT_NULL: usize = 0;
/// `AT_HWCAP2`: the key of the second hardware-capability word.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const AT_HWCAP2: usize = 26;
/// `HWCAP2_SME`: FEAT_SME.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const HWCAP2_SME: u64 = 1 << 23;
/// `HWCAP2_SME_F64F64`: FEAT_SME_F64F64.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const HWCAP2_SME_F64F64: u64 = 1 << 25;
/// `HWCAP2_SME2`: FEAT_SME2.
#[cfg(any(target_os = "linux", target_os = "android", test))]
const HWCAP2_SME2: u64 = 1 << 37;

/// Reads `AT_HWCAP2` from `/proc/self/auxv`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_auxv_hwcap2() -> Option<u64> {
    let bytes = std::fs::read("/proc/self/auxv").ok()?;
    auxv_hwcap2(&bytes)
}

/// The operating-system report on Linux and Android.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_report() -> Option<OsSme> {
    read_auxv_hwcap2().map(linux_report)
}

/// Finds `AT_HWCAP2` in a raw auxiliary vector: native-endian `(key, value)`
/// pairs of `usize` words, terminated by an `AT_NULL` key.
///
/// Returns `None` when no `AT_HWCAP2` entry precedes `AT_NULL`, and also when
/// the data ends before a complete `AT_NULL` pair: a truncated vector is not
/// trusted. With 32-bit words the capability bits above 31 (SME2 among them)
/// cannot occur, so SME2 reads as absent there.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn auxv_hwcap2(bytes: &[u8]) -> Option<u64> {
    const WORD: usize = core::mem::size_of::<usize>();
    let mut hwcap2 = None;
    for pair in bytes.chunks_exact(2 * WORD) {
        let (key, value) = pair.split_at(WORD);
        let key = usize::from_ne_bytes(key.try_into().ok()?);
        let value = usize::from_ne_bytes(value.try_into().ok()?);
        match key {
            AT_NULL => return hwcap2,
            AT_HWCAP2 => hwcap2 = Some(value as u64),
            _ => {}
        }
    }
    None
}

/// Decodes `AT_HWCAP2`.
///
/// Linux reports no maximum streaming vector length, so only the `RDSVL`
/// probe checks it, and no cluster topology the backend relies on, so the
/// unit count is 1.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn linux_report(hwcap2: u64) -> OsSme {
    OsSme {
        sme: hwcap2 & HWCAP2_SME != 0,
        sme2: hwcap2 & HWCAP2_SME2 != 0,
        f64f64: hwcap2 & HWCAP2_SME_F64F64 != 0,
        max_svl_bytes: None,
        units: 1,
    }
}

// ---------------------------------------------------------------------------
// Other operating systems
// ---------------------------------------------------------------------------

/// Other operating systems offer no supported way to ask, so SME2 stays off.
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
fn os_report() -> Option<OsSme> {
    None
}
