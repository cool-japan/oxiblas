//! Tests of where the SME2 GEMM driver checks thread readiness and where
//! its parts run (`parallel.rs`, decision step 5 of `driver.rs`).
//!
//! A kernel may only run on a thread for which `detect::thread_ready()` was
//! true, so the driver checks it twice: once on the calling thread during
//! the decision (a thread that is not ready declines the whole call), and
//! once in every part, on the thread that executes the part (a part whose
//! thread is not ready runs on the NEON sequential driver instead). Under
//! QEMU every thread is ready, so most of these tests use the test-only
//! switches of `detect.rs`: `set_pretend_not_ready` makes `thread_ready()`
//! answer `false` on the thread that set it, and `set_ready_countdown`
//! makes it answer as usual for a number of calls and `false` afterwards.
//! The real-state tests put the calling thread into real states that are
//! not ready instead (`test_sme_state.rs`), which `thread_ready()` reads
//! through its register probe.
//!
//! - **Per-part check** (`only_caller_ready`, `caller_and_one_ready`,
//!   `all_ready`): a dedicated rayon pool of exactly three threads, the call
//!   made inside `install` with three units, the parallel threshold lowered
//!   and the rendezvous of the test tuning on, so the three parts must run
//!   on the three distinct workers, whatever the scheduler does. Chosen
//!   workers pretend not to be ready (set through `broadcast`); every part
//!   must then run on NEON exactly when its own worker pretends, the three
//!   `ThreadId`s must differ, the worker indices must be 0, 1 and 2, no
//!   part may have timed out at the rendezvous, every SME2 part must show
//!   one kernel entry per block on its own thread and every NEON part none,
//!   and C must equal the sequential NEON result bit for bit. A driver that
//!   checked readiness once on the calling thread and handed the verdict to
//!   the parts fails the first two on every run.
//! - **Part 0 on the calling worker** (`part0_not_ready`, `no_part_ready`):
//!   inside `install`, rayon runs the first part of a split call on the
//!   calling worker itself, which the per-part tests above keep ready. Here
//!   the calling worker passes the decision (one call of `thread_ready()`)
//!   and then fails its own check as part 0 (countdown of one call): part 0
//!   must run on NEON on that worker, the others on SME2 (on NEON as well
//!   when every other worker pretends), with every assertion of the
//!   per-part tests. A driver whose part 0 trusts the decision's verdict
//!   fails both on every run.
//! - **Calling thread** (`caller_not_ready`): when the calling thread
//!   pretends not to be ready, `try_gemm` declines: `false`, no decision, no
//!   run report, no kernel entry, C bitwise untouched, below and above the
//!   parallel threshold, from a plain thread and from a pool worker whose
//!   fellow workers are ready.
//! - **Real states** (`real_states_decline`): the calling thread with ZA
//!   storage on outside streaming mode (SVCR = 2) and with TPIDR2_EL0 set
//!   (SVCR = 0): `thread_ready()` and `sme2_active()` are `false` and
//!   `try_gemm` declines as above, while the same call is accepted on the
//!   same thread before and after each state. The pretend switches return
//!   before the registers are read; these are the tests in which the rule
//!   of `ThreadState::ready` decides on what the probe reads. Streaming
//!   mode (SVCR bit 0) has no such test: ordinary code cannot run in it
//!   (`test_sme_state.rs`).
//! - **Parts on workers** (`parts_run_on_workers`): a call made from
//!   outside any pool with two parts runs every part on a rayon worker, not
//!   on the calling thread.
//! - **Flags cleared** (`flags_cleared_after_a_panic`, native too): the drop
//!   guard that clears the flags of a pool's workers does so even when the
//!   test panics with the flags set.
//!
//! The tests that call `try_gemm` start with the backend gate
//! `backend_or_skip`. The per-part tests need rayon (`parallel`).

use std::marker::PhantomData;

use oxiblas_core::parallel::Par;

use super::detect::{self, override_units, set_pretend_not_ready};
#[cfg(feature = "parallel")]
use super::detect::{pretends_not_ready, set_ready_countdown};
#[cfg(feature = "parallel")]
use super::driver::BlockCount;
use super::driver::{TuningOverride, override_tuning};
use super::test_sme_state::{Tpidr2Set, ZaStorageOn};
#[cfg(feature = "parallel")]
use super::test_support::Need;
use super::test_support::{backend_or_skip, kernel_entries_here, sme_state};
use super::tests_gemm_harness::{
    Case, TestElem, assert_declined, assert_matches_neon, run_case, run_case_around,
};
#[cfg(feature = "parallel")]
use super::tests_gemm_harness::{Outcome, assert_runs, assert_same_bits, expected_part};
use crate::level3::trsm::Trans;

/// Workers of the dedicated pool, and parts of each call made in it.
#[cfg(feature = "parallel")]
const WORKERS: usize = 3;

/// The parallel threshold of [`caller_not_ready`] (through the test
/// tuning): its shapes lie on both sides of it.
const CALLER_PAR_MIN_MACS: usize = 1 << 18;

/// Makes the calling thread pretend not to be ready while it lives, then
/// restores the previous setting (it cannot be sent to another thread).
struct PretendHere {
    /// The setting to restore.
    previous: bool,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

impl PretendHere {
    /// Starts pretending on the calling thread.
    fn new() -> Self {
        Self {
            previous: set_pretend_not_ready(true),
            _not_send: PhantomData,
        }
    }
}

impl Drop for PretendHere {
    fn drop(&mut self) {
        set_pretend_not_ready(self.previous);
    }
}

/// Runs a readiness countdown on the calling thread while it lives (the
/// next `calls` calls of `thread_ready()` answer as usual, later ones
/// `false`), then restores the previous countdown (it cannot be sent to
/// another thread).
#[cfg(feature = "parallel")]
struct CountdownHere {
    /// The countdown to restore.
    previous: Option<u32>,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

#[cfg(feature = "parallel")]
impl CountdownHere {
    /// Starts a countdown of `calls` calls on the calling thread.
    fn new(calls: u32) -> Self {
        Self {
            previous: set_ready_countdown(Some(calls)),
            _not_send: PhantomData,
        }
    }
}

#[cfg(feature = "parallel")]
impl Drop for CountdownHere {
    fn drop(&mut self) {
        set_ready_countdown(self.previous);
    }
}

/// Clears the "pretend not ready" flag of every worker of a pool when
/// dropped (a second `broadcast`), so that a failing assertion cannot leave
/// a worker pretending.
#[cfg(feature = "parallel")]
struct ClearFlags<'p> {
    /// The pool whose workers are cleared.
    pool: &'p rayon::ThreadPool,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: PhantomData<*const ()>,
}

#[cfg(feature = "parallel")]
impl<'p> ClearFlags<'p> {
    /// The guard of `pool`.
    fn new(pool: &'p rayon::ThreadPool) -> Self {
        Self {
            pool,
            _not_send: PhantomData,
        }
    }
}

#[cfg(feature = "parallel")]
impl Drop for ClearFlags<'_> {
    fn drop(&mut self) {
        self.pool.broadcast(|_| {
            set_pretend_not_ready(false);
        });
    }
}

/// A rayon pool of exactly `threads` threads.
#[cfg(feature = "parallel")]
fn pool_of(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool")
}

/// The flag of every worker of `pool`, by worker index.
#[cfg(feature = "parallel")]
fn flags(pool: &rayon::ThreadPool) -> Vec<bool> {
    pool.broadcast(|_| pretends_not_ready())
}

/// The calls of the per-part tests: a row split and a column split into
/// three parts, NoTrans and Trans operands, several depth blocks.
#[cfg(feature = "parallel")]
fn split_cases<T: TestElem>() -> Vec<Case<T>> {
    let mut cases = Vec::new();
    for (m, n) in [(200, 2 * T::NR + 1), (70, 25 * T::NR - 3)] {
        for trans in [Trans::NoTrans, Trans::Trans] {
            cases.push(
                Case::<T>::new(m, n, 96)
                    .explicit_kc(32)
                    .trans(trans, trans)
                    .par(Par::RayonWith(WORKERS)),
            );
        }
    }
    cases
}

/// Runs `case` inside `pool`, on the worker that `install` picks (the
/// calling worker), after setting the flags of the workers: the calling
/// worker and the first `others_ready` of the other workers stay ready, the
/// rest pretend not to be. Three units, every work splits, rendezvous on.
/// Returns the outcome, the calling worker's index and the indices of the
/// workers that pretended.
#[cfg(feature = "parallel")]
fn run_in_pool<T: TestElem>(
    pool: &rayon::ThreadPool,
    case: &Case<T>,
    others_ready: usize,
) -> (Outcome<T>, usize, Vec<usize>) {
    pool.install(|| {
        let caller = rayon::current_thread_index().expect("install runs on a worker");
        let flagged: Vec<usize> = (0..WORKERS)
            .filter(|&worker| worker != caller)
            .skip(others_ready)
            .collect();
        rayon::broadcast(|ctx| {
            set_pretend_not_ready(flagged.contains(&ctx.index()));
        });
        let _switches = (
            override_units(WORKERS),
            override_tuning(TuningOverride {
                par_min_macs: Some(0),
                rendezvous: true,
                ..TuningOverride::default()
            }),
        );
        (run_case(case), caller, flagged)
    })
}

/// The per-part tests: for every case of [`split_cases`], the calling
/// worker and `others_ready` other workers ready, the rest pretending (see
/// the module docs for what is asserted).
#[cfg(feature = "parallel")]
fn ready_workers<T: TestElem>(test: &str, others_ready: usize) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let pool = pool_of(WORKERS);
    let clear = ClearFlags::new(&pool);
    for case in split_cases::<T>() {
        let (out, caller, flagged) = run_in_pool(&pool, &case, others_ready);
        let label = format!("{test}: caller {caller}, pretending {flagged:?}, {case:?}");
        assert_eq!(flagged.len(), WORKERS - 1 - others_ready, "{label}");
        assert_same_bits(&out, &label);
        let report = out.report.as_ref().expect("run report");
        assert_eq!(report.layout.parts.len(), WORKERS, "{label}: parts");
        assert_eq!(
            report.timed_out,
            vec![false; WORKERS],
            "{label}: rendezvous"
        );
        // Three distinct threads, the three workers of the pool.
        for (index, thread) in report.threads.iter().enumerate() {
            assert!(
                !report.threads[..index].contains(thread),
                "{label}: parts share a thread: {:?}",
                report.threads
            );
        }
        let mut workers = report.workers.clone();
        workers.sort_unstable();
        assert_eq!(
            workers,
            (0..WORKERS).map(Some).collect::<Vec<_>>(),
            "{label}: worker indices"
        );
        // A part runs on NEON exactly when its own thread pretends.
        let pretended: Vec<bool> = report
            .workers
            .iter()
            .map(|worker| worker.is_some_and(|worker| flagged.contains(&worker)))
            .collect();
        assert_eq!(report.neon, pretended, "{label}: NEON parts");
        assert_eq!(
            report.neon.iter().filter(|neon| !**neon).count(),
            WORKERS - flagged.len(),
            "{label}: SME2 parts"
        );
        // Kernel entries on each part's own thread: one per block of an
        // SME2 part, none for a NEON part.
        assert_runs(&out, &label, |index, part| {
            if pretended[index] {
                (true, BlockCount::default())
            } else {
                (false, expected_part(&out, part))
            }
        });
        if flagged.is_empty() {
            assert_matches_neon(&out, &label);
        }
    }
    drop(clear);
    // A later call on these workers sees no flag.
    assert_eq!(flags(&pool), vec![false; WORKERS], "{test}: flags left set");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f32_only_caller_ready() {
    ready_workers::<f32>("test_sme2_threads_f32_only_caller_ready", 0);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f64_only_caller_ready() {
    ready_workers::<f64>("test_sme2_threads_f64_only_caller_ready", 0);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f32_caller_and_one_ready() {
    ready_workers::<f32>("test_sme2_threads_f32_caller_and_one_ready", 1);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f64_caller_and_one_ready() {
    ready_workers::<f64>("test_sme2_threads_f64_caller_and_one_ready", 1);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f32_all_ready() {
    ready_workers::<f32>("test_sme2_threads_f32_all_ready", WORKERS - 1);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f64_all_ready() {
    ready_workers::<f64>("test_sme2_threads_f64_all_ready", WORKERS - 1);
}

/// Part 0 on a calling worker that is not ready (see the module docs): for
/// every case of [`split_cases`], inside a pool of [`WORKERS`] threads with
/// three units, the threshold lowered and the rendezvous on, the calling
/// worker passes the decision and then answers "not ready" (a countdown of
/// one call, started just before `try_gemm`); the other workers are ready
/// (`others_ready`) or all pretend not to be. Part 0 must have run on the
/// calling worker, on NEON; every other part on its own worker, on SME2
/// exactly when that worker is ready; the bits are those of the sequential
/// NEON result.
#[cfg(feature = "parallel")]
fn part0_not_ready<T: TestElem>(test: &str, others_ready: bool) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let pool = pool_of(WORKERS);
    let clear = ClearFlags::new(&pool);
    for case in split_cases::<T>() {
        let (out, caller, caller_thread) = pool.install(|| {
            let caller = rayon::current_thread_index().expect("install runs on a worker");
            rayon::broadcast(|ctx| {
                set_pretend_not_ready(!others_ready && ctx.index() != caller);
            });
            let _switches = (
                override_units(WORKERS),
                override_tuning(TuningOverride {
                    par_min_macs: Some(0),
                    rendezvous: true,
                    ..TuningOverride::default()
                }),
            );
            // One call answers as usual (decision step 5); the next one,
            // part 0's own check on this worker, answers `false`.
            let out = run_case_around(&case, |call| {
                let _countdown = CountdownHere::new(1);
                call()
            });
            (out, caller, std::thread::current().id())
        });
        let label = format!("{test}: caller {caller}, others ready: {others_ready}, {case:?}");
        assert_same_bits(&out, &label);
        let report = out.report.as_ref().expect("run report");
        assert_eq!(report.layout.parts.len(), WORKERS, "{label}: parts");
        assert_eq!(
            report.timed_out,
            vec![false; WORKERS],
            "{label}: rendezvous"
        );
        // Part 0 on the calling worker (the premise), the others on the
        // other workers, three distinct threads.
        assert_eq!(report.threads[0], caller_thread, "{label}: part 0's thread");
        assert_eq!(report.workers[0], Some(caller), "{label}: part 0's worker");
        for (index, thread) in report.threads.iter().enumerate() {
            assert!(
                !report.threads[..index].contains(thread),
                "{label}: parts share a thread: {:?}",
                report.threads
            );
        }
        let neon: Vec<bool> = (0..WORKERS)
            .map(|index| index == 0 || !others_ready)
            .collect();
        assert_eq!(report.neon, neon, "{label}: NEON parts");
        assert_runs(&out, &label, |index, part| {
            if neon[index] {
                (true, BlockCount::default())
            } else {
                (false, expected_part(&out, part))
            }
        });
    }
    drop(clear);
    assert_eq!(flags(&pool), vec![false; WORKERS], "{test}: flags left set");
    assert_eq!(
        pool.broadcast(|_| set_ready_countdown(None)),
        vec![None; WORKERS],
        "{test}: countdown left set"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f32_part0_not_ready() {
    part0_not_ready::<f32>("test_sme2_threads_f32_part0_not_ready", true);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f64_part0_not_ready() {
    part0_not_ready::<f64>("test_sme2_threads_f64_part0_not_ready", true);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f32_no_part_ready() {
    part0_not_ready::<f32>("test_sme2_threads_f32_no_part_ready", false);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_f64_no_part_ready() {
    part0_not_ready::<f64>("test_sme2_threads_f64_no_part_ready", false);
}

/// The calls of [`caller_not_ready`], all accepted by a ready caller: one
/// below and one above [`CALLER_PAR_MIN_MACS`], NoTrans and Trans.
fn caller_cases<T: TestElem>() -> Vec<Case<T>> {
    #[cfg(feature = "parallel")]
    let par = Par::RayonWith(3);
    #[cfg(not(feature = "parallel"))]
    let par = Par::Seq;
    let mut cases = Vec::new();
    for (m, n, k) in [(70, 2 * T::NR, 64), (200, 3 * T::NR, 96)] {
        for trans in [Trans::NoTrans, Trans::Trans] {
            cases.push(Case::<T>::new(m, n, k).trans(trans, trans).par(par));
        }
    }
    cases
}

/// Runs `case` on the calling thread, which pretends not to be ready, and
/// asserts the decline: `false`, no decision, no run report, no kernel entry
/// on this thread, C bitwise untouched.
fn assert_caller_declines<T: TestElem>(case: &Case<T>, label: &str) {
    let _units = override_units(3);
    let _tuning = override_tuning(TuningOverride {
        par_min_macs: Some(CALLER_PAR_MIN_MACS),
        ..TuningOverride::default()
    });
    let work = case.m * case.n * case.k;
    let _pretend = PretendHere::new();
    let entries = kernel_entries_here();
    let out = run_case(case);
    let label = format!("{label}: work {work} (threshold {CALLER_PAR_MIN_MACS}) {case:?}");
    assert_declined(&out, &label);
    assert_eq!(kernel_entries_here(), entries, "{label}: a kernel ran");
}

/// Decision step 5: a calling thread that is not ready declines the call
/// (see the module docs).
fn caller_not_ready<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let cases = caller_cases::<T>();
    let mut seen = [false; 2];
    for case in &cases {
        seen[usize::from(case.m * case.n * case.k >= CALLER_PAR_MIN_MACS)] = true;
        // The control: the same call from a ready thread is accepted.
        {
            let _units = override_units(3);
            let _tuning = override_tuning(TuningOverride {
                par_min_macs: Some(CALLER_PAR_MIN_MACS),
                ..TuningOverride::default()
            });
            let out = run_case(case);
            assert_matches_neon(&out, &format!("{test}: control"));
        }
        assert_caller_declines(case, &format!("{test}: plain thread"));
    }
    assert_eq!(seen, [true, true], "{test}: both sides of the threshold");
    // From a worker of a pool whose other workers are ready: the caller's
    // verdict decides.
    #[cfg(feature = "parallel")]
    {
        let pool = pool_of(WORKERS);
        let clear = ClearFlags::new(&pool);
        pool.broadcast(|_| {
            set_pretend_not_ready(false);
        });
        pool.install(|| {
            for case in &cases {
                assert_caller_declines(case, &format!("{test}: pool worker"));
            }
        });
        drop(clear);
        assert_eq!(flags(&pool), vec![false; WORKERS], "{test}: flags left set");
    }
}

#[test]
fn test_sme2_threads_f32_caller_not_ready() {
    caller_not_ready::<f32>("test_sme2_threads_f32_caller_not_ready");
}

#[test]
fn test_sme2_threads_f64_caller_not_ready() {
    caller_not_ready::<f64>("test_sme2_threads_f64_caller_not_ready");
}

/// A real state of the calling thread that is not ready (`test_sme_state.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RealState {
    /// ZA storage on outside streaming mode: SVCR = 2, TPIDR2_EL0 = 0.
    ZaOn,
    /// TPIDR2_EL0 set, SVCR = 0.
    Tpidr2Set,
}

impl RealState {
    /// Both states.
    const ALL: [Self; 2] = [Self::ZaOn, Self::Tpidr2Set];

    /// `(SVCR, TPIDR2_EL0)` in the state.
    fn registers(self) -> (u64, u64) {
        match self {
            Self::ZaOn => (2, 0),
            Self::Tpidr2Set => (0, Tpidr2Set::value()),
        }
    }
}

/// What a call made in a real state saw, read before its guard was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InState {
    /// `sme_state()` in the state.
    registers: Option<(u64, u64)>,
    /// `detect::thread_ready()` in the state.
    ready: bool,
    /// `sme2_active()` in the state.
    active: bool,
    /// Kernel entries on this thread during the call.
    kernel_calls: u64,
}

/// Decision step 5 on real states (see the module docs): a call that this
/// thread's ready state accepts (the control, before and after each state)
/// is declined in each state, with C bitwise untouched and no kernel run.
/// Everything the state is checked against is read in the state and
/// asserted after the guard has restored it, so that a failure leaves the
/// thread as it was.
fn real_states_decline<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let case = Case::<T>::new(70, 2 * T::NR, 64).trans(Trans::Trans, Trans::NoTrans);
    let control = |when: &str| {
        assert_eq!(sme_state(), Some((0, 0)), "{test}: state {when}");
        assert_matches_neon(&run_case(&case), &format!("{test}: control {when}"));
    };
    control("before");
    for state in RealState::ALL {
        let mut seen = None;
        let out = run_case_around(&case, |call| {
            let guards = match state {
                RealState::ZaOn => (ZaStorageOn::new(), None),
                RealState::Tpidr2Set => (None, Tpidr2Set::new()),
            };
            let registers = sme_state();
            let ready = detect::thread_ready();
            let active = super::sme2_active();
            let entries = kernel_entries_here();
            let accepted = call();
            seen = Some(InState {
                registers,
                ready,
                active,
                kernel_calls: kernel_entries_here() - entries,
            });
            drop(guards);
            accepted
        });
        let label = format!("{test}: {state:?}, {case:?}");
        assert_eq!(
            seen,
            Some(InState {
                registers: Some(state.registers()),
                ready: false,
                active: false,
                kernel_calls: 0,
            }),
            "{label}: in the state"
        );
        assert_declined(&out, &label);
        control(&format!("after {state:?}"));
    }
}

#[test]
fn test_sme2_threads_f32_real_states_decline() {
    real_states_decline::<f32>("test_sme2_threads_f32_real_states_decline");
}

#[test]
fn test_sme2_threads_f64_real_states_decline() {
    real_states_decline::<f64>("test_sme2_threads_f64_real_states_decline");
}

/// A call made from outside any pool with two parts (rendezvous off, the
/// global pool) runs every part on a rayon worker, not on the calling
/// thread; for both types, a row and a column split.
#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_parts_run_on_workers() {
    let test = "test_sme2_threads_parts_run_on_workers";
    if !backend_or_skip(test, Need::F64) {
        return;
    }
    assert_eq!(rayon::current_thread_index(), None, "{test}: inside a pool");
    let _switches = (
        override_units(2),
        override_tuning(TuningOverride {
            par_min_macs: Some(0),
            ..TuningOverride::default()
        }),
    );
    fn check<T: TestElem>(test: &str) {
        let me = std::thread::current().id();
        for (m, n) in [(200, 2 * T::NR + 1), (70, 25 * T::NR - 3)] {
            let case = Case::<T>::new(m, n, 96)
                .explicit_kc(32)
                .par(Par::RayonWith(2));
            let out = run_case(&case);
            let label = format!("{test}: {case:?}");
            assert_matches_neon(&out, &label);
            let report = out.report.as_ref().expect("run report");
            assert_eq!(report.layout.parts.len(), 2, "{label}: parts");
            assert!(
                report.workers.iter().all(Option::is_some),
                "{label}: a part ran outside rayon: {:?}",
                report.workers
            );
            assert!(
                report.threads.iter().all(|thread| *thread != me),
                "{label}: a part ran on the calling thread"
            );
        }
    }
    check::<f32>(test);
    check::<f64>(test);
}

/// The drop guard clears the flags of every worker even when the test
/// panics while they are set, so a later test on the same workers sees no
/// flag. Executes no SME2 code (runs everywhere).
#[cfg(feature = "parallel")]
#[test]
fn test_sme2_threads_flags_cleared_after_a_panic() {
    let pool = pool_of(WORKERS);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _clear = ClearFlags::new(&pool);
        pool.broadcast(|_| {
            set_pretend_not_ready(true);
        });
        assert_eq!(flags(&pool), vec![true; WORKERS]);
        // Set flags make `thread_ready` false on every worker.
        assert_eq!(
            pool.broadcast(|_| super::detect::thread_ready()),
            vec![false; WORKERS]
        );
        panic!("a failing assertion while the flags are set");
    }));
    assert!(result.is_err(), "the closure did not panic");
    assert_eq!(flags(&pool), vec![false; WORKERS], "flags left set");
}
