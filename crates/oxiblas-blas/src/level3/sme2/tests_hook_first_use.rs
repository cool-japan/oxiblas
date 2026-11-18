//! Tests of the first use of the backend through the public entry points:
//! which thread runs the first-use check, and what concurrent first calls
//! see.
//!
//! The tests are conclusive only in a process whose verdict is still open
//! (for the last one: whose detection has not started at all) when they
//! start, which an exact-filter run gives (each is then the only test that
//! uses the backend). In a full run another test has usually settled the
//! detection first: they then print `inconclusive` and check only what
//! holds in every process (the bits, and that no part of a call runs a
//! check). They are gated by the hardware gate, which reads the candidate
//! and never starts the check (the backend gate would settle the verdict);
//! the last one asks the gate only after its concurrent first calls, since
//! the gate computes the candidate.
//!
//! The tests of this file run one at a time ([`one_at_a_time`], taken
//! before anything else): each of them may be the one that computes the
//! candidate or runs the check, and two of them at once would each see the
//! other's work as their own. A test of another file that computes the
//! candidate or starts the check while one of them runs makes it
//! inconclusive: the process-wide counts (`candidate_runs`,
//! `checks_started`) show that, and the test prints it ([`judge_check`]).

use std::sync::{Arc, Barrier, Mutex, MutexGuard};

#[cfg(feature = "parallel")]
use oxiblas_core::parallel::Par;

use super::SME2_ALLOWED;
use super::detect::{
    FIRST_USE_KERNEL_CALLS, FIRST_USE_KERNEL_CALLS_F32, Verdict, candidate, candidate_known,
    candidate_runs, checks_started, verdict,
};
use super::test_support::{
    Need, backend_or_skip, decisions, force_off, hardware_or_skip, quiet_kernel_entries,
};
use super::tests_gemm_harness::TestElem;
use super::tests_hook::{Call, Entry, Run, assert_same_c, run_public, run_reference};
use crate::level2::gemv::{GemvTrans, gemv};

/// What one thread of the concurrent test computed: the check's kernel
/// calls on it, and one of its four results.
type Outcome = (
    u64,
    Option<Run<f32>>,
    Option<Run<f64>>,
    Option<Vec<f32>>,
    Option<Vec<f64>>,
);

/// Held by the test of this file that runs (module docs).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Waits until no other test of this file runs; the guard lets the next one
/// in. A test that failed (a poisoned lock) does not stop the others.
fn one_at_a_time() -> MutexGuard<'static, ()> {
    match ONE_AT_A_TIME.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

/// The state of the first-use check when a test began: whether it can be
/// the test's threads that run it.
#[derive(Debug, Clone, Copy)]
struct CheckStart {
    /// The verdict was open.
    open: bool,
    /// Checks the process had started (read before the verdict).
    started: usize,
}

impl CheckStart {
    /// The state now.
    fn now() -> Self {
        let started = checks_started();
        Self {
            open: verdict() == Verdict::Open,
            started,
        }
    }

    /// Whether the test's threads must be the ones that run the check: the
    /// verdict was open and no check had started.
    fn fresh(self) -> bool {
        self.open && self.started == 0
    }
}

/// Judges the check kernel calls `ours` made on the test's threads, for a
/// test that began in the state `start` and has made all its first calls:
/// `true` when the test is conclusive (its threads ran the check, once,
/// and the verdict is "passed"), `false` (printed) when it is not. Fails
/// for what holds in every process: the test's threads ran no check or
/// exactly one; none when the verdict was settled or its check had started
/// when the test began; and when the verdict was open and no check had
/// started, a check ran during the test (on the test's threads, or on
/// another test's).
fn judge_check(test: &str, start: CheckStart, ours: u64) -> bool {
    let one = check_calls();
    assert!(
        ours == 0 || ours == one,
        "{test}: the test's threads made {ours} check kernel calls, not 0 and not one check ({one})"
    );
    if !start.fresh() {
        assert_eq!(
            ours, 0,
            "{test}: a check ran on the test's threads although the check was settled or started \
             before the test began ({start:?})"
        );
        println!(
            "{test}: inconclusive: the check was settled or started before the test ({start:?})"
        );
        return false;
    }
    if ours == one {
        let settled = verdict();
        assert_eq!(
            settled,
            Verdict::Passed,
            "{test}: the verdict after the test's check"
        );
        return true;
    }
    let started = checks_started();
    assert!(
        started > start.started,
        "{test}: the verdict was open and no check had started, yet no check ran during the \
         first calls ({ours} check kernel calls on the test's threads, {started} checks started)"
    );
    println!("{test}: inconclusive: another test ran the check while this one ran");
    false
}

/// The kernel calls of the first-use check on this CPU.
fn check_calls() -> u64 {
    if candidate().is_some_and(|hw| hw.f64) {
        FIRST_USE_KERNEL_CALLS
    } else {
        FIRST_USE_KERNEL_CALLS_F32
    }
}

/// Whether the backend can be on in this build at all; prints a note when
/// it cannot (a build that caps SIMD).
fn backend_allowed(test: &str) -> bool {
    if !SME2_ALLOWED {
        println!("note: {test}: this build caps SIMD below SME2; the hooks decline everywhere");
    }
    SME2_ALLOWED
}

/// A GEMV past the crossover on this thread: `y` after the call, and the
/// decisions it took.
fn public_gemv<T: TestElem>(seed: usize) -> (Vec<T>, u64) {
    let (m, n) = (200, 150);
    let a: Vec<T> = (0..m * n)
        .map(|index| T::of(((index * 13 + seed) % 97) as f64 / 64.0 - 0.7))
        .collect();
    let x: Vec<T> = (0..n)
        .map(|index| T::of((index % 11) as f64 * 0.125 - 0.5))
        .collect();
    let mut y: Vec<T> = (0..m)
        .map(|index| T::of((index % 5) as f64 * 0.25))
        .collect();
    let view = match oxiblas_matrix::MatRef::from_column_major(&a, m, n) {
        Some(view) => view,
        None => panic!("GEMV view"),
    };
    let decided = decisions();
    gemv(
        GemvTrans::NoTrans,
        T::of(0.75),
        view,
        &x,
        T::of(0.5),
        &mut y,
    );
    (y, decisions() - decided)
}

/// Several threads start public GEMM and GEMV calls at the same moment in a
/// process whose verdict is open: one of them runs the first-use check
/// (exactly once in the process: the check's kernel calls on these threads
/// add up to one check), the others do not wait for it, every result
/// equals the NEON reference bit for bit, and afterwards the calls run on
/// SME2.
#[test]
fn test_sme2_hook_first_use_concurrent_public_calls() {
    let test = "test_sme2_hook_first_use_concurrent_public_calls";
    let _alone = one_at_a_time();
    if !hardware_or_skip(test, Need::F64) || !backend_allowed(test) {
        return;
    }
    let check_start = CheckStart::now();
    let gemm_f32 = Call::new(Entry::GemmWithPar, 70, 49, 65);
    let gemm_f64 = Call::new(Entry::Transposed, 70, 25, 65).seed(3);
    // References first, under force_off: `enabled` answers before the
    // detection is asked, so the verdict stays as it is.
    let (want_f32, want_f64, want_gemv_f32, want_gemv_f64) = {
        let _off = force_off();
        (
            run_reference::<f32>(&gemm_f32),
            run_reference::<f64>(&gemm_f64),
            public_gemv::<f32>(1).0,
            public_gemv::<f64>(2).0,
        )
    };
    let threads = 6;
    let start = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|index| {
            let start = Arc::clone(&start);
            std::thread::spawn(move || -> Outcome {
                let quiet = quiet_kernel_entries();
                start.wait();
                let (gemm_32, gemm_64, gemv_32, gemv_64) = match index % 4 {
                    0 => (Some(run_public(&gemm_f32)), None, None, None),
                    1 => (None, Some(run_public(&gemm_f64)), None, None),
                    2 => (None, None, Some(public_gemv(1).0), None),
                    _ => (None, None, None, Some(public_gemv(2).0)),
                };
                (
                    quiet_kernel_entries() - quiet,
                    gemm_32,
                    gemm_64,
                    gemv_32,
                    gemv_64,
                )
            })
        })
        .collect();
    let mut check_kernel_calls = 0;
    for (index, handle) in handles.into_iter().enumerate() {
        let Ok((quiet, gemm_32, gemm_64, gemv_32, gemv_64)) = handle.join() else {
            panic!("{test}: thread {index} panicked");
        };
        let label = format!("{test}: thread {index}");
        check_kernel_calls += quiet;
        if let Some(got) = gemm_32 {
            assert_same_c(&got, &want_f32, &label);
        }
        if let Some(got) = gemm_64 {
            assert_same_c(&got, &want_f64, &label);
        }
        if let Some(got) = gemv_32 {
            let same = got
                .iter()
                .zip(&want_gemv_f32)
                .all(|(g, w)| g.to_bits() == w.to_bits());
            assert!(same, "{label}: GEMV f32 differs from NEON");
        }
        if let Some(got) = gemv_64 {
            let same = got
                .iter()
                .zip(&want_gemv_f64)
                .all(|(g, w)| g.to_bits() == w.to_bits());
            assert!(same, "{label}: GEMV f64 differs from NEON");
        }
    }
    if judge_check(test, check_start, check_kernel_calls) {
        println!(
            "{test}: conclusive: one thread ran the check ({check_kernel_calls} kernel calls)"
        );
    }
    // Afterwards the calls run on SME2.
    assert!(
        backend_or_skip(test, Need::F64),
        "{test}: the backend is off after the check"
    );
    assert_eq!(
        run_public::<f32>(&gemm_f32).decisions,
        1,
        "{test}: GEMM f32 after"
    );
    assert_eq!(
        run_public::<f64>(&gemm_f64).decisions,
        1,
        "{test}: GEMM f64 after"
    );
    assert_eq!(public_gemv::<f32>(1).1, 1, "{test}: GEMV f32 after");
    assert_eq!(public_gemv::<f64>(2).1, 1, "{test}: GEMV f64 after");
}

/// No part of an SME2 call is the thread that runs the first-use check: a
/// public call split over three SME units on rayon's global pool, made in a
/// process whose verdict is open, runs the check on the calling thread (in
/// its decision, before any part runs), and none of the pool's workers runs
/// any of it.
#[cfg(feature = "parallel")]
#[test]
fn test_sme2_hook_first_use_runs_on_the_calling_thread() {
    use super::detect::override_units;
    use super::driver::{TuningOverride, override_tuning};

    let test = "test_sme2_hook_first_use_runs_on_the_calling_thread";
    let _alone = one_at_a_time();
    if !hardware_or_skip(test, Need::F64) || !backend_allowed(test) {
        return;
    }
    let pool_quiet = || rayon::broadcast(|_| quiet_kernel_entries());
    let workers_before = pool_quiet();
    let check_start = CheckStart::now();
    let call = Call::new(Entry::TransposedWithPar, 200, 101, 96).par(Par::RayonWith(3));
    let _units = override_units(3);
    let _tuning = override_tuning(TuningOverride {
        par_min_macs: Some(1 << 12),
        ..TuningOverride::default()
    });
    let want = run_reference::<f32>(&call);
    let quiet = quiet_kernel_entries();
    let got = run_public::<f32>(&call);
    let here = quiet_kernel_entries() - quiet;
    let workers_after = pool_quiet();
    assert_eq!(
        workers_before, workers_after,
        "{test}: a worker of the pool ran first-use check kernels"
    );
    assert_same_c(&got, &want, test);
    let conclusive = judge_check(test, check_start, here);
    if conclusive {
        println!("{test}: conclusive: the calling thread ran the check ({here} kernel calls)");
    } else if check_start.open {
        // Another test's check was in progress during the call, which then
        // ran on NEON: nothing more holds in every process.
        return;
    }
    assert_eq!(got.decisions, 1, "{test}: the call did not run on SME2");
    let parts = got.report.as_ref().map(|report| report.neon.clone());
    assert_eq!(parts, Some(vec![false; 3]), "{test}: the parts");
}

/// Several threads make their first public GEMM and GEMV calls at the same
/// moment in a process whose detection has not started: the candidate is
/// computed once (on one of them; the others do not wait for it, nor for
/// the first-use check), the check runs once, every result equals the NEON
/// reference bit for bit, and afterwards the calls run on SME2.
///
/// Nothing before the concurrent calls may start the detection: the
/// references run under `force_off` (`enabled` answers before it asks the
/// detection), and the hardware gate comes after the calls (it computes the
/// candidate). Before the gate the test executes no SME code itself; the
/// public calls run SME2 only where the detection found it.
#[test]
fn test_sme2_hook_first_use_concurrent_first_detection() {
    let test = "test_sme2_hook_first_use_concurrent_first_detection";
    let _alone = one_at_a_time();
    if !SME2_ALLOWED {
        // A build that caps SIMD never runs the detection from the BLAS paths.
        let _ = hardware_or_skip(test, Need::F64);
        let _ = backend_allowed(test);
        return;
    }
    // The count first: a candidate unknown after a count of 0 was not being
    // computed when the count was read.
    let runs_before = candidate_runs();
    let fresh = runs_before == 0 && !candidate_known();
    let check_start = CheckStart::now();
    let gemm_f32 = Call::new(Entry::GemmWithPar, 70, 49, 65);
    let gemm_f64 = Call::new(Entry::Transposed, 70, 25, 65).seed(5);
    let (want_f32, want_f64, want_gemv_f32, want_gemv_f64) = {
        let _off = force_off();
        (
            run_reference::<f32>(&gemm_f32),
            run_reference::<f64>(&gemm_f64),
            public_gemv::<f32>(3).0,
            public_gemv::<f64>(4).0,
        )
    };
    // The references never ask the detection; a candidate computed by now
    // was computed by another test, which makes this one inconclusive.
    let runs_after_references = candidate_runs();
    let computed_elsewhere = fresh && runs_after_references != 0;
    if computed_elsewhere {
        println!(
            "{test}: inconclusive: another test computed the candidate while the references ran \
             ({runs_after_references} computations)"
        );
    }
    let threads = 8;
    let start = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|index| {
            let start = Arc::clone(&start);
            std::thread::spawn(move || -> Outcome {
                let quiet = quiet_kernel_entries();
                start.wait();
                let (gemm_32, gemm_64, gemv_32, gemv_64) = match index % 4 {
                    0 => (Some(run_public(&gemm_f32)), None, None, None),
                    1 => (None, Some(run_public(&gemm_f64)), None, None),
                    2 => (None, None, Some(public_gemv(3).0), None),
                    _ => (None, None, None, Some(public_gemv(4).0)),
                };
                (
                    quiet_kernel_entries() - quiet,
                    gemm_32,
                    gemm_64,
                    gemv_32,
                    gemv_64,
                )
            })
        })
        .collect();
    let mut check_kernel_calls = 0;
    for (index, handle) in handles.into_iter().enumerate() {
        let Ok((quiet, gemm_32, gemm_64, gemv_32, gemv_64)) = handle.join() else {
            panic!("{test}: thread {index} panicked");
        };
        let label = format!("{test}: thread {index}");
        check_kernel_calls += quiet;
        if let Some(got) = gemm_32 {
            assert_same_c(&got, &want_f32, &label);
        }
        if let Some(got) = gemm_64 {
            assert_same_c(&got, &want_f64, &label);
        }
        if let Some(got) = gemv_32 {
            let differ = got
                .iter()
                .zip(&want_gemv_f32)
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                differ, 0,
                "{label}: GEMV f32 differs from NEON in {differ} elements"
            );
        }
        if let Some(got) = gemv_64 {
            let differ = got
                .iter()
                .zip(&want_gemv_f64)
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                differ, 0,
                "{label}: GEMV f64 differs from NEON in {differ} elements"
            );
        }
    }
    // The candidate is computed once per process, and the concurrent calls
    // computed it unless someone else had: 1 in every process.
    let runs = candidate_runs();
    assert_eq!(
        runs, 1,
        "{test}: candidate computations after the concurrent first calls ({runs_before} before)"
    );
    // The gate after the calls: it reads the candidate they computed.
    if !hardware_or_skip(test, Need::F64) {
        let runs = candidate_runs();
        assert_eq!(runs, 1, "{test}: the candidate was computed {runs} times");
        return;
    }
    let checked_here = judge_check(test, check_start, check_kernel_calls);
    if fresh && !computed_elsewhere && checked_here {
        println!(
            "{test}: conclusive: the candidate was computed once and one thread ran the check \
             ({check_kernel_calls} kernel calls)"
        );
    } else if !fresh {
        println!(
            "{test}: inconclusive: the detection had started before the test ({check_start:?})"
        );
    }
    // Afterwards the calls run on SME2, and nothing computed the candidate
    // again.
    assert!(
        backend_or_skip(test, Need::F64),
        "{test}: the backend is off after the check"
    );
    let decided = run_public::<f32>(&gemm_f32).decisions;
    assert_eq!(
        decided, 1,
        "{test}: GEMM f32 afterwards made {decided} decisions"
    );
    let decided = run_public::<f64>(&gemm_f64).decisions;
    assert_eq!(
        decided, 1,
        "{test}: GEMM f64 afterwards made {decided} decisions"
    );
    let decided = public_gemv::<f32>(3).1;
    assert_eq!(
        decided, 1,
        "{test}: GEMV f32 afterwards made {decided} decisions"
    );
    let decided = public_gemv::<f64>(4).1;
    assert_eq!(
        decided, 1,
        "{test}: GEMV f64 afterwards made {decided} decisions"
    );
    let runs = candidate_runs();
    assert_eq!(runs, 1, "{test}: the candidate was computed {runs} times");
}
