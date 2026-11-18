//! Tests of the first-use check (stage 2 of the detection): the verdict's
//! state machine on local [`FirstUse`] values with stand-in checks and
//! readiness (on any host), [`run_checks`] with stand-in checks (order,
//! first failure, floating-point state, quiet counters; on any AArch64
//! CPU), the failure line, and, on SME2 hardware (QEMU), the real check:
//! it passes, settles the process's verdict, is invisible to the test
//! counters and costs the documented kernel calls.

use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use super::super::args::{MacroArgs, SmeElement};
use super::super::few_rows::regions_on_this_thread;
use super::super::fpstate::{fpcr_read, fpcr_write, fpsr_read, fpsr_write};
use super::super::selftest::{KernelFn, self_test_with_detailed};
use super::super::test_support::{
    FPSR_DZC, FPSR_IDC, FPSR_IOC, FPSR_IXC, FPSR_OFC, FPSR_QC, FPSR_UFC, Need, decisions,
    hardware_or_skip, kernel_entries_here, note_kernel_entry, quiet_kernel_entries,
};
use super::*;

/// How long a test waits for another thread before it calls that a hang.
const PATIENCE: Duration = Duration::from_secs(20);

/// A check that passes.
fn pass() -> Result<(), CheckFailure> {
    Ok(())
}

/// A readiness answer: ready.
fn ready() -> bool {
    true
}

/// A readiness answer: not ready.
fn not_ready() -> bool {
    false
}

/// The first mismatch of a plain-Rust stand-in macro-kernel that writes
/// nothing (so its first block differs from the oracle).
fn some_mismatch() -> Mismatch {
    /// A stand-in macro-kernel that leaves C as it is.
    ///
    /// # Safety
    ///
    /// Any argument block (it reads and writes nothing).
    unsafe fn leaves_c(_args: &MacroArgs<f32>) {}
    let kernel: KernelFn<f32> = leaves_c;
    // SAFETY: the stand-in is plain Rust and touches nothing.
    match unsafe { self_test_with_detailed(kernel) } {
        Err(mismatch) => mismatch,
        Ok(()) => panic!("a kernel that writes nothing passed the self-test"),
    }
}

#[test]
fn test_sme2_detect_first_use_failure_line_text() {
    let prefix = "oxiblas: the SME2 backend is off for this process: ";
    let suffix = "; please report this line";
    for (failure, what) in [
        (
            CheckFailure::ZaPackers,
            "the check of the SME2 ZA transposers failed",
        ),
        (
            CheckFailure::FewRows,
            "the check of the SME2 few-rows GEMM failed",
        ),
        (CheckFailure::Gemv, "the check of the SME2 GEMV failed"),
        (
            CheckFailure::Poisoned,
            "a first-use check of SME2 panicked (its lock is poisoned)",
        ),
    ] {
        assert_eq!(failure_line(&failure), format!("{prefix}{what}{suffix}"));
    }
    let mismatch = some_mismatch();
    let line = failure_line(&CheckFailure::MacroKernels(mismatch));
    assert_eq!(
        line,
        format!("{prefix}the self-test of the SME2 macro-kernels failed ({mismatch}){suffix}")
    );
    assert!(
        line.contains("(SME2 self-test mismatch, f32: block "),
        "{line}"
    );
    // One line each.
    for failure in [
        CheckFailure::MacroKernels(mismatch),
        CheckFailure::ZaPackers,
        CheckFailure::FewRows,
        CheckFailure::Gemv,
        CheckFailure::Poisoned,
    ] {
        let line = failure_line(&failure);
        assert!(!line.contains(['\n', '\r']), "{line:?}");
    }
}

#[test]
fn test_sme2_detect_first_use_not_ready_thread_leaves_the_verdict_open() {
    let first_use = FirstUse::new();
    let checks = Cell::new(0_u32);
    let check = || {
        checks.set(checks.get() + 1);
        Ok(())
    };
    let reports = RefCell::new(Vec::new());
    let report = |line: &str| reports.borrow_mut().push(line.to_owned());
    // A thread that is not ready: `false`, no check, the verdict stays open.
    for _ in 0..3 {
        assert!(!first_use.passed(not_ready, check, report));
        assert_eq!(first_use.verdict(), Verdict::Open);
    }
    assert_eq!(checks.get(), 0, "a thread that is not ready ran the check");
    // A failing check offered to a thread that is not ready is not run
    // either: the verdict cannot become "failed" that way.
    let failing = || {
        checks.set(checks.get() + 1);
        Err(CheckFailure::Gemv)
    };
    assert!(!first_use.passed(not_ready, failing, report));
    assert_eq!(first_use.verdict(), Verdict::Open);
    // The next ready thread settles it.
    assert!(first_use.passed(ready, check, report));
    assert_eq!(first_use.verdict(), Verdict::Passed);
    assert_eq!(checks.get(), 1);
    assert!(reports.borrow().is_empty());
}

#[test]
fn test_sme2_detect_first_use_check_runs_once_after_passed() {
    let first_use = FirstUse::new();
    let (checks, readiness) = (Cell::new(0_u32), Cell::new(0_u32));
    let probe = || {
        readiness.set(readiness.get() + 1);
        true
    };
    let check = || {
        checks.set(checks.get() + 1);
        Ok(())
    };
    let failing = || {
        checks.set(checks.get() + 1);
        Err(CheckFailure::FewRows)
    };
    let report = |line: &str| panic!("reported {line}");
    assert!(first_use.passed(probe, check, report));
    for _ in 0..5 {
        assert!(first_use.passed(probe, failing, report));
        // A settled verdict holds for a thread that is not ready as well.
        assert!(first_use.passed(not_ready, failing, report));
    }
    assert_eq!(checks.get(), 1, "the check ran again after it passed");
    assert_eq!(readiness.get(), 1, "a settled verdict probed the thread");
    assert_eq!(first_use.verdict(), Verdict::Passed);
}

#[test]
fn test_sme2_detect_first_use_failed_is_final_and_reported_once() {
    let first_use = FirstUse::new();
    let checks = Cell::new(0_u32);
    let failing = || {
        checks.set(checks.get() + 1);
        Err(CheckFailure::ZaPackers)
    };
    let check = || {
        checks.set(checks.get() + 1);
        Ok(())
    };
    let reports = RefCell::new(Vec::new());
    let report = |line: &str| reports.borrow_mut().push(line.to_owned());
    assert!(!first_use.passed(ready, failing, report));
    let line = failure_line(&CheckFailure::ZaPackers);
    assert_eq!(first_use.verdict(), Verdict::Failed(line.as_str()));
    for _ in 0..5 {
        assert!(!first_use.passed(ready, check, report));
        assert!(!first_use.passed(not_ready, check, report));
    }
    assert_eq!(checks.get(), 1, "the check ran again after it failed");
    assert_eq!(first_use.verdict(), Verdict::Failed(line.as_str()));
    assert_eq!(*reports.borrow(), [line], "the line was not reported once");
}

#[test]
fn test_sme2_detect_first_use_waiters_see_one_answer() {
    let first_use = Arc::new(FirstUse::new());
    let checks = Arc::new(StdMutex::new(0_u32));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    // The thread that runs the check: it holds the lock until released.
    let checker = {
        let (first_use, checks) = (Arc::clone(&first_use), Arc::clone(&checks));
        std::thread::spawn(move || {
            first_use.passed(
                ready,
                || {
                    if let Ok(mut count) = checks.lock() {
                        *count += 1;
                    }
                    let _ = started_tx.send(());
                    match release_rx.recv_timeout(PATIENCE) {
                        Ok(()) => Ok(()),
                        Err(_) => Err(CheckFailure::Gemv),
                    }
                },
                |_| {},
            )
        })
    };
    assert!(
        started_rx.recv_timeout(PATIENCE).is_ok(),
        "the check did not start"
    );
    // A thread that is not ready gets `false` at once from the non-waiting
    // form while the check runs: it does not wait for the lock, and leaves
    // the verdict open. (The waiting form waits for the check:
    // `tests_detect_wait.rs`.)
    let (answer_tx, answer_rx) = mpsc::channel();
    {
        let first_use = Arc::clone(&first_use);
        std::thread::spawn(move || {
            let answer = first_use.passed_now(not_ready, || Err(CheckFailure::FewRows), |_| {});
            let _ = answer_tx.send((answer, first_use.verdict() == Verdict::Open));
        });
    }
    let not_ready_answer = answer_rx.recv_timeout(PATIENCE);
    // Ready threads that arrive while the check runs wait, and then read the
    // settled verdict; their own (failing) checks never run.
    let waiters: Vec<_> = (0..4)
        .map(|_| {
            let (first_use, checks) = (Arc::clone(&first_use), Arc::clone(&checks));
            std::thread::spawn(move || {
                first_use.passed(
                    ready,
                    || {
                        if let Ok(mut count) = checks.lock() {
                            *count += 1;
                        }
                        Err(CheckFailure::FewRows)
                    },
                    |_| {},
                )
            })
        })
        .collect();
    // Give the waiters time to reach the lock (the result does not depend
    // on it), then release the check.
    std::thread::sleep(Duration::from_millis(100));
    let _ = release_tx.send(());
    let checker_answer = checker.join();
    let waiter_answers: Vec<_> = waiters.into_iter().map(|w| w.join()).collect();
    assert!(
        matches!(not_ready_answer, Ok((false, true))),
        "a thread that is not ready waited for the check or changed the verdict: \
         {not_ready_answer:?}"
    );
    assert!(matches!(checker_answer, Ok(true)), "{checker_answer:?}");
    for answer in &waiter_answers {
        assert!(matches!(answer, Ok(true)), "a waiter read {answer:?}");
    }
    assert_eq!(first_use.verdict(), Verdict::Passed);
    let count = checks.lock().map(|count| *count).unwrap_or(0);
    assert_eq!(count, 1, "the check ran {count} times");
}

#[test]
fn test_sme2_detect_first_use_reentrant_call_does_not_wait() {
    let first_use = Arc::new(FirstUse::new());
    let (tx, rx) = mpsc::channel();
    {
        let first_use = Arc::clone(&first_use);
        std::thread::spawn(move || {
            let inner = Cell::new(None);
            let outer = first_use.passed(
                ready,
                || {
                    // A call from inside the check, on the checking thread.
                    inner.set(Some(first_use.passed(ready, pass, |_| {})));
                    Ok(())
                },
                |_| {},
            );
            let _ = tx.send((outer, inner.get()));
        });
    }
    let answer = rx.recv_timeout(PATIENCE);
    assert!(
        matches!(answer, Ok((true, Some(false)))),
        "the call from inside the check: {answer:?} (a timeout is a deadlock)"
    );
    assert_eq!(first_use.verdict(), Verdict::Passed);
}

#[test]
fn test_sme2_detect_first_use_poisoned_lock_fails() {
    let first_use = Arc::new(FirstUse::new());
    let panicked = {
        let first_use = Arc::clone(&first_use);
        std::thread::spawn(move || {
            first_use.passed(
                ready,
                // `resume_unwind` skips the panic hook (no message in the log).
                || std::panic::resume_unwind(Box::new("a check that panics")),
                |_| {},
            )
        })
        .join()
    };
    assert!(panicked.is_err(), "the check did not unwind");
    assert_eq!(first_use.verdict(), Verdict::Open, "the panic settled it");
    let checks = Cell::new(0_u32);
    let reports = RefCell::new(Vec::new());
    let check = || {
        checks.set(checks.get() + 1);
        Ok(())
    };
    let report = |line: &str| reports.borrow_mut().push(line.to_owned());
    assert!(!first_use.passed(ready, check, report));
    assert!(!first_use.passed(ready, check, report));
    let line = failure_line(&CheckFailure::Poisoned);
    assert_eq!(first_use.verdict(), Verdict::Failed(line.as_str()));
    assert_eq!(checks.get(), 0, "the check ran under a poisoned lock");
    assert_eq!(*reports.borrow(), [line]);
}

thread_local! {
    /// The stand-in checks that ran on this thread, with their `f64` flag.
    static LOG: RefCell<Vec<(&'static str, bool)>> = const { RefCell::new(Vec::new()) };
    /// What the stand-in checks saw: (FPCR, FPSR after raising flags).
    static SEEN: RefCell<Vec<(u64, u64)>> = const { RefCell::new(Vec::new()) };
    /// The stand-in that fails (`""`: none).
    static FAILING: Cell<&'static str> = const { Cell::new("") };
}

/// Records a stand-in call: logs it, notes FPCR, raises IXC and IOC, reads
/// FPSR, counts one kernel entry, and answers whether `name` passes.
fn stand_in(name: &'static str, f64_present: bool) -> bool {
    LOG.with(|log| log.borrow_mut().push((name, f64_present)));
    let fpcr = fpcr_read();
    black_box(black_box(1.0_f64) / black_box(3.0_f64));
    black_box(black_box(0.0_f64) * black_box(f64::INFINITY));
    SEEN.with(|seen| seen.borrow_mut().push((fpcr, fpsr_read())));
    note_kernel_entry();
    FAILING.with(Cell::get) != name
}

/// The macro-kernel stand-in.
///
/// # Safety
///
/// None needed (plain Rust); `unsafe` for the type of [`Checks`].
unsafe fn macro_stand_in(f64_present: bool) -> Result<(), Mismatch> {
    if stand_in("macro_kernels", f64_present) {
        Ok(())
    } else {
        Err(some_mismatch())
    }
}

/// The ZA-transposer stand-in.
///
/// # Safety
///
/// None needed (plain Rust).
unsafe fn za_stand_in(f64_present: bool) -> bool {
    stand_in("za_packers", f64_present)
}

/// The few-rows stand-in.
///
/// # Safety
///
/// None needed (plain Rust).
unsafe fn few_rows_stand_in(f64_present: bool) -> bool {
    stand_in("few_rows", f64_present)
}

/// The GEMV stand-in.
///
/// # Safety
///
/// None needed (plain Rust).
unsafe fn gemv_stand_in(f64_present: bool) -> bool {
    stand_in("gemv", f64_present)
}

/// The four stand-ins.
const STAND_INS: Checks = Checks {
    macro_kernels: macro_stand_in,
    za_packers: za_stand_in,
    few_rows: few_rows_stand_in,
    gemv: gemv_stand_in,
};

/// Runs [`run_checks`] on the stand-ins with `failing` failing; returns the
/// outcome and the log.
fn run_stand_ins(
    failing: &'static str,
    f64_present: bool,
) -> (Result<(), CheckFailure>, Vec<(&'static str, bool)>) {
    LOG.with(|log| log.borrow_mut().clear());
    FAILING.with(|name| name.set(failing));
    // SAFETY: the stand-ins are plain Rust.
    let outcome = unsafe { run_checks(&STAND_INS, f64_present) };
    FAILING.with(|name| name.set(""));
    (outcome, LOG.with(|log| log.borrow().clone()))
}

#[test]
fn test_sme2_detect_first_use_runs_the_four_checks_in_order() {
    let order = ["macro_kernels", "za_packers", "few_rows", "gemv"];
    for f64_present in [true, false] {
        let (outcome, log) = run_stand_ins("", f64_present);
        assert_eq!(outcome, Ok(()));
        let want: Vec<_> = order.iter().map(|name| (*name, f64_present)).collect();
        assert_eq!(log, want, "the checks that ran, in order");
    }
    // The first failure ends the check, and names the check that failed.
    for (index, failing) in order.iter().enumerate() {
        let (outcome, log) = run_stand_ins(failing, true);
        let want: Vec<_> = order[..=index].iter().map(|name| (*name, true)).collect();
        assert_eq!(log, want, "{failing} failing");
        let expected = match index {
            0 => CheckFailure::MacroKernels(some_mismatch()),
            1 => CheckFailure::ZaPackers,
            2 => CheckFailure::FewRows,
            _ => CheckFailure::Gemv,
        };
        assert_eq!(outcome, Err(expected), "{failing} failing");
    }
}

#[test]
fn test_sme2_detect_first_use_check_floating_point_state_and_counters() {
    /// FPCR.RMode = round towards minus infinity, and FPCR.DN.
    const CALLER_MODES: u64 = (0b10 << 22) | (1 << 25);
    /// Flags the stand-ins never raise.
    const CALLER_FLAGS: u64 = FPSR_DZC | FPSR_QC;
    let (fpcr, fpsr) = (fpcr_read(), fpsr_read());
    let (here, quiet, made) = (kernel_entries_here(), quiet_kernel_entries(), decisions());
    SEEN.with(|seen| seen.borrow_mut().clear());
    // SAFETY: a test thread; restored below, before any assertion.
    unsafe { fpcr_write(CALLER_MODES) };
    fpsr_write(CALLER_FLAGS);
    let (outcome, log) = run_stand_ins("", true);
    let (fpcr_after, fpsr_after) = (fpcr_read(), fpsr_read());
    // SAFETY: the test thread's own values.
    unsafe { fpcr_write(fpcr) };
    fpsr_write(fpsr);
    let seen = SEEN.with(|seen| seen.borrow().clone());
    assert_eq!(outcome, Ok(()));
    assert_eq!(log.len(), 4);
    assert_eq!(seen.len(), 4);
    for (index, &(fpcr_in, fpsr_in)) in seen.iter().enumerate() {
        assert_eq!(fpcr_in, 0, "check {index}: FPCR inside the check");
        assert_eq!(
            fpsr_in & (FPSR_IXC | FPSR_IOC),
            FPSR_IXC | FPSR_IOC,
            "check {index}: FPSR {fpsr_in:#x}"
        );
    }
    assert_eq!(fpcr_after, CALLER_MODES, "the caller's FPCR afterwards");
    assert_eq!(fpsr_after, CALLER_FLAGS, "the caller's FPSR afterwards");
    // The checks' kernel entries are invisible to the counters, and counted
    // apart.
    assert_eq!(kernel_entries_here(), here, "the check moved the counter");
    assert_eq!(quiet_kernel_entries(), quiet + 4);
    assert_eq!(decisions(), made);
}

#[test]
fn test_sme2_detect_first_use_readers_never_start_the_check() {
    // Without a candidate (an Apple M3, QEMU with a 256-bit streaming vector
    // length) nothing runs, and the verdict stays open for good.
    if candidate().is_none() {
        assert_eq!(hw(), None);
        assert!(!thread_ready());
        assert_eq!(verdict(), Verdict::Open);
    }
    // With one, `hw()` is `Some` exactly when the verdict passed.
    let reported = hw().is_some();
    let passed = verdict() == Verdict::Passed;
    assert_eq!(reported, passed, "hw() {reported}, verdict {:?}", verdict());
    assert!(!passed || candidate().is_some());
}

#[test]
fn test_sme2_detect_first_use_passes_on_sme2_hardware() {
    let test = "test_sme2_detect_first_use_passes_on_sme2_hardware";
    if !hardware_or_skip(test, Need::F32) {
        return;
    }
    let Some(found) = candidate() else {
        panic!("{test}: the hardware gate ran without a candidate");
    };
    // SAFETY: the candidate is present.
    let state = unsafe { probe_thread_state() };
    assert!(
        state.ready(),
        "{test}: the test thread is not ready: {state:?}"
    );
    let reported = hw();
    assert_eq!(reported, Some(found), "{test}: hw() on a ready thread");
    assert_eq!(verdict(), Verdict::Passed, "{test}: the verdict");
    // A build that caps SIMD below SME2 keeps the backend off, whatever the
    // verdict.
    assert_eq!(enabled::<f32>(), SME2_ALLOWED, "{test}: f32");
    assert_eq!(enabled::<f64>(), found.f64 && SME2_ALLOWED, "{test}: f64");
    assert!(thread_ready(), "{test}: thread_ready");
    // `hw()` being `Some` implies "passed" at every later moment, on this
    // thread and on others.
    let others: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                (0..100).all(|_| hw().is_none() || verdict() == Verdict::Passed) && hw().is_some()
            })
        })
        .collect();
    for round in 0..100 {
        assert!(hw().is_some(), "{test}: round {round}");
        assert_eq!(verdict(), Verdict::Passed, "{test}: round {round}");
    }
    for other in others {
        let answer = other.join();
        assert!(matches!(answer, Ok(true)), "{test}: {answer:?}");
    }
}

#[test]
fn test_sme2_detect_first_use_check_is_quiet_and_costs_the_docs() {
    let test = "test_sme2_detect_first_use_check_is_quiet_and_costs_the_docs";
    if !hardware_or_skip(test, Need::F64) {
        return;
    }
    // A caller with round towards minus infinity and default NaN: with
    // those modes the GEMV check's negative-zero case would fail, so the
    // check passes only because it runs with FPCR = 0.
    const CALLER_MODES: u64 = (0b10 << 22) | (1 << 25);
    const CALLER_FLAGS: u64 = FPSR_DZC | FPSR_QC;
    let mut lines = Vec::new();
    for (f64_present, calls) in [
        (true, FIRST_USE_KERNEL_CALLS),
        (false, FIRST_USE_KERNEL_CALLS_F32),
    ] {
        let (fpcr, fpsr) = (fpcr_read(), fpsr_read());
        let previous_countdown = set_ready_countdown(Some(3));
        let (here, quiet, made, regions) = (
            kernel_entries_here(),
            quiet_kernel_entries(),
            decisions(),
            regions_on_this_thread(),
        );
        // SAFETY: the gate found a candidate, so the probe is implemented.
        let state = unsafe { probe_thread_state() };
        assert!(
            state.ready(),
            "{test}: the test thread is not ready: {state:?}"
        );
        // SAFETY: a test thread; restored below, before any assertion.
        unsafe { fpcr_write(CALLER_MODES) };
        fpsr_write(CALLER_FLAGS);
        let start = Instant::now();
        // SAFETY: the gate found SME2 with FEAT_SME_F64F64, and the test
        // thread was just found ready (the assertion above).
        let outcome = unsafe { run_checks(&REAL_CHECKS, f64_present) };
        let took = start.elapsed();
        let (fpcr_after, fpsr_after) = (fpcr_read(), fpsr_read());
        // SAFETY: the test thread's own values.
        unsafe { fpcr_write(fpcr) };
        fpsr_write(fpsr);
        let countdown = set_ready_countdown(previous_countdown);
        let counted = quiet_kernel_entries() - quiet;
        // Integer microseconds: a floating-point conversion here could be
        // scheduled before the FPSR read above and raise IXC there.
        lines.push(format!(
            "{test}: f64 {f64_present}: {counted} kernel calls in {} us",
            took.as_micros()
        ));
        assert_eq!(outcome, Ok(()), "{test}: f64 {f64_present}");
        assert_eq!(counted, calls, "{test}: f64 {f64_present}: kernel calls");
        assert_eq!(
            kernel_entries_here(),
            here,
            "{test}: kernel_entries_here moved"
        );
        assert_eq!(decisions(), made, "{test}: decisions moved");
        assert_eq!(
            regions_on_this_thread(),
            regions,
            "{test}: few-rows regions moved"
        );
        assert_eq!(countdown, Some(3), "{test}: the readiness countdown moved");
        assert_eq!(fpcr_after, CALLER_MODES, "{test}: FPCR afterwards");
        assert_eq!(fpsr_after, CALLER_FLAGS, "{test}: FPSR afterwards");
    }
    // The wall time, for the docs (run with --nocapture to see it).
    for line in lines {
        println!("{line}");
    }
}

// ---------------------------------------------------------------------------
// Interleavings, failure paths and bindings
// ---------------------------------------------------------------------------

/// A ready checker whose check fails while two ready threads wait for the
/// lock, a thread that is not ready arriving meanwhile and one arriving
/// afterwards: the waiters read "failed" without running their own checks,
/// the line is reported once in all, and every later call says `false`.
#[test]
fn test_sme2_detect_first_use_waiters_see_one_failure() {
    let first_use = Arc::new(FirstUse::new());
    let checks = Arc::new(AtomicUsize::new(0));
    let reports = Arc::new(StdMutex::new(Vec::<String>::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let report_into = |reports: &Arc<StdMutex<Vec<String>>>| {
        let reports = Arc::clone(reports);
        move |line: &str| {
            if let Ok(mut lines) = reports.lock() {
                lines.push(line.to_owned());
            }
        }
    };
    let checker = {
        let (first_use, checks, report) = (
            Arc::clone(&first_use),
            Arc::clone(&checks),
            report_into(&reports),
        );
        std::thread::spawn(move || {
            first_use.passed(
                ready,
                || {
                    checks.fetch_add(1, AtomicOrdering::SeqCst);
                    let _ = started_tx.send(());
                    let _ = release_rx.recv_timeout(PATIENCE);
                    Err(CheckFailure::FewRows)
                },
                report,
            )
        })
    };
    assert!(
        started_rx.recv_timeout(PATIENCE).is_ok(),
        "the check did not start"
    );
    let waiters: Vec<_> = (0..2)
        .map(|_| {
            let (first_use, checks, report) = (
                Arc::clone(&first_use),
                Arc::clone(&checks),
                report_into(&reports),
            );
            std::thread::spawn(move || {
                first_use.passed(
                    ready,
                    || {
                        checks.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok(())
                    },
                    report,
                )
            })
        })
        .collect();
    // A thread that is not ready while the lock is held: `false` at once
    // from the non-waiting form, its (passing) check not run, the verdict
    // still open.
    let during = {
        let first_use = Arc::clone(&first_use);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let answer = first_use.passed_now(not_ready, pass, |_| {});
            let _ = tx.send((answer, first_use.verdict() == Verdict::Open));
        });
        rx.recv_timeout(PATIENCE)
    };
    std::thread::sleep(Duration::from_millis(100));
    let _ = release_tx.send(());
    let checker_answer = checker.join();
    let waiter_answers: Vec<_> = waiters.into_iter().map(|w| w.join()).collect();
    let after = {
        let first_use = Arc::clone(&first_use);
        let report = report_into(&reports);
        std::thread::spawn(move || {
            (
                first_use.passed(not_ready, pass, &report),
                first_use.passed(ready, pass, &report),
            )
        })
        .join()
    };
    assert!(
        matches!(during, Ok((false, true))),
        "a thread that is not ready, while the check ran: {during:?}"
    );
    assert!(matches!(checker_answer, Ok(false)), "{checker_answer:?}");
    for answer in &waiter_answers {
        assert!(matches!(answer, Ok(false)), "a waiter read {answer:?}");
    }
    assert!(matches!(after, Ok((false, false))), "afterwards: {after:?}");
    let line = failure_line(&CheckFailure::FewRows);
    assert_eq!(first_use.verdict(), Verdict::Failed(line.as_str()));
    assert_eq!(checks.load(AtomicOrdering::SeqCst), 1, "checks that ran");
    let reported = reports
        .lock()
        .map(|lines| lines.clone())
        .unwrap_or_default();
    assert_eq!(reported, [line], "the line was not reported exactly once");
}

/// A check that panics while two ready threads wait for the lock: one of
/// them settles "failed" (the poisoned lock) without running its check, the
/// other reads it, the line is reported once, and a thread that is not
/// ready afterwards gets `false` too.
#[test]
fn test_sme2_detect_first_use_waiters_see_a_panic_as_failed() {
    let first_use = Arc::new(FirstUse::new());
    let checks = Arc::new(AtomicUsize::new(0));
    let reports = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let checker = {
        let first_use = Arc::clone(&first_use);
        std::thread::spawn(move || {
            first_use.passed(
                ready,
                || {
                    let _ = started_tx.send(());
                    let _ = release_rx.recv_timeout(PATIENCE);
                    std::panic::resume_unwind(Box::new("a check that panics"))
                },
                |_| {},
            )
        })
    };
    assert!(
        started_rx.recv_timeout(PATIENCE).is_ok(),
        "the check did not start"
    );
    let waiters: Vec<_> = (0..2)
        .map(|_| {
            let (first_use, checks, reports) = (
                Arc::clone(&first_use),
                Arc::clone(&checks),
                Arc::clone(&reports),
            );
            std::thread::spawn(move || {
                first_use.passed(
                    ready,
                    || {
                        checks.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok(())
                    },
                    |_| {
                        reports.fetch_add(1, AtomicOrdering::SeqCst);
                    },
                )
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(100));
    let _ = release_tx.send(());
    let checker_answer = checker.join();
    let waiter_answers: Vec<_> = waiters.into_iter().map(|w| w.join()).collect();
    assert!(checker_answer.is_err(), "the check did not unwind");
    for answer in &waiter_answers {
        assert!(matches!(answer, Ok(false)), "a waiter read {answer:?}");
    }
    assert!(!first_use.passed(not_ready, pass, |_| {}));
    let line = failure_line(&CheckFailure::Poisoned);
    assert_eq!(first_use.verdict(), Verdict::Failed(line.as_str()));
    assert_eq!(
        checks.load(AtomicOrdering::SeqCst),
        0,
        "a waiter ran its check"
    );
    assert_eq!(reports.load(AtomicOrdering::SeqCst), 1, "reports");
}

/// The "inside a check" mark ends with the check, also when it unwinds:
/// the same thread then runs the check of another verdict.
#[test]
fn test_sme2_detect_first_use_in_check_mark_ends_with_the_check() {
    let checks = Cell::new(0_u32);
    let check = || {
        checks.set(checks.get() + 1);
        Ok(())
    };
    assert!(FirstUse::new().passed(ready, check, |_| {}));
    assert!(
        FirstUse::new().passed(ready, check, |_| {}),
        "a second verdict on the thread that ran a check"
    );
    let failing = FirstUse::new();
    assert!(!failing.passed(ready, || Err(CheckFailure::Gemv), |_| {}));
    let panicking = FirstUse::new();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        panicking.passed(
            ready,
            || std::panic::resume_unwind(Box::new("a check that panics")),
            |_| {},
        )
    }));
    assert!(unwound.is_err(), "the check did not unwind");
    assert!(
        FirstUse::new().passed(ready, check, |_| {}),
        "a verdict after a check that failed and one that unwound"
    );
    assert_eq!(checks.get(), 3);
    assert!(!IN_CHECK.with(Cell::get), "the mark outlived the checks");
}

/// The macro-kernel stand-in of [`PANIC_STAND_INS`]: the regular stand-in.
///
/// # Safety
///
/// None needed (plain Rust).
unsafe fn panic_macro_stand_in(f64_present: bool) -> Result<(), Mismatch> {
    // SAFETY: plain Rust.
    unsafe { macro_stand_in(f64_present) }
}

/// A GEMV stand-in that records like the others and then unwinds.
///
/// # Safety
///
/// None needed (plain Rust).
unsafe fn panicking_gemv_stand_in(f64_present: bool) -> bool {
    stand_in("gemv", f64_present);
    std::panic::resume_unwind(Box::new("a GEMV check that panics"))
}

/// The stand-ins with a GEMV check that unwinds.
const PANIC_STAND_INS: Checks = Checks {
    macro_kernels: panic_macro_stand_in,
    za_packers: za_stand_in,
    few_rows: few_rows_stand_in,
    gemv: panicking_gemv_stand_in,
};

/// Every caller mode the architecture defines in FPCR (without FEAT_AFP),
/// one by one and together: the three other rounding modes, FZ, DN, AHP,
/// FZ16 and each trap enable. A bit the CPU ignores (trap enables are
/// optional) reads back as 0; the caller's state is what reads back.
const CALLER_FPCR: [u64; 13] = [
    0b01 << 22,
    0b10 << 22,
    0b11 << 22,
    1 << 24,
    1 << 25,
    1 << 26,
    1 << 19,
    1 << 8,
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 15,
];

/// For every caller FPCR mode, two caller FPSR values (none and every
/// cumulative flag) and every outcome of the check (passed, each check
/// failing, the GEMV check unwinding): FPCR is 0 inside every check, and
/// afterwards FPCR, FPSR and the counters are the caller's, bit for bit,
/// including the flags the checks raised (they never reach the caller).
#[test]
fn test_sme2_detect_first_use_floating_point_state_on_every_path() {
    let all_flags = FPSR_IOC | FPSR_DZC | FPSR_OFC | FPSR_UFC | FPSR_IXC | FPSR_IDC | FPSR_QC;
    let modes: Vec<u64> = CALLER_FPCR
        .iter()
        .copied()
        .chain([(0b11 << 22) | (1 << 24) | (1 << 25) | (1 << 26) | (1 << 19)])
        .collect();
    let (fpcr, fpsr) = (fpcr_read(), fpsr_read());
    let mut stuck = Vec::new();
    let mut problems = Vec::new();
    for &mode in &modes {
        for caller_fpsr in [0, all_flags] {
            for outcome in [
                "",
                "macro_kernels",
                "za_packers",
                "few_rows",
                "gemv",
                "panic",
            ] {
                SEEN.with(|seen| seen.borrow_mut().clear());
                let here = kernel_entries_here();
                // SAFETY: a test thread; restored below, before any
                // assertion or formatting.
                unsafe { fpcr_write(mode) };
                let caller_fpcr = fpcr_read();
                fpsr_write(caller_fpsr);
                let result = if outcome == "panic" {
                    std::panic::catch_unwind(|| {
                        // SAFETY: the stand-ins are plain Rust.
                        unsafe { run_checks(&PANIC_STAND_INS, true) }
                    })
                    .map_err(|_| ())
                } else {
                    Ok(run_stand_ins(outcome, true).0)
                };
                let (fpcr_after, fpsr_after) = (fpcr_read(), fpsr_read());
                // SAFETY: the test thread's own values.
                unsafe { fpcr_write(fpcr) };
                fpsr_write(fpsr);
                note_kernel_entry();
                let counted = kernel_entries_here() - here;
                let seen = SEEN.with(|seen| seen.borrow().clone());
                let label = format!(
                    "FPCR {mode:#x} (reads {caller_fpcr:#x}), FPSR {caller_fpsr:#x}, {outcome:?}"
                );
                if caller_fpcr != 0 && !stuck.contains(&caller_fpcr) {
                    stuck.push(caller_fpcr);
                }
                let ran_as_expected = match outcome {
                    "" => result == Ok(Ok(())),
                    "panic" => result.is_err(),
                    _ => matches!(result, Ok(Err(_))),
                };
                if !ran_as_expected {
                    problems.push(format!("{label}: outcome {result:?}"));
                }
                if seen.is_empty() || seen.iter().any(|&(inside, _)| inside != 0) {
                    problems.push(format!("{label}: FPCR inside the checks {seen:x?}"));
                }
                if fpcr_after != caller_fpcr {
                    problems.push(format!("{label}: FPCR afterwards {fpcr_after:#x}"));
                }
                if fpsr_after != caller_fpsr {
                    problems.push(format!("{label}: FPSR afterwards {fpsr_after:#x}"));
                }
                if counted != 1 {
                    problems.push(format!(
                        "{label}: one kernel entry after the check moved the counter by {counted}"
                    ));
                }
            }
        }
    }
    println!("caller FPCR values this CPU keeps: {stuck:#x?}");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    // The rounding modes, FZ and DN are not optional: the test saw them.
    for bit in [0b11 << 22, 1 << 24, 1 << 25] {
        assert!(stuck.contains(&bit), "FPCR {bit:#x} did not read back");
    }
}

/// While the check runs, the process-wide kernel-entry counter does not
/// move either: 100 000 kernel entries inside checks move it by far less
/// than that (other test threads may count their own meanwhile).
#[test]
fn test_sme2_detect_first_use_check_moves_no_process_counter() {
    const RUNS: u64 = 25_000;
    let (total, quiet) = (
        super::super::test_support::kernel_entries(),
        quiet_kernel_entries(),
    );
    for _ in 0..RUNS {
        let (outcome, _) = run_stand_ins("", false);
        assert_eq!(outcome, Ok(()));
    }
    let moved = super::super::test_support::kernel_entries() - total;
    assert_eq!(quiet_kernel_entries() - quiet, 4 * RUNS);
    assert!(
        moved < 4 * RUNS,
        "the process-wide counter moved by {moved} over {} quiet entries",
        4 * RUNS
    );
}

/// [`REAL_CHECKS`] binds each of the four real checks to its own slot. The
/// stand-in tests prove the order and the stop; this proves that the real
/// set holds the four checks, not one twice (the cost test cannot tell the
/// ZA transposers from the GEMV: 12 kernel calls per type each).
#[test]
fn test_sme2_detect_first_use_real_checks_bind_the_four_checks() {
    type Check = unsafe fn(bool) -> bool;
    let macro_kernels: unsafe fn(bool) -> Result<(), Mismatch> = selftest::run_self_test_detailed;
    let (za, few, gemv): (Check, Check, Check) = (
        pack_za::self_check,
        few_rows::self_check,
        gemv_sme2::self_check,
    );
    assert_eq!(
        REAL_CHECKS.macro_kernels as usize, macro_kernels as usize,
        "macro-kernels"
    );
    assert_eq!(
        REAL_CHECKS.za_packers as usize, za as usize,
        "ZA transposers"
    );
    assert_eq!(REAL_CHECKS.few_rows as usize, few as usize, "few-rows GEMM");
    assert_eq!(REAL_CHECKS.gemv as usize, gemv as usize, "GEMV");
    let mut addresses = [
        macro_kernels as usize,
        za as usize,
        few as usize,
        gemv as usize,
    ];
    addresses.sort_unstable();
    assert!(
        addresses.windows(2).all(|pair| pair[0] != pair[1]),
        "two checks share an address"
    );
}

/// The process's own first-use check runs with the candidate's `f64` flag:
/// when this test's thread is the one that settles the verdict, it counts
/// the check's kernel calls (quiet entries on this thread) and requires the
/// documented total for the candidate. When another thread settled it, the
/// test cannot see the check and says so (run it alone, with an exact
/// filter, to make it conclusive).
#[test]
fn test_sme2_detect_first_use_process_check_runs_with_the_candidate_flag() {
    let test = "test_sme2_detect_first_use_process_check_runs_with_the_candidate_flag";
    if !hardware_or_skip(test, Need::F32) {
        return;
    }
    let Some(found) = candidate() else {
        panic!("{test}: the hardware gate ran without a candidate");
    };
    // SAFETY: the candidate is present.
    let state = unsafe { probe_thread_state() };
    assert!(
        state.ready(),
        "{test}: the test thread is not ready: {state:?}"
    );
    let open_before = verdict() == Verdict::Open;
    let quiet = quiet_kernel_entries();
    let reported = hw();
    let counted = quiet_kernel_entries() - quiet;
    assert_eq!(reported, Some(found), "{test}: hw() on a ready thread");
    if !open_before || counted == 0 {
        println!(
            "{test}: inconclusive: another thread settled the verdict (open before: {open_before})"
        );
        return;
    }
    let want = if found.f64 {
        FIRST_USE_KERNEL_CALLS
    } else {
        FIRST_USE_KERNEL_CALLS_F32
    };
    println!("{test}: this thread ran the check: {counted} kernel calls");
    assert_eq!(
        counted, want,
        "{test}: kernel calls of the process's check (f64 {})",
        found.f64
    );
}

thread_local! {
    /// The bits that [`garbage_kernel`] writes.
    static GARBAGE: Cell<u64> = const { Cell::new(0) };
}

/// A stand-in macro-kernel that writes the bits of [`GARBAGE`] to every
/// element of the C block and nothing else.
///
/// # Safety
///
/// As for the macro-kernels: `C(i, j)` for `i < m`, `j < n` is writable.
unsafe fn garbage_kernel<T: SmeElement>(args: &MacroArgs<T>) {
    let value = T::from_bits64(GARBAGE.with(Cell::get));
    for j in 0..args.n {
        for i in 0..args.m {
            // SAFETY: `C(i, j)` lies in the block (the contract above).
            unsafe { *args.c.add(i + j * args.ldc) = value };
        }
    }
}

/// The macro-kernel self-test of one type over kernels that write garbage
/// (NaNs, infinities, -0, the largest and the smallest values): a failure
/// with a one-line report each time, never a panic.
fn garbage_fails<T: SmeElement>(patterns: &[u64]) {
    for &bits in patterns {
        GARBAGE.with(|garbage| garbage.set(bits));
        let kernel: KernelFn<T> = garbage_kernel::<T>;
        let outcome = std::panic::catch_unwind(|| {
            // SAFETY: the stand-in is plain Rust and writes only the block.
            unsafe { self_test_with_detailed(kernel) }
        });
        match outcome {
            Ok(Err(mismatch)) => {
                let line = failure_line(&CheckFailure::MacroKernels(mismatch));
                assert!(!line.contains(['\n', '\r']), "{bits:#x}: {line:?}");
            }
            Ok(Ok(())) => panic!("{} bits {bits:#x}: the garbage kernel passed", T::NR),
            Err(_) => panic!("{} bits {bits:#x}: the self-test panicked", T::NR),
        }
    }
}

#[test]
fn test_sme2_detect_first_use_garbage_macro_kernel_fails_without_panic() {
    garbage_fails::<f32>(&[
        0xffff_ffff,
        0x7fc0_0000,
        0x7f80_0000,
        0xff80_0000,
        0x8000_0000,
        0x7f7f_ffff,
        0x0000_0001,
    ]);
    garbage_fails::<f64>(&[
        u64::MAX,
        0x7ff8_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        1,
    ]);
}

/// What one call of `hw()` saw in a real state that is not ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NotReadyCall {
    /// `sme_state()` before the call.
    before: Option<(u64, u64)>,
    /// `sme_state()` after the call.
    after: Option<(u64, u64)>,
    /// `hw()` was `Some`.
    reported: bool,
    /// The verdict after the call.
    verdict_passed: bool,
    /// Quiet kernel entries on this thread during the call (the check's).
    check_calls: u64,
}

/// A thread whose real SME state is not ready (ZA storage on, TPIDR2_EL0
/// set) never runs the first-use check: its `hw()` runs no kernel on it
/// (no quiet entry on this thread, the state unchanged) and is `None` while
/// the verdict is open. The proof needs the verdict open when the test
/// starts (run it alone, with an exact filter); otherwise the readiness
/// rule is not consulted and the test says so. Afterwards, ready again, the
/// thread gets the hardware.
#[test]
fn test_sme2_detect_first_use_not_ready_thread_never_runs_the_check() {
    use super::super::test_sme_state::{Tpidr2Set, ZaStorageOn};
    use super::super::test_support::sme_state;
    let test = "test_sme2_detect_first_use_not_ready_thread_never_runs_the_check";
    if !hardware_or_skip(test, Need::F32) {
        return;
    }
    let open_before = verdict() == Verdict::Open;
    let mut calls = Vec::new();
    for state in 0..2 {
        // No thread is created and no ZA routine runs while a guard lives;
        // everything is asserted after it is dropped.
        let quiet = quiet_kernel_entries();
        let guards = if state == 0 {
            (ZaStorageOn::new(), None)
        } else {
            (None, Tpidr2Set::new())
        };
        let before = sme_state();
        let reported = hw().is_some();
        let verdict_passed = verdict() == Verdict::Passed;
        let after = sme_state();
        let check_calls = quiet_kernel_entries() - quiet;
        let made = guards.0.is_some() || guards.1.is_some();
        drop(guards);
        assert!(made, "{test}: state switch {state} not made");
        calls.push(NotReadyCall {
            before,
            after,
            reported,
            verdict_passed,
            check_calls,
        });
    }
    for (state, call) in calls.iter().enumerate() {
        assert_eq!(
            call.check_calls, 0,
            "{test}: state {state}: the check ran: {call:?}"
        );
        assert_eq!(call.before, call.after, "{test}: state {state}: {call:?}");
        assert!(
            call.before.is_some_and(|registers| registers != (0, 0)),
            "{test}: state {state}: the thread was ready: {call:?}"
        );
        assert!(
            !call.reported || call.verdict_passed,
            "{test}: state {state}: hw() Some without a passed verdict: {call:?}"
        );
    }
    assert_eq!(sme_state(), Some((0, 0)), "{test}: state restored");
    let reported = hw();
    assert_eq!(
        reported,
        candidate(),
        "{test}: hw() on the ready thread afterwards"
    );
    if !open_before {
        println!("{test}: inconclusive: the verdict was settled before this test");
    } else {
        println!("{test}: the verdict was open; calls in states that are not ready: {calls:?}");
    }
}
