//! Tests of the two forms ([`Form::Wait`] and [`Form::NoWait`]) of both
//! stages of the detection, on any host: the first-use verdict on local
//! [`FirstUse`] values with stand-in checks, and the candidate on local
//! [`CandidateCell`] values with stand-in computations (the process's
//! statics are not touched by these). Then the wiring of the process's
//! detection: source checks that the public BLAS paths reach only the
//! non-waiting form and that [`hw_in`] hands its form to both stages, and a
//! check that the process computes its candidate once.
//!
//! The stand-in check (or computation) of thread A parks until the test
//! releases it (a channel), so every interleaving below is fixed by the
//! test, not by the scheduler. A thread that should answer at once reports
//! through a channel that the test reads with a timeout of its own: a call
//! that waits shows up as a failed assertion, never as a hang of the test
//! binary (the parked thread is released when the test unwinds, because its
//! sender is dropped). Every assertion message names what was observed.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Barrier, Mutex as StdMutex};
use std::time::Duration;

use super::*;

/// How long a test waits for an answer that must come at once, or for a
/// thread that must finish, before it calls that a hang.
const PATIENCE: Duration = Duration::from_secs(20);

/// How long a waiting caller must stay blocked while the check is parked.
const BLOCKED_FOR: Duration = Duration::from_millis(200);

/// A readiness answer: ready.
fn ready() -> bool {
    true
}

/// A readiness answer: not ready.
fn not_ready() -> bool {
    false
}

/// What a test observed from a call on another thread: the answer, an
/// answer that did not come within the patience (`it waited`), or a thread
/// that ended without an answer.
fn observed<T: std::fmt::Debug>(result: &Result<T, RecvTimeoutError>) -> String {
    match result {
        Ok(seen) => format!("it answered {seen:?}"),
        Err(RecvTimeoutError::Timeout) => "no answer in time: it waited".to_owned(),
        Err(RecvTimeoutError::Disconnected) => {
            "its thread ended without an answer (it panicked)".to_owned()
        }
    }
}

/// What one call on another thread saw: its answer, whether the verdict was
/// still open right after it, and how many checks and reports it made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Seen {
    /// The answer of the call.
    answer: bool,
    /// The verdict was open right after the call returned.
    open_after: bool,
    /// Checks the call ran itself.
    checks: usize,
    /// Failure lines the call reported.
    reports: usize,
}

/// Thread A: runs the check (in the waiting form) and parks inside it until
/// `release` delivers or its sender is dropped. A delivered release ends the
/// check with `outcome`; a dropped sender (a test that unwinds) ends it with
/// a `Gemv` failure, so A always finishes. `started` is signalled once A is
/// inside the check; `reports` counts the failure lines A reports. The
/// handle gives A's answer.
fn spawn_parked_checker(
    first_use: &Arc<FirstUse>,
    outcome: Result<(), CheckFailure>,
    started: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    reports: Arc<AtomicUsize>,
) -> std::thread::JoinHandle<bool> {
    let first_use = Arc::clone(first_use);
    std::thread::spawn(move || {
        first_use.passed(
            ready,
            || {
                let _ = started.send(());
                match release.recv() {
                    Ok(()) => outcome,
                    Err(_) => Err(CheckFailure::Gemv),
                }
            },
            |_| {
                reports.fetch_add(1, AtomicOrdering::SeqCst);
            },
        )
    })
}

/// One call of `form` on a new thread, with a stand-in check that would
/// fail (so a check it runs by mistake also changes the verdict); the
/// thread sends what it saw.
fn spawn_call(first_use: &Arc<FirstUse>, form: Form, tx: mpsc::Sender<Seen>) {
    spawn_signalling_call(first_use, form, tx, None);
}

/// [`spawn_call`] for a call that the test wants to see blocked: this
/// function returns once the call's thread runs (it signals right before
/// the call), so a "still blocked" check that follows is not passed by a
/// thread that has not started yet.
fn spawn_started_call(first_use: &Arc<FirstUse>, form: Form, tx: mpsc::Sender<Seen>, label: &str) {
    let (started_tx, started_rx) = mpsc::channel();
    spawn_signalling_call(first_use, form, tx, Some(started_tx));
    let started = started_rx.recv_timeout(PATIENCE);
    assert!(
        started.is_ok(),
        "{label}: the call's thread did not start: {}",
        observed(&started)
    );
}

/// [`spawn_call`]; with `started`, the thread signals on it right before
/// the call.
fn spawn_signalling_call(
    first_use: &Arc<FirstUse>,
    form: Form,
    tx: mpsc::Sender<Seen>,
    started: Option<mpsc::Sender<()>>,
) {
    let first_use = Arc::clone(first_use);
    std::thread::spawn(move || {
        let checks = Cell::new(0);
        let reports = Cell::new(0);
        if let Some(started) = started {
            let _ = started.send(());
        }
        let answer = first_use.passed_in(
            form,
            ready,
            || {
                checks.set(checks.get() + 1);
                Err(CheckFailure::FewRows)
            },
            |_| reports.set(reports.get() + 1),
        );
        let _ = tx.send(Seen {
            answer,
            open_after: first_use.verdict() == Verdict::Open,
            checks: checks.get(),
            reports: reports.get(),
        });
    });
}

/// The scenario of the module docs for a check of thread A that ends with
/// `outcome`: B (non-waiting) answers at once while A is parked, C
/// (waiting) waits and gets A's answer, B afterwards gets A's answer too.
fn parked_scenario(outcome: Result<(), CheckFailure>) {
    let passes = outcome.is_ok();
    let first_use = Arc::new(FirstUse::new());
    let reports = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let checker = spawn_parked_checker(
        &first_use,
        outcome,
        started_tx,
        release_rx,
        Arc::clone(&reports),
    );
    let started = started_rx.recv_timeout(PATIENCE);
    assert!(
        started.is_ok(),
        "thread A did not start its check: {}",
        observed(&started)
    );

    // B, a ready thread in the non-waiting form, while A is parked inside
    // the check: `false` at once, the verdict still open, no check of its
    // own, nothing reported.
    let (b_tx, b_rx) = mpsc::channel();
    spawn_call(&first_use, Form::NoWait, b_tx);
    let b = b_rx.recv_timeout(PATIENCE);
    let want_b = Seen {
        answer: false,
        open_after: true,
        checks: 0,
        reports: 0,
    };
    assert!(
        b == Ok(want_b),
        "the non-waiting form while another thread checks must answer {want_b:?} at once: {}",
        observed(&b)
    );
    let verdict_now = first_use.verdict();
    assert_eq!(
        verdict_now,
        Verdict::Open,
        "after B the verdict is {verdict_now:?}, not open: B touched it"
    );

    // C, a ready thread in the waiting form: it blocks while A is parked.
    let (c_tx, c_rx) = mpsc::channel();
    spawn_started_call(&first_use, Form::Wait, c_tx, "thread C");
    let early = c_rx.recv_timeout(BLOCKED_FOR);
    assert!(
        early.is_err(),
        "the waiting form must wait while the check is running: {}",
        observed(&early)
    );

    // Release A; C then returns with A's answer, without a check of its own.
    let _ = release_tx.send(());
    let a = checker.join();
    assert!(
        matches!(a, Ok(answer) if answer == passes),
        "thread A must answer {passes}: it gave {a:?}"
    );
    let c = c_rx.recv_timeout(PATIENCE);
    let want_after = Seen {
        answer: passes,
        open_after: false,
        checks: 0,
        reports: 0,
    };
    assert!(
        c == Ok(want_after),
        "the waiting form after the check must answer {want_after:?}: {}",
        observed(&c)
    );

    // B again, now that the verdict is settled: A's answer.
    let (b_tx, b_rx) = mpsc::channel();
    spawn_call(&first_use, Form::NoWait, b_tx);
    let b = b_rx.recv_timeout(PATIENCE);
    assert!(
        b == Ok(want_after),
        "the non-waiting form after the check must answer {want_after:?}: {}",
        observed(&b)
    );
    let settled = first_use.verdict();
    match outcome {
        Ok(()) => assert_eq!(
            settled,
            Verdict::Passed,
            "a passing check settled {settled:?}"
        ),
        Err(failure) => {
            let line = failure_line(&failure);
            assert_eq!(
                settled,
                Verdict::Failed(line.as_str()),
                "a failing check ({failure:?}) settled {settled:?}"
            );
        }
    }
    let reported = reports.load(AtomicOrdering::SeqCst);
    assert_eq!(
        reported,
        usize::from(!passes),
        "the failure line was reported {reported} times"
    );
}

#[test]
fn test_sme2_detect_forms_no_wait_answers_while_a_check_passes() {
    parked_scenario(Ok(()));
}

#[test]
fn test_sme2_detect_forms_no_wait_answers_while_a_check_fails() {
    parked_scenario(Err(CheckFailure::ZaPackers));
}

/// Without a concurrent check the non-waiting form behaves as the waiting
/// one: a thread that is not ready gets `false` and leaves the verdict
/// open; a ready thread that finds the verdict open and the lock free runs
/// the check itself, once; a settled verdict is final.
#[test]
fn test_sme2_detect_forms_no_wait_runs_the_check_when_nobody_else_does() {
    for passes in [true, false] {
        let first_use = FirstUse::new();
        let checks = Cell::new(0_u32);
        let reports = RefCell::new(Vec::new());
        let check = || {
            checks.set(checks.get() + 1);
            if passes {
                Ok(())
            } else {
                Err(CheckFailure::FewRows)
            }
        };
        let report = |line: &str| reports.borrow_mut().push(line.to_owned());
        let answer = first_use.passed_now(not_ready, check, report);
        assert!(
            !answer,
            "passes {passes}: a thread that is not ready answered {answer}"
        );
        let verdict_now = first_use.verdict();
        assert_eq!(
            verdict_now,
            Verdict::Open,
            "passes {passes}: a thread that is not ready settled {verdict_now:?}"
        );
        assert_eq!(
            checks.get(),
            0,
            "passes {passes}: a thread that is not ready ran {} checks",
            checks.get()
        );
        let answer = first_use.passed_now(ready, check, report);
        assert_eq!(
            answer, passes,
            "passes {passes}: the ready thread's own check answered {answer}"
        );
        assert_eq!(
            checks.get(),
            1,
            "passes {passes}: the ready thread ran {} checks",
            checks.get()
        );
        for round in 0..3 {
            let now = first_use.passed_now(ready, check, report);
            assert_eq!(
                now, passes,
                "passes {passes}, round {round}: the non-waiting form answered {now}"
            );
            let waited = first_use.passed(ready, check, report);
            assert_eq!(
                waited, passes,
                "passes {passes}, round {round}: the waiting form answered {waited}"
            );
        }
        assert_eq!(
            checks.get(),
            1,
            "passes {passes}: the check ran {} times in all",
            checks.get()
        );
        let settled = first_use.verdict();
        if passes {
            assert_eq!(
                settled,
                Verdict::Passed,
                "a passing check settled {settled:?}"
            );
            assert!(
                reports.borrow().is_empty(),
                "a passing check reported {:?}",
                reports.borrow()
            );
        } else {
            let line = failure_line(&CheckFailure::FewRows);
            assert_eq!(
                settled,
                Verdict::Failed(line.as_str()),
                "a failing check settled {settled:?}"
            );
            assert_eq!(
                *reports.borrow(),
                [line],
                "a failing check reported {:?}",
                reports.borrow()
            );
        }
    }
}

/// A poisoned lock (a check that panicked) settles a failure without
/// running a check in both forms, whichever comes first, and the line is
/// reported once.
#[test]
fn test_sme2_detect_forms_poisoned_lock_fails_in_both_forms() {
    for first in [Form::NoWait, Form::Wait] {
        let second = match first {
            Form::NoWait => Form::Wait,
            Form::Wait => Form::NoWait,
        };
        let first_use = Arc::new(FirstUse::new());
        let panicked = {
            let first_use = Arc::clone(&first_use);
            std::thread::spawn(move || {
                first_use.passed_in(
                    first,
                    ready,
                    // `resume_unwind` skips the panic hook (no message in the
                    // log).
                    || std::panic::resume_unwind(Box::new("a check that panics")),
                    |_| {},
                )
            })
            .join()
        };
        assert!(
            panicked.is_err(),
            "the check must unwind ({first:?}): the thread returned {panicked:?}"
        );
        let verdict_now = first_use.verdict();
        assert_eq!(
            verdict_now,
            Verdict::Open,
            "{first:?}: the panic itself settled {verdict_now:?}"
        );
        let checks = Cell::new(0_u32);
        let reports = RefCell::new(Vec::new());
        let check = || {
            checks.set(checks.get() + 1);
            Ok(())
        };
        let report = |line: &str| reports.borrow_mut().push(line.to_owned());
        let answer = first_use.passed_in(first, ready, check, report);
        assert!(!answer, "{first:?} under a poisoned lock answered {answer}");
        let answer = first_use.passed_in(second, ready, check, report);
        assert!(
            !answer,
            "{second:?} after a poisoned lock answered {answer}"
        );
        let line = failure_line(&CheckFailure::Poisoned);
        let settled = first_use.verdict();
        assert_eq!(
            settled,
            Verdict::Failed(line.as_str()),
            "{first:?} first: a poisoned lock settled {settled:?}"
        );
        assert_eq!(
            checks.get(),
            0,
            "{first:?} first: {} checks ran under a poisoned lock",
            checks.get()
        );
        assert_eq!(
            *reports.borrow(),
            [line],
            "{first:?} first: the reports were {:?}",
            reports.borrow()
        );
    }
}

/// `true` from the non-waiting form always means "passed": whatever another
/// thread does with the lock, an answer of `true` comes with a settled
/// verdict. This is why no part of an SME2 call can be the thread that runs
/// the check (the decision asks `enabled` first, on the calling thread).
#[test]
fn test_sme2_detect_forms_no_wait_true_only_when_settled() {
    let first_use = Arc::new(FirstUse::new());
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let checker = spawn_parked_checker(
        &first_use,
        Ok(()),
        started_tx,
        release_rx,
        Arc::new(AtomicUsize::new(0)),
    );
    let started = started_rx.recv_timeout(PATIENCE);
    assert!(
        started.is_ok(),
        "thread A did not start its check: {}",
        observed(&started)
    );
    // Many non-waiting callers around the release of the check.
    let callers: Vec<_> = (0..8)
        .map(|_| {
            let (first_use, seen) = (Arc::clone(&first_use), Arc::clone(&seen));
            std::thread::spawn(move || {
                for _ in 0..2_000 {
                    let answer = first_use.passed_now(ready, || Err(CheckFailure::Gemv), |_| {});
                    let settled = first_use.verdict() == Verdict::Passed;
                    if answer
                        && !settled
                        && let Ok(mut seen) = seen.lock()
                    {
                        seen.push("true with an unsettled verdict");
                    }
                    std::thread::yield_now();
                }
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(20));
    let _ = release_tx.send(());
    let a = checker.join();
    for (index, caller) in callers.into_iter().enumerate() {
        let joined = caller.join();
        assert!(joined.is_ok(), "caller {index} panicked: {joined:?}");
    }
    assert!(
        matches!(a, Ok(true)),
        "thread A must pass its check: it gave {a:?}"
    );
    let seen = seen.lock().map(|seen| seen.clone()).unwrap_or_default();
    assert!(
        seen.is_empty(),
        "the non-waiting callers saw {} times: {seen:?}",
        seen.len()
    );
    let settled = first_use.verdict();
    assert_eq!(
        settled,
        Verdict::Passed,
        "the verdict is {settled:?}: a caller ran its failing check"
    );
}

/// A non-waiting call that finds the lock taken answers with the verdict as
/// it reads it then: `true` when it was settled to "passed" meanwhile (after
/// the call's first read of the verdict), `false` when it was settled to
/// "failed" or is still open. The call runs no check, reports nothing and
/// changes no state. The test holds the lock itself, and the verdict changes
/// inside the call's readiness closure, which runs between the call's first
/// read and its attempt on the lock, so the interleaving is fixed.
#[test]
fn test_sme2_detect_forms_no_wait_reads_a_verdict_settled_meanwhile() {
    for (settle_to, expected) in [(Some(PASSED), true), (Some(FAILED), false), (None, false)] {
        let first_use = FirstUse::new();
        let held = first_use.lock.lock();
        assert!(
            held.is_ok(),
            "{settle_to:?}: a fresh lock is poisoned: {held:?}"
        );
        let (checks, reports) = (Cell::new(0_usize), Cell::new(0_usize));
        let answer = first_use.passed_now(
            || {
                if let Some(state) = settle_to {
                    first_use.state.store(state, AtomicOrdering::Release);
                }
                true
            },
            || {
                checks.set(checks.get() + 1);
                Ok(())
            },
            |_| reports.set(reports.get() + 1),
        );
        assert_eq!(
            answer, expected,
            "{settle_to:?}: the call answered {answer}"
        );
        let state = first_use.state.load(AtomicOrdering::Acquire);
        assert_eq!(
            state,
            settle_to.unwrap_or(OPEN),
            "{settle_to:?}: the call left the state {state}"
        );
        assert_eq!(
            (checks.get(), reports.get()),
            (0, 0),
            "{settle_to:?}: the call ran {} checks and made {} reports",
            checks.get(),
            reports.get()
        );
        drop(held);
    }
}

/// The body of `fn <name>` in `source` (from its signature to the matching
/// closing brace), or `None`.
fn fn_body<'s>(source: &'s str, name: &str) -> Option<&'s str> {
    let start = source.find(&format!("fn {name}"))?;
    let open = start + source[start..].find('{')?;
    let mut depth = 0_usize;
    for (offset, ch) in source[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[start..open + offset + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `body` calls the waiting form `hw()` (and not only `hw_now()`).
fn calls_waiting_form(body: &str) -> bool {
    body.match_indices("hw()").any(|(index, _)| {
        let before = &body[..index];
        !before.ends_with('_') && !before.ends_with(|c: char| c.is_alphanumeric())
    })
}

/// The detection questions that the public BLAS paths ask (`enabled`,
/// `thread_ready`, `units`) use the non-waiting form, `sme2_active` uses
/// the waiting one, and no file of the GEMM or GEMV paths calls the waiting
/// form directly.
#[test]
fn test_sme2_detect_forms_blas_paths_use_the_non_waiting_form() {
    let detect = include_str!("detect.rs");
    for name in ["enabled<", "thread_ready(", "units("] {
        let body = fn_body(detect, name).unwrap_or_default();
        assert!(!body.is_empty(), "fn {name} not found in detect.rs");
        assert!(
            body.contains("hw_now()"),
            "fn {name} does not ask hw_now(): {body}"
        );
        assert!(
            !calls_waiting_form(body),
            "fn {name} calls the waiting hw(): {body}"
        );
        assert!(
            !body.contains("candidate()"),
            "fn {name} asks the waiting candidate(): {body}"
        );
    }
    let module = include_str!("mod.rs");
    let active = fn_body(module, "sme2_active(").unwrap_or_default();
    assert!(
        calls_waiting_form(active),
        "sme2_active does not settle the verdict with the waiting hw(): {active}"
    );
    for (file, text) in [
        ("driver.rs", include_str!("driver.rs")),
        ("parallel.rs", include_str!("parallel.rs")),
        ("gemv/sme2.rs", include_str!("../../level2/gemv/sme2.rs")),
        ("gemm.rs", include_str!("../gemm.rs")),
        ("gemm_thin.rs", include_str!("../gemm_thin.rs")),
        ("gemv/mod.rs", include_str!("../../level2/gemv/mod.rs")),
    ] {
        // Code lines only: the docs may name the detection.
        let direct = text
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .find(|line| line.contains("detect::hw(") || line.contains("detect::hw_now("));
        assert!(
            direct.is_none(),
            "{file} asks the detection directly instead of enabled / thread_ready / units: \
             {direct:?}"
        );
        let stage1 = text
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .find(|line| line.contains("detect::candidate(") || line.contains("CANDIDATE."));
        assert!(
            stage1.is_none(),
            "{file} asks stage 1 of the detection directly: {stage1:?}"
        );
    }
}

/// The source check above recognises a call of the waiting form.
#[test]
fn test_sme2_detect_forms_source_check_sees_the_waiting_form() {
    for text in ["{ let x = hw(); }", "{ detect::hw().is_some() }"] {
        assert!(
            calls_waiting_form(text),
            "the waiting form in {text:?} was not recognised"
        );
    }
    for text in ["{ hw_now() }", "{ my_hw() }"] {
        assert!(
            !calls_waiting_form(text),
            "{text:?} was taken for the waiting form"
        );
    }
    let body = fn_body("fn a() { if x { b() } } fn c() {}", "a(");
    assert_eq!(
        body,
        Some("fn a() { if x { b() } }"),
        "the body of fn a was cut as {body:?}"
    );
}

// ---------------------------------------------------------------------------
// Stage 1: the candidate
// ---------------------------------------------------------------------------

/// The candidate that thread A's stand-in computation returns.
const STAND_IN: Sme2Hw = Sme2Hw {
    f64: true,
    units: 3,
};

/// The candidate that every other stand-in computation returns: a call that
/// computes by mistake shows up in the answers as well as in the counts.
const INTRUDER: Sme2Hw = Sme2Hw {
    f64: false,
    units: 7,
};

/// What one stage-1 call on another thread saw: its answer, whether the
/// cell's candidate was known right after it, and how many computations the
/// call ran itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeenCandidate {
    /// The answer of the call.
    answer: Option<Sme2Hw>,
    /// The candidate was known right after the call returned.
    known_after: bool,
    /// Computations the call ran itself.
    computed: usize,
}

/// How thread A's parked computation ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    /// It returns this candidate.
    Returns(Option<Sme2Hw>),
    /// It panics.
    Panics,
}

/// Thread A: computes the candidate of `cell` in the form `form` and parks
/// inside the computation until `release` delivers or its sender is
/// dropped; then it ends as `ending` says (a dropped sender, from a test
/// that unwinds, ends it with "no candidate"). `started` is signalled once A
/// is inside the computation; `computed` counts A's computations. The
/// handle gives A's answer (an `Err` when it panicked).
fn spawn_parked_computer(
    cell: &Arc<CandidateCell>,
    form: Form,
    ending: Ending,
    started: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    computed: Arc<AtomicUsize>,
) -> std::thread::JoinHandle<Option<Sme2Hw>> {
    let cell = Arc::clone(cell);
    std::thread::spawn(move || {
        cell.get_in(form, || {
            computed.fetch_add(1, AtomicOrdering::SeqCst);
            let _ = started.send(());
            let released = release.recv();
            match (released, ending) {
                (Ok(()), Ending::Returns(value)) => value,
                // `resume_unwind` skips the panic hook (no message in the
                // log).
                (Ok(()), Ending::Panics) => {
                    std::panic::resume_unwind(Box::new("a computation that panics"))
                }
                (Err(_), _) => None,
            }
        })
        .copied()
    })
}

/// One stage-1 call of `form` on a new thread, with a stand-in computation
/// that returns [`INTRUDER`]; the thread sends what it saw.
fn spawn_candidate_call(cell: &Arc<CandidateCell>, form: Form, tx: mpsc::Sender<SeenCandidate>) {
    spawn_signalling_candidate_call(cell, form, tx, None);
}

/// [`spawn_candidate_call`] for a call that the test wants to see blocked:
/// this function returns once the call's thread runs (it signals right
/// before the call), so a "still blocked" check that follows is not passed
/// by a thread that has not started yet.
fn spawn_started_candidate_call(
    cell: &Arc<CandidateCell>,
    form: Form,
    tx: mpsc::Sender<SeenCandidate>,
    label: &str,
) {
    let (started_tx, started_rx) = mpsc::channel();
    spawn_signalling_candidate_call(cell, form, tx, Some(started_tx));
    let started = started_rx.recv_timeout(PATIENCE);
    assert!(
        started.is_ok(),
        "{label}: the call's thread did not start: {}",
        observed(&started)
    );
}

/// [`spawn_candidate_call`]; with `started`, the thread signals on it
/// right before the call.
fn spawn_signalling_candidate_call(
    cell: &Arc<CandidateCell>,
    form: Form,
    tx: mpsc::Sender<SeenCandidate>,
    started: Option<mpsc::Sender<()>>,
) {
    let cell = Arc::clone(cell);
    std::thread::spawn(move || {
        let computed = Cell::new(0);
        if let Some(started) = started {
            let _ = started.send(());
        }
        let answer = cell
            .get_in(form, || {
                computed.set(computed.get() + 1);
                Some(INTRUDER)
            })
            .copied();
        let _ = tx.send(SeenCandidate {
            answer,
            known_after: cell.known().is_some(),
            computed: computed.get(),
        });
    });
}

/// One stage-1 call of `form` on a new thread, read with the patience.
fn candidate_call(
    cell: &Arc<CandidateCell>,
    form: Form,
) -> Result<SeenCandidate, RecvTimeoutError> {
    let (tx, rx) = mpsc::channel();
    spawn_candidate_call(cell, form, tx);
    rx.recv_timeout(PATIENCE)
}

/// The scenario of stage 1 for thread A computing in the form `a_form` a
/// computation that returns `value`: while A computes, B (non-waiting)
/// answers `None` at once and stores nothing, and C (waiting) waits; A's
/// computation is not disturbed; afterwards C, and B again, get A's value,
/// and nobody else computed.
fn candidate_parked_scenario(a_form: Form, value: Option<Sme2Hw>) {
    let label = format!("A in {a_form:?} computing {value:?}");
    let cell = Arc::new(CandidateCell::new());
    let computed = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let computer = spawn_parked_computer(
        &cell,
        a_form,
        Ending::Returns(value),
        started_tx,
        release_rx,
        Arc::clone(&computed),
    );
    let started = started_rx.recv_timeout(PATIENCE);
    assert!(
        started.is_ok(),
        "{label}: thread A did not start its computation: {}",
        observed(&started)
    );

    // B, non-waiting, while A computes: `None` at once, nothing stored,
    // nothing computed.
    let b = candidate_call(&cell, Form::NoWait);
    let want_b = SeenCandidate {
        answer: None,
        known_after: false,
        computed: 0,
    };
    assert!(
        b == Ok(want_b),
        "{label}: the non-waiting form while another thread computes must answer \
         {want_b:?} at once: {}",
        observed(&b)
    );
    let known = cell.known().map(|known| known.copied());
    assert_eq!(
        known, None,
        "{label}: after B the candidate is known as {known:?}: B stored it"
    );

    // C, waiting: blocked while A computes.
    let (c_tx, c_rx) = mpsc::channel();
    spawn_started_candidate_call(&cell, Form::Wait, c_tx, &label);
    let early = c_rx.recv_timeout(BLOCKED_FOR);
    assert!(
        early.is_err(),
        "{label}: the waiting form must wait while another thread computes: {}",
        observed(&early)
    );

    // Release A: A returns its own value, C then gets it.
    let _ = release_tx.send(());
    let a = computer.join();
    assert!(
        matches!(a, Ok(answer) if answer == value),
        "{label}: thread A must return {value:?}: it gave {a:?}"
    );
    let c = c_rx.recv_timeout(PATIENCE);
    let want_after = SeenCandidate {
        answer: value,
        known_after: true,
        computed: 0,
    };
    assert!(
        c == Ok(want_after),
        "{label}: the waiting form after the computation must answer {want_after:?}: {}",
        observed(&c)
    );

    // B again, now that the candidate is known, in both forms: A's value.
    for form in [Form::NoWait, Form::Wait] {
        let again = candidate_call(&cell, form);
        assert!(
            again == Ok(want_after),
            "{label}: {form:?} after the computation must answer {want_after:?}: {}",
            observed(&again)
        );
    }
    let runs = computed.load(AtomicOrdering::SeqCst);
    assert_eq!(runs, 1, "{label}: thread A computed {runs} times");
    let known = cell.known().map(|known| known.copied());
    assert_eq!(
        known,
        Some(value),
        "{label}: the cell keeps {known:?}, not A's value"
    );
}

/// While thread A computes the candidate, a non-waiting call returns `None`
/// at once (without computing, storing or disturbing A), a waiting call
/// waits and gets A's value, and afterwards every call gets A's value: for
/// A in either form, and for a present and an absent candidate.
#[test]
fn test_sme2_detect_forms_candidate_no_wait_answers_while_another_thread_computes() {
    for a_form in [Form::Wait, Form::NoWait] {
        for value in [Some(STAND_IN), None] {
            candidate_parked_scenario(a_form, value);
        }
    }
}

/// Without a concurrent computation both forms compute the candidate
/// themselves, once; a known candidate is final and read without computing,
/// whether it is present or absent.
#[test]
fn test_sme2_detect_forms_candidate_computed_once_when_nobody_else_does() {
    for first in [Form::NoWait, Form::Wait] {
        for value in [Some(STAND_IN), None] {
            let cell = CandidateCell::new();
            let computed = Cell::new(0_usize);
            let compute = || {
                computed.set(computed.get() + 1);
                value
            };
            let known = cell.known().map(|known| known.copied());
            assert_eq!(known, None, "a new cell knows {known:?}");
            let answer = cell.get_in(first, compute).copied();
            assert_eq!(
                answer, value,
                "{first:?} on an unknown candidate {value:?} answered {answer:?}"
            );
            for round in 0..3 {
                for form in [Form::NoWait, Form::Wait] {
                    let answer = cell.get_in(form, || Some(INTRUDER)).copied();
                    assert_eq!(
                        answer, value,
                        "{first:?} first, {value:?}, round {round}: {form:?} answered {answer:?}"
                    );
                }
            }
            assert_eq!(
                computed.get(),
                1,
                "{first:?} first, {value:?}: computed {} times",
                computed.get()
            );
        }
    }
}

/// Many threads ask for an unknown candidate at once, half of them in each
/// form, with a computation that takes a while: it is computed exactly once,
/// every waiting call gets its value, a non-waiting call gets its value or
/// `None` (never anything else), and afterwards every call gets the value.
#[test]
fn test_sme2_detect_forms_candidate_computed_once_under_contention() {
    const THREADS: usize = 16;
    let mut contended = 0_usize;
    for round in 0..20 {
        let cell = Arc::new(CandidateCell::new());
        let computed = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(THREADS));
        let callers: Vec<_> = (0..THREADS)
            .map(|index| {
                let (cell, computed, start) =
                    (Arc::clone(&cell), Arc::clone(&computed), Arc::clone(&start));
                let form = if index % 2 == 0 {
                    Form::NoWait
                } else {
                    Form::Wait
                };
                std::thread::spawn(move || {
                    start.wait();
                    let answer = cell
                        .get_in(form, || {
                            computed.fetch_add(1, AtomicOrdering::SeqCst);
                            std::thread::sleep(Duration::from_millis(2));
                            Some(STAND_IN)
                        })
                        .copied();
                    (form, answer)
                })
            })
            .collect();
        for (index, caller) in callers.into_iter().enumerate() {
            let joined = caller.join();
            match joined {
                Ok((Form::Wait, answer)) => assert_eq!(
                    answer,
                    Some(STAND_IN),
                    "round {round}: waiting thread {index} answered {answer:?}"
                ),
                Ok((Form::NoWait, None)) => contended += 1,
                Ok((Form::NoWait, answer)) => assert_eq!(
                    answer,
                    Some(STAND_IN),
                    "round {round}: non-waiting thread {index} answered {answer:?}"
                ),
                Err(_) => panic!("round {round}: thread {index} panicked: {joined:?}"),
            }
        }
        let runs = computed.load(AtomicOrdering::SeqCst);
        assert_eq!(
            runs, 1,
            "round {round}: the candidate was computed {runs} times"
        );
        for form in [Form::NoWait, Form::Wait] {
            let answer = cell.get_in(form, || Some(INTRUDER)).copied();
            assert_eq!(
                answer,
                Some(STAND_IN),
                "round {round}: {form:?} afterwards answered {answer:?}"
            );
        }
    }
    println!("non-waiting calls that found the computation in progress: {contended}");
}

/// A computation that panics: the panic leaves the computing call, and the
/// candidate is then settled to "no candidate" for good by the next call of
/// either form (the lock is poisoned), without computing again. A waiting
/// call that was blocked meanwhile returns `None`, a non-waiting call during
/// the computation returned `None` at once, and no later call waits.
#[test]
fn test_sme2_detect_forms_candidate_panic_settles_no_candidate() {
    // With a waiting call blocked while A computes, and a non-waiting call
    // during the computation.
    for a_form in [Form::Wait, Form::NoWait] {
        let label = format!("A in {a_form:?}, C waiting");
        let cell = Arc::new(CandidateCell::new());
        let computed = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let computer = spawn_parked_computer(
            &cell,
            a_form,
            Ending::Panics,
            started_tx,
            release_rx,
            Arc::clone(&computed),
        );
        let started = started_rx.recv_timeout(PATIENCE);
        assert!(
            started.is_ok(),
            "{label}: thread A did not start its computation: {}",
            observed(&started)
        );
        let b = candidate_call(&cell, Form::NoWait);
        let want_none = SeenCandidate {
            answer: None,
            known_after: false,
            computed: 0,
        };
        assert!(
            b == Ok(want_none),
            "{label}: the non-waiting form during the computation must answer {want_none:?} \
             at once: {}",
            observed(&b)
        );
        let (c_tx, c_rx) = mpsc::channel();
        spawn_started_candidate_call(&cell, Form::Wait, c_tx, &label);
        let early = c_rx.recv_timeout(BLOCKED_FOR);
        assert!(
            early.is_err(),
            "{label}: the waiting form must wait while another thread computes: {}",
            observed(&early)
        );
        let _ = release_tx.send(());
        let a = computer.join();
        assert!(
            a.is_err(),
            "{label}: thread A's computation must unwind: it returned {a:?}"
        );
        let c = c_rx.recv_timeout(PATIENCE);
        let want_settled = SeenCandidate {
            answer: None,
            known_after: true,
            computed: 0,
        };
        assert!(
            c == Ok(want_settled),
            "{label}: the blocked waiting call after the panic must answer {want_settled:?}: \
             {}",
            observed(&c)
        );
        for round in 0..3 {
            for form in [Form::NoWait, Form::Wait] {
                let later = candidate_call(&cell, form);
                assert!(
                    later == Ok(want_settled),
                    "{label}, round {round}: {form:?} after the panic must answer \
                     {want_settled:?}: {}",
                    observed(&later)
                );
            }
        }
        let runs = computed.load(AtomicOrdering::SeqCst);
        assert_eq!(runs, 1, "{label}: thread A computed {runs} times");
    }
    // Nobody waiting: the first call after the panic, in either form, finds
    // the lock poisoned and settles "no candidate".
    for first in [Form::NoWait, Form::Wait] {
        let label = format!("{first:?} first after the panic");
        let cell = Arc::new(CandidateCell::new());
        let panicked = {
            let cell = Arc::clone(&cell);
            std::thread::spawn(move || {
                cell.get_in(first, || {
                    std::panic::resume_unwind(Box::new("a computation that panics"))
                })
                .copied()
            })
            .join()
        };
        assert!(
            panicked.is_err(),
            "{label}: the computation must unwind: it returned {panicked:?}"
        );
        let known = cell.known().map(|known| known.copied());
        assert_eq!(known, None, "{label}: the panic itself stored {known:?}");
        let want_settled = SeenCandidate {
            answer: None,
            known_after: true,
            computed: 0,
        };
        for form in [first, Form::NoWait, Form::Wait] {
            let later = candidate_call(&cell, form);
            assert!(
                later == Ok(want_settled),
                "{label}: {form:?} must answer {want_settled:?}: {}",
                observed(&later)
            );
        }
    }
}

/// [`hw_in`] hands its own form to both stages: stage 1 is the cell of
/// `candidate_cell` read with `form`, never the waiting `candidate()` and
/// never a direct computation, and stage 2 is the verdict of
/// `first_use_cell` read with `form`; outside the test seam those are the
/// process's [`CANDIDATE`] and [`FIRST_USE`]. Read from the source
/// (comments removed, whitespace ignored).
///
/// The dynamic tests of `tests_detect_wait.rs` drive the real `hw` and
/// `hw_now` through the seam (`StagesHere`): stage 1 on any host, stage 2
/// only where the process has SME2 hardware, because the real readiness
/// closure of stage 2 executes the SME register probe (SIGILL on a CPU
/// without SME). On other hosts this source check is the only test of the
/// stage-2 wiring, and it also pins that the seam ends in the process's
/// statics.
#[test]
fn test_sme2_detect_forms_hw_in_passes_its_form_to_both_stages() {
    let lexed = super::super::lint_scan::lex(include_str!("detect.rs"));
    let body = fn_body(&lexed.code, "hw_in(").unwrap_or_default();
    assert!(!body.is_empty(), "fn hw_in not found in detect.rs");
    let compact: String = body.split_whitespace().collect();
    for (name, process) in [
        ("candidate_cell(", "&CANDIDATE}"),
        ("first_use_cell(", "&FIRST_USE}"),
    ] {
        let cell = fn_body(&lexed.code, name).unwrap_or_default();
        let cell_compact: String = cell.split_whitespace().collect();
        assert!(
            cell_compact.ends_with(process),
            "fn {name} does not end with the process's {process}: {cell}"
        );
    }
    for wanted in [
        "candidate_cell().get_in(form,compute_candidate)?",
        "first_use_cell().passed_in(form,",
    ] {
        assert!(
            compact.contains(wanted),
            "fn hw_in does not contain {wanted:?}: {body}"
        );
    }
    for unwanted in ["candidate()", "detect_candidate", "Form::"] {
        assert!(
            !compact.contains(unwanted),
            "fn hw_in contains {unwanted:?}: {body}"
        );
    }
}

/// The process computes its candidate once: after detection calls of both
/// forms (and of the questions of the BLAS paths) from several threads, the
/// count of computations is exactly 1, whatever other tests did meanwhile.
/// The test runs on every host: on SME2 hardware these calls run the
/// first-use check (as the backend gate does), without the hardware every
/// call answers `None` without executing SME code, and the count is
/// checked all the same. It therefore writes no gate marker (it is neither
/// skipped nor declined anywhere) and only prints whether the hardware is
/// there.
#[test]
fn test_sme2_detect_forms_process_candidate_computed_once() {
    let test = "test_sme2_detect_forms_process_candidate_computed_once";
    let hardware =
        super::super::test_support::hardware_present(super::super::test_support::Need::F32);
    println!("{test}: SME2 hardware present: {hardware}");
    let callers: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..3 {
                    let now = hw_now().copied();
                    let units_now = units();
                    let waited = hw().copied();
                    let read = candidate().copied();
                    let _ = (now, units_now, waited, read, enabled::<f32>());
                }
            })
        })
        .collect();
    for (index, caller) in callers.into_iter().enumerate() {
        let joined = caller.join();
        assert!(joined.is_ok(), "caller {index} panicked: {joined:?}");
    }
    assert!(candidate_known(), "the candidate is unknown after hw()");
    let runs = candidate_runs();
    assert_eq!(runs, 1, "the process computed its candidate {runs} times");
}
