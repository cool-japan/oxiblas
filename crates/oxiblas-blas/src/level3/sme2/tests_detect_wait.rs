//! Tests of what a waiting call guarantees (module docs, "Waiting, and
//! fork"): the first-use verdict on local [`FirstUse`] values with stand-in
//! checks for a thread that is not ready, in both forms and in every state
//! of the verdict and its lock (on any host); and the wiring of both stages
//! into the real [`hw`] and [`hw_now`], through stand-ins that the test puts
//! on its threads ([`StagesHere`]), so the process's stages are not touched:
//! stage 1 on any host, stage 2 where the process has SME2 hardware (its
//! real readiness closure executes the SME register probe).
//!
//! The stand-in check (or computation) of thread A parks until the test
//! releases it (a channel), so every interleaving is fixed by the test, not
//! by the scheduler. A call that must answer at once reports through a
//! channel that the test reads with a generous timeout: a call that waits
//! shows up as a failed assertion, never as a hang of the test binary. A
//! call that must still be blocked is given [`BLOCKED_FOR`], and only after
//! the test has seen that call arrive: a call on a local verdict signals
//! when it asks its readiness (it is then past the re-entrancy test and in
//! front of the lock, see [`spawn_entered_call`]), and a call of the real
//! [`hw`] signals from its thread right before the call. So the answer a
//! blocked call gives later never depends on when its thread first ran, and
//! a slow machine can only make the "still blocked" check pass more easily,
//! never fail it falsely. Every assertion message names what was observed.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, MutexGuard};
use std::time::Duration;

use super::*;

/// How long a test waits for an answer that must come, or for a thread
/// that must finish, before it calls that a hang.
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
/// answer that did not come in time (`it waited`), or a thread that ended
/// without an answer.
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
/// still open right after it, and how many checks, reports and readiness
/// questions it made.
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
    /// Times the call asked its readiness.
    asked: usize,
}

/// The [`Seen`] of a call that answered `answer`, asked its readiness
/// `asked` times and ran and reported nothing.
fn seen(answer: bool, open_after: bool, asked: usize) -> Seen {
    Seen {
        answer,
        open_after,
        checks: 0,
        reports: 0,
        asked,
    }
}

/// One call of `form` on a new thread whose readiness is `is_ready`, with a
/// stand-in check that would fail (so a check it runs by mistake also
/// changes the verdict); the thread sends what it saw.
fn spawn_call(
    first_use: &Arc<FirstUse>,
    form: Form,
    is_ready: fn() -> bool,
    tx: mpsc::Sender<Seen>,
) {
    spawn_signalling_call(first_use, form, is_ready, tx, None);
}

/// [`spawn_call`]; with `entered`, the thread also signals on it every time
/// the call asks its readiness. The verdict's code asks it once, after its
/// re-entrancy test and before it takes the lock, and only while the
/// verdict is open.
fn spawn_signalling_call(
    first_use: &Arc<FirstUse>,
    form: Form,
    is_ready: fn() -> bool,
    tx: mpsc::Sender<Seen>,
    entered: Option<mpsc::Sender<()>>,
) {
    let first_use = Arc::clone(first_use);
    std::thread::spawn(move || {
        let (checks, reports, asked) = (Cell::new(0), Cell::new(0), Cell::new(0));
        let answer = first_use.passed_in(
            form,
            || {
                asked.set(asked.get() + 1);
                if let Some(entered) = &entered {
                    let _ = entered.send(());
                }
                is_ready()
            },
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
            asked: asked.get(),
        });
    });
}

/// A call as [`spawn_call`] makes it, on an open verdict, that has asked its
/// readiness when this function returns (the test waits for its signal with
/// the patience). From then on the call is in front of the lock or behind
/// it: it has seen the verdict open, so what it answers later does not
/// depend on when its thread first ran, and a "still blocked" check that
/// follows is not passed by a thread that has not started yet.
fn spawn_entered_call(
    first_use: &Arc<FirstUse>,
    form: Form,
    is_ready: fn() -> bool,
    tx: mpsc::Sender<Seen>,
    label: &str,
) {
    let (entered_tx, entered_rx) = mpsc::channel();
    spawn_signalling_call(first_use, form, is_ready, tx, Some(entered_tx));
    let entered = entered_rx.recv_timeout(PATIENCE);
    assert!(
        entered.is_ok(),
        "{label}: the call did not reach its readiness question: {}",
        observed(&entered)
    );
}

/// One call as [`spawn_call`] makes it, read with the patience.
fn call(
    first_use: &Arc<FirstUse>,
    form: Form,
    is_ready: fn() -> bool,
) -> Result<Seen, RecvTimeoutError> {
    let (tx, rx) = mpsc::channel();
    spawn_call(first_use, form, is_ready, tx);
    rx.recv_timeout(PATIENCE)
}

/// How thread A's parked check ends.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Ending {
    /// It returns this outcome.
    Returns(Result<(), CheckFailure>),
    /// It panics.
    Panics,
}

/// Thread A: a ready thread that runs the check of `first_use` (in the
/// waiting form) and parks inside it until `release` delivers or its sender
/// is dropped; then it ends as `ending` says (a dropped sender, from a test
/// that unwinds, ends it with a `Gemv` failure, so A always finishes).
/// `started` is signalled once A is inside the check; `reports` counts the
/// failure lines A reports. The handle gives A's answer (an `Err` when it
/// panicked).
fn spawn_parked_checker(
    first_use: &Arc<FirstUse>,
    ending: Ending,
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
                match (release.recv(), ending) {
                    (Ok(()), Ending::Returns(outcome)) => outcome,
                    // `resume_unwind` skips the panic hook (no message in the
                    // log).
                    (Ok(()), Ending::Panics) => {
                        std::panic::resume_unwind(Box::new("a check that panics"))
                    }
                    (Err(_), _) => Err(CheckFailure::Gemv),
                }
            },
            |_| {
                reports.fetch_add(1, AtomicOrdering::SeqCst);
            },
        )
    })
}

/// Starts thread A on `first_use` with `ending` and waits until it is
/// inside its check: the release sender, A's handle and A's report count.
fn start_parked_checker(
    first_use: &Arc<FirstUse>,
    ending: Ending,
    label: &str,
) -> (
    mpsc::Sender<()>,
    std::thread::JoinHandle<bool>,
    Arc<AtomicUsize>,
) {
    let reports = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let checker = spawn_parked_checker(
        first_use,
        ending,
        started_tx,
        release_rx,
        Arc::clone(&reports),
    );
    let started = started_rx.recv_timeout(PATIENCE);
    assert!(
        started.is_ok(),
        "{label}: thread A did not start its check: {}",
        observed(&started)
    );
    (release_tx, checker, reports)
}

/// The lock of `first_use`, taken by the test thread (a fresh or released
/// lock is never poisoned in these tests).
fn hold<'a>(first_use: &'a FirstUse, label: &str) -> MutexGuard<'a, ()> {
    match first_use.lock.lock() {
        Ok(guard) => guard,
        Err(poison) => panic!("{label}: the lock is poisoned: {poison:?}"),
    }
}

/// Whether the lock of `first_use` is free now (taken and released at once;
/// a poisoned lock counts as free).
fn lock_is_free(first_use: &FirstUse) -> bool {
    !matches!(first_use.lock.try_lock(), Err(TryLockError::WouldBlock))
}

/// A waiting call from a thread that is not ready, made while a ready
/// thread runs the check: it blocks until that check ends, then answers
/// what it settled ("passed" or "failed") without running a check or
/// reporting anything. A non-waiting call from a thread that is not ready
/// answers `false` at once meanwhile and never touches the verdict.
#[test]
fn test_sme2_detect_wait_not_ready_caller_waits_for_a_check_in_progress() {
    for outcome in [Ok(()), Err(CheckFailure::ZaPackers)] {
        let passes = outcome.is_ok();
        let label = format!("a check that ends with {outcome:?}");
        let first_use = Arc::new(FirstUse::new());
        let (release, checker, reports) =
            start_parked_checker(&first_use, Ending::Returns(outcome), &label);

        // N, not ready, waiting form: blocked while A is parked.
        let (n_tx, n_rx) = mpsc::channel();
        spawn_entered_call(&first_use, Form::Wait, not_ready, n_tx, &label);
        let early = n_rx.recv_timeout(BLOCKED_FOR);
        assert!(
            early.is_err(),
            "{label}: the waiting form on a thread that is not ready must wait for the check \
             in progress: {}",
            observed(&early)
        );

        // M, not ready, non-waiting form: `false` at once, verdict open.
        let m = call(&first_use, Form::NoWait, not_ready);
        let want_m = seen(false, true, 1);
        assert!(
            m == Ok(want_m),
            "{label}: the non-waiting form on a thread that is not ready must answer {want_m:?} \
             at once while the check runs: {}",
            observed(&m)
        );

        let _ = release.send(());
        let a = checker.join();
        assert!(
            matches!(a, Ok(answer) if answer == passes),
            "{label}: thread A must answer {passes}: it gave {a:?}"
        );
        let n = n_rx.recv_timeout(PATIENCE);
        let want_n = seen(passes, false, 1);
        assert!(
            n == Ok(want_n),
            "{label}: the waiting form on a thread that is not ready must answer A's verdict \
             {want_n:?} after the check: {}",
            observed(&n)
        );
        let settled = first_use.verdict();
        let as_settled = match settled {
            Verdict::Passed => passes,
            Verdict::Failed(_) => !passes,
            Verdict::Open => false,
        };
        assert!(as_settled, "{label}: the verdict is {settled:?}");
        let reported = reports.load(AtomicOrdering::SeqCst);
        assert_eq!(
            reported,
            usize::from(!passes),
            "{label}: A reported {reported} failure lines"
        );
        assert!(lock_is_free(&first_use), "{label}: the lock is still held");
    }
}

/// A waiting call from a thread that is not ready, made while nobody runs
/// the check, answers `false` at once, runs nothing, leaves the verdict
/// open and the lock free; a later call from a ready thread still runs the
/// check (once). Two such calls that find the lock held by a thread that
/// runs nothing (the test thread) both wait for it, and then both answer
/// `false` with the verdict left open.
#[test]
fn test_sme2_detect_wait_not_ready_caller_returns_when_nobody_checks() {
    let first_use = Arc::new(FirstUse::new());
    for round in 0..3 {
        let n = call(&first_use, Form::Wait, not_ready);
        let want = seen(false, true, 1);
        assert!(
            n == Ok(want),
            "round {round}: the waiting form on a thread that is not ready, with nobody \
             checking, must answer {want:?} at once: {}",
            observed(&n)
        );
        assert!(
            lock_is_free(&first_use),
            "round {round}: the call left the lock held"
        );
    }

    // Two such waiting calls while the test thread holds the lock.
    let held = hold(&first_use, "two waiters");
    let (tx, rx) = mpsc::channel();
    for _ in 0..2 {
        spawn_entered_call(&first_use, Form::Wait, not_ready, tx.clone(), "two waiters");
    }
    let early = rx.recv_timeout(BLOCKED_FOR);
    assert!(
        early.is_err(),
        "two waiters: a waiting call on a thread that is not ready returned while the lock was \
         held: {}",
        observed(&early)
    );
    drop(held);
    for index in 0..2 {
        let n = rx.recv_timeout(PATIENCE);
        let want = seen(false, true, 1);
        assert!(
            n == Ok(want),
            "two waiters: waiter {index} after the lock was released must answer {want:?}: {}",
            observed(&n)
        );
    }
    let verdict_now = first_use.verdict();
    assert_eq!(
        verdict_now,
        Verdict::Open,
        "the calls from threads that are not ready settled {verdict_now:?}"
    );
    assert!(lock_is_free(&first_use), "the waiters left the lock held");

    // A ready thread still runs the check.
    let checks = Cell::new(0_usize);
    let answer = first_use.passed(
        ready,
        || {
            checks.set(checks.get() + 1);
            Ok(())
        },
        |line| panic!("a passing check reported {line}"),
    );
    assert!(answer, "the ready thread's own check answered {answer}");
    assert_eq!(
        checks.get(),
        1,
        "the ready thread ran {} checks",
        checks.get()
    );
}

/// A thread that is not ready in the non-waiting form answers at once in
/// every state of the verdict and its lock, runs no check, reports nothing
/// and changes nothing: open with the lock free, held by the test thread,
/// held by a thread that runs the check, poisoned; passed or failed with
/// the lock held. On a settled verdict the waiting form answers at once as
/// well, without the lock and without asking its readiness.
#[test]
fn test_sme2_detect_wait_not_ready_answers_at_once_without_the_lock() {
    // Open, lock free.
    let first_use = Arc::new(FirstUse::new());
    let m = call(&first_use, Form::NoWait, not_ready);
    assert!(
        m == Ok(seen(false, true, 1)),
        "open, lock free: {}",
        observed(&m)
    );

    // Open, lock held by the test thread (it runs nothing).
    let held = hold(&first_use, "open, held");
    let m = call(&first_use, Form::NoWait, not_ready);
    drop(held);
    assert!(
        m == Ok(seen(false, true, 1)),
        "open, lock held by the test: {}",
        observed(&m)
    );

    // Open, lock held by a thread that runs the check.
    let (release, checker, _) =
        start_parked_checker(&first_use, Ending::Returns(Ok(())), "open, checking");
    let m = call(&first_use, Form::NoWait, not_ready);
    let _ = release.send(());
    let a = checker.join();
    assert!(
        m == Ok(seen(false, true, 1)),
        "open, a check in progress: {}",
        observed(&m)
    );
    assert!(matches!(a, Ok(true)), "open, checking: thread A gave {a:?}");

    // Settled, lock held: both forms answer at once, without asking.
    for passes in [true, false] {
        let first_use = Arc::new(FirstUse::new());
        let settled = first_use.passed(
            ready,
            || {
                if passes {
                    Ok(())
                } else {
                    Err(CheckFailure::Gemv)
                }
            },
            |_| {},
        );
        assert_eq!(settled, passes, "settling to {passes} answered {settled}");
        let held = hold(&first_use, "settled, held");
        let answers: Vec<_> = [Form::NoWait, Form::Wait]
            .into_iter()
            .map(|form| (form, call(&first_use, form, not_ready)))
            .collect();
        drop(held);
        for (form, answer) in answers {
            let want = seen(passes, false, 0);
            assert!(
                answer == Ok(want),
                "settled to {passes}, lock held: {form:?} on a thread that is not ready must \
                 answer {want:?} at once: {}",
                observed(&answer)
            );
        }
    }

    // Open, lock poisoned (a check that panicked, nobody since).
    let first_use = Arc::new(FirstUse::new());
    let panicked = {
        let first_use = Arc::clone(&first_use);
        std::thread::spawn(move || {
            first_use.passed(
                ready,
                || std::panic::resume_unwind(Box::new("a check that panics")),
                |_| {},
            )
        })
        .join()
    };
    assert!(panicked.is_err(), "the check must unwind: {panicked:?}");
    let m = call(&first_use, Form::NoWait, not_ready);
    assert!(
        m == Ok(seen(false, true, 1)),
        "open, lock poisoned: {}",
        observed(&m)
    );
}

/// The check of thread A panics while a waiting call from a thread that is
/// not ready waits for it: the waiter then takes the poisoned lock, settles
/// "failed" (the line of a poisoned lock, reported once, by the waiter)
/// without running a check, and answers `false`; every later call in either
/// form, ready or not, answers `false` without a check or a report. The same
/// when the waiting call from a thread that is not ready comes only after
/// the panic.
#[test]
fn test_sme2_detect_wait_not_ready_caller_after_a_panic() {
    let poisoned_line = failure_line(&CheckFailure::Poisoned);

    // The waiter blocked during the panic.
    let first_use = Arc::new(FirstUse::new());
    let (release, checker, a_reports) =
        start_parked_checker(&first_use, Ending::Panics, "during the panic");
    let (n_tx, n_rx) = mpsc::channel();
    spawn_entered_call(&first_use, Form::Wait, not_ready, n_tx, "during the panic");
    let early = n_rx.recv_timeout(BLOCKED_FOR);
    assert!(
        early.is_err(),
        "during the panic: the waiting form on a thread that is not ready must wait for the \
         check in progress: {}",
        observed(&early)
    );
    let _ = release.send(());
    let a = checker.join();
    assert!(
        a.is_err(),
        "thread A's check must unwind: it returned {a:?}"
    );
    let n = n_rx.recv_timeout(PATIENCE);
    let want_n = Seen {
        answer: false,
        open_after: false,
        checks: 0,
        reports: 1,
        asked: 1,
    };
    assert!(
        n == Ok(want_n),
        "during the panic: the blocked waiter must settle the poisoned lock as a failure, \
         {want_n:?}: {}",
        observed(&n)
    );
    let settled = first_use.verdict();
    assert_eq!(
        settled,
        Verdict::Failed(poisoned_line.as_str()),
        "during the panic: the verdict is {settled:?}"
    );
    let reported = a_reports.load(AtomicOrdering::SeqCst);
    assert_eq!(
        reported, 0,
        "the panicking thread reported {reported} lines"
    );
    for form in [Form::NoWait, Form::Wait] {
        for (who, is_ready) in [("ready", ready as fn() -> bool), ("not ready", not_ready)] {
            let later = call(&first_use, form, is_ready);
            assert!(
                later == Ok(seen(false, false, 0)),
                "after the panic: {form:?} on a thread that is {who}: {}",
                observed(&later)
            );
        }
    }

    // Nobody waiting during the panic: the first call afterwards is the
    // waiting form on a thread that is not ready.
    let first_use = Arc::new(FirstUse::new());
    let panicked = {
        let first_use = Arc::clone(&first_use);
        std::thread::spawn(move || {
            first_use.passed(
                ready,
                || std::panic::resume_unwind(Box::new("a check that panics")),
                |_| {},
            )
        })
        .join()
    };
    assert!(panicked.is_err(), "the check must unwind: {panicked:?}");
    let n = call(&first_use, Form::Wait, not_ready);
    assert!(
        n == Ok(want_n),
        "after the panic: the first waiting call on a thread that is not ready must answer \
         {want_n:?}: {}",
        observed(&n)
    );
    let settled = first_use.verdict();
    assert_eq!(
        settled,
        Verdict::Failed(poisoned_line.as_str()),
        "after the panic: the verdict is {settled:?}"
    );
}

/// A call from inside the check, on the checking thread, returns `false` at
/// once in both forms and for either readiness, without touching the lock
/// that thread holds: with the lock taken it would never return. The test
/// reads the result through a channel with the patience, so a call that
/// takes the lock shows up as "it waited" (or as a panic of that thread),
/// never as a hang of the test binary.
#[test]
fn test_sme2_detect_wait_reentrant_calls_never_take_the_lock() {
    let first_use = Arc::new(FirstUse::new());
    let (tx, rx) = mpsc::channel();
    {
        let first_use = Arc::clone(&first_use);
        std::thread::spawn(move || {
            let inner = std::cell::RefCell::new(Vec::new());
            let outer = first_use.passed(
                ready,
                || {
                    for form in [Form::Wait, Form::NoWait] {
                        for is_ready in [not_ready as fn() -> bool, ready] {
                            let answer = first_use.passed_in(form, is_ready, || Ok(()), |_| {});
                            inner.borrow_mut().push((form, is_ready(), answer));
                        }
                    }
                    Ok(())
                },
                |_| {},
            );
            let _ = tx.send((outer, inner.into_inner()));
        });
    }
    let answer = rx.recv_timeout(PATIENCE);
    let want = (
        true,
        vec![
            (Form::Wait, false, false),
            (Form::Wait, true, false),
            (Form::NoWait, false, false),
            (Form::NoWait, true, false),
        ],
    );
    assert!(
        answer.as_ref() == Ok(&want),
        "the calls from inside the check must answer {want:?} at once: {}",
        observed(&answer)
    );
    let settled = first_use.verdict();
    assert_eq!(settled, Verdict::Passed, "the verdict is {settled:?}");
}

/// [`hw`] and [`hw_now`] hand their form to stage 1: while thread A computes
/// the candidate of a stage-1 cell (parked; it computes "no candidate", so
/// no call goes on to stage 2), the real [`hw_now`] and [`units`] on a
/// thread that reads that cell answer at once without storing anything,
/// and the real [`hw`] waits for A and then answers A's value. The cell is
/// put on each thread with [`StagesHere`]; the process's candidate is
/// never read or computed by these calls.
#[test]
fn test_sme2_detect_wait_hw_in_hands_its_form_to_stage_1() {
    let cell: &'static CandidateCell = Box::leak(Box::new(CandidateCell::new()));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let computed = Arc::new(AtomicUsize::new(0));
    let computer = {
        let computed = Arc::clone(&computed);
        std::thread::spawn(move || {
            cell.get_in(Form::Wait, || {
                computed.fetch_add(1, AtomicOrdering::SeqCst);
                let _ = started_tx.send(());
                let _ = release_rx.recv();
                None
            })
            .copied()
        })
    };
    let started = started_rx.recv_timeout(PATIENCE);
    assert!(
        started.is_ok(),
        "thread A did not start its computation: {}",
        observed(&started)
    );

    // B: the non-waiting questions, at once.
    let non_waiting = || {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _here = StagesHere::candidate(cell);
            let now = hw_now().copied();
            let parts = units();
            let _ = tx.send((now, parts, cell.known().is_some()));
        });
        rx.recv_timeout(PATIENCE)
    };
    let b = non_waiting();
    assert!(
        b == Ok((None, 1, false)),
        "hw_now() and units() while another thread computes the candidate must answer \
         (None, 1) at once and store nothing (false): {}",
        observed(&b)
    );

    // C: the waiting hw(), blocked while A computes.
    let (c_tx, c_rx) = mpsc::channel();
    let (c_started_tx, c_started_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _here = StagesHere::candidate(cell);
        let _ = c_started_tx.send(());
        let waited = hw().copied();
        let _ = c_tx.send((waited, cell.known().is_some()));
    });
    let c_started = c_started_rx.recv_timeout(PATIENCE);
    assert!(
        c_started.is_ok(),
        "thread C did not start: {}",
        observed(&c_started)
    );
    let early = c_rx.recv_timeout(BLOCKED_FOR);
    assert!(
        early.is_err(),
        "hw() must wait while another thread computes the candidate: {}",
        observed(&early)
    );

    let _ = release_tx.send(());
    let a = computer.join();
    assert!(
        matches!(a, Ok(None)),
        "thread A must compute \"no candidate\": it gave {a:?}"
    );
    let c = c_rx.recv_timeout(PATIENCE);
    assert!(
        c == Ok((None, true)),
        "hw() after the computation must answer A's value (None, known): {}",
        observed(&c)
    );
    let b = non_waiting();
    assert!(
        b == Ok((None, 1, true)),
        "hw_now() and units() after the computation: {}",
        observed(&b)
    );
    let runs = computed.load(AtomicOrdering::SeqCst);
    assert_eq!(runs, 1, "the cell's candidate was computed {runs} times");
}

/// [`hw`] and [`hw_now`] hand their form to stage 2: with a stand-in
/// verdict that is open and whose lock the test holds (nobody runs a check
/// under it), the real [`hw_now`] and [`units`] answer at once (`None`, and
/// one unit) and the real [`hw`] waits; once the test settles the stand-in to
/// "passed" and releases the lock, [`hw`] answers the process's candidate,
/// and so does [`hw_now`]. No check runs: the stand-in is settled before its
/// lock is free. Gated by the hardware, because the real readiness closure
/// of stage 2 executes the SME register probe.
#[test]
fn test_sme2_detect_wait_hw_in_hands_its_form_to_stage_2() {
    use super::super::test_support::{Need, hardware_or_skip};
    let test = "test_sme2_detect_wait_hw_in_hands_its_form_to_stage_2";
    if !hardware_or_skip(test, Need::F32) {
        return;
    }
    let want = candidate().copied();
    let Some(want_hw) = want else {
        panic!("{test}: the hardware gate passed without a candidate");
    };
    let first_use: &'static FirstUse = Box::leak(Box::new(FirstUse::new()));
    let held = hold(first_use, test);

    // B: the non-waiting questions, at once.
    let non_waiting = || {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _here = StagesHere::first_use(first_use);
            let now = hw_now().copied();
            let parts = units();
            let _ = tx.send((now, parts, first_use.verdict() == Verdict::Open));
        });
        rx.recv_timeout(PATIENCE)
    };
    let b = non_waiting();
    assert!(
        b == Ok((None, 1, true)),
        "{test}: hw_now() and units() while the verdict's lock is held must answer (None, 1) \
         at once and leave it open (true): {}",
        observed(&b)
    );

    // C: the waiting hw(), blocked while the lock is held.
    let (c_tx, c_rx) = mpsc::channel();
    let (c_started_tx, c_started_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _here = StagesHere::first_use(first_use);
        let _ = c_started_tx.send(());
        let _ = c_tx.send(hw().copied());
    });
    let c_started = c_started_rx.recv_timeout(PATIENCE);
    assert!(
        c_started.is_ok(),
        "{test}: thread C did not start: {}",
        observed(&c_started)
    );
    let early = c_rx.recv_timeout(BLOCKED_FOR);
    assert!(
        early.is_err(),
        "{test}: hw() must wait while the verdict's lock is held: {}",
        observed(&early)
    );

    first_use.state.store(PASSED, AtomicOrdering::Release);
    drop(held);
    let c = c_rx.recv_timeout(PATIENCE);
    assert!(
        c == Ok(want),
        "{test}: hw() after the release must answer the candidate {want:?}: {}",
        observed(&c)
    );
    let b = non_waiting();
    assert!(
        b == Ok((want, want_hw.units.max(1), false)),
        "{test}: hw_now() and units() after the release must answer ({want:?}, {}, false): {}",
        want_hw.units.max(1),
        observed(&b)
    );
}
