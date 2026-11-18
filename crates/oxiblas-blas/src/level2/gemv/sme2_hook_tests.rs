//! Tests of the SME2 GEMV hooks of the public GEMV entry points (`gemv`,
//! and `gemv_with_par` on its sequential and on its parallel path).
//!
//! Every comparison is against the same public call made while the calling
//! thread holds `test_support::force_off` (the hooks decline there and the
//! NEON kernels compute it), bitwise except NaN sign and payload. A call
//! that runs on SME2 moves the decision counter of the calling thread once
//! and leaves the run record of the SME2 GEMV (`sme2::take_last_run`: every
//! part ran on SME2, its kernel entries counted on its own thread). A call
//! below the crossover never reaches the detection: the detection-call
//! counter of the thread (`detect::queries_here`, test builds only) does
//! not move.

use num_complex::Complex64;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::MatRef;

use super::sme2::{MIN_ROWS, MIN_WORK, TRANS_MIN_COLS, take_last_run};
#[cfg(feature = "parallel")]
use super::sme2::{Switches, override_switches};
use super::{GemvTrans, gemv, gemv_with_par};
use crate::level3::sme2::args::{SmeElement, same_result};
#[cfg(feature = "parallel")]
use crate::level3::sme2::detect::override_units;
use crate::level3::sme2::detect::{queries_here, set_pretend_not_ready};
use crate::level3::sme2::test_support::{
    Need, backend_or_skip, backend_present, decisions, force_off, hardware_present, note_declined,
    sme_state,
};

/// The element types of these tests.
trait Elem: SmeElement + oxiblas_core::scalar::Field + Send + Sync + core::fmt::Debug {
    /// The hardware the SME2 GEMV of the type needs.
    const NEED: Need;
    /// `x` as the type.
    fn of(x: f64) -> Self;
}

impl Elem for f32 {
    const NEED: Need = Need::F32;
    fn of(x: f64) -> Self {
        x as f32
    }
}

impl Elem for f64 {
    const NEED: Need = Need::F64;
    fn of(x: f64) -> Self {
        x
    }
}

/// A public GEMV entry point.
#[derive(Debug, Clone, Copy)]
enum Entry {
    /// `gemv`.
    Gemv,
    /// `gemv_with_par` with this `Par` (its parallel path from `256²`
    /// elements with a parallel `Par`, its sequential path otherwise).
    WithPar(Par),
}

/// The entry points: `gemv`, `gemv_with_par` sequential and parallel.
fn entries() -> Vec<Entry> {
    #[cfg(feature = "parallel")]
    let entries = vec![
        Entry::Gemv,
        Entry::WithPar(Par::Seq),
        Entry::WithPar(Par::RayonWith(3)),
    ];
    #[cfg(not(feature = "parallel"))]
    let entries = vec![Entry::Gemv, Entry::WithPar(Par::Seq)];
    entries
}

/// The three transposes.
const TRANSES: [GemvTrans; 3] = [GemvTrans::NoTrans, GemvTrans::Trans, GemvTrans::ConjTrans];

/// One GEMV problem: the stored `m × n` A (leading dimension `m + pad`, or
/// row-major when `row_major`), `x`, `y` and the scalars.
#[derive(Debug, Clone)]
struct Problem<T> {
    /// Transpose.
    trans: GemvTrans,
    /// Rows and columns of the stored A.
    dims: (usize, usize),
    /// A's elements.
    a: Vec<T>,
    /// Elements between columns of A (or between rows when `row_major`).
    lda: usize,
    /// A is stored row-major (no unit column stride).
    row_major: bool,
    /// `x`.
    x: Vec<T>,
    /// `y` before the call.
    y: Vec<T>,
    /// alpha and beta.
    scalars: (T, T),
}

impl<T: Elem> Problem<T> {
    /// A problem with data from `seed`; `y` is NaN when `beta == 0`.
    fn new(trans: GemvTrans, m: usize, n: usize, alpha: f64, beta: f64, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut value = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            T::of(((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0)
        };
        let lda = m + 3;
        let a = (0..lda * n).map(|_| value()).collect();
        let (x_len, y_len) = match trans {
            GemvTrans::NoTrans => (n, m),
            GemvTrans::Trans | GemvTrans::ConjTrans => (m, n),
        };
        let x = (0..x_len).map(|_| value()).collect();
        let y = (0..y_len)
            .map(|_| {
                if beta == 0.0 {
                    T::of(f64::NAN)
                } else {
                    value()
                }
            })
            .collect();
        Self {
            trans,
            dims: (m, n),
            a,
            lda,
            row_major: false,
            x,
            y,
            scalars: (T::of(alpha), T::of(beta)),
        }
    }

    /// The same problem with A stored row-major (`col_stride() != 1`).
    fn row_major(mut self) -> Self {
        let (m, n) = self.dims;
        self.lda = n + 2;
        self.a = (0..m * self.lda)
            .map(|index| T::of(((index * 37) % 101) as f64 / 64.0 - 0.75))
            .collect();
        self.row_major = true;
        self
    }

    /// The view of A.
    fn a(&self) -> MatRef<'_, T> {
        let (m, n) = self.dims;
        let (row_stride, col_stride) = if self.row_major {
            (1, self.lda)
        } else {
            (self.lda, 1)
        };
        match MatRef::from_strided(&self.a, m, n, row_stride, col_stride) {
            Some(view) => view,
            None => panic!("A view of {m}x{n}"),
        }
    }
}

/// What one call left and recorded.
#[derive(Debug)]
struct Out<T> {
    /// `y` after the call.
    y: Vec<T>,
    /// Decisions of the calling thread.
    decisions: u64,
    /// Detection calls of the calling thread.
    queries: u64,
    /// The run record of the SME2 GEMV (`None`: it did not run).
    run: Option<super::sme2::GemvRun>,
}

/// `p` through `entry` on this thread.
fn call<T: Elem>(entry: Entry, p: &Problem<T>) -> Out<T> {
    let mut y = p.y.clone();
    let (alpha, beta) = p.scalars;
    let _ = take_last_run();
    let (decided, asked) = (decisions(), queries_here());
    match entry {
        Entry::Gemv => gemv(p.trans, alpha, p.a(), &p.x, beta, &mut y),
        Entry::WithPar(par) => gemv_with_par(p.trans, alpha, p.a(), &p.x, beta, &mut y, par),
    }
    Out {
        y,
        decisions: decisions() - decided,
        queries: queries_here() - asked,
        run: take_last_run(),
    }
}

/// `p` through `entry` with the backend forced off: the NEON result.
fn reference<T: Elem>(entry: Entry, p: &Problem<T>) -> Out<T> {
    let _off = force_off();
    let out = call(entry, p);
    assert_eq!(out.decisions, 0, "a decision under force_off");
    assert!(out.run.is_none(), "SME2 ran under force_off");
    out
}

/// `got.y` equals `want.y` element by element (bitwise, or NaN in both).
fn assert_same_y<T: Elem>(got: &Out<T>, want: &Out<T>, label: &str) {
    assert_eq!(got.y.len(), want.y.len(), "{label}: length");
    for (index, (g, w)) in got.y.iter().zip(&want.y).enumerate() {
        assert!(
            same_result(*g, *w),
            "{label}: y[{index}] is {:#x}, the NEON call gives {:#x}",
            g.bits(),
            w.bits()
        );
    }
}

/// `p` through `entry` runs on SME2 (one decision, every part on SME2 with
/// kernel entries on its own thread) with the NEON bits; returns the parts.
fn assert_on_sme2<T: Elem>(entry: Entry, p: &Problem<T>, label: &str) -> usize {
    let want = reference(entry, p);
    let got = call(entry, p);
    assert_eq!(got.decisions, 1, "{label}: decisions");
    let Some(run) = got.run.as_ref() else {
        panic!("{label}: no SME2 run recorded");
    };
    for (index, part) in run.parts.iter().enumerate() {
        assert!(part.sme2, "{label}: part {index} ran on NEON");
        assert!(part.kernel_calls > 0, "{label}: part {index} ran no kernel");
    }
    assert_same_y(&got, &want, label);
    assert_eq!(sme_state(), Some((0, 0)), "{label}: SME state");
    run.parts.len()
}

/// `p` through `entry` does not run on SME2 and gives the NEON bits;
/// returns the detection calls it made.
fn assert_not_on_sme2<T: Elem>(entry: Entry, p: &Problem<T>, label: &str) -> u64 {
    let want = reference(entry, p);
    let got = call(entry, p);
    assert_eq!(got.decisions, 0, "{label}: decided");
    assert!(got.run.is_none(), "{label}: SME2 ran");
    assert_same_y(&got, &want, label);
    got.queries
}

/// Every entry point and transpose, on shapes past the crossover (vector
/// residues of both types in `m`, a column count that is not a multiple of
/// the column groups, a padded A), with the scalars of the four `y` updates
/// (beta 0 on a NaN `y`, 1, other) and a NaN alpha: SME2 runs with the
/// NEON bits.
fn entries_run_on_sme2<T: Elem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let shapes = [(64, 256), (65, 300), (100, 165), (257, 64), (300, 300)];
    let scalars = [(1.0, 0.0), (0.75, 1.0), (-1.25, 0.5), (f64::NAN, 1.0)];
    for entry in entries() {
        for trans in TRANSES {
            for (index, &(m, n)) in shapes.iter().enumerate() {
                for &(alpha, beta) in &scalars {
                    let label = format!("{test}: {entry:?} {trans:?} {m}x{n} a {alpha} b {beta}");
                    let p = Problem::<T>::new(trans, m, n, alpha, beta, index as u64 + 3);
                    assert_on_sme2(entry, &p, &label);
                }
            }
        }
    }
}

#[test]
fn test_sme2_gemv_hook_entries_run_on_sme2_f32() {
    entries_run_on_sme2::<f32>("test_sme2_gemv_hook_entries_run_on_sme2_f32");
}

#[test]
fn test_sme2_gemv_hook_entries_run_on_sme2_f64() {
    entries_run_on_sme2::<f64>("test_sme2_gemv_hook_entries_run_on_sme2_f64");
}

/// The parallel entry point with 2 and 3 SME units (the unit count and the
/// size threshold of the split overridden on the calling thread): the call
/// is split into that many parts, every one on SME2, with the NEON bits.
#[cfg(feature = "parallel")]
fn split_runs_on_sme2<T: Elem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for units in [2, 3] {
        let _units = override_units(units);
        let _switches = override_switches(Switches {
            par_min_bytes: Some(1),
            ..Switches::default()
        });
        for trans in TRANSES {
            let label = format!("{test}: {units} units {trans:?}");
            let p = Problem::<T>::new(trans, 300, 280, 0.5, 0.25, units as u64);
            let parts = assert_on_sme2(Entry::WithPar(Par::RayonWith(units)), &p, &label);
            assert_eq!(parts, units, "{label}: parts");
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_hook_split_runs_on_sme2_f32() {
    split_runs_on_sme2::<f32>("test_sme2_gemv_hook_split_runs_on_sme2_f32");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_hook_split_runs_on_sme2_f64() {
    split_runs_on_sme2::<f64>("test_sme2_gemv_hook_split_runs_on_sme2_f64");
}

/// `(m, n, trans, past)` around every crossover threshold: `MIN_ROWS`,
/// `MIN_WORK` and, for `Trans`, `TRANS_MIN_COLS` (one below, at, one
/// above), and the same below the crossover on shapes large enough for the
/// parallel path of `gemv_with_par` (`256²` elements).
fn threshold_cases() -> Vec<(usize, usize, GemvTrans, bool)> {
    let (rows, work, cols) = (MIN_ROWS, MIN_WORK, TRANS_MIN_COLS);
    let side = 128;
    assert_eq!(side * side, work, "MIN_WORK is no longer 128²");
    let mut cases = Vec::new();
    for trans in TRANSES {
        // MIN_ROWS, with plenty of work and columns.
        for (m, past) in [(rows - 1, false), (rows, true), (rows + 1, true)] {
            cases.push((m, 1100, trans, past));
        }
        // MIN_WORK, with enough rows and columns.
        for (n, past) in [(side - 1, false), (side, true), (side + 1, true)] {
            cases.push((side, n, trans, past));
        }
        // TRANS_MIN_COLS (NoTrans has no column minimum).
        let transposed = trans != GemvTrans::NoTrans;
        for (n, past) in [(cols - 1, !transposed), (cols, true), (cols + 1, true)] {
            cases.push((4400, n, trans, past));
        }
    }
    cases
}

/// The entry points at every crossover threshold: past it SME2 runs; below
/// it the call does not reach the detection, and gives the NEON bits.
fn crossover_thresholds<T: Elem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for entry in entries() {
        for (m, n, trans, past) in threshold_cases() {
            let label = format!("{test}: {entry:?} {trans:?} {m}x{n}");
            let p = Problem::<T>::new(trans, m, n, 0.75, 0.5, (m + n) as u64);
            if past {
                assert_on_sme2(entry, &p, &label);
            } else {
                let asked = assert_not_on_sme2(entry, &p, &label);
                assert_eq!(asked, 0, "{label}: asked the detection below the crossover");
            }
        }
    }
}

#[test]
fn test_sme2_gemv_hook_crossover_thresholds_f32() {
    crossover_thresholds::<f32>("test_sme2_gemv_hook_crossover_thresholds_f32");
}

#[test]
fn test_sme2_gemv_hook_crossover_thresholds_f64() {
    crossover_thresholds::<f64>("test_sme2_gemv_hook_crossover_thresholds_f64");
}

/// Below the crossover, on any host: sizes 1 to 32 (and the thresholds
/// minus one) through every entry point and transpose never reach the
/// detection. Ungated: it holds with and without the backend, and on a host
/// without SME it is the only check of it.
fn below_crossover_never_asks<T: Elem>(test: &str) {
    let mut shapes: Vec<(usize, usize)> = (1..=32).map(|s| (s, s)).collect();
    shapes.extend([(MIN_ROWS - 1, 4096), (127, 128), (4096, TRANS_MIN_COLS - 1)]);
    for entry in entries() {
        for trans in TRANSES {
            for &(m, n) in &shapes {
                if (m, n) == (4096, TRANS_MIN_COLS - 1) && trans == GemvTrans::NoTrans {
                    continue;
                }
                let label = format!("{test}: {entry:?} {trans:?} {m}x{n}");
                let p = Problem::<T>::new(trans, m, n, 1.5, 0.5, 9);
                let out = call(entry, &p);
                assert_eq!(out.queries, 0, "{label}: asked the detection");
                assert_eq!(out.decisions, 0, "{label}: decided");
                let want = reference(entry, &p);
                assert_same_y(&out, &want, &label);
            }
        }
    }
}

#[test]
fn test_sme2_gemv_hook_below_crossover_never_asks_f32() {
    below_crossover_never_asks::<f32>("test_sme2_gemv_hook_below_crossover_never_asks_f32");
}

#[test]
fn test_sme2_gemv_hook_below_crossover_never_asks_f64() {
    below_crossover_never_asks::<f64>("test_sme2_gemv_hook_below_crossover_never_asks_f64");
}

/// Past the crossover the hooks still decline (NEON bits, no decision) for
/// a row-major A (`col_stride() != 1`), a held `force_off` and a thread
/// that pretends not to be ready; `alpha == 0` returns before the hook
/// (`y = beta·y` only, no detection call).
fn declines_past_the_crossover<T: Elem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for entry in entries() {
        for trans in TRANSES {
            let label = format!("{test}: {entry:?} {trans:?}");
            let p = Problem::<T>::new(trans, 300, 300, 0.75, 0.5, 21);
            assert_on_sme2(entry, &p, &label);
            assert_not_on_sme2(
                entry,
                &p.clone().row_major(),
                &format!("{label}: row-major"),
            );
            {
                let _off = force_off();
                let asked = assert_not_on_sme2(entry, &p, &format!("{label}: force_off"));
                assert_eq!(asked, 0, "{label}: force_off asked the detection");
            }
            let previous = set_pretend_not_ready(true);
            let pretended = call(entry, &p);
            set_pretend_not_ready(previous);
            let want = reference(entry, &p);
            assert_eq!(pretended.decisions, 0, "{label}: decided while not ready");
            assert!(pretended.run.is_none(), "{label}: SME2 ran while not ready");
            assert_same_y(&pretended, &want, &label);
            let zero = Problem::<T> {
                scalars: (T::of(0.0), T::of(0.5)),
                ..p.clone()
            };
            let asked = assert_not_on_sme2(entry, &zero, &format!("{label}: alpha 0"));
            assert_eq!(asked, 0, "{label}: alpha 0 asked the detection");
        }
    }
}

#[test]
fn test_sme2_gemv_hook_declines_past_the_crossover_f32() {
    declines_past_the_crossover::<f32>("test_sme2_gemv_hook_declines_past_the_crossover_f32");
}

#[test]
fn test_sme2_gemv_hook_declines_past_the_crossover_f64() {
    declines_past_the_crossover::<f64>("test_sme2_gemv_hook_declines_past_the_crossover_f64");
}

/// A complex GEMV past the crossover: the type hand-off declines it before
/// the detection is asked, and the result is the one with the backend off.
/// Ungated (the decline comes before any hardware question).
#[test]
fn test_sme2_gemv_hook_other_types_decline() {
    let test = "test_sme2_gemv_hook_other_types_decline";
    let (m, n) = (300, 200);
    let value = |index: usize| Complex64::new((index % 7) as f64 * 0.25, (index % 5) as f64 * 0.5);
    let a: Vec<Complex64> = (0..m * n).map(value).collect();
    let view = match MatRef::from_column_major(&a, m, n) {
        Some(view) => view,
        None => panic!("{test}: A view"),
    };
    for entry in entries() {
        for trans in TRANSES {
            let (x_len, y_len) = match trans {
                GemvTrans::NoTrans => (n, m),
                _ => (m, n),
            };
            let x: Vec<Complex64> = (0..x_len).map(|i| value(i + 3)).collect();
            let run = |off: bool| {
                let _off = off.then(force_off);
                let mut y: Vec<Complex64> = (0..y_len).map(|i| value(i + 1)).collect();
                let (decided, asked) = (decisions(), queries_here());
                let (alpha, beta) = (value(2), value(4));
                match entry {
                    Entry::Gemv => gemv(trans, alpha, view, &x, beta, &mut y),
                    Entry::WithPar(par) => gemv_with_par(trans, alpha, view, &x, beta, &mut y, par),
                }
                (y, decisions() - decided, queries_here() - asked)
            };
            let (want, _, _) = run(true);
            let (got, decided, asked) = run(false);
            let label = format!("{test}: {entry:?} {trans:?}");
            assert_eq!(decided, 0, "{label}: decided");
            assert_eq!(asked, 0, "{label}: asked the detection");
            assert_eq!(got, want, "{label}: y");
        }
    }
}

/// Where the backend is off for `T`, the calls past the crossover through
/// every entry point decline with the NEON bits. Ungated: where the backend
/// is on it prints a `note:` line, otherwise it writes the `declined`
/// marker.
fn declines_without_backend<T: Elem>(test: &str) {
    if backend_present(T::NEED) {
        println!(
            "note: {test}: the SME2 backend is on for {:?}; this test checks the declines where it is off",
            T::NEED
        );
        return;
    }
    let seen = if hardware_present(T::NEED) {
        "capped build"
    } else {
        "no hardware"
    };
    let mut calls = 0;
    for entry in entries() {
        for trans in TRANSES {
            let label = format!("{test} ({seen}): {entry:?} {trans:?}");
            let p = Problem::<T>::new(trans, 300, 300, 0.75, 0.0, 5);
            assert_not_on_sme2(entry, &p, &label);
            calls += 1;
        }
    }
    note_declined(test);
    println!("{test}: the GEMV hooks declined all {calls} calls ({seen})");
}

#[test]
fn test_sme2_gemv_hook_declines_without_backend_f32() {
    declines_without_backend::<f32>("test_sme2_gemv_hook_declines_without_backend_f32");
}

#[test]
fn test_sme2_gemv_hook_declines_without_backend_f64() {
    declines_without_backend::<f64>("test_sme2_gemv_hook_declines_without_backend_f64");
}
