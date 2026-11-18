//! A grid of public GEMV calls with the SME2 backend live, each compared
//! with the same call made while the calling thread holds
//! `test_support::force_off` (the NEON kernels' result), bit for bit except
//! NaN sign and payload.
//!
//! What the other GEMV hook tests keep fixed: A as a view into a larger
//! matrix (an offset start, a leading dimension that is not a multiple of
//! the vector length), `x` and `y` as unaligned slices inside larger
//! buffers (whose other elements must stay untouched), the parallel path of
//! `gemv_with_par` split over 1, 2 and 3 SME units with uneven pieces, and
//! values that the promise does not excuse (signed zeros from exact
//! cancellation, subnormal products, overflow) under the caller's
//! floating-point modes (the three directed rounding modes, flush-to-zero,
//! default NaN), which the SME2 GEMV must honour as the NEON kernels do.
//!
//! Each call first decides whether it must run on SME2 (the crossover, the
//! element type and the unit column stride of A) and asserts the decision
//! counter accordingly.

use oxiblas_core::parallel::Par;
use oxiblas_matrix::MatRef;

#[cfg(feature = "parallel")]
use super::sme2::{Switches, override_switches};
use super::sme2::{crossover, take_last_run};
use super::{GemvTrans, gemv, gemv_with_par};
use crate::level3::sme2::args::{SmeElement, same_result};
#[cfg(feature = "parallel")]
use crate::level3::sme2::detect::override_units;
use crate::level3::sme2::test_support::{
    Need, backend_or_skip, decisions, force_off, fpcr_read, fpcr_write, sme_state,
};

/// The element types of these tests.
trait Elem: SmeElement + oxiblas_core::scalar::Field + Send + Sync + core::fmt::Debug {
    /// The hardware the SME2 GEMV of the type needs.
    const NEED: Need;
    /// Binary exponent that puts the products of two scaled values in the
    /// subnormal range.
    const SUBNORMAL_SHIFT: i32;
    /// The largest finite value, as `f64`.
    const MAX: f64;
    /// `x` rounded to the type.
    fn of(x: f64) -> Self;
    /// The value as `f64`.
    fn to_f64(self) -> f64;
}

impl Elem for f32 {
    const NEED: Need = Need::F32;
    const SUBNORMAL_SHIFT: i32 = 70;
    const MAX: f64 = f32::MAX as f64;
    fn of(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Elem for f64 {
    const NEED: Need = Need::F64;
    const SUBNORMAL_SHIFT: i32 = 530;
    const MAX: f64 = f64::MAX;
    fn of(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// A signalling NaN of the type: the guard value around `x`, `y` and A.
fn sentinel<T: Elem>() -> T {
    if core::mem::size_of::<T>() == 4 {
        T::of(f64::from(f32::from_bits(0x7fa0_0000)))
    } else {
        T::of(f64::from_bits(0x7ff4_0000_0000_0000))
    }
}

/// Data patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// Random values in `[-1, 1)`.
    Random,
    /// Multiples of 1/4 (exact products), every product of an odd depth
    /// index cancelling the one before it exactly, some signed zeros in A
    /// and `y`.
    Zeros,
    /// Products (and `y`) in the subnormal range.
    Subnormal,
    /// A near the largest finite value: sums overflow.
    Overflow,
}

/// One problem: A as a view at `(row0, col0)` of a column-major buffer with
/// leading dimension `ld`, and `x` / `y` at offset 1 of guarded buffers.
#[derive(Debug, Clone)]
struct Problem<T> {
    /// Transpose.
    trans: GemvTrans,
    /// The stored A of the call is `m × n`.
    dims: (usize, usize),
    /// The buffer of A.
    a: Vec<T>,
    /// Its leading dimension.
    ld: usize,
    /// Offset of `A(0, 0)` in the buffer.
    a_offset: usize,
    /// `x` with one guard element before and after.
    x: Vec<T>,
    /// `y` with one guard element before and after.
    y: Vec<T>,
    /// alpha and beta.
    scalars: (T, T),
}

impl<T: Elem> Problem<T> {
    /// A problem of `pattern` with data from `seed`.
    fn new(trans: GemvTrans, m: usize, n: usize, scalars: (f64, f64), pattern: Pattern) -> Self {
        let mut state = (m * 7919 + n * 104_729 + 17) as u64 | 1;
        let mut value = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        };
        let (row0, col0) = (3, 2);
        let ld = m + row0 + 5;
        let a_offset = row0 + col0 * ld;
        let mut a = vec![sentinel::<T>(); ld * (n + col0 + 1)];
        let (depth_len, out_len) = match trans {
            GemvTrans::NoTrans => (n, m),
            GemvTrans::Trans | GemvTrans::ConjTrans => (m, n),
        };
        let shift = T::SUBNORMAL_SHIFT;
        for j in 0..n {
            for i in 0..m {
                let x = value();
                let depth = if trans == GemvTrans::NoTrans { j } else { i };
                let v = match pattern {
                    Pattern::Random => x,
                    Pattern::Zeros if (i + 2 * j) % 7 == 3 => {
                        if (i + j).is_multiple_of(2) {
                            0.0
                        } else {
                            -0.0
                        }
                    }
                    Pattern::Zeros => (x * 8.0).round() / 4.0,
                    Pattern::Subnormal => x * 2_f64.powi(-shift),
                    Pattern::Overflow => {
                        let sign = if (i + j).is_multiple_of(5) { -1.0 } else { 1.0 };
                        T::MAX * (0.25 + 0.25 * x.abs()) * sign
                    }
                };
                // Zeros: the odd depth index repeats the even one negated.
                let v = if pattern == Pattern::Zeros && depth % 2 == 1 {
                    let (pi, pj) = if trans == GemvTrans::NoTrans {
                        (i, j - 1)
                    } else {
                        (i - 1, j)
                    };
                    -a[a_offset + pi + pj * ld].to_f64()
                } else {
                    v
                };
                a[a_offset + i + j * ld] = T::of(v);
            }
        }
        let mut x = vec![sentinel::<T>(); depth_len + 2];
        for p in 0..depth_len {
            let v = value();
            x[p + 1] = T::of(match pattern {
                Pattern::Random => v,
                // Pairs equal: with A negated, the products cancel.
                Pattern::Zeros if p % 2 == 1 => x[p].to_f64(),
                Pattern::Zeros => (v * 8.0).round() / 4.0,
                Pattern::Subnormal => v * 2_f64.powi(-shift),
                Pattern::Overflow => 0.5 + 0.5 * v.abs(),
            });
        }
        let beta = scalars.1;
        let mut y = vec![sentinel::<T>(); out_len + 2];
        for q in 0..out_len {
            let v = value();
            y[q + 1] = T::of(if beta == 0.0 {
                f64::NAN
            } else {
                match pattern {
                    Pattern::Random => v,
                    Pattern::Zeros if q.is_multiple_of(3) => {
                        if q.is_multiple_of(2) {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    Pattern::Zeros => (v * 8.0).round() / 4.0,
                    Pattern::Subnormal => v * 2_f64.powi(-2 * shift),
                    Pattern::Overflow => T::MAX * 0.5 * v,
                }
            });
        }
        Self {
            trans,
            dims: (m, n),
            a,
            ld,
            a_offset,
            x,
            y,
            scalars: (T::of(scalars.0), T::of(scalars.1)),
        }
    }

    /// The view of A.
    fn a(&self) -> MatRef<'_, T> {
        let (m, n) = self.dims;
        match MatRef::from_strided(&self.a[self.a_offset..], m, n, self.ld, 1) {
            Some(view) => view,
            None => panic!("A view of {m}x{n}"),
        }
    }
}

/// What one call left and recorded.
#[derive(Debug)]
struct Out<T> {
    /// `y` with its guards, after the call.
    y: Vec<T>,
    /// Decisions of the calling thread.
    decisions: u64,
    /// The SME2 GEMV ran (its run record).
    ran: bool,
    /// Parts of the SME2 run, every one of which ran on SME2.
    parts: Option<(usize, bool)>,
}

/// `p` through `gemv` (`par == None`) or `gemv_with_par`, with the caller
/// FPCR `fpcr`, the backend live or off.
fn call<T: Elem>(p: &Problem<T>, par: Option<Par>, fpcr: u64, off: bool) -> Out<T> {
    let mut y = p.y.clone();
    let (alpha, beta) = p.scalars;
    let len = y.len();
    let _guard = off.then(force_off);
    let _ = take_last_run();
    let decided = decisions();
    let saved = fpcr_read();
    // SAFETY: a test thread; only the GEMV call runs under the caller mode,
    // and the thread's own FPCR is restored right after it.
    unsafe { fpcr_write(fpcr) };
    {
        let x = &p.x[1..p.x.len() - 1];
        let y = &mut y[1..len - 1];
        match par {
            None => gemv(p.trans, alpha, p.a(), x, beta, y),
            Some(par) => gemv_with_par(p.trans, alpha, p.a(), x, beta, y, par),
        }
    }
    // SAFETY: the thread's own value from before the call.
    unsafe { fpcr_write(saved) };
    let run = take_last_run();
    Out {
        y,
        decisions: decisions() - decided,
        ran: run.is_some(),
        parts: run.map(|run| (run.parts.len(), run.parts.iter().all(|part| part.sme2))),
    }
}

/// `got` equals `want`: the elements of `y` bitwise or NaN in both, the
/// guards untouched.
fn same_y<T: Elem>(got: &Out<T>, want: &Out<T>) -> Option<String> {
    let last = got.y.len() - 1;
    for (index, (g, w)) in got.y.iter().zip(&want.y).enumerate() {
        let same = if index == 0 || index == last {
            g.bits() == sentinel::<T>().bits() && w.bits() == g.bits()
        } else {
            same_result(*g, *w)
        };
        if !same {
            return Some(format!(
                "y[{}] is {:#x}, the NEON call gives {:#x}",
                index as isize - 1,
                g.bits(),
                w.bits()
            ));
        }
    }
    None
}

/// The entry points: `gemv`, `gemv_with_par` sequential and parallel.
fn entries() -> Vec<Option<Par>> {
    #[cfg(feature = "parallel")]
    let entries = vec![
        None,
        Some(Par::Seq),
        Some(Par::Rayon),
        Some(Par::RayonWith(3)),
    ];
    #[cfg(not(feature = "parallel"))]
    let entries = vec![None, Some(Par::Seq)];
    entries
}

/// The three transposes.
const TRANSES: [GemvTrans; 3] = [GemvTrans::NoTrans, GemvTrans::Trans, GemvTrans::ConjTrans];

/// Shapes around the crossover and on the parallel path, with the views
/// and slices of [`Problem`]: SME2 runs exactly where the crossover says,
/// with the NEON bits, and leaves the guards of `x`, `y` and A untouched.
fn views_match_neon<T: Elem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let shapes = [
        (63, 400),
        (64, 255),
        (64, 256),
        (65, 253),
        (127, 129),
        (129, 127),
        (2000, 15),
        (2000, 16),
        (2000, 17),
        (300, 301),
        (513, 260),
    ];
    let scalars = [(1.0, 0.0), (0.75, 1.0), (-1.25, -0.5), (2.0, -0.0)];
    let mut problems = Vec::new();
    let mut ran = 0;
    let mut index = 0;
    for par in entries() {
        for trans in TRANSES {
            for &(m, n) in &shapes {
                let s = scalars[index % scalars.len()];
                index += 1;
                let p = Problem::<T>::new(trans, m, n, s, Pattern::Random);
                let label = format!("{test}: {par:?} {trans:?} {m}x{n} {s:?}");
                let want = call(&p, par, 0, true);
                let got = call(&p, par, 0, false);
                let expected = crossover(trans != GemvTrans::NoTrans, m, n);
                ran += usize::from(got.ran);
                if got.decisions != u64::from(expected) || got.ran != expected {
                    problems.push(format!(
                        "{label}: decisions {} ran {} (crossover says {expected})",
                        got.decisions, got.ran
                    ));
                }
                if let Some(parts) = got.parts
                    && !parts.1
                {
                    problems.push(format!("{label}: a part ran on NEON"));
                }
                if let Some(problem) = same_y(&got, &want) {
                    problems.push(format!("{label}: {problem}"));
                }
                if want.decisions != 0 {
                    problems.push(format!("{label}: decided under force_off"));
                }
            }
        }
    }
    assert_eq!(sme_state(), Some((0, 0)), "{test}: SME state");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    assert!(ran > 0, "{test}: nothing ran on SME2");
}

#[test]
fn test_sme2_gemv_hook_grid_views_match_neon_f32() {
    views_match_neon::<f32>("test_sme2_gemv_hook_grid_views_match_neon_f32");
}

#[test]
fn test_sme2_gemv_hook_grid_views_match_neon_f64() {
    views_match_neon::<f64>("test_sme2_gemv_hook_grid_views_match_neon_f64");
}

/// The parallel path split over 1, 2 and 3 SME units (overridden on the
/// calling thread, with the split threshold lowered), on shapes whose
/// pieces are uneven: every part runs on SME2 and the result is the NEON
/// bits of the same parallel call.
#[cfg(feature = "parallel")]
fn uneven_parts_match_neon<T: Elem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let shapes = [(257, 301), (301, 257), (1000, 70), (70, 1000)];
    let mut problems = Vec::new();
    for units in [1, 2, 3] {
        let _units = override_units(units);
        let _switches = override_switches(Switches {
            par_min_bytes: Some(1),
            ..Switches::default()
        });
        for trans in TRANSES {
            for &(m, n) in &shapes {
                let label = format!("{test}: {units} units {trans:?} {m}x{n}");
                let p = Problem::<T>::new(trans, m, n, (0.5, 0.25), Pattern::Random);
                let par = Some(Par::RayonWith(units));
                let want = call(&p, par, 0, true);
                let got = call(&p, par, 0, false);
                match got.parts {
                    Some((parts, true)) if parts == units => {}
                    other => problems.push(format!("{label}: parts {other:?}")),
                }
                if let Some(problem) = same_y(&got, &want) {
                    problems.push(format!("{label}: {problem}"));
                }
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_hook_grid_uneven_parts_match_neon_f32() {
    uneven_parts_match_neon::<f32>("test_sme2_gemv_hook_grid_uneven_parts_match_neon_f32");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_hook_grid_uneven_parts_match_neon_f64() {
    uneven_parts_match_neon::<f64>("test_sme2_gemv_hook_grid_uneven_parts_match_neon_f64");
}

/// The caller FPCR modes: the default, the three directed rounding modes,
/// flush-to-zero (alone and with rounding towards minus infinity) and
/// default NaN.
const CALLER_MODES: [(&str, u64); 7] = [
    ("default", 0),
    ("round to +inf", 0b01 << 22),
    ("round to -inf", 0b10 << 22),
    ("round to zero", 0b11 << 22),
    ("flush to zero", 1 << 24),
    ("flush to zero, round to -inf", (1 << 24) | (0b10 << 22)),
    ("default NaN", 1 << 25),
];

/// Every pattern under every caller mode, both transposes, the four `y`
/// updates: SME2 runs and gives the NEON bits under the same mode.
fn special_values_under_caller_modes<T: Elem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let scalars = [(1.0, 0.0), (1.0, 1.0), (-0.5, -1.0), (0.5, -0.0)];
    let mut problems = Vec::new();
    for (mode_name, mode) in CALLER_MODES {
        for pattern in [Pattern::Zeros, Pattern::Subnormal, Pattern::Overflow] {
            for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
                for &s in &scalars {
                    for &(m, n) in &[(130, 132), (200, 90)] {
                        let label =
                            format!("{test}: {mode_name} {pattern:?} {trans:?} {m}x{n} {s:?}");
                        let p = Problem::<T>::new(trans, m, n, s, pattern);
                        let want = call(&p, None, mode, true);
                        let got = call(&p, None, mode, false);
                        if !got.ran || got.decisions != 1 {
                            problems.push(format!("{label}: did not run on SME2"));
                        }
                        if let Some(problem) = same_y(&got, &want) {
                            problems.push(format!("{label}: {problem}"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} mismatches, first ones:\n{}",
        problems.len(),
        problems
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn test_sme2_gemv_hook_grid_special_values_under_caller_modes_f32() {
    special_values_under_caller_modes::<f32>(
        "test_sme2_gemv_hook_grid_special_values_under_caller_modes_f32",
    );
}

#[test]
fn test_sme2_gemv_hook_grid_special_values_under_caller_modes_f64() {
    special_values_under_caller_modes::<f64>(
        "test_sme2_gemv_hook_grid_special_values_under_caller_modes_f64",
    );
}
