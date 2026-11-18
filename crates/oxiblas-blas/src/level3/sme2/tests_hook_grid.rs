//! A grid of public GEMM calls with the SME2 backend live, each compared
//! with the same call made while the calling thread holds
//! `test_support::force_off` (the NEON build's result), bit for bit except
//! NaN sign and payload.
//!
//! The grid covers what the other hook tests keep fixed: sizes around every
//! boundary of the route, the few-rows route, the ZA panels and the depth
//! step (one block and several), every operand storage (column-major and
//! row-major with and without padding, and fully strided operands that the
//! driver declines), all four transpose combinations, the four epilogue
//! modes, `Par::Seq` and `Par::Rayon`; the driver's own blocking and parts
//! changed through its test tuning (small MC and B blocks, the small class
//! switched on, depth blocks sent to the NEON macro-panel, 1 to 3 parts);
//! and values that the promise does not excuse: signed zeros from exact
//! cancellation and from zero products, subnormal products, overflow, all
//! under the caller's floating-point modes (the three directed rounding
//! modes, flush-to-zero, default NaN), which the SME2 path must honour as
//! the NEON path does.
//!
//! Every comparison decides first whether the call must run on SME2 (the
//! route of the production plan, the driver's class, the operands and the
//! depth step) and asserts the decision counter accordingly, so a call that
//! silently declined cannot pass as a match.

use oxiblas_core::parallel::Par;

#[cfg(feature = "parallel")]
use super::detect::override_units;
use super::detect::queries_here;
use super::driver::{SMALL_CLASS_ON, TuningOverride, classify, kc_accepted, override_tuning};
use super::fpstate::{fpcr_read, fpcr_write};
use super::parallel::take_last_run;
use super::test_support::{backend_or_skip, decisions, force_off, sme_state};
use super::tests_gemm_harness::{CBuf, Operand, Store, TestElem};
use super::tests_hook::{Call, Entry, Run, TRANS_PAIRS, assert_same_c, operands, run_public};
use crate::level3::gemm::{GemmBlocking, gemm_transposed_with_blocking, gemm_transposed_with_par};
use crate::level3::gemm_plan::{DEFAULT_POLICY, GemmPlan, GemmRoute, RouteInputs};
use crate::level3::trsm::Trans;

/// Whether the hook must take `call` (on a thread whose backend is active,
/// with the driver's tuning `small_on` for the small class): the production
/// plan routes it to the blocked route, the driver's class accepts it, both
/// operands have a unit stride and the depth step is usable.
fn must_run<T: TestElem>(call: &Call, small_on: bool) -> bool {
    let (m, n, k) = call.dims;
    let route = default_route::<T>(call);
    let strided =
        matches!(call.stores.0, Store::Strided) || matches!(call.stores.1, Store::Strided);
    let kc = call.entry.kc::<T>(m, n, k);
    matches!(route, GemmRoute::Blocked | GemmRoute::Thin)
        && classify(m, n, k).accepted(small_on)
        && !strided
        && kc_accepted(kc, k)
}

/// Whether the thin route's one GEMV of `call` runs on the SME2 GEMV: the
/// GEMV of `gemm_thin` on the stored large operand (A for `n == 1`, B for
/// `m == 1`), against the SME2 GEMV's crossover restated here: at least 64
/// stored rows, `rows·cols >= 128²`, and for the dot form (GEMV `Trans`) at
/// least 16 columns; alpha is never zero in the grid.
///
/// Only calls of the thin route come here, and the default policy takes
/// that route only with both operands stored column-major
/// (`RouteInputs::thin_in_scope`). Every large operand of the grid's thin
/// shapes has at least two rows and two columns, so a row-major store of
/// it has no unit column stride and never reaches the thin route: the
/// large operand is a column-major store here (asserted), and the GEMV
/// reads it as stored. (A row-major store of the vector operand with one
/// column and no padding has a unit column stride and does reach it.)
fn thin_gemv_on_sme2(call: &Call) -> bool {
    let (m, n, k) = call.dims;
    let (trans_a, trans_b) = call.trans;
    let plain = |t: Trans| t == Trans::NoTrans;
    // (store, stored rows, stored cols, dot form)
    let (store, rows, cols, dot) = if n == 1 {
        // gemv(op_A, A, op(B)[:, 0]): the dot form for op(A) = Aᵀ.
        let (r, c) = if plain(trans_a) { (m, k) } else { (k, m) };
        (call.stores.0, r, c, !plain(trans_a))
    } else {
        // gemv(Trans for op(B) = B, NoTrans for op(B) = Bᵀ, B, op(A)[0, :]).
        let (r, c) = if plain(trans_b) { (k, n) } else { (n, k) };
        (call.stores.1, r, c, plain(trans_b))
    };
    assert!(
        matches!(store, Store::ColMajor { .. }),
        "a thin-route call with the large operand stored as {store:?}"
    );
    call.scalars.0 != 0.0 && rows >= 64 && rows * cols >= 128 * 128 && (!dot || cols >= 16)
}

/// The route of `call`'s production plan under the build's default policy.
fn default_route<T: TestElem>(call: &Call) -> GemmRoute {
    let (m, n, k) = call.dims;
    let plan = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k);
    let (a, b, _) = operands::<T>(call);
    let inputs = RouteInputs::new(
        call.trans.0,
        call.trans.1,
        &a.view(),
        &b.view(),
        m,
        n,
        k,
        plan.ctx.tier(),
    );
    plan.route(&inputs)
}

/// `call` with the backend live and off: the same C (bitwise but for NaN
/// sign and payload, border untouched), and exactly one decision when the
/// call must run on SME2, none otherwise. A call that runs on SME2 ran every
/// part on SME2 (the per-part run record: no part fell back to NEON) and
/// entered the kernels on each part's own thread, unless `neon_depths` (the
/// tuning sends depth blocks to the NEON macro-panel, so a part of one depth
/// block may enter no kernel). Returns whether it ran.
fn check_call<T: TestElem>(call: &Call, small_on: bool, neon_depths: bool, label: &str) -> bool {
    let want = {
        let _off = force_off();
        let run = run_public::<T>(call);
        assert_eq!(run.decisions, 0, "{label}: decided under force_off");
        run
    };
    let got = run_public::<T>(call);
    assert_same_c(&got, &want, label);
    let expected = must_run::<T>(call, small_on);
    if default_route::<T>(call) == GemmRoute::Thin {
        // The thin route of the default policy (aarch64: the products of
        // its measured scope) never reaches the GEMM hook; its one GEMV runs
        // on the SME2 GEMV exactly when that GEMV's crossover takes it (one
        // decision), which `tests_hook_thin.rs` checks case by case.
        assert!(!expected, "{label}: the hook must take a thin product");
        assert_eq!(
            got.decisions,
            u64::from(thin_gemv_on_sme2(call)),
            "{label}: decisions of the thin route's GEMV"
        );
    } else {
        assert_eq!(
            got.decisions,
            u64::from(expected),
            "{label}: decisions (route, class, operands and KC say {expected})"
        );
    }
    assert_eq!(
        got.report.is_some(),
        expected,
        "{label}: run record (expected a run: {expected})"
    );
    if let Some(report) = got.report.as_ref() {
        for (index, neon) in report.neon.iter().enumerate() {
            assert!(!neon, "{label}: part {index} ran on NEON");
            let calls = report
                .blocks
                .get(index)
                .map_or(0, |blocks| blocks.kernel_calls);
            assert!(
                neon_depths || calls > 0,
                "{label}: part {index} entered no kernel"
            );
        }
        assert!(!report.neon.is_empty(), "{label}: a run without parts");
    }
    assert_eq!(
        sme_state(),
        Some((0, 0)),
        "{label}: SME state after the call"
    );
    expected
}

/// `Par::Rayon` (`units == 0`) or `Par::RayonWith(units)` when `parallel`
/// is set and the build has the `parallel` feature, `Par::Seq` otherwise.
fn parallel_par(parallel: bool, units: usize) -> Par {
    #[cfg(feature = "parallel")]
    if parallel {
        return if units == 0 {
            Par::Rayon
        } else {
            Par::RayonWith(units)
        };
    }
    let _ = (parallel, units);
    Par::Seq
}

/// The dimensions of the grid for `T`: the `32³` boundary, the few-rows
/// limit, one row and column around the ZA panels (64 rows, `NR` columns),
/// the depth step of the default blocking (one below, at, one above, and
/// two steps), thin and two-wide products, and one product of several
/// blocks in every dimension.
fn grid_dims<T: TestElem>() -> Vec<(usize, usize, usize)> {
    let nr = T::NR;
    let few = T::FEW_ROWS;
    let kc = Entry::Transposed.kc::<T>(200, 4 * nr, 4096);
    let mut dims = vec![
        (32, 32, 32),
        (32, 32, 33),
        (33, 33, 33),
        (few, 2 * nr + 1, 64),
        (few + 1, 2 * nr + 1, 64),
        (63, nr - 1, 65),
        (64, nr, 64),
        (65, nr + 1, 63),
        (127, 2 * nr - 1, 70),
        (129, 3 * nr + 1, 33),
        (2, 70, 300),
        (70, 2, 300),
        (1, 90, 400),
        (90, 1, 400),
    ];
    for depth in [kc - 1, kc, kc + 1, 2 * kc + 1] {
        dims.push((70, nr + 3, depth));
    }
    dims.push((150, 6 * nr + 3, 2 * kc + 7));
    dims
}

/// The storage pairs of the grid.
const STORES: [(Store, Store); 5] = [
    (Store::ColMajor { pad: 0 }, Store::ColMajor { pad: 0 }),
    (Store::ColMajor { pad: 3 }, Store::RowMajor { pad: 0 }),
    (Store::RowMajor { pad: 1 }, Store::ColMajor { pad: 2 }),
    (Store::RowMajor { pad: 0 }, Store::RowMajor { pad: 5 }),
    (Store::Strided, Store::ColMajor { pad: 1 }),
];

/// The scalar pairs of the grid (rotated over the calls): the four epilogue
/// modes and a negative-zero beta.
const SCALARS: [(f64, f64); 5] = [
    (1.0, 0.0),
    (1.0, 1.0),
    (-0.5, 0.0),
    (0.75, -1.25),
    (-2.0, -0.0),
];

/// The grid through `gemm_transposed_with_par`.
fn grid_matches_neon<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut ran = 0_usize;
    let mut index = 0_usize;
    for (m, n, k) in grid_dims::<T>() {
        for &(trans_a, trans_b) in &TRANS_PAIRS {
            for &(store_a, store_b) in &STORES {
                let (alpha, beta) = SCALARS[index % SCALARS.len()];
                let par = parallel_par(index % 3 == 2, 0);
                let mut call = Call::new(Entry::TransposedWithPar, m, n, k)
                    .trans(trans_a, trans_b)
                    .stores(store_a, store_b)
                    .scalars(alpha, beta)
                    .par(par)
                    .seed(index as u64 + 101);
                if beta == 0.0 {
                    call = call.nan_c();
                }
                if index % 4 == 1 {
                    call = call.non_finite();
                }
                let label = format!(
                    "{test}: {m}x{n}x{k} {trans_a:?}/{trans_b:?} {store_a:?}/{store_b:?} \
                     a {alpha} b {beta} {par:?}"
                );
                ran += usize::from(check_call::<T>(&call, SMALL_CLASS_ON, false, &label));
                index += 1;
            }
        }
    }
    println!("{test}: {ran} of {index} calls ran on SME2");
    assert!(
        ran * 3 > index,
        "{test}: only {ran} of {index} calls ran on SME2"
    );
}

#[test]
fn test_sme2_hook_grid_matches_neon_f32() {
    grid_matches_neon::<f32>("test_sme2_hook_grid_matches_neon_f32");
}

#[test]
fn test_sme2_hook_grid_matches_neon_f64() {
    grid_matches_neon::<f64>("test_sme2_hook_grid_matches_neon_f64");
}

/// The driver's blocking and parts changed through its test tuning (none of
/// which may change a bit): MC and the B block cut down to a few panels,
/// the small class switched on (transposed products up to `32³` that the
/// NEON route blocks), depth blocks sent to the NEON macro-panel, and with
/// `parallel` 1, 2 and 3 parts. Every call still gives the NEON bits.
fn tuned_driver_matches_neon<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let nr = T::NR;
    let kc = Entry::Transposed.kc::<T>(200, 4 * nr, 4096);
    let tunings = [
        (
            "small blocks",
            TuningOverride {
                mc_target: Some(64),
                b_bytes: Some(40 * core::mem::size_of::<T>() * 2 * nr),
                in_place_bytes: Some(0),
                ..TuningOverride::default()
            },
        ),
        (
            "small class on",
            TuningOverride {
                small_class: Some(true),
                ..TuningOverride::default()
            },
        ),
        (
            "NEON depth blocks",
            TuningOverride {
                neon_depths: 0b101,
                ..TuningOverride::default()
            },
        ),
    ];
    let dims = [
        (150, 5 * nr + 3, 2 * kc + 9),
        (40, 3 * nr, 20),
        (25, 9, 64),
        (T::FEW_ROWS, 4 * nr + 1, kc + 3),
    ];
    #[cfg(feature = "parallel")]
    let unit_counts = [1_usize, 2, 3];
    #[cfg(not(feature = "parallel"))]
    let unit_counts = [1_usize];
    let mut index = 0_u64;
    for (what, tuning) in tunings {
        for units in unit_counts {
            for &(m, n, k) in &dims {
                for &(trans_a, trans_b) in &TRANS_PAIRS {
                    let label = format!(
                        "{test}: {what}, {units} units, {m}x{n}x{k} {trans_a:?}/{trans_b:?}"
                    );
                    let call = Call::new(Entry::TransposedWithPar, m, n, k)
                        .trans(trans_a, trans_b)
                        .scalars(0.75, if index.is_multiple_of(2) { 1.0 } else { -0.5 })
                        .par(parallel_par(true, units))
                        .seed(index + 501);
                    index += 1;
                    #[cfg(feature = "parallel")]
                    let _units = override_units(units);
                    let _tuning = override_tuning(TuningOverride {
                        par_min_macs: Some(1),
                        ..tuning
                    });
                    let small_on = tuning.small_class.unwrap_or(SMALL_CLASS_ON);
                    let ran = check_call::<T>(&call, small_on, tuning.neon_depths != 0, &label);
                    if ran && units > 1 && m >= 2 * 64 {
                        let parts = run_public::<T>(&call)
                            .report
                            .map_or(0, |report| report.layout.parts.len());
                        assert!(parts >= 2, "{label}: {parts} parts");
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_hook_grid_tuned_driver_matches_neon_f32() {
    tuned_driver_matches_neon::<f32>("test_sme2_hook_grid_tuned_driver_matches_neon_f32");
}

#[test]
fn test_sme2_hook_grid_tuned_driver_matches_neon_f64() {
    tuned_driver_matches_neon::<f64>("test_sme2_hook_grid_tuned_driver_matches_neon_f64");
}

// ---------------------------------------------------------------------------
// Values the promise does not excuse, under the caller's floating-point modes
// ---------------------------------------------------------------------------

/// The caller FPCR modes of the special-value tests: the default, the three
/// directed rounding modes, flush-to-zero (alone and with rounding towards
/// minus infinity) and default NaN.
pub(super) const CALLER_MODES: [(&str, u64); 7] = [
    ("default", 0),
    ("round to +inf", 0b01 << 22),
    ("round to -inf", 0b10 << 22),
    ("round to zero", 0b11 << 22),
    ("flush to zero", 1 << 24),
    ("flush to zero, round to -inf", (1 << 24) | (0b10 << 22)),
    ("default NaN", 1 << 25),
];

/// Data patterns that produce results the promise does not excuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pattern {
    /// Exact cancellation: every product of row `i` of `op(A)` with column
    /// `j` of `op(B)` is paired with its negative, so the exact sum is zero
    /// (`+0` or `-0` by the rounding mode); some rows and columns are all
    /// signed zeros, and C holds signed zeros.
    Zeros,
    /// Products and sums in the subnormal range (flushed under FZ).
    Subnormal,
    /// Values near the largest finite value: sums overflow (to infinity, or
    /// to the largest finite value when rounding towards zero).
    Overflow,
}

/// The value of element `(r, c)` of a stored operand for `pattern`, from
/// the random value `x` in `[-1, 1)`; `big` is the operand that carries the
/// large values of [`Pattern::Overflow`] (the other one holds values in
/// `[0.5, 1]`).
fn pattern_value<T: TestElem>(pattern: Pattern, big: bool, r: usize, c: usize, x: T) -> T {
    let x = x.to_f64();
    match pattern {
        Pattern::Zeros => {
            if (r + 2 * c) % 7 == 3 {
                T::of(if (r + c).is_multiple_of(2) { 0.0 } else { -0.0 })
            } else {
                // Values with few significant bits: exact products.
                T::of((x * 8.0).round() / 4.0)
            }
        }
        // Every product (and C, scaled by the square) is subnormal.
        Pattern::Subnormal => T::of(x * 2_f64.powi(-T::SUBNORMAL_SHIFT)),
        Pattern::Overflow if big => {
            let max = if core::mem::size_of::<T>() == 4 {
                f64::from(f32::MAX)
            } else {
                f64::MAX
            };
            // Mostly positive, so that sums run past the largest value.
            let sign = if (r + c).is_multiple_of(5) { -1.0 } else { 1.0 };
            T::of(max * (0.25 + 0.25 * x.abs()) * sign)
        }
        Pattern::Overflow => T::of(0.5 + 0.5 * x.abs()),
    }
}

/// Rewrites the stored values of `x` for `pattern` (`big`: see
/// [`pattern_value`]). For [`Pattern::Zeros`] the depth index `2q + 1`
/// becomes a copy of depth index `2q`, negated when `negate`: with A
/// negated and B copied, the products of each depth pair cancel exactly.
/// `depth_in_cols` says whether the depth runs along the stored columns.
fn apply_pattern<T: TestElem>(
    x: &mut Operand<T>,
    pattern: Pattern,
    big: bool,
    depth_in_cols: bool,
    negate: bool,
) {
    for c in 0..x.cols {
        for r in 0..x.rows {
            let at = r * x.col_stride + c * x.row_stride;
            x.data[at] = pattern_value(pattern, big, r, c, x.data[at]);
        }
    }
    let sign = if negate { -1.0 } else { 1.0 };
    if pattern != Pattern::Zeros {
        return;
    }
    if depth_in_cols {
        for c in (1..x.cols).step_by(2) {
            for r in 0..x.rows {
                let from = r * x.col_stride + (c - 1) * x.row_stride;
                let to = r * x.col_stride + c * x.row_stride;
                x.data[to] = T::of(sign * x.data[from].to_f64());
            }
        }
    } else {
        for r in (1..x.rows).step_by(2) {
            for c in 0..x.cols {
                let from = (r - 1) * x.col_stride + c * x.row_stride;
                let to = r * x.col_stride + c * x.row_stride;
                x.data[to] = T::of(sign * x.data[from].to_f64());
            }
        }
    }
}

/// `call` with the operands rewritten for `pattern`, through
/// `gemm_transposed_with_par` (or, with `explicit`, through
/// `gemm_transposed_with_blocking` with that blocking), under the caller
/// FPCR `fpcr`, with the backend live or off.
fn run_pattern<T: TestElem>(
    call: &Call,
    pattern: Pattern,
    explicit: Option<GemmBlocking>,
    fpcr: u64,
    off: bool,
) -> Run<T> {
    let (mut a, mut b, mut c) = operands::<T>(call);
    let (trans_a, trans_b) = call.trans;
    // op(A) = A: the depth runs along A's columns; op(A) = Aᵀ: along its rows.
    apply_pattern(&mut a, pattern, true, trans_a == Trans::NoTrans, true);
    // op(B) = B: the depth runs along B's rows.
    apply_pattern(&mut b, pattern, false, trans_b != Trans::NoTrans, false);
    pattern_c(&mut c, pattern);
    let (alpha, beta) = (T::of(call.scalars.0), T::of(call.scalars.1));
    let _guard = off.then(force_off);
    let _ = take_last_run();
    let (decided, asked) = (decisions(), queries_here());
    let saved = fpcr_read();
    // SAFETY: a test thread; only the GEMM call below runs under the
    // caller's mode, and the thread's own FPCR is restored right after it,
    // before anything else.
    unsafe { fpcr_write(fpcr) };
    {
        let (a, b, c) = (a.view(), b.view(), c.view());
        match explicit {
            Some(blocking) => gemm_transposed_with_blocking(
                trans_a, trans_b, alpha, a, b, beta, c, call.par, &blocking,
            ),
            None => gemm_transposed_with_par(trans_a, trans_b, alpha, a, b, beta, c, call.par),
        }
    }
    // SAFETY: the thread's own value from before the call.
    unsafe { fpcr_write(saved) };
    Run {
        c,
        decisions: decisions() - decided,
        queries: queries_here() - asked,
        report: take_last_run(),
    }
}

/// C for `pattern`: signed zeros and small integers ([`Pattern::Zeros`]),
/// subnormals, or large values; the border is left as it is.
fn pattern_c<T: TestElem>(c: &mut CBuf<T>, pattern: Pattern) {
    for j in 0..c.n {
        for i in 0..c.m {
            let at = c.offset + i + j * c.ldc;
            let x = c.data[at];
            if x.to_f64().is_nan() {
                continue;
            }
            c.data[at] = match pattern {
                Pattern::Zeros => {
                    if (i + j).is_multiple_of(3) {
                        T::of(if j.is_multiple_of(2) { -0.0 } else { 0.0 })
                    } else {
                        pattern_value(pattern, false, i, j, x)
                    }
                }
                Pattern::Subnormal => T::of(x.to_f64() * 2_f64.powi(-2 * T::SUBNORMAL_SHIFT)),
                Pattern::Overflow => pattern_value(pattern, true, i, j, x),
            };
        }
    }
}

/// Whether C holds at least one zero of each sign, one subnormal or one
/// infinity (so the pattern produced what it is for).
fn pattern_seen<T: TestElem>(run: &Run<T>, pattern: Pattern) -> bool {
    let c = &run.c;
    let values = (0..c.n).flat_map(|j| (0..c.m).map(move |i| c.at(i, j).to_f64()));
    match pattern {
        Pattern::Zeros => {
            let (mut plus, mut minus) = (false, false);
            for v in values {
                if v == 0.0 {
                    if v.is_sign_negative() {
                        minus = true;
                    } else {
                        plus = true;
                    }
                }
            }
            plus && minus
        }
        Pattern::Subnormal => values.into_iter().any(|v| v != 0.0 && v.abs() < 1e-30),
        Pattern::Overflow => values.into_iter().any(f64::is_infinite),
    }
}

/// Every pattern, under every caller mode, for the four transposes and the
/// epilogue modes, on a one-block product and on a product of several depth
/// steps (explicit KC 32: the later steps add into C): SME2 runs and gives
/// the NEON bits under the same mode.
fn special_values_under_caller_modes<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (70, 3 * T::NR + 1, 100);
    let explicit = GemmBlocking {
        mc: 128,
        kc: 32,
        nc: 256,
    };
    let scalars = [
        (1.0, 0.0),
        (1.0, 1.0),
        (-0.5, 0.0),
        (-1.0, -1.0),
        (0.5, -0.0),
    ];
    let mut problems = Vec::new();
    let mut seen = [false; 3];
    for (mode_name, mode) in CALLER_MODES {
        for (p_index, pattern) in [Pattern::Zeros, Pattern::Subnormal, Pattern::Overflow]
            .into_iter()
            .enumerate()
        {
            for &(trans_a, trans_b) in &TRANS_PAIRS {
                for (s_index, &(alpha, beta)) in scalars.iter().enumerate() {
                    if (s_index + p_index) % 2 == 1 && trans_a != trans_b {
                        continue;
                    }
                    for blocking in [None, Some(explicit)] {
                        let call = Call::new(Entry::TransposedWithPar, m, n, k)
                            .trans(trans_a, trans_b)
                            .scalars(alpha, beta)
                            .seed((p_index * 10 + s_index) as u64 + 900);
                        let want = run_pattern::<T>(&call, pattern, blocking, mode, true);
                        let got = run_pattern::<T>(&call, pattern, blocking, mode, false);
                        let label = format!(
                            "{test}: {mode_name}, {pattern:?}, {trans_a:?}/{trans_b:?}, \
                             a {alpha} b {beta}, explicit KC {}",
                            blocking.is_some()
                        );
                        if got.decisions != 1 || got.report.is_none() {
                            problems.push(format!("{label}: did not run on SME2"));
                            continue;
                        }
                        seen[p_index] |= pattern_seen(&got, pattern);
                        let mismatch = got.c.data.iter().zip(&want.c.data).enumerate().find(
                            |&(index, (g, w))| {
                                if got.c.in_block(index) {
                                    !super::args::same_result(*g, *w)
                                } else {
                                    g.bits() != w.bits() || g.bits() != T::SENTINEL.bits()
                                }
                            },
                        );
                        if let Some((index, (g, w))) = mismatch {
                            problems.push(format!(
                                "{label}: element {index} is {:#x}, the NEON call gives {:#x}",
                                g.bits(),
                                w.bits()
                            ));
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
    assert_eq!(
        seen, [true; 3],
        "{test}: a pattern produced nothing special"
    );
}

#[test]
fn test_sme2_hook_grid_special_values_under_caller_modes_f32() {
    special_values_under_caller_modes::<f32>(
        "test_sme2_hook_grid_special_values_under_caller_modes_f32",
    );
}

#[test]
fn test_sme2_hook_grid_special_values_under_caller_modes_f64() {
    special_values_under_caller_modes::<f64>(
        "test_sme2_hook_grid_special_values_under_caller_modes_f64",
    );
}
