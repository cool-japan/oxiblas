//! Tests of the GEMM hook's declines: the calls that the NEON build does
//! not run on the blocked route of a production plan never reach SME2.
//!
//! For each of them SME2 does not run (no run record), the decision counter
//! of the calling thread does not move, and C equals the same call with the
//! backend forced off, bit for bit: the small routes at the `32³` boundary
//! (one below, at, one above), thin products, forced routes (the forced
//! blocked route too), forced pack scopes, a given micro-kernel, complex
//! element types, a held `force_off`, a calling thread that is not ready
//! (the pretend flag, and the real states of `test_sme_state.rs`), and
//! operands the driver declines. A plan whose context is lowered to the
//! portable tier (and that forces nothing else), on the blocked route and
//! on the thin route, does not run SME2 and does not ask the detection (its
//! result equals the portable kernels', which here equal NEON's, so the
//! counters carry that test). Where the hook is not reached at all (the
//! routes other than the blocked one, a forced plan, a complex type) the
//! detection is not even asked: the detection-call counter of the thread
//! (`detect::queries_here`) does not move.
//!
//! The gated tests run where the backend is on (otherwise there is nothing
//! to decline); the ungated `declines_without_backend` tests cover the
//! hook where it is off and write the `declined` marker.

use num_complex::{Complex32, Complex64};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

use super::detect::{queries_here, set_pretend_not_ready};
use super::parallel::take_last_run;
use super::test_sme_state::{Tpidr2Set, ZaStorageOn};
use super::test_support::{
    backend_or_skip, backend_present, decisions, force_off, hardware_present, note_declined,
};
use super::tests_gemm_harness::{Store, TestElem};
use super::tests_hook::{
    Call, Entry, Run, TRANS_PAIRS, assert_same_c, operands, run_public, run_reference,
};
use crate::level3::gemm::{BlockingSource, gemm_planned, gemm_transposed};
use crate::level3::gemm_plan::{
    CANDIDATE_POLICY, DEFAULT_POLICY, GemmPlan, GemmPolicy, GemmRoute, LEGACY_POLICY, PackScope,
    RouteInputs, SimdTier, ThinPolicy,
};
use crate::level3::trsm::Trans;

/// `call` declines: no decision, no SME2 run, C as the NEON call leaves it.
fn assert_declines<T: TestElem>(call: &Call, label: &str) -> Run<T> {
    let want = run_reference::<T>(call);
    let got = run_public::<T>(call);
    assert_eq!(got.decisions, 0, "{label}: decided");
    assert!(got.report.is_none(), "{label}: SME2 ran");
    assert_same_c(&got, &want, label);
    got
}

/// `call` made through `gemm_planned` with `plan` (the operands and scalars
/// of `call`, the default blocking source).
fn run_planned<T: TestElem>(call: &Call, plan: &GemmPlan<T>) -> Run<T> {
    let (a, b, mut c) = operands::<T>(call);
    let (alpha, beta) = (T::of(call.scalars.0), T::of(call.scalars.1));
    let _ = take_last_run();
    let (decided, asked) = (decisions(), queries_here());
    gemm_planned(
        call.trans.0,
        call.trans.1,
        alpha,
        a.view(),
        b.view(),
        beta,
        c.view(),
        call.par,
        BlockingSource::Default,
        plan,
    );
    Run {
        c,
        decisions: decisions() - decided,
        queries: queries_here() - asked,
        report: take_last_run(),
    }
}

/// The small routes at their boundary: a public call of `m·n·k` one below
/// and at `32³` takes a small route, never asks the detection and runs no
/// SME2; one above takes the blocked route and runs on SME2.
fn small_route_boundary<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for &(trans_a, trans_b) in &TRANS_PAIRS {
        for (k, above) in [(31, false), (32, false), (33, true)] {
            let label = format!("{test}: 32x32x{k} {trans_a:?}/{trans_b:?}");
            let call = Call::new(Entry::Transposed, 32, 32, k).trans(trans_a, trans_b);
            if above {
                let got = run_public::<T>(&call);
                assert_eq!(got.decisions, 1, "{label}: not on SME2");
                assert_same_c(&got, &run_reference::<T>(&call), &label);
            } else {
                let got = assert_declines::<T>(&call, &label);
                assert_eq!(
                    got.queries, 0,
                    "{label}: the small route asked the detection"
                );
            }
        }
    }
}

#[test]
fn test_sme2_hook_declines_small_route_boundary_f32() {
    small_route_boundary::<f32>("test_sme2_hook_declines_small_route_boundary_f32");
}

#[test]
fn test_sme2_hook_declines_small_route_boundary_f64() {
    small_route_boundary::<f64>("test_sme2_hook_declines_small_route_boundary_f64");
}

/// Thin products (`m == 1` or `n == 1`) on the routes of 0.2.3 (the
/// production plan of `LEGACY_POLICY`, whose thin route is off): up to
/// `32³` the small routes (no detection call), above it the blocked route,
/// where the driver declines the thin class. A public call of the build's
/// default policy does the same where that policy does not take the thin
/// route; where it does (the default of aarch64 for the products of its
/// measured scope), the GEMM hook is not reached either (no driver run, the
/// bits of the call with the backend off), and the route's GEMV may run on
/// the SME2 GEMV (`tests_hook_thin.rs`).
fn thin_products<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for &(trans_a, trans_b) in &TRANS_PAIRS {
        // m·n·k one below, at and one above 32³ (with m or n equal to 1).
        for (m, n, k) in [
            (1, 1024, 31),
            (1, 1024, 32),
            (1, 1024, 33),
            (1024, 1, 31),
            (1024, 1, 32),
            (1024, 1, 33),
            (1, 300, 300),
            (300, 1, 300),
        ] {
            let label = format!("{test}: {m}x{n}x{k} {trans_a:?}/{trans_b:?}");
            let call = Call::new(Entry::Transposed, m, n, k).trans(trans_a, trans_b);
            let small = m * n * k <= 32 * 32 * 32;
            // The routes of 0.2.3.
            let legacy = GemmPlan::<T>::for_problem(LEGACY_POLICY, m, n, k);
            let got = run_planned::<T>(&call, &legacy);
            let want = {
                let _off = force_off();
                run_planned::<T>(&call, &legacy)
            };
            assert_eq!(got.decisions, 0, "{label}: legacy plan decided");
            assert!(got.report.is_none(), "{label}: legacy plan ran on SME2");
            assert_same_c(&got, &want, &label);
            if small {
                assert_eq!(
                    got.queries, 0,
                    "{label}: the small route asked the detection"
                );
            }
            // The public call under the build's default policy.
            let (a, b, _) = operands::<T>(&call);
            let default = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k);
            let inputs = RouteInputs::new(
                trans_a,
                trans_b,
                &a.view(),
                &b.view(),
                m,
                n,
                k,
                default.ctx.tier(),
            );
            if default.route(&inputs) == GemmRoute::Thin {
                let got = run_public::<T>(&call);
                assert!(got.report.is_none(), "{label}: the GEMM driver ran");
                assert_same_c(&got, &run_reference::<T>(&call), &label);
            } else {
                let got = assert_declines::<T>(&call, &label);
                if small {
                    assert_eq!(
                        got.queries, 0,
                        "{label}: the small route asked the detection"
                    );
                }
            }
        }
    }
}

#[test]
fn test_sme2_hook_declines_thin_products_f32() {
    thin_products::<f32>("test_sme2_hook_declines_thin_products_f32");
}

#[test]
fn test_sme2_hook_declines_thin_products_f64() {
    thin_products::<f64>("test_sme2_hook_declines_thin_products_f64");
}

/// A plan that forces a route (the blocked route too), a pack scope or a
/// micro-kernel runs what it forced: on a shape that the hook takes from a
/// production plan, no decision, no detection call, and the bits of the
/// same forced call with the backend off.
fn forced_plans<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (70, 3 * T::NR + 1, 65);
    for &(trans_a, trans_b) in &TRANS_PAIRS {
        let call = Call::new(Entry::Transposed, m, n, k).trans(trans_a, trans_b);
        let base = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k);
        // The production plan itself runs on SME2 (so the shape is one the
        // hook takes).
        let production = run_planned::<T>(&call, &base);
        assert_eq!(
            production.decisions, 1,
            "{test}: production plan {trans_a:?}/{trans_b:?}"
        );
        let mut plans = vec![
            (
                "forced blocked route",
                base.with_forced_route(GemmRoute::Blocked),
            ),
            (
                "forced small-trans route",
                base.with_forced_route(GemmRoute::SmallTrans),
            ),
            (
                "forced small-copy route",
                base.with_forced_route(GemmRoute::SmallTransCopy),
            ),
            (
                "forced shared scope",
                base.with_forced_scope(PackScope::Shared),
            ),
            (
                "forced per-tile scope",
                base.with_forced_scope(PackScope::PerTile),
            ),
            ("given kernel", base.with_kernel(T::micro_kernel_dispatch())),
            ("GemmPlan::new", GemmPlan::<T>::new(DEFAULT_POLICY)),
        ];
        if (trans_a, trans_b) == (Trans::NoTrans, Trans::NoTrans) {
            plans.push((
                "forced small route",
                base.with_forced_route(GemmRoute::Small),
            ));
        }
        for (what, plan) in plans {
            let label = format!("{test}: {what} {trans_a:?}/{trans_b:?}");
            let want = {
                let _off = force_off();
                run_planned::<T>(&call, &plan)
            };
            let got = run_planned::<T>(&call, &plan);
            assert_eq!(got.decisions, 0, "{label}: decided");
            assert_eq!(got.queries, 0, "{label}: asked the detection");
            assert!(got.report.is_none(), "{label}: SME2 ran");
            assert_same_c(&got, &want, &label);
        }
    }
}

#[test]
fn test_sme2_hook_declines_forced_plans_f32() {
    forced_plans::<f32>("test_sme2_hook_declines_forced_plans_f32");
}

#[test]
fn test_sme2_hook_declines_forced_plans_f64() {
    forced_plans::<f64>("test_sme2_hook_declines_forced_plans_f64");
}

/// The route `plan` takes for `call`.
fn route_of<T: TestElem>(call: &Call, plan: &GemmPlan<T>) -> GemmRoute {
    let (a, b, _) = operands::<T>(call);
    let (m, n, k) = call.dims;
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

/// A plan as `for_problem` builds it, whose context is then lowered to the
/// portable tier (a forced call: it asks for the portable kernels), does
/// not run SME2. For a blocked-route shape under the default policy and two
/// thin-route shapes under the candidate policy with the whole thin route
/// (`ThinPolicy::All`: the candidate's own scoped thin route needs a tier
/// with SIMD thin kernels, so its lowered plan takes the blocked route,
/// which is asserted too), each of which the same plan with the detected
/// tier runs on SME2: no
/// decision, no detection call, no SME2 run, and the bits of the same
/// lowered call with the backend off. The bits alone cannot tell the
/// portable kernels from SME2 or NEON here (the printout counts the
/// elements in which the portable result differs from NEON: 0 so far);
/// the counters can.
///
/// Every case is checked, also after one failed (its message is printed
/// when it fails), and the test then names all that failed: a defect that
/// shows in one route does not hide one in another.
fn lowered_tier_plans<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let col = Store::ColMajor { pad: 0 };
    let whole_thin = GemmPolicy {
        thin: ThinPolicy::All,
        ..CANDIDATE_POLICY
    };
    let cases = [
        (
            "blocked route",
            DEFAULT_POLICY,
            Call::new(Entry::Transposed, 70, 3 * T::NR + 1, 65),
            GemmRoute::Blocked,
        ),
        (
            "thin route, n == 1",
            whole_thin,
            Call::new(Entry::Transposed, 300, 1, 200)
                .stores(col, col)
                .par(Par::Seq),
            GemmRoute::Thin,
        ),
        (
            "thin route, m == 1",
            whole_thin,
            Call::new(Entry::Transposed, 1, 200, 300)
                .stores(col, col)
                .par(Par::Seq),
            GemmRoute::Thin,
        ),
    ];
    let mut failed = Vec::new();
    for (what, policy, call, route) in cases {
        let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lowered_tier_case::<T>(test, what, policy, &call, route);
        }));
        if checked.is_err() {
            failed.push(what);
        }
    }
    assert!(
        failed.is_empty(),
        "{test}: the lowered plans of these cases failed (messages above): {failed:?}"
    );
}

/// One case of [`lowered_tier_plans`].
fn lowered_tier_case<T: TestElem>(
    test: &str,
    what: &str,
    policy: GemmPolicy,
    call: &Call,
    route: GemmRoute,
) {
    let (m, n, k) = call.dims;
    let label = format!("{test}: {what} {m}x{n}x{k}");
    let detected = GemmPlan::<T>::for_problem(policy, m, n, k);
    let mut lowered = detected;
    lowered.ctx = lowered.ctx.with_portable_tier();
    let tier = lowered.ctx.tier();
    assert_eq!(
        tier,
        SimdTier::Portable,
        "{label}: the lowered tier is {tier:?}"
    );
    let taken = route_of(call, &lowered);
    assert_eq!(
        taken, route,
        "{label}: the lowered plan takes the route {taken:?}"
    );
    let taken = route_of(call, &detected);
    assert_eq!(
        taken, route,
        "{label}: the detected plan takes the route {taken:?}"
    );
    if route == GemmRoute::Thin {
        // The candidate's scoped thin route: thin on the detected (NEON)
        // tier, the blocked route of 0.2.3 on the portable tier.
        let mut scoped = GemmPlan::<T>::for_problem(CANDIDATE_POLICY, m, n, k);
        assert_eq!(route_of(call, &scoped), GemmRoute::Thin, "{label}: scoped");
        scoped.ctx = scoped.ctx.with_portable_tier();
        assert_eq!(
            route_of(call, &scoped),
            GemmRoute::Blocked,
            "{label}: scoped, portable tier"
        );
    }
    // The same plan with the detected tier runs on SME2 (so the shape is
    // one the hooks take).
    let production = run_planned::<T>(call, &detected);
    assert_eq!(
        production.decisions, 1,
        "{label}: the plan with the detected tier made {} decisions",
        production.decisions
    );
    let want = {
        let _off = force_off();
        run_planned::<T>(call, &lowered)
    };
    let got = run_planned::<T>(call, &lowered);
    assert_eq!(
        got.decisions, 0,
        "{label}: the lowered plan made {} decisions",
        got.decisions
    );
    assert_eq!(
        got.queries, 0,
        "{label}: the lowered plan asked the detection {} times",
        got.queries
    );
    assert!(
        got.report.is_none(),
        "{label}: SME2 ran for the lowered plan: {:?}",
        got.report
    );
    assert_same_c(&got, &want, &label);
    let neon = {
        let _off = force_off();
        run_planned::<T>(call, &detected)
    };
    let differ = want
        .c
        .data
        .iter()
        .zip(&neon.c.data)
        .filter(|(portable, neon)| portable.bits() != neon.bits())
        .count();
    println!("{label}: the portable kernels' result differs from NEON in {differ} elements");
}

#[test]
fn test_sme2_hook_declines_lowered_tier_plans_f32() {
    lowered_tier_plans::<f32>("test_sme2_hook_declines_lowered_tier_plans_f32");
}

#[test]
fn test_sme2_hook_declines_lowered_tier_plans_f64() {
    lowered_tier_plans::<f64>("test_sme2_hook_declines_lowered_tier_plans_f64");
}

/// A held `force_off`, a thread that pretends not to be ready, a thread
/// with ZA storage on and one with TPIDR2_EL0 set, and operands that the
/// driver declines (a fully strided A or B): no SME2, NEON bits.
fn calling_thread_and_operands<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (70, 3 * T::NR + 1, 65);
    for &(trans_a, trans_b) in &TRANS_PAIRS {
        let call = Call::new(Entry::TransposedWithPar, m, n, k).trans(trans_a, trans_b);
        let label = format!("{test}: {trans_a:?}/{trans_b:?}");
        let want = run_reference::<T>(&call);
        // The shape runs on SME2 from a ready thread.
        assert_eq!(run_public::<T>(&call).decisions, 1, "{label}: not on SME2");

        // A held force_off (run_reference asserts the decline itself).
        {
            let _off = force_off();
            let _nested = force_off();
            let got = run_public::<T>(&call);
            assert_eq!(got.decisions, 0, "{label}: decided under force_off");
            assert_same_c(&got, &want, &label);
        }

        // The pretend flag.
        let previous = set_pretend_not_ready(true);
        let pretended = run_public::<T>(&call);
        set_pretend_not_ready(previous);
        assert_eq!(pretended.decisions, 0, "{label}: decided while not ready");
        assert!(
            pretended.report.is_none(),
            "{label}: SME2 ran while not ready"
        );
        assert_same_c(&pretended, &want, &label);

        // The real states (Par::Seq: no thread may be created while a guard
        // lives; the verdict was settled by the gate above).
        let seq = call.par(Par::Seq);
        let want_seq = run_reference::<T>(&seq);
        let za = {
            let _za = ZaStorageOn::new().expect("SME present");
            run_public::<T>(&seq)
        };
        let tpidr2 = {
            let _tpidr2 = Tpidr2Set::new().expect("SME present");
            run_public::<T>(&seq)
        };
        for (state, got) in [("ZA on", za), ("TPIDR2 set", tpidr2)] {
            assert_eq!(got.decisions, 0, "{label}: decided with {state}");
            assert!(got.report.is_none(), "{label}: SME2 ran with {state}");
            assert_same_c(&got, &want_seq, &format!("{label} ({state})"));
        }

        // Operands the driver declines: neither stride of A (or B) is 1.
        for stores in [
            (Store::Strided, Store::ColMajor { pad: 0 }),
            (Store::ColMajor { pad: 0 }, Store::Strided),
        ] {
            let strided = call.stores(stores.0, stores.1);
            assert_declines::<T>(&strided, &format!("{label}: {stores:?}"));
        }
    }
}

#[test]
fn test_sme2_hook_declines_calling_thread_and_operands_f32() {
    calling_thread_and_operands::<f32>("test_sme2_hook_declines_calling_thread_and_operands_f32");
}

#[test]
fn test_sme2_hook_declines_calling_thread_and_operands_f64() {
    calling_thread_and_operands::<f64>("test_sme2_hook_declines_calling_thread_and_operands_f64");
}

/// A complex product of the generic entry point on the blocked route: the
/// type hand-off declines before the detection is asked, and the result is
/// the one with the backend off.
fn complex_declines<T>(test: &str, value: impl Fn(usize) -> T)
where
    T: crate::level3::gemm_kernel::GemmKernel + bytemuck::Zeroable + PartialEq + core::fmt::Debug,
{
    let filled = |rows: usize, cols: usize, salt: usize| {
        let mut mat = Mat::<T>::zeros(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                mat[(i, j)] = value(i * 31 + j * 7 + salt);
            }
        }
        mat
    };
    let (m, n, k) = (70, 40, 65);
    let a = filled(k, m, 1);
    let b = filled(k, n, 5);
    let run = |off: bool| {
        let _off = off.then(force_off);
        let mut c = filled(m, n, 3);
        let (decided, asked) = (decisions(), queries_here());
        gemm_transposed(
            Trans::Trans,
            Trans::NoTrans,
            value(1),
            a.as_ref(),
            b.as_ref(),
            value(2),
            c.as_mut(),
        );
        (c, decisions() - decided, queries_here() - asked)
    };
    let (want, _, _) = run(true);
    let (got, decided, asked) = run(false);
    assert_eq!(decided, 0, "{test}: decided");
    assert_eq!(asked, 0, "{test}: asked the detection");
    assert!(take_last_run().is_none(), "{test}: SME2 ran");
    for j in 0..n {
        for i in 0..m {
            assert_eq!(got[(i, j)], want[(i, j)], "{test}: C({i}, {j})");
        }
    }
}

#[test]
fn test_sme2_hook_declines_complex_types() {
    let test = "test_sme2_hook_declines_complex_types";
    // Ungated: the decline happens before any hardware question.
    complex_declines(test, |x| {
        Complex64::new((x % 7) as f64 * 0.25 - 0.5, (x % 5) as f64 * 0.125)
    });
    complex_declines(test, |x| {
        Complex32::new((x % 7) as f32 * 0.25 - 0.5, (x % 5) as f32 * 0.125)
    });
}

/// Where the backend is off for `T` (no hardware for it, or a build that
/// caps SIMD below SME2), the public calls that run on SME2 when it is on
/// (every transpose combination, sequential and parallel) decline: no
/// decision, no run, NEON bits. Ungated: where the backend is on it prints a
/// `note:` line; otherwise it writes the `declined` marker.
fn declines_without_backend<T: TestElem>(test: &str) {
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
    #[cfg(feature = "parallel")]
    let pars = [Par::Seq, Par::RayonWith(2)];
    #[cfg(not(feature = "parallel"))]
    let pars = [Par::Seq];
    let mut calls = 0;
    for &(trans_a, trans_b) in &TRANS_PAIRS {
        for par in pars {
            let call = Call::new(Entry::TransposedWithPar, 70, 3 * T::NR + 1, 65)
                .trans(trans_a, trans_b)
                .par(par);
            let label = format!("{test} ({seen}): {trans_a:?}/{trans_b:?} {par:?}");
            assert_declines::<T>(&call, &label);
            calls += 1;
        }
    }
    note_declined(test);
    println!("{test}: the hook declined all {calls} calls ({seen})");
}

#[test]
fn test_sme2_hook_declines_without_backend_f32() {
    declines_without_backend::<f32>("test_sme2_hook_declines_without_backend_f32");
}

#[test]
fn test_sme2_hook_declines_without_backend_f64() {
    declines_without_backend::<f64>("test_sme2_hook_declines_without_backend_f64");
}
