//! Tests of the SME2 GEMV inside the thin GEMM route (`gemm_thin`): with a
//! policy that turns the thin route on, a production plan's thin product
//! (`m == 1` or `n == 1`) beyond the GEMV crossover runs its GEMV on SME2,
//! with the bits of the NEON thin result; below the crossover, and on a
//! plan that forces the thin route, it does not.
//!
//! The proof that SME2 ran is the decision counter and the kernel entries
//! of the calling thread (`Par::Seq`: the SME2 GEMV runs its one part on
//! the calling thread).

use oxiblas_core::parallel::Par;

use super::detect::queries_here;
use super::test_support::{backend_or_skip, decisions, force_off, kernel_entries_here, sme_state};
use super::tests_gemm_harness::{Store, TestElem};
use super::tests_hook::{Call, Entry, Run, assert_same_c, operands};
use crate::level3::gemm::{BlockingSource, gemm_planned};
use crate::level3::gemm_plan::{CANDIDATE_POLICY, GemmPlan, GemmRoute, RouteInputs, SimdTier};
use crate::level3::trsm::Trans;

/// `call` through `gemm_planned` with `plan`, and the kernel entries of the
/// calling thread during it.
fn run_thin<T: TestElem>(call: &Call, plan: &GemmPlan<T>) -> (Run<T>, u64) {
    let (a, b, mut c) = operands::<T>(call);
    let (alpha, beta) = (T::of(call.scalars.0), T::of(call.scalars.1));
    let (decided, asked, entries) = (decisions(), queries_here(), kernel_entries_here());
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
    let run = Run {
        c,
        decisions: decisions() - decided,
        queries: queries_here() - asked,
        report: None,
    };
    (run, kernel_entries_here() - entries)
}

/// The route the plan takes for `call` (to make sure the test is on the
/// thin route).
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

/// The thin route of a production plan under a policy with the thin route
/// on: `m == 1` (the dot form on B, or the axpy form on a transposed B) and
/// `n == 1` (the axpy form on A, or the dot form on a transposed A), one
/// below, at and one above the GEMV crossovers (`MIN_ROWS` 64, `MIN_WORK`
/// 128², `TRANS_MIN_COLS` 16 for the dot form).
fn thin_route_runs_the_sme2_gemv<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    // (The candidate policy of aarch64 has the thin route on; `route_of`
    // below checks that every case takes it.)
    // (m, n, k, trans_a, trans_b, on SME2): the GEMV runs on the stored
    // large operand (A for n == 1, B for m == 1) with (rows, cols).
    let n_t = Trans::NoTrans;
    let t_t = Trans::Trans;
    let cases = [
        // n == 1, op(A) = A: axpy form on A (m x k): rows m, work m·k.
        (300, 1, 200, n_t, n_t, true),
        (64, 1, 256, n_t, n_t, true),
        (63, 1, 300, n_t, n_t, false),
        (128, 1, 128, n_t, n_t, true),
        (128, 1, 127, n_t, n_t, false),
        // n == 1, op(A) = Aᵀ: dot form on A (k x m stored): rows k, cols m.
        (16, 1, 1024, t_t, n_t, true),
        (15, 1, 1100, t_t, n_t, false),
        (200, 1, 63, t_t, n_t, false),
        // m == 1, op(B) = B: dot form on B (k x n): rows k, cols n.
        (1, 200, 300, n_t, n_t, true),
        (1, 16, 1024, n_t, n_t, true),
        (1, 15, 1100, n_t, n_t, false),
        (1, 300, 63, n_t, n_t, false),
        // m == 1, op(B) = Bᵀ: axpy form on B (n x k stored): rows n, cols k.
        (1, 300, 200, n_t, t_t, true),
        (1, 63, 300, n_t, t_t, false),
    ];
    for (m, n, k, trans_a, trans_b, on_sme2) in cases {
        for (alpha, beta) in [(0.75, 0.0), (1.0, 1.0), (-1.5, 0.25)] {
            let label = format!("{test}: {m}x{n}x{k} {trans_a:?}/{trans_b:?} a {alpha} b {beta}");
            let mut call = Call::new(Entry::Transposed, m, n, k)
                .trans(trans_a, trans_b)
                .scalars(alpha, beta)
                .stores(Store::ColMajor { pad: 0 }, Store::ColMajor { pad: 0 });
            if beta == 0.0 {
                call = call.nan_c();
            }
            let plan = GemmPlan::<T>::for_problem(CANDIDATE_POLICY, m, n, k);
            assert_eq!(route_of(&call, &plan), GemmRoute::Thin, "{label}: route");
            let want = {
                let _off = force_off();
                run_thin(&call, &plan).0
            };
            let (got, entries) = run_thin(&call, &plan);
            assert_eq!(got.decisions, u64::from(on_sme2), "{label}: decisions");
            assert_eq!(entries > 0, on_sme2, "{label}: kernel entries {entries}");
            assert_same_c(&got, &want, &label);
            assert_eq!(sme_state(), Some((0, 0)), "{label}: SME state");
            if !on_sme2 {
                // Below the crossover the detection is not asked.
                assert_eq!(got.queries, 0, "{label}: asked the detection");
            }
        }
    }
}

#[test]
fn test_sme2_hook_thin_route_runs_the_sme2_gemv_f32() {
    thin_route_runs_the_sme2_gemv::<f32>("test_sme2_hook_thin_route_runs_the_sme2_gemv_f32");
}

#[test]
fn test_sme2_hook_thin_route_runs_the_sme2_gemv_f64() {
    thin_route_runs_the_sme2_gemv::<f64>("test_sme2_hook_thin_route_runs_the_sme2_gemv_f64");
}

/// A plan that forces the thin route (the thin tests' plans: a given
/// kernel, a forced route, the portable tier) runs the thin route's own
/// kernels: no decision, no detection call, the bits of the same call with
/// the backend off.
fn forced_thin_route_runs_its_kernels<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for (m, n, k) in [(1, 200, 300), (300, 1, 200)] {
        let call = Call::new(Entry::Transposed, m, n, k)
            .stores(Store::ColMajor { pad: 0 }, Store::ColMajor { pad: 0 })
            .par(Par::Seq);
        let production = GemmPlan::<T>::for_problem(CANDIDATE_POLICY, m, n, k);
        let mut portable = production.with_forced_route(GemmRoute::Thin);
        portable.ctx = portable.ctx.with_portable_tier();
        assert_eq!(portable.ctx.tier(), SimdTier::Portable);
        for (what, plan) in [
            (
                "forced thin route",
                production.with_forced_route(GemmRoute::Thin),
            ),
            (
                "given kernel",
                production.with_kernel(T::micro_kernel_dispatch()),
            ),
            ("portable tier", portable),
        ] {
            let label = format!("{test}: {what} {m}x{n}x{k}");
            assert_eq!(route_of(&call, &plan), GemmRoute::Thin, "{label}: route");
            let want = {
                let _off = force_off();
                run_thin(&call, &plan).0
            };
            let (got, entries) = run_thin(&call, &plan);
            assert_eq!(got.decisions, 0, "{label}: decided");
            assert_eq!(got.queries, 0, "{label}: asked the detection");
            assert_eq!(entries, 0, "{label}: a kernel ran");
            assert_same_c(&got, &want, &label);
        }
        // The same shape with the production plan runs on SME2.
        let (got, _) = run_thin(&call, &production);
        assert_eq!(got.decisions, 1, "{test}: production plan {m}x{n}x{k}");
    }
}

#[test]
fn test_sme2_hook_thin_forced_route_runs_its_kernels_f32() {
    forced_thin_route_runs_its_kernels::<f32>(
        "test_sme2_hook_thin_forced_route_runs_its_kernels_f32",
    );
}

#[test]
fn test_sme2_hook_thin_forced_route_runs_its_kernels_f64() {
    forced_thin_route_runs_its_kernels::<f64>(
        "test_sme2_hook_thin_forced_route_runs_its_kernels_f64",
    );
}
