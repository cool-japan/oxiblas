//! The guards between the correctness pass and the timing: every call of a
//! variant executes what its correctness call executed (fingerprints), and
//! A's own result is checked against the reference in every group.

use super::super::gemm::{BlockingSource, gemm_planned};
use super::super::gemm_plan::{GemmPlan, GemmPolicy, LEGACY_POLICY, SimdTier};
use super::check::{
    AbReal, Operands, SEED_A, SEED_B, check_reference, check_streamed, initial_c, operand,
    reference,
};
use super::engage::{Engage, Facts};
use super::engine::{Env, GemmEngine};
use super::groups::{Case, Dims, GroupSpec, Variant, all_specs};
use super::output::VecSink;
use super::runner::{BuildInfo, GroupResult, check_calls, run_group};
use super::sampler::{CallCount, CheckKind, Engine, Entry, Slot, lineup};
use super::settings::Settings;
use super::specs::{Beta, Dtype, TransPair};
use super::tests_chain::{fake_facts, fake_fingerprint, fake_spec};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;
use std::time::Duration;

/// A tiny f64 NN case `m × k` by `k × n` with the intended flags `intended`.
fn tiny(m: usize, k: usize, n: usize, intended: &[bool], beta: Beta) -> Case {
    Case {
        dtype: Dtype::F64,
        trans: TransPair::NN,
        dims: Dims { m, k, n },
        beta,
        quick: true,
        named_target: false,
        intended: intended.to_vec(),
    }
}

/// Smoke settings for tiny real-engine runs (no re-measurement).
fn tiny_settings() -> Settings {
    Settings {
        rounds_requested: 4,
        rounds: 4,
        min_sample: Duration::from_micros(50),
        threads: vec![2],
        quick: true,
        retries: 0,
        tag: "tiny".to_string(),
        out: None,
    }
}

/// The build of a tiny real-engine run.
fn tiny_build() -> BuildInfo {
    BuildInfo {
        tier: format!("{:?}", SimdTier::current()),
        arch: std::env::consts::ARCH.to_string(),
        debug_build: true,
        features: "test".to_string(),
        cpu: "test".to_string(),
    }
}

/// Runs `spec` on `make`'s engines; the result and the lines, or the panic
/// message.
fn run_caught<'e>(
    spec: &GroupSpec,
    make: &mut dyn FnMut(&Case) -> Box<dyn Engine + 'e>,
) -> Result<(GroupResult, Vec<String>), String> {
    let (settings, build) = (tiny_settings(), tiny_build());
    let mut sink = VecSink::default();
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_group(spec, &settings, &build, &mut sink, make)
    }));
    match run {
        Ok(result) => Ok((result, sink.lines)),
        Err(payload) => {
            let text = payload.downcast_ref::<String>().cloned();
            let text = text.or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()));
            Err(text.unwrap_or_default())
        }
    }
}

/// A real engine whose warm-up and timed calls run every variant with the
/// plan of `policy` (under the variant's own name): the checks run as usual.
struct PlanSwap<'e> {
    inner: Box<dyn Engine + 'e>,
    policy: GemmPolicy,
}

impl PlanSwap<'_> {
    fn swapped(&self, variant: &Variant) -> Variant {
        Variant {
            policy: self.policy,
            ..*variant
        }
    }
}

impl Engine for PlanSwap<'_> {
    fn start_clean(&mut self) -> Result<(), String> {
        self.inner.start_clean()
    }
    fn check(&mut self, entry: &Entry) -> Result<Facts, String> {
        self.inner.check(entry)
    }
    fn apply_limit(&mut self, limit: Option<usize>) {
        self.inner.apply_limit(limit);
    }
    fn verify_limit(&mut self, limit: Option<usize>) -> Result<(), String> {
        self.inner.verify_limit(limit)
    }
    fn call(&mut self, variant: &Variant) {
        let swapped = self.swapped(variant);
        self.inner.call(&swapped);
    }
    fn time(&mut self, variant: &Variant, reps: usize) -> Duration {
        let swapped = self.swapped(variant);
        self.inner.time(&swapped, reps)
    }
    fn take_calls(&mut self) -> Vec<CallCount> {
        self.inner.take_calls()
    }
}

/// Two real engines of one case: `checked` makes the correctness calls,
/// `timed` (built from another spec) the warm-up and timed calls.
#[cfg(feature = "parallel")]
struct TimedElsewhere<'e> {
    checked: Box<dyn Engine + 'e>,
    timed: Box<dyn Engine + 'e>,
}

#[cfg(feature = "parallel")]
impl Engine for TimedElsewhere<'_> {
    fn start_clean(&mut self) -> Result<(), String> {
        self.checked.start_clean()
    }
    fn check(&mut self, entry: &Entry) -> Result<Facts, String> {
        self.checked.check(entry)
    }
    fn apply_limit(&mut self, limit: Option<usize>) {
        self.checked.apply_limit(limit);
    }
    fn verify_limit(&mut self, limit: Option<usize>) -> Result<(), String> {
        self.checked.verify_limit(limit)
    }
    fn call(&mut self, variant: &Variant) {
        self.timed.call(variant);
    }
    fn time(&mut self, variant: &Variant, reps: usize) -> Duration {
        self.timed.time(variant, reps)
    }
    fn take_calls(&mut self) -> Vec<CallCount> {
        let mut calls = self.checked.take_calls();
        calls.extend(self.timed.take_calls());
        calls
    }
}

/// The sequential thin group on a thin target and a non-thin neighbour.
fn tiny_thin_seq() -> GroupSpec {
    let mut spec = super::groups::spec_seq_thin();
    spec.cases = vec![
        tiny(1, 64, 48, &[true, true], Beta::One),
        tiny(2, 16, 16, &[false, false], Beta::Zero),
    ];
    spec
}

#[test]
fn test_gemm_ab_timed_plan_from_another_policy_fails() {
    let spec = tiny_thin_seq();
    let env = Env::default();
    // The control run: the real engine as it is completes.
    let mut honest = |case: &Case| -> Box<dyn Engine + '_> {
        Box::new(GemmEngine::<f64>::new(&spec, case, &env))
    };
    let done = run_caught(&spec, &mut honest);
    assert!(done.as_ref().is_ok_and(|r| r.0.cases == 2), "{done:?}");
    // The timed path builds LEGACY plans under the candidate's name.
    let mut swapped = |case: &Case| -> Box<dyn Engine + '_> {
        let inner = Box::new(GemmEngine::<f64>::new(&spec, case, &env));
        Box::new(PlanSwap {
            inner,
            policy: LEGACY_POLICY,
        })
    };
    let failed = run_caught(&spec, &mut swapped).err().unwrap_or_default();
    // LEGACY sends the 1x64x48 product to the small route.
    assert!(
        failed.contains("calls of thin executed [route=Small ")
            && failed.contains("its correctness call executed [route=Thin "),
        "{failed}"
    );
}

#[cfg(feature = "parallel")]
fn pool_of(threads: usize) -> rayon::ThreadPool {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build();
    pool.unwrap_or_else(|e| panic!("pool of {threads}: {e}"))
}

/// The parallel thin group at 2 threads on tiny cases.
#[cfg(feature = "parallel")]
fn tiny_thin_par() -> GroupSpec {
    let mut spec = super::groups::spec_par_thin(2);
    spec.cases = vec![
        tiny(1, 64, 48, &[true, true], Beta::Zero),
        tiny(48, 64, 1, &[true, true], Beta::One),
        tiny(2, 16, 16, &[false, false], Beta::Zero),
    ];
    spec
}

#[cfg(feature = "parallel")]
#[test]
fn test_gemm_ab_real_engine_parallel_tiny_cases() {
    // The parallel thin group at 2 threads through the real engine, in a
    // pool of 2: every call (correctness, warm-up, timed) runs there with
    // `Par::RayonWith(2)`, and the run completes.
    let spec = tiny_thin_par();
    let pool = pool_of(2);
    let env = Env { pool: Some(&pool) };
    let mut make = |case: &Case| -> Box<dyn Engine + '_> {
        Box::new(GemmEngine::<f64>::new(&spec, case, &env))
    };
    let run = run_caught(&spec, &mut make);
    env.hold(None);
    let (result, lines) = run.unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(result.cases, 3);
    let rows: Vec<&String> = lines.iter().filter(|l| l.starts_with("AB\t")).collect();
    // Four lines per case: A, `thin`, `thin_scoped` and the control.
    assert_eq!(rows.len(), 12, "{lines:?}");
    assert!(rows.iter().all(|l| l.contains(" pool=2")), "{rows:?}");
    for name in ["thin", "thin_scoped"] {
        // The scope needs a tier with SIMD thin kernels: on another tier
        // (`force-scalar`) `thin_scoped` takes A's route on its targets.
        let off_tier = name == "thin_scoped"
            && !super::super::gemm_plan::route_tests::simd_thin_tier(SimdTier::current());
        let role = if off_tier { "target_noop" } else { "target" };
        let target = format!("\t{name}\t{role}\t");
        assert!(rows.iter().any(|l| l.contains(&target)), "{name}: {rows:?}");
        let thin = format!("\t{name}\ttarget\t");
        assert_eq!(
            rows.iter().any(|l| l.contains(&thin)),
            !off_tier,
            "{name}: {rows:?}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_gemm_ab_timed_par_seq_in_parallel_group_fails() {
    let spec = tiny_thin_par();
    let mut seq = spec.clone();
    seq.par = Par::Seq;
    let pool = pool_of(2);
    let env = Env { pool: Some(&pool) };
    // The correctness calls with the group's `Par`, the warm-up and timed
    // calls with `Par::Seq`, all in the pool of 2.
    let mut make = |case: &Case| -> Box<dyn Engine + '_> {
        Box::new(TimedElsewhere {
            checked: Box::new(GemmEngine::<f64>::new(&spec, case, &env)),
            timed: Box::new(GemmEngine::<f64>::new(&seq, case, &env)),
        })
    };
    let failed = run_caught(&spec, &mut make).err().unwrap_or_default();
    env.hold(None);
    assert!(
        failed.contains(" executed [route=")
            && failed.contains("par=Seq pool=Some(2)]")
            && failed.contains("par=RayonWith(2) pool=Some(2)]"),
        "{failed}"
    );
}

#[test]
fn test_gemm_ab_calls_are_checked_against_the_correctness_call() {
    let spec = fake_spec();
    let entries = lineup(&spec, true);
    // `b` engages (Thin); A, `c` and the control do not (Blocked).
    let facts: Vec<Facts> = ["a", "b", "c", "control"]
        .map(|name| fake_facts(name == "b"))
        .to_vec();
    let count = |variant: &'static str, engaged: bool, calls: usize| CallCount {
        variant,
        ran: fake_fingerprint(engaged),
        calls,
    };
    let fine = [
        count("a", false, 3),
        count("b", true, 3),
        count("c", false, 3),
        count("control", false, 3),
    ];
    assert_eq!(check_calls(&entries, &facts, &fine, Some(&[3; 4])), Ok(()));
    assert_eq!(check_calls(&entries, &facts, &fine, None), Ok(()));
    assert_eq!(check_calls(&entries, &facts, &[], None), Ok(()));
    // One call of `b` that executed A's path: both fingerprints are named.
    let mut swapped = fine.to_vec();
    swapped.push(count("b", false, 1));
    let error = check_calls(&entries, &facts, &swapped, None);
    let error = error.err().unwrap_or_default();
    let (got, want) = (
        fake_fingerprint(false).text(),
        fake_fingerprint(true).text(),
    );
    assert_eq!(
        error,
        format!("1 calls of b executed {got}, its correctness call executed {want}")
    );
    // A `Par`, pool, scope or NC other than the checked one is a difference
    // too.
    #[cfg(feature = "parallel")]
    {
        let mut par = count("a", false, 3);
        par.ran.par = Par::RayonWith(2);
        assert!(check_calls(&entries, &facts, &[par], None).is_err());
    }
    let mut pool = count("a", false, 3);
    pool.ran.pool = Some(2);
    let mut scope = count("a", false, 3);
    scope.ran.scope = super::super::gemm_plan::PackScope::PerTile;
    let mut nc = count("a", false, 3);
    nc.ran.par_nc += 1;
    for other in [pool, scope, nc] {
        assert!(check_calls(&entries, &facts, &[other], None).is_err());
    }
    // Calls of a name outside the line-up, and missing or extra calls.
    let stray = [count("d", false, 1)];
    let error = check_calls(&entries, &facts, &stray, None);
    assert!(error.is_err_and(|e| e.contains("not in the line-up")));
    let error = check_calls(&entries, &facts, &fine, Some(&[3, 4, 3, 3]));
    assert_eq!(
        error,
        Err("3 calls ran with the plan of b, 4 were asked for".to_string())
    );
}

#[test]
fn test_gemm_ab_base_checked_against_reference_in_every_group() {
    for spec in all_specs(&[2]) {
        let route = spec.engage == Engage::Route;
        let entries = lineup(&spec, route);
        let ctx = format!("{} {}", spec.name, spec.run);
        let checks: Vec<(Slot, CheckKind)> = entries.iter().map(|e| (e.slot, e.check)).collect();
        let own = if route {
            CheckKind::Reference
        } else {
            CheckKind::Bitwise
        };
        let mut want = vec![(Slot::Base, CheckKind::Reference)];
        want.extend((0..spec.candidates().len()).map(|i| (Slot::Candidate(i), own)));
        want.push((Slot::Control, CheckKind::Bitwise));
        assert_eq!(checks, want, "{ctx}");
    }
}

/// A's result of `case` computed directly (LEGACY plan, sequential).
fn legacy_result<T: AbReal>(case: &Case) -> Mat<T> {
    let Dims { m, k, n } = case.dims;
    let (a, b) = (
        operand::<T>(case.trans.a, m, k, SEED_A),
        operand::<T>(case.trans.b, k, n, SEED_B),
    );
    let mut c = initial_c::<T>(m, n, Beta::Zero);
    let plan = GemmPlan::<T>::for_problem(LEGACY_POLICY, m, n, k);
    let (ta, tb) = (case.trans.a, case.trans.b);
    let blocking = BlockingSource::Default;
    let (one, zero) = (T::one(), T::zero());
    let (a, b) = (a.as_ref(), b.as_ref());
    gemm_planned(
        ta,
        tb,
        one,
        a,
        b,
        zero,
        c.as_mut(),
        Par::Seq,
        blocking,
        &plan,
    );
    c
}

/// A bitwise group checks A against the reference: A's correct result
/// passes, an element left unwritten (NaN) or off by a relative 1e-3 fails.
fn bitwise_group_checks_base<T: AbReal>(dtype: Dtype) {
    let spec = super::groups::spec_seq_b_cap();
    let case = Case {
        dtype,
        ..tiny(24, 40, 32, &[false, false], Beta::Zero)
    };
    let env = Env::default();
    let entries = lineup(&spec, false);
    let base = entries[0];
    let good = legacy_result::<T>(&case);
    let mut engine = GemmEngine::<T>::new(&spec, &case, &env);
    let worst = engine.check_result(&base, &good);
    assert!(
        worst
            .as_ref()
            .is_ok_and(|w| w.is_some_and(|w| (0.0..=1.0).contains(&w))),
        "{worst:?}"
    );
    let mut unwritten = good.clone();
    unwritten[(5, 7)] = T::of(f64::NAN);
    assert!(engine.check_result(&base, &unwritten).is_err());
    let mut off = good.clone();
    off[(23, 31)] = T::of(off[(23, 31)].val() * (1.0 + 1e-3));
    let error = engine.check_result(&base, &off).err().unwrap_or_default();
    assert!(error.starts_with("(23, 31): "), "{error}");
}

#[test]
fn test_gemm_ab_bitwise_group_checks_base_against_reference() {
    bitwise_group_checks_base::<f64>(Dtype::F64);
    bitwise_group_checks_base::<f32>(Dtype::F32);
    // Through the whole chain: A's row says `reference` with a figure, the
    // candidates and the control `bitwise`.
    let mut spec = super::groups::spec_seq_b_cap();
    spec.cases = vec![tiny(24, 40, 32, &[false, false], Beta::Zero)];
    let env = Env::default();
    let mut make = |case: &Case| -> Box<dyn Engine + '_> {
        Box::new(GemmEngine::<f64>::new(&spec, case, &env))
    };
    let (_, lines) = run_caught(&spec, &mut make).unwrap_or_else(|e| panic!("{e}"));
    let rows: Vec<Vec<&str>> = lines
        .iter()
        .filter(|l| l.starts_with("AB\t"))
        .map(|l| l.split('\t').collect())
        .collect();
    let at = |name: &str| super::output::AB_COLUMNS.iter().position(|c| *c == name);
    let (role, check, worst) = (at("role"), at("check"), at("check_worst"));
    fn field<'r>(row: &[&'r str], i: Option<usize>) -> Option<&'r str> {
        i.and_then(|i| row.get(i + 1)).copied()
    }
    for row in &rows {
        let (r, c, w) = (field(row, role), field(row, check), field(row, worst));
        match r {
            Some("base") => {
                assert_eq!(c, Some("reference"), "{row:?}");
                assert!(w.is_some_and(|w| w.parse::<f64>().is_ok()), "{row:?}");
            }
            _ => assert_eq!((c, w), (Some("bitwise"), Some("-")), "{row:?}"),
        }
    }
    assert_eq!(rows.len(), 4);
}

/// `check_streamed` agrees with `check_reference` (figure and error) for
/// `trans`, `beta` and the dimensions `[m, k, n]`.
fn streamed_agrees<T: AbReal>(trans: TransPair, beta: Beta, [m, k, n]: [usize; 3]) {
    let (a, b) = (
        operand::<T>(trans.a, m, k, SEED_A),
        operand::<T>(trans.b, k, n, SEED_B),
    );
    let c0 = (beta == Beta::One).then(|| initial_c::<T>(m, n, Beta::One));
    let ops = Operands {
        trans_a: trans.a,
        trans_b: trans.b,
        a: &a,
        b: &b,
        c0: c0.as_ref(),
    };
    let built = reference(&ops, [m, k, n]);
    let mut got = Mat::<T>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            got[(i, j)] = T::of(built.elems[i + j * m].value);
        }
    }
    let ctx = format!("{} {m}x{k}x{n} {beta:?}", trans.name());
    let (whole, streamed) = (
        check_reference(&got, &built),
        check_streamed(&ops, [m, k, n], &got),
    );
    assert!(whole.is_ok(), "{ctx}: {whole:?}");
    assert_eq!(whole, streamed, "{ctx}");
    // Two elements off: both name the first one in column-major order.
    got[(m - 1, n - 1)] = T::of(f64::NAN);
    got[(m / 2, n / 2)] = T::of(got[(m / 2, n / 2)].val() + 1.0);
    let (whole, streamed) = (
        check_reference(&got, &built),
        check_streamed(&ops, [m, k, n], &got),
    );
    assert!(whole.is_err(), "{ctx}");
    assert_eq!(whole, streamed, "{ctx}");
    assert!(check_streamed(&ops, [m, k, n], &Mat::<T>::zeros(m, n + 1)).is_err());
}

#[test]
fn test_gemm_ab_streamed_check_agrees_with_reference() {
    for trans in [TransPair::NN, TransPair::TN, TransPair::NT, TransPair::TT] {
        for beta in [Beta::Zero, Beta::One] {
            streamed_agrees::<f64>(trans, beta, [7, 33, 5]);
            streamed_agrees::<f32>(trans, beta, [5, 17, 9]);
        }
    }
    // Large enough for several threads (columns shared out in turn).
    streamed_agrees::<f64>(TransPair::NN, Beta::One, [32, 256, 300]);
    streamed_agrees::<f32>(TransPair::TT, Beta::Zero, [24, 256, 400]);
}
