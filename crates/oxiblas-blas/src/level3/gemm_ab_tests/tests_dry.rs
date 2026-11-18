//! The dry run of every group (nothing large is computed), the cache drain,
//! the limit guards, and the real engine on tiny cases.

use super::super::gemm::{BlockingSource, CallBlocking, call_blocking, gemm_planned};
use super::super::gemm_packing::{PACK_CACHE_MAX_BYTES, pack_cache_bytes, pack_cache_limit};
use super::super::gemm_plan::route_tests::{Rule, model_route, simd_thin_tier};
use super::super::gemm_plan::{
    DEFAULT_POLICY, GemmPlan, GemmPolicy, GemmRoute, LEGACY_POLICY, SimdTier,
};
use super::check::AbReal;
use super::engage::{Engage, predicted_cache_use};
use super::engine::{Env, GemmEngine, hold_cache_limit};
use super::groups::{Case, DESIGN_TIER, Dims, GroupSpec, KernelPick, MIB, Variant, all_specs};
use super::output::VecSink;
use super::runner::{BuildInfo, run_group};
use super::sampler::Engine;
use super::settings::Settings;
use super::specs::{Beta, Dtype, TransPair};
use super::stats::Decision;
use oxiblas_matrix::Mat;
use std::time::Duration;

#[cfg(feature = "parallel")]
use super::super::gemm::PARALLEL_GATE;

/// The blocks of `variant`'s plan for `case`, with its limit held on this
/// thread (the B cap reads the calling thread's limit).
fn blocks_of<T: AbReal>(variant: &Variant, case: &Case) -> CallBlocking {
    hold_cache_limit(variant.limit);
    let plan = variant.plan::<T>(case.dims);
    let Dims { m, k, n } = case.dims;
    let blocks = call_blocking(BlockingSource::Default, &plan, m, k, n);
    hold_cache_limit(None);
    blocks
}

/// Whether `cand` engages on `case` on this host as far as can be told
/// without running it: route, kernel and scope groups at the design tier
/// (the intended targets), NC of the driver for the B caps, the cache
/// model for the cache limit.
fn predicted_engagement<T: AbReal>(spec: &GroupSpec, case: &Case, cand: &Variant) -> bool {
    let Some(base) = spec.variants.first() else {
        return false;
    };
    let Dims { m, k, n } = case.dims;
    match spec.engage {
        Engage::Route | Engage::Kernel | Engage::Scope => spec.intended_for(case, cand),
        Engage::BCap => {
            let nc = |v: &Variant| {
                let blocks = blocks_of::<T>(v, case);
                #[cfg(feature = "parallel")]
                {
                    let work = usize::try_from(case.dims.work()).unwrap_or(usize::MAX);
                    if PARALLEL_GATE.should_parallelize(work, spec.expected_par()) {
                        return blocks.parallel().nc;
                    }
                }
                blocks.sequential().nc
            };
            nc(cand) != nc(base)
        }
        Engage::CacheLimit => {
            let use_of = |v: &Variant| {
                let blocks = *blocks_of::<T>(v, case).sequential();
                let shape = v.plan::<T>(case.dims).micro_kernel().shape;
                let limit = v.limit.unwrap_or(PACK_CACHE_MAX_BYTES);
                predicted_cache_use(&blocks, [m, k, n], &shape, size_of::<T>(), limit)
            };
            use_of(cand) != use_of(base)
        }
    }
}

/// Checks that `spec`'s plans for `case` build (no operands are made).
fn dry_case<T: AbReal>(spec: &GroupSpec, case: &Case) {
    let ctx = format!("{} {} {}", spec.name, spec.run, case.label());
    let Dims { m, k, n } = case.dims;
    assert!(m > 0 && k > 0 && n > 0, "{ctx}: empty dimension");
    assert_eq!(
        case.intended.len(),
        spec.candidates().len(),
        "{ctx}: intended targets"
    );
    for v in &spec.variants {
        // The forced 16x6 kernel is AVX2 code: built only on that tier.
        #[cfg(target_arch = "x86_64")]
        if v.kernel == KernelPick::F32x16x6 && SimdTier::current() != SimdTier::Avx2Fma {
            assert_eq!(case.dtype, Dtype::F32, "{ctx}: 16x6 is f32");
            continue;
        }
        let shape = v.plan::<T>(case.dims).micro_kernel().shape;
        assert!(
            shape.mr > 0 && shape.nr > 0,
            "{ctx} {}: kernel shape",
            v.name
        );
    }
}

/// Whether `cand` engages on `case` (by element type).
fn engages_on(spec: &GroupSpec, case: &Case, cand: &Variant) -> bool {
    match case.dtype {
        Dtype::F64 => predicted_engagement::<f64>(spec, case, cand),
        Dtype::F32 => predicted_engagement::<f32>(spec, case, cand),
    }
}

/// Intended targets of candidate `ci` of `spec` on which it engages here.
fn engaged_targets(spec: &GroupSpec, ci: usize) -> usize {
    let cand = &spec.candidates()[ci];
    let engaged = |c: &&Case| c.intended[ci] && engages_on(spec, c, cand);
    spec.cases.iter().filter(engaged).count()
}

/// The group-level checks of the dry run.
fn dry_spec(spec: &GroupSpec) {
    let ctx = format!("{} {}", spec.name, spec.run);
    let base = spec
        .variants
        .first()
        .copied()
        .unwrap_or_else(|| panic!("{ctx}: no A"));
    // A is LEGACY (the cache group's cap-on run: LEGACY with the cap).
    let reset = GemmPolicy {
        b_cap: LEGACY_POLICY.b_cap,
        ..base.policy
    };
    assert_eq!(reset, LEGACY_POLICY, "{ctx}: A is not LEGACY");
    assert!(
        base.kernel == KernelPick::Plan && base.limit.is_none(),
        "{ctx}: A"
    );
    assert!(!spec.candidates().is_empty(), "{ctx}: no candidate");
    let mut labels: Vec<String> = spec.cases.iter().map(Case::label).collect();
    let count = labels.len();
    labels.sort();
    labels.dedup();
    assert_eq!(labels.len(), count, "{ctx}: duplicate case labels");
    assert!(
        spec.selected(true).next().is_some(),
        "{ctx}: empty quick subset"
    );
    let mut names: Vec<&str> = spec.variants.iter().map(|v| v.name).collect();
    names.push("control");
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        spec.variants.len() + 1,
        "{ctx}: duplicate variant names"
    );
    for (ci, v) in spec.candidates().iter().enumerate() {
        let name = v.name;
        let same = v.policy == base.policy && v.kernel == base.kernel && v.limit == base.limit;
        assert!(!same, "{ctx}: {name} is A");
        assert!(
            !name.contains(['\t', ';', ',', ':', ' ']),
            "{ctx}: name {name}"
        );
        // The cache limit's engagement depends on this host's caches more
        // than any other kind: a candidate without an engaged target on some
        // host is a legitimate NO_TARGETS there, so it is checked for the
        // group as a whole (`cache_group_engages`), not per candidate.
        if spec.engage != Engage::CacheLimit {
            assert!(
                engaged_targets(spec, ci) > 0,
                "{ctx}: {name} engages on no intended target here"
            );
        }
    }
    for case in &spec.cases {
        match case.dtype {
            Dtype::F64 => dry_case::<f64>(spec, case),
            Dtype::F32 => dry_case::<f32>(spec, case),
        }
    }
}

#[test]
fn test_gemm_ab_dry_run_every_group() {
    let specs = all_specs(&[2, 4, 8]);
    let mut names: Vec<&str> = specs.iter().map(|s| s.name).collect();
    names.dedup();
    let mut expected = vec![];
    if cfg!(target_arch = "x86_64") {
        expected.extend(["ab_seq_f32_kernel", "ab_seq_small_simd"]);
    }
    if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        expected.push("ab_seq_small_trans_copy");
    }
    expected.extend([
        "ab_seq_thin",
        "ab_seq_thin_grid",
        "ab_seq_b_cap",
        "ab_seq_pack_cache",
    ]);
    if cfg!(feature = "parallel") {
        expected.extend(["ab_par_pack_scope", "ab_par_thin", "ab_par_b_cap"].repeat(3));
    }
    assert_eq!(names, expected);
    // Every group, on every tier: the route groups' engagement here is the
    // router's at the design tier (the run itself measures it at the host
    // tier).
    specs.iter().for_each(dry_spec);
    // Some candidate of the cache group engages on an intended target here.
    let cache_runs = specs.iter().filter(|s| s.engage == Engage::CacheLimit);
    let per_candidate: Vec<usize> = cache_runs
        .flat_map(|s| (0..s.candidates().len()).map(move |ci| engaged_targets(s, ci)))
        .collect();
    assert!(
        per_candidate.iter().any(|&n| n > 0),
        "cache group: {per_candidate:?}"
    );
    // The thin, kernel and scope intended targets are what the design says:
    // `thin` every thin product, `thin_scoped` the products of the scope
    // (the model of the adopted rule at the design tier).
    let thin_groups = specs.iter().filter(|s| s.name.contains("_thin"));
    for spec in thin_groups {
        for case in &spec.cases {
            let Dims { m, n, .. } = case.dims;
            let inp = case.route_inputs(DESIGN_TIER);
            let scoped = model_route(&inp, Rule::AdoptedAarch64) == GemmRoute::Thin;
            let ctx = format!("{} {}", spec.name, case.label());
            assert_eq!(case.intended, [m == 1 || n == 1, scoped], "{ctx}");
        }
    }
    #[cfg(target_arch = "x86_64")]
    for case in &super::groups::spec_seq_f32_kernel().cases {
        use super::super::gemm_kernel::{MicroKernelShape, f32_avx2_picks_16x6};
        use super::super::gemm_plan::F32Avx2Policy;
        let Dims { m, k, n } = case.dims;
        let (tier, canonical) = (SimdTier::Avx2Fma, MicroKernelShape { mr: 8, nr: 8 });
        let picks = f32_avx2_picks_16x6(tier, F32Avx2Policy::Adaptive, canonical, m, n, k);
        assert_eq!(case.intended, [picks, true], "{}", case.label());
    }
    #[cfg(feature = "parallel")]
    for spec in specs.iter().filter(|s| s.engage == Engage::Scope) {
        for case in &spec.cases {
            let work = case.dims.work();
            let gate = work >= 128 * 128 * 128 && work / spec.threads as u128 >= 80 * 80 * 80;
            assert_eq!(case.intended, [gate; 2], "{} {}", spec.run, case.label());
        }
    }
    // The quick subsets have beta = 1 cases in the route groups.
    for spec in specs.iter().filter(|s| s.engage == Engage::Route) {
        let quick_beta = spec.selected(true).any(|c| c.beta == Beta::One);
        assert!(
            quick_beta,
            "{} {}: no quick beta = 1 case",
            spec.name, spec.run
        );
    }
}

#[test]
fn test_gemm_ab_cache_drain_and_limits_restore() {
    // A blocked product parks its buffers; the engine's clean start frees
    // them on every thread and checks it.
    let park = || {
        let (a, b) = (Mat::<f64>::zeros(160, 160), Mat::<f64>::zeros(160, 160));
        let mut c = Mat::<f64>::zeros(160, 160);
        let plan = GemmPlan::<f64>::for_problem(LEGACY_POLICY, 160, 160, 160);
        let par = oxiblas_core::parallel::Par::Seq;
        let (ta, tb) = (
            super::super::trsm::Trans::NoTrans,
            super::super::trsm::Trans::NoTrans,
        );
        gemm_planned(
            ta,
            tb,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
            par,
            BlockingSource::Default,
            &plan,
        );
        pack_cache_bytes()
    };
    let spec = super::groups::spec_seq_thin();
    let case = spec.cases[0].clone();
    let env = Env::default();
    assert!(park() > 0);
    let mut engine = GemmEngine::<f64>::new(&spec, &case, &env);
    assert_eq!(engine.start_clean(), Ok(()));
    assert_eq!(pack_cache_bytes(), 0);
    // Limits: set, checked, restored.
    assert_eq!(pack_cache_limit(), PACK_CACHE_MAX_BYTES);
    for limit in [Some(16 * MIB), Some(32 * MIB), None] {
        engine.apply_limit(limit);
        assert_eq!(pack_cache_limit(), limit.unwrap_or(PACK_CACHE_MAX_BYTES));
        assert_eq!(engine.verify_limit(limit), Ok(()));
    }
    hold_cache_limit(Some(16 * MIB));
    assert!(engine.verify_limit(None).is_err());
    hold_cache_limit(None);
    #[cfg(feature = "parallel")]
    {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build();
        let pool = pool.unwrap_or_else(|e| panic!("pool: {e}"));
        let env = Env { pool: Some(&pool) };
        let limits = || pool.broadcast(|_| pack_cache_limit());
        env.hold(Some(16 * MIB));
        assert_eq!(
            (limits(), env.run(pack_cache_limit)),
            (vec![16 * MIB; 3], 16 * MIB)
        );
        let mut engine = GemmEngine::<f64>::new(&spec, &case, &env);
        assert_eq!(engine.verify_limit(Some(16 * MIB)), Ok(()));
        // A limit missing on one worker is seen.
        pool.broadcast(|ctx| {
            if ctx.index() == 1 {
                hold_cache_limit(None);
            }
        });
        assert!(engine.verify_limit(Some(16 * MIB)).is_err());
        env.hold(None);
        assert_eq!(
            (limits(), pack_cache_limit()),
            (vec![PACK_CACHE_MAX_BYTES; 3], PACK_CACHE_MAX_BYTES)
        );
        // Buffers parked on the workers are drained too.
        let parked: usize = pool.broadcast(|_| park()).iter().sum();
        assert!(parked > 0);
        assert_eq!(engine.start_clean(), Ok(()));
        assert!(
            pool.broadcast(|_| pack_cache_bytes())
                .iter()
                .all(|&b| b == 0)
        );
    }
}

#[test]
fn test_gemm_ab_real_engine_on_tiny_cases() {
    // The thin group on two tiny cases through the real engine: correctness,
    // engagement, timing and lines.
    let mut spec = super::groups::spec_seq_thin();
    let tiny = |m: usize, k: usize, n: usize, intended: bool, beta: Beta| Case {
        dtype: Dtype::F64,
        trans: TransPair::NN,
        dims: Dims { m, k, n },
        beta,
        quick: true,
        named_target: false,
        intended: vec![intended; 2],
    };
    spec.cases = vec![
        tiny(1, 64, 48, true, Beta::One),
        tiny(2, 16, 16, false, Beta::Zero),
    ];
    let settings = Settings {
        rounds_requested: 4,
        rounds: 4,
        min_sample: Duration::from_micros(50),
        threads: vec![1],
        quick: true,
        retries: 0,
        tag: "tiny".to_string(),
        out: None,
    };
    let build = BuildInfo {
        tier: format!("{:?}", SimdTier::current()),
        arch: std::env::consts::ARCH.to_string(),
        debug_build: true,
        features: "none".to_string(),
        cpu: "test".to_string(),
    };
    let env = Env::default();
    let mut sink = VecSink::default();
    let mut make = |case: &Case| -> Box<dyn Engine + '_> {
        Box::new(GemmEngine::<f64>::new(&spec, case, &env))
    };
    let result = run_group(&spec, &settings, &build, &mut sink, &mut make);
    assert_eq!(result.cases, 2);
    assert_eq!(result.verdicts[0].decision, Decision::Smoke);
    let rows: Vec<&String> = sink
        .lines
        .iter()
        .filter(|l| l.starts_with("AB\t"))
        .collect();
    // Four lines per case: A, `thin`, `thin_scoped` and the control.
    assert_eq!(rows.len(), 8, "{:?}", sink.lines);
    for name in ["thin", "thin_scoped"] {
        // The scope needs a tier with SIMD thin kernels: on another tier
        // (`force-scalar`) `thin_scoped` takes A's route on its target.
        let off_tier = name == "thin_scoped" && !simd_thin_tier(SimdTier::current());
        let (role, seen) = if off_tier {
            ("target_noop", "->Small")
        } else {
            ("target", "->Thin")
        };
        let target = format!("\t{name}\t{role}\treference\t");
        let elsewhere = format!("\t{name}\telsewhere\treference\t");
        let thin_target = rows.iter().any(|l| l.contains(&target) && l.contains(seen));
        let thin_elsewhere = rows.iter().any(|l| l.contains(&elsewhere));
        assert!(thin_target && thin_elsewhere, "{name}: {rows:?}");
        let tagged = format!("\t{name}\t");
        let any_thin = rows
            .iter()
            .any(|l| l.contains(&tagged) && l.contains("->Thin"));
        assert_eq!(any_thin, !off_tier, "{name}: {rows:?}");
    }
}

#[test]
fn test_gemm_ab_every_group_a_is_the_legacy_policy() {
    // A of every group is `LEGACY_POLICY` itself, never the build's
    // default: on aarch64 the default has the adopted fields, and a group
    // must still measure against 0.2.3's routes. The one exception is the
    // cache group's `cap_on` run, whose A is `LEGACY_POLICY` with the B cap
    // (the cap is the setting of that run, not a candidate).
    let specs = all_specs(&[1, 2, 4, 8]);
    assert!(specs.iter().any(|s| s.name == "ab_seq_thin_grid"));
    for spec in &specs {
        let ctx = format!("{} {}", spec.name, spec.run);
        let base = spec.variants.first().copied();
        let base = base.unwrap_or_else(|| panic!("{ctx}: no A"));
        let expected = if spec.name == "ab_seq_pack_cache" && spec.run == "cap_on" {
            GemmPolicy {
                b_cap: super::super::gemm_plan::BCapScope::Sequential,
                ..LEGACY_POLICY
            }
        } else {
            LEGACY_POLICY
        };
        assert_eq!(base.policy, expected, "{ctx}: A");
        if DEFAULT_POLICY != LEGACY_POLICY {
            assert_ne!(base.policy, DEFAULT_POLICY, "{ctx}: A is the default");
        }
        // A's plan, as the engine builds it, carries that policy.
        let case = spec.cases.first();
        let case = case.unwrap_or_else(|| panic!("{ctx}: no case"));
        assert_eq!(base.plan::<f64>(case.dims).ctx.policy, expected, "{ctx}");
    }
}

/// `(trans, m, k, n)` of a case.
type ThinKey = (String, usize, usize, usize);

/// The decision set of the thin route, restated from the selection the
/// deciding figures were computed on (`boundary()` and `original()` of the
/// project's evaluation script): per small layout the grid points of 1024
/// or 2048 multiply-adds and of 512, the points just outside a layout's
/// limit, one point of 1024 multiply-adds and one above the small threshold
/// of each other layout, and the original shapes with the dots of 1024 and
/// 32768.
fn thin_decision_set() -> Vec<ThinKey> {
    let mut out: Vec<ThinKey> = Vec::new();
    let mut add = |t: &str, m: usize, k: usize, n: usize| {
        let key = (t.to_string(), m, k, n);
        if !out.contains(&key) {
            out.push(key);
        }
    };
    let diag = [
        (2, 1024),
        (4, 256),
        (8, 128),
        (16, 64),
        (32, 32),
        (64, 16),
        (128, 8),
        (256, 4),
        (1024, 2),
    ];
    let below = [
        (2, 256),
        (4, 128),
        (8, 64),
        (16, 32),
        (32, 16),
        (64, 8),
        (128, 4),
        (256, 2),
    ];
    for (v, k) in diag.into_iter().chain(below) {
        add("NN", v, k, 1);
        add("TN", v, k, 1);
        add("NN", 1, k, v);
        add("NT", 1, k, v);
    }
    for (v, k) in [(128, 8), (256, 8), (2, 4096)] {
        add("TN", v, k, 1);
    }
    for (v, k) in [(2, 1024), (2, 4096)] {
        add("NN", 1, k, v);
        add("NT", 1, k, v);
    }
    for (v, k) in [(1024, 2), (4096, 2)] {
        add("NT", 1, k, v);
    }
    for (t, m, k, n) in [
        ("NT", 64, 16, 1),
        ("TT", 64, 16, 1),
        ("TN", 1, 16, 64),
        ("TT", 1, 16, 64),
        ("NT", 1024, 64, 1),
        ("TT", 1024, 64, 1),
        ("TN", 1, 64, 1024),
        ("TT", 1, 64, 1024),
    ] {
        add(t, m, k, n);
    }
    for s in [256, 1024, 4096, 16_384] {
        add("NN", 1, s, s);
        add("NN", s, s, 1);
        add("NT", 1, s, s);
        add("TN", s, s, 1);
    }
    for k in [4096, 65_536, 1 << 20, 1024, 32_768] {
        add("NN", 1, k, 1);
        add("TT", 1, k, 1);
    }
    for s in [256, 1024, 4096] {
        add("NN", 2, s, s);
        add("NN", s, s, 2);
    }
    out.sort();
    out
}

/// The `beta = 0` keys and the `beta = 1` labels of `spec`'s `f64` cases,
/// sorted, after checking that the `f32` cases are the same shapes.
fn thin_keys(spec: &GroupSpec) -> (Vec<ThinKey>, Vec<String>) {
    let shapes = |dtype: Dtype| {
        let mut keys: Vec<(ThinKey, bool)> = spec
            .cases
            .iter()
            .filter(|c| c.dtype == dtype)
            .map(|c| {
                let key = (c.trans.name(), c.dims.m, c.dims.k, c.dims.n);
                (key, c.beta == Beta::One)
            })
            .collect();
        keys.sort();
        keys
    };
    let f64s = shapes(Dtype::F64);
    assert_eq!(f64s, shapes(Dtype::F32), "{}: f64 and f32", spec.name);
    let zero = f64s.iter().filter(|k| !k.1).map(|k| k.0.clone());
    let one = spec.cases.iter();
    let one = one.filter(|c| c.dtype == Dtype::F64 && c.beta == Beta::One);
    let mut one: Vec<String> = one.map(Case::label).collect();
    one.sort();
    (zero.collect(), one)
}

#[test]
fn test_gemm_ab_thin_decision_group_is_the_decision_set() {
    let decision = super::groups::spec_seq_thin();
    let (keys, beta_one) = thin_keys(&decision);
    let expected = thin_decision_set();
    assert_eq!(keys, expected);
    assert_eq!(
        beta_one,
        [
            "NN_1024x1024x1_f64_beta1",
            "NN_1x1024x1024_f64_beta1",
            "NN_1x256x256_f64_beta1",
            "NN_1x65536x1_f64_beta1",
            "TN_1024x1024x1_f64_beta1",
        ]
    );
    assert_eq!((expected.len(), decision.cases.len()), (113, 236));
    // The information group is the dense grid as measured: 1530 cases,
    // holding every decision shape but the two dots of length 1024, which
    // the measured grid never had.
    let grid = super::groups::spec_seq_thin_grid();
    let (grid_keys, grid_beta_one) = thin_keys(&grid);
    assert_eq!(grid.cases.len(), 1530);
    assert_eq!(grid_beta_one, beta_one);
    let missing: Vec<&ThinKey> = expected.iter().filter(|k| !grid_keys.contains(k)).collect();
    let dots = [
        ("NN".to_string(), 1, 1024, 1),
        ("TT".to_string(), 1, 1024, 1),
    ];
    assert_eq!(missing, dots.iter().collect::<Vec<_>>());
    // The parallel group keeps its shapes: 22 per element type.
    #[cfg(feature = "parallel")]
    assert_eq!(super::groups::spec_par_thin(2).cases.len(), 44);
}
