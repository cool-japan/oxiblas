//! The whole chain of a case and a group run (calibration, warm-ups, order,
//! rounds, statistics, lines, re-measurement, roles, verdict) driven by a
//! fake engine with known per-variant costs.

use super::super::gemm_kernel::MicroKernelShape;
use super::super::gemm_plan::{GemmRoute, LEGACY_POLICY, PackScope};
use super::super::gemm_thin::ThinSplits;
use super::engage::{DriverRuns, Engage, Facts, Fingerprint, Path};
use super::groups::{Case, Dims, GroupSpec, MIB, Variant};
use super::output::{VecSink, parse_line};
use super::runner::{BuildInfo, CaseResult, RunContext, run_case, run_group};
use super::sampler::{CallCount, Engine, Entry, count_calls, round_order};
use super::settings::Settings;
use super::specs::{Beta, Dtype, TransPair};
use super::stats::{Decision, Role, Threshold};
use oxiblas_core::parallel::Par;
use std::time::Duration;

/// What the fake engine was asked to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Clean,
    Check,
    Apply,
    Verify,
    Call,
    Time,
}

/// One logged request: the operation, the variant it named (`""` for none)
/// and the limit in force when it ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Event {
    op: Op,
    variant: &'static str,
    limit: Option<usize>,
    /// Calls made (1 for `Call`, `reps` for `Time`).
    calls: usize,
}

/// The cost of one variant per attempt (the last entry repeats).
#[derive(Debug, Clone)]
struct Cost {
    variant: &'static str,
    per_attempt_us: Vec<f64>,
}

/// Factors on the cost of a variant's timed samples, in turn: sample `s` of
/// the variant (counted over the whole case) costs `factors[s % len]` times
/// its cost. Calibration and every round time each variant once, so equal
/// periods stay in phase across the variants.
#[derive(Debug, Clone)]
pub(super) struct Drift {
    pub(super) variant: &'static str,
    pub(super) factors: Vec<f64>,
}

/// A fake engine: known costs per variant and attempt, a drift per sample,
/// a penalty on a call whose predecessor was another variant, engagement by
/// name.
#[derive(Debug, Clone)]
pub(super) struct FakeEngine {
    costs: Vec<Cost>,
    /// Per-sample factors of some variants' costs.
    pub(super) drift: Vec<Drift>,
    /// Added to a timed sample whose previous call was of another variant.
    switch_penalty_us: f64,
    /// Runs every timed call with A's plan (an engine bug the runner must
    /// see).
    time_with_base: bool,
    /// Candidates that engage (change the path).
    engaged: Vec<&'static str>,
    /// Entries in the line-up (to tell checks from attempts).
    lineup: usize,
    cleans: usize,
    limit: Option<usize>,
    last: &'static str,
    log: Vec<Event>,
    /// Log entries already returned by `take_calls`.
    taken: usize,
}

impl FakeEngine {
    pub(super) fn new(
        costs: &[(&'static str, &[f64])],
        engaged: &[&'static str],
        lineup: usize,
    ) -> Self {
        let costs = costs.iter().map(|&(variant, us)| Cost {
            variant,
            per_attempt_us: us.to_vec(),
        });
        Self {
            costs: costs.collect(),
            drift: Vec::new(),
            switch_penalty_us: 0.0,
            time_with_base: false,
            engaged: engaged.to_vec(),
            lineup,
            cleans: 0,
            limit: None,
            last: "",
            log: Vec::new(),
            taken: 0,
        }
    }

    /// The factor on the next timed sample of `variant` (see [`Drift`]).
    fn drift_of(&self, variant: &str) -> f64 {
        let Some(drift) = self.drift.iter().find(|d| d.variant == variant) else {
            return 1.0;
        };
        let timed = self.log.iter().filter(|e| e.op == Op::Time);
        let done = timed.filter(|e| e.variant == variant).count();
        let at = done.checked_rem(drift.factors.len());
        at.and_then(|i| drift.factors.get(i))
            .copied()
            .unwrap_or(1.0)
    }

    fn attempt(&self) -> usize {
        self.cleans.saturating_sub(self.lineup + 1)
    }

    fn cost_us(&self, variant: &str) -> f64 {
        let cost = self.costs.iter().find(|c| c.variant == variant);
        let per = cost.map_or(&[][..], |c| &c.per_attempt_us[..]);
        let at = self.attempt().min(per.len().saturating_sub(1));
        per.get(at)
            .copied()
            .unwrap_or_else(|| panic!("no cost for {variant}"))
    }

    fn log(&mut self, op: Op, variant: &'static str) {
        self.log_calls(op, variant, 0);
    }

    fn log_calls(&mut self, op: Op, variant: &'static str, calls: usize) {
        let limit = self.limit;
        self.log.push(Event {
            op,
            variant,
            limit,
            calls,
        });
    }
}

/// The fingerprint of a fake call: the path `Thin` when `engaged`,
/// `Blocked` otherwise.
pub(super) fn fake_fingerprint(engaged: bool) -> Fingerprint {
    let route = if engaged {
        GemmRoute::Thin
    } else {
        GemmRoute::Blocked
    };
    Fingerprint {
        path: Path { route, simd: false },
        kernel: MicroKernelShape { mr: 8, nr: 6 },
        kc: 256,
        seq_nc: 64,
        par_nc: 64,
        scope: PackScope::Shared,
        par: Par::Seq,
        pool: None,
    }
}

/// Facts with the path `Thin` when `engaged`, `Blocked` otherwise.
pub(super) fn fake_facts(engaged: bool) -> Facts {
    Facts {
        ran: fake_fingerprint(engaged),
        runs: DriverRuns::default(),
        thin: ThinSplits::default(),
        nc: 64,
        cache: None,
        same_bits_as_base: !engaged,
        worst: Some(0.25),
        detail: format!("fake engaged={engaged}"),
    }
}

impl Engine for FakeEngine {
    fn take_calls(&mut self) -> Vec<CallCount> {
        let mut out = Vec::new();
        for event in self.log.get(self.taken..).unwrap_or(&[]) {
            if event.calls > 0 {
                let ran = fake_fingerprint(self.engaged.contains(&event.variant));
                count_calls(&mut out, event.variant, ran, event.calls);
            }
        }
        self.taken = self.log.len();
        out
    }

    fn start_clean(&mut self) -> Result<(), String> {
        self.cleans += 1;
        self.log(Op::Clean, "");
        Ok(())
    }

    fn check(&mut self, entry: &Entry) -> Result<Facts, String> {
        self.limit = entry.variant.limit;
        self.log(Op::Check, entry.variant.name);
        Ok(fake_facts(self.engaged.contains(&entry.variant.name)))
    }

    fn apply_limit(&mut self, limit: Option<usize>) {
        self.limit = limit;
        self.log(Op::Apply, "");
    }

    fn verify_limit(&mut self, limit: Option<usize>) -> Result<(), String> {
        self.log(Op::Verify, "");
        if self.limit == limit {
            Ok(())
        } else {
            Err(format!(
                "limit {:?} in force, expected {limit:?}",
                self.limit
            ))
        }
    }

    fn call(&mut self, variant: &Variant) {
        self.log_calls(Op::Call, variant.name, 1);
        self.last = variant.name;
    }

    fn time(&mut self, variant: &Variant, reps: usize) -> Duration {
        let ran = if self.time_with_base {
            "a"
        } else {
            variant.name
        };
        let drift = self.drift_of(variant.name);
        self.log_calls(Op::Time, ran, reps);
        let penalty = if self.last == variant.name {
            0.0
        } else {
            self.switch_penalty_us
        };
        self.last = variant.name;
        let us = self.cost_us(variant.name) * drift * reps as f64 + penalty;
        Duration::from_secs_f64(us * 1e-6)
    }
}

/// A route group of A (`a`), `b` (16 MiB limit) and `c` (32 MiB limit), with
/// cases `NN_4x4x4` (intended for `b`) and `NN_8x8x8` (intended for `c`).
pub(super) fn fake_spec() -> GroupSpec {
    let case = |m: usize, intended: Vec<bool>| Case {
        dtype: Dtype::F64,
        trans: TransPair::NN,
        dims: Dims { m, k: m, n: m },
        beta: Beta::Zero,
        quick: true,
        named_target: false,
        intended,
    };
    let a = Variant::with("a", LEGACY_POLICY);
    GroupSpec {
        name: "ab_fake",
        run: "seq".to_string(),
        threads: 1,
        parallel: false,
        par: Par::Seq,
        engage: Engage::Route,
        threshold: Threshold::None,
        variants: vec![a, a.limited("b", 16), a.limited("c", 32)],
        cases: vec![case(4, vec![true, false]), case(8, vec![false, true])],
    }
}

/// Settings of a decision-grade run with `rounds` rounds and `retries`.
pub(super) fn fake_settings(rounds: usize, retries: usize) -> Settings {
    Settings {
        rounds_requested: rounds,
        rounds,
        min_sample: Duration::from_millis(2),
        threads: vec![1],
        quick: false,
        retries,
        tag: "fake".to_string(),
        out: None,
    }
}

/// Runs the first case of `spec` on `engine`; returns the result and lines.
pub(super) fn fake_case(
    spec: &GroupSpec,
    settings: &Settings,
    engine: &mut FakeEngine,
) -> (CaseResult, Vec<String>) {
    let ctx = RunContext {
        spec,
        settings,
        run_id: "run-1",
    };
    let mut sink = VecSink::default();
    let result = run_case(&ctx, &spec.cases[0], engine, &mut sink);
    (result, sink.lines)
}

/// The `AB` lines' fields keyed by column name.
pub(super) fn ab_rows(lines: &[String]) -> Vec<Vec<(&'static str, String)>> {
    let names = super::output::AB_COLUMNS;
    let rows = lines
        .iter()
        .filter_map(|l| parse_line(l))
        .filter(|(kind, _)| *kind == "AB");
    rows.map(|(_, fields)| {
        names
            .iter()
            .copied()
            .zip(fields.into_iter().map(String::from))
            .collect()
    })
    .collect()
}

/// The value of `column` in `row`.
pub(super) fn get<'r>(row: &'r [(&'static str, String)], column: &str) -> &'r str {
    let found = row.iter().find(|(name, _)| *name == column);
    found.map_or("", |(_, value)| value.as_str())
}

/// The rows of `variant` in `attempt`.
pub(super) fn rows_of<'r>(
    rows: &'r [Vec<(&'static str, String)>],
    variant: &str,
    attempt: usize,
) -> Vec<&'r [(&'static str, String)]> {
    let matches = |r: &&Vec<(&'static str, String)>| {
        get(r, "variant") == variant && get(r, "attempt") == attempt.to_string()
    };
    rows.iter().filter(matches).map(Vec::as_slice).collect()
}

/// The number in `column` of `row`.
pub(super) fn number(row: &[(&'static str, String)], column: &str) -> f64 {
    let text = get(row, column);
    text.parse()
        .unwrap_or_else(|e| panic!("{column}={text:?}: {e}"))
}

/// Costs (us per call) of the fake line-up: `b` twice as fast as A, `c`
/// twice as slow, the control 1 % slower.
pub(super) const COSTS: [(&str, &[f64]); 4] = [
    ("a", &[100.0]),
    ("b", &[50.0]),
    ("c", &[200.0]),
    ("control", &[101.0]),
];

#[test]
fn test_gemm_ab_chain_ratio_is_base_over_variant() {
    let (spec, settings) = (fake_spec(), fake_settings(4, 0));
    let mut engine = FakeEngine::new(&COSTS, &["b"], 4);
    let (result, lines) = fake_case(&spec, &settings, &mut engine);
    let rows = ab_rows(&lines);
    let ratio = |v: &str| number(rows_of(&rows, v, 0)[0], "ratio");
    let close = |x: f64, want: f64| (x - want).abs() < 2e-3;
    assert!(
        close(ratio("a"), 1.0) && close(ratio("b"), 2.0),
        "{lines:?}"
    );
    assert!(close(ratio("c"), 0.5) && close(ratio("control"), 100.0 / 101.0));
    let medians = |v: &str| number(rows_of(&rows, v, 0)[0], "ratio_of_medians");
    assert!(close(medians("b"), 2.0) && close(medians("c"), 0.5));
    // The outcomes pair each candidate's role with its own ratio.
    let [b, c] = [result.outcomes[0], result.outcomes[1]];
    assert_eq!((b.role, c.role), (Role::Target, Role::Elsewhere));
    assert!(close(b.ratio, 2.0) && close(c.ratio, 0.5), "{result:?}");
    assert!(result.valid && result.attempts == 1);
}

#[test]
fn test_gemm_ab_chain_spread_from_control() {
    let spec = fake_spec();
    // Candidates far off, control 0.9 % slower than A: valid.
    let mut engine = FakeEngine::new(&COSTS, &["b"], 4);
    let (result, lines) = fake_case(&spec, &fake_settings(4, 0), &mut engine);
    let want = 100.0 / 101.0 - 1.0;
    assert!(
        (result.spread - want).abs() < 1e-3 && result.valid,
        "{result:?}"
    );
    let rows = ab_rows(&lines);
    let spread = number(rows_of(&rows, "b", 0)[0], "spread");
    assert!((spread - want).abs() < 1e-3, "{spread}");
    // The control alone decides validity: 2 % off is invalid although every
    // candidate is identical to A; symmetric in both directions.
    for control in [98.0, 102.0] {
        let costs: [(&str, &[f64]); 4] = [
            ("a", &[100.0]),
            ("b", &[100.0]),
            ("c", &[100.0]),
            ("control", &[control]),
        ];
        let mut engine = FakeEngine::new(&costs, &[], 4);
        let (result, _) = fake_case(&spec, &fake_settings(4, 0), &mut engine);
        assert!(!result.valid, "control {control}: {result:?}");
    }
    for control in [98.6, 101.4] {
        let costs: [(&str, &[f64]); 4] = [
            ("a", &[100.0]),
            ("b", &[70.0]),
            ("c", &[130.0]),
            ("control", &[control]),
        ];
        let mut engine = FakeEngine::new(&costs, &[], 4);
        let (result, _) = fake_case(&spec, &fake_settings(4, 0), &mut engine);
        assert!(result.valid, "control {control}: {result:?}");
    }
}

/// The timed samples of `log`: per sample the events from its first request
/// to its `Time` (the line-up's limits differ, so a sample is Apply, Call,
/// Verify, Time).
fn samples(log: &[Event]) -> Vec<&[Event]> {
    let timed: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, e)| e.op == Op::Time)
        .map(|(i, _)| i)
        .collect();
    timed
        .iter()
        .map(|&i| &log[i.saturating_sub(3)..=i])
        .collect()
}

#[test]
fn test_gemm_ab_chain_timed_calls_use_variant_plan_and_limit() {
    let spec = fake_spec();
    let mut engine = FakeEngine::new(&COSTS, &["b"], 4);
    let _ = fake_case(&spec, &fake_settings(4, 0), &mut engine);
    let limit_of = |name: &str| match name {
        "b" => Some(16 * MIB),
        "c" => Some(32 * MIB),
        _ => None,
    };
    let timed = samples(&engine.log);
    assert!(!timed.is_empty());
    for sample in timed {
        let ops: Vec<Op> = sample.iter().map(|e| e.op).collect();
        assert_eq!(
            ops,
            [Op::Apply, Op::Call, Op::Verify, Op::Time],
            "{sample:?}"
        );
        let (call, time) = (sample[1], sample[3]);
        // The warm-up and the timed calls are of one variant, with its limit.
        assert_eq!(call.variant, time.variant, "{sample:?}");
        assert_eq!(time.limit, limit_of(time.variant), "{sample:?}");
        assert_eq!(call.limit, limit_of(call.variant), "{sample:?}");
    }
    // Every variant is checked with its own limit, each from a clean start.
    let checks: Vec<(&str, Option<usize>)> = engine
        .log
        .iter()
        .filter(|e| e.op == Op::Check)
        .map(|e| (e.variant, e.limit))
        .collect();
    let want = [
        ("a", None),
        ("b", Some(16 * MIB)),
        ("c", Some(32 * MIB)),
        ("control", None),
    ];
    assert_eq!(checks, want);
    let before_checks = engine.log.windows(2).filter(|w| w[1].op == Op::Check);
    assert!(
        before_checks.clone().count() == 4 && before_checks.clone().all(|w| w[0].op == Op::Clean)
    );
}

#[test]
fn test_gemm_ab_chain_order_alternates_with_warmup() {
    let spec = fake_spec();
    let mut engine = FakeEngine::new(&COSTS, &["b"], 4);
    // A first call after another variant costs 1 ms more: the warm-up call
    // before every sample keeps it out of the timed calls.
    engine.switch_penalty_us = 1000.0;
    let (result, lines) = fake_case(&spec, &fake_settings(6, 0), &mut engine);
    let rows = ab_rows(&lines);
    let reps = number(rows_of(&rows, "a", 0)[0], "reps") as usize;
    assert!(reps > 1 && result.valid, "{result:?}");
    let ratio = number(rows_of(&rows, "b", 0)[0], "ratio");
    assert!((ratio - 2.0).abs() < 1e-3, "{ratio}");
    // The last 6 rounds of 4 samples: forward, reverse, forward, ...
    let names = ["a", "b", "c", "control"];
    let timed: Vec<&str> = samples(&engine.log).iter().map(|s| s[3].variant).collect();
    let rounds = &timed[timed.len() - 24..];
    for (round, chunk) in rounds.chunks(4).enumerate() {
        let want: Vec<&str> = round_order(round, 4).iter().map(|&v| names[v]).collect();
        assert_eq!(chunk, want, "round {round}");
    }
    assert_eq!(round_order(1, 3), [2, 1, 0]);
    // Calibration samples also follow a warm-up of the same variant.
    for sample in samples(&engine.log) {
        assert_eq!(sample[1].variant, sample[3].variant);
    }
}

#[test]
fn test_gemm_ab_chain_remeasures_and_keeps_the_last_attempt() {
    let spec = fake_spec();
    // Attempts 0 and 1 have a bad control; attempt 2 is valid.
    let costs: [(&str, &[f64]); 4] = [
        ("a", &[100.0]),
        ("b", &[50.0, 60.0, 80.0]),
        ("c", &[200.0]),
        ("control", &[130.0, 125.0, 100.5]),
    ];
    let mut engine = FakeEngine::new(&costs, &["b"], 4);
    let (result, lines) = fake_case(&spec, &fake_settings(4, 2), &mut engine);
    assert_eq!((result.attempts, result.valid), (3, true));
    assert!((result.outcomes[0].ratio - 1.25).abs() < 1e-3, "{result:?}");
    let rows = ab_rows(&lines);
    for (attempt, valid) in [(0, "no"), (1, "no"), (2, "yes")] {
        let printed = rows_of(&rows, "b", attempt);
        assert!(
            printed.len() == 1 && get(printed[0], "valid") == valid,
            "{attempt}"
        );
    }
    let raw = lines.iter().filter(|l| l.starts_with("ABR\t")).count();
    assert_eq!(raw, 3 * 4, "one ABR line per attempt and variant");
    // Retries exhausted: the last attempt counts, not the best one (attempt
    // 0 has the smallest spread and ratio 2.0).
    let costs: [(&str, &[f64]); 4] = [
        ("a", &[100.0]),
        ("b", &[50.0, 60.0, 80.0]),
        ("c", &[200.0]),
        ("control", &[102.0, 110.0, 130.0]),
    ];
    let mut engine = FakeEngine::new(&costs, &["b"], 4);
    let (result, _) = fake_case(&spec, &fake_settings(4, 2), &mut engine);
    assert_eq!((result.attempts, result.valid), (3, false));
    assert!((result.outcomes[0].ratio - 1.25).abs() < 1e-3, "{result:?}");
    // A valid first attempt is not repeated.
    let mut engine = FakeEngine::new(&COSTS, &["b"], 4);
    let (result, _) = fake_case(&spec, &fake_settings(4, 2), &mut engine);
    assert_eq!(result.attempts, 1);
}

#[test]
fn test_gemm_ab_chain_remeasures_a_control_with_bad_quartiles() {
    let spec = fake_spec();
    // Every sample lasts at least the minimum sample time (2 ms) at one
    // call, so calibration times each variant once and an attempt of 8
    // rounds is 9 samples per variant. In the first attempt the control is
    // 6 % slower in three of its 8 rounds: its paired median stays 1 (the
    // spread is 0), its lower quartile is 1/1.06, below 1/1.015². Such a
    // case is re-measured and counts as re-measured; the second attempt is
    // valid and is the result.
    let costs: [(&str, &[f64]); 4] = [
        ("a", &[4000.0]),
        ("b", &[2000.0, 3200.0]),
        ("c", &[8000.0]),
        ("control", &[4000.0]),
    ];
    let first = [1.0, 1.06, 1.0, 1.06, 1.0, 1.06, 1.0, 1.0, 1.0];
    let engine_with = |factors: Vec<f64>| {
        let mut engine = FakeEngine::new(&costs, &["b"], 4);
        engine.drift = vec![Drift {
            variant: "control",
            factors,
        }];
        engine
    };
    let once = || [&first[..], &[1.0; 9]].concat();
    let mut engine = engine_with(once());
    let (result, lines) = fake_case(&spec, &fake_settings(8, 2), &mut engine);
    assert_eq!((result.attempts, result.valid), (2, true), "{result:?}");
    assert!(result.spread.abs() < 1e-9, "{result:?}");
    assert!((result.outcomes[0].ratio - 1.25).abs() < 1e-3, "{result:?}");
    let rows = ab_rows(&lines);
    let control = rows_of(&rows, "control", 0)[0];
    // The first attempt: the median within its limit, a quartile outside.
    assert!(number(control, "spread").abs() < 1e-9, "{control:?}");
    assert_eq!(get(control, "valid"), "no");
    assert!(
        number(control, "ratio_p25") < 1.0 / 1.030_225,
        "{control:?}"
    );
    assert!((number(control, "ratio") - 1.0).abs() < 1e-9, "{control:?}");
    assert_eq!(get(rows_of(&rows, "control", 1)[0], "valid"), "yes");
    // In a group run the case counts as re-measured, and not as invalid.
    let mut sink = VecSink::default();
    let mut make = |_: &Case| -> Box<dyn Engine> { Box::new(engine_with(once())) };
    let one_case = GroupSpec {
        cases: spec.cases[..1].to_vec(),
        ..spec.clone()
    };
    let settings = fake_settings(8, 2);
    let group = run_group(
        &one_case,
        &settings,
        &fake_build(false),
        &mut sink,
        &mut make,
    );
    let counts = (group.cases, group.remeasured, group.invalid);
    assert_eq!(counts, (1, 1, 0), "{group:?}");
    // A control that stays so in every attempt leaves the case invalid.
    let mut engine = engine_with(first.to_vec());
    let (result, _) = fake_case(&spec, &fake_settings(8, 2), &mut engine);
    assert_eq!((result.attempts, result.valid), (3, false), "{result:?}");
    assert!(result.spread.abs() < 1e-9, "{result:?}");
}

#[test]
fn test_gemm_ab_chain_noop_target_counts_as_elsewhere() {
    let spec = fake_spec();
    // `b` is intended on the first case but engages nowhere.
    let mut engine = FakeEngine::new(&COSTS, &[], 4);
    let (result, lines) = fake_case(&spec, &fake_settings(4, 0), &mut engine);
    assert_eq!(result.outcomes[0].role, Role::TargetNoop);
    assert_eq!(result.outcomes[1].role, Role::Elsewhere);
    let rows = ab_rows(&lines);
    let row = rows_of(&rows, "b", 0)[0];
    assert_eq!(
        (get(row, "role"), get(row, "engaged")),
        ("target_noop", "no")
    );
    let control = rows_of(&rows, "control", 0)[0];
    assert_eq!(
        (get(control, "role"), get(control, "check")),
        ("control", "bitwise")
    );
    assert!(get(row, "detail").contains("seen=Blocked->Blocked"));
}

/// The build of a fake group run (`debug`: a debug build).
pub(super) fn fake_build(debug: bool) -> BuildInfo {
    BuildInfo {
        tier: "Neon".to_string(),
        arch: "aarch64".to_string(),
        debug_build: debug,
        features: "parallel".to_string(),
        cpu: "fake".to_string(),
    }
}

/// Runs `spec` as a group on fake engines (every candidate in `engaged`
/// engages on every case) with `settings` and the debug flag `debug`.
fn fake_group(
    spec: &GroupSpec,
    settings: &Settings,
    engaged: &'static [&'static str],
    debug: bool,
) -> (super::runner::GroupResult, Vec<String>) {
    let build = fake_build(debug);
    let mut sink = VecSink::default();
    let mut make = |_: &Case| -> Box<dyn Engine> { Box::new(FakeEngine::new(&COSTS, engaged, 4)) };
    let result = run_group(spec, settings, &build, &mut sink, &mut make);
    (result, sink.lines)
}

#[test]
fn test_gemm_ab_chain_group_verdicts_and_lines() {
    let spec = fake_spec();
    let settings = fake_settings(22, 2);
    let (result, lines) = fake_group(&spec, &settings, &["b"], false);
    let decisions: Vec<Decision> = result.verdicts.iter().map(|v| v.decision).collect();
    // `b` is twice as fast where it engages (both cases); `c` engages nowhere.
    assert_eq!(decisions, [Decision::Adopt, Decision::NoTargets]);
    assert_eq!(
        (
            result.verdicts[0].figures.targets,
            result.verdicts[1].figures.noop
        ),
        (2, 1)
    );
    // The same run in a debug build, or with the quick subset: SMOKE.
    let (smoke, _) = fake_group(&spec, &settings, &["b"], true);
    assert!(smoke.verdicts.iter().all(|v| v.decision == Decision::Smoke));
    // Lines: headers, SET, AB/ABR per case, attempt and variant, ABV, END.
    let kinds: Vec<(&str, String)> = lines
        .iter()
        .filter_map(|l| parse_line(l))
        .map(|(kind, fields)| {
            (
                kind,
                fields.first().map_or(String::new(), |f| f.to_string()),
            )
        })
        .collect();
    let count = |kind: &str, first: &str| {
        kinds
            .iter()
            .filter(|(k, f)| *k == kind && (first.is_empty() || f == first))
            .count()
    };
    assert_eq!(
        (count("AB#", "AB"), count("AB#", "ABV"), count("AB#", "ABR")),
        (1, 1, 1)
    );
    assert_eq!(
        (count("AB#", "SET"), count("AB#", "END"), count("ABV", "")),
        (1, 1, 1)
    );
    assert_eq!((count("AB", ""), count("ABR", "")), (8, 8));
    let run_id = &result.run_id;
    for line in lines.iter().filter(|l| !l.starts_with("AB#\tAB")) {
        let (kind, fields) = parse_line(line).unwrap_or_else(|| panic!("{line}"));
        let id = if kind == "AB#" { fields[1] } else { fields[0] };
        assert_eq!(id, run_id, "{line}");
        assert!(!id.contains([' ', '\t']));
    }
    let set = lines
        .iter()
        .find(|l| l.starts_with("AB#\tSET\t"))
        .cloned()
        .unwrap_or_default();
    for key in [
        "group",
        "run",
        "tier",
        "arch",
        "threads",
        "rounds",
        "min_sample_ms",
        "quick",
        "debug_build",
        "retries",
        "tag",
        "grade",
    ] {
        assert!(
            set.split('\t').any(|f| f.starts_with(&format!("{key}="))),
            "{key} in {set}"
        );
    }
    assert!(set.contains("\tgrade=decision\t") && set.contains("\ttier=Neon\t"));
    let abv = lines
        .iter()
        .find(|l| l.starts_with("ABV\t"))
        .cloned()
        .unwrap_or_default();
    assert!(
        abv.contains("b:ADOPT,geo=2.0000") && abv.contains(";c:NO_TARGETS,"),
        "{abv}"
    );
    let end = lines.last().cloned().unwrap_or_default();
    assert!(
        end.starts_with("AB#\tEND\t") && end.ends_with("\tcases=2\tinvalid_cases=0"),
        "{end}"
    );
}

#[test]
fn test_gemm_ab_chain_calls_must_use_the_variant_handed_in() {
    let spec = fake_spec();
    let mut engine = FakeEngine::new(&COSTS, &["b"], 4);
    engine.time_with_base = true;
    let settings = fake_settings(4, 0);
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fake_case(&spec, &settings, &mut engine)
    }));
    let message = run.err().and_then(|e| e.downcast_ref::<String>().cloned());
    let message = message.unwrap_or_default();
    assert!(message.contains("calls ran with the plan of"), "{message}");
}
