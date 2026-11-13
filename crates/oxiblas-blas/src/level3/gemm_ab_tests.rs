//! In-crate A/B timing harness for the GEMM policy fields.
//!
//! Every optional GEMM path of 0.2.4 sits behind one field of [`GemmPolicy`];
//! the shipped default [`LEGACY_POLICY`] reproduces 0.2.3 bit for bit. A field
//! may be switched on only where an A/B measurement passes the keep/revert
//! rule, and this module is that measurement. It is a test module because its
//! hooks exist only in test builds (the per-thread packing-cache limit, the
//! second reading of the issue #23 B-block caps).
//!
//! # What a group does
//!
//! A group (`#[test] #[ignore = "timing harness"] fn ab_<group>`) compares
//! plans built from [`LEGACY_POLICY`] with one field, or one coherent set of
//! fields, changed, all run through `gemm_planned` with the default blocking
//! (alpha 1, beta 0). For each case (element type, transposes, `m × k` by
//! `k × n`):
//!
//! 1. **Correctness before timing.** Every variant computes the product once
//!    into a NaN-poisoned C, checked against the base variant A: bit for bit
//!    where the plan promises equal bits (blocked route: kernel shape, pack
//!    scope, B caps, cache limit; and the control A′), and componentwise
//!    `|C − C_A| <= 2·γ_{k+2}/(1 − γ_k) · |op(A)|·|op(B)|` where the route
//!    changes (thin, small SIMD kernels, small transposed copy). It also checks
//!    that each variant takes its path on the target cases (route, f32 kernel,
//!    pack scope; the B caps are checked by the dry run, NC is in `detail`). A
//!    wrong result or a variant that does not engage panics: nothing is timed.
//! 2. **Timing.** A repeat count is calibrated so that one sample of the
//!    fastest variant lasts at least the minimum sample time; then the
//!    variants run interleaved, round after round, in the order A, B, …, A′
//!    (A′ is A's plan, built separately). A variant with its own packing-cache
//!    limit sets it before each sample and makes one untimed call first.
//!
//! The parallel groups run once per thread count, each in an installed rayon
//! pool of exactly that size (`Par::RayonWith(n)` only sizes the partition;
//! the work runs on the ambient pool). A changed cache limit is set on the
//! calling thread and on every worker (`ThreadPool::broadcast`), each thread
//! keeping its (`!Send`) guard in a thread-local for the run.
//!
//! | Group | Candidates (A = LEGACY) | Only on |
//! |---|---|---|
//! | `ab_seq_f32_kernel` | `adaptive` (`f32_avx2`), `forced16x6` | x86-64, AVX2 + FMA |
//! | `ab_seq_small_simd` | `simd<=32^3`, `simd<=80^3` | x86-64, AVX2 + FMA |
//! | `ab_seq_small_trans_copy` | `copy<=32^3` (x86 also `copy<=80^3`) | x86-64, aarch64 |
//! | `ab_seq_thin` | `thin` | |
//! | `ab_seq_b_cap` | `cap_seq`, `cap_seq_alt` (both readings of the rule) | |
//! | `ab_seq_pack_cache` | `limit16MiB`, `limit32MiB`; runs `cap_off`, `cap_on` | |
//! | `ab_par_pack_scope` | `per_tile`, `per_tile_16MiB` | `parallel` |
//! | `ab_par_thin` | `thin` | `parallel` |
//! | `ab_par_b_cap` | `cap_all`, `cap_all_alt` | `parallel` |
//!
//! Each case is a *target* or an *elsewhere* case for each candidate: in the
//! route groups a target is where the pure router takes another path at the
//! architecture's SIMD tier; for the f32 kernel, where the padding model picks
//! 16×6; for the pack scope, above the parallel gate; the B-cap and cache
//! groups name their targets (issue #23 thin-M shapes, large squares).
//!
//! # Running
//!
//! `cargo test -p oxiblas-blas --release --features parallel --lib --
//! --ignored --test-threads=1 --nocapture <filters>`, where the filters select
//! groups by name (`ab_seq_thin`, `ab_par_`, `ab_` for all). On x86-64 set
//! `OXIBLAS_REQUIRE_AVX2=1` so that the AVX2 groups cannot pass by skipping.
//! Settings: `OXIBLAS_AB_ROUNDS` (default 21, at least 3),
//! `OXIBLAS_AB_MIN_SAMPLE_MS` (default 2), `OXIBLAS_AB_THREADS` (thread
//! counts of the parallel groups, comma-separated; default: available cores),
//! `OXIBLAS_AB_QUICK` (non-empty: a small subset of each group, for smoke
//! runs), `OXIBLAS_AB_OUT` (a file the lines are also appended to; choose a
//! path under `target/`). Nothing else is written.
//!
//! # Output (standard output, tab-separated)
//!
//! - `AB#\tAB\t…` and `AB#\tABV\t…`: the column names ([`AB_COLUMNS`],
//!   [`ABV_COLUMNS`]), once per process; `AB#\tSET\t…`: settings and tier of
//!   a group run.
//! - `AB\t…`: one line per (group run, case, variant). Times are per call in
//!   ns (median and quartiles of the samples); `gflops` = `2·m·n·k / median`;
//!   `ratio` = `median(A) / median(variant)` (above 1 = faster than A);
//!   `aa_spread` = the case's `ratio(A′) − 1`; `detail` = route, kernel, KC,
//!   NC (sequential / parallel driver) and cache limit of the variant.
//! - `ABV\t…`: one verdict line per group run (per thread count, or per cap
//!   setting for the cache group); `verdicts` holds one `;`-separated entry per
//!   candidate: `name:DECISION,geo=…,min_t=…,min_e=…,nt=…,ne=…,partial=…`.
//!
//! # Keep/revert rule ([`verdict`])
//!
//! Valid only if every A/A′ spread is within ±1.5 % (`INVALID` otherwise).
//! `ADOPT` when the geometric mean over the targets is at least 1.05, every
//! target at least 1.00 and every elsewhere case at least 0.97; otherwise
//! `KEEP_LEGACY`. `partial=LO..HI/N` is the largest `m·n·k` range (cases
//! ordered by size, equal sizes never split) in which the rule passes, the
//! candidate for a narrower threshold; `none` if no range passes. For the pack
//! scope the threshold is a thread count: compare the thread counts' lines.
//! That the full suite passes under the candidate policy is checked elsewhere.
//!
//! # Protocol for a real measurement
//!
//! A quiet machine (no builds, no other agents), plugged in, Low Power Mode
//! off, `caffeinate -dims` on macOS; on Linux the performance governor and the
//! sequential groups pinned (`taskset -c 2`). Run each group twice in a
//! different order and trust a verdict only when both runs are valid and
//! agree. Timings taken under Rosetta 2, in a debug build, or on a busy
//! machine are meaningless; such runs only show that the harness works.

use super::autotune::{ISSUE_23_B_CAPS, ISSUE_23_B_CAPS_SHALLOW_SINGLE_PASS};
use super::gemm::{BlockingSource, call_blocking, gemm_planned};
use super::gemm_packing::{PACK_CACHE_MAX_BYTES, PackCacheLimitGuard, set_pack_cache_limit};
use super::gemm_plan::{
    BCapScope, GemmPlan, GemmPolicy, GemmRoute, LEGACY_POLICY, RouteInputs, SimdTier, choose_route,
};
use super::trsm::Trans;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::Mat;
use std::cell::RefCell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::gemm_plan::{CANDIDATE_AARCH64, CANDIDATE_X86};
#[cfg(feature = "parallel")]
use super::{gemm::PARALLEL_GATE, gemm_parallel::scope_runs, gemm_plan::PackScope};
#[cfg(target_arch = "x86_64")]
use super::{
    gemm_kernel::{MicroKernel, MicroKernelShape, f32_avx2_prefers_8x8, kernel_for_problem},
    gemm_plan::{F32Avx2Policy, avx2_fma_tier_or_skip},
    gemm_small::SMALL_THRESHOLD,
};

/// The settings of a run: interleaved rounds per case (at least 3), least
/// sample duration, thread counts of the parallel groups, quick subset only,
/// and a file the lines are appended to.
#[derive(Debug, Clone, PartialEq)]
struct Settings(usize, Duration, Vec<usize>, bool, Option<PathBuf>);

/// `value` of the setting `key` parsed as `V`.
fn parse_setting<V: std::str::FromStr>(key: &str, value: &str) -> Result<V, String>
where
    V::Err: std::fmt::Display,
{
    let parsed = value.trim().parse();
    parsed.map_err(|e| format!("{key}={value:?}: {e}"))
}

impl Settings {
    /// Settings from `lookup` (an environment), with `cores` as the default
    /// thread count. An empty value counts as unset.
    fn from_lookup(lookup: impl Fn(&str) -> Option<String>, cores: usize) -> Result<Self, String> {
        let get = |key: &str| lookup(key).filter(|v| !v.trim().is_empty());
        let rounds = match get("OXIBLAS_AB_ROUNDS") {
            None => 21,
            Some(v) => parse_setting::<usize>("OXIBLAS_AB_ROUNDS", &v)?.max(3),
        };
        let min_ms = match get("OXIBLAS_AB_MIN_SAMPLE_MS") {
            None => 2.0,
            Some(v) => match parse_setting::<f64>("OXIBLAS_AB_MIN_SAMPLE_MS", &v)? {
                ms if ms.is_finite() && ms > 0.0 && ms <= 60_000.0 => ms,
                _ => return Err(format!("OXIBLAS_AB_MIN_SAMPLE_MS={v:?}: not in (0, 60000]")),
            },
        };
        let thread_count = |t: &str| match parse_setting::<usize>("OXIBLAS_AB_THREADS", t)? {
            0 => Err("OXIBLAS_AB_THREADS: a thread count is 0".to_string()),
            t => Ok(t),
        };
        let threads = match get("OXIBLAS_AB_THREADS") {
            None => vec![cores.max(1)],
            Some(v) => v.split(',').map(thread_count).collect::<Result<_, _>>()?,
        };
        let min_sample = Duration::from_secs_f64(min_ms / 1000.0);
        let (quick, out) = (get("OXIBLAS_AB_QUICK").is_some(), get("OXIBLAS_AB_OUT"));
        Ok(Self(
            rounds,
            min_sample,
            threads,
            quick,
            out.map(PathBuf::from),
        ))
    }

    /// Settings from the process environment; a malformed value fails the run.
    fn from_env() -> Self {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        let settings = Self::from_lookup(|key| std::env::var(key).ok(), cores);
        settings.unwrap_or_else(|e| panic!("A/B harness settings: {e}"))
    }
}

/// Upper bound of the calibrated repeat count.
const MAX_REPS: usize = 1 << 24;

/// Runs one variant `reps` times and reports how long the timed calls took.
trait SampleRunner {
    /// Number of variants, in interleaving order (A, B, …, A′).
    fn variant_count(&self) -> usize;
    /// One sample: `reps` timed calls of `variant`.
    fn run(&mut self, variant: usize, reps: usize) -> Duration;
}

/// The smallest repeat count found (by extrapolation, at most [`MAX_REPS`])
/// for which one sample of the fastest variant lasts at least `min`.
fn calibrate(runner: &mut impl SampleRunner, min: Duration) -> usize {
    let mut reps = 1_usize;
    loop {
        let times = (0..runner.variant_count()).map(|v| runner.run(v, reps));
        let fastest = times.min().unwrap_or(Duration::ZERO);
        if fastest >= min || reps >= MAX_REPS {
            return reps;
        }
        let scale = min.as_secs_f64() / fastest.as_secs_f64() * 1.1;
        let estimate = (reps as f64 * scale).ceil() as usize;
        reps = estimate.clamp(reps + 1, reps.saturating_mul(16));
        reps = reps.min(MAX_REPS);
    }
}

/// `rounds` rounds with the variants interleaved in their order; per variant,
/// the time per call (ns) of each round.
fn sample_rounds(runner: &mut impl SampleRunner, rounds: usize, reps: usize) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(rounds); runner.variant_count()];
    for _ in 0..rounds {
        for (v, samples) in out.iter_mut().enumerate() {
            let elapsed = runner.run(v, reps);
            samples.push(elapsed.as_secs_f64() * 1e9 / reps.max(1) as f64);
        }
    }
    out
}

/// Median, 25th and 75th percentile of one variant's per-call times.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Stats(f64, f64, f64);

/// The `q`-quantile of ascending `sorted` data, linearly interpolated between
/// the two nearest ranks (NaN for no data).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return f64::NAN;
    };
    let pos = q.clamp(0.0, 1.0) * last as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(last);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Median and quartiles of `samples`.
fn stats(samples: &[f64]) -> Stats {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q = |q| percentile(&sorted, q);
    Stats(q(0.5), q(0.25), q(0.75))
}

/// Geometric mean of positive finite values (NaN when empty or when a value
/// is not positive and finite).
fn geomean(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
        return f64::NAN;
    }
    (values.iter().map(|v| v.ln()).sum::<f64>() / values.len() as f64).exp()
}

/// Largest |A/A′ spread| of a valid run.
const SPREAD_LIMIT: f64 = 0.015;
/// Least geometric mean of the target ratios for adoption.
const ADOPT_GEOMEAN: f64 = 1.05;
/// Least ratio of every target case for adoption.
const TARGET_MIN: f64 = 1.00;
/// Least ratio of every elsewhere case for adoption.
const ELSEWHERE_MIN: f64 = 0.97;

/// What a case is for one candidate: a target (the variant changes what runs
/// here and has to win) or elsewhere (it must not lose).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Target,
    Elsewhere,
}

/// `Target` when `target` holds, `Elsewhere` otherwise.
const fn role_if(target: bool) -> Role {
    [Role::Elsewhere, Role::Target][target as usize]
}

/// One case's result for one candidate: `m·n·k` (orders the cases for
/// partial adoption), role, and `median(A) / median(variant)`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CaseOutcome(u128, Role, f64);

/// The rule's figures over a set of cases: geometric mean of the target
/// ratios, least target and elsewhere ratios, target and elsewhere counts,
/// and whether there is a target and every adoption clause holds.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RuleCheck(f64, f64, f64, usize, usize, bool);

/// The adoption clauses of the rule over `cases`.
fn rule_check<'a>(cases: impl Iterator<Item = &'a CaseOutcome>) -> RuleCheck {
    let (mut target, mut elsewhere) = (Vec::new(), Vec::new());
    for &CaseOutcome(_, role, ratio) in cases {
        match role {
            Role::Target => target.push(ratio),
            Role::Elsewhere => elsewhere.push(ratio),
        }
    }
    let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let (geo, min_t, min_e) = (geomean(&target), min(&target), min(&elsewhere));
    let finite = target.iter().chain(&elsewhere).all(|r| r.is_finite());
    let clauses = geo >= ADOPT_GEOMEAN && min_t >= TARGET_MIN && min_e >= ELSEWHERE_MIN;
    let passes = !target.is_empty() && finite && clauses;
    RuleCheck(geo, min_t, min_e, target.len(), elsewhere.len(), passes)
}

/// The decision for one candidate: `Invalid` (an A/A′ spread outside
/// ±[`SPREAD_LIMIT`]), `Adopt` (every clause holds), `KeepLegacy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decision {
    Invalid,
    Adopt,
    KeepLegacy,
}

/// A size range `[lo, hi]` of `m·n·k` and its number of cases.
type Range = (u128, u128, usize);

/// The verdict of one candidate over one group run: the decision, the rule's
/// figures, and the largest `m·n·k` range in which the rule passes (`None`
/// when none does or the run is invalid).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Verdict(Decision, RuleCheck, Option<Range>);

/// The keep/revert rule over one group run: `spreads` are the A/A′ spreads of
/// all its cases, `cases` the outcomes of one candidate.
fn verdict(spreads: &[f64], cases: &[CaseOutcome]) -> Verdict {
    let check = rule_check(cases.iter());
    let in_limit = |s: &f64| s.is_finite() && s.abs() <= SPREAD_LIMIT;
    let valid = !spreads.is_empty() && spreads.iter().all(in_limit);
    match (valid, check.5) {
        (false, _) => Verdict(Decision::Invalid, check, None),
        (true, true) => Verdict(Decision::Adopt, check, largest_passing_range(cases)),
        (true, false) => Verdict(Decision::KeepLegacy, check, largest_passing_range(cases)),
    }
}

/// The range of sizes with the most cases (then the most targets, then the
/// smallest sizes) whose cases pass the rule.
fn largest_passing_range(cases: &[CaseOutcome]) -> Option<Range> {
    let mut works: Vec<u128> = cases.iter().map(|c| c.0).collect();
    works.sort_unstable();
    works.dedup();
    let mut best: Option<(Range, usize)> = None;
    for (i, &lo) in works.iter().enumerate() {
        for &hi in &works[i..] {
            let inside = || cases.iter().filter(|c| lo <= c.0 && c.0 <= hi);
            let (check, count) = (rule_check(inside()), inside().count());
            let better = best.is_none_or(|((_, _, n), t)| (count, check.3) > (n, t));
            if check.5 && better {
                best = Some(((lo, hi, count), check.3));
            }
        }
    }
    best.map(|(range, _)| range)
}

/// Columns of an `AB` line, in order.
const AB_COLUMNS: &str = "group run threads case dtype trans m k n variant role check reps \
                          rounds median_ns p25_ns p75_ns gflops ratio aa_spread detail";

/// Columns of an `ABV` line, in order.
const ABV_COLUMNS: &str = "group run threads tier valid max_abs_spread verdicts";

/// The two column-header lines.
fn header_lines() -> [String; 2] {
    let line = |kind: &str, cols: &str| {
        let names: Vec<&str> = cols.split_whitespace().collect();
        format!("AB#\t{kind}\t{}", names.join("\t"))
    };
    [line("AB", AB_COLUMNS), line("ABV", ABV_COLUMNS)]
}

/// A line of kind `kind` with `fields` in column order (a tab inside a field
/// becomes a space).
fn data_line(kind: &str, fields: &[String]) -> String {
    let fields: Vec<String> = fields.iter().map(|f| f.replace('\t', " ")).collect();
    format!("{kind}\t{}", fields.join("\t"))
}

/// Splits a harness line into its kind (`AB`, `ABV`, `AB#`) and fields.
fn parse_line(line: &str) -> Option<(&str, Vec<&str>)> {
    let mut parts = line.split('\t');
    let kind = parts.next()?;
    matches!(kind, "AB" | "ABV" | "AB#").then(|| (kind, parts.collect()))
}

/// Appends `lines` to the file at `path` (created if missing).
fn append_lines(path: &Path, lines: &[String]) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    let mut file = options.create(true).append(true).open(path)?;
    lines.iter().try_for_each(|line| writeln!(file, "{line}"))
}

/// Prints `lines` and appends them to the output file of `settings`.
fn emit(settings: &Settings, lines: &[String]) {
    lines.iter().for_each(|line| println!("{line}"));
    if let Some(path) = &settings.4 {
        let shown = path.display();
        let appended = append_lines(path, lines);
        appended.unwrap_or_else(|e| panic!("OXIBLAS_AB_OUT={shown}: {e}"));
    }
}

/// The `verdicts` entry of one candidate.
fn verdict_entry(name: &str, v: &Verdict) -> String {
    let partial = match v.2 {
        Some((lo, hi, n)) => format!("{lo}..{hi}/{n}"),
        None => "none".to_string(),
    };
    let decision = match v.0 {
        Decision::Invalid => "INVALID",
        Decision::Adopt => "ADOPT",
        Decision::KeepLegacy => "KEEP_LEGACY",
    };
    let RuleCheck(geo, min_t, min_e, nt, ne, _) = v.1;
    let figures = format!("geo={geo:.4},min_t={min_t:.4},min_e={min_e:.4},nt={nt},ne={ne}");
    format!("{name}:{decision},{figures},partial={partial}")
}

/// Element type of a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    F32,
    F64,
}

/// The real element types the harness runs.
trait AbReal: Field + super::gemm_kernel::GemmKernel + bytemuck::Zeroable + Send + Sync {
    const UNIT_ROUNDOFF: f64;
    fn of(x: f64) -> Self;
    fn val(self) -> f64;
    fn bits(self) -> u64;
}

impl AbReal for f64 {
    const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;
    fn of(x: f64) -> Self {
        x
    }
    fn val(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl AbReal for f32 {
    const UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;
    fn of(x: f64) -> Self {
        x as f32
    }
    fn val(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// A deterministic value in `[-1, 1)` for element `(i, j)` of operand `seed`
/// (splitmix64 of the position).
fn sample_value(i: usize, j: usize, seed: u64) -> f64 {
    let (i, j) = (i as u64, j as u64);
    let mut z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = z.wrapping_add(j.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    z = z.wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
}

/// Rows and columns of the stored matrix of an `rows × cols` `op(X)`.
const fn stored_shape(trans: Trans, rows: usize, cols: usize) -> (usize, usize) {
    match trans {
        Trans::NoTrans => (rows, cols),
        Trans::Trans | Trans::ConjTrans => (cols, rows),
    }
}

/// The stored operand of an `rows × cols` logical operand `op(X)`.
fn operand<T: AbReal>(trans: Trans, rows: usize, cols: usize, seed: u64) -> Mat<T> {
    let (r, c) = stored_shape(trans, rows, cols);
    let mut x = Mat::<T>::zeros(r, c);
    for j in 0..c {
        (0..r).for_each(|i| x[(i, j)] = T::of(sample_value(i, j, seed)));
    }
    x
}

/// `|op(x)[i, j]|` in f64.
fn op_abs<T: AbReal>(x: &Mat<T>, trans: Trans, i: usize, j: usize) -> f64 {
    let (r, c) = stored_shape(trans, i, j);
    x[(r, c)].val().abs()
}

/// `|op(A)|·|op(B)|` (`m × n`, column-major, f64) for the transposes `ts`,
/// in a loop order that walks the stored operands along their columns.
fn abs_product<T: AbReal>(ts: (Trans, Trans), ab: [&Mat<T>; 2], dims: [usize; 3]) -> Vec<f64> {
    let ([m, k, n], (ta, tb), [a, b]) = (dims, ts, ab);
    let mut s = vec![0.0_f64; m * n];
    let mut axpy = |p: usize, j: usize| {
        let bv = op_abs(b, tb, p, j);
        (0..m).for_each(|i| s[i + j * m] += op_abs(a, ta, i, p) * bv);
    };
    if tb != Trans::NoTrans {
        // op(B)[p, j] = B[j, p]: contiguous in j.
        (0..k).for_each(|p| (0..n).for_each(|j| axpy(p, j)));
    } else if ta == Trans::NoTrans {
        (0..n).for_each(|j| (0..k).for_each(|p| axpy(p, j)));
    } else {
        // Dot form: A[p, i] and B[p, j] are both contiguous in p.
        for (idx, out) in s.iter_mut().enumerate() {
            let (i, j) = (idx % m, idx / m);
            let term = |p| op_abs(a, ta, i, p) * op_abs(b, tb, p, j);
            *out = (0..k).map(term).sum();
        }
    }
    s
}

/// `2·γ_{k+2}/(1 − γ_k)` for unit roundoff `u`: the componentwise distance two
/// results of the same `k`-term product may have, relative to
/// `|op(A)|·|op(B)|` computed in floating point (all terms are positive, so
/// that is within `1/(1 − γ_k)` of exact). Infinite when `k·u` is too large.
fn rounding_factor(k: usize, u: f64) -> f64 {
    let gamma = |j: f64| match j * u {
        ju if ju < 0.5 => ju / (1.0 - ju),
        _ => f64::INFINITY,
    };
    match gamma(k as f64) {
        gk if gk < 1.0 => 2.0 * gamma(k as f64 + 2.0) / (1.0 - gk),
        _ => f64::INFINITY,
    }
}

/// How a variant's result is compared with the base variant's: every element
/// bit for bit, or `|got − reference| <= factor · scale` componentwise
/// (`scale` column-major `m × n`).
#[derive(Debug, Clone, Copy)]
enum Compare<'a> {
    Bitwise,
    Bound(&'a [f64], f64),
}

/// Checks `got` against `reference`; the error names the first mismatch.
fn compare_results<T: AbReal>(
    got: &Mat<T>,
    reference: &Mat<T>,
    how: Compare,
) -> Result<(), String> {
    let (m, n) = (reference.nrows(), reference.ncols());
    let (gm, gn) = (got.nrows(), got.ncols());
    if (gm, gn) != (m, n) {
        return Err(format!("shape {gm}x{gn} against {m}x{n}"));
    }
    for j in 0..n {
        for i in 0..m {
            let (g, r) = (got[(i, j)], reference[(i, j)]);
            let (gv, gb, rv, rb) = (g.val(), g.bits(), r.val(), r.bits());
            let (ok, rule) = match how {
                Compare::Bitwise => (gb == rb, "bitwise".to_string()),
                Compare::Bound(scale, factor) => {
                    let tol = factor * scale.get(i + j * m).copied().unwrap_or(f64::NAN);
                    let ok = gv.is_finite() && (gv - rv).abs() <= tol;
                    (ok, format!("bound {tol:e}"))
                }
            };
            if !ok {
                let at = format!("({i}, {j}): {gv:e} (bits {gb:#x})");
                return Err(format!("{at} against {rv:e} (bits {rb:#x}), {rule}"));
            }
        }
    }
    Ok(())
}

/// The micro-kernel of a variant's plans: the plan's own choice, or (x86-64)
/// the f32 16×6 AVX2 + FMA kernel on every case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelPick {
    Plan,
    #[cfg(target_arch = "x86_64")]
    F32x16x6,
}

/// One side of an A/B comparison. `cache_limit`: the packing-cache limit of
/// the calling thread and of every pool worker while it runs (`None`: the
/// default).
#[derive(Debug, Clone, Copy)]
struct Variant {
    name: &'static str,
    policy: GemmPolicy,
    kernel: KernelPick,
    cache_limit: Option<usize>,
}

/// [`LEGACY_POLICY`] changed by `edit`.
fn legacy_with(edit: impl FnOnce(&mut GemmPolicy)) -> GemmPolicy {
    let mut policy = LEGACY_POLICY;
    edit(&mut policy);
    policy
}

impl Variant {
    /// The variant `name` with `policy`.
    const fn with(name: &'static str, policy: GemmPolicy) -> Self {
        let (kernel, cache_limit) = (KernelPick::Plan, None);
        Self {
            name,
            policy,
            kernel,
            cache_limit,
        }
    }

    /// [`LEGACY_POLICY`] as it ships.
    const fn legacy(name: &'static str) -> Self {
        Self::with(name, LEGACY_POLICY)
    }

    /// This variant renamed, with the packing-cache limit `mib` MiB.
    const fn limit(mut self, name: &'static str, mib: usize) -> Self {
        self.name = name;
        self.cache_limit = Some(mib * MIB);
        self
    }

    /// The variant's plan for an `m × k` by `k × n` product.
    fn plan<T: AbReal>(&self, m: usize, n: usize, k: usize) -> GemmPlan<T> {
        let plan = GemmPlan::<T>::for_problem(self.policy, m, n, k);
        match self.kernel {
            KernelPick::Plan => plan,
            #[cfg(target_arch = "x86_64")]
            KernelPick::F32x16x6 => {
                let kernel = forced_f32_16x6::<T>();
                GemmPlan { kernel, ..plan }
            }
        }
    }
}

/// The f32 16×6 kernel (AVX2 + FMA code: call only on that tier).
#[cfg(target_arch = "x86_64")]
fn forced_f32_16x6<T: AbReal>() -> MicroKernel<T> {
    // A large square: the padding model always picks 16×6 for it.
    let (tier, adaptive) = (SimdTier::Avx2Fma, F32Avx2Policy::Adaptive);
    let kernel = kernel_for_problem::<T>(tier, adaptive, 4096, 4096, 4096);
    let shape = MicroKernelShape { mr: 16, nr: 6 };
    assert_eq!(kernel.shape, shape, "16x6 exists only for f32 (8x8 tier)");
    kernel
}

/// One problem of a group: element type, transposes, `m × k` by `k × n`,
/// whether it is in the quick subset, whether the group names it a target
/// (B-cap and cache groups), and its role for each candidate (`variants[1..]`).
#[derive(Debug, Clone)]
struct Case {
    dtype: Dtype,
    trans_a: Trans,
    trans_b: Trans,
    m: usize,
    k: usize,
    n: usize,
    quick: bool,
    named_target: bool,
    roles: Vec<Role>,
}

impl Case {
    fn work(&self) -> u128 {
        self.m as u128 * self.n as u128 * self.k as u128
    }

    fn trans(&self) -> String {
        let letter = |t| ["N", "T", "C"][t as usize];
        letter(self.trans_a).to_string() + letter(self.trans_b)
    }

    fn dtype_name(&self) -> &'static str {
        ["f32", "f64"][self.dtype as usize]
    }

    fn label(&self) -> String {
        let (m, k, n) = (self.m, self.k, self.n);
        format!("{}_{m}x{k}x{n}_{}", self.trans(), self.dtype_name())
    }

    /// Router inputs for plain column-major operands at `tier`.
    fn route_inputs(&self, tier: SimdTier) -> RouteInputs {
        let (m, n, k, trans_a, trans_b) = (self.m, self.n, self.k, self.trans_a, self.trans_b);
        let (a_unit_col_stride, b_unit_col_stride, simd_real) = (true, true, true);
        let (a_unit_row_stride, b_unit_row_stride) = (false, false);
        RouteInputs {
            m,
            n,
            k,
            trans_a,
            trans_b,
            a_unit_col_stride,
            a_unit_row_stride,
            b_unit_col_stride,
            b_unit_row_stride,
            simd_real,
            tier,
        }
    }
}

/// Cases from a compact table, one per token and element type of `dtypes`.
///
/// A token is a transpose pair (`NN`, `TN`, `NT`, `TT`), which applies to the
/// tokens after it, or `MxKxN` with optional flags: `t` (a named target),
/// `q` (quick for every type), `d` (quick for f64), `s` (quick for f32).
fn cases(dtypes: &[Dtype], table: &str) -> Vec<Case> {
    let mut out = Vec::new();
    for &dtype in dtypes {
        let (mut trans_a, mut trans_b) = (Trans::NoTrans, Trans::NoTrans);
        for token in table.split_whitespace() {
            let letter = |c| [Trans::NoTrans, Trans::Trans][usize::from(c == 'T')];
            if let [a @ ('N' | 'T'), b @ ('N' | 'T')] = token.chars().collect::<Vec<_>>()[..] {
                (trans_a, trans_b) = (letter(a), letter(b));
                continue;
            }
            let digits = token.trim_end_matches(char::is_alphabetic);
            let flags = &token[digits.len()..];
            let dims: Vec<usize> = digits.split('x').filter_map(|d| d.parse().ok()).collect();
            let [m, k, n] = dims[..] else {
                panic!("bad case token {token:?}");
            };
            let own = ['s', 'd'][dtype as usize];
            let quick = flags.contains('q') || flags.contains(own);
            let (named_target, roles) = (flags.contains('t'), Vec::new());
            let case = Case {
                dtype,
                trans_a,
                trans_b,
                m,
                k,
                n,
                quick,
                named_target,
                roles,
            };
            out.push(case);
        }
    }
    out
}

/// What a group's candidates change, which decides the roles and the
/// engagement checks: the route or the small kernels (`true`: only on the
/// architecture's SIMD tier), the f32 kernel shape, NC of the sequential
/// (`false`) or 2D parallel (`true`) driver, the packing-cache limit, or the
/// pack scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engage {
    Route(bool),
    #[cfg(target_arch = "x86_64")]
    Kernel,
    BCap(bool),
    CacheLimit,
    #[cfg(feature = "parallel")]
    Scope,
}

/// The cases and variants of one group run. `run` is `seq`, `t<threads>` or
/// the cache group's cap setting; `variants[0]` is A (the control A′ is added
/// when the group runs).
#[derive(Debug, Clone)]
struct GroupSpec {
    name: &'static str,
    run: String,
    threads: usize,
    par: Par,
    engage: Engage,
    variants: Vec<Variant>,
    cases: Vec<Case>,
}

/// A sequential group run; [`GroupSpec::on`] makes it a parallel one.
fn group(
    name: &'static str,
    engage: Engage,
    variants: Vec<Variant>,
    cases: Vec<Case>,
) -> GroupSpec {
    let (run, threads, par) = ("seq".to_string(), 1, Par::Seq);
    GroupSpec {
        name,
        run,
        threads,
        par,
        engage,
        variants,
        cases,
    }
    .with_roles()
}

impl GroupSpec {
    /// Every candidate promises the bits of A on every case (all groups but
    /// the route groups).
    fn bitwise(&self) -> bool {
        !matches!(self.engage, Engage::Route(_))
    }

    /// This group run on `threads` threads (parallel groups).
    #[cfg(feature = "parallel")]
    fn on(mut self, threads: usize) -> Self {
        (self.run, self.threads) = (format!("t{threads}"), threads);
        self.par = Par::RayonWith(threads);
        self.with_roles()
    }

    /// This group run named `run`.
    fn named(mut self, run: &str) -> Self {
        self.run = run.to_string();
        self
    }

    /// Sets every case's roles from the group's [`Engage`].
    fn with_roles(mut self) -> Self {
        let role = |case: &Case, v: &Variant| match self.engage {
            Engage::Route(_) => route_role(case, v),
            #[cfg(target_arch = "x86_64")]
            Engage::Kernel => role_if(!f32_avx2_prefers_8x8(case.m, case.n, case.k)),
            Engage::BCap(_) | Engage::CacheLimit => role_if(case.named_target),
            #[cfg(feature = "parallel")]
            Engage::Scope => role_if(gate_passes(case, self.par)),
        };
        let roles: Vec<Vec<Role>> = self
            .cases
            .iter()
            .map(|c| self.variants[1..].iter().map(|v| role(c, v)).collect())
            .collect();
        for (case, roles) in self.cases.iter_mut().zip(roles) {
            case.roles = roles;
        }
        self
    }

    /// The cases of this run (all, or the quick subset).
    fn selected(&self, quick: bool) -> impl Iterator<Item = &Case> {
        self.cases.iter().filter(move |c| c.quick || !quick)
    }
}

/// The SIMD tier the candidates of this architecture are designed for.
const DESIGN_TIER: SimdTier = if cfg!(target_arch = "x86_64") {
    SimdTier::Avx2Fma
} else if cfg!(target_arch = "aarch64") {
    SimdTier::Neon
} else {
    SimdTier::Portable
};

/// The route and whether the SIMD small kernels compute it.
fn path_of(inp: &RouteInputs, policy: &GemmPolicy) -> (GemmRoute, bool) {
    let route = choose_route(inp, policy);
    let col_major = inp.a_unit_col_stride && inp.b_unit_col_stride;
    let simd = match route {
        GemmRoute::Small => inp.simd_small(policy) && col_major,
        GemmRoute::SmallTransCopy => true,
        GemmRoute::Thin | GemmRoute::SmallTrans | GemmRoute::Blocked => false,
    };
    (route, simd)
}

/// Route-group role: a target when the candidate takes another path than
/// LEGACY at the design tier.
fn route_role(case: &Case, variant: &Variant) -> Role {
    let inp = case.route_inputs(DESIGN_TIER);
    role_if(path_of(&inp, &variant.policy) != path_of(&inp, &LEGACY_POLICY))
}

/// The parallel gate of the blocked driver passes for `case`.
#[cfg(feature = "parallel")]
fn gate_passes(case: &Case, par: Par) -> bool {
    let work = usize::try_from(case.work()).unwrap_or(usize::MAX);
    PARALLEL_GATE.should_parallelize(work, par)
}

const BOTH: [Dtype; 2] = [Dtype::F64, Dtype::F32];
const MIB: usize = 1 << 20;

/// Shapes of the B-cap groups: the issue #23 thin-M shapes (one MC block) and
/// a shallow one (several MC blocks, one KC block) as targets; small B
/// blocks, deep `k` and squares elsewhere.
const B_CAP_TABLE: &str = "NN 64x512x32768ts 128x256x16384td 32x1024x32768t 1024x256x16384t \
    64x256x1024d 4096x2048x256 1024x1024x1024s 2048x2048x2048 4096x4096x4096";

/// Thin cases: rows `(1, s, s)` and columns `(s, s, 1)` with both transposes
/// of the large operand, dots `(1, k, 1)`, and the non-thin neighbours
/// `(2, s, s)`, `(s, s, 2)`; the first size of each list is quick.
fn thin_table(sizes: &[usize], dots: &[usize], near: &[usize]) -> String {
    let q = |i: usize| if i == 0 { "q" } else { "" };
    let mut table = String::new();
    for (i, s) in sizes.iter().enumerate() {
        let q = q(i);
        table += &format!(" NN 1x{s}x{s}{q} {s}x{s}x1{q} NT 1x{s}x{s}{q} TN {s}x{s}x1{q}");
    }
    for (i, k) in dots.iter().enumerate() {
        table += &format!(" NN 1x{k}x1{0} TT 1x{k}x1{0}", q(i));
    }
    for (i, s) in near.iter().enumerate() {
        table += &format!(" NN 2x{s}x{s}{0} {s}x{s}x2{0}", q(i));
    }
    table
}

/// `ab_seq_f32_kernel`: the f32 AVX2 kernel of the plan; targets where the
/// padding model picks 16×6.
#[cfg(target_arch = "x86_64")]
fn spec_seq_f32_kernel() -> GroupSpec {
    let adaptive = legacy_with(|p| p.f32_avx2 = F32Avx2Policy::Adaptive);
    let mut forced = Variant::legacy("forced16x6");
    forced.kernel = KernelPick::F32x16x6;
    let table = "NN 64x64x64q 128x128x128 256x256x256q 512x512x512 1024x1024x1024 \
        2048x2048x2048 4096x4096x4096 8x4096x4096q 12x4096x4096 16x4096x4096q 24x4096x4096 \
        4096x4096x8 4096x4096x12 4096x4096x16 4096x4096x24 \
        TN 33x33x33q 48x48x48 64x64x64 96x96x96 128x128x128q \
        NT 33x33x33 48x48x48 64x64x64 96x96x96 128x128x128";
    let (base, adaptive) = (
        Variant::legacy("legacy8x8"),
        Variant::with("adaptive", adaptive),
    );
    let cases = cases(&[Dtype::F32], table);
    group(
        "ab_seq_f32_kernel",
        Engage::Kernel,
        vec![base, adaptive, forced],
        cases,
    )
}

/// `ab_seq_small_simd`: the AVX2 + FMA small kernels.
#[cfg(target_arch = "x86_64")]
fn spec_seq_small_simd() -> GroupSpec {
    let up_to = |name, max: usize| {
        let policy = legacy_with(|p| (p.small_simd_x86, p.small_simd_max_work) = (true, max));
        Variant::with(name, policy)
    };
    let table = "NN 5x5x5 8x8x8 12x12x12d 16x16x16 24x24x24 32x32x32 40x40x40 48x48x48s \
        64x64x64 80x80x80 96x96x96d 4x64x4d 5x256x4 8x512x8 16x128x64 64x8x64 40x32x80";
    let small = up_to("simd<=32^3", SMALL_THRESHOLD);
    let wide = up_to("simd<=80^3", CANDIDATE_X86.small_simd_max_work);
    let (variants, cases) = (
        vec![Variant::legacy("off"), small, wide],
        cases(&BOTH, table),
    );
    group("ab_seq_small_simd", Engage::Route(true), variants, cases)
}

/// `ab_seq_small_trans_copy`: transposed small problems copied to
/// column-major for the SIMD small kernels.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn spec_seq_small_trans_copy() -> GroupSpec {
    let mut table = String::from(
        "TN 4x4x4 6x6x6d 8x8x8 12x12x12 16x16x16 24x24x24 32x32x32 16x64x8d 32x8x48 8x128x16 \
         NT 4x4x4 6x6x6 8x8x8 12x12x12 16x16x16s 24x24x24 32x32x32 16x64x8 32x8x48 8x128x16 \
         TT 4x4x4 6x6x6 8x8x8 12x12x12 16x16x16 24x24x24 32x32x32",
    );
    let mut variants = vec![Variant::legacy("off")];
    if cfg!(target_arch = "x86_64") {
        table += " TN 40x40x40 48x48x48 64x64x64 80x80x80 NT 40x40x40 48x48x48 64x64x64 \
                  80x80x80 TT 40x40x40 48x48x48s 64x64x64 80x80x80";
        let range = |hi: usize| {
            CANDIDATE_X86
                .small_trans_copy
                .map(|(lo, t)| (lo, t.max(hi)))
        };
        let copy = |name, hi| {
            let policy =
                legacy_with(|p| (p.small_simd_x86, p.small_trans_copy) = (true, range(hi)));
            Variant::with(name, policy)
        };
        let wide = CANDIDATE_X86.small_simd_max_work;
        variants.extend([copy("copy<=32^3", 0), copy("copy<=80^3", wide)]);
    } else {
        let copy = legacy_with(|p| p.small_trans_copy = CANDIDATE_AARCH64.small_trans_copy);
        variants.push(Variant::with("copy<=32^3", copy));
    }
    let cases = cases(&BOTH, &table);
    group(
        "ab_seq_small_trans_copy",
        Engage::Route(true),
        variants,
        cases,
    )
}

/// `ab_seq_thin` (`ab_par_thin` with `threads`): `m == 1` / `n == 1` as one
/// GEMV, on the sizes `sizes`, dots `dots` and neighbours `near`.
fn spec_thin(name: &'static str, [sizes, dots, near]: [&[usize]; 3]) -> GroupSpec {
    let thin = Variant::with("thin", legacy_with(|p| p.thin = true));
    let cases = cases(&BOTH, &thin_table(sizes, dots, near));
    group(
        name,
        Engage::Route(false),
        vec![Variant::legacy("off"), thin],
        cases,
    )
}

/// `ab_seq_thin`: the sequential thin route.
fn spec_seq_thin() -> GroupSpec {
    let sizes: [&[usize]; 3] = [
        &[256, 1024, 4096, 16_384],
        &[4096, 65_536, 1 << 20],
        &[256, 1024, 4096],
    ];
    spec_thin("ab_seq_thin", sizes)
}

/// A B-cap group with `scope`: both readings of the rule, on [`B_CAP_TABLE`].
fn spec_b_cap(name: &'static str, scope: BCapScope, names: [&'static str; 3]) -> GroupSpec {
    let capped = legacy_with(|p| (p.b_cap, p.b_caps) = (scope, ISSUE_23_B_CAPS));
    let alt = GemmPolicy {
        b_caps: ISSUE_23_B_CAPS_SHALLOW_SINGLE_PASS,
        ..capped
    };
    let (capped, alt) = (
        Variant::with(names[1], capped),
        Variant::with(names[2], alt),
    );
    let variants = vec![Variant::legacy(names[0]), capped, alt];
    let parallel = scope == BCapScope::All;
    group(
        name,
        Engage::BCap(parallel),
        variants,
        cases(&BOTH, B_CAP_TABLE),
    )
}

/// `ab_seq_b_cap`: the issue #23 B-block caps in the sequential driver.
fn spec_seq_b_cap() -> GroupSpec {
    let names = ["cap_off", "cap_seq", "cap_seq_alt"];
    spec_b_cap("ab_seq_b_cap", BCapScope::Sequential, names)
}

/// `ab_seq_pack_cache`: the per-thread cache limit, with the B cap off or on;
/// targets are shapes whose packed blocks can outgrow 8 MiB.
fn spec_seq_pack_cache(cap: BCapScope) -> GroupSpec {
    let base = Variant::with("limit8MiB", legacy_with(|p| p.b_cap = cap));
    let run = if cap == BCapScope::Off {
        "cap_off"
    } else {
        "cap_on"
    };
    let table = "NN 64x512x32768t 128x256x16384td 2048x2048x2048t 4096x4096x4096t 256x256x256d \
        512x512x512 1024x1024x1024";
    let (big, bigger) = (base.limit("limit16MiB", 16), base.limit("limit32MiB", 32));
    let (variants, cases) = (vec![base, big, bigger], cases(&BOTH, table));
    group("ab_seq_pack_cache", Engage::CacheLimit, variants, cases).named(run)
}

/// `ab_par_pack_scope`: shared against per-tile packing on `threads`; targets
/// are above the parallel gate, where the 2D driver runs.
#[cfg(feature = "parallel")]
fn spec_par_pack_scope(threads: usize) -> GroupSpec {
    let per_tile = Variant::with("per_tile", legacy_with(|p| p.per_tile_min_threads = 1));
    // Per tile with room for A and B on every worker (with 8 MiB, A is freed
    // after every task for some shapes).
    let roomy = per_tile.limit("per_tile_16MiB", 16);
    let table = "NN 96x96x96d 256x256x256d 512x512x512s 1024x1024x1024 2048x2048x2048 \
        4096x4096x4096 8192x512x256 256x512x8192 TN 1024x1024x1024 NT 2048x2048x2048";
    let (variants, cases) = (
        vec![Variant::legacy("shared"), per_tile, roomy],
        cases(&BOTH, table),
    );
    group("ab_par_pack_scope", Engage::Scope, variants, cases).on(threads)
}

/// `ab_par_thin`: the parallel thin split on `threads`.
#[cfg(feature = "parallel")]
fn spec_par_thin(threads: usize) -> GroupSpec {
    let sizes: [&[usize]; 3] = [&[1024, 4096, 16_384], &[65_536, 1 << 20], &[1024, 4096]];
    spec_thin("ab_par_thin", sizes).on(threads)
}

/// `ab_par_b_cap`: the B caps in both drivers (`BCapScope::All`) on `threads`.
#[cfg(feature = "parallel")]
fn spec_par_b_cap(threads: usize) -> GroupSpec {
    let names = ["cap_off", "cap_all", "cap_all_alt"];
    spec_b_cap("ab_par_b_cap", BCapScope::All, names).on(threads)
}

/// Every group run this build has (the parallel ones for each of `threads`).
fn all_specs(threads: &[usize]) -> Vec<GroupSpec> {
    let mut specs = vec![spec_seq_thin(), spec_seq_b_cap()];
    specs.extend([BCapScope::Off, BCapScope::Sequential].map(spec_seq_pack_cache));
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    specs.insert(0, spec_seq_small_trans_copy());
    #[cfg(target_arch = "x86_64")]
    {
        specs.insert(0, spec_seq_small_simd());
        specs.insert(0, spec_seq_f32_kernel());
    }
    #[cfg(feature = "parallel")]
    for &t in threads {
        specs.extend([spec_par_pack_scope(t), spec_par_thin(t), spec_par_b_cap(t)]);
    }
    let _ = threads;
    specs
}

thread_local! {
    /// The packing-cache limit guard this thread holds for the A/B run.
    static HELD_LIMIT: RefCell<Option<PackCacheLimitGuard>> = const { RefCell::new(None) };
}

/// Sets this thread's packing-cache limit to `limit` until the next call
/// (`None`: the limit from before the run). The previous guard is dropped
/// first, so it restores the earlier limit.
fn hold_cache_limit(limit: Option<usize>) {
    HELD_LIMIT.with(|held| {
        let mut held = held.borrow_mut();
        *held = None;
        *held = limit.map(set_pack_cache_limit);
    });
}

/// Where a group run executes: the test thread, or an installed pool.
#[derive(Default)]
struct Env<'p> {
    #[cfg(feature = "parallel")]
    pool: Option<&'p rayon::ThreadPool>,
    #[cfg(not(feature = "parallel"))]
    _pool: core::marker::PhantomData<&'p ()>,
}

impl Env<'_> {
    /// Runs `f` where the GEMM calls run (inside the pool when there is one).
    fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        if let Some(pool) = self.pool {
            return pool.install(f);
        }
        f()
    }

    /// Sets the packing-cache limit on the calling thread and on every worker
    /// of the pool.
    fn hold(&self, limit: Option<usize>) {
        hold_cache_limit(limit);
        #[cfg(feature = "parallel")]
        if let Some(pool) = self.pool {
            pool.broadcast(|_| hold_cache_limit(limit));
        }
    }
}

/// `(Shared, PerTile)` runs of the 2D driver on this thread so far.
fn scope_count() -> (usize, usize) {
    #[cfg(feature = "parallel")]
    {
        scope_runs()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0, 0)
    }
}

/// The operands of one case.
struct Problem<'a, T: AbReal>(&'a Case, Mat<T>, Mat<T>, Par);

impl<T: AbReal> Problem<'_, T> {
    /// `C = op(A)·op(B)` with `plan` (alpha 1, beta 0, default blocking).
    fn compute(&self, plan: &GemmPlan<T>, c: &mut Mat<T>) {
        let (Self(case, a, b, par), blocking) = (self, BlockingSource::Default);
        let (ta, tb, one, zero) = (case.trans_a, case.trans_b, T::one(), T::zero());
        let (a, b, c) = (a.as_ref(), b.as_ref(), c.as_mut());
        gemm_planned(ta, tb, one, a, b, zero, c, *par, blocking, plan);
    }

    /// The router inputs of the actual operands on `plan`'s tier.
    fn inputs(&self, plan: &GemmPlan<T>) -> RouteInputs {
        let (c, tier) = (self.0, plan.ctx.tier());
        let (a, b, (ta, tb)) = (self.1.as_ref(), self.2.as_ref(), (c.trans_a, c.trans_b));
        RouteInputs::new(ta, tb, &a, &b, c.m, c.n, c.k, tier)
    }

    /// Route, kernel, KC, NC and cache limit of `plan` (NC for the calling
    /// thread's limit).
    fn detail(&self, plan: &GemmPlan<T>, limit: Option<usize>) -> String {
        let c = self.0;
        let (route, simd) = path_of(&self.inputs(plan), &plan.ctx.policy);
        let simd = if simd { "+simd" } else { "" };
        let blocks = call_blocking(BlockingSource::Default, plan, c.m, c.k, c.n);
        let (seq, shape) = (blocks.sequential(), plan.kernel.shape);
        #[cfg(feature = "parallel")]
        let nc = format!("{}/{}", seq.nc, blocks.parallel().nc);
        #[cfg(not(feature = "parallel"))]
        let nc = seq.nc.to_string();
        let (mr, nr, kc) = (shape.mr, shape.nr, seq.kc);
        let mib = limit.unwrap_or(PACK_CACHE_MAX_BYTES) / MIB;
        format!("route={route:?}{simd} kernel={mr}x{nr} kc={kc} nc={nc} limit={mib}MiB")
    }
}

/// The timed side of one case: the variants' plans on shared operands. With
/// `switch_state` (the variants differ in the cache limit) each sample sets
/// the limit and makes one untimed call first.
struct GemmRunner<'a, 'p, T: AbReal> {
    problem: &'a Problem<'a, T>,
    plans: &'a [GemmPlan<T>],
    limits: &'a [Option<usize>],
    c: &'a mut Mat<T>,
    env: &'a Env<'p>,
    switch_state: bool,
}

impl<T: AbReal> SampleRunner for GemmRunner<'_, '_, T> {
    fn variant_count(&self) -> usize {
        self.plans.len()
    }

    fn run(&mut self, variant: usize, reps: usize) -> Duration {
        let warm = self.switch_state;
        if warm {
            self.env.hold(self.limits[variant]);
        }
        let (problem, plan, c) = (self.problem, &self.plans[variant], &mut *self.c);
        self.env.run(move || {
            if warm {
                problem.compute(plan, c);
            }
            let start = Instant::now();
            (0..reps).for_each(|_| problem.compute(plan, c));
            let elapsed = start.elapsed();
            std::hint::black_box(&*c);
            elapsed
        })
    }
}

/// Checks that the variant with `plan` and role `role` takes its path against
/// A's plan `base` (see [`Engage`]); `scopes` are the 2D driver runs of its
/// correctness call.
fn check_engagement<T: AbReal>(
    (spec, problem): (&GroupSpec, &Problem<'_, T>),
    (base, plan, forced): (&GemmPlan<T>, &GemmPlan<T>, bool),
    (role, scopes): (Role, (usize, usize)),
    ctx: &str,
) {
    let target = role == Role::Target;
    match spec.engage {
        Engage::Route(tier_bound) => {
            if !tier_bound || SimdTier::current() == DESIGN_TIER {
                let path = |p: &GemmPlan<T>| path_of(&problem.inputs(p), &p.ctx.policy);
                let changed = path(plan) != path(base);
                assert_eq!(changed, target, "{ctx}: path against role");
            }
        }
        #[cfg(target_arch = "x86_64")]
        Engage::Kernel => {
            let shape = plan.kernel.shape;
            let is_16x6 = shape == (MicroKernelShape { mr: 16, nr: 6 });
            assert_eq!(
                is_16x6,
                target || forced,
                "{ctx}: kernel {shape:?} against role"
            );
        }
        Engage::BCap(_) | Engage::CacheLimit => {}
        #[cfg(feature = "parallel")]
        Engage::Scope => {
            let per_tile_of = |p: &GemmPlan<T>| p.scope(spec.threads) == PackScope::PerTile;
            let per_tile = per_tile_of(plan);
            let changed = per_tile != per_tile_of(base);
            assert!(changed || !target, "{ctx}: the pack scope does not change");
            let want =
                [(0, 0), (0, 0), (1, 0), (0, 1)][usize::from(target) * 2 + usize::from(per_tile)];
            assert_eq!(scopes, want, "{ctx}: 2D driver runs (Shared, PerTile)");
        }
    }
    let _ = (scopes, forced);
}

/// Runs one case of `spec`: correctness first, then the interleaved timing.
/// Returns its lines, its A/A′ spread and the `ratio` of every variant.
fn run_case<T: AbReal>(
    spec: &GroupSpec,
    case: &Case,
    st: &Settings,
    env: &Env<'_>,
) -> (Vec<String>, f64, Vec<f64>) {
    let (m, k, n, ts) = (case.m, case.k, case.n, (case.trans_a, case.trans_b));
    let (a, b) = (operand::<T>(ts.0, m, k, 1), operand::<T>(ts.1, k, n, 2));
    let problem = Problem::<T>(case, a, b, spec.par);
    // A, candidates, A′ (A rebuilt separately).
    let mut variants = spec.variants.clone();
    variants.push(Variant {
        name: "control",
        ..spec.variants[0]
    });
    let control = variants.len() - 1;
    let plans: Vec<GemmPlan<T>> = variants.iter().map(|v| v.plan::<T>(m, n, k)).collect();
    let limits: Vec<Option<usize>> = variants.iter().map(|v| v.cache_limit).collect();
    let role_of = |vi: usize| (vi > 0 && vi < control).then(|| case.roles[vi - 1]);
    let bound = |vi| !spec.bitwise() && role_of(vi) == Some(Role::Target);

    let needs_scale = (1..control).any(bound);
    let scale = needs_scale.then(|| abs_product(ts, [&problem.1, &problem.2], [m, k, n]));
    let factor = rounding_factor(k, T::UNIT_ROUNDOFF);
    let (mut reference, mut scratch) = (Mat::<T>::zeros(m, n), Mat::<T>::zeros(m, n));
    let mut details = Vec::with_capacity(variants.len());
    for (vi, plan) in plans.iter().enumerate() {
        let ctx = format!(
            "{} {} {} {}",
            spec.name,
            spec.run,
            case.label(),
            variants[vi].name
        );
        env.hold(limits[vi]);
        let out = if vi == 0 {
            &mut reference
        } else {
            &mut scratch
        };
        (0..n).for_each(|j| (0..m).for_each(|i| out[(i, j)] = T::of(f64::NAN)));
        let (pr, limit) = (&problem, limits[vi]);
        let (detail, scopes) = env.run(move || {
            let (detail, before) = (pr.detail(plan, limit), scope_count());
            pr.compute(plan, out);
            let after = scope_count();
            (detail, (after.0 - before.0, after.1 - before.1))
        });
        let how = match scale.as_deref() {
            Some(scale) if bound(vi) => Compare::Bound(scale, factor),
            _ => Compare::Bitwise,
        };
        let checked = (vi > 0).then(|| compare_results(&scratch, &reference, how));
        if let Err(e) = checked.transpose() {
            panic!("{ctx}: wrong result, not timed: {e}");
        }
        if let Some(role) = role_of(vi) {
            #[cfg(target_arch = "x86_64")]
            let forced = variants[vi].kernel == KernelPick::F32x16x6;
            #[cfg(not(target_arch = "x86_64"))]
            let forced = false;
            let plans = (&plans[0], plan, forced);
            check_engagement((spec, &problem), plans, (role, scopes), &ctx);
        }
        details.push(detail);
    }

    let switch_state = limits.iter().any(Option::is_some);
    let (problem, plans, limits, c) = (&problem, &plans, &limits, &mut scratch);
    let mut runner = GemmRunner {
        problem,
        plans,
        limits,
        c,
        env,
        switch_state,
    };
    let reps = calibrate(&mut runner, st.1);
    let samples = sample_rounds(&mut runner, st.0, reps);
    env.hold(None);

    let all: Vec<Stats> = samples.iter().map(|s| stats(s)).collect();
    let ratios: Vec<f64> = all.iter().map(|s| all[0].0 / s.0).collect();
    let spread = ratios[control] - 1.0;
    let flops = 2.0 * case.work() as f64;
    let (group, run, threads, rounds) = (spec.name, &spec.run, spec.threads, st.0);
    let (label, dtype, trans) = (case.label(), case.dtype_name(), case.trans());
    let mut lines = Vec::with_capacity(variants.len());
    for (vi, v) in variants.iter().enumerate() {
        let (role, check) = match role_of(vi) {
            Some(Role::Target) => ("target", if bound(vi) { "bound" } else { "bitwise" }),
            Some(Role::Elsewhere) => ("elsewhere", "bitwise"),
            None if vi == 0 => ("base", "-"),
            None => ("control", "bitwise"),
        };
        let (Stats(med, p25, p75), ratio) = (all[vi], ratios[vi]);
        let (variant, gflops, detail) = (v.name, flops / med, details[vi].replace('\t', " "));
        lines.push(format!(
            "AB\t{group}\t{run}\t{threads}\t{label}\t{dtype}\t{trans}\t{m}\t{k}\t{n}\t{variant}\t\
             {role}\t{check}\t{reps}\t{rounds}\t{med:.1}\t{p25:.1}\t{p75:.1}\t{gflops:.3}\t\
             {ratio:.4}\t{spread:+.4}\t{detail}"
        ));
    }
    (lines, spread, ratios)
}

/// Runs every selected case of `spec` and prints its lines and verdict.
fn run_group(spec: &GroupSpec, st: &Settings, env: &Env<'_>) {
    static HEADERS: std::sync::Once = std::sync::Once::new();
    // libtest prints `test <name> ... ` without a line break: start a new
    // line so that every harness line begins at column 0.
    println!();
    HEADERS.call_once(|| emit(st, &header_lines()));
    let (group, run, threads, tier) = (spec.name, &spec.run, spec.threads, SimdTier::current());
    let (arch, debug, ms) = (
        std::env::consts::ARCH,
        cfg!(debug_assertions),
        st.1.as_secs_f64() * 1e3,
    );
    let (rounds, quick) = (st.0, st.3);
    let set = format!(
        "AB#\tSET\tgroup={group}\trun={run}\ttier={tier:?}\tarch={arch}\trounds={rounds}\t\
         min_sample_ms={ms}\tthreads={threads}\tquick={quick}\tdebug_build={debug}"
    );
    emit(st, &[set]);
    let mut spreads = Vec::new();
    let mut outcomes = vec![Vec::new(); spec.variants.len() - 1];
    for case in spec.selected(quick) {
        let (lines, spread, ratios) = match case.dtype {
            Dtype::F64 => run_case::<f64>(spec, case, st, env),
            Dtype::F32 => run_case::<f32>(spec, case, st, env),
        };
        emit(st, &lines);
        spreads.push(spread);
        for (ci, out) in outcomes.iter_mut().enumerate() {
            out.push(CaseOutcome(case.work(), case.roles[ci], ratios[ci + 1]));
        }
    }
    let candidates = spec.variants[1..].iter().zip(&outcomes);
    let entries = candidates.map(|(v, out)| verdict_entry(v.name, &verdict(&spreads, out)));
    let verdicts = entries.collect::<Vec<_>>().join(";");
    let max_spread = spreads.iter().fold(0.0_f64, |a, s| a.max(s.abs()));
    let valid = !spreads.is_empty() && spreads.iter().all(|s| s.abs() <= SPREAD_LIMIT);
    let valid = if valid { "yes" } else { "no" };
    let (threads, tier, spread) = (
        threads.to_string(),
        format!("{tier:?}"),
        format!("{max_spread:.4}"),
    );
    let fields = [group, run, &threads, &tier, valid, &spread, &verdicts].map(String::from);
    emit(st, &[data_line("ABV", &fields)]);
}

/// Runs a sequential group on the test thread.
fn run_seq(spec: &GroupSpec) {
    run_group(spec, &Settings::from_env(), &Env::default());
}

/// Runs a parallel group once per thread count, each in a pool of exactly
/// that many threads.
#[cfg(feature = "parallel")]
fn run_par(build: fn(usize) -> GroupSpec) {
    let settings = Settings::from_env();
    for &threads in &settings.2 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build();
        let pool = pool.unwrap_or_else(|e| panic!("rayon pool of {threads} threads: {e}"));
        let env = Env { pool: Some(&pool) };
        run_group(&build(threads), &settings, &env);
        env.hold(None);
    }
}

/// Declares the timing tests: `name => body`, each with its own `cfg`.
macro_rules! ab_groups {
    ($($(#[$cfg:meta])* $name:ident => $body:expr;)*) => {$(
        $(#[$cfg])*
        #[test]
        #[ignore = "timing harness"]
        fn $name() {
            $body
        }
    )*};
}

ab_groups! {
    #[cfg(target_arch = "x86_64")]
    ab_seq_f32_kernel => if avx2_fma_tier_or_skip("ab_seq_f32_kernel") {
        run_seq(&spec_seq_f32_kernel());
    };
    #[cfg(target_arch = "x86_64")]
    ab_seq_small_simd => if avx2_fma_tier_or_skip("ab_seq_small_simd") {
        run_seq(&spec_seq_small_simd());
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    ab_seq_small_trans_copy => run_seq(&spec_seq_small_trans_copy());
    ab_seq_thin => run_seq(&spec_seq_thin());
    ab_seq_b_cap => run_seq(&spec_seq_b_cap());
    ab_seq_pack_cache => for cap in [BCapScope::Off, BCapScope::Sequential] {
        run_seq(&spec_seq_pack_cache(cap));
    };
    #[cfg(feature = "parallel")]
    ab_par_pack_scope => run_par(spec_par_pack_scope);
    #[cfg(feature = "parallel")]
    ab_par_thin => run_par(spec_par_thin);
    #[cfg(feature = "parallel")]
    ab_par_b_cap => run_par(spec_par_b_cap);
}

/// A runner with a fixed cost per call for each variant, logging the calls.
struct FakeRunner(Vec<Duration>, Vec<(usize, usize)>);

impl SampleRunner for FakeRunner {
    fn variant_count(&self) -> usize {
        self.0.len()
    }

    fn run(&mut self, variant: usize, reps: usize) -> Duration {
        self.1.push((variant, reps));
        self.0[variant] * u32::try_from(reps).unwrap_or(u32::MAX)
    }
}

/// Outcomes from a table of `<work><T|E><ratio>` tokens (`T`: target, `E`:
/// elsewhere), e.g. `"100T1.10 400E0.975"`.
fn outcomes(table: &str) -> Vec<CaseOutcome> {
    let parse = |token: &str| {
        let at = token.find(['T', 'E'])?;
        let (work, ratio) = (token[..at].parse().ok()?, token[at + 1..].parse().ok()?);
        Some(CaseOutcome(
            work,
            role_if(token[at..].starts_with('T')),
            ratio,
        ))
    };
    let all = table.split_whitespace().map(|t| parse(t).ok_or(t));
    all.collect::<Result<_, _>>()
        .unwrap_or_else(|t| panic!("bad outcome {t:?}"))
}

/// The decision and partial range of `table` (see [`outcomes`]) with no
/// A/A′ spread.
fn decide(table: &str) -> (Decision, Option<Range>) {
    let Verdict(decision, _, partial) = verdict(&[0.0], &outcomes(table));
    (decision, partial)
}

#[test]
fn test_gemm_ab_percentile_and_median() {
    let (five, four): (&[f64], &[f64]) = (&[1.0, 2.0, 3.0, 4.0, 5.0], &[10.0, 20.0, 30.0, 40.0]);
    let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
    let want = [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 17.5, 25.0, 32.5, 40.0]];
    for (data, want) in [five, four].into_iter().zip(want) {
        assert_eq!(quantiles.map(|q| percentile(data, q)), want, "{data:?}");
    }
    assert_eq!(
        (percentile(&[7.0], 0.3), percentile(&[], 0.5).is_nan()),
        (7.0, true)
    );
    // `stats` sorts first.
    let s = stats(&[9.0, 1.0, 5.0, 3.0, 7.0, 100.0, 2.0]);
    assert_eq!(s, Stats(5.0, 2.5, 8.0));
}

#[test]
fn test_gemm_ab_geomean() {
    let close = |values: &[f64], want: f64| (geomean(values) - want).abs() < 1e-12;
    assert!(close(&[2.0, 8.0], 4.0) && close(&[1.05; 7], 1.05));
    assert!(close(&[0.5, 2.0, 1.0], 1.0) && close(&[1.0, 1.0, 1.21], 1.21_f64.cbrt()));
    for bad in [&[][..], &[1.0, 0.0], &[1.0, -1.0], &[1.0, f64::INFINITY]] {
        assert!(geomean(bad).is_nan(), "{bad:?}");
    }
}

#[test]
fn test_gemm_ab_rule_adopts() {
    let cases = outcomes("100T1.10 200T1.02 300T1.08 400E0.975 500E1.0");
    let Verdict(decision, check, partial) = verdict(&[0.01, -0.012, 0.0, 0.015, -0.015], &cases);
    let RuleCheck(geo, min_t, min_e, nt, ne, passes) = check;
    assert_eq!((decision, nt, ne, passes), (Decision::Adopt, 3, 2, true));
    assert!((geo - (1.10_f64 * 1.02 * 1.08).cbrt()).abs() < 1e-12);
    assert_eq!((min_t, min_e, partial), (1.02, 0.975, Some((100, 500, 5))));
    // Geometric mean just below 1.05 with every target >= 1.00: keep (the
    // range without the 1.04 case passes).
    let keep = Decision::KeepLegacy;
    assert_eq!(decide("1T1.04 2T1.05 3E1.0"), (keep, Some((2, 3, 2))));
    // No targets: nothing to adopt.
    assert_eq!(decide("1E1.5"), (keep, None));
}

#[test]
fn test_gemm_ab_rule_target_below_one_keeps_legacy() {
    let Verdict(decision, check, _) = verdict(&[0.0], &outcomes("100T1.30 200T0.99 300T1.20"));
    assert!(check.0 >= ADOPT_GEOMEAN);
    assert_eq!((decision, check.1), (Decision::KeepLegacy, 0.99));
    // Exactly 1.00 is enough.
    assert_eq!(decide("1T1.0 2T1.2").0, Decision::Adopt);
}

#[test]
fn test_gemm_ab_rule_elsewhere_below_limit_keeps_legacy() {
    assert_eq!(
        decide("100T1.10 200T1.10 300E0.969").0,
        Decision::KeepLegacy
    );
    assert_eq!(decide("100T1.10 200T1.10 300E0.97").0, Decision::Adopt);
    assert_eq!(decide("100T1.10 200T1.10 300ENaN").0, Decision::KeepLegacy);
}

#[test]
fn test_gemm_ab_rule_invalid_control() {
    let cases = outcomes("1T1.5 2E1.0");
    for spreads in [&[0.0, 0.0151][..], &[-0.016], &[f64::NAN], &[]] {
        let Verdict(decision, _, partial) = verdict(spreads, &cases);
        assert_eq!(
            (decision, partial),
            (Decision::Invalid, None),
            "{spreads:?}"
        );
    }
    assert_eq!(verdict(&[0.015, -0.015], &cases).0, Decision::Adopt);
}

#[test]
fn test_gemm_ab_rule_partial_adoption() {
    let keep = Decision::KeepLegacy;
    // Small sizes win, a large target loses: the range stops below it.
    let first = "64E1.0 512T1.20 512T1.10 4096T1.08 32768T0.95 262144E1.0";
    assert_eq!(decide(first), (keep, Some((64, 4096, 4))));
    // An elsewhere loss splits the ranges; the larger side wins.
    assert_eq!(
        decide("1T1.2 2E0.9 3T1.1 4T1.1 5E1.0"),
        (keep, Some((3, 5, 3)))
    );
    // Equal sizes are never split: one of the two size-8 cases loses.
    assert_eq!(decide("8T1.3 8T0.98 9T1.2"), (keep, Some((9, 9, 1))));
    // Nothing passes anywhere.
    assert_eq!(decide("1T1.01 2T0.9"), (keep, None));
    let entry = verdict_entry("x", &verdict(&[0.0], &outcomes("1T0.9")));
    assert!(entry.starts_with("x:KEEP_LEGACY,") && entry.ends_with(",partial=none"));
}

#[test]
fn test_gemm_ab_sampler_calibrates_and_interleaves() {
    let (us, min) = (Duration::from_micros, Duration::from_millis(2));
    let mut runner = FakeRunner(vec![us(100), us(300), us(100)], Vec::new());
    let reps = calibrate(&mut runner, min);
    // The fastest variant's sample reaches the minimum, and not by far.
    let reached = us(100) * u32::try_from(reps).unwrap_or(0) >= min;
    assert!(reached && reps <= 42, "reps {reps}");
    // Calibration runs every variant with the same count each time.
    for chunk in runner.1.chunks(3) {
        assert_eq!(chunk, [0, 1, 2].map(|v| (v, chunk[0].1)));
    }
    runner.1.clear();
    let samples = sample_rounds(&mut runner, 5, reps);
    let order: Vec<(usize, usize)> = [0, 1, 2].repeat(5).into_iter().map(|v| (v, reps)).collect();
    assert_eq!(runner.1, order);
    for (per_call, cost) in samples.iter().zip(&runner.0) {
        let want = cost.as_secs_f64() * 1e9;
        assert!(per_call.len() == 5 && per_call.iter().all(|t| (t - want).abs() < 1e-6));
    }
    // A free call stops at the repeat cap; a slow one needs one repetition.
    let mut free = FakeRunner(vec![Duration::ZERO; 2], vec![]);
    let mut slow = FakeRunner(vec![us(5000), us(3000)], vec![]);
    assert_eq!(
        (calibrate(&mut free, min), calibrate(&mut slow, min)),
        (MAX_REPS, 1)
    );
}

#[test]
fn test_gemm_ab_output_line_format() {
    let [ab_header, abv_header] = header_lines();
    let ab_cols: Vec<&str> = AB_COLUMNS.split_whitespace().collect();
    let abv_cols: Vec<&str> = ABV_COLUMNS.split_whitespace().collect();
    assert_eq!((ab_cols.len(), abv_cols.len()), (21, 7));
    let (ab_want, abv_want) = (
        [&["AB"][..], &ab_cols].concat(),
        [&["ABV"][..], &abv_cols].concat(),
    );
    assert_eq!(parse_line(&ab_header), Some(("AB#", ab_want)));
    assert_eq!(parse_line(&abv_header), Some(("AB#", abv_want)));
    // A line parsed back gives each value under its column; a tab inside a
    // field becomes a space.
    let values: Vec<String> = ab_cols.iter().map(|c| format!("{c}-value")).collect();
    let mut sent = values.clone();
    sent[20] = "route=Blocked\tkernel=8x8".to_string();
    let line = data_line("AB", &sent);
    let Some(("AB", fields)) = parse_line(&line) else {
        panic!("line {line}");
    };
    assert_eq!(fields.len(), ab_cols.len());
    for (i, (field, value)) in fields.iter().zip(&values).enumerate() {
        let want = if i == 20 {
            "route=Blocked kernel=8x8"
        } else {
            value
        };
        assert_eq!(*field, want, "column {}", ab_cols[i]);
    }
    assert!(parse_line("test foo ... ok").is_none());
    let entry = verdict_entry("thin", &verdict(&[0.0], &outcomes("1T1.1")));
    assert_eq!(
        entry,
        "thin:ADOPT,geo=1.1000,min_t=1.1000,min_e=inf,nt=1,ne=0,partial=1..1/1"
    );
}

#[test]
fn test_gemm_ab_output_appends_to_file() {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
    let name = format!(
        "oxiblas_ab_{}_{}.tsv",
        std::process::id(),
        now.map_or(0, |d| d.as_nanos())
    );
    let path = std::env::temp_dir().join(name);
    let (first, second) = (
        ["AB\ta".to_string()],
        ["ABV\tb".to_string(), "AB\tc".to_string()],
    );
    let result = append_lines(&path, &first).and_then(|()| append_lines(&path, &second));
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string());
    let _ = std::fs::remove_file(&path);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(text, Ok("AB\ta\nABV\tb\nAB\tc\n".to_string()));
}

#[test]
fn test_gemm_ab_settings_from_environment() {
    let parse = |table: &str, cores| {
        let pairs: Vec<(&str, &str)> = table.split(' ').filter_map(|p| p.split_once('=')).collect();
        let env = |key: &str| pairs.iter().find(|p| p.0 == key).map(|p| p.1.to_string());
        Settings::from_lookup(env, cores)
    };
    let two_ms = Duration::from_millis(2);
    assert_eq!(parse("", 6), Ok(Settings(21, two_ms, vec![6], false, None)));
    let all = "OXIBLAS_AB_ROUNDS=31 OXIBLAS_AB_MIN_SAMPLE_MS=0.5 OXIBLAS_AB_THREADS=2,4,8 \
               OXIBLAS_AB_QUICK=1 OXIBLAS_AB_OUT=target/ab.tsv";
    let (half_ms, out) = (
        Duration::from_micros(500),
        Some(PathBuf::from("target/ab.tsv")),
    );
    assert_eq!(
        parse(all, 6),
        Ok(Settings(31, half_ms, vec![2, 4, 8], true, out))
    );
    // At least three rounds; empty values count as unset.
    let low = parse(
        "OXIBLAS_AB_ROUNDS=1 OXIBLAS_AB_QUICK= OXIBLAS_AB_THREADS=",
        0,
    );
    assert_eq!(low, Ok(Settings(3, two_ms, vec![1], false, None)));
    let bad = "OXIBLAS_AB_ROUNDS=many OXIBLAS_AB_MIN_SAMPLE_MS=0 OXIBLAS_AB_MIN_SAMPLE_MS=-1 \
               OXIBLAS_AB_MIN_SAMPLE_MS=NaN OXIBLAS_AB_THREADS=4,0 OXIBLAS_AB_THREADS=4,,8 \
               OXIBLAS_AB_THREADS=x";
    for pair in bad.split_whitespace() {
        assert!(parse(pair, 4).is_err(), "{pair}");
    }
}

#[test]
fn test_gemm_ab_compare_bitwise_and_bound() {
    let base = operand::<f64>(Trans::NoTrans, 5, 4, 9);
    let copy = || operand::<f64>(Trans::NoTrans, 5, 4, 9);
    assert_eq!(compare_results(&copy(), &base, Compare::Bitwise), Ok(()));
    // One ulp is a bitwise failure, but within a rounding bound.
    let mut ulp = copy();
    ulp[(3, 2)] = f64::from_bits(ulp[(3, 2)].to_bits() + 1);
    assert!(compare_results(&ulp, &base, Compare::Bitwise).is_err());
    let (scale, factor) = (vec![1.0; 20], rounding_factor(16, f64::UNIT_ROUNDOFF));
    let eps = f64::EPSILON;
    assert!(factor > 16.0 * eps && factor < 20.0 * eps, "{factor}");
    let bound = Compare::Bound(&scale, factor);
    assert_eq!(compare_results(&ulp, &base, bound), Ok(()));
    // A large error, a NaN, and a shape mismatch fail.
    let mut wrong = copy();
    wrong[(0, 1)] += 1e-6;
    assert!(compare_results(&wrong, &base, bound).is_err());
    wrong[(0, 1)] = f64::NAN;
    assert!(compare_results(&wrong, &base, bound).is_err());
    let other = operand::<f64>(Trans::NoTrans, 4, 5, 9);
    assert!(compare_results(&other, &base, Compare::Bitwise).is_err());
    assert!(rounding_factor(1 << 40, f32::UNIT_ROUNDOFF).is_infinite());
    // |op(A)|·|op(B)| for every transpose pair (f32 operands), and the case
    // table parser.
    let (no, tr) = (Trans::NoTrans, Trans::Trans);
    for (ta, tb) in [(no, tr), (tr, no), (no, no), (tr, tr)] {
        let (a, b) = (operand::<f32>(ta, 3, 7, 1), operand::<f32>(tb, 7, 2, 2));
        for (idx, got) in abs_product((ta, tb), [&a, &b], [3, 7, 2])
            .iter()
            .enumerate()
        {
            let (i, j) = (idx % 3, idx / 3);
            let term = |p| op_abs(&a, ta, i, p) * op_abs(&b, tb, p, j);
            let want: f64 = (0..7).map(term).sum();
            assert!((got - want).abs() <= 1e-12 * want, "{ta:?}{tb:?} ({i},{j})");
        }
    }
    let parsed = cases(&BOTH, "NN 2x3x4tq TN 5x6x7d NT 8x9x10s");
    let got: Vec<String> = parsed
        .iter()
        .map(|c| format!("{}{}{}", c.label(), c.quick, c.named_target))
        .collect();
    let want = "NN_2x3x4_f64truetrue TN_5x6x7_f64truefalse NT_8x9x10_f64falsefalse \
                NN_2x3x4_f32truetrue TN_5x6x7_f32falsefalse NT_8x9x10_f32truefalse";
    assert_eq!(got, want.split_whitespace().collect::<Vec<_>>());
}

/// Checks that `spec`'s plans for `case` build and its shapes are consistent
/// (no operands are allocated).
fn dry_case<T: AbReal>(spec: &GroupSpec, case: &Case) {
    let ctx = format!("{} {} {}", spec.name, spec.run, case.label());
    let (m, k, n, ta, tb) = (case.m, case.k, case.n, case.trans_a, case.trans_b);
    assert!(m > 0 && k > 0 && n > 0, "{ctx}: empty dimension");
    assert_eq!(case.roles.len(), spec.variants.len() - 1, "{ctx}: roles");
    let (sa, sb) = (stored_shape(ta, m, k), stored_shape(tb, k, n));
    let ops = (stored_shape(ta, sa.0, sa.1), stored_shape(tb, sb.0, sb.1));
    assert_eq!(ops, ((m, k), (k, n)), "{ctx}: operand shapes");
    for v in &spec.variants {
        // The forced 16x6 kernel is AVX2 code: built only on that tier.
        #[cfg(target_arch = "x86_64")]
        if v.kernel == KernelPick::F32x16x6 && SimdTier::current() != SimdTier::Avx2Fma {
            assert_eq!(case.dtype, Dtype::F32, "{ctx}: 16x6 is f32");
            continue;
        }
        let (shape, name) = (v.plan::<T>(m, n, k).kernel.shape, v.name);
        assert!(shape.mr > 0 && shape.nr > 0, "{ctx} {name}: kernel shape");
    }
}

/// NC of `variant`'s plan for `case` in the driver the B-cap group measures.
fn capped_nc_of<T: AbReal>(variant: &Variant, case: &Case, parallel: bool) -> usize {
    let plan = variant.plan::<T>(case.m, case.n, case.k);
    let blocks = call_blocking(BlockingSource::Default, &plan, case.m, case.k, case.n);
    #[cfg(feature = "parallel")]
    if parallel {
        return blocks.parallel().nc;
    }
    let _ = parallel;
    blocks.sequential().nc
}

/// The group-level checks of the dry run.
fn dry_spec(spec: &GroupSpec) {
    let ctx = format!("{} {}", spec.name, spec.run);
    let base = &spec.variants[0];
    // A is LEGACY (the cache group's cap-on run: LEGACY with the cap).
    let reset = GemmPolicy {
        b_cap: LEGACY_POLICY.b_cap,
        ..base.policy
    };
    assert_eq!(reset, LEGACY_POLICY, "{ctx}: A is not LEGACY");
    let plain = base.kernel == KernelPick::Plan && base.cache_limit.is_none();
    assert!(
        plain && spec.variants.len() >= 2,
        "{ctx}: A, or no candidate"
    );
    let mut labels: Vec<String> = spec.cases.iter().map(Case::label).collect();
    let count = labels.len();
    labels.sort();
    labels.dedup();
    assert_eq!(labels.len(), count, "{ctx}: duplicate cases");
    for (ci, v) in spec.variants.iter().enumerate().skip(1) {
        let name = v.name;
        let same = v.policy == base.policy && v.kernel == base.kernel;
        assert!(
            !same || v.cache_limit != base.cache_limit,
            "{ctx}: {name} is A"
        );
        assert!(!name.contains(['\t', ';', ',', ':']), "{ctx}: name {name}");
        let roles = |quick| spec.selected(quick).map(move |c| c.roles[ci - 1]);
        for (quick, role) in [false, true]
            .map(|q| (q, Role::Target))
            .into_iter()
            .chain([false, true].map(|q| (q, Role::Elsewhere)))
        {
            let has = roles(quick).any(|r| r == role);
            assert!(has, "{ctx}: {name} lacks {role:?} cases (quick {quick})");
        }
        // Every B-cap candidate caps NC on at least one target case.
        if let Engage::BCap(parallel) = spec.engage {
            let nc = |var: &Variant, c: &Case| match c.dtype {
                Dtype::F64 => capped_nc_of::<f64>(var, c, parallel),
                Dtype::F32 => capped_nc_of::<f32>(var, c, parallel),
            };
            let target = |c: &&Case| c.roles[ci - 1] == Role::Target;
            let mut targets = spec.cases.iter().filter(target);
            assert!(
                targets.any(|c| nc(v, c) < nc(base, c)),
                "{ctx}: {name} never caps"
            );
        }
    }
    for case in &spec.cases {
        match case.dtype {
            Dtype::F64 => dry_case::<f64>(spec, case),
            Dtype::F32 => dry_case::<f32>(spec, case),
        }
        // Route groups: the roles are the router's (elsewhere cases keep A's
        // path, so they are compared bit for bit).
        let label = case.label();
        if let Engage::Route(_) = spec.engage {
            let routed: Vec<Role> = spec.variants[1..]
                .iter()
                .map(|v| route_role(case, v))
                .collect();
            assert_eq!(case.roles, routed, "{ctx}: {label}");
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
    expected.extend(["ab_seq_thin", "ab_seq_b_cap", "ab_seq_pack_cache"]);
    if cfg!(feature = "parallel") {
        expected.extend(["ab_par_pack_scope", "ab_par_thin", "ab_par_b_cap"].repeat(3));
    }
    assert_eq!(names, expected);
    specs.iter().for_each(dry_spec);
    // The thin, kernel and scope targets are what the design says.
    for case in &spec_seq_thin().cases {
        let (thin, label) = (case.m == 1 || case.n == 1, case.label());
        assert_eq!(case.roles, [role_if(thin)], "{label}");
    }
    #[cfg(target_arch = "x86_64")]
    for case in &spec_seq_f32_kernel().cases {
        let (canonical, (m, n, k)) = (MicroKernelShape { mr: 8, nr: 8 }, (case.m, case.n, case.k));
        let (tier, adaptive) = (SimdTier::Avx2Fma, F32Avx2Policy::Adaptive);
        let picks = super::gemm_kernel::f32_avx2_picks_16x6(tier, adaptive, canonical, m, n, k);
        assert_eq!(case.roles, [role_if(picks); 2], "{}", case.label());
    }
    #[cfg(feature = "parallel")]
    for spec in specs.iter().filter(|s| s.engage == Engage::Scope) {
        for case in &spec.cases {
            let gate = role_if(
                case.work() >= 128 * 128 * 128
                    && case.work() / spec.threads as u128 >= 80 * 80 * 80,
            );
            assert_eq!(case.roles, [gate; 2], "{} {}", spec.run, case.label());
        }
    }
}

#[test]
fn test_gemm_ab_cache_limit_guards_restore() {
    use super::gemm_packing::pack_cache_limit;
    assert_eq!(pack_cache_limit(), PACK_CACHE_MAX_BYTES);
    for limit in [Some(16 * MIB), Some(32 * MIB), None] {
        hold_cache_limit(limit);
        assert_eq!(pack_cache_limit(), limit.unwrap_or(PACK_CACHE_MAX_BYTES));
    }
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
        env.hold(None);
        let default = PACK_CACHE_MAX_BYTES;
        assert_eq!((limits(), pack_cache_limit()), (vec![default; 3], default));
    }
}
