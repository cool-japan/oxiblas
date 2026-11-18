//! In-crate A/B timing harness for the GEMM policy fields.
//!
//! # Purpose
//!
//! Every optional GEMM path of 0.2.4 sits behind one field of `GemmPolicy`;
//! `LEGACY_POLICY` reproduces 0.2.3 (the bits of every result that is not a
//! NaN and the positions of the NaNs; which NaN a result carries where an
//! operation meets two NaNs may differ, see `LEGACY_POLICY`). A field of the
//! default policy is switched on only where an A/B measurement passes the
//! keep/revert rule below, and this module is that measurement: on aarch64
//! (NEON) the default has the scoped thin route (`thin_scoped` of
//! `ab_seq_thin`) and the small transposed copy where `op(A)` is not read in
//! place (`copy_a_only>=12^3` of `ab_seq_small_trans_copy`), measured on an
//! Apple M3; every other field and target keeps 0.2.3 (see `gemm_policy`).
//! Whatever the build's default is, A of every group is `LEGACY_POLICY`
//! itself (the variants are built from it, never from the default), so a
//! group always measures against 0.2.3's routes. It is a test module because
//! its hooks exist only in test builds (the per-thread packing-cache limit
//! and drain, the driver counters, the second reading of the issue #23
//! B-block caps).
//!
//! # Groups
//!
//! A group (`#[test] #[ignore = "timing harness"] fn ab_<group>`) compares A
//! (LEGACY) with candidates that change one field, or one coherent set of
//! fields. A *group run* is one group at one thread count (parallel groups)
//! or one cap setting (`ab_seq_pack_cache`: runs `cap_off`, `cap_on`).
//!
//! | Group | Candidates | Engagement | Threshold | Only on |
//! |---|---|---|---|---|
//! | `ab_seq_f32_kernel` | `adaptive`, `forced16x6` | kernel shape | lower bound on `m·n·k` | x86-64, AVX2 + FMA |
//! | `ab_seq_small_simd` | `simd<=32^3`, `simd<=80^3` | router path | upper bound on `m·n·k` | x86-64, AVX2 + FMA |
//! | `ab_seq_small_trans_copy` | `copy<=32^3`, `copy_a_only>=12^3` (x86 also `copy<=80^3`) | router path | interval of `m·n·k` | x86-64 (AVX2 + FMA), aarch64 |
//! | `ab_seq_thin` | `thin`, `thin_scoped` | router path | none | |
//! | `ab_seq_thin_grid` | `thin`, `thin_scoped` (informational) | router path | none | |
//! | `ab_seq_b_cap` | `cap_seq`, `cap_seq_alt` | NC of the driver | none | |
//! | `ab_seq_pack_cache` | `limit16MiB`, `limit32MiB` | allocations, parked bytes | none | |
//! | `ab_par_pack_scope` | `per_tile`, `per_tile_16MiB` | scope and 2D runs | none | `parallel` |
//! | `ab_par_thin` | `thin`, `thin_scoped` | router path | none | `parallel` |
//! | `ab_par_b_cap` | `cap_all`, `cap_all_alt` | NC of the driver | none | `parallel` |
//!
//! What each candidate changes (every one is `LEGACY_POLICY` with these
//! fields, A is `LEGACY_POLICY`):
//! - `adaptive`: `f32_avx2 = Adaptive`; `forced16x6`: no field, the f32
//!   16×6 kernel forced on every case (informational);
//! - `simd<=32^3`, `simd<=80^3`: `small_simd_x86` with `small_simd_max_work`
//!   32³ or 80³;
//! - `copy<=32^3`: `small_trans_copy` from 8³ to 32³ for every transposed or
//!   strided product (x86-64 also `small_simd_x86`); `copy<=80^3`: the same
//!   up to 80³; `copy_a_only>=12^3`: from 12³ to 32³ with
//!   `small_trans_copy_a_only` (only products with A transposed and both
//!   operands stored column-major), the field adopted on aarch64;
//! - `thin`: `thin = All` (every thin product); `thin_scoped`:
//!   `thin = Scoped` (the measured scope, `RouteInputs::thin_in_scope`),
//!   adopted on aarch64;
//! - `cap_seq`, `cap_seq_alt`, `cap_all`, `cap_all_alt`: `b_cap` with the
//!   two readings of the issue #23 caps;
//! - `limit16MiB`, `limit32MiB`, `per_tile_16MiB`: the packing-cache limit;
//!   `per_tile`: `per_tile_min_threads = 1`.
//!
//! `ab_seq_thin` is the decision group of the thin route: the original
//! shapes (the squares, the dots, the non-thin neighbours, the `beta = 1`
//! shapes) and the boundary set of the scope (per small layout the points
//! of 512, 1024 and 2048 multiply-adds, the points just outside a layout's
//! limit, and points of the four layouts outside the scope), 236 cases.
//! `ab_seq_thin_grid` is its information group: the original shapes and the
//! dense grid of the four small layouts with the coarse grid of the other
//! four, 1530 cases (20 to 30 minutes per pass on an Apple M3). It is the
//! measurement a scope is derived from; its verdicts never decide, because
//! the keep rule asks every one of its several hundred non-target cases
//! (which run the same instructions under both policies) to stay above
//! 0.97, which a single disturbed case breaks.
//!
//! The parallel groups run once per thread count, each in an installed rayon
//! pool of exactly that size (`Par::RayonWith(n)` only sizes the partition;
//! the work runs on the ambient pool). A cache limit is set on the calling
//! thread and on every worker (`ThreadPool::broadcast`).
//!
//! # Method: one case
//!
//! Every case of every group has its operands stored column-major (a
//! `Shape` has transposes, sizes and beta, no storage layout: each operand
//! is a `Mat`, read through its transpose where the case says so). What a
//! group measures is therefore that storage only, and the adopted routes
//! are limited to it (`RouteInputs::thin_in_scope`,
//! `RouteInputs::a_transposed_col_major`); the routes for operands stored
//! otherwise remain candidates until cases with such storage exist.
//!
//! The harness runs in a test build, where every timed call also executes
//! the test-only call probes of the plan (the count of tier reads and the
//! record of the route taken: thread-local writes of a few instructions).
//! A and every candidate pay them alike.
//!
//! 1. **Clean start.** The packing caches of the calling thread and of every
//!    pool worker are emptied (`gemm_packing::drain_pack_cache`) and checked
//!    to hold 0 bytes, before every variant's correctness call and before
//!    every timed attempt: buffers parked by an earlier case (also of the
//!    other element type) cannot change what a later one measures.
//! 2. **Correctness and engagement before timing**, per variant with its
//!    cache limit in force on the caller and every worker. A's result is
//!    checked against an independent high-precision reference in every group
//!    (see `check`). Groups whose candidates promise A's bits (kernel shape,
//!    pack scope, B caps, cache limit) and every control are compared with
//!    A bit for bit. Route groups (thin, small SIMD, small transposed copy)
//!    compare every candidate with the reference too, and a candidate that
//!    takes A's path also bit for bit with A. The call also records what ran
//!    (see *Roles* and *What a timed call ran*), checks the pool size and
//!    that the call took the parallel path exactly where the gates say. A
//!    failure panics with the group, run, case and variant: nothing is
//!    timed.
//! 3. **Timing.** The timed unit is what a user's call costs: the plan is
//!    built inside the timed loop as the public entry points build it
//!    (`GemmPlan::for_problem`, plus the forced kernel where a variant forces
//!    one), then `gemm_planned` runs. Before every timed sample the variant's
//!    cache limit is put in force (when the variants differ in it), one
//!    untimed call of the same variant is made and the limits are checked;
//!    then `reps` calls are timed. So every timed call follows a call of the
//!    same variant. `reps` is calibrated (after warm-up calls too) so that a
//!    sample of the fastest variant lasts at least the minimum sample time.
//! 4. **Order.** The line-up is A, the candidates in order, A′ (the control:
//!    A's variant, built separately). Round `r` (from 0) runs it forward
//!    when `r` is even and in reverse (A′ first, A last) when `r` is odd; the
//!    number of rounds is even.
//! 5. **Statistics.** `t_v(r)` is the time per call of variant `v` in round
//!    `r`. The primary figure of a variant is the median over the rounds of
//!    the paired ratio `q_v(r) = t_A(r) / t_v(r)` (above 1: faster than A),
//!    with its quartiles; the secondary one is `median(t_A) / median(t_v)`.
//!    The *spread* is the control's paired median minus 1; a case is valid
//!    when `|ln(median q_A′)| <= ln(1.015)` and both quartiles of `q_A′` are
//!    within twice that on the logarithmic scale,
//!    `|ln(p25 q_A′)|, |ln(p75 q_A′)| <= ln(1.015²)`
//!    (`stats::QUARTILE_SPREAD_LIMIT`): load that arrives in bursts can
//!    disturb a quarter or more of a case's rounds and still leave the
//!    control's median within 1.5 %.
//! 6. **Re-measurement.** A case whose control is outside either limit is
//!    measured again as a whole (clean start, new calibration, all rounds),
//!    at most `OXIBLAS_AB_RETRIES` more times, and counts as re-measured.
//!    Every attempt is printed; the last attempt is the case's result, never
//!    the best one, and a case whose last attempt is still outside a limit
//!    is invalid. The decision to measure again looks only at A and A′.
//!
//! # What a timed call ran
//!
//! One function (`engine::Problem::call`) builds the plan of every call,
//! correctness, probe, warm-up and timed alike, and hands it with the
//! problem's `Par` to `gemm_planned`. Each call records a *fingerprint* of
//! what it executed (`engage::Fingerprint`): the path (route and SIMD small
//! kernel) the plan takes for the operands, the kernel shape, KC and the
//! fitted NC of both drivers (B cap and limit included), the pack scope at
//! the `Par`'s thread count, the `Par` and the pool size the call saw.
//! Inside the timed loop a call only compares its plan and `Par` with the
//! previous ones (equal plans and `Par` on one thread give equal
//! fingerprints); the fingerprint of each distinct plan is computed after
//! the clock stopped. After the correctness pass and after every attempt
//! the runner requires every call of a variant to show the fingerprint of
//! that variant's correctness call, and every variant to have made exactly
//! the calls asked of it; a difference fails the run with both
//! fingerprints, before any figure is printed.
//!
//! # Roles and engagement
//!
//! A candidate *engages* on a case when it changes what runs there, decided
//! on the machine from its correctness call: route groups, the router's
//! path (route, SIMD small kernel or not) at the host tier differs from A's;
//! `ab_seq_f32_kernel`, the plan's kernel shape; `ab_par_pack_scope`, the
//! plan's scope at the thread count and the 2D driver's run counters;
//! B-cap groups, NC of the driver that ran (the 2D one when it ran); the
//! cache group, the buffers allocated during two calls after a warm-up or
//! the bytes left parked (caller and workers) differ from A's. Roles: A is
//! `base`, A′ `control`; a candidate's case is a `target` where it engages,
//! `elsewhere` where it does not. The groups name their *intended* targets
//! (the router at the design tier, the padding model, the parallel gate,
//! the table's flags); an intended target that does not engage is printed
//! as `target_noop` and counts as elsewhere (a candidate that changes
//! nothing on a "target" would face the 1.00 floor on noise alone).
//!
//! # Keep/revert rule (per candidate and group run)
//!
//! In this order:
//! - `SMOKE` when the run is not decision grade: fewer than 21 rounds
//!   requested, a minimum sample time below 2 ms, the quick subset, or a
//!   debug build (the figures are still printed);
//! - `INVALID` when the last attempt of any case is invalid;
//! - `NO_TARGETS` when the candidate engages on no case;
//! - `ADOPT` when the geometric mean of the target ratios is at least 1.05,
//!   every target ratio at least 1.00 and every elsewhere ratio (no-op
//!   targets included) at least 0.97;
//! - `KEEP_LEGACY` otherwise.
//!
//! `partial` is the largest `m·n·k` range in which the rule passes, among the
//! ranges the group's policy threshold can express: upper bound (ranges from
//! the smallest engaged size), lower bound (ranges up to the largest engaged
//! size) or interval; cases of equal `m·n·k` are never split. `n/a` where
//! the field has no work threshold, `-` when not evaluated (`INVALID`,
//! `NO_TARGETS`), `none` when no range passes, `informational` for a
//! candidate that is no policy field (`forced16x6`, a forced kernel) and for
//! the candidates of an information group (`ab_seq_thin_grid`): their
//! decision only informs, no range is computed for them. That the full test
//! suite passes under the candidate policy is checked elsewhere.
//!
//! # Running
//!
//! `cargo test -p oxiblas-blas --release --features parallel --lib --
//! --ignored --test-threads=1 --nocapture <filters>`, where the filters are
//! substrings of the group names (`ab_par_`, `ab_` for all). A filter
//! selects every group whose name contains it: `ab_seq_thin` selects
//! `ab_seq_thin_grid` too, so one group alone is run with `--exact` and its
//! full test path (`--exact level3::gemm_ab_tests::ab_seq_thin`).
//! On x86-64 set `OXIBLAS_REQUIRE_AVX2=1` so that the AVX2-gated groups
//! (`ab_seq_f32_kernel`, `ab_seq_small_simd`, `ab_seq_small_trans_copy`)
//! cannot pass by skipping.
//!
//! Settings (a value that is empty or only white space counts as unset; a
//! malformed one, also one that is not valid Unicode, fails the run):
//! - `OXIBLAS_AB_ROUNDS`: rounds per attempt (default 22; an odd value is
//!   rounded up to the next even one; at least 4);
//! - `OXIBLAS_AB_MIN_SAMPLE_MS`: least sample time in ms (default 2);
//! - `OXIBLAS_AB_THREADS`: thread counts of the parallel groups,
//!   comma-separated (default: the available cores);
//! - `OXIBLAS_AB_QUICK`: non-empty for a small subset of each group (smoke
//!   runs);
//! - `OXIBLAS_AB_RETRIES`: re-measurements of a case with a bad control
//!   (default 2, at most 10). A run is invalid when one case stays invalid:
//!   on an Apple M3 with the default of 2, passes of the 1530 cases of
//!   `ab_seq_thin_grid` kept 1 to 27 cases with a control outside 1.5 %
//!   (background daemons of the operating system), and the deciding runs
//!   used 8, with no invalid case. Use 8 for the large groups;
//! - `OXIBLAS_AB_TAG`: free text without tabs, recorded on every `SET` line;
//! - `OXIBLAS_AB_OUT`: a file every line is also appended to.
//!
//! # Output
//!
//! Tab-separated lines on standard output, each also appended to the
//! `OXIBLAS_AB_OUT` file, each written with one write call. The first field
//! is the kind; every line of a group run carries its `run_id`
//! (`<pid>-<unix ms>-<sequence>`) as the first field after the kind (after
//! `SET`/`END` on those lines). Per group run, in order:
//! - `AB#\tAB\t…`, `AB#\tABV\t…`, `AB#\tABR\t…`: the column names
//!   (`output::AB_COLUMNS`, `ABV_COLUMNS`, `ABR_COLUMNS`);
//! - `AB#\tSET\t<run_id>\tkey=value…`: `group`, `run`, `tier` (the Debug
//!   name of `SimdTier`), `arch`, `threads`, `rounds`, `rounds_requested`,
//!   `min_sample_ms`, `quick`, `debug_build`, `retries`, `tag`, `grade`
//!   (`decision` or `smoke`), `features` (of `parallel`, `force-scalar`,
//!   `max-simd-128`, `max-simd-256`, `sme2`; `none`), `cpu`, `pid`; printed
//!   after the AVX2 gate passed;
//! - per case and attempt, per variant, one `AB` line and one `ABR` line;
//! - `ABV`: the verdict of the group run;
//! - `AB#\tEND\t<run_id>\tgroup=…\trun=…\tcases=…\tinvalid_cases=…` once the
//!   run completed (a `SET` without `END` is a run that panicked).
//!
//! `AB` columns: `run_id group run threads case dtype trans m k n beta
//! attempt valid variant role check reps rounds median_ns p25_ns p75_ns
//! gflops ratio ratio_p25 ratio_p75 ratio_of_medians spread check_worst
//! engaged detail`. Times are per call in ns over the rounds; `gflops` =
//! `2·m·n·k / median_ns`; `ratio` and its quartiles are the paired ratios;
//! `spread` is the attempt's control spread (from the control's median);
//! `valid` whether the attempt's control is within both limits (median and
//! quartiles, step 5); `role` one of `base`, `control`, `target`,
//! `elsewhere`, `target_noop`; `check` `reference` or `bitwise`;
//! `check_worst` the largest `|error| / tolerance` of the reference check
//! (`-` for bitwise only); `engaged` `yes`/`no` for candidates; `detail` the
//! route, kernel, KC, NC (sequential / parallel driver), planned scope, the
//! 2D driver's runs, the thin splits (gated / split), the limit, the cache
//! probe, the pool, and for candidates `seen=<A>-><candidate>`, what the
//! engagement test compared.
//!
//! `ABR` columns: `run_id group run case attempt variant reps ns_per_call`,
//! the last a comma-separated list of the time per call of every round in
//! round order (round `r` ran forward when `r` is even, in reverse when odd).
//!
//! `ABV` columns: `run_id group run threads tier grade cases remeasured
//! invalid max_abs_spread verdicts`, `verdicts` one `;`-separated entry per
//! candidate: `name:DECISION,geo=…,min_t=…,min_e=…,nt=…,ne=…,noop=…,partial=…`
//! (`ne` counts the no-op targets too, `noop` only them).
//!
//! # Protocol for a real measurement
//!
//! A quiet machine (no builds, no emulators, no other jobs), plugged in,
//! Low Power Mode off and `caffeinate -dims` on macOS; on Linux the
//! performance governor and the sequential groups pinned (`taskset -c 2`).
//! Run each group twice, in a different order (each group in its own
//! process), and trust a verdict only when both runs are valid and agree.
//! Timings taken under Rosetta 2, in a debug build or on a busy machine are
//! meaningless: such runs only show that the harness works (their verdicts
//! say `SMOKE` where the settings are not decision grade, but a busy machine
//! is not detected).

mod check;
mod engage;
mod engine;
mod groups;
mod output;
mod runner;
mod sampler;
mod settings;
mod specs;
mod stats;

mod tests;
mod tests_chain;
mod tests_check;
mod tests_dry;
mod tests_figures;
mod tests_guard;

use super::gemm_plan::{BCapScope, SimdTier};
use check::AbReal;
use engine::{Env, GemmEngine};
use groups::{Case, GroupSpec};
use output::OutputSink;
use runner::{BuildInfo, run_group};
use sampler::Engine;
use settings::Settings;
use specs::Dtype;

#[cfg(target_arch = "x86_64")]
use super::gemm_plan::avx2_fma_tier_or_skip;

/// The crate features that change what runs, `+`-separated (`none`).
fn features() -> String {
    let flags = [
        (cfg!(feature = "parallel"), "parallel"),
        (cfg!(feature = "force-scalar"), "force-scalar"),
        (cfg!(feature = "max-simd-128"), "max-simd-128"),
        (cfg!(feature = "max-simd-256"), "max-simd-256"),
        (cfg!(feature = "sme2"), "sme2"),
    ];
    let on: Vec<&str> = flags.iter().filter(|f| f.0).map(|f| f.1).collect();
    if on.is_empty() {
        "none".to_string()
    } else {
        on.join("+")
    }
}

/// The CPU's name (Linux `/proc/cpuinfo`, macOS `sysctl`), `unknown`
/// otherwise; spaces become `_`.
fn cpu_name() -> String {
    let from_proc = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            let line = text.lines().find(|l| l.starts_with("model name"))?;
            Some(line.split_once(':')?.1.trim().to_string())
        });
    let from_sysctl = || {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        let name = String::from_utf8(out.stdout).ok()?;
        Some(name.trim().to_string()).filter(|n| !n.is_empty())
    };
    let name = from_proc
        .or_else(from_sysctl)
        .unwrap_or_else(|| "unknown".to_string());
    name.split_whitespace().collect::<Vec<_>>().join("_")
}

/// The build and machine of this process.
fn build_info() -> BuildInfo {
    BuildInfo {
        tier: format!("{:?}", SimdTier::current()),
        arch: std::env::consts::ARCH.to_string(),
        debug_build: cfg!(debug_assertions),
        features: features(),
        cpu: cpu_name(),
    }
}

/// The real engine of `case` (by element type).
fn engine_for<'e, 'p>(spec: &'e GroupSpec, case: &Case, env: &'e Env<'p>) -> Box<dyn Engine + 'e> {
    fn boxed<'e, 'p, T: AbReal>(
        spec: &'e GroupSpec,
        case: &Case,
        env: &'e Env<'p>,
    ) -> Box<dyn Engine + 'e> {
        Box::new(GemmEngine::<T>::new(spec, case, env))
    }
    match case.dtype {
        Dtype::F64 => boxed::<f64>(spec, case, env),
        Dtype::F32 => boxed::<f32>(spec, case, env),
    }
}

/// Runs a group run in `env` with `settings`, printing to standard output
/// and the output file.
fn run_in(spec: &GroupSpec, settings: &Settings, env: &Env<'_>) {
    let mut sink = OutputSink {
        out: settings.out.clone(),
    };
    sink.begin();
    let mut make = |case: &Case| engine_for(spec, case, env);
    run_group(spec, settings, &build_info(), &mut sink, &mut make);
}

/// Runs a sequential group run on the test thread.
fn run_seq(spec: &GroupSpec) {
    let settings = Settings::from_env();
    let env = Env::default();
    run_in(spec, &settings, &env);
    env.hold(None);
}

/// Runs a parallel group once per thread count, each in a pool of exactly
/// that many threads.
#[cfg(feature = "parallel")]
fn run_par(build: fn(usize) -> GroupSpec) {
    let settings = Settings::from_env();
    for &threads in &settings.threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build();
        let pool = pool.unwrap_or_else(|e| panic!("rayon pool of {threads} threads: {e}"));
        let env = Env { pool: Some(&pool) };
        run_in(&build(threads), &settings, &env);
        env.hold(None);
    }
}

/// Whether an AVX2-gated group may run here (x86-64: the AVX2 + FMA tier,
/// see `avx2_fma_tier_or_skip`; elsewhere always).
fn gate_avx2(group: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        avx2_fma_tier_or_skip(group)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = group;
        true
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "timing harness"]
fn ab_seq_f32_kernel() {
    if gate_avx2("ab_seq_f32_kernel") {
        run_seq(&groups::spec_seq_f32_kernel());
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "timing harness"]
fn ab_seq_small_simd() {
    if gate_avx2("ab_seq_small_simd") {
        run_seq(&groups::spec_seq_small_simd());
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
#[ignore = "timing harness"]
fn ab_seq_small_trans_copy() {
    if gate_avx2("ab_seq_small_trans_copy") {
        run_seq(&groups::spec_seq_small_trans_copy());
    }
}

#[test]
#[ignore = "timing harness"]
fn ab_seq_thin() {
    run_seq(&groups::spec_seq_thin());
}

#[test]
#[ignore = "timing harness"]
fn ab_seq_thin_grid() {
    run_seq(&groups::spec_seq_thin_grid());
}

#[test]
#[ignore = "timing harness"]
fn ab_seq_b_cap() {
    run_seq(&groups::spec_seq_b_cap());
}

#[test]
#[ignore = "timing harness"]
fn ab_seq_pack_cache() {
    for cap in [BCapScope::Off, BCapScope::Sequential] {
        run_seq(&groups::spec_seq_pack_cache(cap));
    }
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "timing harness"]
fn ab_par_pack_scope() {
    run_par(groups::spec_par_pack_scope);
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "timing harness"]
fn ab_par_thin() {
    run_par(groups::spec_par_thin);
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "timing harness"]
fn ab_par_b_cap() {
    run_par(groups::spec_par_b_cap);
}
