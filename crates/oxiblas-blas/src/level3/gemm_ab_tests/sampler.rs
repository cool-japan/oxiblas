//! The engine of one case, the line-up of its variants, calibration and the
//! timed rounds.
//!
//! The generic code here chooses which variant runs when; an [`Engine`]
//! only executes what it is handed (the real one builds the variant's plan
//! and calls the GEMM, a fake one in the tests has known costs).

use super::engage::{Facts, Fingerprint};
use super::groups::{GroupSpec, Variant};
use std::time::Duration;

/// The name of the control A′ in every line-up.
pub(super) const CONTROL_NAME: &str = "control";

/// Upper bound of the calibrated repeat count.
pub(super) const MAX_REPS: usize = 1 << 24;

/// Which place a variant has in a case's line-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Slot {
    /// A, the reference of the ratios.
    Base,
    /// Candidate number `index` of the group (0-based).
    Candidate(usize),
    /// A′: A's variant, built separately; its ratio is the spread.
    Control,
}

/// How the correctness pass checks a variant's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CheckKind {
    /// Bit for bit against A's result (A's own check is `Reference`).
    Bitwise,
    /// Against the independent reference, with the rounding tolerance.
    Reference,
}

impl CheckKind {
    /// The word of the `check` column.
    pub(super) const fn word(self) -> &'static str {
        match self {
            Self::Bitwise => "bitwise",
            Self::Reference => "reference",
        }
    }
}

/// One variant of a case's line-up.
#[derive(Debug, Clone, Copy)]
pub(super) struct Entry {
    pub(super) variant: Variant,
    pub(super) slot: Slot,
    pub(super) check: CheckKind,
}

/// The line-up of a case of `spec`: A, the candidates in order, A′.
///
/// A is checked against the reference in every group. In route groups
/// (whose candidates change the rounding) the candidates are too; in every
/// other group they promise A's bits. A′ is compared with A bit for bit.
pub(super) fn lineup(spec: &GroupSpec, route_group: bool) -> Vec<Entry> {
    let own = if route_group {
        CheckKind::Reference
    } else {
        CheckKind::Bitwise
    };
    let base = spec.variants.first().copied();
    let mut out: Vec<Entry> = base
        .iter()
        .map(|&variant| Entry {
            variant,
            slot: Slot::Base,
            check: CheckKind::Reference,
        })
        .collect();
    out.extend(
        spec.candidates()
            .iter()
            .enumerate()
            .map(|(i, &variant)| Entry {
                variant,
                slot: Slot::Candidate(i),
                check: own,
            }),
    );
    out.extend(base.map(|variant| Entry {
        variant: Variant {
            name: CONTROL_NAME,
            ..variant
        },
        slot: Slot::Control,
        check: CheckKind::Bitwise,
    }));
    out
}

/// Executes the calls of one case.
pub(super) trait Engine {
    /// Empties the packing caches of the calling thread and of every pool
    /// worker and checks that nothing is parked on any of them.
    fn start_clean(&mut self) -> Result<(), String>;
    /// The correctness and engagement pass of `entry` (with its limit in
    /// force): computes, checks and reports the facts. Called first for A.
    fn check(&mut self, entry: &Entry) -> Result<Facts, String>;
    /// Puts the packing-cache limit `limit` in force on the calling thread
    /// and on every pool worker.
    fn apply_limit(&mut self, limit: Option<usize>);
    /// Checks that the limit `limit` is in force on the calling thread and
    /// on every pool worker.
    fn verify_limit(&mut self, limit: Option<usize>) -> Result<(), String>;
    /// One untimed call of `variant` (its plan built for the call).
    fn call(&mut self, variant: &Variant);
    /// `reps` timed calls of `variant` (each building its plan); how long
    /// they took.
    fn time(&mut self, variant: &Variant, reps: usize) -> Duration;
    /// The GEMM calls made since the previous `take_calls` (correctness,
    /// probe, warm-up and timed calls), per variant name and fingerprint of
    /// what each call executed; the record starts empty again.
    fn take_calls(&mut self) -> Vec<CallCount>;
}

/// Calls of one variant that executed one [`Fingerprint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CallCount {
    pub(super) variant: &'static str,
    pub(super) ran: Fingerprint,
    pub(super) calls: usize,
}

/// Adds `calls` calls of `variant` that executed `ran` to `record`.
pub(super) fn count_calls(
    record: &mut Vec<CallCount>,
    variant: &'static str,
    ran: Fingerprint,
    calls: usize,
) {
    let same = |c: &&mut CallCount| c.variant == variant && c.ran == ran;
    match record.iter_mut().find(same) {
        Some(count) => count.calls += calls,
        None => record.push(CallCount {
            variant,
            ran,
            calls,
        }),
    }
}

/// The order of the variants in round `round` of a line-up of `count`:
/// forward (A, the candidates, A′) in even rounds, reverse (A′, the
/// candidates backwards, A) in odd rounds.
pub(super) fn round_order(round: usize, count: usize) -> Vec<usize> {
    if round.is_multiple_of(2) {
        (0..count).collect()
    } else {
        (0..count).rev().collect()
    }
}

/// One timed sample of `entry`: its limit put in force when the variants
/// differ in it (`switch`), one untimed call of the same variant, the limit
/// checked, then `reps` timed calls.
pub(super) fn sample(
    engine: &mut dyn Engine,
    entry: &Entry,
    reps: usize,
    switch: bool,
) -> Result<Duration, String> {
    if switch {
        engine.apply_limit(entry.variant.limit);
    }
    engine.call(&entry.variant);
    engine.verify_limit(entry.variant.limit)?;
    Ok(engine.time(&entry.variant, reps))
}

/// The smallest repeat count found (by extrapolation, at most [`MAX_REPS`])
/// for which one sample of the fastest variant lasts at least `min`; every
/// sample of the calibration follows a warm-up call (see [`sample`]).
/// `requested[v]` counts the calls asked of entry `v`.
pub(super) fn calibrate(
    engine: &mut dyn Engine,
    entries: &[Entry],
    min: Duration,
    switch: bool,
    requested: &mut [usize],
) -> Result<usize, String> {
    let mut reps = 1_usize;
    loop {
        let mut fastest = Duration::MAX;
        for (entry, asked) in entries.iter().zip(requested.iter_mut()) {
            fastest = fastest.min(sample(engine, entry, reps, switch)?);
            *asked += 1 + reps;
        }
        if fastest >= min || reps >= MAX_REPS {
            return Ok(reps);
        }
        let scale = min.as_secs_f64() / fastest.as_secs_f64() * 1.1;
        let estimate = (reps as f64 * scale).ceil();
        let estimate = if estimate.is_finite() && estimate < MAX_REPS as f64 {
            estimate as usize
        } else {
            MAX_REPS
        };
        reps = estimate
            .clamp(reps + 1, reps.saturating_mul(16))
            .min(MAX_REPS);
    }
}

/// `rounds` rounds (see [`round_order`]) of one sample per variant; per
/// variant (in line-up order), the time per call (ns) of each round.
/// `requested[v]` counts the calls asked of entry `v`.
pub(super) fn sample_rounds(
    engine: &mut dyn Engine,
    entries: &[Entry],
    (rounds, reps): (usize, usize),
    switch: bool,
    requested: &mut [usize],
) -> Result<Vec<Vec<f64>>, String> {
    let mut times = vec![vec![f64::NAN; rounds]; entries.len()];
    for round in 0..rounds {
        for v in round_order(round, entries.len()) {
            let elapsed = sample(engine, &entries[v], reps, switch)?;
            requested[v] += 1 + reps;
            times[v][round] = elapsed.as_secs_f64() * 1e9 / reps.max(1) as f64;
        }
    }
    Ok(times)
}
