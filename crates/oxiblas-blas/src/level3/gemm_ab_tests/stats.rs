//! Statistics and the keep/revert rule (pure functions).

use super::settings::Grade;

/// Largest factor between A and its control A′ (paired median) of a valid
/// case, either way: `|ln(median q_A′)| <= ln(SPREAD_LIMIT)`.
pub(super) const SPREAD_LIMIT: f64 = 1.015;
/// Largest factor between A and its control A′ at the quartiles of the
/// paired ratio of a valid case, either way.
///
/// Other load that arrives in bursts can disturb a quarter or more of the
/// rounds of a case and still leave the control's median within
/// [`SPREAD_LIMIT`]; the medians of the candidates measured in those rounds
/// can then be off by several percent (on an Apple M3 with background
/// daemons of the operating system: one case in about 700 moved by 4 to 5 %
/// per run, with the control's quartiles at 0.94 and at 0.81 to 1.36). Twice
/// the limit of the median, on the logarithmic scale.
pub(super) const QUARTILE_SPREAD_LIMIT: f64 = SPREAD_LIMIT * SPREAD_LIMIT;
/// Least geometric mean of the target ratios for adoption.
pub(super) const ADOPT_GEOMEAN: f64 = 1.05;
/// Least ratio of every target case for adoption.
pub(super) const TARGET_MIN: f64 = 1.00;
/// Least ratio of every elsewhere case (and no-op intended target) for
/// adoption.
pub(super) const ELSEWHERE_MIN: f64 = 0.97;

/// The `q`-quantile of ascending `sorted` data, linearly interpolated between
/// the two nearest ranks (NaN for no data).
pub(super) fn percentile(sorted: &[f64], q: f64) -> f64 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return f64::NAN;
    };
    let pos = q.clamp(0.0, 1.0) * last as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(last);
    let (Some(&low), Some(&high)) = (sorted.get(lo), sorted.get(hi)) else {
        return f64::NAN;
    };
    low + (high - low) * (pos - lo as f64)
}

/// Median and quartiles of a sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Quartiles {
    pub(super) median: f64,
    pub(super) p25: f64,
    pub(super) p75: f64,
}

/// Median and quartiles of `samples` (in any order).
pub(super) fn quartiles(samples: &[f64]) -> Quartiles {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Quartiles {
        median: percentile(&sorted, 0.5),
        p25: percentile(&sorted, 0.25),
        p75: percentile(&sorted, 0.75),
    }
}

/// Geometric mean of positive finite values (NaN when empty or when a value
/// is not positive and finite).
pub(super) fn geomean(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
        return f64::NAN;
    }
    (values.iter().map(|v| v.ln()).sum::<f64>() / values.len() as f64).exp()
}

/// The figures of one variant in one attempt of a case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VariantFigures {
    /// Time per call (ns) over the rounds.
    pub(super) time: Quartiles,
    /// The paired ratio `q(r) = t_A(r) / t_variant(r)` over the rounds
    /// (above 1: faster than A). The median is the primary figure.
    pub(super) ratio: Quartiles,
    /// `median(t_A) / median(t_variant)` (secondary figure).
    pub(super) ratio_of_medians: f64,
}

/// The figures of every variant from `times[v][r]`, the time per call of
/// variant `v` in round `r`; `times[0]` is A.
pub(super) fn variant_figures(times: &[Vec<f64>]) -> Vec<VariantFigures> {
    let Some(base) = times.first() else {
        return Vec::new();
    };
    let base_median = quartiles(base).median;
    times
        .iter()
        .map(|own| {
            let paired: Vec<f64> = base.iter().zip(own).map(|(a, v)| a / v).collect();
            let time = quartiles(own);
            VariantFigures {
                time,
                ratio: quartiles(&paired),
                ratio_of_medians: base_median / time.median,
            }
        })
        .collect()
}

/// The spread of a case: the control's paired median minus 1.
pub(super) fn spread_of(control_ratio: f64) -> f64 {
    control_ratio - 1.0
}

/// Whether a case whose control has the paired ratio `control` (median and
/// quartiles over the rounds) is valid: the median within
/// [`SPREAD_LIMIT`] (`|ln(median)| <= ln(1.015)`) and both quartiles within
/// [`QUARTILE_SPREAD_LIMIT`], each either way (symmetric).
pub(super) fn control_valid(control: &Quartiles) -> bool {
    let within = |ratio: f64, limit: f64| {
        ratio.is_finite() && ratio > 0.0 && ratio.ln().abs() <= limit.ln() + 1e-12
    };
    within(control.median, SPREAD_LIMIT)
        && within(control.p25, QUARTILE_SPREAD_LIMIT)
        && within(control.p75, QUARTILE_SPREAD_LIMIT)
}

/// What a variant is on one case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Role {
    /// A, the reference of the ratios.
    Base,
    /// A′, A built separately: its ratio is the spread.
    Control,
    /// A candidate that changes what runs here (it has to win).
    Target,
    /// A candidate that does not change what runs here (it must not lose).
    Elsewhere,
    /// A candidate on a case the group names as a target, where it does not
    /// change what runs on this machine: counted as elsewhere.
    TargetNoop,
}

impl Role {
    /// The word of the `role` column.
    pub(super) const fn word(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Control => "control",
            Self::Target => "target",
            Self::Elsewhere => "elsewhere",
            Self::TargetNoop => "target_noop",
        }
    }

    /// The role of a candidate on a case: a target where it engages, a no-op
    /// target where the group intends a target but it does not engage,
    /// elsewhere otherwise.
    pub(super) const fn of_candidate(intended: bool, engaged: bool) -> Self {
        match (engaged, intended) {
            (true, _) => Self::Target,
            (false, true) => Self::TargetNoop,
            (false, false) => Self::Elsewhere,
        }
    }
}

/// One case's result for one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CaseOutcome {
    /// `m·n·k`, which orders the cases for a partial adoption.
    pub(super) work: u128,
    /// Target, elsewhere or no-op target.
    pub(super) role: Role,
    /// The paired median ratio of the case's last attempt.
    pub(super) ratio: f64,
}

/// The rule's figures over a set of cases of one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RuleFigures {
    /// Geometric mean of the target ratios (NaN without targets).
    pub(super) geomean: f64,
    /// Least target ratio (infinite without targets).
    pub(super) min_target: f64,
    /// Least ratio of the elsewhere and no-op target cases (infinite
    /// without any).
    pub(super) min_elsewhere: f64,
    /// Number of targets.
    pub(super) targets: usize,
    /// Number of elsewhere cases, the no-op targets included.
    pub(super) elsewhere: usize,
    /// Number of no-op intended targets (among `elsewhere`).
    pub(super) noop: usize,
    /// There is a target and every adoption clause holds.
    pub(super) passes: bool,
}

/// The adoption clauses of the rule over `cases`.
pub(super) fn rule_figures<'a>(cases: impl Iterator<Item = &'a CaseOutcome>) -> RuleFigures {
    let (mut target, mut elsewhere, mut noop) = (Vec::new(), Vec::new(), 0);
    for case in cases {
        match case.role {
            Role::Target => target.push(case.ratio),
            Role::TargetNoop => {
                noop += 1;
                elsewhere.push(case.ratio);
            }
            Role::Elsewhere | Role::Base | Role::Control => elsewhere.push(case.ratio),
        }
    }
    let least = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let (geo, min_t, min_e) = (geomean(&target), least(&target), least(&elsewhere));
    let finite = target.iter().chain(&elsewhere).all(|r| r.is_finite());
    let clauses = geo >= ADOPT_GEOMEAN && min_t >= TARGET_MIN && min_e >= ELSEWHERE_MIN;
    RuleFigures {
        geomean: geo,
        min_target: min_t,
        min_elsewhere: min_e,
        targets: target.len(),
        elsewhere: elsewhere.len(),
        noop,
        passes: !target.is_empty() && finite && clauses,
    }
}

/// The decision for one candidate over one group run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Decision {
    /// The run is not decision grade; the figures are informative only.
    Smoke,
    /// A case's last attempt has its control outside a limit (median or
    /// quartiles, [`control_valid`]).
    Invalid,
    /// The candidate changes what runs on no case of this machine.
    NoTargets,
    /// Every clause of the rule holds.
    Adopt,
    /// Some clause fails.
    KeepLegacy,
}

impl Decision {
    /// The decision word of an `ABV` entry.
    pub(super) const fn word(self) -> &'static str {
        match self {
            Self::Smoke => "SMOKE",
            Self::Invalid => "INVALID",
            Self::NoTargets => "NO_TARGETS",
            Self::Adopt => "ADOPT",
            Self::KeepLegacy => "KEEP_LEGACY",
        }
    }
}

/// Which policy threshold of a group a partial adoption could narrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Threshold {
    /// No work threshold (thin, B caps, cache limit, pack scope).
    None,
    /// An upper bound on `m·n·k` (small SIMD kernels): ranges from the
    /// smallest engaged size up.
    UpperWork,
    /// A lower bound on `m·n·k` (the f32 kernel's least work): ranges up to
    /// the largest engaged size.
    LowerWork,
    /// An interval of `m·n·k` (small transposed copy): any range.
    WorkInterval,
    /// The candidate is no policy field (`forced16x6`): its verdict is
    /// informational and it has no range to adopt.
    Informational,
}

/// The largest `m·n·k` range in which the rule passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Partial {
    /// The group's policy field has no work threshold (`n/a`).
    NotApplicable,
    /// Not evaluated: the run is invalid or the candidate has no target
    /// (`-`).
    NotEvaluated,
    /// No range passes (`none`).
    NoRange,
    /// `[lo, hi]` with `cases` cases.
    Range { lo: u128, hi: u128, cases: usize },
    /// The candidate is no policy field, its verdict only informs
    /// (`informational`).
    Informational,
}

impl Partial {
    /// The text of the `partial` field.
    pub(super) fn text(self) -> String {
        match self {
            Self::Informational => "informational".to_string(),
            Self::NotApplicable => "n/a".to_string(),
            Self::NotEvaluated => "-".to_string(),
            Self::NoRange => "none".to_string(),
            Self::Range { lo, hi, cases } => format!("{lo}..{hi}/{cases}"),
        }
    }
}

/// The verdict of one candidate over one group run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Verdict {
    pub(super) decision: Decision,
    pub(super) figures: RuleFigures,
    pub(super) partial: Partial,
}

/// The keep/revert rule for one candidate: `cases` its outcomes, `all_valid`
/// whether the last attempt of every case of the run is valid, `grade` the
/// run's grade, `threshold` the group's threshold kind.
///
/// In order: `SMOKE` (not decision grade), `INVALID`, `NO_TARGETS`, then
/// `ADOPT` when every clause holds, else `KEEP_LEGACY`.
pub(super) fn verdict(
    cases: &[CaseOutcome],
    all_valid: bool,
    grade: Grade,
    threshold: Threshold,
) -> Verdict {
    let figures = rule_figures(cases.iter());
    let decision = if grade == Grade::Smoke {
        Decision::Smoke
    } else if !all_valid {
        Decision::Invalid
    } else if figures.targets == 0 {
        Decision::NoTargets
    } else if figures.passes {
        Decision::Adopt
    } else {
        Decision::KeepLegacy
    };
    let partial = match (threshold, decision) {
        (Threshold::Informational, _) => Partial::Informational,
        (Threshold::None, _) => Partial::NotApplicable,
        (_, Decision::Invalid | Decision::NoTargets) => Partial::NotEvaluated,
        (_, Decision::Smoke | Decision::Adopt | Decision::KeepLegacy) => {
            largest_passing_range(cases, threshold)
        }
    };
    Verdict {
        decision,
        figures,
        partial,
    }
}

/// The ranges of sizes a threshold of kind `threshold` can express over
/// `works` (ascending, distinct), given the smallest and largest engaged
/// sizes.
fn candidate_ranges(
    works: &[u128],
    threshold: Threshold,
    engaged: Option<(u128, u128)>,
) -> Vec<(u128, u128)> {
    let all = || {
        let pairs = works.iter().enumerate();
        pairs.flat_map(move |(i, &lo)| works[i..].iter().map(move |&hi| (lo, hi)))
    };
    match (threshold, engaged) {
        (Threshold::None | Threshold::Informational, _) | (_, None) => Vec::new(),
        (Threshold::WorkInterval, Some(_)) => all().collect(),
        (Threshold::UpperWork, Some((first, _))) => all().filter(|r| r.0 == first).collect(),
        (Threshold::LowerWork, Some((_, last))) => all().filter(|r| r.1 == last).collect(),
    }
}

/// The range of sizes with the most cases (then the most targets, then the
/// smallest sizes) whose cases pass the rule, among the ranges `threshold`
/// can express. Cases of equal `m·n·k` are never split.
pub(super) fn largest_passing_range(cases: &[CaseOutcome], threshold: Threshold) -> Partial {
    match threshold {
        Threshold::None => return Partial::NotApplicable,
        Threshold::Informational => return Partial::Informational,
        Threshold::UpperWork | Threshold::LowerWork | Threshold::WorkInterval => {}
    }
    let mut works: Vec<u128> = cases.iter().map(|c| c.work).collect();
    works.sort_unstable();
    works.dedup();
    let targets = cases
        .iter()
        .filter(|c| c.role == Role::Target)
        .map(|c| c.work);
    let engaged = targets.clone().min().zip(targets.max());
    let mut best: Option<(Partial, usize, usize)> = None;
    for (lo, hi) in candidate_ranges(&works, threshold, engaged) {
        let inside = || cases.iter().filter(|c| lo <= c.work && c.work <= hi);
        let (figures, count) = (rule_figures(inside()), inside().count());
        let better = best
            .as_ref()
            .is_none_or(|(_, n, t)| (count, figures.targets) > (*n, *t));
        if figures.passes && better {
            let range = Partial::Range {
                lo,
                hi,
                cases: count,
            };
            best = Some((range, count, figures.targets));
        }
    }
    best.map_or(Partial::NoRange, |(range, _, _)| range)
}
