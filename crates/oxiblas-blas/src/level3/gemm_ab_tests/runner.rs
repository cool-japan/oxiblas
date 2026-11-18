//! One case and one group run, generic over the [`Engine`]: the correctness
//! pass, the roles, the timed attempts, the lines and the verdict.

use super::engage::{Engage, Facts, engages, seen};
use super::groups::{Case, GroupSpec};
use super::output::{
    AbLine, AbrLine, AbvLine, CaseIds, EndLine, SetLine, Sink, header_lines, new_run_id,
    verdict_entry,
};
use super::sampler::{CallCount, Engine, Entry, Slot, calibrate, lineup, sample_rounds};
use super::settings::Settings;
use super::specs::Beta;
use super::stats::{
    CaseOutcome, Role, Threshold, Verdict, control_valid, spread_of, variant_figures, verdict,
};

/// What the `SET` line says about the build and the machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BuildInfo {
    /// `SimdTier` of the host (Debug name).
    pub(super) tier: String,
    pub(super) arch: String,
    pub(super) debug_build: bool,
    /// The crate features that matter, `+`-separated (`none`).
    pub(super) features: String,
    pub(super) cpu: String,
}

/// The result of one case for the verdict.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CaseResult {
    /// Per candidate: work, role and the last attempt's paired median.
    pub(super) outcomes: Vec<CaseOutcome>,
    /// The last attempt's control is within both limits (median and
    /// quartiles, `stats::control_valid`).
    pub(super) valid: bool,
    /// Attempts made (1 + re-measurements).
    pub(super) attempts: usize,
    /// The last attempt's spread.
    pub(super) spread: f64,
}

/// The result of one group run.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct GroupResult {
    pub(super) run_id: String,
    /// Per candidate.
    pub(super) verdicts: Vec<Verdict>,
    pub(super) cases: usize,
    pub(super) remeasured: usize,
    pub(super) invalid: usize,
}

/// A group run's fixed context.
#[derive(Debug, Clone, Copy)]
pub(super) struct RunContext<'a> {
    pub(super) spec: &'a GroupSpec,
    pub(super) settings: &'a Settings,
    pub(super) run_id: &'a str,
}

/// The role of every entry, whether it engages, and its detail.
struct EntryRoles {
    roles: Vec<Role>,
    engaged: Vec<Option<bool>>,
    details: Vec<String>,
}

/// The roles from the correctness facts (`facts[0]` is A's).
fn roles_of(
    spec: &GroupSpec,
    case: &Case,
    entries: &[Entry],
    facts: &[Facts],
    ctx: &str,
) -> EntryRoles {
    let base = &facts[0];
    let mut out = EntryRoles {
        roles: Vec::new(),
        engaged: Vec::new(),
        details: Vec::new(),
    };
    for (entry, own) in entries.iter().zip(facts) {
        let (role, engaged, detail) = match entry.slot {
            Slot::Base => (Role::Base, None, own.detail.clone()),
            Slot::Control => (Role::Control, None, own.detail.clone()),
            Slot::Candidate(ci) => {
                let engaged = engages(spec.engage, base, own);
                let intended = case.intended.get(ci).copied().unwrap_or(false);
                // A route-group candidate that takes A's path must give A's
                // bits.
                let name = entry.variant.name;
                if spec.engage == Engage::Route && !engaged && !own.same_bits_as_base {
                    panic!("{ctx} {name}: A's path but not A's bits, not timed");
                }
                let detail = format!("{} {}", own.detail, seen(spec.engage, base, own));
                (Role::of_candidate(intended, engaged), Some(engaged), detail)
            }
        };
        out.roles.push(role);
        out.engaged.push(engaged);
        out.details.push(detail);
    }
    out
}

/// Checks the calls an engine recorded against the line-up `entries` and
/// their correctness `facts`: every call of a variant executed the
/// fingerprint of that variant's correctness call (the error shows both),
/// no call names a variant outside the line-up, and with `requested` (per
/// entry) every variant made exactly the calls asked of it.
pub(super) fn check_calls(
    entries: &[Entry],
    facts: &[Facts],
    calls: &[CallCount],
    requested: Option<&[usize]>,
) -> Result<(), String> {
    for call in calls {
        let at = entries.iter().position(|e| e.variant.name == call.variant);
        let Some(checked) = at.and_then(|v| facts.get(v)) else {
            let (n, name) = (call.calls, call.variant);
            return Err(format!("{n} calls of {name}, which is not in the line-up"));
        };
        if call.ran != checked.ran {
            let (n, name) = (call.calls, call.variant);
            return Err(format!(
                "{n} calls of {name} executed {}, its correctness call executed {}",
                call.ran.text(),
                checked.ran.text()
            ));
        }
    }
    let Some(requested) = requested else {
        return Ok(());
    };
    for (entry, asked) in entries.iter().zip(requested) {
        let name = entry.variant.name;
        let of = calls.iter().filter(|c| c.variant == name);
        let made: usize = of.map(|c| c.calls).sum();
        if made != *asked {
            return Err(format!(
                "{made} calls ran with the plan of {name}, {asked} were asked for"
            ));
        }
    }
    Ok(())
}

/// Runs one case: the correctness pass of every variant (each from a clean
/// packing cache), then up to `1 + retries` timed attempts until the control
/// is within both limits (`stats::control_valid`). Every attempt is printed;
/// the last one counts.
pub(super) fn run_case(
    ctx: &RunContext<'_>,
    case: &Case,
    engine: &mut dyn Engine,
    sink: &mut dyn Sink,
) -> CaseResult {
    let (spec, settings) = (ctx.spec, ctx.settings);
    let label = case.label();
    let at = format!("{} {} {label}", spec.name, spec.run);
    let entries = lineup(spec, spec.engage == Engage::Route);
    let mut facts = Vec::with_capacity(entries.len());
    for entry in &entries {
        let name = entry.variant.name;
        let clean = engine.start_clean();
        clean.unwrap_or_else(|e| panic!("{at} {name}: {e}; not timed"));
        let checked = engine.check(entry);
        facts.push(checked.unwrap_or_else(|e| panic!("{at} {name}: {e}; not timed")));
    }
    // The correctness and probe calls executed what the checks saw.
    let checked = check_calls(&entries, &facts, &engine.take_calls(), None);
    checked.unwrap_or_else(|e| panic!("{at}: {e}; not timed"));
    let roles = roles_of(spec, case, &entries, &facts, &at);

    let switch = spec.limits_differ();
    let control = entries.iter().position(|e| e.slot == Slot::Control);
    let control = control.unwrap_or_else(|| panic!("{at}: no control"));
    let ids = CaseIds {
        run_id: ctx.run_id,
        group: spec.name,
        run: &spec.run,
        threads: spec.threads,
        case: &label,
        dtype: case.dtype.name(),
        trans: &case.trans.name(),
        dims: case.dims,
        beta: u8::from(case.beta == Beta::One),
    };
    let mut last = None;
    for attempt in 0..=settings.retries {
        let fail = |e: String| -> ! { panic!("{at} attempt {attempt}: {e}") };
        engine.start_clean().unwrap_or_else(|e| fail(e));
        if !switch {
            engine.apply_limit(entries[0].variant.limit);
        }
        // Nothing runs between the checks or attempts: anything recorded
        // must still be what the checks saw.
        let between = check_calls(&entries, &facts, &engine.take_calls(), None);
        between.unwrap_or_else(|e| fail(e));
        let mut requested = vec![0; entries.len()];
        let min = settings.min_sample;
        let reps = calibrate(engine, &entries, min, switch, &mut requested);
        let reps = reps.unwrap_or_else(|e| fail(e));
        let shape = (settings.rounds, reps);
        let times = sample_rounds(engine, &entries, shape, switch, &mut requested);
        let times = times.unwrap_or_else(|e| fail(e));
        // Every warm-up and timed call executed what its variant's
        // correctness call executed, and each variant made the calls asked.
        let made = engine.take_calls();
        check_calls(&entries, &facts, &made, Some(&requested)).unwrap_or_else(|e| fail(e));
        let figures = variant_figures(&times);
        let control_ratio = figures[control].ratio.median;
        let valid = control_valid(&figures[control].ratio);
        let spread = spread_of(control_ratio);
        for (v, entry) in entries.iter().enumerate() {
            let ab = AbLine {
                ids,
                attempt,
                valid,
                variant: entry.variant.name,
                role: roles.roles[v],
                check: entry.check.word(),
                reps,
                rounds: settings.rounds,
                figures: &figures[v],
                spread,
                check_worst: facts[v].worst,
                engaged: roles.engaged[v],
                detail: &roles.details[v],
            };
            sink.emit(&ab.line());
            let abr = AbrLine {
                ids,
                attempt,
                variant: entry.variant.name,
                reps,
                per_round: &times[v],
            };
            sink.emit(&abr.line());
        }
        last = Some((figures, valid, spread, attempt + 1));
        if valid {
            break;
        }
    }
    let (figures, valid, spread, attempts) = last.unwrap_or_else(|| panic!("{at}: no attempt"));
    let outcomes = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e.slot, Slot::Candidate(_)))
        .map(|(v, _)| CaseOutcome {
            work: case.dims.work(),
            role: roles.roles[v],
            ratio: figures[v].ratio.median,
        })
        .collect();
    CaseResult {
        outcomes,
        valid,
        attempts,
        spread,
    }
}

/// Runs every selected case of `spec`, each on the engine `engine_for`
/// makes for it, and prints the headers, the `SET` line, the case lines, the
/// `ABV` line and the `END` line.
pub(super) fn run_group<'e>(
    spec: &GroupSpec,
    settings: &Settings,
    build: &BuildInfo,
    sink: &mut dyn Sink,
    engine_for: &mut dyn FnMut(&Case) -> Box<dyn Engine + 'e>,
) -> GroupResult {
    let run_id = new_run_id();
    let grade = settings.grade(build.debug_build);
    header_lines().iter().for_each(|line| sink.emit(line));
    let set = SetLine {
        run_id: run_id.clone(),
        group: spec.name.to_string(),
        run: spec.run.clone(),
        tier: build.tier.clone(),
        arch: build.arch.clone(),
        threads: spec.threads,
        rounds: settings.rounds,
        rounds_requested: settings.rounds_requested,
        min_sample_ms: settings.min_sample_ms(),
        quick: settings.quick,
        debug_build: build.debug_build,
        retries: settings.retries,
        tag: settings.tag.clone(),
        grade: grade.word().to_string(),
        features: build.features.clone(),
        cpu: build.cpu.clone(),
    };
    sink.emit(&set.line());
    let ctx = RunContext {
        spec,
        settings,
        run_id: &run_id,
    };
    let mut results = Vec::new();
    for case in spec.selected(settings.quick) {
        let mut engine = engine_for(case);
        results.push(run_case(&ctx, case, engine.as_mut(), sink));
    }
    let all_valid = results.iter().all(|r| r.valid);
    let verdicts: Vec<Verdict> = (spec.candidates().iter().enumerate())
        .map(|(ci, cand)| {
            let outcomes: Vec<CaseOutcome> = results.iter().map(|r| r.outcomes[ci]).collect();
            // A candidate that is no policy field has no range to adopt.
            let threshold = if cand.informational {
                Threshold::Informational
            } else {
                spec.threshold
            };
            verdict(&outcomes, all_valid, grade, threshold)
        })
        .collect();
    let entries = spec.candidates().iter().zip(&verdicts);
    let invalid = results.iter().filter(|r| !r.valid).count();
    let remeasured = results.iter().filter(|r| r.attempts > 1).count();
    let max_abs_spread = results.iter().fold(0.0_f64, |a, r| a.max(r.spread.abs()));
    let abv = AbvLine {
        run_id: &run_id,
        group: spec.name,
        run: &spec.run,
        threads: spec.threads,
        tier: &build.tier,
        grade: grade.word(),
        cases: results.len(),
        remeasured,
        invalid,
        max_abs_spread,
        verdicts: entries
            .map(|(v, verdict)| verdict_entry(v.name, verdict))
            .collect(),
    };
    sink.emit(&abv.line());
    let end = EndLine {
        run_id: run_id.clone(),
        group: spec.name.to_string(),
        run: spec.run.clone(),
        cases: results.len(),
        invalid_cases: invalid,
    };
    sink.emit(&end.line());
    GroupResult {
        run_id,
        verdicts,
        cases: results.len(),
        remeasured,
        invalid,
    }
}
