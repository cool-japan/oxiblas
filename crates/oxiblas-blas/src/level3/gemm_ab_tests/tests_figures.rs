//! Which figures reach the verdict: the paired ratios under a drift that
//! separates them from the ratio of medians, an invalid case in a group
//! run, and a candidate whose verdict is informational.

use super::groups::{Case, all_specs};
use super::output::{ABV_COLUMNS, VecSink, parse_line};
use super::runner::run_group;
use super::sampler::Engine;
use super::settings::Grade;
use super::stats::{
    CaseOutcome, Decision, Partial, Role, Threshold, largest_passing_range, verdict,
};
use super::tests_chain::{
    COSTS, Drift, FakeEngine, ab_rows, fake_build, fake_case, fake_settings, fake_spec, get,
    number, rows_of,
};

/// The median of `values` (the mean of the two middle ones for an even
/// count), computed here independently of `stats`.
fn median_of(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// The per-round times (ns per call) of `variant` in `attempt` from the
/// `ABR` lines.
fn per_round(lines: &[String], variant: &str, attempt: usize) -> Vec<f64> {
    let attempt = attempt.to_string();
    let rows = lines.iter().filter_map(|l| parse_line(l));
    let mut found = rows.filter(|(kind, fields)| {
        let (at, name) = (fields.get(4).copied(), fields.get(5).copied());
        *kind == "ABR" && name == Some(variant) && at == Some(attempt.as_str())
    });
    let (_, fields) = found
        .next()
        .unwrap_or_else(|| panic!("no ABR line of {variant}"));
    let times = fields.get(7).copied().unwrap_or_default().split(',');
    times
        .map(|t| t.parse().unwrap_or_else(|e| panic!("{t:?}: {e}")))
        .collect()
}

/// A fake engine whose costs drift from sample to sample: A is three times
/// slower in every other sample, `b` four times slower in every fourth, the
/// control 2 % slower in every fourth sample where A is fast (within the
/// control's quartile limit, so the case stays valid). The paired median
/// ratios are then `b` 2 and control 1, the ratios of medians `b` 4 and
/// control 200/201.
fn drift_engine() -> FakeEngine {
    let costs: [(&'static str, &[f64]); 4] = [
        ("a", &[100.0]),
        ("b", &[50.0]),
        ("c", &[200.0]),
        ("control", &[100.0]),
    ];
    let mut engine = FakeEngine::new(&costs, &["b"], 4);
    engine.drift = vec![
        Drift {
            variant: "a",
            factors: vec![1.0, 3.0],
        },
        Drift {
            variant: "b",
            factors: vec![1.0, 1.0, 1.0, 4.0],
        },
        Drift {
            variant: "control",
            factors: vec![1.02, 3.0, 1.0, 3.0],
        },
    ];
    engine
}

#[test]
fn test_gemm_ab_verdict_uses_paired_figures_under_drift() {
    let spec = fake_spec();
    let mut engine = drift_engine();
    let (result, lines) = fake_case(&spec, &fake_settings(8, 0), &mut engine);
    let close = |x: f64, want: f64| (x - want).abs() < 1e-3;
    // The two figures differ clearly under this drift (computed here from
    // the raw rounds).
    let base = per_round(&lines, "a", 0);
    let paired = |v: &str| {
        let own = per_round(&lines, v, 0);
        let ratios: Vec<f64> = base.iter().zip(&own).map(|(a, x)| a / x).collect();
        median_of(&ratios)
    };
    let of_medians = |v: &str| median_of(&base) / median_of(&per_round(&lines, v, 0));
    assert!(
        close(paired("b"), 2.0) && close(of_medians("b"), 4.0),
        "{lines:?}"
    );
    assert!(close(paired("control"), 1.0), "{lines:?}");
    assert!(close(of_medians("control"), 200.0 / 201.0), "{lines:?}");
    // The outcome, the AB ratio and the spread are the paired figures; the
    // ratio of medians is printed beside them.
    assert!(close(result.outcomes[0].ratio, 2.0), "{result:?}");
    assert!(close(result.spread, 0.0) && result.valid, "{result:?}");
    let rows = ab_rows(&lines);
    let b = rows_of(&rows, "b", 0)[0];
    assert!(close(number(b, "ratio"), 2.0) && close(number(b, "ratio_of_medians"), 4.0));
    assert!(close(number(b, "spread"), 0.0) && get(b, "valid") == "yes");
    // Over a decision-grade group run: the geometric mean of the paired
    // ratios, and the largest spread from the control's paired ratio.
    let settings = fake_settings(24, 0);
    let mut sink = VecSink::default();
    let mut make = |_: &Case| -> Box<dyn Engine> { Box::new(drift_engine()) };
    let group = run_group(&spec, &settings, &fake_build(false), &mut sink, &mut make);
    assert_eq!(group.verdicts[0].decision, Decision::Adopt);
    assert!(close(group.verdicts[0].figures.geomean, 2.0), "{group:?}");
    let abv = abv_fields(&sink.lines);
    assert_eq!(abv_field(&abv, "max_abs_spread"), "0.0000", "{abv:?}");
    assert_eq!(abv_field(&abv, "invalid"), "0", "{abv:?}");
    assert!(
        abv_field(&abv, "verdicts").starts_with("b:ADOPT,geo=2.0000,"),
        "{abv:?}"
    );
}

/// The fields of the `ABV` line of `lines`.
fn abv_fields(lines: &[String]) -> Vec<String> {
    let abv = lines.iter().filter_map(|l| parse_line(l));
    let mut abv = abv.filter(|(kind, _)| *kind == "ABV");
    let (_, fields) = abv.next().unwrap_or_else(|| panic!("no ABV in {lines:?}"));
    fields.iter().map(|f| f.to_string()).collect()
}

/// The `column` of an `ABV` line's fields.
fn abv_field<'f>(fields: &'f [String], column: &str) -> &'f str {
    let at = ABV_COLUMNS.iter().position(|c| *c == column);
    at.and_then(|i| fields.get(i)).map_or("", String::as_str)
}

#[test]
fn test_gemm_ab_invalid_case_makes_every_candidate_invalid() {
    let spec = fake_spec();
    // The second case's control stays 30 % off through every retry.
    let bad: [(&'static str, &[f64]); 4] = [
        ("a", &[100.0]),
        ("b", &[50.0]),
        ("c", &[200.0]),
        ("control", &[130.0]),
    ];
    let mut made = 0;
    let mut make = |_: &Case| -> Box<dyn Engine> {
        made += 1;
        let costs = if made == 2 { &bad } else { &COSTS };
        Box::new(FakeEngine::new(costs, &["b"], 4))
    };
    let mut sink = VecSink::default();
    let settings = fake_settings(22, 2);
    let result = run_group(&spec, &settings, &fake_build(false), &mut sink, &mut make);
    let lines = sink.lines;
    let decisions: Vec<Decision> = result.verdicts.iter().map(|v| v.decision).collect();
    assert_eq!(decisions, [Decision::Invalid, Decision::Invalid]);
    assert_eq!((result.cases, result.invalid, result.remeasured), (2, 1, 1));
    let abv = abv_fields(&lines);
    let counts = ["cases", "remeasured", "invalid"].map(|c| abv_field(&abv, c).to_string());
    assert_eq!(counts, ["2", "1", "1"], "{abv:?}");
    // The invalid case's outcome is still in the figures: `b` engages on
    // both cases, `c` on none (a no-op target on the second).
    let verdicts = abv_field(&abv, "verdicts");
    assert!(
        verdicts.starts_with("b:INVALID,")
            && verdicts.contains(",nt=2,ne=0,noop=0,partial=n/a;c:INVALID,")
            && verdicts.ends_with(",nt=0,ne=2,noop=1,partial=n/a"),
        "{verdicts}"
    );
    let end = lines.last().cloned().unwrap_or_default();
    assert!(end.ends_with("\tcases=2\tinvalid_cases=1"), "{end}");
    // Every attempt of the invalid case is printed, each invalid.
    let rows = ab_rows(&lines);
    let of_case = |r: &&[(&'static str, String)]| get(r, "case") == "NN_8x8x8_f64";
    for attempt in 0..3 {
        let printed: Vec<_> = rows_of(&rows, "b", attempt)
            .into_iter()
            .filter(of_case)
            .collect();
        assert!(
            printed.len() == 1 && get(printed[0], "valid") == "no",
            "attempt {attempt}: {lines:?}"
        );
    }
}

#[test]
fn test_gemm_ab_informational_candidate_has_no_range() {
    let outcome = |work: u128, role: Role, ratio: f64| CaseOutcome { work, role, ratio };
    let cases = [
        outcome(64, Role::Target, 1.2),
        outcome(512, Role::Target, 1.1),
        outcome(4096, Role::Elsewhere, 1.0),
    ];
    // The decision is still made; no range is computed and the text says so.
    for grade in [Grade::Decision, Grade::Smoke] {
        let v = verdict(&cases, true, grade, Threshold::Informational);
        assert_eq!(v.partial, Partial::Informational);
        assert_eq!(v.partial.text(), "informational");
    }
    let decided = verdict(&cases, true, Grade::Decision, Threshold::Informational);
    assert_eq!(decided.decision, Decision::Adopt);
    let range = largest_passing_range(&cases, Threshold::Informational);
    assert_eq!(range, Partial::Informational);
    assert!(matches!(
        largest_passing_range(&cases, Threshold::UpperWork),
        Partial::Range { .. }
    ));
    // In a group run with a work threshold: `b` gets its range, the
    // informational `c` none.
    let mut spec = fake_spec();
    spec.threshold = Threshold::UpperWork;
    spec.variants[2].informational = true;
    let mut sink = VecSink::default();
    let mut make =
        |_: &Case| -> Box<dyn Engine> { Box::new(FakeEngine::new(&COSTS, &["b", "c"], 4)) };
    let settings = fake_settings(22, 0);
    let result = run_group(&spec, &settings, &fake_build(false), &mut sink, &mut make);
    let partials: Vec<Partial> = result.verdicts.iter().map(|v| v.partial).collect();
    let range = Partial::Range {
        lo: 64,
        hi: 512,
        cases: 2,
    };
    assert_eq!(partials, [range, Partial::Informational]);
    let verdicts = abv_field(&abv_fields(&sink.lines), "verdicts").to_string();
    assert!(
        verdicts.contains(",partial=64..512/2;c:KEEP_LEGACY,")
            && verdicts.ends_with(",partial=informational"),
        "{verdicts}"
    );
    // Only the forced kernel and the candidates of the information group
    // are informational among the groups.
    for spec in all_specs(&[2]) {
        for v in &spec.variants {
            let forced = v.name == "forced16x6";
            let grid = spec.name == "ab_seq_thin_grid" && v.name != "off";
            assert_eq!(v.informational, forced || grid, "{} {}", spec.name, v.name);
        }
    }
    let grid = super::groups::spec_seq_thin_grid();
    let flags: Vec<(&str, bool)> = grid
        .candidates()
        .iter()
        .map(|v| (v.name, v.informational))
        .collect();
    assert_eq!(flags, [("thin", true), ("thin_scoped", true)]);
    #[cfg(target_arch = "x86_64")]
    {
        let kernel = super::groups::spec_seq_f32_kernel();
        let flags: Vec<(&str, bool)> = kernel
            .candidates()
            .iter()
            .map(|v| (v.name, v.informational))
            .collect();
        assert_eq!(flags, [("adaptive", false), ("forced16x6", true)]);
    }
}
