//! Logic tests of the pure parts: statistics, the rule, partial ranges,
//! settings and line formats.

use super::groups::Dims;
use super::output::{
    AB_COLUMNS, ABR_COLUMNS, ABV_COLUMNS, AbLine, AbrLine, AbvLine, CaseIds, EndLine, SetLine,
    append_text, data_line, header_lines, parse_line, verdict_entry,
};
use super::settings::{Grade, Settings, effective_rounds};
use super::stats::{
    CaseOutcome, Decision, Partial, QUARTILE_SPREAD_LIMIT, Quartiles, Role, SPREAD_LIMIT,
    Threshold, VariantFigures, control_valid, geomean, largest_passing_range, percentile,
    quartiles, rule_figures, spread_of, variant_figures, verdict,
};
use std::env::VarError;
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

/// An outcome of `work` with `role` and `ratio`.
const fn outcome(work: u128, role: Role, ratio: f64) -> CaseOutcome {
    CaseOutcome { work, role, ratio }
}

const T: Role = Role::Target;
const E: Role = Role::Elsewhere;
const N: Role = Role::TargetNoop;

/// The decision of a decision-grade, valid run.
fn decide(cases: &[CaseOutcome]) -> Decision {
    verdict(cases, true, Grade::Decision, Threshold::None).decision
}

#[test]
fn test_gemm_ab_percentile_and_median() {
    let (five, four): (&[f64], &[f64]) = (&[1.0, 2.0, 3.0, 4.0, 5.0], &[10.0, 20.0, 30.0, 40.0]);
    let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
    let want = [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 17.5, 25.0, 32.5, 40.0]];
    for (data, want) in [five, four].into_iter().zip(want) {
        assert_eq!(quantiles.map(|q| percentile(data, q)), want, "{data:?}");
    }
    assert_eq!(percentile(&[7.0], 0.3), 7.0);
    assert!(percentile(&[], 0.5).is_nan());
    // `quartiles` sorts first.
    let q = quartiles(&[9.0, 1.0, 5.0, 3.0, 7.0, 100.0, 2.0]);
    let want = Quartiles {
        median: 5.0,
        p25: 2.5,
        p75: 8.0,
    };
    assert_eq!(q, want);
}

#[test]
fn test_gemm_ab_geomean() {
    let close = |values: &[f64], want: f64| (geomean(values) - want).abs() < 1e-12;
    assert!(close(&[2.0, 8.0], 4.0) && close(&[1.05; 7], 1.05));
    assert!(close(&[0.5, 2.0, 1.0], 1.0) && close(&[1.0, 1.0, 1.21], 1.21_f64.cbrt()));
    for bad in [
        &[][..],
        &[1.0, 0.0],
        &[1.0, -1.0],
        &[1.0, f64::INFINITY],
        &[f64::NAN],
    ] {
        assert!(geomean(bad).is_nan(), "{bad:?}");
    }
}

#[test]
fn test_gemm_ab_paired_ratio_is_base_over_variant() {
    // Rounds of A, a variant twice as fast, a control with one slow round.
    let times = vec![
        vec![100.0, 110.0, 90.0, 100.0],
        vec![50.0, 55.0, 45.0, 50.0],
        vec![100.0, 110.0, 180.0, 100.0],
    ];
    let figures: Vec<VariantFigures> = variant_figures(&times);
    assert_eq!(figures[1].ratio.median, 2.0);
    assert_eq!((figures[1].ratio.p25, figures[1].ratio.p75), (2.0, 2.0));
    assert_eq!(figures[1].ratio_of_medians, 2.0);
    // Pairing removes the drift between rounds: the control's paired median
    // is 1 although one round was twice as slow.
    assert_eq!(figures[2].ratio.median, 1.0);
    assert_eq!(figures[0].ratio.median, 1.0);
    assert_eq!(figures[1].time.median, 50.0);
    assert_eq!(spread_of(figures[2].ratio.median), 0.0);
}

#[test]
fn test_gemm_ab_control_limit_is_symmetric() {
    let flat = |ratio: f64| Quartiles {
        median: ratio,
        p25: ratio,
        p75: ratio,
    };
    for ratio in [1.0, 1.015, 1.0 / 1.015, 1.01, 0.99] {
        assert!(control_valid(&flat(ratio)), "{ratio}");
    }
    for ratio in [
        1.0151,
        1.0 / 1.0151,
        0.98,
        1.02,
        f64::NAN,
        0.0,
        f64::INFINITY,
    ] {
        assert!(!control_valid(&flat(ratio)), "{ratio}");
    }
}

#[test]
fn test_gemm_ab_control_quartiles_limit() {
    // The limits: 1.5 % for the median, 1.015² (3.0225 %) for each
    // quartile, either way.
    assert_eq!(SPREAD_LIMIT, 1.015);
    assert!((QUARTILE_SPREAD_LIMIT - 1.030_225).abs() < 1e-12);
    let control = |p25: f64, median: f64, p75: f64| Quartiles { median, p25, p75 };
    let q = 1.030_225;
    // Valid: the median within 1.5 % and both quartiles within 1.015², at
    // the exact limits too.
    for (p25, median, p75) in [
        (1.0, 1.0, 1.0),
        (1.0 / q, 1.0, q),
        (1.0 / q, 1.015, q),
        (1.0 / q, 1.0 / 1.015, 1.0),
        (0.98, 1.0, 1.02),
        (1.02, 1.01, 1.03),
    ] {
        let c = control(p25, median, p75);
        assert!(control_valid(&c), "{c:?}");
    }
    // Invalid: the median alone (quartiles fine), each quartile alone (the
    // median fine), both quartiles, either side, and non-finite or
    // non-positive values in any of the three.
    for (p25, median, p75) in [
        (1.0, 1.0151, 1.0),
        (1.0, 1.0 / 1.0151, 1.0),
        (0.97, 1.0, 1.0),
        (1.0 / (q * 1.0001), 1.0, 1.0),
        (1.0, 1.0, 1.031),
        (1.0, 1.0, q * 1.0001),
        (0.94, 1.0, 1.003),
        (0.814, 1.0, 1.363),
        (0.95, 1.0, 1.05),
        (1.04, 1.01, 1.05),
        (0.95, 0.99, 0.96),
        (f64::NAN, 1.0, 1.0),
        (1.0, 1.0, f64::NAN),
        (1.0, f64::NAN, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, f64::INFINITY),
        (-1.0, 1.0, 1.0),
    ] {
        let c = control(p25, median, p75);
        assert!(!control_valid(&c), "{c:?}");
    }
}

#[test]
fn test_gemm_ab_rule_clauses() {
    let cases = [
        outcome(100, T, 1.10),
        outcome(200, T, 1.02),
        outcome(300, T, 1.08),
        outcome(400, E, 0.975),
        outcome(500, N, 1.0),
    ];
    let v = verdict(&cases, true, Grade::Decision, Threshold::None);
    assert_eq!(v.decision, Decision::Adopt);
    let f = v.figures;
    assert_eq!((f.targets, f.elsewhere, f.noop, f.passes), (3, 2, 1, true));
    assert!((f.geomean - (1.10_f64 * 1.02 * 1.08).cbrt()).abs() < 1e-12);
    assert_eq!(
        (f.min_target, f.min_elsewhere, v.partial),
        (1.02, 0.975, Partial::NotApplicable)
    );
    // Geometric mean just below 1.05 with every target >= 1.00.
    assert_eq!(
        decide(&[outcome(1, T, 1.04), outcome(2, T, 1.05)]),
        Decision::KeepLegacy
    );
    // A target below 1.00 although the mean is high; exactly 1.00 is enough.
    let low = [
        outcome(1, T, 1.30),
        outcome(2, T, 0.99),
        outcome(3, T, 1.20),
    ];
    assert_eq!(decide(&low), Decision::KeepLegacy);
    assert_eq!(
        decide(&[outcome(1, T, 1.0), outcome(2, T, 1.2)]),
        Decision::Adopt
    );
    // Elsewhere (and a no-op target, which counts as elsewhere) below 0.97.
    let targets = [outcome(1, T, 1.10), outcome(2, T, 1.10)];
    for (role, ratio, want) in [
        (E, 0.969, Decision::KeepLegacy),
        (N, 0.969, Decision::KeepLegacy),
        (E, 0.97, Decision::Adopt),
        (E, f64::NAN, Decision::KeepLegacy),
    ] {
        let cases = [&targets[..], &[outcome(3, role, ratio)]].concat();
        assert_eq!(decide(&cases), want, "{role:?} {ratio}");
    }
    let noop_only = rule_figures([outcome(1, N, 1.0)].iter());
    assert_eq!(
        (noop_only.targets, noop_only.elsewhere, noop_only.noop),
        (0, 1, 1)
    );
}

#[test]
fn test_gemm_ab_rule_order_of_decisions() {
    let adopt = [outcome(1, T, 1.5), outcome(2, E, 1.0)];
    let none = [outcome(1, N, 1.5), outcome(2, E, 1.0)];
    let check = |cases: &[CaseOutcome], valid, grade| {
        verdict(cases, valid, grade, Threshold::None).decision
    };
    // SMOKE first, whatever else holds.
    for (cases, valid) in [
        (&adopt, true),
        (&adopt, false),
        (&none, false),
        (&none, true),
    ] {
        assert_eq!(check(cases, valid, Grade::Smoke), Decision::Smoke);
    }
    // Then INVALID, then NO_TARGETS, then ADOPT / KEEP_LEGACY.
    assert_eq!(check(&adopt, false, Grade::Decision), Decision::Invalid);
    assert_eq!(check(&none, false, Grade::Decision), Decision::Invalid);
    assert_eq!(check(&none, true, Grade::Decision), Decision::NoTargets);
    assert_eq!(check(&adopt, true, Grade::Decision), Decision::Adopt);
    let keep = [outcome(1, T, 1.01)];
    assert_eq!(check(&keep, true, Grade::Decision), Decision::KeepLegacy);
    // The figures are printed in every case; partial only where evaluated.
    let v = verdict(&adopt, false, Grade::Decision, Threshold::WorkInterval);
    assert_eq!((v.partial, v.figures.targets), (Partial::NotEvaluated, 1));
    let smoke = verdict(&adopt, true, Grade::Smoke, Threshold::WorkInterval);
    assert!(matches!(smoke.partial, Partial::Range { .. }));
}

#[test]
fn test_gemm_ab_partial_ranges_per_threshold_kind() {
    let range = |lo, hi, cases| Partial::Range { lo, hi, cases };
    // Small sizes win, a large target loses.
    let first = [
        outcome(64, E, 1.0),
        outcome(512, T, 1.20),
        outcome(512, T, 1.10),
        outcome(4096, T, 1.08),
        outcome(32768, T, 0.95),
        outcome(262_144, E, 1.0),
    ];
    // Interval: any range; the largest passing one is 64..4096.
    assert_eq!(
        largest_passing_range(&first, Threshold::WorkInterval),
        range(64, 4096, 4)
    );
    // Upper bound: ranges start at the smallest engaged size (512).
    assert_eq!(
        largest_passing_range(&first, Threshold::UpperWork),
        range(512, 4096, 3)
    );
    // Lower bound: ranges end at the largest engaged size, which loses.
    assert_eq!(
        largest_passing_range(&first, Threshold::LowerWork),
        Partial::NoRange
    );
    assert_eq!(
        largest_passing_range(&first, Threshold::None),
        Partial::NotApplicable
    );
    // An interior range is an interval only.
    let middle = [
        outcome(1, T, 1.2),
        outcome(2, E, 0.9),
        outcome(3, T, 1.1),
        outcome(4, T, 1.1),
        outcome(5, E, 1.0),
    ];
    assert_eq!(
        largest_passing_range(&middle, Threshold::WorkInterval),
        range(3, 5, 3)
    );
    assert_eq!(
        largest_passing_range(&middle, Threshold::UpperWork),
        range(1, 1, 1)
    );
    // Lower bound: ranges end at the largest engaged size (4), not at 5.
    assert_eq!(
        largest_passing_range(&middle, Threshold::LowerWork),
        range(3, 4, 2)
    );
    // Lower bound: a loss at small sizes is cut off.
    let large = [
        outcome(8, T, 0.9),
        outcome(64, T, 1.2),
        outcome(512, T, 1.1),
    ];
    assert_eq!(
        largest_passing_range(&large, Threshold::LowerWork),
        range(64, 512, 2)
    );
    // Equal sizes are never split: one of the two size-8 cases loses.
    let equal = [outcome(8, T, 1.3), outcome(8, T, 0.98), outcome(9, T, 1.2)];
    assert_eq!(
        largest_passing_range(&equal, Threshold::WorkInterval),
        range(9, 9, 1)
    );
    // Nothing passes; no target at all.
    let lose = [outcome(1, T, 1.01), outcome(2, T, 0.9)];
    assert_eq!(
        largest_passing_range(&lose, Threshold::WorkInterval),
        Partial::NoRange
    );
    let idle = [outcome(1, E, 1.5)];
    assert_eq!(
        largest_passing_range(&idle, Threshold::UpperWork),
        Partial::NoRange
    );
    assert_eq!(range(1, 2, 3).text(), "1..2/3");
    let texts = [
        Partial::NotApplicable,
        Partial::NotEvaluated,
        Partial::NoRange,
    ]
    .map(Partial::text);
    assert_eq!(texts, ["n/a", "-", "none"]);
}

#[test]
fn test_gemm_ab_roles_of_candidates() {
    assert_eq!(Role::of_candidate(true, true), Role::Target);
    assert_eq!(Role::of_candidate(false, true), Role::Target);
    assert_eq!(Role::of_candidate(true, false), Role::TargetNoop);
    assert_eq!(Role::of_candidate(false, false), Role::Elsewhere);
    let words = [Role::Base, Role::Control, T, E, N].map(Role::word);
    assert_eq!(
        words,
        ["base", "control", "target", "elsewhere", "target_noop"]
    );
}

/// Settings from a space-separated `KEY=value` table.
fn settings_of(table: &str, cores: usize) -> Result<Settings, String> {
    let pairs: Vec<(&str, &str)> = table.split(' ').filter_map(|p| p.split_once('=')).collect();
    let env = |key: &str| {
        let found = pairs.iter().find(|p| p.0 == key);
        found.map(|p| p.1.to_string()).ok_or(VarError::NotPresent)
    };
    Settings::from_lookup(env, cores)
}

#[test]
fn test_gemm_ab_settings_from_environment() {
    let defaults = settings_of("", 6).unwrap_or_else(|e| panic!("{e}"));
    let want = Settings {
        rounds_requested: 22,
        rounds: 22,
        min_sample: Duration::from_millis(2),
        threads: vec![6],
        quick: false,
        retries: 2,
        tag: String::new(),
        out: None,
    };
    assert_eq!(defaults, want);
    assert_eq!(defaults.grade(false), Grade::Decision);
    assert_eq!(defaults.grade(true), Grade::Smoke);
    let all = "OXIBLAS_AB_ROUNDS=31 OXIBLAS_AB_MIN_SAMPLE_MS=0.5 OXIBLAS_AB_THREADS=2,4,8 \
               OXIBLAS_AB_QUICK=1 OXIBLAS_AB_OUT=target/ab.tsv OXIBLAS_AB_RETRIES=0 \
               OXIBLAS_AB_TAG=pass1";
    let set = settings_of(all, 6).unwrap_or_else(|e| panic!("{e}"));
    let want = Settings {
        rounds_requested: 31,
        rounds: 32,
        min_sample: Duration::from_micros(500),
        threads: vec![2, 4, 8],
        quick: true,
        retries: 0,
        tag: "pass1".to_string(),
        out: Some(PathBuf::from("target/ab.tsv")),
    };
    assert_eq!(set, want);
    // Even rounds, at least 4; empty values count as unset.
    assert_eq!(
        [1, 3, 4, 5, 20, 21, 22].map(effective_rounds),
        [4, 4, 4, 6, 20, 22, 22]
    );
    let low = settings_of(
        "OXIBLAS_AB_ROUNDS=1 OXIBLAS_AB_QUICK= OXIBLAS_AB_THREADS=",
        0,
    );
    let low = low.unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (low.rounds, low.threads.clone(), low.quick),
        (4, vec![1], false)
    );
    // Grades: each smoke condition alone.
    for table in [
        "OXIBLAS_AB_ROUNDS=20",
        "OXIBLAS_AB_MIN_SAMPLE_MS=1.9",
        "OXIBLAS_AB_QUICK=1",
    ] {
        let s = settings_of(table, 4).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.grade(false), Grade::Smoke, "{table}");
    }
    let s = settings_of("OXIBLAS_AB_ROUNDS=21 OXIBLAS_AB_MIN_SAMPLE_MS=2", 4);
    assert_eq!(s.map(|s| s.grade(false)), Ok(Grade::Decision));
    let bad = "OXIBLAS_AB_ROUNDS=many OXIBLAS_AB_MIN_SAMPLE_MS=0 OXIBLAS_AB_MIN_SAMPLE_MS=-1 \
               OXIBLAS_AB_MIN_SAMPLE_MS=NaN OXIBLAS_AB_THREADS=4,0 OXIBLAS_AB_THREADS=4,,8 \
               OXIBLAS_AB_THREADS=x OXIBLAS_AB_RETRIES=11 OXIBLAS_AB_RETRIES=-1";
    for pair in bad.split_whitespace() {
        assert!(settings_of(pair, 4).is_err(), "{pair}");
    }
    let only = |key: &'static str, value: &'static str| {
        move |k: &str| {
            let found = (k == key).then(|| value.to_string());
            found.ok_or(VarError::NotPresent)
        }
    };
    let tab = Settings::from_lookup(only("OXIBLAS_AB_TAG", "a\tb"), 4);
    assert!(tab.is_err());
    // A value that is not valid Unicode is an error, for every key.
    let keys = [
        "OXIBLAS_AB_ROUNDS",
        "OXIBLAS_AB_MIN_SAMPLE_MS",
        "OXIBLAS_AB_THREADS",
        "OXIBLAS_AB_QUICK",
        "OXIBLAS_AB_RETRIES",
        "OXIBLAS_AB_TAG",
        "OXIBLAS_AB_OUT",
    ];
    for key in keys {
        let raw = move |k: &str| {
            if k == key {
                Err(VarError::NotUnicode(OsString::from("1")))
            } else {
                Err(VarError::NotPresent)
            }
        };
        let read = Settings::from_lookup(raw, 4);
        assert!(
            read.as_ref()
                .is_err_and(|e| e.contains("not valid Unicode")),
            "{key}: {read:?}"
        );
    }
    // A value of only white space counts as unset, for every key.
    let defaults = settings_of("", 4).unwrap_or_else(|e| panic!("{e}"));
    for key in keys {
        for blank in [" ", "\t", " \t "] {
            let read = Settings::from_lookup(only(key, blank), 4);
            assert_eq!(read.as_ref(), Ok(&defaults), "{key}={blank:?}");
        }
    }
}

#[test]
fn test_gemm_ab_features_list_sme2() {
    let features = super::features();
    let listed: Vec<&str> = features.split('+').collect();
    assert_eq!(
        listed.contains(&"sme2"),
        cfg!(feature = "sme2"),
        "{features}"
    );
    assert_eq!(
        listed.contains(&"parallel"),
        cfg!(feature = "parallel"),
        "{features}"
    );
}

/// The ids of a test case line.
fn ids() -> CaseIds<'static> {
    CaseIds {
        run_id: "123-456-7",
        group: "ab_seq_thin",
        run: "seq",
        threads: 1,
        case: "NN_1x256x256_f64",
        dtype: "f64",
        trans: "NN",
        dims: Dims {
            m: 1,
            k: 256,
            n: 256,
        },
        beta: 0,
    }
}

#[test]
fn test_gemm_ab_output_line_formats() {
    let [ab, abv, abr] = header_lines();
    fn header<'s>(kind: &'s str, cols: &[&'s str]) -> Vec<&'s str> {
        let mut fields = vec![kind];
        fields.extend(cols);
        fields
    }
    assert_eq!(parse_line(&ab), Some(("AB#", header("AB", &AB_COLUMNS))));
    assert_eq!(parse_line(&abv), Some(("AB#", header("ABV", &ABV_COLUMNS))));
    assert_eq!(parse_line(&abr), Some(("AB#", header("ABR", &ABR_COLUMNS))));
    // An AB line parses back with one field per column.
    let figures = VariantFigures {
        time: Quartiles {
            median: 2000.0,
            p25: 1900.0,
            p75: 2100.0,
        },
        ratio: Quartiles {
            median: 1.25,
            p25: 1.2,
            p75: 1.3,
        },
        ratio_of_medians: 1.24,
    };
    let line = AbLine {
        ids: ids(),
        attempt: 1,
        valid: true,
        variant: "thin",
        role: Role::TargetNoop,
        check: "reference",
        reps: 3,
        rounds: 22,
        figures: &figures,
        spread: -0.004,
        check_worst: Some(0.0123),
        engaged: Some(false),
        detail: "route=Thin\tkernel=8x8",
    }
    .line();
    let Some(("AB", fields)) = parse_line(&line) else {
        panic!("{line}");
    };
    assert_eq!(fields.len(), AB_COLUMNS.len());
    let field = |name: &str| {
        let at = AB_COLUMNS.iter().position(|c| *c == name);
        at.and_then(|i| fields.get(i)).copied().unwrap_or("?")
    };
    let pairs = [
        ("run_id", "123-456-7"),
        ("attempt", "1"),
        ("valid", "yes"),
        ("role", "target_noop"),
        ("median_ns", "2000.0"),
        ("gflops", "65.536"),
        ("ratio", "1.2500"),
        ("ratio_p25", "1.2000"),
        ("ratio_of_medians", "1.2400"),
        ("spread", "-0.0040"),
        ("check_worst", "1.230e-2"),
        ("engaged", "no"),
        ("detail", "route=Thin kernel=8x8"),
    ];
    for (name, want) in pairs {
        assert_eq!(field(name), want, "{name}");
    }
    let raw = AbrLine {
        ids: ids(),
        attempt: 0,
        variant: "off",
        reps: 3,
        per_round: &[10.0, 20.5],
    }
    .line();
    assert_eq!(
        raw,
        "ABR\t123-456-7\tab_seq_thin\tseq\tNN_1x256x256_f64\t0\toff\t3\t10.0,20.5"
    );
    assert_eq!(
        parse_line(&raw).map(|(_, f)| f.len()),
        Some(ABR_COLUMNS.len())
    );
    let cases = [outcome(1, T, 1.1)];
    let v = verdict(&cases, true, Grade::Decision, Threshold::UpperWork);
    let entry = verdict_entry("thin", &v);
    assert_eq!(
        entry,
        "thin:ADOPT,geo=1.1000,min_t=1.1000,min_e=inf,nt=1,ne=0,noop=0,partial=1..1/1"
    );
    let abv_line = AbvLine {
        run_id: "123-456-7",
        group: "ab_seq_thin",
        run: "seq",
        threads: 1,
        tier: "Neon",
        grade: "decision",
        cases: 1,
        remeasured: 0,
        invalid: 0,
        max_abs_spread: 0.004,
        verdicts: vec![entry.clone(), entry],
    }
    .line();
    let Some(("ABV", abv_fields)) = parse_line(&abv_line) else {
        panic!("{abv_line}");
    };
    assert_eq!(abv_fields.len(), ABV_COLUMNS.len());
    assert_eq!(abv_fields[10].split(';').count(), 2);
    let set = SetLine {
        run_id: "1-2-3".to_string(),
        group: "g".to_string(),
        run: "seq".to_string(),
        tier: "Avx2Fma".to_string(),
        arch: "x86_64".to_string(),
        threads: 1,
        rounds: 22,
        rounds_requested: 21,
        min_sample_ms: 2.0,
        quick: false,
        debug_build: false,
        retries: 2,
        tag: "pass 1".to_string(),
        grade: "decision".to_string(),
        features: "parallel".to_string(),
        cpu: "x".to_string(),
    }
    .line();
    assert!(
        set.starts_with("AB#\tSET\t1-2-3\tgroup=g\trun=seq\ttier=Avx2Fma\t"),
        "{set}"
    );
    assert!(set.contains("\ttag=pass 1\tgrade=decision\t"), "{set}");
    let end = EndLine {
        run_id: "1-2-3".to_string(),
        group: "g".to_string(),
        run: "seq".to_string(),
        cases: 4,
        invalid_cases: 1,
    }
    .line();
    assert_eq!(
        end,
        "AB#\tEND\t1-2-3\tgroup=g\trun=seq\tcases=4\tinvalid_cases=1"
    );
    assert_eq!(data_line("AB", &["a\nb".to_string()]), "AB\ta b");
    assert!(parse_line("test foo ... ok").is_none());
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
    let result = append_text(&path, "AB\ta\n").and_then(|()| append_text(&path, "ABV\tb\n"));
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string());
    let _ = std::fs::remove_file(&path);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(text, Ok("AB\ta\nABV\tb\n".to_string()));
}

#[test]
fn test_gemm_ab_engagement_per_kind() {
    use super::super::gemm_kernel::MicroKernelShape;
    use super::super::gemm_plan::{GemmRoute, PackScope};
    use super::super::gemm_thin::ThinSplits;
    use super::engage::{CacheUse, DriverRuns, Engage, Facts, Fingerprint, Path, engages, seen};
    let base = Facts {
        ran: Fingerprint {
            path: Path {
                route: GemmRoute::Blocked,
                simd: false,
            },
            kernel: MicroKernelShape { mr: 8, nr: 8 },
            kc: 256,
            seq_nc: 4096,
            par_nc: 4096,
            scope: PackScope::Shared,
            par: oxiblas_core::parallel::Par::Seq,
            pool: None,
        },
        runs: DriverRuns {
            shared: 1,
            per_tile: 0,
            declined: 0,
        },
        thin: ThinSplits::default(),
        nc: 4096,
        cache: Some(CacheUse {
            allocations: 2,
            parked: 0,
        }),
        same_bits_as_base: true,
        worst: None,
        detail: String::new(),
    };
    let kinds = [
        Engage::Route,
        Engage::Kernel,
        Engage::Scope,
        Engage::BCap,
        Engage::CacheLimit,
    ];
    // Identical facts engage nowhere.
    assert!(kinds.iter().all(|&kind| !engages(kind, &base, &base)));
    let mut route = base.clone();
    route.ran.path.simd = true;
    let mut kernel = base.clone();
    kernel.ran.kernel = MicroKernelShape { mr: 16, nr: 6 };
    let mut scope = base.clone();
    scope.ran.scope = PackScope::PerTile;
    let mut scope_ran = scope.clone();
    scope_ran.runs = DriverRuns {
        shared: 0,
        per_tile: 1,
        declined: 0,
    };
    let mut nc = base.clone();
    nc.nc = 2048;
    let mut cache = base.clone();
    cache.cache = Some(CacheUse {
        allocations: 0,
        parked: 1 << 20,
    });
    // Each kind sees only its own fact; a planned scope that did not run
    // does not engage.
    let table = [
        (&route, [true, false, false, false, false]),
        (&kernel, [false, true, false, false, false]),
        (&scope, [false; 5]),
        (&scope_ran, [false, false, true, false, false]),
        (&nc, [false, false, false, true, false]),
        (&cache, [false, false, false, false, true]),
    ];
    for (cand, want) in table {
        assert_eq!(
            kinds.map(|kind| engages(kind, &base, cand)),
            want,
            "{cand:?}"
        );
    }
    assert_eq!(seen(Engage::BCap, &base, &nc), "seen=4096->2048");
    assert_eq!(
        seen(Engage::Route, &base, &route),
        "seen=Blocked->Blocked+simd"
    );
    assert_eq!(
        seen(Engage::CacheLimit, &base, &cache),
        "seen=allocs:2,parked:0->allocs:0,parked:1048576"
    );
}

#[test]
fn test_gemm_ab_cache_model() {
    use super::super::gemm::GemmBlocking;
    use super::super::gemm_kernel::MicroKernelShape;
    use super::engage::{CacheUse, predicted_cache_use};
    let shape = MicroKernelShape { mr: 8, nr: 6 };
    let blocks = GemmBlocking {
        mc: 64,
        kc: 256,
        nc: 8196,
    };
    // A: 64·256·8 = 128 KiB; B: 256·8196·8 ≈ 16.0 MiB.
    let at = |limit| predicted_cache_use(&blocks, [64, 512, 32_768], &shape, 8, limit);
    let free_b = CacheUse {
        allocations: 1,
        parked: 64 * 256 * 8,
    };
    assert_eq!(at(8 << 20), free_b);
    let both = CacheUse {
        allocations: 0,
        parked: 64 * 256 * 8 + 256 * 8196 * 8,
    };
    assert_eq!(at(32 << 20), both);
    // B fills the cache exactly: A has no room left.
    assert_eq!(at(256 * 8196 * 8).allocations, 1);
}
