//! The output lines (see the module documentation for the columns) and the
//! sinks they go to.

use super::groups::Dims;
use super::stats::{Role, VariantFigures, Verdict};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Columns of an `AB` line after the kind, in order.
pub(super) const AB_COLUMNS: [&str; 30] = [
    "run_id",
    "group",
    "run",
    "threads",
    "case",
    "dtype",
    "trans",
    "m",
    "k",
    "n",
    "beta",
    "attempt",
    "valid",
    "variant",
    "role",
    "check",
    "reps",
    "rounds",
    "median_ns",
    "p25_ns",
    "p75_ns",
    "gflops",
    "ratio",
    "ratio_p25",
    "ratio_p75",
    "ratio_of_medians",
    "spread",
    "check_worst",
    "engaged",
    "detail",
];

/// Columns of an `ABR` line after the kind, in order.
pub(super) const ABR_COLUMNS: [&str; 8] = [
    "run_id",
    "group",
    "run",
    "case",
    "attempt",
    "variant",
    "reps",
    "ns_per_call",
];

/// Columns of an `ABV` line after the kind, in order.
pub(super) const ABV_COLUMNS: [&str; 11] = [
    "run_id",
    "group",
    "run",
    "threads",
    "tier",
    "grade",
    "cases",
    "remeasured",
    "invalid",
    "max_abs_spread",
    "verdicts",
];

/// The three column-header lines (`AB#` kind, then `AB`, `ABV`, `ABR`).
pub(super) fn header_lines() -> [String; 3] {
    let line = |kind: &str, cols: &[&str]| format!("AB#\t{kind}\t{}", cols.join("\t"));
    [
        line("AB", &AB_COLUMNS),
        line("ABV", &ABV_COLUMNS),
        line("ABR", &ABR_COLUMNS),
    ]
}

/// A line of kind `kind` with `fields` in column order (a tab or line break
/// inside a field becomes a space).
pub(super) fn data_line(kind: &str, fields: &[String]) -> String {
    let clean = |f: &String| f.replace(['\t', '\n', '\r'], " ");
    let fields: Vec<String> = fields.iter().map(clean).collect();
    format!("{kind}\t{}", fields.join("\t"))
}

/// Splits a harness line into its kind (`AB`, `ABR`, `ABV`, `AB#`) and
/// fields.
pub(super) fn parse_line(line: &str) -> Option<(&str, Vec<&str>)> {
    let mut parts = line.split('\t');
    let kind = parts.next()?;
    matches!(kind, "AB" | "ABR" | "ABV" | "AB#").then(|| (kind, parts.collect()))
}

/// A unique id of one group run: `<pid>-<unix ms>-<sequence>` (no tab, no
/// space).
pub(super) fn new_run_id() -> String {
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
    let ms = now.map_or(0, |d| d.as_millis());
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("{}-{ms}-{seq}", std::process::id())
}

/// A `true`/`false` field as `yes`/`no`.
const fn yes_no(flag: bool) -> &'static str {
    if flag { "yes" } else { "no" }
}

/// The settings and build of one group run: `AB#\tSET\t<run_id>\tkey=value…`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SetLine {
    pub(super) run_id: String,
    pub(super) group: String,
    pub(super) run: String,
    pub(super) tier: String,
    pub(super) arch: String,
    pub(super) threads: usize,
    pub(super) rounds: usize,
    pub(super) rounds_requested: usize,
    pub(super) min_sample_ms: f64,
    pub(super) quick: bool,
    pub(super) debug_build: bool,
    pub(super) retries: usize,
    pub(super) tag: String,
    pub(super) grade: String,
    pub(super) features: String,
    pub(super) cpu: String,
}

impl SetLine {
    /// The line.
    pub(super) fn line(&self) -> String {
        let pairs = [
            ("group", self.group.clone()),
            ("run", self.run.clone()),
            ("tier", self.tier.clone()),
            ("arch", self.arch.clone()),
            ("threads", self.threads.to_string()),
            ("rounds", self.rounds.to_string()),
            ("rounds_requested", self.rounds_requested.to_string()),
            ("min_sample_ms", format!("{}", self.min_sample_ms)),
            ("quick", self.quick.to_string()),
            ("debug_build", self.debug_build.to_string()),
            ("retries", self.retries.to_string()),
            ("tag", self.tag.clone()),
            ("grade", self.grade.clone()),
            ("features", self.features.clone()),
            ("cpu", self.cpu.clone()),
            ("pid", std::process::id().to_string()),
        ];
        let mut fields = vec!["SET".to_string(), self.run_id.clone()];
        fields.extend(pairs.iter().map(|(k, v)| format!("{k}={v}")));
        data_line("AB#", &fields)
    }
}

/// The end of a completed group run: `AB#\tEND\t<run_id>\tkey=value…`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct EndLine {
    pub(super) run_id: String,
    pub(super) group: String,
    pub(super) run: String,
    pub(super) cases: usize,
    pub(super) invalid_cases: usize,
}

impl EndLine {
    /// The line.
    pub(super) fn line(&self) -> String {
        let fields = [
            "END".to_string(),
            self.run_id.clone(),
            format!("group={}", self.group),
            format!("run={}", self.run),
            format!("cases={}", self.cases),
            format!("invalid_cases={}", self.invalid_cases),
        ];
        data_line("AB#", &fields)
    }
}

/// Where a line's case and run come from (shared by `AB` and `ABR` lines).
#[derive(Debug, Clone, Copy)]
pub(super) struct CaseIds<'a> {
    pub(super) run_id: &'a str,
    pub(super) group: &'a str,
    pub(super) run: &'a str,
    pub(super) threads: usize,
    pub(super) case: &'a str,
    pub(super) dtype: &'a str,
    pub(super) trans: &'a str,
    pub(super) dims: Dims,
    /// 0 or 1.
    pub(super) beta: u8,
}

/// One `AB` line: one variant in one attempt of one case.
#[derive(Debug, Clone, Copy)]
pub(super) struct AbLine<'a> {
    pub(super) ids: CaseIds<'a>,
    pub(super) attempt: usize,
    pub(super) valid: bool,
    pub(super) variant: &'a str,
    pub(super) role: Role,
    pub(super) check: &'a str,
    pub(super) reps: usize,
    pub(super) rounds: usize,
    pub(super) figures: &'a VariantFigures,
    pub(super) spread: f64,
    /// The largest `|error| / tolerance` of the correctness check (`None`:
    /// bitwise only).
    pub(super) check_worst: Option<f64>,
    /// Whether this variant engages (`None` for A and the control).
    pub(super) engaged: Option<bool>,
    pub(super) detail: &'a str,
}

impl AbLine<'_> {
    /// The line.
    pub(super) fn line(&self) -> String {
        let ids = &self.ids;
        let fig = self.figures;
        let flops = 2.0 * ids.dims.work() as f64;
        let worst = self
            .check_worst
            .map_or("-".to_string(), |w| format!("{w:.3e}"));
        let engaged = self.engaged.map_or("-", yes_no);
        let fields = [
            ids.run_id.to_string(),
            ids.group.to_string(),
            ids.run.to_string(),
            ids.threads.to_string(),
            ids.case.to_string(),
            ids.dtype.to_string(),
            ids.trans.to_string(),
            ids.dims.m.to_string(),
            ids.dims.k.to_string(),
            ids.dims.n.to_string(),
            ids.beta.to_string(),
            self.attempt.to_string(),
            yes_no(self.valid).to_string(),
            self.variant.to_string(),
            self.role.word().to_string(),
            self.check.to_string(),
            self.reps.to_string(),
            self.rounds.to_string(),
            format!("{:.1}", fig.time.median),
            format!("{:.1}", fig.time.p25),
            format!("{:.1}", fig.time.p75),
            format!("{:.3}", flops / fig.time.median),
            format!("{:.4}", fig.ratio.median),
            format!("{:.4}", fig.ratio.p25),
            format!("{:.4}", fig.ratio.p75),
            format!("{:.4}", fig.ratio_of_medians),
            format!("{:+.4}", self.spread),
            worst,
            engaged.to_string(),
            self.detail.to_string(),
        ];
        data_line("AB", &fields)
    }
}

/// One `ABR` line: the time per call (ns) of every round, in round order.
#[derive(Debug, Clone, Copy)]
pub(super) struct AbrLine<'a> {
    pub(super) ids: CaseIds<'a>,
    pub(super) attempt: usize,
    pub(super) variant: &'a str,
    pub(super) reps: usize,
    pub(super) per_round: &'a [f64],
}

impl AbrLine<'_> {
    /// The line.
    pub(super) fn line(&self) -> String {
        let ids = &self.ids;
        let rounds: Vec<String> = self.per_round.iter().map(|t| format!("{t:.1}")).collect();
        let fields = [
            ids.run_id.to_string(),
            ids.group.to_string(),
            ids.run.to_string(),
            ids.case.to_string(),
            self.attempt.to_string(),
            self.variant.to_string(),
            self.reps.to_string(),
            rounds.join(","),
        ];
        data_line("ABR", &fields)
    }
}

/// The `verdicts` entry of one candidate:
/// `name:DECISION,geo=…,min_t=…,min_e=…,nt=…,ne=…,noop=…,partial=…`.
pub(super) fn verdict_entry(name: &str, v: &Verdict) -> String {
    let f = &v.figures;
    let figures = format!(
        "geo={:.4},min_t={:.4},min_e={:.4},nt={},ne={},noop={}",
        f.geomean, f.min_target, f.min_elsewhere, f.targets, f.elsewhere, f.noop
    );
    let (decision, partial) = (v.decision.word(), v.partial.text());
    format!("{name}:{decision},{figures},partial={partial}")
}

/// The verdict line of one group run.
#[derive(Debug, Clone)]
pub(super) struct AbvLine<'a> {
    pub(super) run_id: &'a str,
    pub(super) group: &'a str,
    pub(super) run: &'a str,
    pub(super) threads: usize,
    pub(super) tier: &'a str,
    pub(super) grade: &'a str,
    pub(super) cases: usize,
    pub(super) remeasured: usize,
    pub(super) invalid: usize,
    pub(super) max_abs_spread: f64,
    /// One entry per candidate ([`verdict_entry`]).
    pub(super) verdicts: Vec<String>,
}

impl AbvLine<'_> {
    /// The line.
    pub(super) fn line(&self) -> String {
        let fields = [
            self.run_id.to_string(),
            self.group.to_string(),
            self.run.to_string(),
            self.threads.to_string(),
            self.tier.to_string(),
            self.grade.to_string(),
            self.cases.to_string(),
            self.remeasured.to_string(),
            self.invalid.to_string(),
            format!("{:.4}", self.max_abs_spread),
            self.verdicts.join(";"),
        ];
        data_line("ABV", &fields)
    }
}

/// Where the lines go.
pub(super) trait Sink {
    /// Emits one line (without its line break).
    fn emit(&mut self, line: &str);
}

/// Standard output, and the `OXIBLAS_AB_OUT` file when set; each line goes
/// out with one write call per destination.
#[derive(Debug, Clone)]
pub(super) struct OutputSink {
    pub(super) out: Option<PathBuf>,
}

impl OutputSink {
    /// Ends the line libtest left open (`test <name> ... `) so that every
    /// harness line starts at column 0 (standard output only).
    pub(super) fn begin(&self) {
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(b"\n").and_then(|()| stdout.flush());
    }
}

/// Appends `text` to the file at `path` (created if missing) with one write
/// call.
pub(super) fn append_text(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    let mut file = options.create(true).append(true).open(path)?;
    file.write_all(text.as_bytes())
}

impl Sink for OutputSink {
    fn emit(&mut self, line: &str) {
        let text = format!("{line}\n");
        {
            let mut stdout = std::io::stdout().lock();
            let written = stdout
                .write_all(text.as_bytes())
                .and_then(|()| stdout.flush());
            written.unwrap_or_else(|e| panic!("standard output: {e}"));
        }
        if let Some(path) = &self.out {
            let appended = append_text(path, &text);
            appended.unwrap_or_else(|e| panic!("OXIBLAS_AB_OUT={}: {e}", path.display()));
        }
    }
}

/// Collects the lines (tests).
#[derive(Debug, Clone, Default)]
pub(super) struct VecSink {
    pub(super) lines: Vec<String>,
}

impl Sink for VecSink {
    fn emit(&mut self, line: &str) {
        self.lines.push(line.to_string());
    }
}
