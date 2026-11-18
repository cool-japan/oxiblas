//! The settings of a harness run, read from the environment.

use std::env::VarError;
use std::path::PathBuf;
use std::time::Duration;

/// Rounds when `OXIBLAS_AB_ROUNDS` is unset.
pub(super) const DEFAULT_ROUNDS: usize = 22;
/// Fewest rounds of a run (smaller requests are raised to it).
pub(super) const MIN_ROUNDS: usize = 4;
/// Fewest requested rounds of a decision-grade run.
pub(super) const DECISION_ROUNDS: usize = 21;
/// Least sample time (ms) when `OXIBLAS_AB_MIN_SAMPLE_MS` is unset, and the
/// least of a decision-grade run.
pub(super) const DEFAULT_MIN_SAMPLE_MS: f64 = 2.0;
/// Re-measurements of a case with a bad control when `OXIBLAS_AB_RETRIES`
/// is unset.
pub(super) const DEFAULT_RETRIES: usize = 2;
/// Most re-measurements a run may ask for.
pub(super) const MAX_RETRIES: usize = 10;

/// Whether a run may decide (`Decision`) or only shows that the harness
/// works (`Smoke`: its verdicts are `SMOKE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Grade {
    Decision,
    Smoke,
}

impl Grade {
    /// The word of the `grade` key of a `SET` line.
    pub(super) const fn word(self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::Smoke => "smoke",
        }
    }
}

/// The settings of a run (see the module documentation for the variables).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Settings {
    /// Rounds as requested (`OXIBLAS_AB_ROUNDS`, or the default).
    pub(super) rounds_requested: usize,
    /// Rounds run: the request rounded up to an even number, at least
    /// [`MIN_ROUNDS`].
    pub(super) rounds: usize,
    /// Least duration of one timed sample of the fastest variant.
    pub(super) min_sample: Duration,
    /// Thread counts of the parallel groups.
    pub(super) threads: Vec<usize>,
    /// Only the quick subset of each group.
    pub(super) quick: bool,
    /// How often a case whose control is outside a limit (median or
    /// quartiles) is measured again.
    pub(super) retries: usize,
    /// Free text recorded with every group run (no tab, no line break).
    pub(super) tag: String,
    /// A file every line is also appended to.
    pub(super) out: Option<PathBuf>,
}

/// `value` of the setting `key` parsed as `V`.
fn parse_setting<V: std::str::FromStr>(key: &str, value: &str) -> Result<V, String>
where
    V::Err: std::fmt::Display,
{
    let parsed = value.trim().parse();
    parsed.map_err(|e| format!("{key}={value:?}: {e}"))
}

/// One thread count of `OXIBLAS_AB_THREADS` (not 0).
fn thread_count(text: &str) -> Result<usize, String> {
    match parse_setting::<usize>("OXIBLAS_AB_THREADS", text)? {
        0 => Err("OXIBLAS_AB_THREADS: a thread count is 0".to_string()),
        count => Ok(count),
    }
}

/// `requested` rounds as run: even, at least [`MIN_ROUNDS`].
pub(super) const fn effective_rounds(requested: usize) -> usize {
    let even = requested.saturating_add(requested % 2);
    if even < MIN_ROUNDS { MIN_ROUNDS } else { even }
}

/// The value of the setting `key` as `lookup` gave it: `None` when it is
/// unset, empty or only white space; an error when it is not valid Unicode.
fn setting_value(key: &str, looked_up: Result<String, VarError>) -> Result<Option<String>, String> {
    match looked_up {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(VarError::NotUnicode(raw)) => Err(format!("{key}={raw:?}: not valid Unicode")),
    }
}

/// The variable `key` of the process environment.
fn process_var(key: &str) -> Result<String, VarError> {
    std::env::var(key)
}

impl Settings {
    /// Settings from `lookup` (an environment, as `std::env::var` reads it),
    /// with `cores` as the default thread count. A value that is empty or
    /// only white space counts as unset; a value that is not valid Unicode
    /// is an error like any malformed value.
    pub(super) fn from_lookup(
        lookup: impl Fn(&str) -> Result<String, VarError>,
        cores: usize,
    ) -> Result<Self, String> {
        let get = |key: &str| setting_value(key, lookup(key));
        let rounds_requested = match get("OXIBLAS_AB_ROUNDS")? {
            None => DEFAULT_ROUNDS,
            Some(v) => parse_setting::<usize>("OXIBLAS_AB_ROUNDS", &v)?,
        };
        let min_ms = match get("OXIBLAS_AB_MIN_SAMPLE_MS")? {
            None => DEFAULT_MIN_SAMPLE_MS,
            Some(v) => match parse_setting::<f64>("OXIBLAS_AB_MIN_SAMPLE_MS", &v)? {
                ms if ms.is_finite() && ms > 0.0 && ms <= 60_000.0 => ms,
                _ => return Err(format!("OXIBLAS_AB_MIN_SAMPLE_MS={v:?}: not in (0, 60000]")),
            },
        };
        let threads = match get("OXIBLAS_AB_THREADS")? {
            None => vec![cores.max(1)],
            Some(v) => v.split(',').map(thread_count).collect::<Result<_, _>>()?,
        };
        let retries = match get("OXIBLAS_AB_RETRIES")? {
            None => DEFAULT_RETRIES,
            Some(v) => match parse_setting::<usize>("OXIBLAS_AB_RETRIES", &v)? {
                r if r <= MAX_RETRIES => r,
                _ => return Err(format!("OXIBLAS_AB_RETRIES={v:?}: above {MAX_RETRIES}")),
            },
        };
        let tag = get("OXIBLAS_AB_TAG")?.unwrap_or_default();
        if tag.contains(['\t', '\n', '\r']) {
            return Err(format!("OXIBLAS_AB_TAG={tag:?}: a tab or line break"));
        }
        Ok(Self {
            rounds_requested,
            rounds: effective_rounds(rounds_requested),
            min_sample: Duration::from_secs_f64(min_ms / 1000.0),
            threads,
            quick: get("OXIBLAS_AB_QUICK")?.is_some(),
            retries,
            tag: tag.trim().to_string(),
            out: get("OXIBLAS_AB_OUT")?.map(PathBuf::from),
        })
    }

    /// Settings from the process environment; a malformed value fails the run.
    pub(super) fn from_env() -> Self {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        let settings = Self::from_lookup(process_var, cores);
        settings.unwrap_or_else(|e| panic!("A/B harness settings: {e}"))
    }

    /// The grade of a run with these settings in a build with (`true`) or
    /// without debug assertions: `Smoke` with fewer than 21 rounds requested,
    /// samples shorter than 2 ms, the quick subset or a debug build.
    pub(super) fn grade(&self, debug_build: bool) -> Grade {
        let short = self.min_sample.as_secs_f64() * 1e3 < DEFAULT_MIN_SAMPLE_MS;
        let few = self.rounds_requested < DECISION_ROUNDS;
        if few || short || self.quick || debug_build {
            Grade::Smoke
        } else {
            Grade::Decision
        }
    }

    /// The least sample time in milliseconds.
    pub(super) fn min_sample_ms(&self) -> f64 {
        self.min_sample.as_secs_f64() * 1e3
    }
}
