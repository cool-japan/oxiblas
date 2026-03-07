//! Performance baseline storage and regression detection.
//!
//! Provides types and utilities for capturing, persisting, loading, and comparing
//! performance baselines so that CI can flag throughput regressions automatically.

use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Baseline performance data for an entire benchmark run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerfBaseline {
    /// OxiBLAS crate version (e.g. `"0.2.0"`).
    pub version: String,
    /// Target triple / platform string (e.g. `"aarch64-apple-darwin"`).
    pub platform: String,
    /// ISO 8601 timestamp of the run (e.g. `"2026-03-06T12:00:00Z"`).
    pub timestamp: String,
    /// Individual operation measurements.
    pub measurements: Vec<PerfMeasurement>,
}

/// A single measured operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerfMeasurement {
    /// Operation identifier (e.g. `"gemm_f64_256x256"`).
    pub name: String,
    /// Arithmetic mean wall-clock time in nanoseconds.
    pub mean_ns: f64,
    /// Sample standard deviation in nanoseconds.
    pub std_dev_ns: f64,
    /// Sustained throughput in GFLOP/s.
    pub throughput_gflops: f64,
    /// Leading dimension of the (square) matrix used.
    pub matrix_size: usize,
    /// Element data type: `"f64"` or `"f32"`.
    pub dtype: String,
}

// ---------------------------------------------------------------------------
// Regression report
// ---------------------------------------------------------------------------

/// A detected performance regression for one operation.
#[derive(Debug, Clone)]
pub struct Regression {
    /// Operation name matching [`PerfMeasurement::name`].
    pub name: String,
    /// GFLOP/s achieved in the baseline run.
    pub baseline_gflops: f64,
    /// GFLOP/s achieved in the current run.
    pub current_gflops: f64,
    /// Percentage slowdown relative to the baseline (positive = slower).
    /// Formula: `(baseline - current) / baseline * 100`.
    pub degradation_pct: f64,
}

impl fmt::Display for Regression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.3} -> {:.3} GFLOP/s  ({:.1}% regression)",
            self.name, self.baseline_gflops, self.current_gflops, self.degradation_pct
        )
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when persisting or loading baselines.
#[derive(Debug)]
pub enum RegressionError {
    /// An I/O error while reading or writing a file.
    Io(std::io::Error),
    /// A JSON serialisation / deserialisation error.
    Json(serde_json::Error),
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegressionError::Io(e) => write!(f, "I/O error: {e}"),
            RegressionError::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for RegressionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RegressionError::Io(e) => Some(e),
            RegressionError::Json(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for RegressionError {
    fn from(e: std::io::Error) -> Self {
        RegressionError::Io(e)
    }
}

impl From<serde_json::Error> for RegressionError {
    fn from(e: serde_json::Error) -> Self {
        RegressionError::Json(e)
    }
}

// ---------------------------------------------------------------------------
// Checker
// ---------------------------------------------------------------------------

/// Compares two baselines and surfaces operations whose throughput dropped
/// beyond a configurable threshold.
pub struct RegressionChecker {
    /// Minimum percentage slowdown that counts as a regression (default `5.0`).
    pub threshold_pct: f64,
}

impl Default for RegressionChecker {
    fn default() -> Self {
        Self { threshold_pct: 5.0 }
    }
}

impl RegressionChecker {
    /// Create a checker with a custom threshold percentage.
    pub fn new(threshold_pct: f64) -> Self {
        Self { threshold_pct }
    }

    /// Compare `current` against `baseline`.
    ///
    /// Returns every operation whose throughput fell by more than
    /// [`Self::threshold_pct`] percent.  Operations that appear only in one
    /// baseline are silently ignored (no panic).
    pub fn check(&self, baseline: &PerfBaseline, current: &PerfBaseline) -> Vec<Regression> {
        let mut regressions = Vec::new();

        for base_m in &baseline.measurements {
            // Find the matching measurement in current by name.
            let Some(cur_m) = current.measurements.iter().find(|m| m.name == base_m.name) else {
                continue;
            };

            // Avoid division-by-zero when baseline was 0 GFLOP/s.
            if base_m.throughput_gflops <= 0.0 {
                continue;
            }

            let degradation_pct = (base_m.throughput_gflops - cur_m.throughput_gflops)
                / base_m.throughput_gflops
                * 100.0;

            if degradation_pct > self.threshold_pct {
                regressions.push(Regression {
                    name: base_m.name.clone(),
                    baseline_gflops: base_m.throughput_gflops,
                    current_gflops: cur_m.throughput_gflops,
                    degradation_pct,
                });
            }
        }

        regressions
    }

    /// Serialise `baseline` as pretty-printed JSON and write it to `path`.
    pub fn save_baseline(baseline: &PerfBaseline, path: &Path) -> Result<(), RegressionError> {
        let json = serde_json::to_string_pretty(baseline)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Read and deserialise a baseline from a JSON file at `path`.
    pub fn load_baseline(path: &Path) -> Result<PerfBaseline, RegressionError> {
        let bytes = std::fs::read(path)?;
        let baseline = serde_json::from_slice(&bytes)?;
        Ok(baseline)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_measurement(name: &str, gflops: f64, size: usize, dtype: &str) -> PerfMeasurement {
        PerfMeasurement {
            name: name.to_owned(),
            mean_ns: 1_000_000.0,
            std_dev_ns: 50_000.0,
            throughput_gflops: gflops,
            matrix_size: size,
            dtype: dtype.to_owned(),
        }
    }

    fn make_baseline(version: &str, measurements: Vec<PerfMeasurement>) -> PerfBaseline {
        PerfBaseline {
            version: version.to_owned(),
            platform: "aarch64-apple-darwin".to_owned(),
            timestamp: "2026-03-06T00:00:00Z".to_owned(),
            measurements,
        }
    }

    // ------------------------------------------------------------------
    // 1. Roundtrip serialisation through JSON bytes
    // ------------------------------------------------------------------
    #[test]
    fn test_serialization_roundtrip_json_bytes() {
        let baseline = make_baseline(
            "0.2.0",
            vec![make_measurement("gemm_f64_256x256", 42.0, 256, "f64")],
        );

        let json = serde_json::to_string(&baseline).expect("serialise");
        let loaded: PerfBaseline = serde_json::from_str(&json).expect("deserialise");

        assert_eq!(loaded.version, baseline.version);
        assert_eq!(loaded.platform, baseline.platform);
        assert_eq!(loaded.measurements.len(), 1);
        assert!((loaded.measurements[0].throughput_gflops - 42.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // 2. Roundtrip serialisation through a temp file
    // ------------------------------------------------------------------
    #[test]
    fn test_serialization_roundtrip_file() {
        let baseline = make_baseline(
            "0.2.0",
            vec![
                make_measurement("gemm_f64_128x128", 20.0, 128, "f64"),
                make_measurement("gemm_f32_128x128", 38.0, 128, "f32"),
            ],
        );

        let path = std::env::temp_dir().join("oxiblas_regression_test_roundtrip.json");
        RegressionChecker::save_baseline(&baseline, &path).expect("save");
        let loaded = RegressionChecker::load_baseline(&path).expect("load");

        assert_eq!(loaded.measurements.len(), 2);
        assert_eq!(loaded.measurements[0].name, "gemm_f64_128x128");
        assert_eq!(loaded.measurements[1].name, "gemm_f32_128x128");

        // Clean up temp file
        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // 3. Regression is detected when current is 10 % slower
    // ------------------------------------------------------------------
    #[test]
    fn test_regression_detected_at_10pct() {
        let baseline = make_baseline(
            "0.1.0",
            vec![make_measurement("gemm_f64_256x256", 100.0, 256, "f64")],
        );
        let current = make_baseline(
            "0.2.0",
            vec![make_measurement("gemm_f64_256x256", 90.0, 256, "f64")],
        );

        let checker = RegressionChecker::default(); // threshold = 5 %
        let regressions = checker.check(&baseline, &current);

        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].name, "gemm_f64_256x256");
        assert!((regressions[0].degradation_pct - 10.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // 4. No regression when current equals baseline
    // ------------------------------------------------------------------
    #[test]
    fn test_no_regression_same_baselines() {
        let baseline = make_baseline(
            "0.1.0",
            vec![make_measurement("gemm_f64_256x256", 100.0, 256, "f64")],
        );
        let current = baseline.clone();

        let checker = RegressionChecker::default();
        let regressions = checker.check(&baseline, &current);

        assert!(regressions.is_empty());
    }

    // ------------------------------------------------------------------
    // 5. No regression when current is faster
    // ------------------------------------------------------------------
    #[test]
    fn test_no_regression_current_faster() {
        let baseline = make_baseline(
            "0.1.0",
            vec![make_measurement("gemm_f64_256x256", 80.0, 256, "f64")],
        );
        let current = make_baseline(
            "0.2.0",
            vec![make_measurement("gemm_f64_256x256", 100.0, 256, "f64")],
        );

        let checker = RegressionChecker::default();
        let regressions = checker.check(&baseline, &current);

        assert!(regressions.is_empty());
    }

    // ------------------------------------------------------------------
    // 6. Custom threshold: a 3 % drop triggers regression at threshold=2
    // ------------------------------------------------------------------
    #[test]
    fn test_custom_threshold_triggers() {
        let baseline = make_baseline(
            "0.1.0",
            vec![make_measurement("gemm_f64_256x256", 100.0, 256, "f64")],
        );
        let current = make_baseline(
            "0.2.0",
            vec![make_measurement("gemm_f64_256x256", 97.0, 256, "f64")],
        );

        let checker = RegressionChecker::new(2.0); // 2 % threshold
        let regressions = checker.check(&baseline, &current);

        assert_eq!(regressions.len(), 1);
        assert!((regressions[0].degradation_pct - 3.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // 7. Custom threshold: a 3 % drop does NOT trigger at threshold=5
    // ------------------------------------------------------------------
    #[test]
    fn test_custom_threshold_no_trigger() {
        let baseline = make_baseline(
            "0.1.0",
            vec![make_measurement("gemm_f64_256x256", 100.0, 256, "f64")],
        );
        let current = make_baseline(
            "0.2.0",
            vec![make_measurement("gemm_f64_256x256", 97.0, 256, "f64")],
        );

        let checker = RegressionChecker::new(5.0);
        let regressions = checker.check(&baseline, &current);

        assert!(regressions.is_empty());
    }

    // ------------------------------------------------------------------
    // 8. Operations missing from current are silently ignored
    // ------------------------------------------------------------------
    #[test]
    fn test_missing_operation_in_current_ignored() {
        let baseline = make_baseline(
            "0.1.0",
            vec![
                make_measurement("gemm_f64_256x256", 100.0, 256, "f64"),
                make_measurement("cholesky_f64_256x256", 50.0, 256, "f64"),
            ],
        );
        let current = make_baseline(
            "0.2.0",
            // only one of the two operations
            vec![make_measurement("gemm_f64_256x256", 90.0, 256, "f64")],
        );

        let checker = RegressionChecker::default();
        let regressions = checker.check(&baseline, &current);

        // Only gemm shows up as a regression; cholesky is silently skipped
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].name, "gemm_f64_256x256");
    }

    // ------------------------------------------------------------------
    // 9. Zero-GFLOP/s baseline does not panic
    // ------------------------------------------------------------------
    #[test]
    fn test_zero_baseline_gflops_no_panic() {
        let baseline = make_baseline(
            "0.1.0",
            vec![make_measurement("gemm_f64_1x1", 0.0, 1, "f64")],
        );
        let current = make_baseline(
            "0.2.0",
            vec![make_measurement("gemm_f64_1x1", 0.0, 1, "f64")],
        );

        let checker = RegressionChecker::default();
        let regressions = checker.check(&baseline, &current);

        // A zero baseline is skipped entirely
        assert!(regressions.is_empty());
    }

    // ------------------------------------------------------------------
    // 10. Multiple simultaneous regressions are all captured
    // ------------------------------------------------------------------
    #[test]
    fn test_multiple_regressions_all_captured() {
        let baseline = make_baseline(
            "0.1.0",
            vec![
                make_measurement("gemm_f64_256x256", 100.0, 256, "f64"),
                make_measurement("gemm_f32_256x256", 180.0, 256, "f32"),
                make_measurement("cholesky_f64_256x256", 60.0, 256, "f64"),
            ],
        );
        let current = make_baseline(
            "0.2.0",
            vec![
                make_measurement("gemm_f64_256x256", 80.0, 256, "f64"), // -20%
                make_measurement("gemm_f32_256x256", 160.0, 256, "f32"), // -11%
                make_measurement("cholesky_f64_256x256", 59.0, 256, "f64"), // ~-1.7% (ok)
            ],
        );

        let checker = RegressionChecker::default(); // 5 % threshold
        let regressions = checker.check(&baseline, &current);

        assert_eq!(regressions.len(), 2);
        let names: Vec<&str> = regressions.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"gemm_f64_256x256"));
        assert!(names.contains(&"gemm_f32_256x256"));
    }

    // ------------------------------------------------------------------
    // 11. Regression Display formatting is non-empty and contains name
    // ------------------------------------------------------------------
    #[test]
    fn test_regression_display_contains_name() {
        let r = Regression {
            name: "gemm_f64_512x512".to_owned(),
            baseline_gflops: 200.0,
            current_gflops: 170.0,
            degradation_pct: 15.0,
        };
        let s = format!("{r}");
        assert!(s.contains("gemm_f64_512x512"));
        assert!(s.contains("15.0%") || s.contains("15.0 %") || s.contains("regression"));
    }
}
