//! Lightweight, criterion-free performance measurement utilities.
//!
//! Each function runs a fixed protocol:
//!   * 5 warm-up iterations (not measured)
//!   * 20 measurement iterations
//!
//! The reported `mean_ns` / `std_dev_ns` are computed from those 20 samples.
//! GFLOP/s is derived from the standard floating-point operation counts for
//! each algorithm:
//!
//! | Operation        | FLOPs                      |
//! |------------------|----------------------------|
//! | GEMM n×n         | 2 n³                       |
//! | Cholesky n×n     | n³ / 3                     |

use crate::regression::{PerfBaseline, PerfMeasurement};
use oxiblas_blas::level3::gemm;
use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_matrix::Mat;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const WARMUP_ITERS: usize = 5;
const MEASURE_ITERS: usize = 20;

/// Run `f` for `WARMUP_ITERS` warm-up rounds then collect `MEASURE_ITERS`
/// wall-clock durations (nanoseconds) and return them as a Vec.
fn timed_samples<F>(mut f: F) -> Vec<f64>
where
    F: FnMut(),
{
    for _ in 0..WARMUP_ITERS {
        f();
    }

    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    samples
}

/// Compute mean and sample standard deviation from a slice of values.
fn mean_stddev(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance.sqrt())
}

// ---------------------------------------------------------------------------
// Public measurement functions
// ---------------------------------------------------------------------------

/// Time an n×n double-precision general matrix multiplication (DGEMM).
///
/// FLOPs = 2 n³ (multiply-add pairs for C = A B).
pub fn quick_gemm_f64(n: usize) -> PerfMeasurement {
    let a_data: Vec<f64> = (0..n * n).map(|i| (i % 97) as f64 * 0.01).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| (i % 89) as f64 * 0.01).collect();
    let a = Mat::from_slice(n, n, &a_data);
    let b = Mat::from_slice(n, n, &b_data);
    let mut c: Mat<f64> = Mat::zeros(n, n);

    let samples = timed_samples(|| {
        gemm(
            black_box(1.0f64),
            black_box(a.as_ref()),
            black_box(b.as_ref()),
            black_box(0.0f64),
            black_box(c.as_mut()),
        );
    });

    let (mean_ns, std_dev_ns) = mean_stddev(&samples);
    let flops = 2.0 * (n as f64).powi(3);
    let throughput_gflops = flops / mean_ns; // ns * 1e-9 s, GFlops = flops/1e9 / s => flops/ns

    PerfMeasurement {
        name: format!("gemm_f64_{n}x{n}"),
        mean_ns,
        std_dev_ns,
        throughput_gflops,
        matrix_size: n,
        dtype: "f64".to_owned(),
    }
}

/// Time an n×n single-precision general matrix multiplication (SGEMM).
///
/// FLOPs = 2 n³.
pub fn quick_gemm_f32(n: usize) -> PerfMeasurement {
    let a_data: Vec<f32> = (0..n * n).map(|i| (i % 97) as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| (i % 89) as f32 * 0.01).collect();
    let a = Mat::from_slice(n, n, &a_data);
    let b = Mat::from_slice(n, n, &b_data);
    let mut c: Mat<f32> = Mat::zeros(n, n);

    let samples = timed_samples(|| {
        gemm(
            black_box(1.0f32),
            black_box(a.as_ref()),
            black_box(b.as_ref()),
            black_box(0.0f32),
            black_box(c.as_mut()),
        );
    });

    let (mean_ns, std_dev_ns) = mean_stddev(&samples);
    let flops = 2.0 * (n as f64).powi(3);
    let throughput_gflops = flops / mean_ns;

    PerfMeasurement {
        name: format!("gemm_f32_{n}x{n}"),
        mean_ns,
        std_dev_ns,
        throughput_gflops,
        matrix_size: n,
        dtype: "f32".to_owned(),
    }
}

/// Time an n×n double-precision Cholesky decomposition.
///
/// FLOPs ≈ n³ / 3.
pub fn quick_cholesky(n: usize) -> PerfMeasurement {
    // Build a symmetric positive-definite matrix: A = D + n*I so all
    // eigenvalues are positive.
    let mut a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let val = ((i * 17 + j * 31) % 100) as f64 * 0.01;
            a[(i, j)] = val;
            a[(j, i)] = val;
        }
        // Strong diagonal dominance ensures positive definiteness.
        a[(i, i)] += n as f64 + 1.0;
    }

    let samples = timed_samples(|| {
        let _ = Cholesky::compute(black_box(a.as_ref()));
    });

    let (mean_ns, std_dev_ns) = mean_stddev(&samples);
    let flops = (n as f64).powi(3) / 3.0;
    let throughput_gflops = flops / mean_ns;

    PerfMeasurement {
        name: format!("cholesky_f64_{n}x{n}"),
        mean_ns,
        std_dev_ns,
        throughput_gflops,
        matrix_size: n,
        dtype: "f64".to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Full suite
// ---------------------------------------------------------------------------

/// Run the complete quick benchmark suite and return a [`PerfBaseline`].
///
/// Sizes chosen to be fast enough for CI (all complete in well under a minute
/// on any modern machine):
///
/// | Op        | Sizes     |
/// |-----------|-----------|
/// | GEMM f64  | 64, 128   |
/// | GEMM f32  | 64, 128   |
/// | Cholesky  | 64, 128   |
pub fn benchmark_suite() -> PerfBaseline {
    let platform = std::env::consts::ARCH.to_owned() + "-" + std::env::consts::OS;

    let mut measurements = Vec::new();

    for &n in &[64_usize, 128] {
        measurements.push(quick_gemm_f64(n));
        measurements.push(quick_gemm_f32(n));
        measurements.push(quick_cholesky(n));
    }

    PerfBaseline {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        platform,
        timestamp: chrono_like_timestamp(),
        measurements,
    }
}

/// Produce an ISO-8601-ish timestamp without pulling in `chrono`.
///
/// Uses [`std::time::SystemTime`] and formats a UTC-offset string.
fn chrono_like_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Manual conversion: seconds-since-epoch -> YYYY-MM-DDTHH:MM:SSZ
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hh = time_of_day / 3600;
    let mm = (time_of_day % 3600) / 60;
    let ss = time_of_day % 60;

    // Gregorian calendar reconstruction from days-since-epoch.
    let (year, month, day) = days_to_ymd(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u32, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    // (civil_from_days, shifted to epoch 1970-01-01)
    let z = days as i64 + 719_468;
    let era: i64 = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month prime [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    // Suppress the unused_mut warning - days is consumed above via z.
    let _ = &mut days;

    (y as u32, m as u32, d as u32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. quick_gemm_f64 returns a measurement with the right dtype/name
    // ------------------------------------------------------------------
    #[test]
    fn test_quick_gemm_f64_basic() {
        let m = quick_gemm_f64(32);
        assert_eq!(m.dtype, "f64");
        assert_eq!(m.matrix_size, 32);
        assert!(m.name.contains("gemm_f64_32x32"));
    }

    // ------------------------------------------------------------------
    // 2. quick_gemm_f64 reports positive timing and GFLOP/s
    // ------------------------------------------------------------------
    #[test]
    fn test_quick_gemm_f64_positive_values() {
        let m = quick_gemm_f64(32);
        assert!(m.mean_ns > 0.0, "mean_ns must be positive");
        assert!(m.std_dev_ns >= 0.0, "std_dev_ns must be non-negative");
        assert!(m.throughput_gflops > 0.0, "throughput must be positive");
    }

    // ------------------------------------------------------------------
    // 3. quick_gemm_f32 returns a measurement with the right dtype/name
    // ------------------------------------------------------------------
    #[test]
    fn test_quick_gemm_f32_basic() {
        let m = quick_gemm_f32(32);
        assert_eq!(m.dtype, "f32");
        assert_eq!(m.matrix_size, 32);
        assert!(m.name.contains("gemm_f32_32x32"));
    }

    // ------------------------------------------------------------------
    // 4. quick_gemm_f32 reports positive values
    // ------------------------------------------------------------------
    #[test]
    fn test_quick_gemm_f32_positive_values() {
        let m = quick_gemm_f32(32);
        assert!(m.mean_ns > 0.0);
        assert!(m.throughput_gflops > 0.0);
    }

    // ------------------------------------------------------------------
    // 5. quick_cholesky returns a measurement with the right dtype/name
    // ------------------------------------------------------------------
    #[test]
    fn test_quick_cholesky_basic() {
        let m = quick_cholesky(32);
        assert_eq!(m.dtype, "f64");
        assert_eq!(m.matrix_size, 32);
        assert!(m.name.contains("cholesky_f64_32x32"));
    }

    // ------------------------------------------------------------------
    // 6. quick_cholesky reports positive values
    // ------------------------------------------------------------------
    #[test]
    fn test_quick_cholesky_positive_values() {
        let m = quick_cholesky(32);
        assert!(m.mean_ns > 0.0);
        assert!(m.throughput_gflops > 0.0);
    }

    // ------------------------------------------------------------------
    // 7. benchmark_suite returns the expected number of measurements
    // ------------------------------------------------------------------
    #[test]
    fn test_benchmark_suite_measurement_count() {
        let suite = benchmark_suite();
        // 2 sizes × 3 ops = 6 measurements
        assert_eq!(suite.measurements.len(), 6);
    }

    // ------------------------------------------------------------------
    // 8. benchmark_suite returns non-empty version and platform strings
    // ------------------------------------------------------------------
    #[test]
    fn test_benchmark_suite_metadata() {
        let suite = benchmark_suite();
        assert!(!suite.version.is_empty());
        assert!(!suite.platform.is_empty());
        assert!(!suite.timestamp.is_empty());
    }

    // ------------------------------------------------------------------
    // 9. benchmark_suite results are all positive
    // ------------------------------------------------------------------
    #[test]
    fn test_benchmark_suite_all_positive() {
        let suite = benchmark_suite();
        for m in &suite.measurements {
            assert!(
                m.throughput_gflops > 0.0,
                "Expected positive GFLOP/s for {}",
                m.name
            );
            assert!(m.mean_ns > 0.0, "Expected positive mean_ns for {}", m.name);
        }
    }

    // ------------------------------------------------------------------
    // 10. benchmark_suite can be serialised and deserialised via JSON file
    // ------------------------------------------------------------------
    #[test]
    fn test_benchmark_suite_json_roundtrip() {
        use crate::regression::RegressionChecker;

        let suite = benchmark_suite();
        let path = std::env::temp_dir().join("oxiblas_quick_perf_suite_test.json");
        RegressionChecker::save_baseline(&suite, &path).expect("save");
        let loaded = RegressionChecker::load_baseline(&path).expect("load");

        assert_eq!(loaded.measurements.len(), suite.measurements.len());
        for (orig, loaded_m) in suite.measurements.iter().zip(loaded.measurements.iter()) {
            assert_eq!(orig.name, loaded_m.name);
            assert!((orig.throughput_gflops - loaded_m.throughput_gflops).abs() < 1e-9);
        }

        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------
    // 11. mean_stddev produces correct values for a known input
    // ------------------------------------------------------------------
    #[test]
    fn test_mean_stddev_known_values() {
        let samples = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
        let (mean, stddev) = mean_stddev(&samples);
        assert!((mean - 30.0).abs() < 1e-9, "mean should be 30");
        // population stddev = ~14.142, sample stddev = sqrt(250) = ~15.811
        assert!(
            (stddev - 15.811_388_3).abs() < 1e-4,
            "sample stddev mismatch: {stddev}"
        );
    }

    // ------------------------------------------------------------------
    // 12. timestamp string has expected length and format hint
    // ------------------------------------------------------------------
    #[test]
    fn test_timestamp_format() {
        let ts = super::chrono_like_timestamp();
        // "YYYY-MM-DDTHH:MM:SSZ" = 20 characters
        assert_eq!(ts.len(), 20, "Unexpected timestamp length: {ts}");
        assert!(ts.ends_with('Z'), "Timestamp should end with Z: {ts}");
        assert!(ts.contains('T'), "Timestamp should contain T: {ts}");
    }
}
