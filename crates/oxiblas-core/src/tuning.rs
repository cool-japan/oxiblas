//! Auto-tuning utilities for optimal block sizes and algorithm selection.
//!
//! This module provides runtime performance tuning capabilities to automatically
//! select optimal block sizes for matrix operations based on the target architecture
//! and matrix dimensions.

use crate::simd::{SimdLevel, detect_simd_level};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

/// Default block size for M dimension in GEMM operations.
pub const DEFAULT_BLOCK_M: usize = 64;

/// Default block size for N dimension in GEMM operations.
pub const DEFAULT_BLOCK_N: usize = 64;

/// Default block size for K dimension in GEMM operations.
pub const DEFAULT_BLOCK_K: usize = 256;

/// Default block size for Level 2 BLAS operations.
pub const DEFAULT_L2_BLOCK: usize = 256;

/// L1 cache size in bytes (32 KB typical for modern CPUs).
pub const L1_CACHE_SIZE: usize = 32 * 1024;

/// L2 cache size in bytes (256 KB typical for modern CPUs).
pub const L2_CACHE_SIZE: usize = 256 * 1024;

/// L3 cache size in bytes (8 MB typical for modern CPUs).
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;

/// Tuning configuration for matrix operations.
#[derive(Debug, Clone, Copy)]
pub struct TuningConfig {
    /// Block size for M dimension in GEMM.
    pub block_m: usize,
    /// Block size for N dimension in GEMM.
    pub block_n: usize,
    /// Block size for K dimension in GEMM.
    pub block_k: usize,
    /// Block size for Level 2 operations.
    pub l2_block: usize,
    /// SIMD level to use.
    pub simd_level: SimdLevel,
    /// Whether to use parallel execution.
    pub parallel: bool,
    /// Threshold for parallelization (matrix size).
    pub par_threshold: usize,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            block_m: DEFAULT_BLOCK_M,
            block_n: DEFAULT_BLOCK_N,
            block_k: DEFAULT_BLOCK_K,
            l2_block: DEFAULT_L2_BLOCK,
            simd_level: detect_simd_level(),
            parallel: false,
            par_threshold: 64 * 64,
        }
    }
}

impl TuningConfig {
    /// Creates a new tuning configuration with architecture-specific defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a tuning configuration optimized for the given matrix dimensions.
    ///
    /// This uses heuristics based on cache sizes and SIMD width to determine
    /// optimal block sizes.
    #[must_use]
    pub fn for_dimensions(m: usize, n: usize, _k: usize) -> Self {
        let mut config = Self::new();
        let simd_level = detect_simd_level();

        // Adjust block sizes based on matrix dimensions and SIMD level
        match simd_level {
            SimdLevel::Scalar => {
                config.block_m = 32;
                config.block_n = 32;
                config.block_k = 128;
            }
            SimdLevel::Simd128 => {
                config.block_m = 64;
                config.block_n = 64;
                config.block_k = 256;
            }
            SimdLevel::Simd256 => {
                config.block_m = 96;
                config.block_n = 96;
                config.block_k = 384;
            }
            SimdLevel::Simd512 => {
                config.block_m = 128;
                config.block_n = 128;
                config.block_k = 512;
            }
        }

        // For small matrices, reduce block sizes
        if m < 128 || n < 128 {
            config.block_m = config.block_m.min(m);
            config.block_n = config.block_n.min(n);
        }

        // Adjust K blocking to fit in L2 cache
        let element_size = 8; // Assume f64 for sizing
        let panel_size = config.block_m * config.block_k * element_size;
        if panel_size > L2_CACHE_SIZE / 2 {
            config.block_k = (L2_CACHE_SIZE / 2) / (config.block_m * element_size);
        }

        // Enable parallelization for large matrices
        config.parallel = m * n >= config.par_threshold;

        config
    }

    /// Returns the optimal block size for GEMV operations.
    #[must_use]
    pub fn gemv_block_size(&self) -> usize {
        match self.simd_level {
            SimdLevel::Scalar => 128,
            SimdLevel::Simd128 => 256,
            SimdLevel::Simd256 => 512,
            SimdLevel::Simd512 => 1024,
        }
    }

    /// Returns the optimal panel width for factorizations.
    #[must_use]
    pub fn factorization_panel_width(&self) -> usize {
        match self.simd_level {
            SimdLevel::Scalar => 16,
            SimdLevel::Simd128 => 32,
            SimdLevel::Simd256 => 48,
            SimdLevel::Simd512 => 64,
        }
    }
}

/// Global tuning cache to avoid repeated auto-tuning.
pub struct TuningCache {
    initialized: AtomicBool,
    block_m: AtomicUsize,
    block_n: AtomicUsize,
    block_k: AtomicUsize,
}

static TUNING_CACHE: TuningCache = TuningCache {
    initialized: AtomicBool::new(false),
    block_m: AtomicUsize::new(DEFAULT_BLOCK_M),
    block_n: AtomicUsize::new(DEFAULT_BLOCK_N),
    block_k: AtomicUsize::new(DEFAULT_BLOCK_K),
};

impl TuningCache {
    /// Gets the cached tuning configuration, initializing if needed.
    pub fn get() -> TuningConfig {
        if !TUNING_CACHE.initialized.load(Ordering::Relaxed) {
            let config = TuningConfig::new();
            TUNING_CACHE
                .block_m
                .store(config.block_m, Ordering::Relaxed);
            TUNING_CACHE
                .block_n
                .store(config.block_n, Ordering::Relaxed);
            TUNING_CACHE
                .block_k
                .store(config.block_k, Ordering::Relaxed);
            TUNING_CACHE.initialized.store(true, Ordering::Relaxed);
        }

        TuningConfig {
            block_m: TUNING_CACHE.block_m.load(Ordering::Relaxed),
            block_n: TUNING_CACHE.block_n.load(Ordering::Relaxed),
            block_k: TUNING_CACHE.block_k.load(Ordering::Relaxed),
            ..TuningConfig::new()
        }
    }

    /// Updates the cached configuration.
    pub fn set(config: &TuningConfig) {
        TUNING_CACHE
            .block_m
            .store(config.block_m, Ordering::Relaxed);
        TUNING_CACHE
            .block_n
            .store(config.block_n, Ordering::Relaxed);
        TUNING_CACHE
            .block_k
            .store(config.block_k, Ordering::Relaxed);
        TUNING_CACHE.initialized.store(true, Ordering::Relaxed);
    }
}

/// Dimension of the fixed, representative square matrices used to time
/// candidate block-size configurations in [`time_candidate`].
///
/// This is intentionally small and independent of the caller's actual
/// `(m, n, k)` so that `AutoTuner::tune_gemm` completes in bounded time
/// regardless of how large the real workload is -- it is a timing *probe*,
/// not the real computation. `96` keeps the working set (3 matrices of
/// `96 * 96 * 8` bytes each, ~216 KiB total) close to [`L2_CACHE_SIZE`] so
/// that different block sizes produce a genuine, measurable difference in
/// cache behavior.
#[cfg(feature = "std")]
const BENCH_DIM: usize = 96;

/// Number of untimed warm-up passes run before timing starts, to prime
/// caches and pay one-time setup costs outside the measurement window.
#[cfg(feature = "std")]
const WARMUP_ITERS: u32 = 1;

/// Number of timed passes per candidate. The minimum observed duration is
/// kept, which is the standard way to filter out scheduler/OS noise in
/// short micro-benchmarks.
#[cfg(feature = "std")]
const TIMED_ITERS: u32 = 3;

/// Percentage scale factors (relative to the heuristic baseline) used to
/// generate alternative block-size candidates for `tune_gemm` to measure.
#[cfg(feature = "std")]
const CANDIDATE_SCALE_FACTORS_PERCENT: [usize; 4] = [50, 75, 150, 200];

/// Naive cache-blocked triple-loop matrix multiply (`C = A * B`) used only
/// as a timing probe for the auto-tuner.
///
/// It deliberately avoids SIMD/parallelism so the *only* variable between
/// runs is the block size under test -- this isolates the effect of
/// blocking on cache behavior, which is exactly what [`time_candidate`]
/// needs to measure. It is not, and is not meant to be, a production GEMM
/// kernel (those live in `oxiblas-blas`, which depends on this crate, not
/// the other way around).
#[cfg(feature = "std")]
fn blocked_matmul(a: &[f64], b: &[f64], c: &mut [f64], dim: usize, config: &TuningConfig) {
    let block_m = config.block_m.clamp(1, dim);
    let block_n = config.block_n.clamp(1, dim);
    let block_k = config.block_k.clamp(1, dim);

    for elem in c.iter_mut() {
        *elem = 0.0;
    }

    let mut ii = 0;
    while ii < dim {
        let i_end = (ii + block_m).min(dim);
        let mut kk = 0;
        while kk < dim {
            let k_end = (kk + block_k).min(dim);
            let mut jj = 0;
            while jj < dim {
                let j_end = (jj + block_n).min(dim);
                for i in ii..i_end {
                    let a_row = i * dim;
                    let c_row = i * dim;
                    for k in kk..k_end {
                        let a_ik = a[a_row + k];
                        let b_row = k * dim;
                        for j in jj..j_end {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
                jj = j_end;
            }
            kk = k_end;
        }
        ii = i_end;
    }
}

/// Runs [`blocked_matmul`] with `config`'s block sizes over a fixed-size
/// representative problem and returns the best (minimum) wall-clock time
/// observed across [`TIMED_ITERS`] timed runs, after [`WARMUP_ITERS`]
/// untimed warm-up runs.
#[cfg(feature = "std")]
fn time_candidate(config: &TuningConfig) -> Duration {
    let dim = BENCH_DIM;
    let a = vec![1.0_f64; dim * dim];
    let b = vec![1.0_f64; dim * dim];
    let mut c = vec![0.0_f64; dim * dim];

    for _ in 0..WARMUP_ITERS {
        blocked_matmul(&a, &b, &mut c, dim, config);
    }

    let mut best = Duration::MAX;
    for _ in 0..TIMED_ITERS {
        let start = Instant::now();
        blocked_matmul(&a, &b, &mut c, dim, config);
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }

    // Keep the optimizer from proving `c` is dead and eliding the loops
    // above entirely.
    std::hint::black_box(&c);
    best
}

/// Generates the heuristic baseline (from [`TuningConfig::for_dimensions`])
/// plus a handful of scaled variations for [`AutoTuner::tune_gemm`] to
/// benchmark against each other. Candidates are deduplicated so time is
/// never spent timing the same configuration twice.
#[cfg(feature = "std")]
fn generate_candidates(m: usize, n: usize, k: usize) -> Vec<TuningConfig> {
    let baseline = TuningConfig::for_dimensions(m, n, k);
    let mut candidates = Vec::with_capacity(CANDIDATE_SCALE_FACTORS_PERCENT.len() + 1);
    candidates.push(baseline);

    for &percent in &CANDIDATE_SCALE_FACTORS_PERCENT {
        let scaled = TuningConfig {
            block_m: (baseline.block_m * percent / 100).max(8),
            block_n: (baseline.block_n * percent / 100).max(8),
            block_k: (baseline.block_k * percent / 100).max(8),
            ..baseline
        };
        let is_duplicate = candidates.iter().any(|existing: &TuningConfig| {
            existing.block_m == scaled.block_m
                && existing.block_n == scaled.block_n
                && existing.block_k == scaled.block_k
        });
        if !is_duplicate {
            candidates.push(scaled);
        }
    }

    candidates
}

/// Runtime auto-tuner for GEMM block sizes.
///
/// `tune_gemm` runs a small, bounded set of *real* timed micro-benchmarks
/// on the current machine (see [`generate_candidates`] and
/// [`time_candidate`]) and keeps whichever candidate measured fastest.
/// Only that genuinely-measured winner is written into the shared
/// [`TuningCache`], so other code reading the cache never observes a
/// value that was fabricated rather than measured.
///
/// On targets built without the `std` feature (`std::time::Instant` is
/// unavailable there), no timing can be performed; see `tune_gemm` for the
/// honest fallback used in that case.
pub struct AutoTuner {
    config: TuningConfig,
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoTuner {
    /// Creates a new auto-tuner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TuningCache::get(),
        }
    }

    /// Tunes for GEMM operations with the given dimensions.
    ///
    /// This runs real, timed micro-benchmarks (using `std::time::Instant`)
    /// comparing the heuristic baseline from
    /// [`TuningConfig::for_dimensions`] against a few scaled block-size
    /// variations on a small representative problem, and keeps whichever
    /// configuration measured fastest on this machine. Only that
    /// genuinely-measured winner is written into the shared
    /// [`TuningCache`].
    ///
    /// On targets built without the `std` feature, `std::time::Instant` is
    /// unavailable, so no measurement can be performed. In that case this
    /// falls back to the heuristic from [`TuningConfig::for_dimensions`]
    /// and deliberately does **not** write to [`TuningCache`], so an
    /// unmeasured value can never be presented as a measured one to other
    /// code that relies on the shared cache.
    pub fn tune_gemm(&mut self, m: usize, n: usize, k: usize) -> &TuningConfig {
        #[cfg(feature = "std")]
        {
            let candidates = generate_candidates(m, n, k);
            // `generate_candidates` always pushes the baseline first, so
            // this is never empty.
            let mut best_config = candidates[0];
            let mut best_time = time_candidate(&best_config);
            for candidate in candidates.into_iter().skip(1) {
                let elapsed = time_candidate(&candidate);
                if elapsed < best_time {
                    best_time = elapsed;
                    best_config = candidate;
                }
            }
            self.config = best_config;
            TuningCache::set(&self.config);
        }

        #[cfg(not(feature = "std"))]
        {
            // No `std::time::Instant` available on this target: we cannot
            // measure anything, so we honestly fall back to the heuristic
            // and skip `TuningCache::set` entirely rather than poisoning
            // the shared cache with an unmeasured value.
            self.config = TuningConfig::for_dimensions(m, n, k);
        }

        &self.config
    }

    /// Returns the current tuning configuration.
    #[must_use]
    pub const fn config(&self) -> &TuningConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TuningConfig::default();
        assert_eq!(config.block_m, DEFAULT_BLOCK_M);
        assert_eq!(config.block_n, DEFAULT_BLOCK_N);
        assert_eq!(config.block_k, DEFAULT_BLOCK_K);
    }

    #[test]
    fn test_dimension_based_tuning() {
        let config_small = TuningConfig::for_dimensions(32, 32, 32);
        let config_large = TuningConfig::for_dimensions(1024, 1024, 1024);

        // Small matrices should have smaller or equal block sizes
        assert!(config_small.block_m <= 32);
        assert!(config_small.block_n <= 32);

        // Large matrices should enable parallelization
        assert!(config_large.parallel);
    }

    #[test]
    fn test_tuning_cache() {
        let config = TuningConfig {
            block_m: 77,
            block_n: 88,
            block_k: 99,
            ..TuningConfig::default()
        };

        TuningCache::set(&config);
        let cached = TuningCache::get();

        assert_eq!(cached.block_m, 77);
        assert_eq!(cached.block_n, 88);
        assert_eq!(cached.block_k, 99);
    }

    #[test]
    fn test_auto_tuner() {
        let mut tuner = AutoTuner::new();
        let config = tuner.tune_gemm(512, 512, 512);

        // Should produce sensible block sizes
        assert!(config.block_m > 0);
        assert!(config.block_n > 0);
        assert!(config.block_k > 0);
        assert!(config.block_m <= 512);
        assert!(config.block_n <= 512);
    }

    // ------------------------------------------------------------------
    // Regression tests for the "AutoTuner runs no benchmarks / poisons the
    // cache" bug: `tune_gemm` must perform genuine, measurable timing and
    // must only ever write a genuinely-measured winner into the shared
    // `TuningCache`.
    // ------------------------------------------------------------------

    #[cfg(feature = "std")]
    #[test]
    fn test_generate_candidates_produces_multiple_distinct_configs() {
        let candidates = generate_candidates(512, 512, 512);

        // A real auto-tuner needs at least two distinct options to choose
        // between -- otherwise there is nothing to "tune".
        assert!(
            candidates.len() >= 2,
            "expected multiple candidates to benchmark, got {}",
            candidates.len()
        );

        let first = candidates[0];
        assert!(
            candidates
                .iter()
                .any(|c| c.block_m != first.block_m
                    || c.block_n != first.block_n
                    || c.block_k != first.block_k),
            "all generated candidates were identical; nothing would actually be tuned"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_time_candidate_measures_nonzero_duration() {
        let config = TuningConfig::default();
        let elapsed = time_candidate(&config);

        // A stub that never actually runs the benchmark workload would
        // return instantly (zero, or an unmeasured constant). Real work
        // over a 96x96x96 problem always takes a measurable, nonzero
        // amount of wall-clock time.
        assert!(
            elapsed.as_nanos() > 0,
            "expected a real, nonzero measured duration from time_candidate"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_auto_tuner_writes_only_measured_result_to_cache() {
        // Seed the cache with a sentinel that no candidate generated by
        // `generate_candidates` could ever naturally produce (candidates
        // are always >= 8), so we can detect whether `tune_gemm` truly
        // overwrote it with a real measurement rather than leaving stale
        // or fabricated data behind.
        let sentinel = TuningConfig {
            block_m: 1,
            block_n: 1,
            block_k: 1,
            ..TuningConfig::default()
        };
        TuningCache::set(&sentinel);

        let mut tuner = AutoTuner::new();
        let config = *tuner.tune_gemm(256, 256, 256);

        assert_ne!(
            (config.block_m, config.block_n, config.block_k),
            (1, 1, 1),
            "tune_gemm must overwrite the cache with a genuinely measured winner"
        );
        assert!(config.block_m > 0);
        assert!(config.block_n > 0);
        assert!(config.block_k > 0);

        // The winning configuration must actually be one of the timed
        // candidates, not an arbitrary/default value bolted on afterwards.
        let candidates = generate_candidates(256, 256, 256);
        assert!(
            candidates.iter().any(|c| c.block_m == config.block_m
                && c.block_n == config.block_n
                && c.block_k == config.block_k),
            "tune_gemm's result must be one of the benchmarked candidates"
        );

        // The shared cache must be kept in sync with the measured winner.
        let cached = TuningCache::get();
        assert_eq!(cached.block_m, config.block_m);
        assert_eq!(cached.block_n, config.block_n);
        assert_eq!(cached.block_k, config.block_k);
    }

    #[test]
    fn test_gemv_block_size() {
        let config = TuningConfig::default();
        let block_size = config.gemv_block_size();

        // Should be reasonable for vectorization
        assert!(block_size >= 128);
        assert!(block_size <= 1024);
    }

    #[test]
    fn test_factorization_panel_width() {
        let config = TuningConfig::default();
        let panel_width = config.factorization_panel_width();

        // Should be a reasonable panel width
        assert!(panel_width >= 16);
        assert!(panel_width <= 128);
    }
}
