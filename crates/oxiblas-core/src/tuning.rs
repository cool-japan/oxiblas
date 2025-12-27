//! Auto-tuning utilities for optimal block sizes and algorithm selection.
//!
//! This module provides runtime performance tuning capabilities to automatically
//! select optimal block sizes for matrix operations based on the target architecture
//! and matrix dimensions.

use crate::simd::{SimdLevel, detect_simd_level};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

/// Micro-benchmark based auto-tuner (simplified version).
///
/// In a production implementation, this would run actual benchmarks.
/// For now, it uses heuristics based on architecture and cache sizes.
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
    /// This is a simplified version that uses heuristics. A full implementation
    /// would run micro-benchmarks to find optimal parameters.
    pub fn tune_gemm(&mut self, m: usize, n: usize, k: usize) -> &TuningConfig {
        self.config = TuningConfig::for_dimensions(m, n, k);
        TuningCache::set(&self.config);
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
