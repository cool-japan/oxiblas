//! Runtime auto-tuning infrastructure for GEMM block sizes.
//!
//! This module provides a `RuntimeAutoTuner` that benchmarks different block size
//! combinations (MC, KC, NC) at runtime and caches optimal parameters per operation
//! type, scalar type, and matrix size range.
//!
//! # Feature Gate
//!
//! This module is available when the `runtime-tuning` feature is enabled.
//! Without it, the existing heuristic-based auto-tuning in [`super::autotune`]
//! is used as the fallback.
//!
//! # Thread Safety
//!
//! The tuning cache uses `std::sync::OnceLock` for lock-free, thread-safe
//! initialization. Once a tuning result is computed for a given key, it is
//! stored permanently and never recomputed.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxiblas_blas::level3::runtime_autotune::{RuntimeAutoTuner, auto_tune_gemm};
//! use oxiblas_blas::level3::GemmBlocking;
//!
//! // Automatic tuning for f64 GEMM
//! let blocking = auto_tune_gemm::<f64>();
//! println!("Optimal: MC={}, KC={}, NC={}", blocking.mc, blocking.kc, blocking.nc);
//! ```

use crate::level3::gemm::{GemmBlocking, gemm_with_blocking};
use crate::level3::gemm_kernel::{GemmKernel, MicroKernelShape};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::Mat;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identifies a matrix size range for cache lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizeClass {
    /// Matrices up to 128 in all dimensions.
    Small,
    /// Matrices up to 512 in all dimensions.
    Medium,
    /// Matrices larger than 512 in any dimension.
    Large,
}

impl SizeClass {
    /// Classifies matrix dimensions into a size category.
    #[must_use]
    pub fn from_dims(m: usize, k: usize, n: usize) -> Self {
        let max_dim = m.max(k).max(n);
        if max_dim <= 128 {
            SizeClass::Small
        } else if max_dim <= 512 {
            SizeClass::Medium
        } else {
            SizeClass::Large
        }
    }

    /// Returns the representative sample dimension for benchmarking this class.
    #[must_use]
    pub const fn sample_dim(self) -> usize {
        match self {
            SizeClass::Small => 64,
            SizeClass::Medium => 256,
            SizeClass::Large => 512,
        }
    }
}

/// Identifies an operation type for the tuning cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// General matrix-matrix multiplication.
    Gemm,
    /// Triangular solve with multiple right-hand sides.
    Trsm,
    /// Symmetric rank-k update.
    Syrk,
}

/// Key used to look up cached tuning results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuningKey {
    /// The operation being tuned.
    pub op: OpKind,
    /// Size of one scalar element in bytes (4 for f32, 8 for f64, etc.).
    pub elem_size: usize,
    /// Matrix size range.
    pub size_class: SizeClass,
}

/// A single benchmark sample: blocking parameters and measured throughput.
#[derive(Debug, Clone, Copy)]
struct BenchSample {
    blocking: GemmBlocking,
    gflops: f64,
}

// ---------------------------------------------------------------------------
// Candidate generation
// ---------------------------------------------------------------------------

/// Generates a set of candidate block-size combinations to benchmark.
///
/// The candidates are centered around the heuristic defaults and explore
/// a small neighborhood. This keeps the tuning time bounded.
fn generate_candidates(
    elem_size: usize,
    shape: &MicroKernelShape,
    size_class: SizeClass,
) -> Vec<GemmBlocking> {
    let mr = shape.mr;
    let nr = shape.nr;
    let dim = size_class.sample_dim();

    // Heuristic baseline (from existing autotune)
    let base = GemmBlocking::auto_tuned_raw(dim, dim, dim, elem_size, shape);

    let mut candidates = Vec::with_capacity(16);
    candidates.push(base);

    // Explore MC variations (0.5x, 0.75x, 1.25x, 1.5x of base)
    for &factor_num in &[50_usize, 75, 125, 150] {
        let mc_raw = base.mc * factor_num / 100;
        let mc = align_down(mc_raw, mr).max(mr);
        if mc != base.mc && mc <= dim * 2 {
            candidates.push(GemmBlocking {
                mc,
                kc: base.kc,
                nc: base.nc,
            });
        }
    }

    // Explore KC variations
    for &factor_num in &[50_usize, 75, 125, 150] {
        let kc = (base.kc * factor_num / 100).max(32);
        if kc != base.kc {
            candidates.push(GemmBlocking {
                mc: base.mc,
                kc,
                nc: base.nc,
            });
        }
    }

    // Explore NC variations
    for &factor_num in &[50_usize, 75, 125, 150] {
        let nc_raw = base.nc * factor_num / 100;
        let nc = align_down(nc_raw, nr).max(nr);
        if nc != base.nc && nc <= dim * 8 {
            candidates.push(GemmBlocking {
                mc: base.mc,
                kc: base.kc,
                nc,
            });
        }
    }

    // Add the static defaults as a candidate too
    let static_default = GemmBlocking::for_kernel_with_elem_size(elem_size, shape);
    if !candidates.iter().any(|c| {
        c.mc == static_default.mc && c.kc == static_default.kc && c.nc == static_default.nc
    }) {
        candidates.push(static_default);
    }

    candidates
}

/// Aligns `n` down to a multiple of `align`.
#[inline]
const fn align_down(n: usize, align: usize) -> usize {
    if align == 0 {
        return n;
    }
    (n / align) * align
}

// ---------------------------------------------------------------------------
// Benchmarking
// ---------------------------------------------------------------------------

/// Measures the throughput of GEMM with a given blocking configuration.
///
/// Uses a small sample matrix and runs several iterations to get a stable
/// measurement. Returns GFLOP/s.
fn benchmark_blocking<T: Field + GemmKernel + bytemuck::Zeroable>(
    blocking: &GemmBlocking,
    dim: usize,
) -> f64 {
    let a: Mat<T> = Mat::filled(dim, dim, T::one());
    let b: Mat<T> = Mat::filled(dim, dim, T::one());
    let mut c: Mat<T> = Mat::zeros(dim, dim);

    // Warm-up run
    gemm_with_blocking(
        T::one(),
        a.as_ref(),
        b.as_ref(),
        T::zero(),
        c.as_mut(),
        Par::Seq,
        blocking,
    );

    // Timed runs
    let num_iters = 3_u32;
    let start = Instant::now();
    for _ in 0..num_iters {
        gemm_with_blocking(
            T::one(),
            a.as_ref(),
            b.as_ref(),
            T::zero(),
            c.as_mut(),
            Par::Seq,
            blocking,
        );
    }
    let elapsed = start.elapsed().as_secs_f64();

    // 2 * M * N * K floating-point operations per GEMM
    let flops_per_iter = 2.0 * (dim as f64) * (dim as f64) * (dim as f64);
    let total_flops = flops_per_iter * f64::from(num_iters);
    total_flops / elapsed / 1e9 // GFLOP/s
}

// ---------------------------------------------------------------------------
// RuntimeAutoTuner
// ---------------------------------------------------------------------------

/// Runtime auto-tuner that benchmarks block-size combinations and caches results.
///
/// The tuner runs a small set of micro-benchmarks for each (operation, scalar,
/// size-class) combination the first time it is requested, then stores the
/// winner in a thread-safe cache.
pub struct RuntimeAutoTuner {
    /// Cached results keyed by (op, elem_size, size_class).
    cache: Mutex<HashMap<TuningKey, GemmBlocking>>,
}

impl Default for RuntimeAutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeAutoTuner {
    /// Creates a new, empty auto-tuner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Looks up a cached result, returning `None` if the key has not been tuned.
    #[must_use]
    pub fn get(&self, key: &TuningKey) -> Option<GemmBlocking> {
        self.cache
            .lock()
            .ok()
            .and_then(|guard| guard.get(key).copied())
    }

    /// Stores a tuning result in the cache.
    pub fn insert(&self, key: TuningKey, blocking: GemmBlocking) {
        if let Ok(mut guard) = self.cache.lock() {
            guard.insert(key, blocking);
        }
    }

    /// Tunes GEMM for the given scalar type and size class.
    ///
    /// If a cached result exists, it is returned immediately.
    /// Otherwise, a set of candidates is benchmarked and the best one stored.
    pub fn tune_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
        &self,
        size_class: SizeClass,
    ) -> GemmBlocking {
        let key = TuningKey {
            op: OpKind::Gemm,
            elem_size: core::mem::size_of::<T>(),
            size_class,
        };

        // Fast path: already cached
        if let Some(cached) = self.get(&key) {
            return cached;
        }

        // Benchmark candidates
        let shape = T::micro_kernel_shape();
        let candidates = generate_candidates(key.elem_size, &shape, size_class);
        let dim = size_class.sample_dim();

        let mut best: Option<BenchSample> = None;
        for candidate in &candidates {
            let gflops = benchmark_blocking::<T>(candidate, dim);
            let dominated = best.as_ref().is_some_and(|b| gflops <= b.gflops);
            if !dominated {
                best = Some(BenchSample {
                    blocking: *candidate,
                    gflops,
                });
            }
        }

        let winner = best
            .map(|b| b.blocking)
            .unwrap_or_else(|| GemmBlocking::for_kernel_with_elem_size(key.elem_size, &shape));

        self.insert(key, winner);
        winner
    }

    /// Returns the number of cached entries.
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.cache.lock().map_or(0, |g| g.len())
    }

    /// Clears all cached tuning results.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.cache.lock() {
            guard.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

/// Returns a reference to the global `RuntimeAutoTuner` singleton.
///
/// The tuner is lazily initialized on first access.
fn global_tuner() -> &'static RuntimeAutoTuner {
    static INSTANCE: OnceLock<RuntimeAutoTuner> = OnceLock::new();
    INSTANCE.get_or_init(RuntimeAutoTuner::new)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Tunes GEMM for the given scalar type and returns the optimal blocking.
///
/// This function uses the global tuning cache. On the first call for a given
/// (T, SizeClass) pair it will run micro-benchmarks (a few hundred
/// milliseconds). Subsequent calls return the cached result instantly.
///
/// # Example
///
/// ```rust,no_run
/// use oxiblas_blas::level3::runtime_autotune::auto_tune_gemm;
///
/// let blocking = auto_tune_gemm::<f64>();
/// println!("MC={}, KC={}, NC={}", blocking.mc, blocking.kc, blocking.nc);
/// ```
#[must_use]
pub fn auto_tune_gemm<T: Field + GemmKernel + bytemuck::Zeroable>() -> GemmBlocking {
    auto_tune_gemm_for_size::<T>(SizeClass::Large)
}

/// Tunes GEMM for a specific size class.
#[must_use]
pub fn auto_tune_gemm_for_size<T: Field + GemmKernel + bytemuck::Zeroable>(
    size_class: SizeClass,
) -> GemmBlocking {
    global_tuner().tune_gemm::<T>(size_class)
}

/// GEMM with runtime-tuned parameters.
///
/// Selects the optimal blocking for the given matrix dimensions by:
/// 1. Classifying the matrix size (small / medium / large).
/// 2. Looking up (or computing) the optimal blocking in the global cache.
/// 3. Executing GEMM with those parameters.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A * B
/// * `a`     - Left matrix (m x k)
/// * `b`     - Right matrix (k x n)
/// * `beta`  - Scalar multiplier for C
/// * `c`     - Output matrix (m x n)
pub fn gemm_auto_tuned<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: oxiblas_matrix::MatRef<'_, T>,
    b: oxiblas_matrix::MatRef<'_, T>,
    beta: T,
    c: oxiblas_matrix::MatMut<'_, T>,
) {
    gemm_auto_tuned_with_par(alpha, a, b, beta, c, Par::Seq);
}

/// GEMM with runtime-tuned parameters and parallelization control.
pub fn gemm_auto_tuned_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: oxiblas_matrix::MatRef<'_, T>,
    b: oxiblas_matrix::MatRef<'_, T>,
    beta: T,
    c: oxiblas_matrix::MatMut<'_, T>,
    par: Par,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let size_class = SizeClass::from_dims(m, k, n);
    let blocking = auto_tune_gemm_for_size::<T>(size_class);
    gemm_with_blocking(alpha, a, b, beta, c, par, &blocking);
}

/// Returns the heuristic-fallback blocking for when runtime tuning is skipped.
///
/// This delegates to the existing `compute_blocking_adaptive` heuristic.
#[must_use]
pub fn heuristic_fallback<T: Field + GemmKernel>(m: usize, k: usize, n: usize) -> GemmBlocking {
    let shape = T::micro_kernel_shape();
    GemmBlocking::auto_tuned::<T>(m, k, n, &shape)
}

// ---------------------------------------------------------------------------
// Helper: GemmBlocking extensions used internally
// ---------------------------------------------------------------------------

impl GemmBlocking {
    /// Raw version that takes elem_size directly (avoids generic bound on Field).
    #[must_use]
    pub(crate) fn auto_tuned_raw(
        m: usize,
        k: usize,
        n: usize,
        elem_size: usize,
        shape: &MicroKernelShape,
    ) -> Self {
        let tuned = crate::level3::autotune::compute_blocking_adaptive(m, k, n, elem_size, shape);
        Self {
            mc: tuned.mc,
            kc: tuned.kc,
            nc: tuned.nc,
        }
    }

    /// Static defaults keyed by elem_size (avoids requiring Field bound).
    #[must_use]
    pub(crate) fn for_kernel_with_elem_size(elem_size: usize, shape: &MicroKernelShape) -> Self {
        let mr = shape.mr;
        let nr = shape.nr;

        let (mc, kc) = if elem_size >= 8 {
            (576, 448)
        } else {
            (576, 896)
        };

        Self {
            nc: (2048 / nr) * nr,
            kc,
            mc: (mc / mr) * mr,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_from_dims() {
        assert_eq!(SizeClass::from_dims(32, 32, 32), SizeClass::Small);
        assert_eq!(SizeClass::from_dims(256, 256, 256), SizeClass::Medium);
        assert_eq!(SizeClass::from_dims(1024, 512, 768), SizeClass::Large);
        // Edge: one large dimension is enough
        assert_eq!(SizeClass::from_dims(32, 32, 1024), SizeClass::Large);
    }

    #[test]
    fn test_tuning_key_hash() {
        let k1 = TuningKey {
            op: OpKind::Gemm,
            elem_size: 8,
            size_class: SizeClass::Large,
        };
        let k2 = TuningKey {
            op: OpKind::Gemm,
            elem_size: 4,
            size_class: SizeClass::Large,
        };
        assert_ne!(k1, k2);

        let k3 = TuningKey {
            op: OpKind::Gemm,
            elem_size: 8,
            size_class: SizeClass::Large,
        };
        assert_eq!(k1, k3);
    }

    #[test]
    fn test_generate_candidates() {
        let shape = MicroKernelShape { mr: 8, nr: 6 };
        let candidates = generate_candidates(8, &shape, SizeClass::Large);

        // Should have at least the base + some variations
        assert!(
            candidates.len() >= 2,
            "Expected at least 2 candidates, got {}",
            candidates.len()
        );

        // All MC values should be multiples of mr
        for c in &candidates {
            assert_eq!(
                c.mc % shape.mr,
                0,
                "MC={} not aligned to MR={}",
                c.mc,
                shape.mr
            );
        }

        // All NC values should be multiples of nr
        for c in &candidates {
            assert_eq!(
                c.nc % shape.nr,
                0,
                "NC={} not aligned to NR={}",
                c.nc,
                shape.nr
            );
        }
    }

    #[test]
    fn test_runtime_auto_tuner_new() {
        let tuner = RuntimeAutoTuner::new();
        assert_eq!(tuner.cache_len(), 0);
    }

    #[test]
    fn test_runtime_auto_tuner_insert_get() {
        let tuner = RuntimeAutoTuner::new();
        let key = TuningKey {
            op: OpKind::Gemm,
            elem_size: 8,
            size_class: SizeClass::Medium,
        };
        let blocking = GemmBlocking {
            mc: 128,
            kc: 256,
            nc: 512,
        };

        assert!(tuner.get(&key).is_none());

        tuner.insert(key, blocking);
        let got = tuner.get(&key);
        assert!(got.is_some());
        let got = got.expect("just inserted");
        assert_eq!(got.mc, 128);
        assert_eq!(got.kc, 256);
        assert_eq!(got.nc, 512);
    }

    #[test]
    fn test_runtime_auto_tuner_clear() {
        let tuner = RuntimeAutoTuner::new();
        let key = TuningKey {
            op: OpKind::Gemm,
            elem_size: 8,
            size_class: SizeClass::Small,
        };
        tuner.insert(
            key,
            GemmBlocking {
                mc: 64,
                kc: 64,
                nc: 64,
            },
        );
        assert_eq!(tuner.cache_len(), 1);

        tuner.clear();
        assert_eq!(tuner.cache_len(), 0);
        assert!(tuner.get(&key).is_none());
    }

    #[test]
    fn test_tune_gemm_f64_produces_valid_blocking() {
        let tuner = RuntimeAutoTuner::new();
        let blocking = tuner.tune_gemm::<f64>(SizeClass::Small);

        let shape = f64::micro_kernel_shape();
        assert!(blocking.mc > 0, "MC must be > 0");
        assert!(blocking.kc > 0, "KC must be > 0");
        assert!(blocking.nc > 0, "NC must be > 0");
        assert_eq!(blocking.mc % shape.mr, 0, "MC must be aligned to MR");
    }

    #[test]
    fn test_tune_gemm_f32_produces_valid_blocking() {
        let tuner = RuntimeAutoTuner::new();
        let blocking = tuner.tune_gemm::<f32>(SizeClass::Small);

        let shape = f32::micro_kernel_shape();
        assert!(blocking.mc > 0, "MC must be > 0");
        assert!(blocking.kc > 0, "KC must be > 0");
        assert!(blocking.nc > 0, "NC must be > 0");
        assert_eq!(blocking.mc % shape.mr, 0, "MC must be aligned to MR");
    }

    #[test]
    fn test_tune_gemm_caches_result() {
        let tuner = RuntimeAutoTuner::new();
        assert_eq!(tuner.cache_len(), 0);

        let b1 = tuner.tune_gemm::<f64>(SizeClass::Medium);
        assert_eq!(tuner.cache_len(), 1);

        // Second call should return cached value
        let b2 = tuner.tune_gemm::<f64>(SizeClass::Medium);
        assert_eq!(tuner.cache_len(), 1);
        assert_eq!(b1.mc, b2.mc);
        assert_eq!(b1.kc, b2.kc);
        assert_eq!(b1.nc, b2.nc);
    }

    #[test]
    fn test_gemm_auto_tuned_correctness_f64() {
        // Verify that runtime-tuned GEMM produces correct results
        let m = 64;
        let k = 48;
        let n = 32;

        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        gemm_auto_tuned(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        let expected = (k as f64) * 2.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_gemm_auto_tuned_correctness_f32() {
        let m = 48;
        let k = 32;
        let n = 64;

        let a: Mat<f32> = Mat::filled(m, k, 1.0_f32);
        let b: Mat<f32> = Mat::filled(k, n, 3.0_f32);
        let mut c: Mat<f32> = Mat::zeros(m, n);

        gemm_auto_tuned(1.0_f32, a.as_ref(), b.as_ref(), 0.0_f32, c.as_mut());

        let expected = (k as f32) * 3.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-3,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_gemm_auto_tuned_with_alpha_beta() {
        let m = 32;
        let k = 16;
        let n = 32;

        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::filled(m, n, 10.0);

        // C = 2 * A * B + 3 * C
        // A*B each element = 16 * 2 = 32
        // Result = 2 * 32 + 3 * 10 = 94
        gemm_auto_tuned(2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());

        let expected = 2.0 * (k as f64) * 2.0 + 3.0 * 10.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_heuristic_fallback() {
        let blocking = heuristic_fallback::<f64>(256, 256, 256);
        assert!(blocking.mc > 0);
        assert!(blocking.kc > 0);
        assert!(blocking.nc > 0);
    }

    #[test]
    fn test_global_tuner_singleton() {
        // Exercise the global singleton through the public API
        let b1 = auto_tune_gemm::<f64>();
        let b2 = auto_tune_gemm::<f64>();

        // Should be identical (cached)
        assert_eq!(b1.mc, b2.mc);
        assert_eq!(b1.kc, b2.kc);
        assert_eq!(b1.nc, b2.nc);
    }

    #[test]
    fn test_different_size_classes_may_differ() {
        let tuner = RuntimeAutoTuner::new();

        let small = tuner.tune_gemm::<f64>(SizeClass::Small);
        let large = tuner.tune_gemm::<f64>(SizeClass::Large);

        // They are allowed to be the same, but both must be valid
        assert!(small.mc > 0);
        assert!(large.mc > 0);
        assert!(small.kc > 0);
        assert!(large.kc > 0);
    }

    #[test]
    fn test_auto_tune_gemm_for_size() {
        let blocking = auto_tune_gemm_for_size::<f64>(SizeClass::Medium);
        assert!(blocking.mc > 0);
        assert!(blocking.kc > 0);
        assert!(blocking.nc > 0);
    }

    #[test]
    fn test_benchmark_blocking_runs() {
        // Smoke test: benchmark_blocking should not panic
        let blocking = GemmBlocking::default();
        let gflops = benchmark_blocking::<f64>(&blocking, 32);
        assert!(gflops > 0.0, "GFLOP/s must be positive, got {}", gflops);
    }
}
