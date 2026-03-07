//! Parallelization primitives for OxiBLAS.
//!
//! This module provides:
//! - Parallel execution modes
//! - Work partitioning utilities
//! - Thread-local accumulation patterns

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Global flag to disable parallelism.
static PARALLELISM_DISABLED: AtomicBool = AtomicBool::new(false);

/// Disables global parallelism.
///
/// This can be useful for debugging or when running in environments
/// where threading is problematic.
pub fn disable_global_parallelism() {
    PARALLELISM_DISABLED.store(true, Ordering::SeqCst);
}

/// Enables global parallelism.
pub fn enable_global_parallelism() {
    PARALLELISM_DISABLED.store(false, Ordering::SeqCst);
}

/// Returns true if parallelism is enabled.
pub fn is_parallelism_enabled() -> bool {
    !PARALLELISM_DISABLED.load(Ordering::SeqCst)
}

/// Parallelization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Par {
    /// Sequential execution.
    Seq,
    /// Parallel execution with the default thread pool.
    #[cfg(feature = "parallel")]
    Rayon,
    /// Parallel execution with a specific number of threads.
    #[cfg(feature = "parallel")]
    RayonWith(usize),
}

// Manual impl because the default variant depends on feature flags
// (Rayon when "parallel" is enabled, Seq otherwise)
#[allow(clippy::derivable_impls)]
impl Default for Par {
    fn default() -> Self {
        #[cfg(feature = "parallel")]
        {
            Par::Rayon
        }
        #[cfg(not(feature = "parallel"))]
        {
            Par::Seq
        }
    }
}

impl Par {
    /// Returns true if this mode is sequential.
    #[inline]
    pub fn is_sequential(&self) -> bool {
        match self {
            Par::Seq => true,
            #[cfg(feature = "parallel")]
            _ => !is_parallelism_enabled(),
        }
    }

    /// Returns the number of threads to use.
    #[cfg(feature = "parallel")]
    pub fn num_threads(&self) -> usize {
        if !is_parallelism_enabled() {
            return 1;
        }

        match self {
            Par::Seq => 1,
            Par::Rayon => rayon::current_num_threads(),
            Par::RayonWith(n) => *n,
        }
    }

    /// Returns the number of threads to use (always 1 without parallel feature).
    #[cfg(not(feature = "parallel"))]
    pub fn num_threads(&self) -> usize {
        1
    }
}

/// Threshold configuration for parallel operations.
#[derive(Debug, Clone, Copy)]
pub struct ParThreshold {
    /// Minimum number of elements for parallelization.
    pub min_elements: usize,
    /// Minimum work per thread (elements).
    pub min_work_per_thread: usize,
}

impl Default for ParThreshold {
    fn default() -> Self {
        ParThreshold {
            min_elements: 4096,
            min_work_per_thread: 256,
        }
    }
}

impl ParThreshold {
    /// Creates a new threshold configuration.
    pub const fn new(min_elements: usize, min_work_per_thread: usize) -> Self {
        ParThreshold {
            min_elements,
            min_work_per_thread,
        }
    }

    /// Returns true if parallelization should be used for the given work size.
    #[inline]
    pub fn should_parallelize(&self, total_work: usize, par: Par) -> bool {
        if par.is_sequential() {
            return false;
        }

        if total_work < self.min_elements {
            return false;
        }

        let threads = par.num_threads();
        if threads <= 1 {
            return false;
        }

        total_work / threads >= self.min_work_per_thread
    }
}

/// Work range for parallel iteration.
#[derive(Debug, Clone, Copy)]
pub struct WorkRange {
    /// Start index (inclusive).
    pub start: usize,
    /// End index (exclusive).
    pub end: usize,
}

impl WorkRange {
    /// Creates a new work range.
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        WorkRange { start, end }
    }

    /// Returns the length of the range.
    #[inline]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns true if the range is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Partitions work into chunks for parallel execution.
pub fn partition_work(total: usize, num_threads: usize) -> Vec<WorkRange> {
    if num_threads == 0 || total == 0 {
        return vec![];
    }

    if num_threads == 1 {
        return vec![WorkRange::new(0, total)];
    }

    let chunk_size = total.div_ceil(num_threads);
    let mut ranges = Vec::with_capacity(num_threads);

    let mut start = 0;
    while start < total {
        let end = (start + chunk_size).min(total);
        ranges.push(WorkRange::new(start, end));
        start = end;
    }

    ranges
}

/// Executes a closure in parallel over work ranges.
///
/// If parallelism is disabled or the work is too small, executes sequentially.
#[inline]
pub fn for_each_range<F>(total: usize, par: Par, threshold: &ParThreshold, f: F)
where
    F: Fn(WorkRange) + Send + Sync,
{
    if !threshold.should_parallelize(total, par) {
        f(WorkRange::new(0, total));
        return;
    }

    #[cfg(feature = "parallel")]
    {
        let ranges = partition_work(total, par.num_threads());
        ranges.into_par_iter().for_each(|range| {
            f(range);
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        f(WorkRange::new(0, total));
    }
}

/// Parallel map-reduce operation.
///
/// Maps each work range to a value, then reduces all values.
#[allow(unused_variables)]
pub fn map_reduce<T, Map, Reduce>(
    total: usize,
    par: Par,
    threshold: &ParThreshold,
    identity: T,
    map: Map,
    reduce: Reduce,
) -> T
where
    T: Clone + Send + Sync,
    Map: Fn(WorkRange) -> T + Send + Sync,
    Reduce: Fn(T, T) -> T + Send + Sync,
{
    if !threshold.should_parallelize(total, par) {
        return map(WorkRange::new(0, total));
    }

    #[cfg(feature = "parallel")]
    {
        let ranges = partition_work(total, par.num_threads());
        ranges
            .into_par_iter()
            .map(map)
            .reduce(|| identity.clone(), reduce)
    }

    #[cfg(not(feature = "parallel"))]
    {
        map(WorkRange::new(0, total))
    }
}

/// Parallel for_each with index.
pub fn for_each_indexed<F>(total: usize, par: Par, threshold: &ParThreshold, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if !threshold.should_parallelize(total, par) {
        for i in 0..total {
            f(i);
        }
        return;
    }

    #[cfg(feature = "parallel")]
    {
        (0..total).into_par_iter().for_each(f);
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..total {
            f(i);
        }
    }
}

// =============================================================================
// Custom thread pool support
// =============================================================================

/// Trait for custom thread pool implementations.
///
/// This allows using thread pools other than rayon's global pool.
pub trait ThreadPool: Send + Sync {
    /// Returns the number of threads in the pool.
    fn num_threads(&self) -> usize;

    /// Executes a closure on the thread pool.
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static;

    /// Joins two closures, executing them potentially in parallel.
    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;

    /// Parallel for_each over a range.
    fn for_each<F>(&self, range: core::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync;

    /// Parallel map-reduce over a range.
    fn map_reduce<T, Map, Reduce>(
        &self,
        range: core::ops::Range<usize>,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> T
    where
        T: Clone + Send + Sync,
        Map: Fn(usize) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync;
}

/// A single-threaded "pool" for sequential execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialPool;

impl ThreadPool for SequentialPool {
    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        f();
    }

    #[inline]
    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        (a(), b())
    }

    fn for_each<F>(&self, range: core::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        for i in range {
            f(i);
        }
    }

    fn map_reduce<T, Map, Reduce>(
        &self,
        range: core::ops::Range<usize>,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> T
    where
        T: Clone + Send + Sync,
        Map: Fn(usize) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        let mut acc = identity;
        for i in range {
            acc = reduce(acc, map(i));
        }
        acc
    }
}

/// Wrapper for rayon's global thread pool.
#[cfg(feature = "parallel")]
#[derive(Debug, Clone, Copy, Default)]
pub struct RayonGlobalPool;

#[cfg(feature = "parallel")]
impl ThreadPool for RayonGlobalPool {
    #[inline]
    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }

    #[inline]
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        rayon::spawn(f);
    }

    #[inline]
    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        rayon::join(a, b)
    }

    fn for_each<F>(&self, range: core::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        range.into_par_iter().for_each(f);
    }

    fn map_reduce<T, Map, Reduce>(
        &self,
        range: core::ops::Range<usize>,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> T
    where
        T: Clone + Send + Sync,
        Map: Fn(usize) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        range
            .into_par_iter()
            .map(map)
            .reduce(|| identity.clone(), reduce)
    }
}

/// Wrapper for a custom rayon thread pool.
#[cfg(feature = "parallel")]
pub struct CustomRayonPool {
    pool: rayon::ThreadPool,
}

#[cfg(feature = "parallel")]
impl CustomRayonPool {
    /// Creates a new custom rayon pool with the specified number of threads.
    pub fn new(num_threads: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;
        Ok(CustomRayonPool { pool })
    }

    /// Creates a new custom rayon pool with the specified number of threads.
    ///
    /// This is an alias for [`CustomRayonPool::new`] that matches the naming
    /// convention used in the `OxiblasThreadConfig` builder API.
    pub fn with_num_threads(n: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        Self::new(n)
    }

    /// Creates a new custom rayon pool with builder configuration.
    pub fn with_builder<F>(configure: F) -> Result<Self, rayon::ThreadPoolBuildError>
    where
        F: FnOnce(rayon::ThreadPoolBuilder) -> rayon::ThreadPoolBuilder,
    {
        let builder = rayon::ThreadPoolBuilder::new();
        let pool = configure(builder).build()?;
        Ok(CustomRayonPool { pool })
    }

    /// Installs this pool for the duration of the closure.
    pub fn install<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }
}

#[cfg(feature = "parallel")]
impl ThreadPool for CustomRayonPool {
    #[inline]
    fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    #[inline]
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.pool.spawn(f);
    }

    #[inline]
    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.pool.join(a, b)
    }

    fn for_each<F>(&self, range: core::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        self.pool.install(|| {
            range.into_par_iter().for_each(f);
        });
    }

    fn map_reduce<T, Map, Reduce>(
        &self,
        range: core::ops::Range<usize>,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> T
    where
        T: Clone + Send + Sync,
        Map: Fn(usize) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        self.pool.install(|| {
            range
                .into_par_iter()
                .map(map)
                .reduce(|| identity.clone(), reduce)
        })
    }
}

/// Scoped execution context for a thread pool.
///
/// This provides a convenient way to run operations with a specific thread pool.
pub struct PoolScope<'a, P: ThreadPool> {
    pool: &'a P,
    threshold: ParThreshold,
}

impl<'a, P: ThreadPool> PoolScope<'a, P> {
    /// Creates a new pool scope with default threshold.
    pub fn new(pool: &'a P) -> Self {
        PoolScope {
            pool,
            threshold: ParThreshold::default(),
        }
    }

    /// Creates a new pool scope with a custom threshold.
    pub fn with_threshold(pool: &'a P, threshold: ParThreshold) -> Self {
        PoolScope { pool, threshold }
    }

    /// Returns the number of threads in the pool.
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.pool.num_threads()
    }

    /// Joins two closures.
    #[inline]
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.pool.join(a, b)
    }

    /// Parallel for_each over a range.
    pub fn for_each<F>(&self, total: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if total < self.threshold.min_elements || self.pool.num_threads() <= 1 {
            for i in 0..total {
                f(i);
            }
        } else {
            self.pool.for_each(0..total, f);
        }
    }

    /// Parallel for_each over work ranges.
    pub fn for_each_range<F>(&self, total: usize, f: F)
    where
        F: Fn(WorkRange) + Send + Sync,
    {
        if total < self.threshold.min_elements || self.pool.num_threads() <= 1 {
            f(WorkRange::new(0, total));
        } else {
            let ranges = partition_work(total, self.pool.num_threads());
            for range in ranges {
                f(range);
            }
        }
    }

    /// Parallel map-reduce operation.
    pub fn map_reduce<T, Map, Reduce>(
        &self,
        total: usize,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> T
    where
        T: Clone + Send + Sync,
        Map: Fn(usize) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        if total < self.threshold.min_elements || self.pool.num_threads() <= 1 {
            let mut acc = identity;
            for i in 0..total {
                acc = reduce(acc, map(i));
            }
            acc
        } else {
            self.pool.map_reduce(0..total, identity, map, reduce)
        }
    }
}

/// Gets the default thread pool based on feature flags.
#[cfg(feature = "parallel")]
pub fn default_pool() -> RayonGlobalPool {
    RayonGlobalPool
}

/// Gets the default thread pool (sequential without parallel feature).
#[cfg(not(feature = "parallel"))]
pub fn default_pool() -> SequentialPool {
    SequentialPool
}

/// Executes work with the default pool.
///
/// This is a convenience wrapper that creates a PoolScope with the default pool.
#[cfg(feature = "parallel")]
pub fn with_default_pool<R, F>(f: F) -> R
where
    F: FnOnce(PoolScope<'_, RayonGlobalPool>) -> R,
{
    let pool = RayonGlobalPool;
    f(PoolScope::new(&pool))
}

/// Executes work with the default pool (sequential version).
#[cfg(not(feature = "parallel"))]
pub fn with_default_pool<R, F>(f: F) -> R
where
    F: FnOnce(PoolScope<'_, SequentialPool>) -> R,
{
    let pool = SequentialPool;
    f(PoolScope::new(&pool))
}

// =============================================================================
// Global thread pool management
// =============================================================================

/// Configuration for the OxiBLAS thread pool.
///
/// `OxiblasThreadConfig` gathers all knobs that influence how OxiBLAS
/// chooses threads for parallel operations.  Build one with the fluent
/// builder methods, then apply it via [`set_global_thread_pool`] or
/// [`with_thread_count`].
///
/// # Example
///
/// ```rust
/// use oxiblas_core::parallel::OxiblasThreadConfig;
///
/// let cfg = OxiblasThreadConfig::new()
///     .num_threads(4)
///     .stack_size(2 * 1024 * 1024);
/// println!("threads: {}", cfg.num_threads);
/// ```
#[derive(Debug, Clone, Default)]
pub struct OxiblasThreadConfig {
    /// Number of worker threads.  `0` means "use all logical CPUs".
    pub num_threads: usize,
    /// Per-thread stack size in bytes.  `0` means "use OS default".
    pub stack_size: usize,
    /// Human-readable name prefix for spawned threads.
    pub thread_name: Option<String>,
}

impl OxiblasThreadConfig {
    /// Creates a new configuration with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the desired thread count.  Pass `0` for "all CPUs".
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Sets the per-thread stack size.  Pass `0` for the OS default.
    pub fn stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = bytes;
        self
    }

    /// Sets a human-readable name prefix for spawned threads.
    pub fn thread_name(mut self, name: impl Into<String>) -> Self {
        self.thread_name = Some(name.into());
        self
    }

    /// Returns the effective thread count, substituting the available
    /// logical CPU count when `num_threads` is `0`.
    pub fn effective_threads(&self) -> usize {
        if self.num_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            self.num_threads
        }
    }

    /// Builds a [`CustomRayonPool`] from this configuration.
    ///
    /// Returns an error if rayon fails to construct the pool.
    #[cfg(feature = "parallel")]
    pub fn build_pool(&self) -> Result<CustomRayonPool, rayon::ThreadPoolBuildError> {
        let mut builder = rayon::ThreadPoolBuilder::new().num_threads(self.effective_threads());
        if self.stack_size > 0 {
            builder = builder.stack_size(self.stack_size);
        }
        if let Some(name) = &self.thread_name {
            let name = name.clone();
            builder = builder.thread_name(move |i| format!("{name}-{i}"));
        }
        let pool = builder.build()?;
        Ok(CustomRayonPool { pool })
    }
}

// ---------------------------------------------------------------------------
// Global pool registry
// ---------------------------------------------------------------------------

/// A type-erased, `Send + Sync` trait object for thread pools stored
/// in the global registry.
#[cfg(feature = "std")]
trait AnyPool: Send + Sync {
    fn num_threads_dyn(&self) -> usize;
}

#[cfg(all(feature = "std", feature = "parallel"))]
impl AnyPool for CustomRayonPool {
    fn num_threads_dyn(&self) -> usize {
        self.num_threads()
    }
}

#[cfg(feature = "std")]
impl AnyPool for SequentialPool {
    fn num_threads_dyn(&self) -> usize {
        1
    }
}

#[cfg(feature = "std")]
static GLOBAL_POOL: std::sync::OnceLock<Box<dyn AnyPool>> = std::sync::OnceLock::new();

/// Sets the global OxiBLAS thread pool.
///
/// The pool is stored in a `OnceLock` so it can only be set **once** per
/// process.  Subsequent calls are silently ignored (the first writer wins).
///
/// # Arguments
///
/// * `pool` – Any value that implements [`ThreadPool`] and is
///   `'static + Send + Sync`.  Typically a [`CustomRayonPool`] built via
///   [`OxiblasThreadConfig::build_pool`] or
///   [`CustomRayonPool::with_num_threads`].
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "parallel")]
/// # {
/// use oxiblas_core::parallel::{CustomRayonPool, set_global_thread_pool};
/// let pool = CustomRayonPool::with_num_threads(4).expect("build pool");
/// set_global_thread_pool(pool);
/// # }
/// ```
#[cfg(all(feature = "std", feature = "parallel"))]
pub fn set_global_thread_pool(pool: CustomRayonPool) {
    let _ = GLOBAL_POOL.set(Box::new(pool));
}

/// Sets the global OxiBLAS thread pool to a sequential (single-threaded)
/// pool (available without the `parallel` feature).
#[cfg(all(feature = "std", not(feature = "parallel")))]
pub fn set_global_thread_pool(pool: SequentialPool) {
    let _ = GLOBAL_POOL.set(Box::new(pool));
}

/// Returns the number of threads in the global pool, or `1` if no pool has
/// been registered.
#[cfg(feature = "std")]
pub fn global_num_threads() -> usize {
    GLOBAL_POOL.get().map(|p| p.num_threads_dyn()).unwrap_or(1)
}

/// Executes `f` inside a temporary rayon pool with exactly `n` threads.
///
/// This is useful for benchmarks or tests that need deterministic
/// parallelism without replacing the global pool.  On platforms without
/// the `parallel` feature the closure is called directly on the current
/// thread.
///
/// # Example
///
/// ```rust
/// use oxiblas_core::parallel::with_thread_count;
///
/// with_thread_count(2, || {
///     // work here runs with (up to) 2 rayon threads
/// });
/// ```
#[cfg(feature = "parallel")]
pub fn with_thread_count(n: usize, f: impl FnOnce() + Send) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build();
    match pool {
        Ok(p) => p.install(f),
        Err(_) => f(), // fallback: run sequentially if build fails
    }
}

/// Sequential fallback when the `parallel` feature is disabled.
#[cfg(not(feature = "parallel"))]
pub fn with_thread_count(_n: usize, f: impl FnOnce()) {
    f();
}

// =============================================================================
// Thread-local accumulation
// =============================================================================

/// Thread-local accumulator for parallel reduction.
///
/// This is useful for operations like parallel summation where each thread
/// maintains its own accumulator to avoid synchronization.
///
/// Requires the `parallel` feature (which implies `std`).
#[cfg(feature = "parallel")]
pub struct ThreadLocalAccum<T> {
    values: Vec<std::sync::Mutex<T>>,
}

#[cfg(feature = "parallel")]
impl<T: Clone + Send> ThreadLocalAccum<T> {
    /// Creates a new thread-local accumulator.
    pub fn new(identity: T) -> Self {
        let num_threads = rayon::current_num_threads();
        let values = (0..num_threads)
            .map(|_| std::sync::Mutex::new(identity.clone()))
            .collect();
        ThreadLocalAccum { values }
    }

    /// Gets or initializes the accumulator for the current thread.
    pub fn get(&self) -> std::sync::MutexGuard<'_, T> {
        let thread_idx = rayon::current_thread_index().unwrap_or(0) % self.values.len();
        self.values[thread_idx]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Reduces all thread-local values into a single result.
    pub fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self.values
            .into_iter()
            .map(|m| {
                m.into_inner()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
            .reduce(f)
            .expect("ThreadLocalAccum should have at least one value")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_work() {
        let ranges = partition_work(100, 4);
        assert_eq!(ranges.len(), 4);

        // Check that ranges cover everything
        let mut covered = [false; 100];
        for range in &ranges {
            for (offset, covered_elem) in covered[range.start..range.end].iter_mut().enumerate() {
                let i = range.start + offset;
                assert!(!*covered_elem, "Overlap at {}", i);
                *covered_elem = true;
            }
        }
        assert!(covered.iter().all(|&x| x), "Not all elements covered");
    }

    #[test]
    fn test_partition_work_uneven() {
        let ranges = partition_work(10, 4);

        // Total should equal original
        let total: usize = ranges.iter().map(|r| r.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_partition_work_single() {
        let ranges = partition_work(100, 1);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[0].end, 100);
    }

    #[test]
    fn test_threshold() {
        let threshold = ParThreshold::new(100, 10);

        assert!(!threshold.should_parallelize(50, Par::Seq));
        assert!(!threshold.should_parallelize(50, Par::default()));

        #[cfg(feature = "parallel")]
        {
            // Only tests with parallel feature
            assert!(threshold.should_parallelize(1000, Par::Rayon));
        }
    }

    #[test]
    fn test_global_parallelism() {
        // Save current state
        let was_enabled = is_parallelism_enabled();

        disable_global_parallelism();
        assert!(!is_parallelism_enabled());

        enable_global_parallelism();
        assert!(is_parallelism_enabled());

        // Restore
        if !was_enabled {
            disable_global_parallelism();
        }
    }

    #[test]
    fn test_sequential_map_reduce() {
        let result = map_reduce(
            100,
            Par::Seq,
            &ParThreshold::default(),
            0usize,
            |range| range.len(),
            |a, b| a + b,
        );
        assert_eq!(result, 100);
    }

    // Thread pool tests
    #[test]
    fn test_sequential_pool() {
        let pool = SequentialPool;

        assert_eq!(pool.num_threads(), 1);

        // Test join
        let (a, b) = pool.join(|| 1 + 1, || 2 + 2);
        assert_eq!(a, 2);
        assert_eq!(b, 4);

        // Test for_each
        let sum = std::sync::atomic::AtomicUsize::new(0);
        pool.for_each(0..10, |i| {
            sum.fetch_add(i, std::sync::atomic::Ordering::SeqCst);
        });
        assert_eq!(sum.load(std::sync::atomic::Ordering::SeqCst), 45);

        // Test map_reduce
        let result = pool.map_reduce(0..10, 0, |i| i, |a, b| a + b);
        assert_eq!(result, 45);
    }

    #[test]
    fn test_pool_scope() {
        let pool = SequentialPool;
        let scope = PoolScope::new(&pool);

        assert_eq!(scope.num_threads(), 1);

        // Test map_reduce
        let result = scope.map_reduce(100, 0usize, |i| i, |a, b| a + b);
        assert_eq!(result, (0..100).sum::<usize>());

        // Test for_each
        let sum = std::sync::atomic::AtomicUsize::new(0);
        scope.for_each(10, |i| {
            sum.fetch_add(i, std::sync::atomic::Ordering::SeqCst);
        });
        assert_eq!(sum.load(std::sync::atomic::Ordering::SeqCst), 45);
    }

    #[test]
    fn test_pool_scope_with_threshold() {
        let pool = SequentialPool;
        let threshold = ParThreshold::new(50, 10);
        let scope = PoolScope::with_threshold(&pool, threshold);

        // Should work the same for sequential pool
        let result = scope.map_reduce(100, 0usize, |i| i, |a, b| a + b);
        assert_eq!(result, (0..100).sum::<usize>());
    }

    #[test]
    fn test_default_pool() {
        let pool = default_pool();
        // Should have at least 1 thread
        assert!(pool.num_threads() >= 1);
    }

    #[test]
    fn test_with_default_pool() {
        let result = with_default_pool(|scope| scope.num_threads());
        assert!(result >= 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_rayon_global_pool() {
        let pool = RayonGlobalPool;

        // Should have multiple threads on most systems
        assert!(pool.num_threads() >= 1);

        // Test join
        let (a, b) = pool.join(|| 1 + 1, || 2 + 2);
        assert_eq!(a, 2);
        assert_eq!(b, 4);

        // Test map_reduce
        let result = pool.map_reduce(0..100, 0, |i| i, |a, b| a + b);
        assert_eq!(result, (0..100).sum::<usize>());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_custom_rayon_pool() {
        let pool = CustomRayonPool::new(2).expect("Failed to create pool");

        assert_eq!(pool.num_threads(), 2);

        // Test map_reduce
        let result = pool.map_reduce(0..100, 0, |i| i, |a, b| a + b);
        assert_eq!(result, (0..100).sum::<usize>());

        // Test install
        let result = pool.install(|| (0..100).into_par_iter().sum::<usize>());
        assert_eq!(result, (0..100).sum());
    }

    // ---- OxiblasThreadConfig tests ------------------------------------------

    #[test]
    fn test_thread_config_default() {
        let cfg = OxiblasThreadConfig::default();
        assert_eq!(cfg.num_threads, 0);
        assert_eq!(cfg.stack_size, 0);
        assert!(cfg.thread_name.is_none());
    }

    #[test]
    fn test_thread_config_builder() {
        let cfg = OxiblasThreadConfig::new()
            .num_threads(4)
            .stack_size(1024 * 1024)
            .thread_name("oxiblas-worker");
        assert_eq!(cfg.num_threads, 4);
        assert_eq!(cfg.stack_size, 1024 * 1024);
        assert_eq!(cfg.thread_name.as_deref(), Some("oxiblas-worker"));
    }

    #[test]
    fn test_thread_config_effective_threads_zero() {
        let cfg = OxiblasThreadConfig::new().num_threads(0);
        // effective_threads should fall back to available parallelism (>= 1)
        assert!(cfg.effective_threads() >= 1);
    }

    #[test]
    fn test_thread_config_effective_threads_explicit() {
        let cfg = OxiblasThreadConfig::new().num_threads(3);
        assert_eq!(cfg.effective_threads(), 3);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_custom_rayon_pool_with_num_threads() {
        let pool = CustomRayonPool::with_num_threads(2).expect("build pool");
        assert_eq!(pool.num_threads(), 2);
        let sum: usize = pool.map_reduce(0..50, 0, |i| i, |a, b| a + b);
        assert_eq!(sum, (0..50).sum::<usize>());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_oxiblas_thread_config_build_pool() {
        let cfg = OxiblasThreadConfig::new().num_threads(2);
        let pool = cfg.build_pool().expect("build pool");
        assert_eq!(pool.num_threads(), 2);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_with_thread_count() {
        // Run inside a 2-thread pool and verify rayon sees 2 threads.
        with_thread_count(2, || {
            assert_eq!(rayon::current_num_threads(), 2);
        });
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_with_thread_count_sequential() {
        // Without parallel feature, should just call the closure directly.
        let mut called = false;
        with_thread_count(4, || {
            called = true;
        });
        assert!(called);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_global_num_threads_default() {
        // Before any pool is registered the answer must be at least 1.
        // (May be > 1 if a sibling test already set the global pool.)
        assert!(global_num_threads() >= 1);
    }
}
