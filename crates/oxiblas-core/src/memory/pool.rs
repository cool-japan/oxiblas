//! Memory management utilities for OxiBLAS.
//!
//! This module provides:
//! - Aligned memory allocation
//! - Stack-based temporary allocation (StackReq pattern)
//! - Cache-aware data layout utilities
//! - Prefetch hints for cache optimization
//! - Memory pool for temporary allocations
//! - Custom allocator support via the `Alloc` trait

use core::mem::size_of;

use super::aligned_vec::AlignedVec;
use super::alloc::*;

// =============================================================================
// MemoryPool - Pooled allocation for temporary buffers
// =============================================================================

/// A memory pool for temporary allocations.
///
/// This provides fast allocation for scratch space by reusing previously
/// allocated buffers. Useful for algorithms that repeatedly allocate and
/// free temporary storage of similar sizes.
///
/// # Thread Safety
///
/// This pool is NOT thread-safe. Use one pool per thread or wrap in a mutex.
///
/// # Example
///
/// ```
/// use oxiblas_core::memory::MemoryPool;
///
/// let mut pool = MemoryPool::new();
///
/// // Acquire a buffer
/// let buffer: Vec<f64> = pool.acquire(100);
/// // ... use buffer ...
///
/// // Return buffer to pool for reuse
/// pool.release(buffer);
///
/// // Next acquire may reuse the buffer
/// let buffer2: Vec<f64> = pool.acquire(50);
/// ```
pub struct MemoryPool<T> {
    /// Cached buffers, sorted by capacity (smallest first).
    buffers: Vec<Vec<T>>,
    /// Maximum number of buffers to keep.
    max_cached: usize,
    /// Maximum total bytes to keep cached.
    max_bytes: usize,
    /// Current cached bytes.
    cached_bytes: usize,
}

impl<T> Default for MemoryPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MemoryPool<T> {
    /// Creates a new memory pool with default limits.
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            max_cached: 16,
            max_bytes: 16 * 1024 * 1024, // 16 MB default
            cached_bytes: 0,
        }
    }

    /// Creates a new memory pool with custom limits.
    ///
    /// # Arguments
    /// * `max_cached` - Maximum number of buffers to cache
    /// * `max_bytes` - Maximum total bytes to cache
    pub fn with_limits(max_cached: usize, max_bytes: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_cached,
            max_bytes,
            cached_bytes: 0,
        }
    }

    /// Acquires a buffer with at least the given capacity.
    ///
    /// If a suitable buffer exists in the pool, it is reused.
    /// Otherwise, a new buffer is allocated.
    pub fn acquire(&mut self, min_capacity: usize) -> Vec<T> {
        // Find the smallest buffer that fits
        let pos = self
            .buffers
            .iter()
            .position(|b| b.capacity() >= min_capacity);

        if let Some(idx) = pos {
            let mut buffer = self.buffers.remove(idx);
            self.cached_bytes -= buffer.capacity() * size_of::<T>();
            buffer.clear();
            buffer
        } else {
            Vec::with_capacity(min_capacity)
        }
    }

    /// Returns a buffer to the pool for future reuse.
    ///
    /// The buffer is cleared before being stored.
    pub fn release(&mut self, mut buffer: Vec<T>) {
        let bytes = buffer.capacity() * size_of::<T>();

        // Check if we can cache this buffer
        if self.buffers.len() >= self.max_cached {
            // Pool is full, just drop the buffer
            return;
        }

        if self.cached_bytes + bytes > self.max_bytes {
            // Would exceed byte limit, drop the buffer
            return;
        }

        buffer.clear();
        self.cached_bytes += bytes;

        // Insert in sorted order by capacity
        let pos = self
            .buffers
            .iter()
            .position(|b| b.capacity() >= buffer.capacity())
            .unwrap_or(self.buffers.len());

        self.buffers.insert(pos, buffer);
    }

    /// Returns the number of cached buffers.
    pub fn cached_count(&self) -> usize {
        self.buffers.len()
    }

    /// Returns the total cached bytes.
    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    /// Clears all cached buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.cached_bytes = 0;
    }

    /// Shrinks the pool to fit within the current limits.
    ///
    /// Removes the largest buffers first to stay within limits.
    pub fn shrink_to_limits(&mut self) {
        while self.buffers.len() > self.max_cached || self.cached_bytes > self.max_bytes {
            if let Some(buffer) = self.buffers.pop() {
                self.cached_bytes -= buffer.capacity() * size_of::<T>();
            } else {
                break;
            }
        }
    }
}

/// A typed memory pool for aligned allocations.
///
/// Unlike `MemoryPool`, this uses `AlignedVec` for SIMD-friendly allocations.
pub struct AlignedPool<T, const ALIGN: usize = DEFAULT_ALIGN> {
    /// Cached buffers.
    buffers: Vec<AlignedVec<T, ALIGN>>,
    /// Maximum number of buffers to keep.
    max_cached: usize,
}

impl<T: Clone, const ALIGN: usize> Default for AlignedPool<T, ALIGN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const ALIGN: usize> AlignedPool<T, ALIGN> {
    /// Creates a new aligned memory pool.
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            max_cached: 8,
        }
    }

    /// Creates a pool with a custom cache limit.
    pub fn with_limit(max_cached: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_cached,
        }
    }

    /// Acquires an aligned buffer with at least the given capacity.
    pub fn acquire(&mut self, min_capacity: usize) -> AlignedVec<T, ALIGN> {
        // Find a suitable buffer
        let pos = self
            .buffers
            .iter()
            .position(|b| b.capacity() >= min_capacity);

        if let Some(idx) = pos {
            let mut buffer = self.buffers.remove(idx);
            buffer.clear();
            buffer
        } else {
            AlignedVec::with_capacity(min_capacity)
        }
    }

    /// Returns an aligned buffer to the pool.
    pub fn release(&mut self, mut buffer: AlignedVec<T, ALIGN>) {
        if self.buffers.len() >= self.max_cached {
            // Pool is full
            return;
        }

        buffer.clear();

        // Insert in sorted order by capacity
        let pos = self
            .buffers
            .iter()
            .position(|b| b.capacity() >= buffer.capacity())
            .unwrap_or(self.buffers.len());

        self.buffers.insert(pos, buffer);
    }

    /// Clears all cached buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}
