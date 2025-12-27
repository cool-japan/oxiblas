//! Prefetch utilities for improved cache performance.
//!
//! This module provides prefetch hints that can improve performance for
//! large matrix operations by bringing data into cache before it's needed.
//!
//! # Cache Hierarchy
//!
//! Modern CPUs have multiple cache levels:
//! - L1 (fastest, smallest, ~32KB per core)
//! - L2 (fast, medium, ~256KB-1MB per core)
//! - L3 (slower, shared, ~8-32MB)
//!
//! # Usage
//!
//! Prefetching is most effective when:
//! - Processing large matrices that don't fit in cache
//! - Access patterns are predictable (sequential or strided)
//! - There's enough distance between prefetch and use
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_matrix::prefetch::{prefetch_read, PrefetchLocality};
//!
//! // Prefetch data for upcoming reads
//! for i in (0..n).step_by(64 / size_of::<f64>()) {
//!     prefetch_read(&data[i + PREFETCH_DISTANCE], PrefetchLocality::Medium);
//! }
//! ```

/// Cache locality hint for prefetch operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLocality {
    /// Non-temporal: Data will be used once and not reused.
    /// Prefetches to L1 but may be evicted quickly.
    NonTemporal,
    /// Low: Data will be used a few times.
    /// Typically prefetches to L3.
    Low,
    /// Medium: Data will be used moderately.
    /// Typically prefetches to L2.
    Medium,
    /// High: Data will be heavily reused.
    /// Prefetches to L1 for fastest access.
    High,
}

/// Prefetch data for reading.
///
/// Issues a prefetch hint to bring the cache line containing `ptr` into cache.
/// This is a hint and may be ignored by the CPU.
///
/// # Safety
///
/// The pointer must be valid for at least one byte, but doesn't need to be
/// aligned to a cache line. The CPU will prefetch the entire cache line
/// containing the address.
#[inline]
pub fn prefetch_read<T>(ptr: *const T, locality: PrefetchLocality) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            match locality {
                PrefetchLocality::NonTemporal => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
                PrefetchLocality::Low => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
                PrefetchLocality::Medium => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
                PrefetchLocality::High => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON prefetch using inline assembly
        // PRFM instruction with PLDL1KEEP, PLDL2KEEP, PLDL3KEEP
        unsafe {
            match locality {
                PrefetchLocality::NonTemporal | PrefetchLocality::Low => {
                    core::arch::asm!(
                        "prfm pldl3keep, [{0}]",
                        in(reg) ptr,
                        options(nostack, preserves_flags)
                    );
                }
                PrefetchLocality::Medium => {
                    core::arch::asm!(
                        "prfm pldl2keep, [{0}]",
                        in(reg) ptr,
                        options(nostack, preserves_flags)
                    );
                }
                PrefetchLocality::High => {
                    core::arch::asm!(
                        "prfm pldl1keep, [{0}]",
                        in(reg) ptr,
                        options(nostack, preserves_flags)
                    );
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No prefetch available - ignore
        let _ = (ptr, locality);
    }
}

/// Prefetch data for writing.
///
/// Similar to `prefetch_read` but hints that the data will be written.
/// This can help avoid read-for-ownership overhead on some architectures.
#[inline]
pub fn prefetch_write<T>(ptr: *mut T, locality: PrefetchLocality) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        // x86 prefetchw is not always available, use prefetcht0 as fallback
        unsafe {
            match locality {
                PrefetchLocality::NonTemporal => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
                PrefetchLocality::Low => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
                PrefetchLocality::Medium => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
                PrefetchLocality::High => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON prefetch for store using PSTL1KEEP, PSTL2KEEP, PSTL3KEEP
        unsafe {
            match locality {
                PrefetchLocality::NonTemporal | PrefetchLocality::Low => {
                    core::arch::asm!(
                        "prfm pstl3keep, [{0}]",
                        in(reg) ptr,
                        options(nostack, preserves_flags)
                    );
                }
                PrefetchLocality::Medium => {
                    core::arch::asm!(
                        "prfm pstl2keep, [{0}]",
                        in(reg) ptr,
                        options(nostack, preserves_flags)
                    );
                }
                PrefetchLocality::High => {
                    core::arch::asm!(
                        "prfm pstl1keep, [{0}]",
                        in(reg) ptr,
                        options(nostack, preserves_flags)
                    );
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (ptr, locality);
    }
}

/// Cache line size in bytes (typical for modern CPUs).
pub const CACHE_LINE_SIZE: usize = 64;

/// Suggested prefetch distance in cache lines for sequential access.
///
/// This is the number of cache lines ahead to prefetch. The optimal value
/// depends on memory latency and processing speed.
pub const PREFETCH_DISTANCE_LINES: usize = 8;

/// Suggested prefetch distance in bytes for sequential access.
pub const PREFETCH_DISTANCE_BYTES: usize = PREFETCH_DISTANCE_LINES * CACHE_LINE_SIZE;

/// Prefetch a range of memory for reading.
///
/// Prefetches cache lines covering the range `[ptr, ptr + len)`.
/// Useful for preparing a contiguous block of data.
#[inline]
pub fn prefetch_range_read<T>(ptr: *const T, len: usize, locality: PrefetchLocality) {
    if len == 0 {
        return;
    }

    let elem_size = core::mem::size_of::<T>();
    let byte_len = len * elem_size;
    let num_lines = byte_len.div_ceil(CACHE_LINE_SIZE);

    for i in 0..num_lines {
        let offset = i * CACHE_LINE_SIZE;
        let addr = unsafe { (ptr as *const u8).add(offset) };
        prefetch_read(addr, locality);
    }
}

/// Prefetch a range of memory for writing.
#[inline]
pub fn prefetch_range_write<T>(ptr: *mut T, len: usize, locality: PrefetchLocality) {
    if len == 0 {
        return;
    }

    let elem_size = core::mem::size_of::<T>();
    let byte_len = len * elem_size;
    let num_lines = byte_len.div_ceil(CACHE_LINE_SIZE);

    for i in 0..num_lines {
        let offset = i * CACHE_LINE_SIZE;
        let addr = unsafe { (ptr as *mut u8).add(offset) };
        prefetch_write(addr, locality);
    }
}

/// Prefetch a column of a matrix for reading.
///
/// For column-major storage, this prefetches contiguous memory.
/// For row-major or strided access, this prefetches with the given stride.
#[inline]
pub fn prefetch_column<T>(
    ptr: *const T,
    nrows: usize,
    row_stride: usize,
    locality: PrefetchLocality,
) {
    let elem_size = core::mem::size_of::<T>();

    // If contiguous (row_stride == nrows for column-major), prefetch as range
    if row_stride == 1 || (row_stride * elem_size) <= CACHE_LINE_SIZE {
        prefetch_range_read(ptr, nrows, locality);
    } else {
        // Strided access - prefetch individual cache lines
        // Only prefetch if the stride is large enough to warrant it
        let lines_per_column = (nrows * elem_size).div_ceil(CACHE_LINE_SIZE);
        for i in 0..lines_per_column.min(nrows) {
            let row = i * (CACHE_LINE_SIZE / elem_size).max(1);
            if row < nrows {
                let addr = unsafe { ptr.add(row * row_stride) };
                prefetch_read(addr, locality);
            }
        }
    }
}

/// Prefetch a block of a matrix for reading.
///
/// Prefetches a rectangular block starting at `ptr` with dimensions
/// `block_rows × block_cols`.
#[inline]
pub fn prefetch_block<T>(
    ptr: *const T,
    block_rows: usize,
    block_cols: usize,
    row_stride: usize,
    locality: PrefetchLocality,
) {
    for j in 0..block_cols {
        let col_ptr = unsafe { ptr.add(j * row_stride) };
        prefetch_column(col_ptr, block_rows, 1, locality);
    }
}

/// Prefetch hint for matrix operations.
///
/// This struct provides a convenient interface for prefetching during
/// matrix operations with predictable access patterns.
pub struct MatrixPrefetcher<T> {
    /// Base pointer.
    ptr: *const T,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Row stride.
    row_stride: usize,
    /// Current prefetch column.
    current_col: usize,
    /// Prefetch distance in columns.
    distance: usize,
    /// Locality hint.
    locality: PrefetchLocality,
}

impl<T> MatrixPrefetcher<T> {
    /// Creates a new matrix prefetcher.
    ///
    /// # Parameters
    /// - `ptr`: Pointer to matrix data
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns
    /// - `row_stride`: Stride between rows (leading dimension)
    /// - `distance`: Number of columns to prefetch ahead
    /// - `locality`: Cache locality hint
    #[inline]
    pub fn new(
        ptr: *const T,
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        distance: usize,
        locality: PrefetchLocality,
    ) -> Self {
        let prefetcher = MatrixPrefetcher {
            ptr,
            nrows,
            ncols,
            row_stride,
            current_col: 0,
            distance,
            locality,
        };

        // Prefetch initial columns
        for j in 0..distance.min(ncols) {
            let col_ptr = unsafe { ptr.add(j * row_stride) };
            prefetch_column(col_ptr, nrows, 1, locality);
        }

        prefetcher
    }

    /// Advance to the next column and prefetch ahead.
    ///
    /// Call this as you process each column to keep data prefetched.
    #[inline]
    pub fn advance(&mut self) {
        self.current_col += 1;

        let prefetch_col = self.current_col + self.distance;
        if prefetch_col < self.ncols {
            let col_ptr = unsafe { self.ptr.add(prefetch_col * self.row_stride) };
            prefetch_column(col_ptr, self.nrows, 1, self.locality);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_locality() {
        assert_ne!(PrefetchLocality::High, PrefetchLocality::Low);
        assert_eq!(PrefetchLocality::Medium, PrefetchLocality::Medium);
    }

    // Prefetch tests use inline assembly which miri doesn't support
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_read_safety() {
        // Prefetching should not crash even with unusual inputs
        let data = [1.0f64; 1024];

        prefetch_read(data.as_ptr(), PrefetchLocality::High);
        prefetch_read(data.as_ptr().wrapping_add(100), PrefetchLocality::Medium);
        prefetch_read(data.as_ptr().wrapping_add(500), PrefetchLocality::Low);
        prefetch_read(
            data.as_ptr().wrapping_add(900),
            PrefetchLocality::NonTemporal,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_write_safety() {
        let mut data = [1.0f64; 1024];

        prefetch_write(data.as_mut_ptr(), PrefetchLocality::High);
        prefetch_write(
            data.as_mut_ptr().wrapping_add(100),
            PrefetchLocality::Medium,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_range() {
        let data = vec![1.0f64; 4096];

        // Should not crash
        prefetch_range_read(data.as_ptr(), data.len(), PrefetchLocality::Medium);
        prefetch_range_read(data.as_ptr(), 0, PrefetchLocality::High); // Empty range
        prefetch_range_read(data.as_ptr(), 1, PrefetchLocality::Low); // Single element
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_column() {
        let data = vec![1.0f64; 1000];

        // Contiguous column
        prefetch_column(data.as_ptr(), 100, 1, PrefetchLocality::High);

        // Strided column (simulating row-major access)
        prefetch_column(data.as_ptr(), 10, 100, PrefetchLocality::Medium);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_block() {
        let data = vec![1.0f64; 10000];

        // Prefetch a 64x64 block with stride 100
        prefetch_block(data.as_ptr(), 64, 64, 100, PrefetchLocality::High);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_matrix_prefetcher() {
        let data = vec![1.0f64; 10000];

        let mut prefetcher = MatrixPrefetcher::new(
            data.as_ptr(),
            100, // nrows
            100, // ncols
            100, // row_stride
            8,   // distance
            PrefetchLocality::Medium,
        );

        // Simulate processing columns
        for _ in 0..100 {
            prefetcher.advance();
        }
    }

    #[test]
    fn test_cache_constants() {
        assert_eq!(CACHE_LINE_SIZE, 64);
        const { assert!(PREFETCH_DISTANCE_LINES > 0) };
        assert_eq!(
            PREFETCH_DISTANCE_BYTES,
            PREFETCH_DISTANCE_LINES * CACHE_LINE_SIZE
        );
    }
}
