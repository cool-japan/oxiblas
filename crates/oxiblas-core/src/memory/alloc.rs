//! Memory management utilities for OxiBLAS.
//!
//! This module provides:
//! - Aligned memory allocation
//! - Stack-based temporary allocation (StackReq pattern)
//! - Cache-aware data layout utilities
//! - Prefetch hints for cache optimization
//! - Memory pool for temporary allocations
//! - Custom allocator support via the `Alloc` trait

use core::alloc::Layout;
use core::mem::size_of;

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, alloc_zeroed, dealloc};
#[cfg(feature = "std")]
use std::alloc::{alloc, alloc_zeroed, dealloc};

// =============================================================================
// Custom Allocator Trait
// =============================================================================

/// A stable-Rust compatible allocator trait.
///
/// This trait provides a simplified interface for custom memory allocators,
/// similar to `std::alloc::Allocator` but available on stable Rust.
///
/// # Safety
///
/// Implementations must ensure that:
/// - `allocate` returns a valid pointer or null on failure
/// - `deallocate` is called with the same layout used for allocation
/// - Memory is properly aligned as specified by the layout
pub unsafe trait Alloc: Clone {
    /// Allocates memory with the specified layout.
    ///
    /// Returns a pointer to the allocated memory, or null on failure.
    fn allocate(&self, layout: Layout) -> *mut u8;

    /// Allocates zero-initialized memory with the specified layout.
    ///
    /// Returns a pointer to the allocated memory, or null on failure.
    fn allocate_zeroed(&self, layout: Layout) -> *mut u8;

    /// Deallocates memory previously allocated with `allocate`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was previously returned by `allocate` or `allocate_zeroed`
    /// - `layout` matches the layout used for allocation
    /// - The memory has not already been deallocated
    unsafe fn deallocate(&self, ptr: *mut u8, layout: Layout);
}

/// The global allocator.
///
/// This is the default allocator used by `AlignedVec` and other types.
/// It delegates to the Rust global allocator (`std::alloc::alloc`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Global;

// SAFETY: Global allocator delegates to the Rust global allocator
// which is guaranteed to be safe.
unsafe impl Alloc for Global {
    #[inline]
    fn allocate(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            // Return a properly aligned dangling pointer for ZSTs
            return layout.align() as *mut u8;
        }
        unsafe { alloc(layout) }
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            return layout.align() as *mut u8;
        }
        unsafe { alloc_zeroed(layout) }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() != 0 {
            dealloc(ptr, layout);
        }
    }
}

// =============================================================================
// Prefetch hints
// =============================================================================

/// Prefetch locality hint levels.
///
/// These correspond to the x86 prefetch instruction locality hints.
/// Higher locality means the data is expected to be accessed more times
/// and should be kept in closer cache levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLocality {
    /// Non-temporal: Data will be accessed once and then not reused.
    /// Minimizes cache pollution.
    NonTemporal,
    /// Low locality: Data will be accessed a few times.
    /// Kept in L3 cache.
    Low,
    /// Medium locality: Data will be accessed several times.
    /// Kept in L2 cache.
    Medium,
    /// High locality: Data will be accessed many times.
    /// Kept in L1 cache.
    High,
}

/// Prefetches data for reading.
///
/// This is a hint to the processor that data at the given address will be
/// read in the near future. The processor may choose to load the data into
/// cache ahead of time.
///
/// # Arguments
/// * `ptr` - Pointer to the data to prefetch
/// * `locality` - Hint about how often the data will be reused
///
/// # Safety
/// The pointer doesn't need to be valid - invalid prefetches are simply ignored.
/// However, prefetching to invalid addresses may have performance implications.
#[inline]
pub fn prefetch_read<T>(ptr: *const T, locality: PrefetchLocality) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            match locality {
                PrefetchLocality::NonTemporal => _mm_prefetch(ptr.cast(), _MM_HINT_NTA),
                PrefetchLocality::Low => _mm_prefetch(ptr.cast(), _MM_HINT_T2),
                PrefetchLocality::Medium => _mm_prefetch(ptr.cast(), _MM_HINT_T1),
                PrefetchLocality::High => _mm_prefetch(ptr.cast(), _MM_HINT_T0),
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
        // No prefetch support - silently ignore
        let _ = (ptr, locality);
    }
}

/// Prefetches data for writing.
///
/// This is a hint to the processor that data at the given address will be
/// written in the near future. This is useful for write-allocate cache policies.
///
/// # Arguments
/// * `ptr` - Pointer to the data to prefetch
/// * `locality` - Hint about how often the data will be reused
///
/// # Safety
/// The pointer doesn't need to be valid - invalid prefetches are simply ignored.
#[inline]
pub fn prefetch_write<T>(ptr: *mut T, locality: PrefetchLocality) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            // x86 doesn't have separate prefetch-for-write in SSE/AVX
            // PREFETCHW is available with 3DNow! or PREFETCHWT1 with newer CPUs
            // Fall back to regular prefetch
            match locality {
                PrefetchLocality::NonTemporal => _mm_prefetch(ptr.cast(), _MM_HINT_NTA),
                PrefetchLocality::Low => _mm_prefetch(ptr.cast(), _MM_HINT_T2),
                PrefetchLocality::Medium => _mm_prefetch(ptr.cast(), _MM_HINT_T1),
                PrefetchLocality::High => _mm_prefetch(ptr.cast(), _MM_HINT_T0),
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

/// Prefetches a range of memory for reading.
///
/// This prefetches multiple cache lines starting at `ptr` and covering
/// `count` elements of type `T`.
///
/// # Arguments
/// * `ptr` - Pointer to the start of the data
/// * `count` - Number of elements to prefetch
/// * `locality` - Hint about data reuse
#[inline]
pub fn prefetch_read_range<T>(ptr: *const T, count: usize, locality: PrefetchLocality) {
    let bytes = count * size_of::<T>();
    let num_lines = bytes.div_ceil(CACHE_LINE_SIZE);

    for i in 0..num_lines {
        let offset = i * CACHE_LINE_SIZE;
        prefetch_read(unsafe { (ptr as *const u8).add(offset) }, locality);
    }
}

/// Prefetches a range of memory for writing.
#[inline]
pub fn prefetch_write_range<T>(ptr: *mut T, count: usize, locality: PrefetchLocality) {
    let bytes = count * size_of::<T>();
    let num_lines = bytes.div_ceil(CACHE_LINE_SIZE);

    for i in 0..num_lines {
        let offset = i * CACHE_LINE_SIZE;
        prefetch_write(unsafe { (ptr as *mut u8).add(offset) }, locality);
    }
}

/// Prefetch distance calculator for streaming access patterns.
///
/// This calculates the optimal prefetch distance (in elements) based on
/// memory bandwidth and latency estimates.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchDistance {
    /// Number of cache lines to prefetch ahead
    pub lines_ahead: usize,
}

impl Default for PrefetchDistance {
    fn default() -> Self {
        // Default: prefetch 8 cache lines ahead
        // This works well for most streaming workloads
        Self { lines_ahead: 8 }
    }
}

impl PrefetchDistance {
    /// Creates a new prefetch distance calculator.
    pub const fn new(lines_ahead: usize) -> Self {
        Self { lines_ahead }
    }

    /// Calculates the prefetch offset in bytes.
    #[inline]
    pub const fn offset_bytes(&self) -> usize {
        self.lines_ahead * CACHE_LINE_SIZE
    }

    /// Calculates the prefetch offset in elements.
    #[inline]
    pub const fn offset_elements<T>(&self) -> usize {
        self.offset_bytes() / size_of::<T>()
    }
}

/// Default alignment for SIMD operations (cache line size).
///
/// Apple Silicon (M1/M2/M3) uses 128-byte cache lines for optimal performance.
/// x86_64 and most other architectures use 64-byte cache lines.
#[cfg(target_arch = "aarch64")]
pub const CACHE_LINE_SIZE: usize = 128;

/// Cache line size for the target architecture.
///
/// x86_64 and most other architectures use 64-byte cache lines.
#[cfg(not(target_arch = "aarch64"))]
pub const CACHE_LINE_SIZE: usize = 64;

/// Default alignment for matrices.
///
/// This is set to match the cache line size for optimal memory access patterns.
/// - Apple Silicon (aarch64): 128 bytes
/// - x86_64 and others: 64 bytes
pub const DEFAULT_ALIGN: usize = CACHE_LINE_SIZE;

/// Computes the aligned size for a given element count and type.
#[inline]
pub const fn aligned_size<T>(count: usize, align: usize) -> usize {
    let size = count * size_of::<T>();
    (size + align - 1) & !(align - 1)
}

/// Computes the number of elements that fit in a given number of bytes with alignment.
#[inline]
pub const fn elements_per_aligned_bytes<T>(bytes: usize) -> usize {
    bytes / size_of::<T>()
}

/// Rounds up to the next multiple of a power of 2.
#[inline]
pub const fn round_up_pow2(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Checks if a pointer is aligned to the given alignment.
#[inline]
pub fn is_aligned<T>(ptr: *const T, align: usize) -> bool {
    (ptr as usize) % align == 0
}
