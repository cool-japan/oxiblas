//! Memory management utilities for OxiBLAS.
//!
//! This module provides:
//! - Aligned memory allocation
//! - Stack-based temporary allocation (StackReq pattern)
//! - Cache-aware data layout utilities
//! - Prefetch hints for cache optimization
//! - Memory pool for temporary allocations
//! - Custom allocator support via the `Alloc` trait

use core::mem::{MaybeUninit, align_of, size_of};

use super::aligned_vec::AlignedVec;
use super::alloc::*;

// =============================================================================
// Arena - Bump allocator for temporary matrices
// =============================================================================

/// A bump allocator (arena) for temporary matrix allocations.
///
/// This arena provides extremely fast allocations by simply bumping a pointer.
/// All allocations are freed at once when the arena is reset or dropped.
///
/// # Use Case
///
/// BLAS operations like GEMM require temporary buffers for packing matrices.
/// Using an arena avoids repeated calls to the system allocator, which can
/// significantly improve performance for repeated operations.
///
/// # Example
///
/// ```
/// use oxiblas_core::memory::Arena;
///
/// let arena: Arena = Arena::with_capacity(1024 * 1024); // 1 MB arena
///
/// // Allocate multiple temporary buffers (concurrent access supported)
/// let buf1 = arena.alloc_vec::<f64>(1000);
/// let buf2 = arena.alloc_vec::<f32>(2000);
///
/// // Reset arena for reuse
/// arena.reset();
///
/// // Previous allocations are now invalid
/// let buf3 = arena.alloc_vec::<f64>(500);
/// ```
///
/// # Thread Safety
///
/// Arena is NOT thread-safe. Use one arena per thread or wrap in a mutex.
pub struct Arena<const ALIGN: usize = DEFAULT_ALIGN> {
    buffer: AlignedVec<u8, ALIGN>,
    /// Offset uses Cell for interior mutability, allowing multiple allocations
    offset: std::cell::Cell<usize>,
    high_water_mark: std::cell::Cell<usize>,
}

impl<const ALIGN: usize> Arena<ALIGN> {
    /// Creates a new arena with the given capacity in bytes.
    pub fn with_capacity(capacity: usize) -> Self {
        Arena {
            buffer: AlignedVec::zeros(capacity),
            offset: std::cell::Cell::new(0),
            high_water_mark: std::cell::Cell::new(0),
        }
    }

    /// Returns the total capacity of the arena in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the currently used bytes.
    #[inline]
    pub fn used(&self) -> usize {
        self.offset.get()
    }

    /// Returns the remaining capacity in bytes.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buffer.len().saturating_sub(self.offset.get())
    }

    /// Returns the high water mark (maximum bytes ever used).
    #[inline]
    pub fn high_water_mark(&self) -> usize {
        self.high_water_mark.get()
    }

    /// Resets the arena, invalidating all previous allocations.
    ///
    /// This is an O(1) operation that simply resets the internal pointer.
    /// Previously allocated memory becomes available for reuse.
    #[inline]
    pub fn reset(&self) {
        self.offset.set(0);
    }

    /// Allocates uninitialized memory for `count` elements of type `T`.
    ///
    /// Returns a mutable slice of `MaybeUninit<T>` that must be initialized
    /// before reading.
    ///
    /// # Panics
    ///
    /// Panics if the arena doesn't have enough remaining capacity.
    ///
    /// # Safety Note
    ///
    /// The returned slice is valid until `reset()` is called. Using slices
    /// after reset leads to undefined behavior.
    ///
    /// # Interior Mutability
    ///
    /// This function returns a mutable slice from a shared reference.
    /// This is safe because the arena owns the memory and uses interior
    /// mutability (Cell) to track allocations.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T>(&self, count: usize) -> &mut [MaybeUninit<T>] {
        let align = align_of::<T>().max(ALIGN);
        let current_offset = self.offset.get();
        let aligned_offset = round_up_pow2(current_offset, align);
        let size = count * size_of::<T>();
        let new_offset = aligned_offset + size;

        assert!(
            new_offset <= self.buffer.len(),
            "Arena overflow: requested {} bytes but only {} available (capacity: {})",
            size,
            self.remaining(),
            self.capacity()
        );

        // SAFETY: buffer is not reallocated until grow() is called with &mut self
        let ptr =
            unsafe { (self.buffer.as_ptr() as *mut u8).add(aligned_offset) as *mut MaybeUninit<T> };
        self.offset.set(new_offset);
        let hwm = self.high_water_mark.get();
        if new_offset > hwm {
            self.high_water_mark.set(new_offset);
        }

        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// Allocates zero-initialized memory for `count` elements of type `T`.
    ///
    /// Returns a mutable slice of properly initialized (zeroed) elements.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_zeroed<T: bytemuck::Zeroable>(&self, count: usize) -> &mut [T] {
        let slice = self.alloc::<T>(count);
        // Zero the memory
        unsafe {
            core::ptr::write_bytes(slice.as_mut_ptr() as *mut u8, 0, count * size_of::<T>());
            core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, count)
        }
    }

    /// Allocates an `ArenaVec` buffer from the arena.
    ///
    /// This is the recommended way to get multiple concurrent allocations
    /// from the same arena. Each ArenaVec can be used independently.
    pub fn alloc_vec<T: bytemuck::Zeroable>(&self, len: usize) -> ArenaVec<'_, T, ALIGN> {
        let slice = self.alloc_zeroed::<T>(len);
        ArenaVec {
            ptr: slice.as_mut_ptr(),
            len,
            _marker: core::marker::PhantomData,
        }
    }

    /// Tries to allocate memory, returning `None` if there's not enough space.
    #[allow(clippy::mut_from_ref)]
    pub fn try_alloc<T>(&self, count: usize) -> Option<&mut [MaybeUninit<T>]> {
        let align = align_of::<T>().max(ALIGN);
        let current_offset = self.offset.get();
        let aligned_offset = round_up_pow2(current_offset, align);
        let size = count * size_of::<T>();
        let new_offset = aligned_offset + size;

        if new_offset > self.buffer.len() {
            return None;
        }

        let ptr =
            unsafe { (self.buffer.as_ptr() as *mut u8).add(aligned_offset) as *mut MaybeUninit<T> };
        self.offset.set(new_offset);
        let hwm = self.high_water_mark.get();
        if new_offset > hwm {
            self.high_water_mark.set(new_offset);
        }

        Some(unsafe { core::slice::from_raw_parts_mut(ptr, count) })
    }

    /// Tries to allocate zeroed memory, returning `None` if there's not enough space.
    #[allow(clippy::mut_from_ref)]
    pub fn try_alloc_zeroed<T: bytemuck::Zeroable>(&self, count: usize) -> Option<&mut [T]> {
        let slice = self.try_alloc::<T>(count)?;
        unsafe {
            core::ptr::write_bytes(slice.as_mut_ptr() as *mut u8, 0, count * size_of::<T>());
            Some(core::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut T,
                count,
            ))
        }
    }

    /// Saves the current arena state for later restoration.
    ///
    /// This allows nested usage patterns where inner operations can
    /// allocate and then "free" their allocations by restoring the state.
    #[inline]
    pub fn save(&self) -> ArenaState {
        ArenaState {
            offset: self.offset.get(),
        }
    }

    /// Restores the arena to a previously saved state.
    ///
    /// All allocations made after the save point are invalidated.
    ///
    /// # Panics
    ///
    /// Panics if the saved offset is greater than the current offset
    /// (indicating the save point was corrupted or from a different arena).
    #[inline]
    pub fn restore(&self, state: ArenaState) {
        let current = self.offset.get();
        assert!(
            state.offset <= current,
            "Invalid arena state: saved offset {} > current offset {}",
            state.offset,
            current
        );
        self.offset.set(state.offset);
    }

    /// Grows the arena capacity to at least the specified size.
    ///
    /// Note: This creates a new buffer and copies existing data.
    /// Use sparingly as it's an expensive operation.
    ///
    /// # Warning
    ///
    /// This invalidates all existing allocations from this arena.
    pub fn grow(&mut self, min_capacity: usize) {
        if min_capacity <= self.buffer.len() {
            return;
        }

        let new_capacity = min_capacity.max(self.buffer.len() * 2);
        let mut new_buffer: AlignedVec<u8, ALIGN> = AlignedVec::zeros(new_capacity);
        let current_offset = self.offset.get();

        // Copy existing data
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.buffer.as_ptr(),
                new_buffer.as_mut_ptr(),
                current_offset,
            );
        }

        self.buffer = new_buffer;
    }
}

impl Default for Arena<DEFAULT_ALIGN> {
    fn default() -> Self {
        // Default 16 MB arena
        Self::with_capacity(16 * 1024 * 1024)
    }
}

/// Saved arena state for nested usage.
#[derive(Debug, Clone, Copy)]
pub struct ArenaState {
    offset: usize,
}

/// A vector-like view into arena memory.
///
/// This provides a familiar interface for working with arena-allocated arrays.
pub struct ArenaVec<'a, T, const ALIGN: usize = DEFAULT_ALIGN> {
    ptr: *mut T,
    len: usize,
    _marker: core::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const ALIGN: usize> ArenaVec<'a, T, ALIGN> {
    /// Returns the length of the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a mutable pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns a slice of the vector.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice of the vector.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<'a, T, const ALIGN: usize> core::ops::Deref for ArenaVec<'a, T, ALIGN> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T, const ALIGN: usize> core::ops::DerefMut for ArenaVec<'a, T, ALIGN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, T, const ALIGN: usize> core::ops::Index<usize> for ArenaVec<'a, T, ALIGN> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<'a, T, const ALIGN: usize> core::ops::IndexMut<usize> for ArenaVec<'a, T, ALIGN> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

// =============================================================================
// Thread-local arena for BLAS operations
// =============================================================================

/// Gets or creates a thread-local arena for BLAS temporary allocations.
///
/// This function provides a convenient way to access a reusable arena
/// without manually managing arena lifetime. The arena is automatically
/// reset before each use.
///
/// # Example
///
/// ```
/// use oxiblas_core::memory::with_blas_arena;
///
/// // Use the thread-local arena for temporary allocations
/// with_blas_arena(|arena| {
///     let buf: &mut [f64] = arena.alloc_zeroed(1000);
///     buf[0] = 42.0;
///     // Use buffer...
/// });
///
/// // Arena is reset, memory can be reused next time
/// ```
pub fn with_blas_arena<F, R>(f: F) -> R
where
    F: FnOnce(&mut Arena) -> R,
{
    thread_local! {
        static ARENA: std::cell::RefCell<Arena> = std::cell::RefCell::new(Arena::with_capacity(32 * 1024 * 1024)); // 32 MB
    }

    ARENA.with(|cell| {
        let mut arena = cell.borrow_mut();
        arena.reset();
        f(&mut arena)
    })
}

/// Configuration for BLAS arena sizing.
#[derive(Debug, Clone, Copy)]
pub struct BlasArenaConfig {
    /// Default arena capacity in bytes.
    pub capacity: usize,
    /// Whether to grow the arena automatically when needed.
    pub auto_grow: bool,
    /// Maximum arena size if auto_grow is enabled.
    pub max_capacity: usize,
}

impl Default for BlasArenaConfig {
    fn default() -> Self {
        BlasArenaConfig {
            capacity: 32 * 1024 * 1024, // 32 MB
            auto_grow: true,
            max_capacity: 512 * 1024 * 1024, // 512 MB max
        }
    }
}

impl BlasArenaConfig {
    /// Creates a configuration for small matrices.
    pub const fn small() -> Self {
        BlasArenaConfig {
            capacity: 4 * 1024 * 1024,
            auto_grow: true,
            max_capacity: 32 * 1024 * 1024,
        }
    }

    /// Creates a configuration for large matrices.
    pub const fn large() -> Self {
        BlasArenaConfig {
            capacity: 128 * 1024 * 1024,
            auto_grow: true,
            max_capacity: 1024 * 1024 * 1024,
        }
    }

    /// Estimates required arena size for GEMM.
    ///
    /// # Arguments
    ///
    /// * `m` - Rows of A and C
    /// * `k` - Columns of A, rows of B
    /// * `n` - Columns of B and C
    /// * `elem_size` - Size of each element in bytes
    pub const fn gemm_arena_size(m: usize, k: usize, n: usize, elem_size: usize) -> usize {
        // GEMM needs pack_a (MC × KC) and pack_b (KC × NC)
        // Use conservative estimates
        let mc = if m < 512 { m } else { 512 };
        let kc = if k < 256 { k } else { 256 };
        let nc = if n < 2048 { n } else { 2048 };

        let pack_a_size = mc * kc * elem_size;
        let pack_b_size = kc * nc * elem_size;

        // Add 20% overhead for alignment
        (pack_a_size + pack_b_size) * 12 / 10
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena: Arena = Arena::with_capacity(1024);

        // Allocate and use the slice
        {
            let slice: &mut [f64] = arena.alloc_zeroed(10);
            assert_eq!(slice.len(), 10);
            slice[0] = 1.0;
            slice[9] = 9.0;
            assert_eq!(slice[0], 1.0);
            assert_eq!(slice[9], 9.0);
        }
        // Now we can access arena again
        assert_eq!(arena.used(), 80); // 10 * 8 bytes
    }

    #[test]
    fn test_arena_reset() {
        let arena: Arena = Arena::with_capacity(1024);

        {
            let _slice1: &mut [f64] = arena.alloc_zeroed(100);
        }
        assert_eq!(arena.used(), 800);

        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.high_water_mark(), 800);

        // Can reuse the memory
        {
            let _slice2: &mut [f64] = arena.alloc_zeroed(50);
        }
        assert_eq!(arena.used(), 400);
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let arena: Arena = Arena::with_capacity(4096);

        // Use ArenaVec for multiple concurrent allocations
        let mut buf1 = arena.alloc_vec::<f64>(10);
        let mut buf2 = arena.alloc_vec::<f32>(20);
        let mut buf3 = arena.alloc_vec::<u8>(100);

        buf1[0] = 1.0;
        buf2[0] = 2.0;
        buf3[0] = 3;

        assert_eq!(buf1[0], 1.0);
        assert_eq!(buf2[0], 2.0);
        assert_eq!(buf3[0], 3);
    }

    #[test]
    fn test_arena_save_restore() {
        let arena: Arena = Arena::with_capacity(4096);

        {
            let _buf1: &mut [f64] = arena.alloc_zeroed(10);
        }
        let saved_offset = arena.used();
        let state = arena.save();
        assert!(saved_offset > 0);

        {
            let _buf2: &mut [f64] = arena.alloc_zeroed(10);
        }
        let after_second = arena.used();
        assert!(after_second > saved_offset);

        arena.restore(state);
        assert_eq!(arena.used(), saved_offset);
    }

    #[test]
    fn test_arena_try_alloc() {
        let arena: Arena = Arena::with_capacity(100);

        // Should succeed
        {
            let result: Option<&mut [f64]> = arena.try_alloc_zeroed(10);
            assert!(result.is_some());
        }

        // Should fail - not enough space
        let result: Option<&mut [f64]> = arena.try_alloc_zeroed(100);
        assert!(result.is_none());
    }

    #[test]
    fn test_arena_vec() {
        let arena: Arena = Arena::with_capacity(1024);

        let mut vec = arena.alloc_vec::<f64>(10);
        assert_eq!(vec.len(), 10);
        assert!(!vec.is_empty());

        vec[0] = 1.0;
        vec[9] = 9.0;
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[9], 9.0);
        assert_eq!(vec.as_slice()[0], 1.0);
    }

    #[test]
    fn test_arena_alignment() {
        let arena: Arena<128> = Arena::with_capacity(4096);

        let buf: &mut [f64] = arena.alloc_zeroed(10);
        let ptr = buf.as_ptr() as usize;

        // Should be aligned to 128 bytes
        assert_eq!(ptr % 128, 0);
    }

    #[test]
    fn test_arena_grow() {
        let mut arena: Arena = Arena::with_capacity(100);

        let _buf1: &mut [f64] = arena.alloc_zeroed(10);
        assert_eq!(arena.capacity(), 100);

        arena.grow(500);
        assert!(arena.capacity() >= 500);
        assert_eq!(arena.used(), 80); // Previous allocation preserved
    }

    #[test]
    fn test_with_blas_arena() {
        with_blas_arena(|arena| {
            let buf: &mut [f64] = arena.alloc_zeroed(1000);
            buf[0] = 42.0;
            assert_eq!(buf[0], 42.0);
        });

        // Arena should be reset on next use
        with_blas_arena(|arena| {
            assert_eq!(arena.used(), 0);
        });
    }

    #[test]
    fn test_blas_arena_config() {
        let config = BlasArenaConfig::default();
        assert_eq!(config.capacity, 32 * 1024 * 1024);
        assert!(config.auto_grow);

        let small = BlasArenaConfig::small();
        assert_eq!(small.capacity, 4 * 1024 * 1024);

        let large = BlasArenaConfig::large();
        assert_eq!(large.capacity, 128 * 1024 * 1024);
    }

    #[test]
    fn test_gemm_arena_size() {
        // 1024 x 1024 x 1024 GEMM with f64
        let size = BlasArenaConfig::gemm_arena_size(1024, 1024, 1024, 8);
        assert!(size > 0);
        // mc = min(1024, 512) = 512
        // kc = min(1024, 256) = 256
        // nc = min(1024, 2048) = 1024  (not 2048!)
        // (512 * 256 + 256 * 1024) * 8 * 1.2
        let expected = ((512 * 256 + 256 * 1024) * 8) * 12 / 10;
        assert_eq!(size, expected);
    }

    #[test]
    #[should_panic(expected = "Arena overflow")]
    fn test_arena_overflow() {
        let arena: Arena = Arena::with_capacity(100);
        let _buf: &mut [f64] = arena.alloc_zeroed(100); // Needs 800 bytes
    }
}
