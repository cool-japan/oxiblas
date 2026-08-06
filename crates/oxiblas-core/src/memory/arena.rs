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

/// A bump allocator (arena) for temporary allocations.
///
/// This arena provides extremely fast allocations by simply bumping a pointer.
/// All allocations are freed at once when the arena is reset or dropped.
///
/// # Use Case
///
/// It is a general-purpose scratch allocator for short-lived temporary buffers
/// (for example, the packing buffers used by GEMM-style kernels): bumping a
/// pointer avoids repeated round-trips through the system allocator.
///
/// Note: the built-in BLAS kernels currently manage their own scratch space and
/// do **not** yet route allocations through this arena — it is provided as a
/// reusable building block. Grow-on-demand behavior driven by
/// [`BlasArenaConfig`] is available via [`Arena::from_config`] and
/// [`Arena::reserve`].
///
/// # Aliasing / soundness
///
/// [`Arena::alloc`] hands out a `&mut` slice from a shared `&self` borrow. This
/// is sound because every allocation returns a *disjoint* region and the
/// returned slice keeps the arena immutably borrowed for its whole lifetime.
/// The operations that could make regions overlap again — [`Arena::reset`],
/// [`Arena::restore`], and [`Arena::grow`] — take `&mut self`, so the borrow
/// checker guarantees no outstanding allocation is alive when they run. This is
/// the same soundness model used by other bump allocators (e.g. `bumpalo`).
///
/// # Example
///
/// ```
/// use oxiblas_core::memory::Arena;
///
/// let mut arena: Arena = Arena::with_capacity(1024 * 1024); // 1 MB arena
///
/// // Allocate multiple temporary buffers from the same shared borrow.
/// {
///     let buf1 = arena.alloc_vec::<f64>(1000);
///     let buf2 = arena.alloc_vec::<f32>(2000);
///     assert_eq!(buf1.len() + buf2.len(), 3000);
/// } // borrows end here
///
/// // `reset` takes `&mut self`, so all outstanding allocations must be dead.
/// arena.reset();
///
/// // The memory is reused for new allocations.
/// let buf3 = arena.alloc_vec::<f64>(500);
/// assert_eq!(buf3.len(), 500);
/// ```
///
/// # Thread Safety
///
/// Arena is NOT thread-safe. Use one arena per thread or wrap in a mutex.
pub struct Arena<const ALIGN: usize = DEFAULT_ALIGN> {
    buffer: AlignedVec<u8, ALIGN>,
    /// Bump offset. Uses `Cell` for interior mutability so multiple disjoint
    /// allocations can be handed out from a shared `&self` borrow.
    offset: core::cell::Cell<usize>,
    high_water_mark: core::cell::Cell<usize>,
    /// When set, [`Arena::reserve`] may reallocate the backing buffer (up to
    /// `max_capacity`) instead of reporting exhaustion.
    auto_grow: bool,
    /// Upper bound on the backing-buffer size honored by [`Arena::reserve`]
    /// when `auto_grow` is enabled.
    max_capacity: usize,
}

/// Rounds `value` up to the next multiple of `align` (which must be a power of
/// two), returning `None` if the rounding would overflow `usize`.
///
/// This is the overflow-checked counterpart of [`round_up_pow2`]; the arena
/// uses it so that an astronomically large allocation request cannot wrap the
/// offset past the capacity check and produce an out-of-bounds pointer.
#[inline]
fn round_up_pow2_checked(value: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    // `align >= 1`, so `align - 1` never underflows; the mask clears low bits.
    Some(value.checked_add(align - 1)? & !(align - 1))
}

impl<const ALIGN: usize> Arena<ALIGN> {
    /// Creates a new arena with the given capacity in bytes.
    ///
    /// The arena is fixed-size: [`Arena::reserve`] reports exhaustion rather
    /// than silently reallocating (which would invalidate outstanding borrows).
    /// Use [`Arena::from_config`] to opt into grow-on-demand behavior.
    pub fn with_capacity(capacity: usize) -> Self {
        Arena {
            buffer: AlignedVec::zeros(capacity),
            offset: core::cell::Cell::new(0),
            high_water_mark: core::cell::Cell::new(0),
            auto_grow: false,
            max_capacity: capacity,
        }
    }

    /// Creates an arena from a [`BlasArenaConfig`], honoring its `capacity`,
    /// `auto_grow`, and `max_capacity` settings.
    ///
    /// The `auto_grow` and `max_capacity` fields take effect through
    /// [`Arena::reserve`]: with `auto_grow` enabled, `reserve` grows the backing
    /// buffer on demand up to `max_capacity`.
    pub fn from_config(config: BlasArenaConfig) -> Self {
        // `max_capacity` must be at least the initial capacity, otherwise the
        // arena would be born already over its own growth ceiling.
        let max_capacity = config.max_capacity.max(config.capacity);
        Arena {
            buffer: AlignedVec::zeros(config.capacity),
            offset: core::cell::Cell::new(0),
            high_water_mark: core::cell::Cell::new(0),
            auto_grow: config.auto_grow,
            max_capacity,
        }
    }

    /// Computes `(aligned_start_offset, end_offset)` for an allocation of
    /// `bytes` with alignment `align`, or `None` on `usize` overflow.
    ///
    /// # Why address- vs. offset-based alignment
    ///
    /// The backing buffer is only guaranteed to be aligned to `ALIGN`. When a
    /// type's alignment does not exceed `ALIGN`, aligning the *offset* is
    /// sufficient — the base is `ALIGN`-aligned, hence also `align`-aligned, so
    /// `base + round_up(offset, align)` is `align`-aligned. When `align > ALIGN`
    /// the base address itself may not be `align`-aligned, so aligning the
    /// offset is not enough; we instead round the absolute *address*
    /// (`base + offset`) up to `align` and translate it back into a buffer
    /// offset. This keeps the returned pointer `align`-aligned in every case
    /// without rejecting legitimate over-aligned requests.
    #[inline]
    fn compute_bounds(&self, align: usize, bytes: usize) -> Option<(usize, usize)> {
        let current_offset = self.offset.get();
        let aligned_offset = if align <= ALIGN {
            round_up_pow2_checked(current_offset, align)?
        } else {
            let base = self.buffer.as_ptr() as usize;
            let current_addr = base.checked_add(current_offset)?;
            let aligned_addr = round_up_pow2_checked(current_addr, align)?;
            // `aligned_addr >= current_addr >= base`, so this cannot underflow.
            aligned_addr - base
        };
        let new_offset = aligned_offset.checked_add(bytes)?;
        Some((aligned_offset, new_offset))
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
    ///
    /// Takes `&mut self` so that the borrow checker forbids resetting while any
    /// allocation from [`Arena::alloc`] (which borrows `&self`) is still live —
    /// without this exclusion, reset-then-alloc could hand out a second `&mut`
    /// aliasing a still-borrowed region (undefined behavior).
    ///
    /// The following must **not** compile: `reset` while `buf` is still borrowed
    /// is rejected, which is exactly what makes the arena sound.
    ///
    /// ```compile_fail,E0502
    /// use oxiblas_core::memory::Arena;
    /// let mut arena: Arena = Arena::with_capacity(1024);
    /// let buf = arena.alloc_vec::<f64>(10);
    /// arena.reset();     // cannot borrow `arena` as mutable ...
    /// let _ = buf[0];    // ... because `buf` still borrows it here
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.offset.set(0);
    }

    /// Allocates uninitialized memory for `count` elements of type `T`.
    ///
    /// Returns a mutable slice of `MaybeUninit<T>` that must be initialized
    /// before reading. The returned pointer is aligned to
    /// `align_of::<T>().max(ALIGN)`, even when `align_of::<T>()` exceeds the
    /// arena's fixed `ALIGN`.
    ///
    /// # Panics
    ///
    /// Panics if the arena doesn't have enough remaining capacity, or if the
    /// requested size (`count * size_of::<T>()`) or the resulting offset would
    /// overflow `usize`. Use [`Arena::try_alloc`] for a non-panicking variant.
    ///
    /// # Safety Note
    ///
    /// The returned slice is valid until [`Arena::reset`], [`Arena::restore`],
    /// or [`Arena::grow`] is called; those take `&mut self`, so the borrow
    /// checker prevents use-after-reset from safe code.
    ///
    /// # Interior Mutability
    ///
    /// This function returns a mutable slice from a shared reference. This is
    /// sound because the arena hands out disjoint regions and tracks the bump
    /// offset with `Cell`; see the type-level "Aliasing / soundness" section.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T>(&self, count: usize) -> &mut [MaybeUninit<T>] {
        let align = align_of::<T>().max(ALIGN);
        let bytes = count
            .checked_mul(size_of::<T>())
            .expect("Arena overflow: allocation size (count * size_of::<T>()) overflows usize");
        let (aligned_offset, new_offset) = self
            .compute_bounds(align, bytes)
            .expect("Arena overflow: allocation offset overflows usize");

        assert!(
            new_offset <= self.buffer.len(),
            "Arena overflow: requested {} bytes but only {} available (capacity: {})",
            bytes,
            self.remaining(),
            self.capacity()
        );

        // SAFETY: `aligned_offset <= new_offset <= buffer.len()` (checked above),
        // so `base + aligned_offset` is in-bounds. The buffer is never
        // reallocated except through `grow` (`&mut self`), so this pointer stays
        // valid for the returned slice's lifetime. `offset` only advances, so
        // successive allocations return non-overlapping regions.
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

    /// Tries to allocate memory, returning `None` if there's not enough space
    /// or if the size/offset arithmetic would overflow `usize`.
    #[allow(clippy::mut_from_ref)]
    pub fn try_alloc<T>(&self, count: usize) -> Option<&mut [MaybeUninit<T>]> {
        let align = align_of::<T>().max(ALIGN);
        let bytes = count.checked_mul(size_of::<T>())?;
        let (aligned_offset, new_offset) = self.compute_bounds(align, bytes)?;

        if new_offset > self.buffer.len() {
            return None;
        }

        // SAFETY: identical invariants to `alloc` — `aligned_offset` is
        // in-bounds and the returned region is disjoint from prior allocations.
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
    /// Takes `&mut self` for the same reason as [`Arena::reset`]: rewinding the
    /// bump pointer while an allocation from that range is still borrowed would
    /// allow a later `alloc` to alias it.
    ///
    /// # Panics
    ///
    /// Panics if the saved offset is greater than the current offset
    /// (indicating the save point was corrupted or from a different arena).
    #[inline]
    pub fn restore(&mut self, state: ArenaState) {
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
    /// This invalidates all existing allocations from this arena. Because it
    /// takes `&mut self`, the borrow checker guarantees none are outstanding.
    pub fn grow(&mut self, min_capacity: usize) {
        if min_capacity <= self.buffer.len() {
            return;
        }

        let new_capacity = min_capacity.max(self.buffer.len().saturating_mul(2));
        self.grow_to(new_capacity);
    }

    /// Reallocates the backing buffer to exactly `new_capacity` bytes (a no-op
    /// if that is not larger than the current capacity), preserving the bytes
    /// already used.
    fn grow_to(&mut self, new_capacity: usize) {
        if new_capacity <= self.buffer.len() {
            return;
        }

        let mut new_buffer: AlignedVec<u8, ALIGN> = AlignedVec::zeros(new_capacity);
        let current_offset = self.offset.get();

        // SAFETY: `current_offset <= old capacity < new_capacity`, so both
        // buffers are at least `current_offset` bytes long, and they do not
        // overlap (`new_buffer` was just allocated).
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.buffer.as_ptr(),
                new_buffer.as_mut_ptr(),
                current_offset,
            );
        }

        self.buffer = new_buffer;
    }

    /// Ensures the arena can serve at least `additional_bytes` more, growing the
    /// backing buffer on demand when `auto_grow` is enabled.
    ///
    /// This is the mechanism through which [`BlasArenaConfig::auto_grow`] and
    /// [`BlasArenaConfig::max_capacity`] take effect. Construct an arena with
    /// [`Arena::from_config`], then call `reserve` before a batch of
    /// allocations. Growth reallocates the backing buffer (invalidating every
    /// outstanding allocation), which is why this takes `&mut self`.
    ///
    /// # Errors
    ///
    /// - [`ArenaError::CapacityOverflow`] if `used() + additional_bytes`
    ///   overflows `usize`.
    /// - [`ArenaError::GrowthDisabled`] if more room is needed but `auto_grow`
    ///   is disabled (the default for [`Arena::with_capacity`]).
    /// - [`ArenaError::MaxCapacityExceeded`] if the required capacity exceeds
    ///   `max_capacity`.
    pub fn reserve(&mut self, additional_bytes: usize) -> Result<(), ArenaError> {
        let needed = self
            .offset
            .get()
            .checked_add(additional_bytes)
            .ok_or(ArenaError::CapacityOverflow)?;

        if needed <= self.buffer.len() {
            return Ok(());
        }
        if !self.auto_grow {
            return Err(ArenaError::GrowthDisabled {
                needed,
                capacity: self.buffer.len(),
            });
        }
        if needed > self.max_capacity {
            return Err(ArenaError::MaxCapacityExceeded {
                needed,
                max_capacity: self.max_capacity,
            });
        }

        // Exponential growth to amortize reallocation, capped at `max_capacity`.
        let doubled = self.buffer.len().saturating_mul(2);
        let target = needed.max(doubled).min(self.max_capacity);
        self.grow_to(target);
        Ok(())
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

/// Errors that can occur when growing an [`Arena`] via [`Arena::reserve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArenaError {
    /// Growth was required, but `auto_grow` is disabled for this arena.
    GrowthDisabled {
        /// Bytes required (current offset plus the requested amount).
        needed: usize,
        /// Current backing-buffer capacity in bytes.
        capacity: usize,
    },
    /// Growth would exceed the arena's configured `max_capacity`.
    MaxCapacityExceeded {
        /// Bytes required.
        needed: usize,
        /// Configured maximum capacity in bytes.
        max_capacity: usize,
    },
    /// The requested size or the resulting offset overflowed `usize`.
    CapacityOverflow,
}

impl core::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ArenaError::GrowthDisabled { needed, capacity } => write!(
                f,
                "arena needs {needed} bytes but capacity is {capacity} and auto_grow is disabled"
            ),
            ArenaError::MaxCapacityExceeded {
                needed,
                max_capacity,
            } => write!(
                f,
                "arena needs {needed} bytes, exceeding the configured max_capacity of {max_capacity}"
            ),
            ArenaError::CapacityOverflow => {
                write!(f, "arena size/offset arithmetic overflowed usize")
            }
        }
    }
}

impl core::error::Error for ArenaError {}

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
#[cfg(feature = "std")]
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

/// Configuration for sizing an [`Arena`] used for BLAS scratch space.
///
/// Turn a config into an arena with [`Arena::from_config`]. The `auto_grow` and
/// `max_capacity` fields are consumed by [`Arena::reserve`], which grows the
/// backing buffer on demand (up to `max_capacity`) when `auto_grow` is set.
#[derive(Debug, Clone, Copy)]
pub struct BlasArenaConfig {
    /// Initial arena capacity in bytes (see [`Arena::from_config`]).
    pub capacity: usize,
    /// Whether [`Arena::reserve`] may grow the backing buffer on demand.
    pub auto_grow: bool,
    /// Upper bound on the backing-buffer size honored by [`Arena::reserve`]
    /// when `auto_grow` is enabled.
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
        let mut arena: Arena = Arena::with_capacity(1024);

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
        let mut arena: Arena = Arena::with_capacity(4096);

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

    #[cfg(feature = "std")]
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

    // A type whose alignment (64) exceeds the arena's ALIGN (8), used to
    // exercise the address-based realignment path (finding #2).
    #[repr(align(64))]
    #[allow(dead_code)]
    struct Over64 {
        payload: [u8; 8],
    }

    #[test]
    fn test_arena_overaligned_type() {
        // ALIGN = 8 is smaller than align_of::<Over64>() = 64.
        let arena: Arena<8> = Arena::with_capacity(4096);

        // Bump the offset to a position that is very unlikely to already be
        // 64-aligned, so the realignment logic is actually exercised.
        let _pad = arena.alloc::<u8>(1);
        let buf = arena.alloc::<Over64>(3);
        let addr = buf.as_ptr() as usize;

        assert_eq!(
            addr % 64,
            0,
            "over-aligned allocation must honor align_of::<T>() even when it exceeds ALIGN"
        );
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_arena_try_alloc_size_overflow() {
        let arena: Arena = Arena::with_capacity(1024);

        // count * size_of::<f64>() overflows usize; must not wrap into a small
        // allocation that then passes the capacity check.
        let huge = usize::MAX / core::mem::size_of::<f64>() + 1;
        let result = arena.try_alloc::<f64>(huge);
        assert!(
            result.is_none(),
            "overflowing size must return None, not a wrapped allocation"
        );
        // The failed attempt must leave the arena untouched.
        assert_eq!(arena.used(), 0);
    }

    #[test]
    #[should_panic(expected = "Arena overflow")]
    fn test_arena_alloc_size_overflow_panics() {
        let arena: Arena = Arena::with_capacity(1024);
        let huge = usize::MAX / core::mem::size_of::<f64>() + 1;
        let _buf = arena.alloc::<f64>(huge);
    }

    #[test]
    fn test_arena_from_config_auto_grow() {
        let config = BlasArenaConfig {
            capacity: 64,
            auto_grow: true,
            max_capacity: 1024,
        };
        let mut arena: Arena = Arena::from_config(config);
        assert_eq!(arena.capacity(), 64);

        // Reserve beyond the initial capacity: should grow up to max_capacity.
        arena
            .reserve(500)
            .expect("reserve within max_capacity should succeed");
        assert!(arena.capacity() >= 500);
        assert!(arena.capacity() <= 1024);

        // Allocations up to the reserved amount now succeed.
        let buf: &mut [f64] = arena.alloc_zeroed(60); // 480 bytes
        assert_eq!(buf.len(), 60);
    }

    #[test]
    fn test_arena_reserve_growth_disabled() {
        // `with_capacity` arenas have auto_grow == false.
        let mut arena: Arena = Arena::with_capacity(64);

        // Within capacity: Ok, and no growth happens.
        assert!(arena.reserve(64).is_ok());
        assert_eq!(arena.capacity(), 64);

        // Beyond capacity with auto_grow off: GrowthDisabled.
        match arena.reserve(65) {
            Err(ArenaError::GrowthDisabled { needed, capacity }) => {
                assert_eq!(needed, 65);
                assert_eq!(capacity, 64);
            }
            other => panic!("expected GrowthDisabled, got {other:?}"),
        }
        assert_eq!(arena.capacity(), 64);
    }

    #[test]
    fn test_arena_reserve_max_capacity_exceeded() {
        let mut arena: Arena = Arena::from_config(BlasArenaConfig {
            capacity: 64,
            auto_grow: true,
            max_capacity: 128,
        });

        match arena.reserve(200) {
            Err(ArenaError::MaxCapacityExceeded {
                needed,
                max_capacity,
            }) => {
                assert_eq!(needed, 200);
                assert_eq!(max_capacity, 128);
            }
            other => panic!("expected MaxCapacityExceeded, got {other:?}"),
        }
        // A rejected reservation must not have grown the buffer.
        assert_eq!(arena.capacity(), 64);
    }
}
