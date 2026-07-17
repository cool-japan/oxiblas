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
use core::mem::{align_of, size_of};
use core::ptr::NonNull;

#[cfg(not(feature = "std"))]
use alloc::alloc::handle_alloc_error;
#[cfg(feature = "std")]
use std::alloc::handle_alloc_error;

use super::alloc::*;

// =============================================================================
// AlignedVec - Aligned heap allocation
// =============================================================================

/// Compile-time check that an alignment const generic is a power of two, as
/// required by [`core::alloc::Layout::from_size_align`].
///
/// Call sites wrap the call in an inline `const { .. }` block (stable since
/// Rust 1.79) so the assertion is evaluated at monomorphization time: if
/// `ALIGN` is not a power of two, compilation fails with a clear message
/// instead of `AlignedVec` producing an invalid `Layout` (or silently
/// rounding up) the first time it is used.
const fn assert_align_is_power_of_two<const ALIGN: usize>() {
    assert!(
        ALIGN.is_power_of_two(),
        "AlignedVec: ALIGN const generic parameter must be a power of two"
    );
}

/// A vector with guaranteed alignment and custom allocator support.
///
/// Unlike `Vec<T>`, this type ensures the underlying buffer is aligned
/// to at least `ALIGN` bytes, which is required for efficient SIMD operations.
///
/// # Type Parameters
///
/// - `T`: The element type
/// - `ALIGN`: The minimum alignment in bytes (default: 64 for cache line alignment)
/// - `A`: The allocator type (default: `Global`)
///
/// # Custom Allocators
///
/// You can use a custom allocator by specifying the third type parameter:
///
/// ```ignore
/// use oxiblas_core::memory::{AlignedVec, Alloc, Global};
///
/// // Use global allocator (default)
/// let vec: AlignedVec<f64> = AlignedVec::zeros(100);
///
/// // Use custom allocator
/// let custom_vec: AlignedVec<f64, 64, MyAlloc> = AlignedVec::zeros_in(100, MyAlloc::new());
/// ```
pub struct AlignedVec<T, const ALIGN: usize = DEFAULT_ALIGN, A: Alloc = Global> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    alloc: A,
}

// Convenience methods using Global allocator
impl<T, const ALIGN: usize> AlignedVec<T, ALIGN, Global> {
    /// Creates a new empty aligned vector.
    #[inline]
    pub const fn new() -> Self {
        // Compile-time proof that `ALIGN` is a power of two, as required by
        // `core::alloc::Layout`. Evaluated at monomorphization time, so an
        // invalid `ALIGN` fails to compile rather than panicking (or worse,
        // producing a malformed `Layout`) the first time an instance of this
        // type is actually allocated.
        const { assert_align_is_power_of_two::<ALIGN>() };

        AlignedVec {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            alloc: Global,
        }
    }

    /// Creates a new aligned vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }

    /// Creates a new aligned vector filled with zeros.
    ///
    /// This is more efficient than creating and then filling, as it uses
    /// zeroed allocation.
    pub fn zeros(len: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        Self::zeros_in(len, Global)
    }

    /// Creates a new aligned vector filled with a value.
    pub fn filled(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self::filled_in(len, value, Global)
    }

    /// Creates a new aligned vector from a slice.
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self::from_slice_in(slice, Global)
    }
}

// Methods that work with any allocator
impl<T, const ALIGN: usize, A: Alloc> AlignedVec<T, ALIGN, A> {
    /// Creates a new empty aligned vector with the specified allocator.
    #[inline]
    pub fn new_in(alloc: A) -> Self {
        const { assert_align_is_power_of_two::<ALIGN>() };

        AlignedVec {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            alloc,
        }
    }

    /// Creates a new aligned vector with the given capacity and allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        if capacity == 0 {
            return Self::new_in(alloc);
        }

        let layout = Self::layout_for(capacity);
        let ptr = alloc.allocate(layout) as *mut T;

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        AlignedVec {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len: 0,
            cap: capacity,
            alloc,
        }
    }

    /// Creates a new aligned vector filled with zeros using the specified allocator.
    pub fn zeros_in(len: usize, alloc: A) -> Self
    where
        T: bytemuck::Zeroable,
    {
        if len == 0 {
            return Self::new_in(alloc);
        }

        let layout = Self::layout_for(len);
        let ptr = alloc.allocate_zeroed(layout) as *mut T;

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        AlignedVec {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len,
            cap: len,
            alloc,
        }
    }

    /// Creates a new aligned vector filled with a value using the specified allocator.
    pub fn filled_in(len: usize, value: T, alloc: A) -> Self
    where
        T: Clone,
    {
        let mut vec = Self::with_capacity_in(len, alloc);
        for _ in 0..len {
            vec.push(value.clone());
        }
        vec
    }

    /// Creates a new aligned vector from a slice using the specified allocator.
    pub fn from_slice_in(slice: &[T], alloc: A) -> Self
    where
        T: Clone,
    {
        let mut vec = Self::with_capacity_in(slice.len(), alloc);
        for item in slice {
            vec.push(item.clone());
        }
        vec
    }

    /// Returns a reference to the allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns the layout for a given capacity.
    ///
    /// # Panics
    ///
    /// Panics (via [`Self::capacity_overflow`]) if `capacity * size_of::<T>()`
    /// overflows `usize`, or if the resulting size -- rounded up to
    /// `ALIGN.max(align_of::<T>())` -- would exceed `isize::MAX` bytes.
    ///
    /// A naive `capacity * size_of::<T>()` would silently wrap around on
    /// overflow in release builds (multiplication overflow checks are
    /// disabled outside of `debug_assertions`), yielding a small, wrong
    /// `size` that produces a *successfully allocated but undersized*
    /// buffer. Callers such as [`Self::with_capacity_in`] would then record
    /// the huge, un-wrapped `capacity` in `self.cap`, so later writes up to
    /// that bogus capacity (e.g. via [`Self::push`] past `self.len`) would
    /// write past the real allocation: a heap-buffer overflow. Using
    /// `checked_mul` turns that silent memory-corruption bug into a loud,
    /// immediate panic instead.
    fn layout_for(capacity: usize) -> Layout {
        const { assert_align_is_power_of_two::<ALIGN>() };

        let size = match capacity.checked_mul(size_of::<T>()) {
            Some(size) => size,
            None => Self::capacity_overflow(),
        };
        let align = ALIGN.max(align_of::<T>());
        match Layout::from_size_align(size, align) {
            Ok(layout) => layout,
            Err(_) => Self::capacity_overflow(),
        }
    }

    /// Reports that `capacity` does not correspond to a valid, addressable
    /// [`Layout`] for `T` and aborts via panic.
    ///
    /// This mirrors the strategy `alloc::raw_vec::RawVec` uses for oversized
    /// capacities: rather than proceeding with a silently truncated
    /// (wrapped) allocation size -- which would desynchronize the vector's
    /// tracked capacity from its real allocation -- fail loudly and
    /// immediately. Marked `#[cold]`/`#[inline(never)]` so the (exceedingly
    /// rare) overflow path does not bloat the hot allocation path, and
    /// implemented as a named function rather than `.unwrap()`/`.expect()`
    /// so the panic message is specific to `AlignedVec` and its type/align
    /// parameters.
    #[cold]
    #[inline(never)]
    fn capacity_overflow() -> ! {
        panic!(
            "AlignedVec<{}>: capacity overflow -- requested capacity does not \
             fit in a valid memory layout (capacity * size_of::<T>() overflows \
             usize, or exceeds isize::MAX bytes when rounded up to align={ALIGN})",
            core::any::type_name::<T>()
        );
    }

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

    /// Returns the capacity of the vector.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns a pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns a slice of the vector.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice of the vector.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Pushes a value onto the vector.
    ///
    /// # Panics
    /// Panics if the vector is at capacity.
    pub fn push(&mut self, value: T) {
        if self.len >= self.cap {
            self.grow();
        }

        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }

    /// Pops a value from the vector.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        unsafe { Some(self.ptr.as_ptr().add(self.len).read()) }
    }

    /// Clears the vector.
    pub fn clear(&mut self) {
        while self.pop().is_some() {}
    }

    /// Resizes the vector to the given length.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.len {
            self.reserve(new_len - self.len);
            for _ in self.len..new_len {
                self.push(value.clone());
            }
        } else {
            while self.len > new_len {
                self.pop();
            }
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        let required = self.len + additional;
        if required > self.cap {
            let new_cap = required.max(self.cap * 2).max(8);
            self.realloc(new_cap);
        }
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 8 } else { self.cap * 2 };
        self.realloc(new_cap);
    }

    fn realloc(&mut self, new_cap: usize) {
        let new_layout = Self::layout_for(new_cap);
        let new_ptr = self.alloc.allocate(new_layout) as *mut T;

        if new_ptr.is_null() {
            handle_alloc_error(new_layout);
        }

        // Copy existing data
        if self.cap > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.len);
                let old_layout = Self::layout_for(self.cap);
                self.alloc
                    .deallocate(self.ptr.as_ptr() as *mut u8, old_layout);
            }
        }

        self.ptr = unsafe { NonNull::new_unchecked(new_ptr) };
        self.cap = new_cap;
    }
}

impl<T, const ALIGN: usize, A: Alloc> Drop for AlignedVec<T, ALIGN, A> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                core::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }

        // Deallocate
        if self.cap > 0 {
            let layout = Self::layout_for(self.cap);
            unsafe {
                self.alloc.deallocate(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T, const ALIGN: usize> Default for AlignedVec<T, ALIGN, Global> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const ALIGN: usize, A: Alloc> Clone for AlignedVec<T, ALIGN, A> {
    fn clone(&self) -> Self {
        Self::from_slice_in(self.as_slice(), self.alloc.clone())
    }
}

impl<T, const ALIGN: usize, A: Alloc> core::ops::Deref for AlignedVec<T, ALIGN, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const ALIGN: usize, A: Alloc> core::ops::DerefMut for AlignedVec<T, ALIGN, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, const ALIGN: usize, A: Alloc> core::ops::Index<usize> for AlignedVec<T, ALIGN, A> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T, const ALIGN: usize, A: Alloc> core::ops::IndexMut<usize> for AlignedVec<T, ALIGN, A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

// Safety: AlignedVec is Send/Sync if T and A are
unsafe impl<T: Send, const ALIGN: usize, A: Alloc + Send> Send for AlignedVec<T, ALIGN, A> {}
unsafe impl<T: Sync, const ALIGN: usize, A: Alloc + Sync> Sync for AlignedVec<T, ALIGN, A> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_capacity_allocates_correctly_aligned_and_sized_buffer() {
        const ALIGN: usize = 64;
        let vec: AlignedVec<f32, ALIGN> = AlignedVec::with_capacity(37);

        assert_eq!(vec.capacity(), 37);
        assert_eq!(vec.len(), 0);
        assert_eq!(
            vec.as_ptr() as usize % ALIGN,
            0,
            "buffer must be ALIGN-aligned"
        );
    }

    #[test]
    fn zeros_and_push_round_trip() {
        let mut vec: AlignedVec<f64> = AlignedVec::zeros(4);
        assert_eq!(vec.as_slice(), &[0.0, 0.0, 0.0, 0.0]);

        vec.push(1.0);
        vec.push(2.0);
        assert_eq!(vec.len(), 6);
        assert_eq!(&vec.as_slice()[4..], &[1.0, 2.0]);
    }

    #[test]
    fn reserve_and_grow_preserve_existing_elements() {
        let mut vec: AlignedVec<u64> = AlignedVec::with_capacity(2);
        vec.push(10);
        vec.push(20);
        // Forces `grow` -> `realloc` -> `layout_for` with a larger capacity.
        vec.push(30);
        vec.reserve(64);

        assert!(vec.capacity() >= 67);
        assert_eq!(vec.as_slice(), &[10, 20, 30]);
    }

    // Regression test for: `layout_for` computing
    // `capacity * size_of::<T>()` with an unchecked multiplication. In a
    // release build (where integer-overflow checks are compiled out), a
    // capacity just large enough to overflow `usize` would silently wrap
    // around to a small `size`, so the allocator would hand back a tiny
    // buffer while `self.cap` kept recording the huge, un-wrapped capacity
    // requested by the caller -- a heap-buffer-overflow-in-waiting the
    // moment anything wrote up to that bogus capacity. It must now fail
    // loudly via `capacity_overflow` instead.
    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn with_capacity_overflowing_size_panics_instead_of_wrapping() {
        // For `f64` (8 bytes), `usize::MAX / 2` multiplied by 8 overflows
        // `usize` by a wide margin, so the old unchecked multiplication
        // would have wrapped rather than triggering `Layout::from_size_align`'s
        // own (unrelated) `isize::MAX` check.
        let huge_capacity = usize::MAX / 2;
        let _vec: AlignedVec<f64> = AlignedVec::with_capacity(huge_capacity);
    }

    // Regression test for the same bug reached via `zeros_in`, which builds
    // its `Layout` the same way as `with_capacity_in`.
    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn zeros_overflowing_size_panics_instead_of_wrapping() {
        let huge_len = usize::MAX / 2;
        let _vec: AlignedVec<f64> = AlignedVec::zeros(huge_len);
    }

    // A capacity that does not overflow the `checked_mul` but whose size,
    // once rounded up to `ALIGN`, exceeds `isize::MAX` must also be rejected
    // by `Layout::from_size_align` and surfaced as `capacity_overflow`
    // rather than propagating an `Err` (or, previously, panicking via a
    // generic `.expect("Invalid layout")`).
    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn with_capacity_isize_max_exceeded_panics() {
        // `capacity * size_of::<u8>()` does not overflow `usize` here, but
        // it does exceed `isize::MAX`, which `Layout::from_size_align`
        // rejects.
        let capacity = isize::MAX as usize;
        let _vec: AlignedVec<u8> = AlignedVec::with_capacity(capacity);
    }
}
