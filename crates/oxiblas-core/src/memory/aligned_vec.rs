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
    fn layout_for(capacity: usize) -> Layout {
        let size = capacity * size_of::<T>();
        let align = ALIGN.max(align_of::<T>());
        Layout::from_size_align(size, align).expect("Invalid layout")
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
