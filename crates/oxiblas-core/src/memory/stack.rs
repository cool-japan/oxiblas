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
// StackReq - Scratch space requirements
// =============================================================================

/// Represents the memory requirements for an operation.
///
/// This is used to pre-compute the scratch space needed for algorithms,
/// allowing efficient allocation strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackReq {
    /// Size in bytes.
    pub size: usize,
    /// Alignment in bytes.
    pub align: usize,
}

impl StackReq {
    /// Zero requirements.
    pub const ZERO: StackReq = StackReq { size: 0, align: 1 };

    /// Creates a new stack requirement.
    #[inline]
    pub const fn new(size: usize, align: usize) -> Self {
        StackReq { size, align }
    }

    /// Creates a requirement for a given type and count.
    #[inline]
    pub const fn new_for<T>(count: usize) -> Self {
        StackReq {
            size: count * size_of::<T>(),
            align: align_of::<T>(),
        }
    }

    /// Combines two requirements (both must be satisfied).
    #[inline]
    pub const fn and(self, other: Self) -> Self {
        let align = if self.align > other.align {
            self.align
        } else {
            other.align
        };
        let size1 = round_up_pow2(self.size, other.align);
        StackReq {
            size: size1 + other.size,
            align,
        }
    }

    /// Takes the maximum of two requirements (either one is sufficient).
    #[inline]
    pub const fn or(self, other: Self) -> Self {
        let align = if self.align > other.align {
            self.align
        } else {
            other.align
        };
        let size = if self.size > other.size {
            self.size
        } else {
            other.size
        };
        StackReq { size, align }
    }

    /// Returns the requirement aligned to a larger alignment.
    #[inline]
    pub const fn with_align(self, align: usize) -> Self {
        let new_align = if self.align > align {
            self.align
        } else {
            align
        };
        StackReq {
            size: self.size,
            align: new_align,
        }
    }
}

/// Combines multiple stack requirements (all must be satisfied).
#[macro_export]
macro_rules! stack_req_all {
    ($($req:expr),* $(,)?) => {{
        let mut result = $crate::memory::StackReq::ZERO;
        $(
            result = result.and($req);
        )*
        result
    }};
}

/// Takes the maximum of multiple stack requirements.
#[macro_export]
macro_rules! stack_req_any {
    ($($req:expr),* $(,)?) => {{
        let mut result = $crate::memory::StackReq::ZERO;
        $(
            result = result.or($req);
        )*
        result
    }};
}

// =============================================================================
// MemStack - Stack-based temporary allocation
// =============================================================================

/// A memory stack for temporary allocations.
///
/// This provides fast, stack-based allocation for scratch space needed
/// by algorithms. Allocations are invalidated when the stack is reset.
pub struct MemStack {
    buffer: AlignedVec<u8>,
    offset: usize,
}

impl MemStack {
    /// Creates a new memory stack with the given requirement.
    pub fn new(req: StackReq) -> Self {
        let size = round_up_pow2(req.size, req.align);
        MemStack {
            buffer: AlignedVec::zeros(size),
            offset: 0,
        }
    }

    /// Creates a new memory stack with the given size.
    pub fn with_size(size: usize) -> Self {
        MemStack {
            buffer: AlignedVec::zeros(size),
            offset: 0,
        }
    }

    /// Returns the remaining capacity.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.offset
    }

    /// Resets the stack, invalidating all allocations.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocates a slice of the given type.
    ///
    /// # Panics
    /// Panics if there's not enough space.
    pub fn alloc<T>(&mut self, count: usize) -> &mut [MaybeUninit<T>] {
        let align = align_of::<T>();
        let aligned_offset = round_up_pow2(self.offset, align);
        let size = count * size_of::<T>();
        let new_offset = aligned_offset + size;

        assert!(new_offset <= self.buffer.len(), "MemStack overflow");

        let ptr = unsafe { self.buffer.as_mut_ptr().add(aligned_offset) as *mut MaybeUninit<T> };
        self.offset = new_offset;

        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// Allocates and zeros a slice of the given type.
    pub fn alloc_zeroed<T: bytemuck::Zeroable>(&mut self, count: usize) -> &mut [T] {
        let slice = self.alloc::<T>(count);
        // Zero the memory
        unsafe {
            core::ptr::write_bytes(slice.as_mut_ptr() as *mut u8, 0, count * size_of::<T>());
            core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, count)
        }
    }

    /// Creates a sub-stack with the remaining memory.
    ///
    /// This is useful for recursive algorithms that need to pass
    /// scratch space to sub-operations.
    ///
    /// Note: This is a placeholder implementation. A full implementation
    /// would use raw pointers to share the buffer without ownership transfer.
    pub fn make_sub_stack(&mut self) -> MemStack {
        let _remaining = self.remaining();
        let _ptr = unsafe { self.buffer.as_mut_ptr().add(self.offset) };

        // Mark all remaining as used
        self.offset = self.buffer.len();

        // TODO: Implement proper sub-stack that shares buffer
        MemStack {
            buffer: AlignedVec::new(),
            offset: 0,
        }
    }
}
