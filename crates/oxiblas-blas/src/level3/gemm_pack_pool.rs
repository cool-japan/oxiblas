//! Thread-local reuse of GEMM packing buffers.
//!
//! The blocked drivers need an A~ and a B~ buffer per call (per tile in the
//! parallel driver). Allocating them fresh costs more than it looks: buffers
//! of 128 KiB and up are `mmap`ed by glibc, so every call page-faults them in
//! and zeroes them, and for mid-size problems that is a large share of the
//! run time (f64 n = 128 on a Zen 2 core: 34 page faults per call, 27 vs
//! 49 GFLOP/s after this change).
//!
//! [`take`] hands out a buffer of at least `len` elements, reusing one this
//! thread returned earlier through [`give`]. A reused buffer is *not*
//! zeroed: the packers write every element the micro-kernels read,
//! including the zero padding of partial panels.

use super::gemm_kernel::GemmKernel;
use oxiblas_core::memory::AlignedVec;
use std::any::Any;
use std::cell::RefCell;

/// Buffers kept per thread (an A~ and a B~ per scalar type in flight).
const MAX_POOLED: usize = 8;

/// Largest buffer kept, in bytes; bigger ones are freed on return.
const MAX_POOLED_BYTES: usize = 16 << 20;

thread_local! {
    static POOL: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// Returns a packing buffer of at least `len` elements.
pub(crate) fn take<T: GemmKernel + bytemuck::Zeroable>(len: usize) -> AlignedVec<T> {
    let reused = POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let same_type = |b: &Box<dyn Any>| b.downcast_ref::<AlignedVec<T>>().map(AlignedVec::len);
        if let Some(i) = pool
            .iter()
            .position(|b| same_type(b).is_some_and(|l| l >= len))
        {
            return pool.swap_remove(i).downcast::<AlignedVec<T>>().ok();
        }
        // Too small for this request: drop one, so that the pool does not
        // fill up with buffers no later request fits in. The new, larger
        // buffer takes its place on `give`.
        if let Some(i) = pool.iter().position(|b| same_type(b).is_some()) {
            pool.swap_remove(i);
        }
        None
    });
    match reused {
        Some(buf) => *buf,
        None => AlignedVec::zeros(len),
    }
}

/// Returns `buf` to this thread's pool for a later [`take`].
pub(crate) fn give<T: GemmKernel>(buf: AlignedVec<T>) {
    if buf.len() * core::mem::size_of::<T>() > MAX_POOLED_BYTES {
        return;
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() == MAX_POOLED {
            pool.remove(0);
        }
        pool.push(Box::new(buf));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_buffers_by_type_and_size() {
        let a: AlignedVec<f64> = take(1000);
        let ptr = a.as_ptr();
        give(a);
        // A smaller request of the same type reuses it ...
        let b: AlignedVec<f64> = take(500);
        assert_eq!(b.as_ptr(), ptr);
        assert!(b.len() >= 500);
        give(b);
        // ... a larger one or another type does not.
        let c: AlignedVec<f64> = take(2000);
        assert!(c.len() >= 2000);
        let d: AlignedVec<f32> = take(10);
        assert!(d.len() >= 10);
        give(c);
        give(d);
    }

    #[test]
    fn growing_requests_do_not_clog_the_pool() {
        // Ever larger requests (a size sweep) must keep being served from the
        // pool once each size has been seen.
        let mut last = Vec::new();
        for len in (1..=3 * MAX_POOLED).map(|i| i * 100) {
            let a: AlignedVec<f32> = take(len);
            let b: AlignedVec<f32> = take(len);
            last = vec![a.as_ptr(), b.as_ptr()];
            give(a);
            give(b);
        }
        let again: AlignedVec<f32> = take(3 * MAX_POOLED * 100);
        assert!(
            last.contains(&again.as_ptr()),
            "largest buffers were evicted"
        );
        give(again);
    }

    #[test]
    fn oversized_buffers_are_not_kept() {
        let big: AlignedVec<f64> = take((MAX_POOLED_BYTES / 8) + 1);
        let ptr = big.as_ptr();
        give(big);
        let again: AlignedVec<f64> = take((MAX_POOLED_BYTES / 8) + 1);
        assert_ne!(again.as_ptr(), ptr);
    }
}
