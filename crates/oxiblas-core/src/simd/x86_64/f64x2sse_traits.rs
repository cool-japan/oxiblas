//! # `F64x2Sse` - Trait Implementations
//!
//! This module contains trait implementations for `F64x2Sse`.
//!
//! ## Implemented Traits
//!
//! - `SimdRegister`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::simd::SimdRegister;

use super::functions::lane_index_out_of_range;
use super::types::F64x2Sse;

impl SimdRegister for F64x2Sse {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn zero() -> Self {
        unsafe { F64x2Sse(_mm_setzero_pd()) }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { F64x2Sse(_mm_set1_pd(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x2Sse(_mm_load_pd(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x2Sse(_mm_loadu_pd(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        _mm_store_pd(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        _mm_storeu_pd(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_add_pd(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_sub_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_mul_pd(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_div_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // SSE doesn't have native FMA, emulate it
        // If FMA is available, use it
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fmadd_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).add(b)
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fmsub_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).sub(b)
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fnmadd_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            b.sub(self.mul(a))
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        // SSE2-only: the 128-bit tier is granted on bare SSE2, but `haddpd` is
        // an SSE3 instruction. Bring the high lane down with `unpckhpd` (SSE2)
        // and add the two low doubles with `addsd` (SSE2).
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0); // [a1, a1]
            let sum = _mm_add_sd(self.0, high); // low lane = a0 + a1
            _mm_cvtsd_f64(sum)
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0);
            let max = _mm_max_pd(self.0, high);
            _mm_cvtsd_f64(max)
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0);
            let min = _mm_min_pd(self.0, high);
            _mm_cvtsd_f64(min)
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        let arr: [f64; 2] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        let mut arr: [f64; 2] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F64x2Sse(unsafe { core::mem::transmute(arr) })
    }
}
