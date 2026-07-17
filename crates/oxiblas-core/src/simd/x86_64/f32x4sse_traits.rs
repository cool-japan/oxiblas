//! # `F32x4Sse` - Trait Implementations
//!
//! This module contains trait implementations for `F32x4Sse`.
//!
//! ## Implemented Traits
//!
//! - `SimdRegister`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::simd::SimdRegister;

use super::types::F32x4Sse;
use super::functions::lane_index_out_of_range;

impl SimdRegister for F32x4Sse {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe { F32x4Sse(_mm_setzero_ps()) }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { F32x4Sse(_mm_set1_ps(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x4Sse(_mm_load_ps(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x4Sse(_mm_loadu_ps(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        _mm_store_ps(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        _mm_storeu_ps(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_add_ps(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_sub_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_mul_ps(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_div_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F32x4Sse(_mm_fmadd_ps(self.0, a.0, b.0))
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
            F32x4Sse(_mm_fmsub_ps(self.0, a.0, b.0))
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
            F32x4Sse(_mm_fnmadd_ps(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            b.sub(self.mul(a))
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        // SSE2-only: `haddps` is SSE3. Reduce [a0,a1,a2,a3] using `movhlps`
        // (SSE) to fold the upper pair onto the lower pair, then a shuffle +
        // scalar add (SSE/SSE2) to combine the remaining two partial sums.
        unsafe {
            let high = _mm_movehl_ps(self.0, self.0); // [a2, a3, a2, a3]
            let sum1 = _mm_add_ps(self.0, high); // lane0 = a0+a2, lane1 = a1+a3
            let shuf = _mm_shuffle_ps(sum1, sum1, 0b00_00_00_01); // lane0 = a1+a3
            let sum2 = _mm_add_ss(sum1, shuf); // lane0 = (a0+a2)+(a1+a3)
            _mm_cvtss_f32(sum2)
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe {
            let shuffled = _mm_shuffle_ps(self.0, self.0, 0b10_11_00_01);
            let max1 = _mm_max_ps(self.0, shuffled);
            let shuffled2 = _mm_shuffle_ps(max1, max1, 0b00_00_10_10);
            let max2 = _mm_max_ps(max1, shuffled2);
            _mm_cvtss_f32(max2)
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe {
            let shuffled = _mm_shuffle_ps(self.0, self.0, 0b10_11_00_01);
            let min1 = _mm_min_ps(self.0, shuffled);
            let shuffled2 = _mm_shuffle_ps(min1, min1, 0b00_00_10_10);
            let min2 = _mm_min_ps(min1, shuffled2);
            _mm_cvtss_f32(min2)
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        let arr: [f32; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        let mut arr: [f32; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F32x4Sse(unsafe { core::mem::transmute(arr) })
    }
}
