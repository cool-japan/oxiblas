//! # `F32x8` - Trait Implementations
//!
//! This module contains trait implementations for `F32x8`.
//!
//! ## Implemented Traits
//!
//! - `SimdRegister`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use core::arch::x86_64::*;
use crate::simd::SimdRegister;

use super::types::F32x8;
use super::functions::{has_avx2_fma, lane_index_out_of_range, scalar_binop, scalar_load, scalar_reduce, scalar_splat, scalar_store, scalar_ternop};

impl SimdRegister for F32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_setzero_ps()) }
        } else {
            unsafe { scalar_splat::<F32x8, f32, 8>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_set1_ps(value)) }
        } else {
            unsafe { scalar_splat::<F32x8, f32, 8>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        if has_avx2_fma() {
            F32x8(_mm256_load_ps(ptr))
        } else {
            scalar_load::<F32x8, f32, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        if has_avx2_fma() {
            F32x8(_mm256_loadu_ps(ptr))
        } else {
            scalar_load::<F32x8, f32, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        if has_avx2_fma() {
            _mm256_store_ps(ptr, self.0);
        } else {
            scalar_store::<F32x8, f32, 8>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        if has_avx2_fma() {
            _mm256_storeu_ps(ptr, self.0);
        } else {
            scalar_store::<F32x8, f32, 8>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_add_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_sub_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_mul_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_div_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_fmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x8, f32, 8>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_fmsub_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x8, f32, 8>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_fnmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x8, f32, 8>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        if has_avx2_fma() {
            unsafe {
                // Horizontal add pairs
                let sum1 = _mm256_hadd_ps(self.0, self.0);
                let sum2 = _mm256_hadd_ps(sum1, sum1);
                // Extract and add high/low halves
                let high = _mm256_extractf128_ps(sum2, 1);
                let low = _mm256_castps256_ps128(sum2);
                let sum3 = _mm_add_ps(low, high);
                _mm_cvtss_f32(sum3)
            }
        } else {
            unsafe { scalar_reduce::<F32x8, f32, 8>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        if has_avx2_fma() {
            unsafe {
                let high = _mm256_extractf128_ps(self.0, 1);
                let low = _mm256_castps256_ps128(self.0);
                let max1 = _mm_max_ps(low, high);
                // Shuffle and compare
                let max2 = _mm_shuffle_ps(max1, max1, 0b10_11_00_01);
                let max3 = _mm_max_ps(max1, max2);
                let max4 = _mm_shuffle_ps(max3, max3, 0b00_00_10_10);
                let max5 = _mm_max_ps(max3, max4);
                _mm_cvtss_f32(max5)
            }
        } else {
            unsafe { scalar_reduce::<F32x8, f32, 8>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        if has_avx2_fma() {
            unsafe {
                let high = _mm256_extractf128_ps(self.0, 1);
                let low = _mm256_castps256_ps128(self.0);
                let min1 = _mm_min_ps(low, high);
                let min2 = _mm_shuffle_ps(min1, min1, 0b10_11_00_01);
                let min3 = _mm_min_ps(min1, min2);
                let min4 = _mm_shuffle_ps(min3, min3, 0b00_00_10_10);
                let min5 = _mm_min_ps(min3, min4);
                _mm_cvtss_f32(min5)
            }
        } else {
            unsafe { scalar_reduce::<F32x8, f32, 8>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        let arr: [f32; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        let mut arr: [f32; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F32x8(unsafe { core::mem::transmute(arr) })
    }
}
