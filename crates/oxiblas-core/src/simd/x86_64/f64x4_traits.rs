//! # `F64x4` - Trait Implementations
//!
//! This module contains trait implementations for `F64x4`.
//!
//! ## Implemented Traits
//!
//! - `SimdRegister`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::simd::SimdRegister;

use super::functions::{has_avx2_fma, lane_index_out_of_range, scalar_binop, scalar_load, scalar_reduce, scalar_splat, scalar_store, scalar_ternop};
use super::types::F64x4;

impl SimdRegister for F64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_setzero_pd()) }
        } else {
            unsafe { scalar_splat::<F64x4, f64, 4>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_set1_pd(value)) }
        } else {
            unsafe { scalar_splat::<F64x4, f64, 4>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        if has_avx2_fma() {
            F64x4(_mm256_load_pd(ptr))
        } else {
            scalar_load::<F64x4, f64, 4>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        if has_avx2_fma() {
            F64x4(_mm256_loadu_pd(ptr))
        } else {
            scalar_load::<F64x4, f64, 4>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        if has_avx2_fma() {
            _mm256_store_pd(ptr, self.0);
        } else {
            scalar_store::<F64x4, f64, 4>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        if has_avx2_fma() {
            _mm256_storeu_pd(ptr, self.0);
        } else {
            scalar_store::<F64x4, f64, 4>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_add_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_sub_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_mul_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_div_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // FMA: self * a + b. The scalar fallback uses two roundings (mul then
        // add), matching the SSE non-FMA path; `core`'s single-rounding
        // `f64::mul_add` is `std`-only, so it cannot be used on the no_std path.
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_fmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x4, f64, 4>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        // FMA: self * a - b
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_fmsub_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x4, f64, 4>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        // FMA: -(self * a) + b = b - self * a
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_fnmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x4, f64, 4>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        if has_avx2_fma() {
            unsafe {
                // Horizontal add: [a0+a1, a2+a3, a0+a1, a2+a3]
                let sum1 = _mm256_hadd_pd(self.0, self.0);
                // Extract high 128 bits and add to low 128 bits
                let high = _mm256_extractf128_pd(sum1, 1);
                let low = _mm256_castpd256_pd128(sum1);
                let sum2 = _mm_add_pd(low, high);
                _mm_cvtsd_f64(sum2)
            }
        } else {
            unsafe { scalar_reduce::<F64x4, f64, 4>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        if has_avx2_fma() {
            unsafe {
                // Compare and take max of pairs
                let high = _mm256_extractf128_pd(self.0, 1);
                let low = _mm256_castpd256_pd128(self.0);
                let max1 = _mm_max_pd(low, high);
                // Shuffle and compare again
                let max2 = _mm_unpackhi_pd(max1, max1);
                let max3 = _mm_max_pd(max1, max2);
                _mm_cvtsd_f64(max3)
            }
        } else {
            unsafe { scalar_reduce::<F64x4, f64, 4>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        if has_avx2_fma() {
            unsafe {
                let high = _mm256_extractf128_pd(self.0, 1);
                let low = _mm256_castpd256_pd128(self.0);
                let min1 = _mm_min_pd(low, high);
                let min2 = _mm_unpackhi_pd(min1, min1);
                let min3 = _mm_min_pd(min1, min2);
                _mm_cvtsd_f64(min3)
            }
        } else {
            unsafe { scalar_reduce::<F64x4, f64, 4>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        let arr: [f64; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        let mut arr: [f64; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F64x4(unsafe { core::mem::transmute(arr) })
    }
}
