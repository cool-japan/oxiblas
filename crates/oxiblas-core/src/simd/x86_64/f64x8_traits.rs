//! # `F64x8` - Trait Implementations
//!
//! This module contains trait implementations for `F64x8`.
//!
//! ## Implemented Traits
//!
//! - `SimdRegister`
//! - `SimdMask`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use core::arch::x86_64::*;
use crate::simd::{SimdMask, SimdRegister};

use super::functions::{has_avx512f, lane_index_out_of_range, scalar_binop, scalar_load, scalar_reduce, scalar_splat, scalar_store, scalar_ternop};
use super::types::F64x8;

impl SimdRegister for F64x8 {
    type Scalar = f64;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_setzero_pd()) }
        } else {
            unsafe { scalar_splat::<F64x8, f64, 8>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_set1_pd(value)) }
        } else {
            unsafe { scalar_splat::<F64x8, f64, 8>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        if has_avx512f() {
            F64x8(_mm512_load_pd(ptr))
        } else {
            scalar_load::<F64x8, f64, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        if has_avx512f() {
            F64x8(_mm512_loadu_pd(ptr))
        } else {
            scalar_load::<F64x8, f64, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        if has_avx512f() {
            _mm512_store_pd(ptr, self.0);
        } else {
            scalar_store::<F64x8, f64, 8>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        if has_avx512f() {
            _mm512_storeu_pd(ptr, self.0);
        } else {
            scalar_store::<F64x8, f64, 8>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_add_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_sub_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_mul_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_div_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_fmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x8, f64, 8>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_fmsub_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x8, f64, 8>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_fnmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x8, f64, 8>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        if has_avx512f() {
            unsafe { _mm512_reduce_add_pd(self.0) }
        } else {
            unsafe { scalar_reduce::<F64x8, f64, 8>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        if has_avx512f() {
            unsafe { _mm512_reduce_max_pd(self.0) }
        } else {
            unsafe { scalar_reduce::<F64x8, f64, 8>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        if has_avx512f() {
            unsafe { _mm512_reduce_min_pd(self.0) }
        } else {
            unsafe { scalar_reduce::<F64x8, f64, 8>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        let arr: [f64; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        let mut arr: [f64; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F64x8(unsafe { core::mem::transmute(arr) })
    }
}
impl SimdMask for F64x8 {
    type Mask = __mmask8;

    #[inline]
    fn mask_from_bools(bools: &[bool]) -> Self::Mask {
        let mut mask: u8 = 0;
        for (i, &b) in bools.iter().take(8).enumerate() {
            if b {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[inline]
    unsafe fn load_masked(ptr: *const f64, mask: Self::Mask, default: Self) -> Self {
        if has_avx512f() {
            F64x8(_mm512_mask_loadu_pd(default.0, mask, ptr))
        } else {
            let dd: [f64; 8] = core::mem::transmute(default.0);
            let rr: [f64; 8] = core::array::from_fn(|i| {
                if (mask >> i) & 1 == 1 {
                    unsafe { ptr.add(i).read_unaligned() }
                } else {
                    dd[i]
                }
            });
            F64x8(core::mem::transmute(rr))
        }
    }

    #[inline]
    unsafe fn store_masked(self, ptr: *mut f64, mask: Self::Mask) {
        if has_avx512f() {
            _mm512_mask_storeu_pd(ptr, mask, self.0);
        } else {
            let vv: [f64; 8] = core::mem::transmute(self.0);
            for i in 0..8 {
                if (mask >> i) & 1 == 1 {
                    ptr.add(i).write_unaligned(vv[i]);
                }
            }
        }
    }

    #[inline]
    fn blend(mask: Self::Mask, a: Self, b: Self) -> Self {
        if has_avx512f() {
            // `_mm512_mask_blend_pd(k, x, y)` yields `k[i] ? y[i] : x[i]`; passing
            // `(mask, b, a)` therefore selects `a` where the mask bit is set.
            unsafe { F64x8(_mm512_mask_blend_pd(mask, b.0, a.0)) }
        } else {
            unsafe {
                let aa: [f64; 8] = core::mem::transmute(a.0);
                let bb: [f64; 8] = core::mem::transmute(b.0);
                let rr: [f64; 8] =
                    core::array::from_fn(|i| if (mask >> i) & 1 == 1 { aa[i] } else { bb[i] });
                F64x8(core::mem::transmute(rr))
            }
        }
    }
}
