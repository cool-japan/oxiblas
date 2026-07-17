//! # `F32x16` - Trait Implementations
//!
//! This module contains trait implementations for `F32x16`.
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
use super::types::F32x16;

impl SimdRegister for F32x16 {
    type Scalar = f32;
    const LANES: usize = 16;

    #[inline]
    fn zero() -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_setzero_ps()) }
        } else {
            unsafe { scalar_splat::<F32x16, f32, 16>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_set1_ps(value)) }
        } else {
            unsafe { scalar_splat::<F32x16, f32, 16>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        if has_avx512f() {
            F32x16(_mm512_load_ps(ptr))
        } else {
            scalar_load::<F32x16, f32, 16>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        if has_avx512f() {
            F32x16(_mm512_loadu_ps(ptr))
        } else {
            scalar_load::<F32x16, f32, 16>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        if has_avx512f() {
            _mm512_store_ps(ptr, self.0);
        } else {
            scalar_store::<F32x16, f32, 16>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        if has_avx512f() {
            _mm512_storeu_ps(ptr, self.0);
        } else {
            scalar_store::<F32x16, f32, 16>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_add_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_sub_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_mul_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_div_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_fmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x16, f32, 16>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_fmsub_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x16, f32, 16>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_fnmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x16, f32, 16>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        if has_avx512f() {
            unsafe { _mm512_reduce_add_ps(self.0) }
        } else {
            unsafe { scalar_reduce::<F32x16, f32, 16>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        if has_avx512f() {
            unsafe { _mm512_reduce_max_ps(self.0) }
        } else {
            unsafe { scalar_reduce::<F32x16, f32, 16>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        if has_avx512f() {
            unsafe { _mm512_reduce_min_ps(self.0) }
        } else {
            unsafe { scalar_reduce::<F32x16, f32, 16>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        let arr: [f32; 16] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        let mut arr: [f32; 16] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F32x16(unsafe { core::mem::transmute(arr) })
    }
}
impl SimdMask for F32x16 {
    type Mask = __mmask16;

    #[inline]
    fn mask_from_bools(bools: &[bool]) -> Self::Mask {
        let mut mask: u16 = 0;
        for (i, &b) in bools.iter().take(16).enumerate() {
            if b {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[inline]
    unsafe fn load_masked(ptr: *const f32, mask: Self::Mask, default: Self) -> Self {
        if has_avx512f() {
            F32x16(_mm512_mask_loadu_ps(default.0, mask, ptr))
        } else {
            let dd: [f32; 16] = core::mem::transmute(default.0);
            let rr: [f32; 16] = core::array::from_fn(|i| {
                if (mask >> i) & 1 == 1 {
                    unsafe { ptr.add(i).read_unaligned() }
                } else {
                    dd[i]
                }
            });
            F32x16(core::mem::transmute(rr))
        }
    }

    #[inline]
    unsafe fn store_masked(self, ptr: *mut f32, mask: Self::Mask) {
        if has_avx512f() {
            _mm512_mask_storeu_ps(ptr, mask, self.0);
        } else {
            let vv: [f32; 16] = core::mem::transmute(self.0);
            for i in 0..16 {
                if (mask >> i) & 1 == 1 {
                    ptr.add(i).write_unaligned(vv[i]);
                }
            }
        }
    }

    #[inline]
    fn blend(mask: Self::Mask, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_mask_blend_ps(mask, b.0, a.0)) }
        } else {
            unsafe {
                let aa: [f32; 16] = core::mem::transmute(a.0);
                let bb: [f32; 16] = core::mem::transmute(b.0);
                let rr: [f32; 16] =
                    core::array::from_fn(|i| if (mask >> i) & 1 == 1 { aa[i] } else { bb[i] });
                F32x16(core::mem::transmute(rr))
            }
        }
    }
}
