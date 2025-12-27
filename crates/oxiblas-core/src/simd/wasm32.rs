//! WebAssembly SIMD implementations using WASM SIMD128.
//!
//! This module provides SIMD register types and operations for WebAssembly
//! targets with SIMD support (requires `simd128` feature).
//!
//! WASM SIMD128 provides 128-bit vectors. For 256-bit registers, we emulate
//! using two 128-bit registers, similar to the AArch64 approach.

#![cfg(target_arch = "wasm32")]

use crate::simd::{SimdRegister, SimdScalar};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

// =============================================================================
// WASM SIMD128 (128-bit) implementations
// =============================================================================

/// 128-bit SIMD register for f64 (2 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x2(v128);

impl SimdRegister for F64x2 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn zero() -> Self {
        unsafe { F64x2(f64x2_splat(0.0)) }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { F64x2(f64x2_splat(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x2(v128_load(ptr as *const v128))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x2(v128_load(ptr as *const v128))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        v128_store(ptr as *mut v128, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        v128_store(ptr as *mut v128, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F64x2(f64x2_add(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F64x2(f64x2_sub(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F64x2(f64x2_mul(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F64x2(f64x2_div(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // WASM doesn't have FMA, so emulate it
        self.mul(a).add(b)
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        b.sub(self.mul(a))
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        unsafe {
            let arr: [f64; 2] = core::mem::transmute(self.0);
            arr[0] + arr[1]
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        unsafe {
            let arr: [f64; 2] = core::mem::transmute(self.0);
            arr[0].max(arr[1])
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe {
            let arr: [f64; 2] = core::mem::transmute(self.0);
            arr[0].min(arr[1])
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        debug_assert!(index < 2);
        unsafe {
            let arr: [f64; 2] = core::mem::transmute(self.0);
            arr[index]
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        debug_assert!(index < 2);
        unsafe {
            let mut arr: [f64; 2] = core::mem::transmute(self.0);
            arr[index] = value;
            F64x2(core::mem::transmute(arr))
        }
    }
}

/// 128-bit SIMD register for f32 (4 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x4(v128);

impl SimdRegister for F32x4 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe { F32x4(f32x4_splat(0.0)) }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { F32x4(f32x4_splat(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x4(v128_load(ptr as *const v128))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x4(v128_load(ptr as *const v128))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        v128_store(ptr as *mut v128, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        v128_store(ptr as *mut v128, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F32x4(f32x4_add(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F32x4(f32x4_sub(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F32x4(f32x4_mul(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F32x4(f32x4_div(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // WASM doesn't have FMA, so emulate it
        self.mul(a).add(b)
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        b.sub(self.mul(a))
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        unsafe {
            let arr: [f32; 4] = core::mem::transmute(self.0);
            arr[0] + arr[1] + arr[2] + arr[3]
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe {
            let arr: [f32; 4] = core::mem::transmute(self.0);
            arr[0].max(arr[1]).max(arr[2]).max(arr[3])
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe {
            let arr: [f32; 4] = core::mem::transmute(self.0);
            arr[0].min(arr[1]).min(arr[2]).min(arr[3])
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        debug_assert!(index < 4);
        unsafe {
            match index {
                0 => f32x4_extract_lane::<0>(self.0),
                1 => f32x4_extract_lane::<1>(self.0),
                2 => f32x4_extract_lane::<2>(self.0),
                3 => f32x4_extract_lane::<3>(self.0),
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        debug_assert!(index < 4);
        unsafe {
            let result = match index {
                0 => f32x4_replace_lane::<0>(self.0, value),
                1 => f32x4_replace_lane::<1>(self.0, value),
                2 => f32x4_replace_lane::<2>(self.0, value),
                3 => f32x4_replace_lane::<3>(self.0, value),
                _ => core::hint::unreachable_unchecked(),
            };
            F32x4(result)
        }
    }
}

// =============================================================================
// "256-bit" emulation using two 128-bit registers
// =============================================================================

/// Emulated 256-bit register for f64 using two WASM SIMD128 registers (4 lanes).
#[derive(Clone, Copy)]
pub struct F64x4 {
    lo: v128,
    hi: v128,
}

impl SimdRegister for F64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe {
            F64x4 {
                lo: f64x2_splat(0.0),
                hi: f64x2_splat(0.0),
            }
        }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe {
            F64x4 {
                lo: f64x2_splat(value),
                hi: f64x2_splat(value),
            }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x4 {
            lo: v128_load(ptr as *const v128),
            hi: v128_load(ptr.add(2) as *const v128),
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x4 {
            lo: v128_load(ptr as *const v128),
            hi: v128_load(ptr.add(2) as *const v128),
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        v128_store(ptr as *mut v128, self.lo);
        v128_store(ptr.add(2) as *mut v128, self.hi);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        v128_store(ptr as *mut v128, self.lo);
        v128_store(ptr.add(2) as *mut v128, self.hi);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: f64x2_add(self.lo, other.lo),
                hi: f64x2_add(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: f64x2_sub(self.lo, other.lo),
                hi: f64x2_sub(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: f64x2_mul(self.lo, other.lo),
                hi: f64x2_mul(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: f64x2_div(self.lo, other.lo),
                hi: f64x2_div(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // WASM doesn't have FMA, emulate
        self.mul(a).add(b)
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        b.sub(self.mul(a))
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        unsafe {
            let lo_arr: [f64; 2] = core::mem::transmute(self.lo);
            let hi_arr: [f64; 2] = core::mem::transmute(self.hi);
            lo_arr[0] + lo_arr[1] + hi_arr[0] + hi_arr[1]
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        unsafe {
            let lo_arr: [f64; 2] = core::mem::transmute(self.lo);
            let hi_arr: [f64; 2] = core::mem::transmute(self.hi);
            lo_arr[0].max(lo_arr[1]).max(hi_arr[0]).max(hi_arr[1])
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe {
            let lo_arr: [f64; 2] = core::mem::transmute(self.lo);
            let hi_arr: [f64; 2] = core::mem::transmute(self.hi);
            lo_arr[0].min(lo_arr[1]).min(hi_arr[0]).min(hi_arr[1])
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        debug_assert!(index < 4);
        unsafe {
            if index < 2 {
                let arr: [f64; 2] = core::mem::transmute(self.lo);
                arr[index]
            } else {
                let arr: [f64; 2] = core::mem::transmute(self.hi);
                arr[index - 2]
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        debug_assert!(index < 4);
        unsafe {
            if index < 2 {
                let mut arr: [f64; 2] = core::mem::transmute(self.lo);
                arr[index] = value;
                F64x4 {
                    lo: core::mem::transmute(arr),
                    hi: self.hi,
                }
            } else {
                let mut arr: [f64; 2] = core::mem::transmute(self.hi);
                arr[index - 2] = value;
                F64x4 {
                    lo: self.lo,
                    hi: core::mem::transmute(arr),
                }
            }
        }
    }
}

/// Emulated 256-bit register for f32 using two WASM SIMD128 registers (8 lanes).
#[derive(Clone, Copy)]
pub struct F32x8 {
    lo: v128,
    hi: v128,
}

impl SimdRegister for F32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        unsafe {
            F32x8 {
                lo: f32x4_splat(0.0),
                hi: f32x4_splat(0.0),
            }
        }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe {
            F32x8 {
                lo: f32x4_splat(value),
                hi: f32x4_splat(value),
            }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x8 {
            lo: v128_load(ptr as *const v128),
            hi: v128_load(ptr.add(4) as *const v128),
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x8 {
            lo: v128_load(ptr as *const v128),
            hi: v128_load(ptr.add(4) as *const v128),
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        v128_store(ptr as *mut v128, self.lo);
        v128_store(ptr.add(4) as *mut v128, self.hi);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        v128_store(ptr as *mut v128, self.lo);
        v128_store(ptr.add(4) as *mut v128, self.hi);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: f32x4_add(self.lo, other.lo),
                hi: f32x4_add(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: f32x4_sub(self.lo, other.lo),
                hi: f32x4_sub(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: f32x4_mul(self.lo, other.lo),
                hi: f32x4_mul(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: f32x4_div(self.lo, other.lo),
                hi: f32x4_div(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // WASM doesn't have FMA, emulate
        self.mul(a).add(b)
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        b.sub(self.mul(a))
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        unsafe {
            let lo_arr: [f32; 4] = core::mem::transmute(self.lo);
            let hi_arr: [f32; 4] = core::mem::transmute(self.hi);
            lo_arr[0]
                + lo_arr[1]
                + lo_arr[2]
                + lo_arr[3]
                + hi_arr[0]
                + hi_arr[1]
                + hi_arr[2]
                + hi_arr[3]
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe {
            let lo_arr: [f32; 4] = core::mem::transmute(self.lo);
            let hi_arr: [f32; 4] = core::mem::transmute(self.hi);
            lo_arr[0]
                .max(lo_arr[1])
                .max(lo_arr[2])
                .max(lo_arr[3])
                .max(hi_arr[0])
                .max(hi_arr[1])
                .max(hi_arr[2])
                .max(hi_arr[3])
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe {
            let lo_arr: [f32; 4] = core::mem::transmute(self.lo);
            let hi_arr: [f32; 4] = core::mem::transmute(self.hi);
            lo_arr[0]
                .min(lo_arr[1])
                .min(lo_arr[2])
                .min(lo_arr[3])
                .min(hi_arr[0])
                .min(hi_arr[1])
                .min(hi_arr[2])
                .min(hi_arr[3])
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        debug_assert!(index < 8);
        unsafe {
            if index < 4 {
                match index {
                    0 => f32x4_extract_lane::<0>(self.lo),
                    1 => f32x4_extract_lane::<1>(self.lo),
                    2 => f32x4_extract_lane::<2>(self.lo),
                    3 => f32x4_extract_lane::<3>(self.lo),
                    _ => core::hint::unreachable_unchecked(),
                }
            } else {
                match index {
                    4 => f32x4_extract_lane::<0>(self.hi),
                    5 => f32x4_extract_lane::<1>(self.hi),
                    6 => f32x4_extract_lane::<2>(self.hi),
                    7 => f32x4_extract_lane::<3>(self.hi),
                    _ => core::hint::unreachable_unchecked(),
                }
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        debug_assert!(index < 8);
        unsafe {
            if index < 4 {
                let result = match index {
                    0 => f32x4_replace_lane::<0>(self.lo, value),
                    1 => f32x4_replace_lane::<1>(self.lo, value),
                    2 => f32x4_replace_lane::<2>(self.lo, value),
                    3 => f32x4_replace_lane::<3>(self.lo, value),
                    _ => core::hint::unreachable_unchecked(),
                };
                F32x8 {
                    lo: result,
                    hi: self.hi,
                }
            } else {
                let result = match index {
                    4 => f32x4_replace_lane::<0>(self.hi, value),
                    5 => f32x4_replace_lane::<1>(self.hi, value),
                    6 => f32x4_replace_lane::<2>(self.hi, value),
                    7 => f32x4_replace_lane::<3>(self.hi, value),
                    _ => core::hint::unreachable_unchecked(),
                };
                F32x8 {
                    lo: self.lo,
                    hi: result,
                }
            }
        }
    }
}

// =============================================================================
// SimdScalar implementations for WASM
// =============================================================================

impl SimdScalar for f64 {
    type Simd256 = F64x4;
    type Simd512 = F64x4; // No 512-bit on WASM, use 256-bit emulation
}

impl SimdScalar for f32 {
    type Simd256 = F32x8;
    type Simd512 = F32x8; // No 512-bit on WASM, use 256-bit emulation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64x2_basic() {
        let a = F64x2::splat(2.0);
        let b = F64x2::splat(3.0);
        let c = a.add(b);

        assert_eq!(c.extract(0), 5.0);
        assert_eq!(c.extract(1), 5.0);
    }

    #[test]
    fn test_f64x2_mul() {
        let a = F64x2::splat(2.0);
        let b = F64x2::splat(3.0);
        let c = a.mul(b);

        assert_eq!(c.extract(0), 6.0);
        assert_eq!(c.extract(1), 6.0);
    }

    #[test]
    fn test_f64x2_reduce_sum() {
        let a = F64x2::zero().insert(0, 1.0).insert(1, 2.0);
        assert_eq!(a.reduce_sum(), 3.0);
    }

    #[test]
    fn test_f64x4_emulated() {
        let a = F64x4::splat(2.0);
        let b = F64x4::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(2), 5.0);
        assert_eq!(sum.reduce_sum(), 20.0);

        // Test FMA emulation
        let c = F64x4::splat(1.0);
        let fma = a.mul_add(b, c); // 2*3 + 1 = 7
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f32x4_basic() {
        let a = F32x4::splat(2.0);
        let b = F32x4::splat(3.0);
        let c = a.add(b);

        assert_eq!(c.extract(0), 5.0);
        assert_eq!(c.extract(1), 5.0);
        assert_eq!(c.extract(2), 5.0);
        assert_eq!(c.extract(3), 5.0);
    }

    #[test]
    fn test_f32x4_reduce_sum() {
        let a = F32x4::zero()
            .insert(0, 1.0)
            .insert(1, 2.0)
            .insert(2, 3.0)
            .insert(3, 4.0);
        assert_eq!(a.reduce_sum(), 10.0);
    }

    #[test]
    fn test_f32x8_emulated() {
        let a = F32x8::splat(2.0);
        let b = F32x8::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(4), 5.0);
        assert_eq!(sum.reduce_sum(), 40.0);

        // Test FMA emulation
        let c = F32x8::splat(1.0);
        let fma = a.mul_add(b, c); // 2*3 + 1 = 7
        assert_eq!(fma.extract(0), 7.0);
    }
}
