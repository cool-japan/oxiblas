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

/// Reports an out-of-range SIMD lane index as a *defined* panic.
///
/// `f32x4_extract_lane` / `f32x4_replace_lane` require a **compile-time-constant**
/// lane, so the runtime `index` is dispatched through a `match`. An index
/// `>= LANES` is a caller programming error; we turn it into a panic (like slice
/// indexing) instead of `core::hint::unreachable_unchecked()`, which was
/// **undefined behavior reachable from safe code** in release builds — the
/// `SimdRegister::extract` / `insert` trait methods are safe, and the
/// `debug_assert!` that used to guard them is compiled out of release profiles.
/// Kept `#[cold]`/`#[inline(never)]` so the valid-index path stays
/// branch-predictable and the panic string is not duplicated into every lane
/// accessor. Mirrors the `aarch64`/`x86_64` helpers of the same name.
#[cold]
#[inline(never)]
fn lane_index_out_of_range(index: usize, lanes: usize) -> ! {
    panic!("SIMD lane index {index} out of range (register has {lanes} lanes)");
}

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

    /// Computes `self * a + b`.
    ///
    /// **Not a genuine fused multiply-add.** WASM SIMD128 exposes no native
    /// FMA instruction, so this is computed as a separate multiply followed
    /// by a separate add — two roundings, not the single rounding of a true
    /// hardware FMA (e.g. AVX2+FMA or NEON `vfma`). Results can differ from
    /// a real FMA in the last bit or two.
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // NOTE: WASM SIMD128 has no fused multiply-add instruction. This is
        // an UNFUSED multiply-then-add (two roundings), slightly less
        // accurate than a genuine single-rounding FMA.
        self.mul(a).add(b)
    }

    /// Computes `self * a - b`.
    ///
    /// **Not a genuine fused multiply-subtract** — see [`Self::mul_add`]:
    /// WASM SIMD128 has no FMA instruction, so this is an unfused
    /// multiply-then-subtract (two roundings).
    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    /// Computes `-(self * a) + b = b - self * a`.
    ///
    /// **Not a genuine fused negative-multiply-add** — see
    /// [`Self::mul_add`]: WASM SIMD128 has no FMA instruction, so this is
    /// an unfused multiply-then-subtract (two roundings).
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

    /// Computes `self * a + b`.
    ///
    /// **Not a genuine fused multiply-add.** WASM SIMD128 exposes no native
    /// FMA instruction, so this is computed as a separate multiply followed
    /// by a separate add — two roundings, not the single rounding of a true
    /// hardware FMA (e.g. AVX2+FMA or NEON `vfma`). Results can differ from
    /// a real FMA in the last bit or two.
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // NOTE: WASM SIMD128 has no fused multiply-add instruction. This is
        // an UNFUSED multiply-then-add (two roundings), slightly less
        // accurate than a genuine single-rounding FMA.
        self.mul(a).add(b)
    }

    /// Computes `self * a - b`.
    ///
    /// **Not a genuine fused multiply-subtract** — see [`Self::mul_add`]:
    /// WASM SIMD128 has no FMA instruction, so this is an unfused
    /// multiply-then-subtract (two roundings).
    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    /// Computes `-(self * a) + b = b - self * a`.
    ///
    /// **Not a genuine fused negative-multiply-add** — see
    /// [`Self::mul_add`]: WASM SIMD128 has no FMA instruction, so this is
    /// an unfused multiply-then-subtract (two roundings).
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
        match index {
            0 => f32x4_extract_lane::<0>(self.0),
            1 => f32x4_extract_lane::<1>(self.0),
            2 => f32x4_extract_lane::<2>(self.0),
            3 => f32x4_extract_lane::<3>(self.0),
            _ => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        let result = match index {
            0 => f32x4_replace_lane::<0>(self.0, value),
            1 => f32x4_replace_lane::<1>(self.0, value),
            2 => f32x4_replace_lane::<2>(self.0, value),
            3 => f32x4_replace_lane::<3>(self.0, value),
            _ => lane_index_out_of_range(index, Self::LANES),
        };
        F32x4(result)
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

    /// Computes `self * a + b`.
    ///
    /// **Not a genuine fused multiply-add.** WASM SIMD128 exposes no native
    /// FMA instruction, so this is computed as a separate multiply followed
    /// by a separate add — two roundings, not the single rounding of a true
    /// hardware FMA (e.g. AVX2+FMA or NEON `vfma`). Results can differ from
    /// a real FMA in the last bit or two.
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // NOTE: WASM SIMD128 has no fused multiply-add instruction. This is
        // an UNFUSED multiply-then-add (two roundings), slightly less
        // accurate than a genuine single-rounding FMA.
        self.mul(a).add(b)
    }

    /// Computes `self * a - b`.
    ///
    /// **Not a genuine fused multiply-subtract** — see [`Self::mul_add`]:
    /// WASM SIMD128 has no FMA instruction, so this is an unfused
    /// multiply-then-subtract (two roundings).
    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    /// Computes `-(self * a) + b = b - self * a`.
    ///
    /// **Not a genuine fused negative-multiply-add** — see
    /// [`Self::mul_add`]: WASM SIMD128 has no FMA instruction, so this is
    /// an unfused multiply-then-subtract (two roundings).
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

    /// Computes `self * a + b`.
    ///
    /// **Not a genuine fused multiply-add.** WASM SIMD128 exposes no native
    /// FMA instruction, so this is computed as a separate multiply followed
    /// by a separate add — two roundings, not the single rounding of a true
    /// hardware FMA (e.g. AVX2+FMA or NEON `vfma`). Results can differ from
    /// a real FMA in the last bit or two.
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // NOTE: WASM SIMD128 has no fused multiply-add instruction. This is
        // an UNFUSED multiply-then-add (two roundings), slightly less
        // accurate than a genuine single-rounding FMA.
        self.mul(a).add(b)
    }

    /// Computes `self * a - b`.
    ///
    /// **Not a genuine fused multiply-subtract** — see [`Self::mul_add`]:
    /// WASM SIMD128 has no FMA instruction, so this is an unfused
    /// multiply-then-subtract (two roundings).
    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self.mul(a).sub(b)
    }

    /// Computes `-(self * a) + b = b - self * a`.
    ///
    /// **Not a genuine fused negative-multiply-add** — see
    /// [`Self::mul_add`]: WASM SIMD128 has no FMA instruction, so this is
    /// an unfused multiply-then-subtract (two roundings).
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
        if index < 4 {
            match index {
                0 => f32x4_extract_lane::<0>(self.lo),
                1 => f32x4_extract_lane::<1>(self.lo),
                2 => f32x4_extract_lane::<2>(self.lo),
                3 => f32x4_extract_lane::<3>(self.lo),
                _ => lane_index_out_of_range(index, Self::LANES),
            }
        } else {
            match index {
                4 => f32x4_extract_lane::<0>(self.hi),
                5 => f32x4_extract_lane::<1>(self.hi),
                6 => f32x4_extract_lane::<2>(self.hi),
                7 => f32x4_extract_lane::<3>(self.hi),
                _ => lane_index_out_of_range(index, Self::LANES),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        if index < 4 {
            let result = match index {
                0 => f32x4_replace_lane::<0>(self.lo, value),
                1 => f32x4_replace_lane::<1>(self.lo, value),
                2 => f32x4_replace_lane::<2>(self.lo, value),
                3 => f32x4_replace_lane::<3>(self.lo, value),
                _ => lane_index_out_of_range(index, Self::LANES),
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
                _ => lane_index_out_of_range(index, Self::LANES),
            };
            F32x8 {
                lo: self.lo,
                hi: result,
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

    // --- Regression: out-of-range lane index must be a *defined* panic, not UB
    // via `unreachable_unchecked()` in release builds (the `debug_assert!` that
    // used to "guard" these safe trait methods is compiled out of release) ----

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_f32x4_extract_out_of_range_panics() {
        let a = F32x4::splat(1.0);
        // Index 4 is out of range for a 4-lane register.
        let _ = a.extract(4);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_f32x4_insert_out_of_range_panics() {
        let a = F32x4::splat(1.0);
        let _ = a.insert(4, 9.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_f32x8_extract_out_of_range_panics() {
        let a = F32x8::splat(1.0);
        // Index 8 is out of range for the 8-lane emulated register.
        let _ = a.extract(8);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_f32x8_insert_out_of_range_panics() {
        let a = F32x8::splat(1.0);
        let _ = a.insert(8, 9.0);
    }

    #[test]
    fn test_f32x8_all_lanes_roundtrip() {
        // Guard against regressions in the lane dispatch after removing the
        // `debug_assert!`: every in-range lane must still read/write correctly,
        // including across the emulated lo/hi 128-bit boundary (lanes 3 -> 4).
        let mut v = F32x8::zero();
        for lane in 0..F32x8::LANES {
            v = v.insert(lane, (lane as f32) + 0.5);
        }
        for lane in 0..F32x8::LANES {
            assert_eq!(v.extract(lane), (lane as f32) + 0.5);
        }
    }

    #[test]
    fn test_f32x4_all_lanes_roundtrip() {
        let mut v = F32x4::zero();
        for lane in 0..F32x4::LANES {
            v = v.insert(lane, (lane as f32) + 0.5);
        }
        for lane in 0..F32x4::LANES {
            assert_eq!(v.extract(lane), (lane as f32) + 0.5);
        }
    }
}
