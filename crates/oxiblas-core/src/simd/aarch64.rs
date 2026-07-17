//! AArch64 SIMD implementations using NEON.
//!
//! This module provides SIMD register types and operations for AArch64
//! processors with NEON support (always available on AArch64).

// Note: cfg(target_arch = "aarch64") is already on the module declaration in simd.rs

use crate::simd::{SimdRegister, SimdScalar};

use core::arch::aarch64::*;

// =============================================================================
// NEON (128-bit) implementations
// =============================================================================

/// 128-bit SIMD register for f64 (2 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x2(float64x2_t);

impl SimdRegister for F64x2 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn zero() -> Self {
        unsafe { F64x2(vdupq_n_f64(0.0)) }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { F64x2(vdupq_n_f64(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x2(vld1q_f64(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x2(vld1q_f64(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        vst1q_f64(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        vst1q_f64(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F64x2(vaddq_f64(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F64x2(vsubq_f64(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F64x2(vmulq_f64(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F64x2(vdivq_f64(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // FMA: self * a + b
        unsafe { F64x2(vfmaq_f64(b.0, self.0, a.0)) }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        // self * a - b = -(b - self * a)
        unsafe { F64x2(vnegq_f64(vfmsq_f64(b.0, self.0, a.0))) }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        // -(self * a) + b = b - self * a
        unsafe { F64x2(vfmsq_f64(b.0, self.0, a.0)) }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        unsafe { vaddvq_f64(self.0) }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        unsafe { vmaxvq_f64(self.0) }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe { vminvq_f64(self.0) }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        debug_assert!(index < 2);
        unsafe {
            match index {
                0 => vgetq_lane_f64(self.0, 0),
                1 => vgetq_lane_f64(self.0, 1),
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        debug_assert!(index < 2);
        unsafe {
            match index {
                0 => F64x2(vsetq_lane_f64(value, self.0, 0)),
                1 => F64x2(vsetq_lane_f64(value, self.0, 1)),
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }
}

/// 128-bit SIMD register for f32 (4 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x4(float32x4_t);

impl SimdRegister for F32x4 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe { F32x4(vdupq_n_f32(0.0)) }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { F32x4(vdupq_n_f32(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x4(vld1q_f32(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x4(vld1q_f32(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        vst1q_f32(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        vst1q_f32(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F32x4(vaddq_f32(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F32x4(vsubq_f32(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F32x4(vmulq_f32(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F32x4(vdivq_f32(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { F32x4(vfmaq_f32(b.0, self.0, a.0)) }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        unsafe { F32x4(vnegq_f32(vfmsq_f32(b.0, self.0, a.0))) }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        unsafe { F32x4(vfmsq_f32(b.0, self.0, a.0)) }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        unsafe { vaddvq_f32(self.0) }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe { vmaxvq_f32(self.0) }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe { vminvq_f32(self.0) }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        debug_assert!(index < 4);
        unsafe {
            match index {
                0 => vgetq_lane_f32(self.0, 0),
                1 => vgetq_lane_f32(self.0, 1),
                2 => vgetq_lane_f32(self.0, 2),
                3 => vgetq_lane_f32(self.0, 3),
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        debug_assert!(index < 4);
        unsafe {
            match index {
                0 => F32x4(vsetq_lane_f32(value, self.0, 0)),
                1 => F32x4(vsetq_lane_f32(value, self.0, 1)),
                2 => F32x4(vsetq_lane_f32(value, self.0, 2)),
                3 => F32x4(vsetq_lane_f32(value, self.0, 3)),
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }
}

// =============================================================================
// "256-bit" emulation using two 128-bit registers
// =============================================================================

/// Emulated 256-bit register for f64 using two NEON registers (4 lanes).
#[derive(Clone, Copy)]
pub struct F64x4 {
    lo: float64x2_t,
    hi: float64x2_t,
}

impl SimdRegister for F64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe {
            F64x4 {
                lo: vdupq_n_f64(0.0),
                hi: vdupq_n_f64(0.0),
            }
        }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe {
            F64x4 {
                lo: vdupq_n_f64(value),
                hi: vdupq_n_f64(value),
            }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x4 {
            lo: vld1q_f64(ptr),
            hi: vld1q_f64(ptr.add(2)),
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x4 {
            lo: vld1q_f64(ptr),
            hi: vld1q_f64(ptr.add(2)),
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        vst1q_f64(ptr, self.lo);
        vst1q_f64(ptr.add(2), self.hi);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        vst1q_f64(ptr, self.lo);
        vst1q_f64(ptr.add(2), self.hi);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: vaddq_f64(self.lo, other.lo),
                hi: vaddq_f64(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: vsubq_f64(self.lo, other.lo),
                hi: vsubq_f64(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: vmulq_f64(self.lo, other.lo),
                hi: vmulq_f64(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe {
            F64x4 {
                lo: vdivq_f64(self.lo, other.lo),
                hi: vdivq_f64(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            F64x4 {
                lo: vfmaq_f64(b.lo, self.lo, a.lo),
                hi: vfmaq_f64(b.hi, self.hi, a.hi),
            }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        unsafe {
            F64x4 {
                lo: vnegq_f64(vfmsq_f64(b.lo, self.lo, a.lo)),
                hi: vnegq_f64(vfmsq_f64(b.hi, self.hi, a.hi)),
            }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            F64x4 {
                lo: vfmsq_f64(b.lo, self.lo, a.lo),
                hi: vfmsq_f64(b.hi, self.hi, a.hi),
            }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        unsafe { vaddvq_f64(self.lo) + vaddvq_f64(self.hi) }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        unsafe {
            let max_lo = vmaxvq_f64(self.lo);
            let max_hi = vmaxvq_f64(self.hi);
            if max_lo > max_hi { max_lo } else { max_hi }
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe {
            let min_lo = vminvq_f64(self.lo);
            let min_hi = vminvq_f64(self.hi);
            if min_lo < min_hi { min_lo } else { min_hi }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        debug_assert!(index < 4);
        unsafe {
            match index {
                0 => vgetq_lane_f64(self.lo, 0),
                1 => vgetq_lane_f64(self.lo, 1),
                2 => vgetq_lane_f64(self.hi, 0),
                3 => vgetq_lane_f64(self.hi, 1),
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        debug_assert!(index < 4);
        unsafe {
            match index {
                0 => F64x4 {
                    lo: vsetq_lane_f64(value, self.lo, 0),
                    hi: self.hi,
                },
                1 => F64x4 {
                    lo: vsetq_lane_f64(value, self.lo, 1),
                    hi: self.hi,
                },
                2 => F64x4 {
                    lo: self.lo,
                    hi: vsetq_lane_f64(value, self.hi, 0),
                },
                3 => F64x4 {
                    lo: self.lo,
                    hi: vsetq_lane_f64(value, self.hi, 1),
                },
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }
}

/// Emulated 256-bit register for f32 using two NEON registers (8 lanes).
#[derive(Clone, Copy)]
pub struct F32x8 {
    lo: float32x4_t,
    hi: float32x4_t,
}

impl SimdRegister for F32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        unsafe {
            F32x8 {
                lo: vdupq_n_f32(0.0),
                hi: vdupq_n_f32(0.0),
            }
        }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe {
            F32x8 {
                lo: vdupq_n_f32(value),
                hi: vdupq_n_f32(value),
            }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x8 {
            lo: vld1q_f32(ptr),
            hi: vld1q_f32(ptr.add(4)),
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x8 {
            lo: vld1q_f32(ptr),
            hi: vld1q_f32(ptr.add(4)),
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        vst1q_f32(ptr, self.lo);
        vst1q_f32(ptr.add(4), self.hi);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        vst1q_f32(ptr, self.lo);
        vst1q_f32(ptr.add(4), self.hi);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: vaddq_f32(self.lo, other.lo),
                hi: vaddq_f32(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: vsubq_f32(self.lo, other.lo),
                hi: vsubq_f32(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: vmulq_f32(self.lo, other.lo),
                hi: vmulq_f32(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe {
            F32x8 {
                lo: vdivq_f32(self.lo, other.lo),
                hi: vdivq_f32(self.hi, other.hi),
            }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            F32x8 {
                lo: vfmaq_f32(b.lo, self.lo, a.lo),
                hi: vfmaq_f32(b.hi, self.hi, a.hi),
            }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        unsafe {
            F32x8 {
                lo: vnegq_f32(vfmsq_f32(b.lo, self.lo, a.lo)),
                hi: vnegq_f32(vfmsq_f32(b.hi, self.hi, a.hi)),
            }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            F32x8 {
                lo: vfmsq_f32(b.lo, self.lo, a.lo),
                hi: vfmsq_f32(b.hi, self.hi, a.hi),
            }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        unsafe { vaddvq_f32(self.lo) + vaddvq_f32(self.hi) }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe {
            let max_lo = vmaxvq_f32(self.lo);
            let max_hi = vmaxvq_f32(self.hi);
            if max_lo > max_hi { max_lo } else { max_hi }
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe {
            let min_lo = vminvq_f32(self.lo);
            let min_hi = vminvq_f32(self.hi);
            if min_lo < min_hi { min_lo } else { min_hi }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        debug_assert!(index < 8);
        unsafe {
            match index {
                0 => vgetq_lane_f32(self.lo, 0),
                1 => vgetq_lane_f32(self.lo, 1),
                2 => vgetq_lane_f32(self.lo, 2),
                3 => vgetq_lane_f32(self.lo, 3),
                4 => vgetq_lane_f32(self.hi, 0),
                5 => vgetq_lane_f32(self.hi, 1),
                6 => vgetq_lane_f32(self.hi, 2),
                7 => vgetq_lane_f32(self.hi, 3),
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        debug_assert!(index < 8);
        unsafe {
            match index {
                0 => F32x8 {
                    lo: vsetq_lane_f32(value, self.lo, 0),
                    hi: self.hi,
                },
                1 => F32x8 {
                    lo: vsetq_lane_f32(value, self.lo, 1),
                    hi: self.hi,
                },
                2 => F32x8 {
                    lo: vsetq_lane_f32(value, self.lo, 2),
                    hi: self.hi,
                },
                3 => F32x8 {
                    lo: vsetq_lane_f32(value, self.lo, 3),
                    hi: self.hi,
                },
                4 => F32x8 {
                    lo: self.lo,
                    hi: vsetq_lane_f32(value, self.hi, 0),
                },
                5 => F32x8 {
                    lo: self.lo,
                    hi: vsetq_lane_f32(value, self.hi, 1),
                },
                6 => F32x8 {
                    lo: self.lo,
                    hi: vsetq_lane_f32(value, self.hi, 2),
                },
                7 => F32x8 {
                    lo: self.lo,
                    hi: vsetq_lane_f32(value, self.hi, 3),
                },
                _ => core::hint::unreachable_unchecked(),
            }
        }
    }
}

// =============================================================================
// SimdScalar implementations for AArch64
// =============================================================================

impl SimdScalar for f64 {
    type Simd256 = F64x4;
    type Simd512 = F64x4; // No 512-bit on ARM, use 256-bit emulation
}

impl SimdScalar for f32 {
    type Simd256 = F32x8;
    type Simd512 = F32x8; // No 512-bit on ARM, use 256-bit emulation
}

// =============================================================================
// ARM SVE (Scalable Vector Extension) support
// =============================================================================
//
// SVE provides scalable vectors from 128 to 2048 bits.
// The actual vector length is implementation-defined and determined at runtime.

/// SVE feature detection and utilities.
pub struct SveSupport;

impl SveSupport {
    /// Check if SVE is supported on this CPU.
    #[inline]
    pub fn is_available() -> bool {
        #[cfg(target_feature = "sve")]
        {
            true
        }
        #[cfg(not(target_feature = "sve"))]
        {
            // Runtime detection - SVE is indicated by ID_AA64ZFR0_EL1 register
            // For now, we use a simple approach
            false
        }
    }

    /// Check if SVE2 is supported on this CPU.
    #[inline]
    pub fn is_sve2_available() -> bool {
        #[cfg(target_feature = "sve2")]
        {
            true
        }
        #[cfg(not(target_feature = "sve2"))]
        {
            false
        }
    }

    /// Returns the SVE vector length in bits.
    ///
    /// Returns 0 if SVE is not available.
    #[inline]
    pub fn vector_length_bits() -> usize {
        #[cfg(target_feature = "sve")]
        unsafe {
            // Use svcntb() to get vector length in bytes
            use core::arch::aarch64::svcntb;
            svcntb() * 8
        }
        #[cfg(not(target_feature = "sve"))]
        {
            0
        }
    }

    /// Returns the SVE vector length in bytes.
    #[inline]
    pub fn vector_length_bytes() -> usize {
        #[cfg(target_feature = "sve")]
        unsafe {
            use core::arch::aarch64::svcntb;
            svcntb()
        }
        #[cfg(not(target_feature = "sve"))]
        {
            0
        }
    }

    /// Returns the number of f64 elements in an SVE vector.
    #[inline]
    pub fn f64_lanes() -> usize {
        #[cfg(target_feature = "sve")]
        unsafe {
            use core::arch::aarch64::svcntd;
            svcntd()
        }
        #[cfg(not(target_feature = "sve"))]
        {
            0
        }
    }

    /// Returns the number of f32 elements in an SVE vector.
    #[inline]
    pub fn f32_lanes() -> usize {
        #[cfg(target_feature = "sve")]
        unsafe {
            use core::arch::aarch64::svcntw;
            svcntw()
        }
        #[cfg(not(target_feature = "sve"))]
        {
            0
        }
    }
}

/// SVE scalable vector for f64.
///
/// This type wraps the scalable `svfloat64_t` type and provides
/// SIMD operations that work regardless of the actual vector length.
#[cfg(target_feature = "sve")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct SveF64(core::arch::aarch64::svfloat64_t);

#[cfg(target_feature = "sve")]
impl SveF64 {
    /// Creates a vector with all elements set to zero.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn zero() -> Self {
        use core::arch::aarch64::{svdup_n_f64, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svdup_n_f64(0.0))
    }

    /// Creates a vector with all elements set to the same value.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn splat(value: f64) -> Self {
        use core::arch::aarch64::svdup_n_f64;
        SveF64(svdup_n_f64(value))
    }

    /// Loads a vector from memory.
    ///
    /// # Safety
    /// The pointer must be valid for at least `SveSupport::f64_lanes()` elements.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn load(ptr: *const f64) -> Self {
        use core::arch::aarch64::{svld1_f64, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svld1_f64(pred, ptr))
    }

    /// Stores a vector to memory.
    ///
    /// # Safety
    /// The pointer must be valid for at least `SveSupport::f64_lanes()` elements.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn store(self, ptr: *mut f64) {
        use core::arch::aarch64::{svptrue_b64, svst1_f64};
        let pred = svptrue_b64();
        svst1_f64(pred, ptr, self.0);
    }

    /// Element-wise addition.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn add(self, other: Self) -> Self {
        use core::arch::aarch64::{svadd_f64_x, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svadd_f64_x(pred, self.0, other.0))
    }

    /// Element-wise subtraction.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn sub(self, other: Self) -> Self {
        use core::arch::aarch64::{svptrue_b64, svsub_f64_x};
        let pred = svptrue_b64();
        SveF64(svsub_f64_x(pred, self.0, other.0))
    }

    /// Element-wise multiplication.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn mul(self, other: Self) -> Self {
        use core::arch::aarch64::{svmul_f64_x, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svmul_f64_x(pred, self.0, other.0))
    }

    /// Element-wise division.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn div(self, other: Self) -> Self {
        use core::arch::aarch64::{svdiv_f64_x, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svdiv_f64_x(pred, self.0, other.0))
    }

    /// Fused multiply-add: self * a + b
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn fma(self, a: Self, b: Self) -> Self {
        use core::arch::aarch64::{svmla_f64_x, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svmla_f64_x(pred, b.0, self.0, a.0))
    }

    /// Fused multiply-subtract: self * a - b
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn fms(self, a: Self, b: Self) -> Self {
        use core::arch::aarch64::{svnmls_f64_x, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svnmls_f64_x(pred, b.0, self.0, a.0))
    }

    /// Horizontal sum of all elements.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn reduce_add(self) -> f64 {
        use core::arch::aarch64::{svaddv_f64, svptrue_b64};
        let pred = svptrue_b64();
        svaddv_f64(pred, self.0)
    }

    /// Horizontal maximum.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn reduce_max(self) -> f64 {
        use core::arch::aarch64::{svmaxv_f64, svptrue_b64};
        let pred = svptrue_b64();
        svmaxv_f64(pred, self.0)
    }

    /// Horizontal minimum.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn reduce_min(self) -> f64 {
        use core::arch::aarch64::{svminv_f64, svptrue_b64};
        let pred = svptrue_b64();
        svminv_f64(pred, self.0)
    }

    /// Absolute value.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn abs(self) -> Self {
        use core::arch::aarch64::{svabs_f64_x, svptrue_b64};
        let pred = svptrue_b64();
        SveF64(svabs_f64_x(pred, self.0))
    }

    /// Square root.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn sqrt(self) -> Self {
        use core::arch::aarch64::{svptrue_b64, svsqrt_f64_x};
        let pred = svptrue_b64();
        SveF64(svsqrt_f64_x(pred, self.0))
    }
}

/// SVE scalable vector for f32.
#[cfg(target_feature = "sve")]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct SveF32(core::arch::aarch64::svfloat32_t);

#[cfg(target_feature = "sve")]
impl SveF32 {
    /// Creates a vector with all elements set to zero.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn zero() -> Self {
        use core::arch::aarch64::svdup_n_f32;
        SveF32(svdup_n_f32(0.0))
    }

    /// Creates a vector with all elements set to the same value.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn splat(value: f32) -> Self {
        use core::arch::aarch64::svdup_n_f32;
        SveF32(svdup_n_f32(value))
    }

    /// Loads a vector from memory.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn load(ptr: *const f32) -> Self {
        use core::arch::aarch64::{svld1_f32, svptrue_b32};
        let pred = svptrue_b32();
        SveF32(svld1_f32(pred, ptr))
    }

    /// Stores a vector to memory.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn store(self, ptr: *mut f32) {
        use core::arch::aarch64::{svptrue_b32, svst1_f32};
        let pred = svptrue_b32();
        svst1_f32(pred, ptr, self.0);
    }

    /// Element-wise addition.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn add(self, other: Self) -> Self {
        use core::arch::aarch64::{svadd_f32_x, svptrue_b32};
        let pred = svptrue_b32();
        SveF32(svadd_f32_x(pred, self.0, other.0))
    }

    /// Element-wise subtraction.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn sub(self, other: Self) -> Self {
        use core::arch::aarch64::{svptrue_b32, svsub_f32_x};
        let pred = svptrue_b32();
        SveF32(svsub_f32_x(pred, self.0, other.0))
    }

    /// Element-wise multiplication.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn mul(self, other: Self) -> Self {
        use core::arch::aarch64::{svmul_f32_x, svptrue_b32};
        let pred = svptrue_b32();
        SveF32(svmul_f32_x(pred, self.0, other.0))
    }

    /// Element-wise division.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn div(self, other: Self) -> Self {
        use core::arch::aarch64::{svdiv_f32_x, svptrue_b32};
        let pred = svptrue_b32();
        SveF32(svdiv_f32_x(pred, self.0, other.0))
    }

    /// Fused multiply-add: self * a + b
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn fma(self, a: Self, b: Self) -> Self {
        use core::arch::aarch64::{svmla_f32_x, svptrue_b32};
        let pred = svptrue_b32();
        SveF32(svmla_f32_x(pred, b.0, self.0, a.0))
    }

    /// Horizontal sum of all elements.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn reduce_add(self) -> f32 {
        use core::arch::aarch64::{svaddv_f32, svptrue_b32};
        let pred = svptrue_b32();
        svaddv_f32(pred, self.0)
    }

    /// Horizontal maximum.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn reduce_max(self) -> f32 {
        use core::arch::aarch64::{svmaxv_f32, svptrue_b32};
        let pred = svptrue_b32();
        svmaxv_f32(pred, self.0)
    }

    /// Horizontal minimum.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn reduce_min(self) -> f32 {
        use core::arch::aarch64::{svminv_f32, svptrue_b32};
        let pred = svptrue_b32();
        svminv_f32(pred, self.0)
    }

    /// Absolute value.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn abs(self) -> Self {
        use core::arch::aarch64::{svabs_f32_x, svptrue_b32};
        let pred = svptrue_b32();
        SveF32(svabs_f32_x(pred, self.0))
    }

    /// Square root.
    #[inline]
    #[target_feature(enable = "sve")]
    pub unsafe fn sqrt(self) -> Self {
        use core::arch::aarch64::{svptrue_b32, svsqrt_f32_x};
        let pred = svptrue_b32();
        SveF32(svsqrt_f32_x(pred, self.0))
    }
}

/// Helper for SVE-accelerated dot product.
///
/// Computes the dot product of two slices using SVE instructions.
#[cfg(target_feature = "sve")]
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn sve_dot_f64(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::aarch64::{
        svaddv_f64, svcntd, svdup_n_f64, svld1_f64, svmla_f64_x, svptrue_b64, svwhilelt_b64,
    };

    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let lanes = svcntd();

    let mut acc = svdup_n_f64(0.0);
    let mut i = 0usize;

    // Main loop with full vectors
    while i + lanes <= n {
        let pred = svptrue_b64();
        let va = svld1_f64(pred, x.as_ptr().add(i));
        let vb = svld1_f64(pred, y.as_ptr().add(i));
        acc = svmla_f64_x(pred, acc, va, vb);
        i += lanes;
    }

    // Handle remaining elements with predicate
    if i < n {
        let pred = svwhilelt_b64(i as u64, n as u64);
        let va = svld1_f64(pred, x.as_ptr().add(i));
        let vb = svld1_f64(pred, y.as_ptr().add(i));
        acc = svmla_f64_x(pred, acc, va, vb);
    }

    svaddv_f64(svptrue_b64(), acc)
}

/// Helper for SVE-accelerated dot product (f32).
#[cfg(target_feature = "sve")]
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn sve_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::aarch64::{
        svaddv_f32, svcntw, svdup_n_f32, svld1_f32, svmla_f32_x, svptrue_b32, svwhilelt_b32,
    };

    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let lanes = svcntw();

    let mut acc = svdup_n_f32(0.0);
    let mut i = 0usize;

    while i + lanes <= n {
        let pred = svptrue_b32();
        let va = svld1_f32(pred, x.as_ptr().add(i));
        let vb = svld1_f32(pred, y.as_ptr().add(i));
        acc = svmla_f32_x(pred, acc, va, vb);
        i += lanes;
    }

    if i < n {
        let pred = svwhilelt_b32(i as u64, n as u64);
        let va = svld1_f32(pred, x.as_ptr().add(i));
        let vb = svld1_f32(pred, y.as_ptr().add(i));
        acc = svmla_f32_x(pred, acc, va, vb);
    }

    svaddv_f32(svptrue_b32(), acc)
}

/// AXPY operation using SVE: y = alpha * x + y
#[cfg(target_feature = "sve")]
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn sve_axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::aarch64::{
        svcntd, svdup_n_f64, svld1_f64, svmla_n_f64_x, svptrue_b64, svst1_f64, svwhilelt_b64,
    };

    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let lanes = svcntd();
    let mut i = 0usize;

    while i + lanes <= n {
        let pred = svptrue_b64();
        let vx = svld1_f64(pred, x.as_ptr().add(i));
        let vy = svld1_f64(pred, y.as_ptr().add(i));
        let result = svmla_n_f64_x(pred, vy, vx, alpha);
        svst1_f64(pred, y.as_mut_ptr().add(i), result);
        i += lanes;
    }

    if i < n {
        let pred = svwhilelt_b64(i as u64, n as u64);
        let vx = svld1_f64(pred, x.as_ptr().add(i));
        let vy = svld1_f64(pred, y.as_ptr().add(i));
        let result = svmla_n_f64_x(pred, vy, vx, alpha);
        svst1_f64(pred, y.as_mut_ptr().add(i), result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64x2_basic() {
        let a = F64x2::splat(2.0);
        let b = F64x2::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(1), 5.0);

        let prod = a.mul(b);
        assert_eq!(prod.extract(0), 6.0);

        // Test FMA
        let c = F64x2::splat(1.0);
        let fma = a.mul_add(b, c); // 2*3 + 1 = 7
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f64x4_emulated() {
        let a = F64x4::splat(2.0);
        let b = F64x4::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(2), 5.0);

        assert_eq!(sum.reduce_sum(), 20.0);
    }

    #[test]
    fn test_f32x4_basic() {
        let a = F32x4::splat(2.0);
        let b = F32x4::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.reduce_sum(), 20.0);
    }

    #[test]
    fn test_sve_support_detection() {
        // Test that SVE detection doesn't panic
        let is_available = SveSupport::is_available();
        let is_sve2 = SveSupport::is_sve2_available();
        let vlen_bits = SveSupport::vector_length_bits();
        let vlen_bytes = SveSupport::vector_length_bytes();
        let f64_lanes = SveSupport::f64_lanes();
        let f32_lanes = SveSupport::f32_lanes();

        println!("SVE available: {}", is_available);
        println!("SVE2 available: {}", is_sve2);
        println!("Vector length: {} bits / {} bytes", vlen_bits, vlen_bytes);
        println!("f64 lanes: {}, f32 lanes: {}", f64_lanes, f32_lanes);

        // If SVE is not available, all values should be 0
        if !is_available {
            assert_eq!(vlen_bits, 0);
            assert_eq!(vlen_bytes, 0);
            assert_eq!(f64_lanes, 0);
            assert_eq!(f32_lanes, 0);
        } else {
            // SVE vectors are at least 128 bits
            assert!(vlen_bits >= 128);
            assert!(vlen_bytes >= 16);
            assert!(f64_lanes >= 2);
            assert!(f32_lanes >= 4);
        }
    }
}
