//! AArch64 SIMD implementations using NEON.
//!
//! This module provides SIMD register types and operations for AArch64
//! processors with NEON support (always available on AArch64).

// Note: cfg(target_arch = "aarch64") is already on the module declaration in simd.rs

use crate::simd::{SimdRegister, SimdScalar};

use core::arch::aarch64::*;

/// Cold, never-inlined panic path for an out-of-range SIMD lane index.
///
/// `SimdRegister::extract` / `insert` are *safe* fns, but the underlying NEON
/// `vgetq_lane_*` / `vsetq_lane_*` intrinsics require a **compile-time-constant**
/// lane, so the runtime `index` is dispatched through a `match`. An index
/// `>= LANES` is a caller programming error; we turn it into a *defined* panic
/// (like slice indexing) instead of `core::hint::unreachable_unchecked()`, which
/// was **undefined behavior reachable from safe code** in release builds (where
/// the old `debug_assert!` was compiled out). Kept `#[cold]`/`#[inline(never)]`
/// so the valid-index path stays branch-predictable and the panic string is not
/// duplicated into every lane accessor.
#[cold]
#[inline(never)]
fn lane_index_out_of_range(index: usize, lanes: usize) -> ! {
    panic!("SIMD lane index {index} out of range (register has {lanes} lanes)");
}

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
        unsafe {
            match index {
                0 => vgetq_lane_f64(self.0, 0),
                1 => vgetq_lane_f64(self.0, 1),
                _ => lane_index_out_of_range(index, Self::LANES),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        unsafe {
            match index {
                0 => F64x2(vsetq_lane_f64(value, self.0, 0)),
                1 => F64x2(vsetq_lane_f64(value, self.0, 1)),
                _ => lane_index_out_of_range(index, Self::LANES),
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
        unsafe {
            match index {
                0 => vgetq_lane_f32(self.0, 0),
                1 => vgetq_lane_f32(self.0, 1),
                2 => vgetq_lane_f32(self.0, 2),
                3 => vgetq_lane_f32(self.0, 3),
                _ => lane_index_out_of_range(index, Self::LANES),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        unsafe {
            match index {
                0 => F32x4(vsetq_lane_f32(value, self.0, 0)),
                1 => F32x4(vsetq_lane_f32(value, self.0, 1)),
                2 => F32x4(vsetq_lane_f32(value, self.0, 2)),
                3 => F32x4(vsetq_lane_f32(value, self.0, 3)),
                _ => lane_index_out_of_range(index, Self::LANES),
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
        unsafe {
            match index {
                0 => vgetq_lane_f64(self.lo, 0),
                1 => vgetq_lane_f64(self.lo, 1),
                2 => vgetq_lane_f64(self.hi, 0),
                3 => vgetq_lane_f64(self.hi, 1),
                _ => lane_index_out_of_range(index, Self::LANES),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
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
                _ => lane_index_out_of_range(index, Self::LANES),
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
                _ => lane_index_out_of_range(index, Self::LANES),
            }
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
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
                _ => lane_index_out_of_range(index, Self::LANES),
            }
        }
    }
}

// =============================================================================
// SimdScalar implementations for AArch64
// =============================================================================

// AArch64/NEON has no true 512-bit register. `Simd512` therefore aliases the
// 256-bit *emulated* register (two 128-bit NEON lanes). The `SimdScalar` trait's
// default `LANES_512 = 64 / size_of::<Self>()` assumes a real 512-bit width and
// would report 8 (f64) / 16 (f32) lanes, contradicting the aliased register which
// only holds 4 / 8. Any loop that strided by the constant while operating on a
// `Simd512` value would run off the end. We override `LANES_512` (and, for
// symmetry/robustness, `LANES_256`) to the *actual* lane count of the concrete
// register type so the constant can never disagree with the width it describes.
impl SimdScalar for f64 {
    type Simd256 = F64x4;
    type Simd512 = F64x4; // No true 512-bit on ARM; reuse the 256-bit emulation.
    const LANES_256: usize = <F64x4 as SimdRegister>::LANES;
    const LANES_512: usize = <F64x4 as SimdRegister>::LANES;
}

impl SimdScalar for f32 {
    type Simd256 = F32x8;
    type Simd512 = F32x8; // No true 512-bit on ARM; reuse the 256-bit emulation.
    const LANES_256: usize = <F32x8 as SimdRegister>::LANES;
    const LANES_512: usize = <F32x8 as SimdRegister>::LANES;
}

// =============================================================================
// ARM SVE (Scalable Vector Extension) support
// =============================================================================
//
// SVE provides scalable vectors from 128 to 2048 bits.
// The actual vector length is implementation-defined and determined at runtime.
//
// # Toolchain requirements
//
// The SVE scalable-vector intrinsics in `core::arch::aarch64` (`svfloat64_t`,
// `svld1_f64`, `svcntb`, ...) are **unstable** and only exist on a *nightly*
// toolchain behind `#![feature(stdarch_aarch64_sve)]`. Gating this code merely
// on `target_feature = "sve"` would try to compile these intrinsics on stable
// whenever the target enables SVE (e.g. `-C target-cpu=neoverse-v1`), breaking
// the build. All SVE items below are therefore gated on
// `all(feature = "sve-nightly", target_feature = "sve")`: the `sve-nightly`
// cargo feature is **off by default** and, when enabled, requires a nightly
// compiler (see the crate-root `#![cfg_attr(..., feature(stdarch_aarch64_sve))]`).
// On stable, or with the feature off, the crate falls back to the always-present
// NEON (`F64x2`/`F32x4`) and emulated 256-bit (`F64x4`/`F32x8`) paths above.

/// SVE feature detection and utilities.
pub struct SveSupport;

impl SveSupport {
    /// Check if SVE is supported on this CPU.
    #[inline]
    pub fn is_available() -> bool {
        #[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
        {
            true
        }
        #[cfg(not(all(feature = "sve-nightly", target_feature = "sve")))]
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
        #[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
        unsafe {
            // `svcntb` returns the vector length in bytes as `u64`; widen/narrow
            // to `usize` (a real SVE vector is 16..=256 bytes, always in range).
            use core::arch::aarch64::svcntb;
            svcntb() as usize * 8
        }
        #[cfg(not(all(feature = "sve-nightly", target_feature = "sve")))]
        {
            0
        }
    }

    /// Returns the SVE vector length in bytes.
    #[inline]
    pub fn vector_length_bytes() -> usize {
        #[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
        unsafe {
            use core::arch::aarch64::svcntb;
            svcntb() as usize
        }
        #[cfg(not(all(feature = "sve-nightly", target_feature = "sve")))]
        {
            0
        }
    }

    /// Returns the number of f64 elements in an SVE vector.
    #[inline]
    pub fn f64_lanes() -> usize {
        #[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
        unsafe {
            use core::arch::aarch64::svcntd;
            svcntd() as usize
        }
        #[cfg(not(all(feature = "sve-nightly", target_feature = "sve")))]
        {
            0
        }
    }

    /// Returns the number of f32 elements in an SVE vector.
    #[inline]
    pub fn f32_lanes() -> usize {
        #[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
        unsafe {
            use core::arch::aarch64::svcntw;
            svcntw() as usize
        }
        #[cfg(not(all(feature = "sve-nightly", target_feature = "sve")))]
        {
            0
        }
    }
}

// -----------------------------------------------------------------------------
// NOTE: no `SveF64` / `SveF32` register-wrapper types are provided.
//
// A natural design would mirror the NEON `F64x2` / `F32x4` newtypes with, e.g.,
// `#[repr(transparent)] struct SveF64(svfloat64_t)`. That is **not expressible**
// in current Rust: SVE scalable vectors are *sizeless* types (their length is
// only known at run time), so they are not `Sized` and rustc rejects them as
// struct fields -- "scalable vectors cannot be fields of a struct" -- even under
// `repr(transparent)`. Wrapping them behind the `sve-nightly` feature therefore
// could never compile, so we do not ship a type that only pretends to exist.
//
// Until Rust gains first-class support for sizeless/scalable SIMD types, SVE is
// exposed only through the free-function kernels below (`sve_dot_f64`,
// `sve_dot_f32`, `sve_axpy_f64`), which keep the scalable values in locals
// inside `#[target_feature(enable = "sve")]` fns where the language permits
// them. The always-available NEON path (`F64x2` / `F32x4`) and the emulated
// 256-bit path (`F64x4` / `F32x8`) remain the default SIMD abstraction.
// -----------------------------------------------------------------------------

/// Helper for SVE-accelerated dot product.
///
/// Computes the dot product of two slices using SVE instructions.
///
/// # Safety
/// - The target CPU must actually support SVE. This function emits SVE
///   instructions with **no internal runtime guard**, so callers must confirm
///   availability first (e.g. via [`SveSupport::is_available`]).
/// - `x` and `y` must have the **same length**: the loop bound is `x.len()` and
///   `y` is read up to that index, so a shorter `y` is an out-of-bounds read
///   (checked only by `debug_assert` in debug builds).
#[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn sve_dot_f64(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::aarch64::{
        svaddv_f64, svcntd, svdup_n_f64, svld1_f64, svmla_f64_m, svmla_f64_x, svptrue_b64,
        svwhilelt_b64_u64,
    };

    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    // `svcntd` returns the f64 lane count as `u64`; use `usize` for index math.
    let lanes = svcntd() as usize;

    let mut acc = svdup_n_f64(0.0);
    let mut i = 0usize;

    // Main loop with full vectors: every lane is active, so `_x` (don't-care
    // on inactive lanes) is optimal and correct here.
    while i + lanes <= n {
        let pred = svptrue_b64();
        let va = svld1_f64(pred, x.as_ptr().add(i));
        let vb = svld1_f64(pred, y.as_ptr().add(i));
        acc = svmla_f64_x(pred, acc, va, vb);
        i += lanes;
    }

    // Tail: only the first `n - i` lanes are active. `acc` already holds valid
    // per-lane partial sums (from the main loop) in *all* lanes, and the final
    // reduction below sums *all* lanes with a full predicate. We must therefore
    // use **merging** predication (`svmla_f64_m`): active lanes get
    // `acc + va*vb`, inactive lanes keep the existing `acc`. The previous `_x`
    // form left inactive lanes *unspecified*, so garbage from the inactive lanes
    // leaked into the full-predicate reduction and corrupted the result whenever
    // `n` was not a whole multiple of the vector length. (`_z` would be equally
    // wrong here: it would zero the inactive lanes and destroy the main-loop
    // partial sums they carry.)
    if i < n {
        let pred = svwhilelt_b64_u64(i as u64, n as u64);
        let va = svld1_f64(pred, x.as_ptr().add(i));
        let vb = svld1_f64(pred, y.as_ptr().add(i));
        acc = svmla_f64_m(pred, acc, va, vb);
    }

    svaddv_f64(svptrue_b64(), acc)
}

/// Helper for SVE-accelerated dot product (f32).
///
/// # Safety
/// - The target CPU must actually support SVE (no internal runtime guard; see
///   [`SveSupport::is_available`]).
/// - `x` and `y` must have the **same length** (the loop bound is `x.len()` and
///   `y` is read up to that index; checked only by `debug_assert`).
#[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn sve_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::aarch64::{
        svaddv_f32, svcntw, svdup_n_f32, svld1_f32, svmla_f32_m, svmla_f32_x, svptrue_b32,
        svwhilelt_b32_u64,
    };

    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    // `svcntw` returns the f32 lane count as `u64`; use `usize` for index math.
    let lanes = svcntw() as usize;

    let mut acc = svdup_n_f32(0.0);
    let mut i = 0usize;

    // Main loop with full vectors: all lanes active, `_x` is correct/optimal.
    while i + lanes <= n {
        let pred = svptrue_b32();
        let va = svld1_f32(pred, x.as_ptr().add(i));
        let vb = svld1_f32(pred, y.as_ptr().add(i));
        acc = svmla_f32_x(pred, acc, va, vb);
        i += lanes;
    }

    // Tail: merging predication (`svmla_f32_m`) so the inactive lanes retain the
    // main-loop partial sums that the full-predicate reduction below still adds
    // in. See `sve_dot_f64` for the detailed rationale; `_x` left inactive lanes
    // unspecified and corrupted non-multiple-of-vector-length reductions.
    if i < n {
        let pred = svwhilelt_b32_u64(i as u64, n as u64);
        let va = svld1_f32(pred, x.as_ptr().add(i));
        let vb = svld1_f32(pred, y.as_ptr().add(i));
        acc = svmla_f32_m(pred, acc, va, vb);
    }

    svaddv_f32(svptrue_b32(), acc)
}

/// AXPY operation using SVE: y = alpha * x + y
///
/// # Safety
/// - The target CPU must actually support SVE (no internal runtime guard; see
///   [`SveSupport::is_available`]).
/// - `x` and `y` must have the **same length**: the loop bound is `x.len()` and
///   `y` is both read and written up to that index, so a shorter `y` is an
///   out-of-bounds access (checked only by `debug_assert`).
#[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn sve_axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::aarch64::{
        svcntd, svld1_f64, svmla_n_f64_x, svptrue_b64, svst1_f64, svwhilelt_b64_u64,
    };

    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    // `svcntd` returns the f64 lane count as `u64`; use `usize` for index math.
    let lanes = svcntd() as usize;
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
        let pred = svwhilelt_b64_u64(i as u64, n as u64);
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
        #[cfg(feature = "std")]
        let is_sve2 = SveSupport::is_sve2_available();
        let vlen_bits = SveSupport::vector_length_bits();
        let vlen_bytes = SveSupport::vector_length_bytes();
        let f64_lanes = SveSupport::f64_lanes();
        let f32_lanes = SveSupport::f32_lanes();

        #[cfg(feature = "std")]
        {
            println!("SVE available: {}", is_available);
            println!("SVE2 available: {}", is_sve2);
            println!("Vector length: {} bits / {} bytes", vlen_bits, vlen_bytes);
            println!("f64 lanes: {}, f32 lanes: {}", f64_lanes, f32_lanes);
        }

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

    // --- Regression: Finding 1 (out-of-range lane index must be a *defined*
    // panic, not UB via `unreachable_unchecked()` in release builds) ---------

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_f64x2_extract_out_of_range_panics() {
        let a = F64x2::splat(1.0);
        // Index 2 is out of range for a 2-lane register. This must panic with a
        // clear message on *every* profile, not invoke `unreachable_unchecked`.
        let _ = a.extract(2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_f32x8_insert_out_of_range_panics() {
        let a = F32x8::splat(1.0);
        // Index 8 is out of range for the 8-lane emulated register.
        let _ = a.insert(8, 9.0);
    }

    #[test]
    fn test_f64x4_all_lanes_roundtrip() {
        // Guard against regressions in the lane dispatch after removing the
        // `debug_assert!`: every in-range lane must still read/write correctly,
        // including across the emulated lo/hi 128-bit boundary (lanes 1 -> 2).
        let mut v = F64x4::zero();
        for lane in 0..F64x4::LANES {
            v = v.insert(lane, (lane as f64) + 0.5);
        }
        for lane in 0..F64x4::LANES {
            assert_eq!(v.extract(lane), (lane as f64) + 0.5);
        }
    }

    // --- Regression: Finding 2 (LANES_256/LANES_512 must equal the actual
    // width of the concrete Simd256/Simd512 register type on NEON) -----------

    #[test]
    fn test_simd_lane_constants_match_register_width() {
        // On AArch64 the 512-bit alias is the 4-lane (f64) / 8-lane (f32)
        // emulation, so the trait-default LANES_512 (8 / 16) would over-report
        // the width. These asserts fail on the pre-fix code.
        assert_eq!(
            <f64 as SimdScalar>::LANES_512,
            <<f64 as SimdScalar>::Simd512 as SimdRegister>::LANES
        );
        assert_eq!(
            <f32 as SimdScalar>::LANES_512,
            <<f32 as SimdScalar>::Simd512 as SimdRegister>::LANES
        );
        assert_eq!(
            <f64 as SimdScalar>::LANES_256,
            <<f64 as SimdScalar>::Simd256 as SimdRegister>::LANES
        );
        assert_eq!(
            <f32 as SimdScalar>::LANES_256,
            <<f32 as SimdScalar>::Simd256 as SimdRegister>::LANES
        );
        // Concrete expected values for the NEON emulation.
        assert_eq!(<f64 as SimdScalar>::LANES_512, 4);
        assert_eq!(<f32 as SimdScalar>::LANES_512, 8);
    }

    // --- Regression: Finding 4 (SVE dot-product tail must use merging
    // predication so inactive lanes do not corrupt the final reduction).
    //
    // This test only compiles/runs on a nightly toolchain with the
    // `sve-nightly` feature and an SVE-enabled target; it exercises a length
    // that is deliberately *not* a multiple of the (runtime) vector length so
    // the predicated tail path is taken. On the pre-fix `_x` code the inactive
    // lanes were unspecified and this assertion would fail. --------------------

    #[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
    #[test]
    fn test_sve_dot_f64_partial_tail_matches_scalar() {
        let lanes = SveSupport::f64_lanes();
        // Guarantee a non-empty, non-full tail: 3 full vectors + 1 element.
        let n = lanes * 3 + 1;
        let x: std::vec::Vec<f64> = (0..n).map(|k| (k as f64) * 0.5 + 1.0).collect();
        let y: std::vec::Vec<f64> = (0..n).map(|k| (k as f64).mul_add(-0.25, 2.0)).collect();
        let expected: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let got = unsafe { sve_dot_f64(&x, &y) };
        assert!(
            (got - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "sve_dot_f64 tail mismatch: got {got}, expected {expected}"
        );
    }

    #[cfg(all(feature = "sve-nightly", target_feature = "sve"))]
    #[test]
    fn test_sve_dot_f32_partial_tail_matches_scalar() {
        let lanes = SveSupport::f32_lanes();
        let n = lanes * 3 + 1;
        let x: std::vec::Vec<f32> = (0..n).map(|k| (k as f32) * 0.5 + 1.0).collect();
        let y: std::vec::Vec<f32> = (0..n).map(|k| (k as f32).mul_add(-0.25, 2.0)).collect();
        let expected: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let got = unsafe { sve_dot_f32(&x, &y) };
        assert!(
            (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "sve_dot_f32 tail mismatch: got {got}, expected {expected}"
        );
    }
}
