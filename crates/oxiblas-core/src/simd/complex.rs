//! Complex number SIMD types.
//!
//! This module provides SIMD types for complex numbers in interleaved format.
//! Complex numbers are stored as `[re0, im0, re1, im1, ...]` which allows
//! efficient SIMD operations.
//!
//! # Layout
//!
//! - `C64x2`: 2 complex f64 values using 256-bit register (4 f64 lanes)
//! - `C32x4`: 4 complex f32 values using 256-bit register (8 f32 lanes)
//!
//! # Operations
//!
//! Complex multiplication: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`
//! This requires shuffle operations to separate real/imaginary parts.

use num_complex::{Complex32, Complex64};

/// Trait for complex SIMD register types.
pub trait ComplexSimdRegister: Copy + Clone {
    /// The underlying real scalar type.
    type Real;
    /// The complex scalar type.
    type Complex;
    /// Number of complex values in this register.
    const COMPLEX_LANES: usize;

    /// Creates a register with all complex values set to zero.
    fn zero() -> Self;

    /// Creates a register with all complex values set to the same value.
    fn splat(value: Self::Complex) -> Self;

    /// Loads complex values from aligned memory.
    ///
    /// # Safety
    /// The pointer must be aligned and point to at least `COMPLEX_LANES` complex values.
    unsafe fn load_aligned(ptr: *const Self::Complex) -> Self;

    /// Loads complex values from unaligned memory.
    ///
    /// # Safety
    /// The pointer must point to at least `COMPLEX_LANES` complex values.
    unsafe fn load_unaligned(ptr: *const Self::Complex) -> Self;

    /// Stores complex values to aligned memory.
    ///
    /// # Safety
    /// The pointer must be aligned and point to space for at least `COMPLEX_LANES` complex values.
    unsafe fn store_aligned(self, ptr: *mut Self::Complex);

    /// Stores complex values to unaligned memory.
    ///
    /// # Safety
    /// The pointer must point to space for at least `COMPLEX_LANES` complex values.
    unsafe fn store_unaligned(self, ptr: *mut Self::Complex);

    /// Complex addition.
    fn add(self, other: Self) -> Self;

    /// Complex subtraction.
    fn sub(self, other: Self) -> Self;

    /// Complex multiplication.
    fn mul(self, other: Self) -> Self;

    /// Multiplies by a real scalar.
    fn scale_real(self, scalar: Self::Real) -> Self;

    /// Conjugate: negates the imaginary part.
    fn conj(self) -> Self;

    /// Extracts a complex value at the given index.
    fn extract(self, index: usize) -> Self::Complex;

    /// Inserts a complex value at the given index.
    fn insert(self, index: usize, value: Self::Complex) -> Self;

    /// Horizontal sum of all complex values.
    fn reduce_sum(self) -> Self::Complex;
}

// =============================================================================
// Scalar fallback for complex SIMD
// =============================================================================

/// Scalar "register" for Complex64 - processes one complex value at a time.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ScalarC64(pub Complex64);

impl ComplexSimdRegister for ScalarC64 {
    type Real = f64;
    type Complex = Complex64;
    const COMPLEX_LANES: usize = 1;

    #[inline]
    fn zero() -> Self {
        ScalarC64(Complex64::new(0.0, 0.0))
    }

    #[inline]
    fn splat(value: Complex64) -> Self {
        ScalarC64(value)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const Complex64) -> Self {
        ScalarC64(*ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const Complex64) -> Self {
        ScalarC64(*ptr)
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut Complex64) {
        *ptr = self.0;
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut Complex64) {
        *ptr = self.0;
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        ScalarC64(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        ScalarC64(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        ScalarC64(self.0 * other.0)
    }

    #[inline]
    fn scale_real(self, scalar: f64) -> Self {
        ScalarC64(Complex64::new(self.0.re * scalar, self.0.im * scalar))
    }

    #[inline]
    fn conj(self) -> Self {
        ScalarC64(self.0.conj())
    }

    #[inline]
    fn extract(self, _index: usize) -> Complex64 {
        self.0
    }

    #[inline]
    fn insert(self, _index: usize, value: Complex64) -> Self {
        ScalarC64(value)
    }

    #[inline]
    fn reduce_sum(self) -> Complex64 {
        self.0
    }
}

/// Scalar "register" for Complex32 - processes one complex value at a time.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ScalarC32(pub Complex32);

impl ComplexSimdRegister for ScalarC32 {
    type Real = f32;
    type Complex = Complex32;
    const COMPLEX_LANES: usize = 1;

    #[inline]
    fn zero() -> Self {
        ScalarC32(Complex32::new(0.0, 0.0))
    }

    #[inline]
    fn splat(value: Complex32) -> Self {
        ScalarC32(value)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const Complex32) -> Self {
        ScalarC32(*ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const Complex32) -> Self {
        ScalarC32(*ptr)
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut Complex32) {
        *ptr = self.0;
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut Complex32) {
        *ptr = self.0;
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        ScalarC32(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        ScalarC32(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        ScalarC32(self.0 * other.0)
    }

    #[inline]
    fn scale_real(self, scalar: f32) -> Self {
        ScalarC32(Complex32::new(self.0.re * scalar, self.0.im * scalar))
    }

    #[inline]
    fn conj(self) -> Self {
        ScalarC32(self.0.conj())
    }

    #[inline]
    fn extract(self, _index: usize) -> Complex32 {
        self.0
    }

    #[inline]
    fn insert(self, _index: usize, value: Complex32) -> Self {
        ScalarC32(value)
    }

    #[inline]
    fn reduce_sum(self) -> Complex32 {
        self.0
    }
}

// =============================================================================
// AArch64 NEON complex SIMD
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod aarch64_impl {
    use super::*;
    use core::arch::aarch64::*;

    /// 2 complex f64 values using two 128-bit NEON registers.
    #[derive(Clone, Copy)]
    pub struct C64x2 {
        /// First complex value (re0, im0)
        c0: float64x2_t,
        /// Second complex value (re1, im1)
        c1: float64x2_t,
    }

    impl ComplexSimdRegister for C64x2 {
        type Real = f64;
        type Complex = Complex64;
        const COMPLEX_LANES: usize = 2;

        #[inline]
        fn zero() -> Self {
            unsafe {
                C64x2 {
                    c0: vdupq_n_f64(0.0),
                    c1: vdupq_n_f64(0.0),
                }
            }
        }

        #[inline]
        fn splat(value: Complex64) -> Self {
            unsafe {
                let c = vld1q_f64([value.re, value.im].as_ptr());
                C64x2 { c0: c, c1: c }
            }
        }

        #[inline]
        unsafe fn load_aligned(ptr: *const Complex64) -> Self {
            let p = ptr as *const f64;
            C64x2 {
                c0: vld1q_f64(p),
                c1: vld1q_f64(p.add(2)),
            }
        }

        #[inline]
        unsafe fn load_unaligned(ptr: *const Complex64) -> Self {
            Self::load_aligned(ptr)
        }

        #[inline]
        unsafe fn store_aligned(self, ptr: *mut Complex64) {
            let p = ptr as *mut f64;
            vst1q_f64(p, self.c0);
            vst1q_f64(p.add(2), self.c1);
        }

        #[inline]
        unsafe fn store_unaligned(self, ptr: *mut Complex64) {
            self.store_aligned(ptr);
        }

        #[inline]
        fn add(self, other: Self) -> Self {
            unsafe {
                C64x2 {
                    c0: vaddq_f64(self.c0, other.c0),
                    c1: vaddq_f64(self.c1, other.c1),
                }
            }
        }

        #[inline]
        fn sub(self, other: Self) -> Self {
            unsafe {
                C64x2 {
                    c0: vsubq_f64(self.c0, other.c0),
                    c1: vsubq_f64(self.c1, other.c1),
                }
            }
        }

        #[inline]
        fn mul(self, other: Self) -> Self {
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            unsafe {
                // For c0: self.c0 = [a, b], other.c0 = [c, d]
                let a = vdupq_laneq_f64(self.c0, 0); // [a, a]
                let b = vdupq_laneq_f64(self.c0, 1); // [b, b]
                let c = vdupq_laneq_f64(other.c0, 0); // [c, c]
                let d = vdupq_laneq_f64(other.c0, 1); // [d, d]

                // ac, ad
                let ac = vmulq_f64(a, c);
                let ad = vmulq_f64(a, d);
                // bd, bc
                let bd = vmulq_f64(b, d);
                let bc = vmulq_f64(b, c);

                // [ac - bd, ad + bc]
                let re0 = vsubq_f64(ac, bd);
                let im0 = vaddq_f64(ad, bc);
                let c0_new = vzip1q_f64(re0, im0);

                // Same for c1
                let a1 = vdupq_laneq_f64(self.c1, 0);
                let b1 = vdupq_laneq_f64(self.c1, 1);
                let c1 = vdupq_laneq_f64(other.c1, 0);
                let d1 = vdupq_laneq_f64(other.c1, 1);

                let ac1 = vmulq_f64(a1, c1);
                let ad1 = vmulq_f64(a1, d1);
                let bd1 = vmulq_f64(b1, d1);
                let bc1 = vmulq_f64(b1, c1);

                let re1 = vsubq_f64(ac1, bd1);
                let im1 = vaddq_f64(ad1, bc1);
                let c1_new = vzip1q_f64(re1, im1);

                C64x2 {
                    c0: c0_new,
                    c1: c1_new,
                }
            }
        }

        #[inline]
        fn scale_real(self, scalar: f64) -> Self {
            unsafe {
                let s = vdupq_n_f64(scalar);
                C64x2 {
                    c0: vmulq_f64(self.c0, s),
                    c1: vmulq_f64(self.c1, s),
                }
            }
        }

        #[inline]
        fn conj(self) -> Self {
            unsafe {
                // Negate imaginary parts: [re, -im]
                let neg_mask = vld1q_f64([1.0, -1.0].as_ptr());
                C64x2 {
                    c0: vmulq_f64(self.c0, neg_mask),
                    c1: vmulq_f64(self.c1, neg_mask),
                }
            }
        }

        #[inline]
        fn extract(self, index: usize) -> Complex64 {
            debug_assert!(index < 2);
            unsafe {
                let arr = if index == 0 {
                    let mut a = [0.0f64; 2];
                    vst1q_f64(a.as_mut_ptr(), self.c0);
                    a
                } else {
                    let mut a = [0.0f64; 2];
                    vst1q_f64(a.as_mut_ptr(), self.c1);
                    a
                };
                Complex64::new(arr[0], arr[1])
            }
        }

        #[inline]
        fn insert(self, index: usize, value: Complex64) -> Self {
            debug_assert!(index < 2);
            unsafe {
                let new_c = vld1q_f64([value.re, value.im].as_ptr());
                if index == 0 {
                    C64x2 {
                        c0: new_c,
                        c1: self.c1,
                    }
                } else {
                    C64x2 {
                        c0: self.c0,
                        c1: new_c,
                    }
                }
            }
        }

        #[inline]
        fn reduce_sum(self) -> Complex64 {
            unsafe {
                let sum = vaddq_f64(self.c0, self.c1);
                let mut arr = [0.0f64; 2];
                vst1q_f64(arr.as_mut_ptr(), sum);
                Complex64::new(arr[0], arr[1])
            }
        }
    }

    /// 4 complex f32 values using two 128-bit NEON registers.
    #[derive(Clone, Copy)]
    pub struct C32x4 {
        /// First two complex values (re0, im0, re1, im1)
        lo: float32x4_t,
        /// Second two complex values (re2, im2, re3, im3)
        hi: float32x4_t,
    }

    impl ComplexSimdRegister for C32x4 {
        type Real = f32;
        type Complex = Complex32;
        const COMPLEX_LANES: usize = 4;

        #[inline]
        fn zero() -> Self {
            unsafe {
                C32x4 {
                    lo: vdupq_n_f32(0.0),
                    hi: vdupq_n_f32(0.0),
                }
            }
        }

        #[inline]
        fn splat(value: Complex32) -> Self {
            unsafe {
                let vals = [value.re, value.im, value.re, value.im];
                let v = vld1q_f32(vals.as_ptr());
                C32x4 { lo: v, hi: v }
            }
        }

        #[inline]
        unsafe fn load_aligned(ptr: *const Complex32) -> Self {
            let p = ptr as *const f32;
            C32x4 {
                lo: vld1q_f32(p),
                hi: vld1q_f32(p.add(4)),
            }
        }

        #[inline]
        unsafe fn load_unaligned(ptr: *const Complex32) -> Self {
            Self::load_aligned(ptr)
        }

        #[inline]
        unsafe fn store_aligned(self, ptr: *mut Complex32) {
            let p = ptr as *mut f32;
            vst1q_f32(p, self.lo);
            vst1q_f32(p.add(4), self.hi);
        }

        #[inline]
        unsafe fn store_unaligned(self, ptr: *mut Complex32) {
            self.store_aligned(ptr);
        }

        #[inline]
        fn add(self, other: Self) -> Self {
            unsafe {
                C32x4 {
                    lo: vaddq_f32(self.lo, other.lo),
                    hi: vaddq_f32(self.hi, other.hi),
                }
            }
        }

        #[inline]
        fn sub(self, other: Self) -> Self {
            unsafe {
                C32x4 {
                    lo: vsubq_f32(self.lo, other.lo),
                    hi: vsubq_f32(self.hi, other.hi),
                }
            }
        }

        #[inline]
        fn mul(self, other: Self) -> Self {
            // For each pair [a, b] * [c, d] = [ac-bd, ad+bc]
            // Manual implementation using shuffle and FMA
            unsafe {
                // lo = [a0, b0, a1, b1], other.lo = [c0, d0, c1, d1]
                // We need: [a0*c0 - b0*d0, a0*d0 + b0*c0, a1*c1 - b1*d1, a1*d1 + b1*c1]

                // Extract real and imaginary parts using zip/unzip
                // uzp1 gives [a0, a1, c0, c1] when applied to two interleaved vectors
                // uzp2 gives [b0, b1, d0, d1]

                // For lo register:
                let reals_self_lo = vuzp1q_f32(self.lo, self.lo); // [a0, a1, a0, a1]
                let imags_self_lo = vuzp2q_f32(self.lo, self.lo); // [b0, b1, b0, b1]
                let reals_other_lo = vuzp1q_f32(other.lo, other.lo); // [c0, c1, c0, c1]
                let imags_other_lo = vuzp2q_f32(other.lo, other.lo); // [d0, d1, d0, d1]

                // ac, bd, ad, bc
                let ac_lo = vmulq_f32(reals_self_lo, reals_other_lo);
                let bd_lo = vmulq_f32(imags_self_lo, imags_other_lo);
                let ad_lo = vmulq_f32(reals_self_lo, imags_other_lo);
                let bc_lo = vmulq_f32(imags_self_lo, reals_other_lo);

                // ac - bd (real part), ad + bc (imag part)
                let re_lo = vsubq_f32(ac_lo, bd_lo);
                let im_lo = vaddq_f32(ad_lo, bc_lo);

                // Interleave back: [re0, im0, re1, im1]
                let lo_result = vzip1q_f32(re_lo, im_lo);

                // Same for hi register
                let reals_self_hi = vuzp1q_f32(self.hi, self.hi);
                let imags_self_hi = vuzp2q_f32(self.hi, self.hi);
                let reals_other_hi = vuzp1q_f32(other.hi, other.hi);
                let imags_other_hi = vuzp2q_f32(other.hi, other.hi);

                let ac_hi = vmulq_f32(reals_self_hi, reals_other_hi);
                let bd_hi = vmulq_f32(imags_self_hi, imags_other_hi);
                let ad_hi = vmulq_f32(reals_self_hi, imags_other_hi);
                let bc_hi = vmulq_f32(imags_self_hi, reals_other_hi);

                let re_hi = vsubq_f32(ac_hi, bd_hi);
                let im_hi = vaddq_f32(ad_hi, bc_hi);
                let hi_result = vzip1q_f32(re_hi, im_hi);

                C32x4 {
                    lo: lo_result,
                    hi: hi_result,
                }
            }
        }

        #[inline]
        fn scale_real(self, scalar: f32) -> Self {
            unsafe {
                let s = vdupq_n_f32(scalar);
                C32x4 {
                    lo: vmulq_f32(self.lo, s),
                    hi: vmulq_f32(self.hi, s),
                }
            }
        }

        #[inline]
        fn conj(self) -> Self {
            unsafe {
                let neg_mask = vld1q_f32([1.0, -1.0, 1.0, -1.0].as_ptr());
                C32x4 {
                    lo: vmulq_f32(self.lo, neg_mask),
                    hi: vmulq_f32(self.hi, neg_mask),
                }
            }
        }

        #[inline]
        fn extract(self, index: usize) -> Complex32 {
            debug_assert!(index < 4);
            unsafe {
                let mut arr = [0.0f32; 8];
                vst1q_f32(arr.as_mut_ptr(), self.lo);
                vst1q_f32(arr.as_mut_ptr().add(4), self.hi);
                Complex32::new(arr[index * 2], arr[index * 2 + 1])
            }
        }

        #[inline]
        fn insert(self, index: usize, value: Complex32) -> Self {
            debug_assert!(index < 4);
            unsafe {
                let mut arr = [0.0f32; 8];
                vst1q_f32(arr.as_mut_ptr(), self.lo);
                vst1q_f32(arr.as_mut_ptr().add(4), self.hi);
                arr[index * 2] = value.re;
                arr[index * 2 + 1] = value.im;
                C32x4 {
                    lo: vld1q_f32(arr.as_ptr()),
                    hi: vld1q_f32(arr.as_ptr().add(4)),
                }
            }
        }

        #[inline]
        fn reduce_sum(self) -> Complex32 {
            unsafe {
                let sum = vaddq_f32(self.lo, self.hi);
                // sum = [a, b, c, d] where (a,b) and (c,d) are complex
                let mut arr = [0.0f32; 4];
                vst1q_f32(arr.as_mut_ptr(), sum);
                Complex32::new(arr[0] + arr[2], arr[1] + arr[3])
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use aarch64_impl::{C32x4, C64x2};

// =============================================================================
// x86_64 AVX2 / AVX-512 complex SIMD
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "x86_64")]
pub use x86_64::{C32x4, C32x8, C64x2, C64x4};

// =============================================================================
// Trait for complex scalar types to select SIMD register
// =============================================================================

/// Trait for complex scalar types with associated SIMD register types.
///
/// Mirrors the real-scalar [`SimdScalar`](crate::simd::SimdScalar) trait: it
/// exposes both a 256-bit and a 512-bit register type. The concrete types are
/// selected at compile time per target architecture:
///
/// | Target  | `Simd256`      | `Simd512`      |
/// |---------|----------------|----------------|
/// | x86_64  | AVX2           | AVX-512F       |
/// | aarch64 | NEON (2×128)   | NEON (2×128)\* |
/// | other   | scalar         | scalar         |
///
/// \* AArch64 has no 512-bit ISA, so `Simd512` reuses the NEON type, matching
/// the convention used by the real-scalar SIMD layer.
///
/// The associated types are a compile-time *capability ceiling*; whether a
/// given register is actually used is decided at runtime by the dispatch
/// helpers in this module (see [`complex64_mul`] and friends), which fall back
/// to scalar when the CPU lacks the required feature.
pub trait ComplexSimdScalar: Copy {
    /// 256-bit SIMD register type for this complex scalar (AVX2 / NEON).
    type Simd256: ComplexSimdRegister<Complex = Self, Real = Self::Real>;
    /// 512-bit SIMD register type for this complex scalar (AVX-512F).
    type Simd512: ComplexSimdRegister<Complex = Self, Real = Self::Real>;
    /// The underlying real scalar type (`f32` or `f64`).
    type Real: Copy;
}

impl ComplexSimdScalar for Complex64 {
    type Real = f64;

    #[cfg(target_arch = "aarch64")]
    type Simd256 = C64x2;
    #[cfg(target_arch = "aarch64")]
    type Simd512 = C64x2;

    #[cfg(target_arch = "x86_64")]
    type Simd256 = C64x2;
    #[cfg(target_arch = "x86_64")]
    type Simd512 = C64x4;

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    type Simd256 = ScalarC64;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    type Simd512 = ScalarC64;
}

impl ComplexSimdScalar for Complex32 {
    type Real = f32;

    #[cfg(target_arch = "aarch64")]
    type Simd256 = C32x4;
    #[cfg(target_arch = "aarch64")]
    type Simd512 = C32x4;

    #[cfg(target_arch = "x86_64")]
    type Simd256 = C32x4;
    #[cfg(target_arch = "x86_64")]
    type Simd512 = C32x8;

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    type Simd256 = ScalarC32;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    type Simd512 = ScalarC32;
}

// =============================================================================
// Runtime-dispatched complex SIMD batch operations
// =============================================================================
//
// These free functions are the *safe* public entry points for complex SIMD.
// Each one queries [`detect_simd_level`] at runtime and selects the widest
// register the CPU actually supports, falling back to scalar otherwise. This is
// why the compile-time `Simd256` / `Simd512` associated types above can name
// AVX2 / AVX-512 unconditionally on x86_64 without risking illegal-instruction
// faults: the register is only ever constructed inside a branch guarded by the
// matching runtime capability check.

use crate::simd::{SimdLevel, detect_simd_level};

/// Which complex SIMD backend a batch operation will use on this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexSimdBackend {
    /// Scalar fallback (one complex value at a time).
    Scalar,
    /// 128-bit SIMD (AArch64 NEON).
    Simd128,
    /// 256-bit SIMD (x86_64 AVX2).
    Simd256,
    /// 512-bit SIMD (x86_64 AVX-512F).
    Simd512,
}

/// Returns the complex SIMD backend selected for the current CPU at runtime.
///
/// This reflects the exact branch the batch helpers ([`complex64_mul`] etc.)
/// will take, so tests and diagnostics can confirm that a genuine SIMD path is
/// active (and not a silent scalar no-op) when the hardware supports it.
#[inline]
pub fn complex_simd_backend() -> ComplexSimdBackend {
    match detect_simd_level() {
        SimdLevel::Simd512 => {
            #[cfg(target_arch = "x86_64")]
            {
                ComplexSimdBackend::Simd512
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                ComplexSimdBackend::Simd128
            }
        }
        SimdLevel::Simd256 => {
            #[cfg(target_arch = "x86_64")]
            {
                ComplexSimdBackend::Simd256
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                ComplexSimdBackend::Simd128
            }
        }
        SimdLevel::Simd128 => {
            #[cfg(target_arch = "aarch64")]
            {
                ComplexSimdBackend::Simd128
            }
            // On x86_64, 128-bit (SSE-only, no AVX2) has no complex SIMD kernel,
            // so we fall back to scalar rather than claim a SIMD path.
            #[cfg(not(target_arch = "aarch64"))]
            {
                ComplexSimdBackend::Scalar
            }
        }
        SimdLevel::Scalar => ComplexSimdBackend::Scalar,
    }
}

/// Applies a binary vector op over aligned chunks, with a scalar tail.
///
/// # Safety
/// The register type `R` must correspond to a SIMD feature the current CPU
/// supports; callers guarantee this via runtime dispatch.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
unsafe fn zip_map<R>(
    a: &[R::Complex],
    b: &[R::Complex],
    out: &mut [R::Complex],
    vop: impl Fn(R, R) -> R,
    sop: impl Fn(R::Complex, R::Complex) -> R::Complex,
) where
    R: ComplexSimdRegister,
    R::Complex: Copy,
{
    let n = a.len().min(b.len()).min(out.len());
    let lanes = R::COMPLEX_LANES;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    if lanes > 1 {
        while i + lanes <= n {
            let va = R::load_unaligned(a_ptr.add(i));
            let vb = R::load_unaligned(b_ptr.add(i));
            vop(va, vb).store_unaligned(out_ptr.add(i));
            i += lanes;
        }
    }
    while i < n {
        *out_ptr.add(i) = sop(*a_ptr.add(i), *b_ptr.add(i));
        i += 1;
    }
}

/// Applies a unary vector op over aligned chunks, with a scalar tail.
///
/// # Safety
/// See [`zip_map`].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
unsafe fn map1<R>(
    a: &[R::Complex],
    out: &mut [R::Complex],
    vop: impl Fn(R) -> R,
    sop: impl Fn(R::Complex) -> R::Complex,
) where
    R: ComplexSimdRegister,
    R::Complex: Copy,
{
    let n = a.len().min(out.len());
    let lanes = R::COMPLEX_LANES;
    let a_ptr = a.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    if lanes > 1 {
        while i + lanes <= n {
            let va = R::load_unaligned(a_ptr.add(i));
            vop(va).store_unaligned(out_ptr.add(i));
            i += lanes;
        }
    }
    while i < n {
        *out_ptr.add(i) = sop(*a_ptr.add(i));
        i += 1;
    }
}

/// Scalar binary map (no panic path: iteration is bounded by the shortest slice).
#[inline]
fn zip_map_scalar<C: Copy>(a: &[C], b: &[C], out: &mut [C], sop: impl Fn(C, C) -> C) {
    for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *o = sop(x, y);
    }
}

/// Scalar unary map (no panic path: iteration is bounded by the shortest slice).
#[inline]
fn map1_scalar<C: Copy>(a: &[C], out: &mut [C], sop: impl Fn(C) -> C) {
    for (o, &x) in out.iter_mut().zip(a.iter()) {
        *o = sop(x);
    }
}

/// Generates a runtime-dispatched element-wise binary complex batch function.
macro_rules! define_complex_binop {
    ($(#[$meta:meta])* $fn_name:ident, $scalar:ty, $method:ident, $scalar_op:expr) => {
        $(#[$meta])*
        pub fn $fn_name(a: &[$scalar], b: &[$scalar], out: &mut [$scalar]) {
            let _level = detect_simd_level();
            #[cfg(target_arch = "x86_64")]
            {
                unsafe {
                    match _level {
                        SimdLevel::Simd512 => zip_map::<<$scalar as ComplexSimdScalar>::Simd512>(
                            a, b, out, |x, y| x.$method(y), $scalar_op,
                        ),
                        SimdLevel::Simd256 => zip_map::<<$scalar as ComplexSimdScalar>::Simd256>(
                            a, b, out, |x, y| x.$method(y), $scalar_op,
                        ),
                        _ => zip_map_scalar(a, b, out, $scalar_op),
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    match _level {
                        SimdLevel::Scalar => zip_map_scalar(a, b, out, $scalar_op),
                        _ => zip_map::<<$scalar as ComplexSimdScalar>::Simd256>(
                            a, b, out, |x, y| x.$method(y), $scalar_op,
                        ),
                    }
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                zip_map_scalar(a, b, out, $scalar_op);
            }
        }
    };
}

/// Generates a runtime-dispatched element-wise unary complex batch function.
macro_rules! define_complex_unop {
    ($(#[$meta:meta])* $fn_name:ident, $scalar:ty, $method:ident, $scalar_op:expr) => {
        $(#[$meta])*
        pub fn $fn_name(a: &[$scalar], out: &mut [$scalar]) {
            let _level = detect_simd_level();
            #[cfg(target_arch = "x86_64")]
            {
                unsafe {
                    match _level {
                        SimdLevel::Simd512 => map1::<<$scalar as ComplexSimdScalar>::Simd512>(
                            a, out, |x| x.$method(), $scalar_op,
                        ),
                        SimdLevel::Simd256 => map1::<<$scalar as ComplexSimdScalar>::Simd256>(
                            a, out, |x| x.$method(), $scalar_op,
                        ),
                        _ => map1_scalar(a, out, $scalar_op),
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    match _level {
                        SimdLevel::Scalar => map1_scalar(a, out, $scalar_op),
                        _ => map1::<<$scalar as ComplexSimdScalar>::Simd256>(
                            a, out, |x| x.$method(), $scalar_op,
                        ),
                    }
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                map1_scalar(a, out, $scalar_op);
            }
        }
    };
}

define_complex_binop!(
    /// Element-wise complex addition `out[i] = a[i] + b[i]` (`Complex64`).
    complex64_add, Complex64, add, |x: Complex64, y: Complex64| x + y
);
define_complex_binop!(
    /// Element-wise complex subtraction `out[i] = a[i] - b[i]` (`Complex64`).
    complex64_sub, Complex64, sub, |x: Complex64, y: Complex64| x - y
);
define_complex_binop!(
    /// Element-wise complex multiplication `out[i] = a[i] * b[i]` (`Complex64`).
    complex64_mul, Complex64, mul, |x: Complex64, y: Complex64| x * y
);
define_complex_binop!(
    /// Element-wise complex addition `out[i] = a[i] + b[i]` (`Complex32`).
    complex32_add, Complex32, add, |x: Complex32, y: Complex32| x + y
);
define_complex_binop!(
    /// Element-wise complex subtraction `out[i] = a[i] - b[i]` (`Complex32`).
    complex32_sub, Complex32, sub, |x: Complex32, y: Complex32| x - y
);
define_complex_binop!(
    /// Element-wise complex multiplication `out[i] = a[i] * b[i]` (`Complex32`).
    complex32_mul, Complex32, mul, |x: Complex32, y: Complex32| x * y
);

define_complex_unop!(
    /// Element-wise complex conjugation `out[i] = conj(a[i])` (`Complex64`).
    complex64_conj, Complex64, conj, |x: Complex64| x.conj()
);
define_complex_unop!(
    /// Element-wise complex conjugation `out[i] = conj(a[i])` (`Complex32`).
    complex32_conj, Complex32, conj, |x: Complex32| x.conj()
);

/// Element-wise real scaling `out[i] = a[i] * scalar` (`Complex64`).
pub fn complex64_scale_real(a: &[Complex64], scalar: f64, out: &mut [Complex64]) {
    let sop = move |x: Complex64| Complex64::new(x.re * scalar, x.im * scalar);
    let _level = detect_simd_level();
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            match _level {
                SimdLevel::Simd512 => map1::<C64x4>(a, out, |x| x.scale_real(scalar), sop),
                SimdLevel::Simd256 => map1::<C64x2>(a, out, |x| x.scale_real(scalar), sop),
                _ => map1_scalar(a, out, sop),
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            match _level {
                SimdLevel::Scalar => map1_scalar(a, out, sop),
                _ => map1::<C64x2>(a, out, |x| x.scale_real(scalar), sop),
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        map1_scalar(a, out, sop);
    }
}

/// Element-wise real scaling `out[i] = a[i] * scalar` (`Complex32`).
pub fn complex32_scale_real(a: &[Complex32], scalar: f32, out: &mut [Complex32]) {
    let sop = move |x: Complex32| Complex32::new(x.re * scalar, x.im * scalar);
    let _level = detect_simd_level();
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            match _level {
                SimdLevel::Simd512 => map1::<C32x8>(a, out, |x| x.scale_real(scalar), sop),
                SimdLevel::Simd256 => map1::<C32x4>(a, out, |x| x.scale_real(scalar), sop),
                _ => map1_scalar(a, out, sop),
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            match _level {
                SimdLevel::Scalar => map1_scalar(a, out, sop),
                _ => map1::<C32x4>(a, out, |x| x.scale_real(scalar), sop),
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        map1_scalar(a, out, sop);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_scalar_c64_basic() {
        let a = ScalarC64::splat(Complex64::new(2.0, 3.0));
        let b = ScalarC64::splat(Complex64::new(4.0, 5.0));

        // Addition
        let sum = a.add(b);
        assert_eq!(sum.0, Complex64::new(6.0, 8.0));

        // Multiplication: (2+3i)(4+5i) = 8 + 10i + 12i - 15 = -7 + 22i
        let prod = a.mul(b);
        assert_eq!(prod.0, Complex64::new(-7.0, 22.0));

        // Conjugate
        let conj = a.conj();
        assert_eq!(conj.0, Complex64::new(2.0, -3.0));
    }

    #[test]
    fn test_scalar_c32_basic() {
        let a = ScalarC32::splat(Complex32::new(1.0, 2.0));
        let b = ScalarC32::splat(Complex32::new(3.0, 4.0));

        let sum = a.add(b);
        assert_eq!(sum.0, Complex32::new(4.0, 6.0));

        // (1+2i)(3+4i) = 3 + 4i + 6i - 8 = -5 + 10i
        let prod = a.mul(b);
        assert_eq!(prod.0, Complex32::new(-5.0, 10.0));
    }

    #[test]
    fn test_scalar_scale_real() {
        let a = ScalarC64::splat(Complex64::new(2.0, 3.0));
        let scaled = a.scale_real(2.0);
        assert_eq!(scaled.0, Complex64::new(4.0, 6.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_c64x2_basic() {
        let a = C64x2::splat(Complex64::new(2.0, 3.0));
        let b = C64x2::splat(Complex64::new(4.0, 5.0));

        let sum = a.add(b);
        assert_eq!(sum.extract(0), Complex64::new(6.0, 8.0));
        assert_eq!(sum.extract(1), Complex64::new(6.0, 8.0));

        let conj = a.conj();
        assert_eq!(conj.extract(0), Complex64::new(2.0, -3.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_c64x2_mul() {
        let a = C64x2::splat(Complex64::new(2.0, 3.0));
        let b = C64x2::splat(Complex64::new(4.0, 5.0));

        // (2+3i)(4+5i) = 8 + 10i + 12i - 15 = -7 + 22i
        let prod = a.mul(b);
        let result = prod.extract(0);

        assert!((result.re - (-7.0)).abs() < 1e-10);
        assert!((result.im - 22.0).abs() < 1e-10);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_c64x2_reduce_sum() {
        let a = C64x2::zero()
            .insert(0, Complex64::new(1.0, 2.0))
            .insert(1, Complex64::new(3.0, 4.0));

        let sum = a.reduce_sum();
        assert_eq!(sum, Complex64::new(4.0, 6.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_c32x4_basic() {
        let a = C32x4::splat(Complex32::new(1.0, 2.0));
        let b = C32x4::splat(Complex32::new(3.0, 4.0));

        let sum = a.add(b);
        assert_eq!(sum.extract(0), Complex32::new(4.0, 6.0));
        assert_eq!(sum.extract(3), Complex32::new(4.0, 6.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_c32x4_reduce_sum() {
        let a = C32x4::zero()
            .insert(0, Complex32::new(1.0, 0.0))
            .insert(1, Complex32::new(2.0, 0.0))
            .insert(2, Complex32::new(3.0, 0.0))
            .insert(3, Complex32::new(4.0, 0.0));

        let sum = a.reduce_sum();
        assert_eq!(sum, Complex32::new(10.0, 0.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_c32x4_mul() {
        let a = C32x4::splat(Complex32::new(1.0, 2.0));
        let b = C32x4::splat(Complex32::new(3.0, 4.0));

        // (1+2i)(3+4i) = 3 + 4i + 6i - 8 = -5 + 10i
        let prod = a.mul(b);
        let result = prod.extract(0);

        assert!((result.re - (-5.0)).abs() < 1e-5);
        assert!((result.im - 10.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_c32x4_load_store() {
        unsafe {
            let data = [
                Complex32::new(1.0, 2.0),
                Complex32::new(3.0, 4.0),
                Complex32::new(5.0, 6.0),
                Complex32::new(7.0, 8.0),
            ];

            let v = C32x4::load_unaligned(data.as_ptr());

            assert_eq!(v.extract(0), Complex32::new(1.0, 2.0));
            assert_eq!(v.extract(1), Complex32::new(3.0, 4.0));
            assert_eq!(v.extract(2), Complex32::new(5.0, 6.0));
            assert_eq!(v.extract(3), Complex32::new(7.0, 8.0));

            let mut out = [Complex32::new(0.0, 0.0); 4];
            v.store_unaligned(out.as_mut_ptr());

            assert_eq!(out, data);
        }
    }

    // =========================================================================
    // Randomized SIMD-vs-scalar correctness for the runtime-dispatched batch
    // operations and the x86_64 AVX2 / AVX-512 register types.
    // =========================================================================

    use crate::simd::{SimdLevel, detect_simd_level};

    /// Deterministic SplitMix64 PRNG (no external `rand` dependency).
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            SplitMix64(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform `f64` in `[-1.0, 1.0)`.
        fn next_f64(&mut self) -> f64 {
            let bits = self.next_u64() >> 11; // 53 significant bits
            let unit = bits as f64 / (1u64 << 53) as f64;
            unit * 2.0 - 1.0
        }

        /// Uniform `f32` in `[-1.0, 1.0)`.
        fn next_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as u32; // 24 significant bits
            let unit = bits as f32 / (1u32 << 24) as f32;
            unit * 2.0 - 1.0
        }
    }

    fn rand_c64(rng: &mut SplitMix64) -> Complex64 {
        Complex64::new(rng.next_f64(), rng.next_f64())
    }

    fn rand_c32(rng: &mut SplitMix64) -> Complex32 {
        Complex32::new(rng.next_f32(), rng.next_f32())
    }

    fn c64_close(a: Complex64, b: Complex64, atol: f64, rtol: f64) -> bool {
        (a.re - b.re).abs() <= atol + rtol * b.re.abs()
            && (a.im - b.im).abs() <= atol + rtol * b.im.abs()
    }

    fn c32_close(a: Complex32, b: Complex32, atol: f32, rtol: f32) -> bool {
        (a.re - b.re).abs() <= atol + rtol * b.re.abs()
            && (a.im - b.im).abs() <= atol + rtol * b.im.abs()
    }

    /// Confirms a genuine SIMD path is selected on this environment's CPU and
    /// that the complex batch dispatch does not silently degrade to scalar.
    ///
    /// The assertions respect the crate's SIMD-limiting feature flags: with
    /// `force-scalar` the scalar path is the *correct* outcome, and with
    /// `max-simd-128` x86_64 has no complex kernel (SSE-only), so the SIMD
    /// expectation only applies when neither cap is active.
    #[test]
    fn test_complex_simd_path_is_taken() {
        let level = detect_simd_level();
        let backend = complex_simd_backend();
        #[cfg(feature = "std")]
        println!("detect_simd_level() = {level:?}, complex_simd_backend() = {backend:?}");
        let _ = (&level, &backend);

        #[cfg(all(not(feature = "force-scalar"), not(feature = "max-simd-128")))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                // The dev/CI host for this crate exposes at least AVX2 (256-bit).
                assert!(
                    level >= SimdLevel::Simd256,
                    "expected >= Simd256 on x86_64, got {level:?}"
                );
                assert_ne!(
                    backend,
                    ComplexSimdBackend::Scalar,
                    "complex SIMD dispatch fell back to scalar on a SIMD-capable x86_64 CPU"
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                assert_eq!(level, SimdLevel::Simd128);
                assert_eq!(backend, ComplexSimdBackend::Simd128);
            }
        }

        // When scalar is explicitly forced, the dispatch must honor it.
        #[cfg(feature = "force-scalar")]
        {
            assert_eq!(level, SimdLevel::Scalar);
            assert_eq!(backend, ComplexSimdBackend::Scalar);
        }
    }

    #[test]
    fn test_complex64_batch_matches_scalar() {
        let mut rng = SplitMix64::new(0x0000_C0FF_EE00_0064);
        for &len in &[0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 33, 64, 129, 256] {
            let a: Vec<Complex64> = (0..len).map(|_| rand_c64(&mut rng)).collect();
            let b: Vec<Complex64> = (0..len).map(|_| rand_c64(&mut rng)).collect();
            let mut out = vec![Complex64::new(0.0, 0.0); len];

            // Addition and subtraction are elementwise IEEE ops -> exact match.
            complex64_add(&a, &b, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i] + b[i], "add len={len} i={i}");
            }
            complex64_sub(&a, &b, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i] - b[i], "sub len={len} i={i}");
            }

            // Multiplication uses fused multiply-add-sub; allow FP tolerance.
            complex64_mul(&a, &b, &mut out);
            for i in 0..len {
                let expect = a[i] * b[i];
                assert!(
                    c64_close(out[i], expect, 1e-12, 1e-12),
                    "mul len={len} i={i}: got {:?} expected {:?}",
                    out[i],
                    expect
                );
            }

            // Conjugation and real scaling are exact.
            complex64_conj(&a, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i].conj(), "conj len={len} i={i}");
            }
            let s = rng.next_f64() * 4.0;
            complex64_scale_real(&a, s, &mut out);
            for i in 0..len {
                assert_eq!(
                    out[i],
                    Complex64::new(a[i].re * s, a[i].im * s),
                    "scale len={len} i={i}"
                );
            }
        }
    }

    #[test]
    fn test_complex32_batch_matches_scalar() {
        let mut rng = SplitMix64::new(0x0000_C0FF_EE00_0032);
        for &len in &[0usize, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 33, 64, 129, 256] {
            let a: Vec<Complex32> = (0..len).map(|_| rand_c32(&mut rng)).collect();
            let b: Vec<Complex32> = (0..len).map(|_| rand_c32(&mut rng)).collect();
            let mut out = vec![Complex32::new(0.0, 0.0); len];

            complex32_add(&a, &b, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i] + b[i], "add len={len} i={i}");
            }
            complex32_sub(&a, &b, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i] - b[i], "sub len={len} i={i}");
            }

            complex32_mul(&a, &b, &mut out);
            for i in 0..len {
                let expect = a[i] * b[i];
                assert!(
                    c32_close(out[i], expect, 1e-5, 1e-5),
                    "mul len={len} i={i}: got {:?} expected {:?}",
                    out[i],
                    expect
                );
            }

            complex32_conj(&a, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i].conj(), "conj len={len} i={i}");
            }
            let s = rng.next_f32() * 4.0;
            complex32_scale_real(&a, s, &mut out);
            for i in 0..len {
                assert_eq!(
                    out[i],
                    Complex32::new(a[i].re * s, a[i].im * s),
                    "scale len={len} i={i}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Direct exercises of the x86_64 register types, guarded by runtime feature
    // detection so they never execute an unsupported instruction.
    // -------------------------------------------------------------------------

    // `is_x86_feature_detected!`/`eprintln!` below are `std`-only, so these
    // direct register-type exercises additionally require `feature = "std"`
    // (the batch/scalar-vs-SIMD tests above already cover the no_std-safe
    // dispatch path via `complex_simd_backend()`).
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn test_x86_c64x2_avx2_vs_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: AVX2/FMA not available");
            return;
        }
        let mut rng = SplitMix64::new(0x00A2_0000_0000_0064);
        for _ in 0..2000 {
            let a0 = rand_c64(&mut rng);
            let a1 = rand_c64(&mut rng);
            let b0 = rand_c64(&mut rng);
            let b1 = rand_c64(&mut rng);
            let a = C64x2::zero().insert(0, a0).insert(1, a1);
            let b = C64x2::zero().insert(0, b0).insert(1, b1);

            assert_eq!(a.extract(0), a0);
            assert_eq!(a.extract(1), a1);

            let sum = a.add(b);
            assert_eq!(sum.extract(0), a0 + b0);
            assert_eq!(sum.extract(1), a1 + b1);

            let diff = a.sub(b);
            assert_eq!(diff.extract(0), a0 - b0);
            assert_eq!(diff.extract(1), a1 - b1);

            let prod = a.mul(b);
            assert!(c64_close(prod.extract(0), a0 * b0, 1e-12, 1e-12));
            assert!(c64_close(prod.extract(1), a1 * b1, 1e-12, 1e-12));

            let conj = a.conj();
            assert_eq!(conj.extract(0), a0.conj());
            assert_eq!(conj.extract(1), a1.conj());

            let s = rng.next_f64() * 3.0;
            let scaled = a.scale_real(s);
            assert_eq!(scaled.extract(0), Complex64::new(a0.re * s, a0.im * s));

            let rs = a.reduce_sum();
            assert!(c64_close(rs, a0 + a1, 1e-12, 1e-12));
        }

        unsafe {
            let data = [Complex64::new(1.5, -2.5), Complex64::new(-3.5, 4.5)];
            let v = C64x2::load_unaligned(data.as_ptr());
            let mut out = [Complex64::new(0.0, 0.0); 2];
            v.store_unaligned(out.as_mut_ptr());
            assert_eq!(out, data);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn test_x86_c32x4_avx2_vs_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: AVX2/FMA not available");
            return;
        }
        let mut rng = SplitMix64::new(0x00A2_0000_0000_0032);
        for _ in 0..2000 {
            let vals: [Complex32; 4] = [
                rand_c32(&mut rng),
                rand_c32(&mut rng),
                rand_c32(&mut rng),
                rand_c32(&mut rng),
            ];
            let bvals: [Complex32; 4] = [
                rand_c32(&mut rng),
                rand_c32(&mut rng),
                rand_c32(&mut rng),
                rand_c32(&mut rng),
            ];
            let a = unsafe { C32x4::load_unaligned(vals.as_ptr()) };
            let b = unsafe { C32x4::load_unaligned(bvals.as_ptr()) };

            let prod = a.mul(b);
            let conj = a.conj();
            let sum = a.add(b);
            for i in 0..4 {
                assert!(c32_close(prod.extract(i), vals[i] * bvals[i], 1e-5, 1e-5));
                assert_eq!(conj.extract(i), vals[i].conj());
                assert_eq!(sum.extract(i), vals[i] + bvals[i]);
            }

            let expect_sum: Complex32 = vals.iter().copied().sum();
            assert!(c32_close(a.reduce_sum(), expect_sum, 1e-4, 1e-4));
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn test_x86_c64x4_avx512_vs_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: AVX-512F not available");
            return;
        }
        let mut rng = SplitMix64::new(0x0512_0000_0000_0064);
        for _ in 0..2000 {
            let vals: [Complex64; 4] = [
                rand_c64(&mut rng),
                rand_c64(&mut rng),
                rand_c64(&mut rng),
                rand_c64(&mut rng),
            ];
            let bvals: [Complex64; 4] = [
                rand_c64(&mut rng),
                rand_c64(&mut rng),
                rand_c64(&mut rng),
                rand_c64(&mut rng),
            ];
            let a = unsafe { C64x4::load_unaligned(vals.as_ptr()) };
            let b = unsafe { C64x4::load_unaligned(bvals.as_ptr()) };

            let prod = a.mul(b);
            let sum = a.add(b);
            let diff = a.sub(b);
            let conj = a.conj();
            let s = rng.next_f64() * 3.0;
            let scaled = a.scale_real(s);
            for i in 0..4 {
                assert!(c64_close(prod.extract(i), vals[i] * bvals[i], 1e-12, 1e-12));
                assert_eq!(sum.extract(i), vals[i] + bvals[i]);
                assert_eq!(diff.extract(i), vals[i] - bvals[i]);
                assert_eq!(conj.extract(i), vals[i].conj());
                assert_eq!(
                    scaled.extract(i),
                    Complex64::new(vals[i].re * s, vals[i].im * s)
                );
            }

            let expect_sum: Complex64 = vals.iter().copied().sum();
            assert!(c64_close(a.reduce_sum(), expect_sum, 1e-12, 1e-12));

            unsafe {
                let mut out = [Complex64::new(0.0, 0.0); 4];
                a.store_unaligned(out.as_mut_ptr());
                assert_eq!(out, vals);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn test_x86_c32x8_avx512_vs_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: AVX-512F not available");
            return;
        }
        let mut rng = SplitMix64::new(0x0512_0000_0000_0032);
        for _ in 0..2000 {
            let mut vals = [Complex32::new(0.0, 0.0); 8];
            let mut bvals = [Complex32::new(0.0, 0.0); 8];
            for k in 0..8 {
                vals[k] = rand_c32(&mut rng);
                bvals[k] = rand_c32(&mut rng);
            }
            let a = unsafe { C32x8::load_unaligned(vals.as_ptr()) };
            let b = unsafe { C32x8::load_unaligned(bvals.as_ptr()) };

            let prod = a.mul(b);
            let sum = a.add(b);
            let conj = a.conj();
            for i in 0..8 {
                assert!(c32_close(prod.extract(i), vals[i] * bvals[i], 1e-5, 1e-5));
                assert_eq!(sum.extract(i), vals[i] + bvals[i]);
                assert_eq!(conj.extract(i), vals[i].conj());
            }

            let expect_sum: Complex32 = vals.iter().copied().sum();
            assert!(c32_close(a.reduce_sum(), expect_sum, 1e-3, 1e-4));

            unsafe {
                let mut out = [Complex32::new(0.0, 0.0); 8];
                a.store_unaligned(out.as_mut_ptr());
                assert_eq!(out, vals);
            }
        }
    }
}
