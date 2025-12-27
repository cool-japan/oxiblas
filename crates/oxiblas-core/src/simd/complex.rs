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
// Trait for complex scalar types to select SIMD register
// =============================================================================

/// Trait for complex scalar types with associated SIMD register.
pub trait ComplexSimdScalar: Copy {
    /// 256-bit SIMD register type for this complex scalar.
    type Simd256: ComplexSimdRegister<Complex = Self>;
}

impl ComplexSimdScalar for Complex64 {
    #[cfg(target_arch = "aarch64")]
    type Simd256 = C64x2;
    #[cfg(not(target_arch = "aarch64"))]
    type Simd256 = ScalarC64;
}

impl ComplexSimdScalar for Complex32 {
    #[cfg(target_arch = "aarch64")]
    type Simd256 = C32x4;
    #[cfg(not(target_arch = "aarch64"))]
    type Simd256 = ScalarC32;
}

#[cfg(test)]
mod tests {
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
}
