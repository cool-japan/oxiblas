//! x86_64 complex-number SIMD implementations (AVX2 and AVX-512).
//!
//! This module provides SIMD register types for complex numbers stored in
//! interleaved format `[re0, im0, re1, im1, ...]`, mirroring the AArch64 NEON
//! implementation in the parent module but using x86_64 intrinsics:
//!
//! - AVX2 (256-bit): [`C64x2`] (2 complex `f64`), [`C32x4`] (4 complex `f32`)
//! - AVX-512F (512-bit): [`C64x4`] (4 complex `f64`), [`C32x8`] (8 complex `f32`)
//!
//! # Complex multiplication
//!
//! For interleaved pairs `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, the
//! standard shuffle + fused-multiply-add-sub technique is used:
//!
//! ```text
//! xr  = movedup(x)          // [a, a, ...]  duplicate real lanes
//! xi  = permute(x, odd)     // [b, b, ...]  duplicate imaginary lanes
//! ysw = permute(y, swap)    // [d, c, ...]  swap real/imaginary of y
//! t   = xi * ysw            // [bd, bc, ...]
//! res = fmaddsub(xr, y, t)  // [ac - bd, ad + bc, ...]
//! ```
//!
//! `fmaddsub` subtracts on even (real) lanes and adds on odd (imaginary) lanes,
//! which is exactly what complex multiplication needs.
//!
//! # Safety and dispatch
//!
//! Every intrinsic is wrapped in a `#[target_feature]` function so the compiler
//! emits the correct instructions.  These types must only be used on a CPU that
//! actually supports the corresponding feature; the parent module's runtime
//! dispatch (driven by [`detect_simd_level`](crate::simd::detect_simd_level))
//! guarantees this and falls back to scalar otherwise.

// AVX-512 intrinsics require a newer Rust than the crate MSRV, but every use is
// gated behind runtime feature detection, so silence the MSRV lint here (this
// matches the convention in `simd/x86_64.rs`).
#![allow(clippy::incompatible_msrv)]

use super::ComplexSimdRegister;
use core::arch::x86_64::*;
use num_complex::{Complex32, Complex64};

// =============================================================================
// AVX2 (256-bit): 2 complex f64
// =============================================================================

/// 2 complex `f64` values packed into a single 256-bit AVX register
/// (`[re0, im0, re1, im1]`).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct C64x2(__m256d);

impl ComplexSimdRegister for C64x2 {
    type Real = f64;
    type Complex = Complex64;
    const COMPLEX_LANES: usize = 2;

    #[inline]
    fn zero() -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp() -> __m256d {
            _mm256_setzero_pd()
        }
        unsafe { C64x2(imp()) }
    }

    #[inline]
    fn splat(value: Complex64) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(re: f64, im: f64) -> __m256d {
            _mm256_set_pd(im, re, im, re)
        }
        unsafe { C64x2(imp(value.re, value.im)) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn load_aligned(ptr: *const Complex64) -> Self {
        C64x2(_mm256_load_pd(ptr as *const f64))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn load_unaligned(ptr: *const Complex64) -> Self {
        C64x2(_mm256_loadu_pd(ptr as *const f64))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn store_aligned(self, ptr: *mut Complex64) {
        _mm256_store_pd(ptr as *mut f64, self.0);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn store_unaligned(self, ptr: *mut Complex64) {
        _mm256_storeu_pd(ptr as *mut f64, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(a: __m256d, b: __m256d) -> __m256d {
            _mm256_add_pd(a, b)
        }
        unsafe { C64x2(imp(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(a: __m256d, b: __m256d) -> __m256d {
            _mm256_sub_pd(a, b)
        }
        unsafe { C64x2(imp(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        #[target_feature(enable = "avx,fma")]
        unsafe fn imp(x: __m256d, y: __m256d) -> __m256d {
            let xr = _mm256_movedup_pd(x); // [a0, a0, a1, a1]
            let xi = _mm256_permute_pd::<0b1111>(x); // [b0, b0, b1, b1]
            let ysw = _mm256_permute_pd::<0b0101>(y); // [d0, c0, d1, c1]
            let t = _mm256_mul_pd(xi, ysw); // [b0*d0, b0*c0, b1*d1, b1*c1]
            _mm256_fmaddsub_pd(xr, y, t) // [ac-bd, ad+bc, ...]
        }
        unsafe { C64x2(imp(self.0, other.0)) }
    }

    #[inline]
    fn scale_real(self, scalar: f64) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(x: __m256d, s: f64) -> __m256d {
            _mm256_mul_pd(x, _mm256_set1_pd(s))
        }
        unsafe { C64x2(imp(self.0, scalar)) }
    }

    #[inline]
    fn conj(self) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(x: __m256d) -> __m256d {
            // Multiply imaginary lanes by -1 (exact).
            _mm256_mul_pd(x, _mm256_set_pd(-1.0, 1.0, -1.0, 1.0))
        }
        unsafe { C64x2(imp(self.0)) }
    }

    #[inline]
    fn extract(self, index: usize) -> Complex64 {
        debug_assert!(index < 2);
        let arr = unsafe { core::mem::transmute::<__m256d, [f64; 4]>(self.0) };
        Complex64::new(arr[index * 2], arr[index * 2 + 1])
    }

    #[inline]
    fn insert(self, index: usize, value: Complex64) -> Self {
        debug_assert!(index < 2);
        let mut arr = unsafe { core::mem::transmute::<__m256d, [f64; 4]>(self.0) };
        arr[index * 2] = value.re;
        arr[index * 2 + 1] = value.im;
        Self(unsafe { core::mem::transmute::<[f64; 4], __m256d>(arr) })
    }

    #[inline]
    fn reduce_sum(self) -> Complex64 {
        #[target_feature(enable = "avx")]
        unsafe fn imp(x: __m256d) -> [f64; 2] {
            let hi = _mm256_extractf128_pd::<1>(x); // [re1, im1]
            let lo = _mm256_castpd256_pd128(x); // [re0, im0]
            let s = _mm_add_pd(lo, hi); // [re0+re1, im0+im1]
            let mut out = [0.0f64; 2];
            _mm_storeu_pd(out.as_mut_ptr(), s);
            out
        }
        let out = unsafe { imp(self.0) };
        Complex64::new(out[0], out[1])
    }
}

// =============================================================================
// AVX2 (256-bit): 4 complex f32
// =============================================================================

/// 4 complex `f32` values packed into a single 256-bit AVX register
/// (`[re0, im0, re1, im1, re2, im2, re3, im3]`).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct C32x4(__m256);

impl ComplexSimdRegister for C32x4 {
    type Real = f32;
    type Complex = Complex32;
    const COMPLEX_LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp() -> __m256 {
            _mm256_setzero_ps()
        }
        unsafe { C32x4(imp()) }
    }

    #[inline]
    fn splat(value: Complex32) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(re: f32, im: f32) -> __m256 {
            _mm256_set_ps(im, re, im, re, im, re, im, re)
        }
        unsafe { C32x4(imp(value.re, value.im)) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn load_aligned(ptr: *const Complex32) -> Self {
        C32x4(_mm256_load_ps(ptr as *const f32))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn load_unaligned(ptr: *const Complex32) -> Self {
        C32x4(_mm256_loadu_ps(ptr as *const f32))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn store_aligned(self, ptr: *mut Complex32) {
        _mm256_store_ps(ptr as *mut f32, self.0);
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn store_unaligned(self, ptr: *mut Complex32) {
        _mm256_storeu_ps(ptr as *mut f32, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(a: __m256, b: __m256) -> __m256 {
            _mm256_add_ps(a, b)
        }
        unsafe { C32x4(imp(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(a: __m256, b: __m256) -> __m256 {
            _mm256_sub_ps(a, b)
        }
        unsafe { C32x4(imp(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        #[target_feature(enable = "avx,fma")]
        unsafe fn imp(x: __m256, y: __m256) -> __m256 {
            let xr = _mm256_moveldup_ps(x); // [a0,a0,a1,a1,a2,a2,a3,a3]
            let xi = _mm256_movehdup_ps(x); // [b0,b0,b1,b1,b2,b2,b3,b3]
            let ysw = _mm256_permute_ps::<0b10_11_00_01>(y); // swap re/im of y
            let t = _mm256_mul_ps(xi, ysw);
            _mm256_fmaddsub_ps(xr, y, t)
        }
        unsafe { C32x4(imp(self.0, other.0)) }
    }

    #[inline]
    fn scale_real(self, scalar: f32) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(x: __m256, s: f32) -> __m256 {
            _mm256_mul_ps(x, _mm256_set1_ps(s))
        }
        unsafe { C32x4(imp(self.0, scalar)) }
    }

    #[inline]
    fn conj(self) -> Self {
        #[target_feature(enable = "avx")]
        unsafe fn imp(x: __m256) -> __m256 {
            _mm256_mul_ps(
                x,
                _mm256_set_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            )
        }
        unsafe { C32x4(imp(self.0)) }
    }

    #[inline]
    fn extract(self, index: usize) -> Complex32 {
        debug_assert!(index < 4);
        let arr = unsafe { core::mem::transmute::<__m256, [f32; 8]>(self.0) };
        Complex32::new(arr[index * 2], arr[index * 2 + 1])
    }

    #[inline]
    fn insert(self, index: usize, value: Complex32) -> Self {
        debug_assert!(index < 4);
        let mut arr = unsafe { core::mem::transmute::<__m256, [f32; 8]>(self.0) };
        arr[index * 2] = value.re;
        arr[index * 2 + 1] = value.im;
        Self(unsafe { core::mem::transmute::<[f32; 8], __m256>(arr) })
    }

    #[inline]
    fn reduce_sum(self) -> Complex32 {
        #[target_feature(enable = "avx")]
        unsafe fn imp(x: __m256) -> [f32; 4] {
            let hi = _mm256_extractf128_ps::<1>(x); // [re2,im2,re3,im3]
            let lo = _mm256_castps256_ps128(x); // [re0,im0,re1,im1]
            let s = _mm_add_ps(lo, hi); // [re0+re2, im0+im2, re1+re3, im1+im3]
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), s);
            out
        }
        let out = unsafe { imp(self.0) };
        Complex32::new(out[0] + out[2], out[1] + out[3])
    }
}

// =============================================================================
// AVX-512F (512-bit): 4 complex f64
// =============================================================================

/// 4 complex `f64` values packed into a single 512-bit AVX-512 register.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct C64x4(__m512d);

impl ComplexSimdRegister for C64x4 {
    type Real = f64;
    type Complex = Complex64;
    const COMPLEX_LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp() -> __m512d {
            _mm512_setzero_pd()
        }
        unsafe { C64x4(imp()) }
    }

    #[inline]
    fn splat(value: Complex64) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(re: f64, im: f64) -> __m512d {
            _mm512_set_pd(im, re, im, re, im, re, im, re)
        }
        unsafe { C64x4(imp(value.re, value.im)) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_aligned(ptr: *const Complex64) -> Self {
        C64x4(_mm512_load_pd(ptr as *const f64))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_unaligned(ptr: *const Complex64) -> Self {
        C64x4(_mm512_loadu_pd(ptr as *const f64))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_aligned(self, ptr: *mut Complex64) {
        _mm512_store_pd(ptr as *mut f64, self.0);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_unaligned(self, ptr: *mut Complex64) {
        _mm512_storeu_pd(ptr as *mut f64, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(a: __m512d, b: __m512d) -> __m512d {
            _mm512_add_pd(a, b)
        }
        unsafe { C64x4(imp(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(a: __m512d, b: __m512d) -> __m512d {
            _mm512_sub_pd(a, b)
        }
        unsafe { C64x4(imp(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(x: __m512d, y: __m512d) -> __m512d {
            let xr = _mm512_movedup_pd(x);
            let xi = _mm512_permute_pd::<0b1111_1111>(x);
            let ysw = _mm512_permute_pd::<0b0101_0101>(y);
            let t = _mm512_mul_pd(xi, ysw);
            _mm512_fmaddsub_pd(xr, y, t)
        }
        unsafe { C64x4(imp(self.0, other.0)) }
    }

    #[inline]
    fn scale_real(self, scalar: f64) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(x: __m512d, s: f64) -> __m512d {
            _mm512_mul_pd(x, _mm512_set1_pd(s))
        }
        unsafe { C64x4(imp(self.0, scalar)) }
    }

    #[inline]
    fn conj(self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(x: __m512d) -> __m512d {
            _mm512_mul_pd(
                x,
                _mm512_set_pd(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            )
        }
        unsafe { C64x4(imp(self.0)) }
    }

    #[inline]
    fn extract(self, index: usize) -> Complex64 {
        debug_assert!(index < 4);
        let arr = unsafe { core::mem::transmute::<__m512d, [f64; 8]>(self.0) };
        Complex64::new(arr[index * 2], arr[index * 2 + 1])
    }

    #[inline]
    fn insert(self, index: usize, value: Complex64) -> Self {
        debug_assert!(index < 4);
        let mut arr = unsafe { core::mem::transmute::<__m512d, [f64; 8]>(self.0) };
        arr[index * 2] = value.re;
        arr[index * 2 + 1] = value.im;
        Self(unsafe { core::mem::transmute::<[f64; 8], __m512d>(arr) })
    }

    #[inline]
    fn reduce_sum(self) -> Complex64 {
        let arr = unsafe { core::mem::transmute::<__m512d, [f64; 8]>(self.0) };
        Complex64::new(
            arr[0] + arr[2] + arr[4] + arr[6],
            arr[1] + arr[3] + arr[5] + arr[7],
        )
    }
}

// =============================================================================
// AVX-512F (512-bit): 8 complex f32
// =============================================================================

/// 8 complex `f32` values packed into a single 512-bit AVX-512 register.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct C32x8(__m512);

impl ComplexSimdRegister for C32x8 {
    type Real = f32;
    type Complex = Complex32;
    const COMPLEX_LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp() -> __m512 {
            _mm512_setzero_ps()
        }
        unsafe { C32x8(imp()) }
    }

    #[inline]
    fn splat(value: Complex32) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(re: f32, im: f32) -> __m512 {
            _mm512_set_ps(
                im, re, im, re, im, re, im, re, im, re, im, re, im, re, im, re,
            )
        }
        unsafe { C32x8(imp(value.re, value.im)) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_aligned(ptr: *const Complex32) -> Self {
        C32x8(_mm512_load_ps(ptr as *const f32))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_unaligned(ptr: *const Complex32) -> Self {
        C32x8(_mm512_loadu_ps(ptr as *const f32))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_aligned(self, ptr: *mut Complex32) {
        _mm512_store_ps(ptr as *mut f32, self.0);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_unaligned(self, ptr: *mut Complex32) {
        _mm512_storeu_ps(ptr as *mut f32, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(a: __m512, b: __m512) -> __m512 {
            _mm512_add_ps(a, b)
        }
        unsafe { C32x8(imp(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(a: __m512, b: __m512) -> __m512 {
            _mm512_sub_ps(a, b)
        }
        unsafe { C32x8(imp(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(x: __m512, y: __m512) -> __m512 {
            let xr = _mm512_moveldup_ps(x);
            let xi = _mm512_movehdup_ps(x);
            let ysw = _mm512_permute_ps::<0b10_11_00_01>(y);
            let t = _mm512_mul_ps(xi, ysw);
            _mm512_fmaddsub_ps(xr, y, t)
        }
        unsafe { C32x8(imp(self.0, other.0)) }
    }

    #[inline]
    fn scale_real(self, scalar: f32) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(x: __m512, s: f32) -> __m512 {
            _mm512_mul_ps(x, _mm512_set1_ps(s))
        }
        unsafe { C32x8(imp(self.0, scalar)) }
    }

    #[inline]
    fn conj(self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn imp(x: __m512) -> __m512 {
            _mm512_mul_ps(
                x,
                _mm512_set_ps(
                    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                    -1.0, 1.0,
                ),
            )
        }
        unsafe { C32x8(imp(self.0)) }
    }

    #[inline]
    fn extract(self, index: usize) -> Complex32 {
        debug_assert!(index < 8);
        let arr = unsafe { core::mem::transmute::<__m512, [f32; 16]>(self.0) };
        Complex32::new(arr[index * 2], arr[index * 2 + 1])
    }

    #[inline]
    fn insert(self, index: usize, value: Complex32) -> Self {
        debug_assert!(index < 8);
        let mut arr = unsafe { core::mem::transmute::<__m512, [f32; 16]>(self.0) };
        arr[index * 2] = value.re;
        arr[index * 2 + 1] = value.im;
        Self(unsafe { core::mem::transmute::<[f32; 16], __m512>(arr) })
    }

    #[inline]
    fn reduce_sum(self) -> Complex32 {
        let arr = unsafe { core::mem::transmute::<__m512, [f32; 16]>(self.0) };
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let mut i = 0;
        while i < 16 {
            re += arr[i];
            im += arr[i + 1];
            i += 2;
        }
        Complex32::new(re, im)
    }
}
