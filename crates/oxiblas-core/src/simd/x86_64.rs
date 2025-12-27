//! x86_64 SIMD implementations using SSE4.2, AVX2, and AVX512.
//!
//! This module provides SIMD register types and operations for x86_64
//! processors. It includes:
//! - SSE4.2 (128-bit): F64x2Sse, F32x4Sse
//! - AVX2 (256-bit): F64x4, F32x8
//! - AVX512 (512-bit): F64x8, F32x16

// Note: This module is only included when target_arch = "x86_64" (see simd.rs)

// Allow these clippy lints for SIMD code:
// - should_implement_trait: We use add/sub/mul/neg methods on trait implementations
// - missing_transmute_annotations: Transmutes in SIMD are clear from context
// - incompatible_msrv: AVX-512 intrinsics require newer Rust but we gate with runtime detection
// - needless_range_loop: Index-based loops are clearer for SIMD element access patterns
#![allow(clippy::should_implement_trait)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::needless_range_loop)]

use crate::simd::{SimdMask, SimdRegister, SimdScalar};
use core::arch::x86_64::*;

// =============================================================================
// SSE4.2 (128-bit) implementations
// =============================================================================

/// 128-bit SIMD register for f64 (2 lanes) using SSE.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x2Sse(__m128d);

impl SimdRegister for F64x2Sse {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn zero() -> Self {
        unsafe { F64x2Sse(_mm_setzero_pd()) }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { F64x2Sse(_mm_set1_pd(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x2Sse(_mm_load_pd(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x2Sse(_mm_loadu_pd(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        _mm_store_pd(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        _mm_storeu_pd(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_add_pd(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_sub_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_mul_pd(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_div_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // SSE doesn't have native FMA, emulate it
        // If FMA is available, use it
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fmadd_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).add(b)
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fmsub_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).sub(b)
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fnmadd_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            b.sub(self.mul(a))
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        unsafe {
            // Horizontal add: [a0+a1, a0+a1]
            let sum = _mm_hadd_pd(self.0, self.0);
            _mm_cvtsd_f64(sum)
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0);
            let max = _mm_max_pd(self.0, high);
            _mm_cvtsd_f64(max)
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0);
            let min = _mm_min_pd(self.0, high);
            _mm_cvtsd_f64(min)
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
            F64x2Sse(core::mem::transmute(arr))
        }
    }
}

/// 128-bit SIMD register for f32 (4 lanes) using SSE.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x4Sse(__m128);

impl SimdRegister for F32x4Sse {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe { F32x4Sse(_mm_setzero_ps()) }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { F32x4Sse(_mm_set1_ps(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x4Sse(_mm_load_ps(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x4Sse(_mm_loadu_ps(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        _mm_store_ps(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        _mm_storeu_ps(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_add_ps(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_sub_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_mul_ps(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_div_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F32x4Sse(_mm_fmadd_ps(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).add(b)
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F32x4Sse(_mm_fmsub_ps(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).sub(b)
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F32x4Sse(_mm_fnmadd_ps(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            b.sub(self.mul(a))
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        unsafe {
            // Horizontal add pairs
            let sum1 = _mm_hadd_ps(self.0, self.0);
            let sum2 = _mm_hadd_ps(sum1, sum1);
            _mm_cvtss_f32(sum2)
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe {
            let shuffled = _mm_shuffle_ps(self.0, self.0, 0b10_11_00_01);
            let max1 = _mm_max_ps(self.0, shuffled);
            let shuffled2 = _mm_shuffle_ps(max1, max1, 0b00_00_10_10);
            let max2 = _mm_max_ps(max1, shuffled2);
            _mm_cvtss_f32(max2)
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe {
            let shuffled = _mm_shuffle_ps(self.0, self.0, 0b10_11_00_01);
            let min1 = _mm_min_ps(self.0, shuffled);
            let shuffled2 = _mm_shuffle_ps(min1, min1, 0b00_00_10_10);
            let min2 = _mm_min_ps(min1, shuffled2);
            _mm_cvtss_f32(min2)
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        debug_assert!(index < 4);
        unsafe {
            let arr: [f32; 4] = core::mem::transmute(self.0);
            arr[index]
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        debug_assert!(index < 4);
        unsafe {
            let mut arr: [f32; 4] = core::mem::transmute(self.0);
            arr[index] = value;
            F32x4Sse(core::mem::transmute(arr))
        }
    }
}

/// 128-bit SIMD register type alias for compatibility.
pub type Simd128F64 = F64x2Sse;
/// 128-bit SIMD register type alias for compatibility.
pub type Simd128F32 = F32x4Sse;

// =============================================================================
// AVX2 (256-bit) implementations
// =============================================================================

/// 256-bit SIMD register for f64 (4 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x4(__m256d);

impl SimdRegister for F64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe { F64x4(_mm256_setzero_pd()) }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { F64x4(_mm256_set1_pd(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x4(_mm256_load_pd(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x4(_mm256_loadu_pd(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        _mm256_store_pd(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        _mm256_storeu_pd(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F64x4(_mm256_add_pd(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F64x4(_mm256_sub_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F64x4(_mm256_mul_pd(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F64x4(_mm256_div_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // FMA: self * a + b
        unsafe { F64x4(_mm256_fmadd_pd(self.0, a.0, b.0)) }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        // FMA: self * a - b
        unsafe { F64x4(_mm256_fmsub_pd(self.0, a.0, b.0)) }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        // FMA: -(self * a) + b = b - self * a
        unsafe { F64x4(_mm256_fnmadd_pd(self.0, a.0, b.0)) }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        unsafe {
            // Horizontal add: [a0+a1, a2+a3, a0+a1, a2+a3]
            let sum1 = _mm256_hadd_pd(self.0, self.0);
            // Extract high 128 bits and add to low 128 bits
            let high = _mm256_extractf128_pd(sum1, 1);
            let low = _mm256_castpd256_pd128(sum1);
            let sum2 = _mm_add_pd(low, high);
            _mm_cvtsd_f64(sum2)
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
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
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe {
            let high = _mm256_extractf128_pd(self.0, 1);
            let low = _mm256_castpd256_pd128(self.0);
            let min1 = _mm_min_pd(low, high);
            let min2 = _mm_unpackhi_pd(min1, min1);
            let min3 = _mm_min_pd(min1, min2);
            _mm_cvtsd_f64(min3)
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        debug_assert!(index < 4);
        unsafe {
            let arr: [f64; 4] = core::mem::transmute(self.0);
            arr[index]
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        debug_assert!(index < 4);
        unsafe {
            let mut arr: [f64; 4] = core::mem::transmute(self.0);
            arr[index] = value;
            F64x4(core::mem::transmute(arr))
        }
    }
}

/// 256-bit SIMD register for f32 (8 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x8(__m256);

impl SimdRegister for F32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        unsafe { F32x8(_mm256_setzero_ps()) }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { F32x8(_mm256_set1_ps(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x8(_mm256_load_ps(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x8(_mm256_loadu_ps(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        _mm256_store_ps(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        _mm256_storeu_ps(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F32x8(_mm256_add_ps(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F32x8(_mm256_sub_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F32x8(_mm256_mul_ps(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F32x8(_mm256_div_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { F32x8(_mm256_fmadd_ps(self.0, a.0, b.0)) }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        unsafe { F32x8(_mm256_fmsub_ps(self.0, a.0, b.0)) }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        unsafe { F32x8(_mm256_fnmadd_ps(self.0, a.0, b.0)) }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        unsafe {
            // Horizontal add pairs
            let sum1 = _mm256_hadd_ps(self.0, self.0);
            let sum2 = _mm256_hadd_ps(sum1, sum1);
            // Extract and add high/low halves
            let high = _mm256_extractf128_ps(sum2, 1);
            let low = _mm256_castps256_ps128(sum2);
            let sum3 = _mm_add_ps(low, high);
            _mm_cvtss_f32(sum3)
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe {
            let high = _mm256_extractf128_ps(self.0, 1);
            let low = _mm256_castps256_ps128(self.0);
            let max1 = _mm_max_ps(low, high);
            // Shuffle and compare
            let max2 = _mm_shuffle_ps(max1, max1, 0b10_11_00_01);
            let max3 = _mm_max_ps(max1, max2);
            let max4 = _mm_shuffle_ps(max3, max3, 0b00_00_10_10);
            let max5 = _mm_max_ps(max3, max4);
            _mm_cvtss_f32(max5)
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe {
            let high = _mm256_extractf128_ps(self.0, 1);
            let low = _mm256_castps256_ps128(self.0);
            let min1 = _mm_min_ps(low, high);
            let min2 = _mm_shuffle_ps(min1, min1, 0b10_11_00_01);
            let min3 = _mm_min_ps(min1, min2);
            let min4 = _mm_shuffle_ps(min3, min3, 0b00_00_10_10);
            let min5 = _mm_min_ps(min3, min4);
            _mm_cvtss_f32(min5)
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        debug_assert!(index < 8);
        unsafe {
            let arr: [f32; 8] = core::mem::transmute(self.0);
            arr[index]
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        debug_assert!(index < 8);
        unsafe {
            let mut arr: [f32; 8] = core::mem::transmute(self.0);
            arr[index] = value;
            F32x8(core::mem::transmute(arr))
        }
    }
}

// =============================================================================
// AVX512 (512-bit) implementations
// =============================================================================

/// 512-bit SIMD register for f64 (8 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x8(__m512d);

impl SimdRegister for F64x8 {
    type Scalar = f64;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512d {
            _mm512_setzero_pd()
        }
        unsafe { F64x8(zero_impl()) }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn splat_impl(value: f64) -> __m512d {
            _mm512_set1_pd(value)
        }
        unsafe { F64x8(splat_impl(value)) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x8(_mm512_load_pd(ptr))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x8(_mm512_loadu_pd(ptr))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        _mm512_store_pd(ptr, self.0);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        _mm512_storeu_pd(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn add_impl(a: __m512d, b: __m512d) -> __m512d {
            _mm512_add_pd(a, b)
        }
        unsafe { F64x8(add_impl(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn sub_impl(a: __m512d, b: __m512d) -> __m512d {
            _mm512_sub_pd(a, b)
        }
        unsafe { F64x8(sub_impl(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mul_impl(a: __m512d, b: __m512d) -> __m512d {
            _mm512_mul_pd(a, b)
        }
        unsafe { F64x8(mul_impl(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn div_impl(a: __m512d, b: __m512d) -> __m512d {
            _mm512_div_pd(a, b)
        }
        unsafe { F64x8(div_impl(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mul_add_impl(s: __m512d, a: __m512d, b: __m512d) -> __m512d {
            _mm512_fmadd_pd(s, a, b)
        }
        unsafe { F64x8(mul_add_impl(self.0, a.0, b.0)) }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mul_sub_impl(s: __m512d, a: __m512d, b: __m512d) -> __m512d {
            _mm512_fmsub_pd(s, a, b)
        }
        unsafe { F64x8(mul_sub_impl(self.0, a.0, b.0)) }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn neg_mul_add_impl(s: __m512d, a: __m512d, b: __m512d) -> __m512d {
            _mm512_fnmadd_pd(s, a, b)
        }
        unsafe { F64x8(neg_mul_add_impl(self.0, a.0, b.0)) }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_sum_impl(v: __m512d) -> f64 {
            _mm512_reduce_add_pd(v)
        }
        unsafe { reduce_sum_impl(self.0) }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_max_impl(v: __m512d) -> f64 {
            _mm512_reduce_max_pd(v)
        }
        unsafe { reduce_max_impl(self.0) }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_min_impl(v: __m512d) -> f64 {
            _mm512_reduce_min_pd(v)
        }
        unsafe { reduce_min_impl(self.0) }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        debug_assert!(index < 8);
        unsafe {
            let arr: [f64; 8] = core::mem::transmute(self.0);
            arr[index]
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        debug_assert!(index < 8);
        unsafe {
            let mut arr: [f64; 8] = core::mem::transmute(self.0);
            arr[index] = value;
            F64x8(core::mem::transmute(arr))
        }
    }
}

/// 512-bit SIMD register for f32 (16 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x16(__m512);

impl SimdRegister for F32x16 {
    type Scalar = f32;
    const LANES: usize = 16;

    #[inline]
    fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512 {
            _mm512_setzero_ps()
        }
        unsafe { F32x16(zero_impl()) }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn splat_impl(value: f32) -> __m512 {
            _mm512_set1_ps(value)
        }
        unsafe { F32x16(splat_impl(value)) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x16(_mm512_load_ps(ptr))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x16(_mm512_loadu_ps(ptr))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        _mm512_store_ps(ptr, self.0);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        _mm512_storeu_ps(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn add_impl(a: __m512, b: __m512) -> __m512 {
            _mm512_add_ps(a, b)
        }
        unsafe { F32x16(add_impl(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn sub_impl(a: __m512, b: __m512) -> __m512 {
            _mm512_sub_ps(a, b)
        }
        unsafe { F32x16(sub_impl(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mul_impl(a: __m512, b: __m512) -> __m512 {
            _mm512_mul_ps(a, b)
        }
        unsafe { F32x16(mul_impl(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn div_impl(a: __m512, b: __m512) -> __m512 {
            _mm512_div_ps(a, b)
        }
        unsafe { F32x16(div_impl(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mul_add_impl(s: __m512, a: __m512, b: __m512) -> __m512 {
            _mm512_fmadd_ps(s, a, b)
        }
        unsafe { F32x16(mul_add_impl(self.0, a.0, b.0)) }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mul_sub_impl(s: __m512, a: __m512, b: __m512) -> __m512 {
            _mm512_fmsub_ps(s, a, b)
        }
        unsafe { F32x16(mul_sub_impl(self.0, a.0, b.0)) }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn neg_mul_add_impl(s: __m512, a: __m512, b: __m512) -> __m512 {
            _mm512_fnmadd_ps(s, a, b)
        }
        unsafe { F32x16(neg_mul_add_impl(self.0, a.0, b.0)) }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_sum_impl(v: __m512) -> f32 {
            _mm512_reduce_add_ps(v)
        }
        unsafe { reduce_sum_impl(self.0) }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_max_impl(v: __m512) -> f32 {
            _mm512_reduce_max_ps(v)
        }
        unsafe { reduce_max_impl(self.0) }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_min_impl(v: __m512) -> f32 {
            _mm512_reduce_min_ps(v)
        }
        unsafe { reduce_min_impl(self.0) }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        debug_assert!(index < 16);
        unsafe {
            let arr: [f32; 16] = core::mem::transmute(self.0);
            arr[index]
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        debug_assert!(index < 16);
        unsafe {
            let mut arr: [f32; 16] = core::mem::transmute(self.0);
            arr[index] = value;
            F32x16(core::mem::transmute(arr))
        }
    }
}

// =============================================================================
// SimdScalar implementations
// =============================================================================

impl SimdScalar for f64 {
    type Simd256 = F64x4;
    type Simd512 = F64x8;
}

impl SimdScalar for f32 {
    type Simd256 = F32x8;
    type Simd512 = F32x16;
}

// =============================================================================
// Masked operations for AVX512
// =============================================================================

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
    #[target_feature(enable = "avx512f")]
    unsafe fn load_masked(ptr: *const f64, mask: Self::Mask, default: Self) -> Self {
        F64x8(_mm512_mask_loadu_pd(default.0, mask, ptr))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_masked(self, ptr: *mut f64, mask: Self::Mask) {
        _mm512_mask_storeu_pd(ptr, mask, self.0);
    }

    #[inline]
    fn blend(mask: Self::Mask, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn blend_impl(mask: __mmask8, a: __m512d, b: __m512d) -> __m512d {
            _mm512_mask_blend_pd(mask, b, a)
        }
        unsafe { F64x8(blend_impl(mask, a.0, b.0)) }
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
    #[target_feature(enable = "avx512f")]
    unsafe fn load_masked(ptr: *const f32, mask: Self::Mask, default: Self) -> Self {
        F32x16(_mm512_mask_loadu_ps(default.0, mask, ptr))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_masked(self, ptr: *mut f32, mask: Self::Mask) {
        _mm512_mask_storeu_ps(ptr, mask, self.0);
    }

    #[inline]
    fn blend(mask: Self::Mask, a: Self, b: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn blend_impl(mask: __mmask16, a: __m512, b: __m512) -> __m512 {
            _mm512_mask_blend_ps(mask, b, a)
        }
        unsafe { F32x16(blend_impl(mask, a.0, b.0)) }
    }
}

// =============================================================================
// AVX-512BW (Byte/Word) implementations
// =============================================================================

/// 512-bit SIMD register for i16 (32 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I16x32(__m512i);

impl I16x32 {
    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { I16x32(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: i16) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn splat_impl(value: i16) -> __m512i {
            _mm512_set1_epi16(value)
        }
        unsafe { I16x32(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 32 valid i16 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn load_unaligned(ptr: *const i16) -> Self {
        I16x32(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 32 valid writable i16 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn store_unaligned(self, ptr: *mut i16) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi16(a, b)
        }
        unsafe { I16x32(add_impl(self.0, other.0)) }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn sub_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_sub_epi16(a, b)
        }
        unsafe { I16x32(sub_impl(self.0, other.0)) }
    }

    /// Element-wise multiplication (low 16 bits of each 32-bit product).
    #[inline]
    pub fn mullo(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn mullo_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_mullo_epi16(a, b)
        }
        unsafe { I16x32(mullo_impl(self.0, other.0)) }
    }

    /// Saturating addition.
    #[inline]
    pub fn adds(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn adds_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_adds_epi16(a, b)
        }
        unsafe { I16x32(adds_impl(self.0, other.0)) }
    }

    /// Saturating subtraction.
    #[inline]
    pub fn subs(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn subs_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_subs_epi16(a, b)
        }
        unsafe { I16x32(subs_impl(self.0, other.0)) }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn min_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_min_epi16(a, b)
        }
        unsafe { I16x32(min_impl(self.0, other.0)) }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn max_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_max_epi16(a, b)
        }
        unsafe { I16x32(max_impl(self.0, other.0)) }
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn abs_impl(a: __m512i) -> __m512i {
            _mm512_abs_epi16(a)
        }
        unsafe { I16x32(abs_impl(self.0)) }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> i32 {
        // Sum in i32 to avoid overflow
        unsafe {
            let arr: [i16; 32] = core::mem::transmute(self.0);
            arr.iter().map(|&x| x as i32).sum()
        }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> i16 {
        debug_assert!(index < 32);
        unsafe {
            let arr: [i16; 32] = core::mem::transmute(self.0);
            arr[index]
        }
    }
}

/// 512-bit SIMD register for i8 (64 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I8x64(__m512i);

impl I8x64 {
    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { I8x64(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: i8) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn splat_impl(value: i8) -> __m512i {
            _mm512_set1_epi8(value)
        }
        unsafe { I8x64(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid i8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn load_unaligned(ptr: *const i8) -> Self {
        I8x64(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid writable i8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn store_unaligned(self, ptr: *mut i8) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi8(a, b)
        }
        unsafe { I8x64(add_impl(self.0, other.0)) }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn sub_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_sub_epi8(a, b)
        }
        unsafe { I8x64(sub_impl(self.0, other.0)) }
    }

    /// Saturating addition.
    #[inline]
    pub fn adds(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn adds_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_adds_epi8(a, b)
        }
        unsafe { I8x64(adds_impl(self.0, other.0)) }
    }

    /// Saturating subtraction.
    #[inline]
    pub fn subs(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn subs_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_subs_epi8(a, b)
        }
        unsafe { I8x64(subs_impl(self.0, other.0)) }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn min_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_min_epi8(a, b)
        }
        unsafe { I8x64(min_impl(self.0, other.0)) }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn max_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_max_epi8(a, b)
        }
        unsafe { I8x64(max_impl(self.0, other.0)) }
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn abs_impl(a: __m512i) -> __m512i {
            _mm512_abs_epi8(a)
        }
        unsafe { I8x64(abs_impl(self.0)) }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> i32 {
        unsafe {
            let arr: [i8; 64] = core::mem::transmute(self.0);
            arr.iter().map(|&x| x as i32).sum()
        }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> i8 {
        debug_assert!(index < 64);
        unsafe {
            let arr: [i8; 64] = core::mem::transmute(self.0);
            arr[index]
        }
    }
}

/// 512-bit SIMD register for u8 (64 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct U8x64(__m512i);

impl U8x64 {
    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { U8x64(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: u8) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn splat_impl(value: u8) -> __m512i {
            _mm512_set1_epi8(value as i8)
        }
        unsafe { U8x64(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid u8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn load_unaligned(ptr: *const u8) -> Self {
        U8x64(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid writable u8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn store_unaligned(self, ptr: *mut u8) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi8(a, b)
        }
        unsafe { U8x64(add_impl(self.0, other.0)) }
    }

    /// Saturating addition.
    #[inline]
    pub fn adds(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn adds_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_adds_epu8(a, b)
        }
        unsafe { U8x64(adds_impl(self.0, other.0)) }
    }

    /// Saturating subtraction.
    #[inline]
    pub fn subs(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn subs_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_subs_epu8(a, b)
        }
        unsafe { U8x64(subs_impl(self.0, other.0)) }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn min_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_min_epu8(a, b)
        }
        unsafe { U8x64(min_impl(self.0, other.0)) }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn max_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_max_epu8(a, b)
        }
        unsafe { U8x64(max_impl(self.0, other.0)) }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> u32 {
        unsafe {
            let arr: [u8; 64] = core::mem::transmute(self.0);
            arr.iter().map(|&x| x as u32).sum()
        }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> u8 {
        debug_assert!(index < 64);
        unsafe {
            let arr: [u8; 64] = core::mem::transmute(self.0);
            arr[index]
        }
    }
}

// =============================================================================
// AVX-512VNNI (Vector Neural Network Instructions)
// =============================================================================

/// AVX-512VNNI operations for neural network acceleration.
///
/// These operations are essential for quantized neural network inference.
pub struct Avx512Vnni;

impl Avx512Vnni {
    /// Checks if AVX-512VNNI is supported at runtime.
    #[inline]
    pub fn is_supported() -> bool {
        is_x86_feature_detected!("avx512vnni")
    }

    /// Dot product of 4-element vectors of u8 and i8, accumulated to i32.
    ///
    /// This performs: dst[i] = src[i] + sum(a[i*4+j] * b[i*4+j]) for j in 0..4
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpbusd(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpbusd_epi32(src, a, b)
    }

    /// Dot product of 4-element u8*i8 vectors with saturation.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpbusds(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpbusds_epi32(src, a, b)
    }

    /// Dot product of 2-element i16 vectors accumulated to i32.
    ///
    /// This performs: dst[i] = src[i] + sum(a[i*2+j] * b[i*2+j]) for j in 0..2
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpwssd(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpwssd_epi32(src, a, b)
    }

    /// Dot product of 2-element i16 vectors with saturation.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpwssds(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpwssds_epi32(src, a, b)
    }
}

/// 512-bit integer vector for VNNI operations.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I32x16(__m512i);

impl I32x16 {
    /// Number of lanes.
    pub const LANES: usize = 16;

    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { I32x16(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: i32) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn splat_impl(value: i32) -> __m512i {
            _mm512_set1_epi32(value)
        }
        unsafe { I32x16(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 16 valid i32 elements.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn load_unaligned(ptr: *const i32) -> Self {
        I32x16(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 16 valid writable i32 elements.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn store_unaligned(self, ptr: *mut i32) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi32(a, b)
        }
        unsafe { I32x16(add_impl(self.0, other.0)) }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn sub_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_sub_epi32(a, b)
        }
        unsafe { I32x16(sub_impl(self.0, other.0)) }
    }

    /// Element-wise multiplication (low 32 bits).
    #[inline]
    pub fn mullo(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mullo_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_mullo_epi32(a, b)
        }
        unsafe { I32x16(mullo_impl(self.0, other.0)) }
    }

    /// VNNI dot product: u8 * i8 -> i32 accumulation.
    ///
    /// Computes 4-element dot products and adds to accumulator.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    pub unsafe fn dpbusd(self, a: U8x64, b: I8x64) -> Self {
        if Avx512Vnni::is_supported() {
            I32x16(Avx512Vnni::vpdpbusd(self.0, a.0, b.0))
        } else {
            // Fallback implementation
            self.dpbusd_fallback(a, b)
        }
    }

    /// Fallback implementation for VNNI dpbusd.
    #[inline]
    fn dpbusd_fallback(self, a: U8x64, b: I8x64) -> Self {
        unsafe {
            let a_arr: [u8; 64] = core::mem::transmute(a.0);
            let b_arr: [i8; 64] = core::mem::transmute(b.0);
            let mut result: [i32; 16] = core::mem::transmute(self.0);

            for i in 0..16 {
                let base = i * 4;
                for j in 0..4 {
                    result[i] += (a_arr[base + j] as i32) * (b_arr[base + j] as i32);
                }
            }
            I32x16(core::mem::transmute(result))
        }
    }

    /// VNNI dot product: i16 * i16 -> i32 accumulation.
    ///
    /// Computes 2-element dot products and adds to accumulator.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    pub unsafe fn dpwssd(self, a: I16x32, b: I16x32) -> Self {
        if Avx512Vnni::is_supported() {
            I32x16(Avx512Vnni::vpdpwssd(self.0, a.0, b.0))
        } else {
            self.dpwssd_fallback(a, b)
        }
    }

    /// Fallback implementation for VNNI dpwssd.
    #[inline]
    fn dpwssd_fallback(self, a: I16x32, b: I16x32) -> Self {
        unsafe {
            let a_arr: [i16; 32] = core::mem::transmute(a.0);
            let b_arr: [i16; 32] = core::mem::transmute(b.0);
            let mut result: [i32; 16] = core::mem::transmute(self.0);

            for i in 0..16 {
                let base = i * 2;
                for j in 0..2 {
                    result[i] += (a_arr[base + j] as i32) * (b_arr[base + j] as i32);
                }
            }
            I32x16(core::mem::transmute(result))
        }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> i32 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_impl(v: __m512i) -> i32 {
            _mm512_reduce_add_epi32(v)
        }
        unsafe { reduce_impl(self.0) }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> i32 {
        debug_assert!(index < 16);
        unsafe {
            let arr: [i32; 16] = core::mem::transmute(self.0);
            arr[index]
        }
    }

    /// Get the raw __m512i register.
    #[inline]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw __m512i register.
    #[inline]
    pub fn from_raw(v: __m512i) -> Self {
        I32x16(v)
    }
}

/// Feature detection for AVX-512 extensions.
pub struct Avx512Features;

impl Avx512Features {
    /// Check if AVX-512BW (Byte/Word) is supported.
    #[inline]
    pub fn has_avx512bw() -> bool {
        is_x86_feature_detected!("avx512bw")
    }

    /// Check if AVX-512VNNI (Vector Neural Network) is supported.
    #[inline]
    pub fn has_avx512vnni() -> bool {
        is_x86_feature_detected!("avx512vnni")
    }

    /// Check if AVX-512VBMI (Vector Byte Manipulation) is supported.
    #[inline]
    pub fn has_avx512vbmi() -> bool {
        is_x86_feature_detected!("avx512vbmi")
    }

    /// Check if AVX-512DQ (Doubleword and Quadword) is supported.
    #[inline]
    pub fn has_avx512dq() -> bool {
        is_x86_feature_detected!("avx512dq")
    }

    /// Check if AVX-512VL (Vector Length Extensions) is supported.
    #[inline]
    pub fn has_avx512vl() -> bool {
        is_x86_feature_detected!("avx512vl")
    }

    /// Check if all AVX-512 extensions needed for BLAS acceleration are supported.
    #[inline]
    pub fn has_full_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // SSE4.2 tests (always available on x86_64)
    #[test]
    fn test_f64x2_sse_basic() {
        let a = F64x2Sse::splat(2.0);
        let b = F64x2Sse::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(1), 5.0);

        let prod = a.mul(b);
        assert_eq!(prod.extract(0), 6.0);

        // Test emulated FMA
        let c = F64x2Sse::splat(1.0);
        let fma = a.mul_add(b, c); // 2*3 + 1 = 7
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f64x2_sse_reduce() {
        unsafe {
            let data = [1.0f64, 2.0];
            let v = F64x2Sse::load_unaligned(data.as_ptr());
            assert_eq!(v.reduce_sum(), 3.0);
            assert_eq!(v.reduce_max(), 2.0);
            assert_eq!(v.reduce_min(), 1.0);
        }
    }

    #[test]
    fn test_f32x4_sse_basic() {
        let a = F32x4Sse::splat(2.0);
        let b = F32x4Sse::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);

        let fma = a.mul_add(b, F32x4Sse::splat(1.0));
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f32x4_sse_reduce() {
        unsafe {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let v = F32x4Sse::load_unaligned(data.as_ptr());
            assert_eq!(v.reduce_sum(), 10.0);
            assert_eq!(v.reduce_max(), 4.0);
            assert_eq!(v.reduce_min(), 1.0);
        }
    }

    // AVX2 tests
    #[test]
    fn test_f64x4_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = F64x4::splat(2.0);
        let b = F64x4::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(1), 5.0);
        assert_eq!(sum.extract(2), 5.0);
        assert_eq!(sum.extract(3), 5.0);

        let prod = a.mul(b);
        assert_eq!(prod.extract(0), 6.0);

        // Test FMA
        let c = F64x4::splat(1.0);
        let fma = a.mul_add(b, c); // 2*3 + 1 = 7
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f64x4_reduce() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            #[repr(C, align(32))]
            struct Aligned([f64; 4]);
            let data = Aligned([1.0f64, 2.0, 3.0, 4.0]);
            let v = F64x4::load_aligned(data.0.as_ptr());
            assert_eq!(v.reduce_sum(), 10.0);
            assert_eq!(v.reduce_max(), 4.0);
            assert_eq!(v.reduce_min(), 1.0);
        }
    }

    #[test]
    fn test_f32x8_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = F32x8::splat(2.0);
        let b = F32x8::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);

        let fma = a.mul_add(b, F32x8::splat(1.0));
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_load_store() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            let src = [1.0f64, 2.0, 3.0, 4.0];
            let mut dst = [0.0f64; 4];

            let v = F64x4::load_unaligned(src.as_ptr());
            v.store_unaligned(dst.as_mut_ptr());

            assert_eq!(src, dst);
        }
    }

    // AVX-512BW tests
    #[test]
    fn test_i16x32_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        // Test using fallback implementations (transmute-based)
        let a = I16x32::splat(2);
        let b = I16x32::splat(3);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5);
        assert_eq!(sum.extract(15), 5);
        assert_eq!(sum.extract(31), 5);

        let prod = a.mullo(b);
        assert_eq!(prod.extract(0), 6);

        // Test reduce_add
        let ones = I16x32::splat(1);
        assert_eq!(ones.reduce_add(), 32);

        // Test abs
        let neg = I16x32::splat(-5);
        let abs = neg.abs();
        assert_eq!(abs.extract(0), 5);
    }

    #[test]
    fn test_i8x64_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let a = I8x64::splat(2);
        let b = I8x64::splat(3);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5);
        assert_eq!(sum.extract(63), 5);

        // Test abs
        let neg = I8x64::splat(-5);
        let abs = neg.abs();
        assert_eq!(abs.extract(0), 5);

        // Test reduce_add
        let ones = I8x64::splat(1);
        assert_eq!(ones.reduce_add(), 64);
    }

    #[test]
    fn test_u8x64_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let a = U8x64::splat(200);
        let b = U8x64::splat(100);

        let min = a.min(b);
        assert_eq!(min.extract(0), 100);

        let max = a.max(b);
        assert_eq!(max.extract(0), 200);

        // Test saturating add (should saturate at 255)
        let sat_add = a.adds(b);
        assert_eq!(sat_add.extract(0), 255);

        // Test reduce_add
        let ones = U8x64::splat(1);
        assert_eq!(ones.reduce_add(), 64);
    }

    // AVX-512VNNI tests
    #[test]
    fn test_i32x16_basic() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let a = I32x16::splat(2);
        let b = I32x16::splat(3);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5);
        assert_eq!(sum.extract(15), 5);

        let prod = a.mullo(b);
        assert_eq!(prod.extract(0), 6);
    }

    #[test]
    fn test_vnni_dpbusd_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        // Test the fallback implementation
        let acc = I32x16::zero();

        // Create test vectors: 4 elements of u8 and i8 per i32 output lane
        let a_data: [u8; 64] = [1; 64];
        let b_data: [i8; 64] = [2; 64];

        let a = unsafe { U8x64::load_unaligned(a_data.as_ptr()) };
        let b = unsafe { I8x64::load_unaligned(b_data.as_ptr()) };

        let result = acc.dpbusd_fallback(a, b);

        // Each lane should be: 1*2 + 1*2 + 1*2 + 1*2 = 8
        assert_eq!(result.extract(0), 8);
        assert_eq!(result.extract(15), 8);
    }

    #[test]
    fn test_vnni_dpwssd_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let acc = I32x16::zero();

        // Create test vectors: 2 elements of i16 per i32 output lane
        let a_data: [i16; 32] = [3; 32];
        let b_data: [i16; 32] = [4; 32];

        let a = unsafe { I16x32::load_unaligned(a_data.as_ptr()) };
        let b = unsafe { I16x32::load_unaligned(b_data.as_ptr()) };

        let result = acc.dpwssd_fallback(a, b);

        // Each lane should be: 3*4 + 3*4 = 24
        assert_eq!(result.extract(0), 24);
        assert_eq!(result.extract(15), 24);
    }

    #[test]
    fn test_avx512_feature_detection() {
        // Just test that feature detection doesn't panic
        let _bw = Avx512Features::has_avx512bw();
        let _vnni = Avx512Features::has_avx512vnni();
        let _vbmi = Avx512Features::has_avx512vbmi();
        let _dq = Avx512Features::has_avx512dq();
        let _vl = Avx512Features::has_avx512vl();
        let _full = Avx512Features::has_full_avx512();

        println!(
            "AVX-512 features: BW={}, VNNI={}, DQ={}, VL={}, Full={}",
            _bw, _vnni, _dq, _vl, _full
        );
    }
}
