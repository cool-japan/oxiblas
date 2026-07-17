//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{
    has_avx512bw, has_avx512dq, has_avx512f, has_avx512vbmi, has_avx512vl, has_avx512vnni,
    lane_index_out_of_range,
};
use core::arch::x86_64::*;

/// 512-bit SIMD register for i8 (64 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I8x64(pub(super) __m512i);
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
        let arr: [i8; 64] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 64),
        }
    }
}
/// AVX-512VNNI operations for neural network acceleration.
///
/// These operations are essential for quantized neural network inference.
pub struct Avx512Vnni;
impl Avx512Vnni {
    /// Checks if AVX-512VNNI is supported at runtime.
    #[inline]
    pub fn is_supported() -> bool {
        has_avx512vnni()
    }
    /// Dot product of 4-element vectors of u8 and i8, accumulated to i32.
    ///
    /// This performs: `dst[i] = src[i] + sum(a[i*4+j] * b[i*4+j])` for j in 0..4
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
    /// This performs: `dst[i] = src[i] + sum(a[i*2+j] * b[i*2+j])` for j in 0..2
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
/// 512-bit SIMD register for f32 (16 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x16(pub(super) __m512);
/// 256-bit SIMD register for f32 (8 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x8(pub(super) __m256);
/// 512-bit SIMD register for f64 (8 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x8(pub(super) __m512d);
/// 128-bit SIMD register for f64 (2 lanes) using SSE.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x2Sse(pub(super) __m128d);
/// 512-bit SIMD register for i16 (32 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I16x32(pub(super) __m512i);
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
        unsafe {
            let arr: [i16; 32] = core::mem::transmute(self.0);
            arr.iter().map(|&x| x as i32).sum()
        }
    }
    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> i16 {
        let arr: [i16; 32] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 32),
        }
    }
}
/// 256-bit SIMD register for f64 (4 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x4(pub(super) __m256d);
/// 512-bit SIMD register for u8 (64 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct U8x64(pub(super) __m512i);
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
        let arr: [u8; 64] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 64),
        }
    }
}
/// 512-bit integer vector for VNNI operations.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I32x16(pub(super) __m512i);
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
            self.dpbusd_fallback(a, b)
        }
    }
    /// Fallback implementation for VNNI dpbusd.
    #[inline]
    pub(super) fn dpbusd_fallback(self, a: U8x64, b: I8x64) -> Self {
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
    pub(super) fn dpwssd_fallback(self, a: I16x32, b: I16x32) -> Self {
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
        let arr: [i32; 16] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 16),
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
        has_avx512bw()
    }
    /// Check if AVX-512VNNI (Vector Neural Network) is supported.
    #[inline]
    pub fn has_avx512vnni() -> bool {
        has_avx512vnni()
    }
    /// Check if AVX-512VBMI (Vector Byte Manipulation) is supported.
    #[inline]
    pub fn has_avx512vbmi() -> bool {
        has_avx512vbmi()
    }
    /// Check if AVX-512DQ (Doubleword and Quadword) is supported.
    #[inline]
    pub fn has_avx512dq() -> bool {
        has_avx512dq()
    }
    /// Check if AVX-512VL (Vector Length Extensions) is supported.
    #[inline]
    pub fn has_avx512vl() -> bool {
        has_avx512vl()
    }
    /// Check if all AVX-512 extensions needed for BLAS acceleration are supported.
    #[inline]
    pub fn has_full_avx512() -> bool {
        has_avx512f() && has_avx512bw() && has_avx512dq()
    }
}
/// 128-bit SIMD register for f32 (4 lanes) using SSE.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x4Sse(pub(super) __m128);
