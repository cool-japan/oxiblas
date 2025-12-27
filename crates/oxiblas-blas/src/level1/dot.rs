//! Dot product (inner product).
//!
//! Computes x·y = Σ x\[i\] * y\[i\]
//!
//! This module provides optimized implementations using SIMD instructions
//! on supported platforms (AVX2/FMA on `x86_64`, NEON on aarch64).

use num_complex::{Complex32, Complex64};
use oxiblas_core::scalar::Field;

/// Computes the dot product of two vectors.
///
/// x·y = Σ x\[i\] * y\[i\]
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::dot;
///
/// let x = [1.0f64, 2.0, 3.0];
/// let y = [4.0f64, 5.0, 6.0];
///
/// // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// let result = dot(&x, &y);
/// assert!((result - 32.0).abs() < 1e-10);
/// ```
pub fn dot<T: Field>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    // Use 4-way accumulation for better numerical stability and pipelining
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += x[base] * y[base];
        acc1 += x[base + 1] * y[base + 1];
        acc2 += x[base + 2] * y[base + 2];
        acc3 += x[base + 3] * y[base + 3];
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += x[base + i] * y[base + i];
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Computes the conjugate dot product (for complex vectors).
///
/// x^H · y = Σ conj(x\[i\]) * y\[i\]
///
/// For real vectors, this is the same as `dot`.
pub fn dotc<T: Field>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    let mut acc = T::zero();
    for i in 0..n {
        acc += x[i].conj() * y[i];
    }

    acc
}

/// SIMD-optimized dot product for f64.
///
/// Uses NEON FMA on aarch64 and AVX2/FMA on `x86_64` for performance.
/// For very large vectors, uses blocked accumulation for better cache utilization.
#[inline]
#[must_use]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    // For small vectors, use simple implementation
    if n < 32 {
        return dot_f64_scalar(x, y);
    }

    // For medium vectors, use SIMD directly
    if n < 8192 {
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { dot_f64_neon(x, y) };
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { dot_f64_avx2(x, y) };
            }
        }
    }

    // For large vectors, use blocked approach for better cache efficiency
    #[cfg(target_arch = "aarch64")]
    {
        return dot_f64_blocked(x, y);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f64_blocked(x, y);
        }
    }

    #[allow(unreachable_code)]
    dot_f64_scalar(x, y)
}

/// SIMD-optimized dot product for f32.
///
/// Uses NEON FMA on aarch64 and AVX2/FMA on `x86_64` for performance.
#[inline]
#[must_use]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    // For small vectors, use simple implementation
    if n < 32 {
        return dot_f32_scalar(x, y);
    }

    // For medium vectors, use SIMD directly
    if n < 8192 {
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { dot_f32_neon(x, y) };
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { dot_f32_avx2(x, y) };
            }
        }
    }

    // For large vectors, use blocked approach
    #[cfg(target_arch = "aarch64")]
    {
        return dot_f32_blocked(x, y);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f32_blocked(x, y);
        }
    }

    #[allow(unreachable_code)]
    dot_f32_scalar(x, y)
}

/// Scalar dot product for f64 with 8-way unrolling.
#[inline]
fn dot_f64_scalar(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();

    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 = x[base].mul_add(y[base], acc0);
        acc1 = x[base + 1].mul_add(y[base + 1], acc1);
        acc2 = x[base + 2].mul_add(y[base + 2], acc2);
        acc3 = x[base + 3].mul_add(y[base + 3], acc3);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 = x[base + i].mul_add(y[base + i], acc0);
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Scalar dot product for f32 with 8-way unrolling.
#[inline]
fn dot_f32_scalar(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 = x[base].mul_add(y[base], acc0);
        acc1 = x[base + 1].mul_add(y[base + 1], acc1);
        acc2 = x[base + 2].mul_add(y[base + 2], acc2);
        acc3 = x[base + 3].mul_add(y[base + 3], acc3);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 = x[base + i].mul_add(y[base + i], acc0);
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Blocked dot product for f64 (better cache efficiency for large vectors).
///
/// Processes the vector in blocks that fit well in L1/L2 cache.
fn dot_f64_blocked(x: &[f64], y: &[f64]) -> f64 {
    const BLOCK_SIZE: usize = 1024; // ~8KB per vector block, fits in L1 cache

    let n = x.len();
    let num_blocks = n.div_ceil(BLOCK_SIZE);

    let mut total = 0.0;

    for block in 0..num_blocks {
        let start = block * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n);

        #[cfg(target_arch = "aarch64")]
        {
            total += unsafe { dot_f64_neon(&x[start..end], &y[start..end]) };
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                total += unsafe { dot_f64_avx2(&x[start..end], &y[start..end]) };
            } else {
                total += dot_f64_scalar(&x[start..end], &y[start..end]);
            }
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            total += dot_f64_scalar(&x[start..end], &y[start..end]);
        }
    }

    total
}

/// Blocked dot product for f32.
fn dot_f32_blocked(x: &[f32], y: &[f32]) -> f32 {
    const BLOCK_SIZE: usize = 2048; // ~8KB per vector block, fits in L1 cache

    let n = x.len();
    let num_blocks = n.div_ceil(BLOCK_SIZE);

    let mut total = 0.0f32;

    for block in 0..num_blocks {
        let start = block * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n);

        #[cfg(target_arch = "aarch64")]
        {
            total += unsafe { dot_f32_neon(&x[start..end], &y[start..end]) };
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                total += unsafe { dot_f32_avx2(&x[start..end], &y[start..end]) };
            } else {
                total += dot_f32_scalar(&x[start..end], &y[start..end]);
            }
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            total += dot_f32_scalar(&x[start..end], &y[start..end]);
        }
    }

    total
}

/// NEON optimized dot product for f64.
///
/// Processes 8 elements per iteration (4 NEON registers × 2 f64 each).
#[cfg(target_arch = "aarch64")]
unsafe fn dot_f64_neon(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::aarch64::{vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64};

    let n = x.len();

    let mut sum_vec0 = vdupq_n_f64(0.0);
    let mut sum_vec1 = vdupq_n_f64(0.0);
    let mut sum_vec2 = vdupq_n_f64(0.0);
    let mut sum_vec3 = vdupq_n_f64(0.0);

    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let base = i * 8;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_ptr().add(base);

        let x0 = vld1q_f64(x_ptr);
        let x1 = vld1q_f64(x_ptr.add(2));
        let x2 = vld1q_f64(x_ptr.add(4));
        let x3 = vld1q_f64(x_ptr.add(6));

        let y0 = vld1q_f64(y_ptr);
        let y1 = vld1q_f64(y_ptr.add(2));
        let y2 = vld1q_f64(y_ptr.add(4));
        let y3 = vld1q_f64(y_ptr.add(6));

        sum_vec0 = vfmaq_f64(sum_vec0, x0, y0);
        sum_vec1 = vfmaq_f64(sum_vec1, x1, y1);
        sum_vec2 = vfmaq_f64(sum_vec2, x2, y2);
        sum_vec3 = vfmaq_f64(sum_vec3, x3, y3);
    }

    // Combine accumulators
    let sum_01 = vaddq_f64(sum_vec0, sum_vec1);
    let sum_23 = vaddq_f64(sum_vec2, sum_vec3);
    let sum_all = vaddq_f64(sum_01, sum_23);
    let mut sum = vaddvq_f64(sum_all);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        sum = x[base + i].mul_add(y[base + i], sum);
    }

    sum
}

/// NEON optimized dot product for f32.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_f32_neon(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let n = x.len();

    let mut sum_vec0 = vdupq_n_f32(0.0);
    let mut sum_vec1 = vdupq_n_f32(0.0);
    let mut sum_vec2 = vdupq_n_f32(0.0);
    let mut sum_vec3 = vdupq_n_f32(0.0);

    let chunks = n / 16;
    let remainder = n % 16;

    for i in 0..chunks {
        let base = i * 16;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_ptr().add(base);

        let x0 = vld1q_f32(x_ptr);
        let x1 = vld1q_f32(x_ptr.add(4));
        let x2 = vld1q_f32(x_ptr.add(8));
        let x3 = vld1q_f32(x_ptr.add(12));

        let y0 = vld1q_f32(y_ptr);
        let y1 = vld1q_f32(y_ptr.add(4));
        let y2 = vld1q_f32(y_ptr.add(8));
        let y3 = vld1q_f32(y_ptr.add(12));

        sum_vec0 = vfmaq_f32(sum_vec0, x0, y0);
        sum_vec1 = vfmaq_f32(sum_vec1, x1, y1);
        sum_vec2 = vfmaq_f32(sum_vec2, x2, y2);
        sum_vec3 = vfmaq_f32(sum_vec3, x3, y3);
    }

    // Combine accumulators
    let sum_01 = vaddq_f32(sum_vec0, sum_vec1);
    let sum_23 = vaddq_f32(sum_vec2, sum_vec3);
    let sum_all = vaddq_f32(sum_01, sum_23);
    let mut sum = vaddvq_f32(sum_all);

    // Handle remainder
    let base = chunks * 16;
    for i in 0..remainder {
        sum = x[base + i].mul_add(y[base + i], sum);
    }

    sum
}

/// AVX2/FMA optimized dot product for f64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f64_avx2(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::x86_64::*;

    let n = x.len();

    let mut sum_vec0 = _mm256_setzero_pd();
    let mut sum_vec1 = _mm256_setzero_pd();
    let mut sum_vec2 = _mm256_setzero_pd();
    let mut sum_vec3 = _mm256_setzero_pd();

    let chunks = n / 16;
    let remainder = n % 16;

    for i in 0..chunks {
        let base = i * 16;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_ptr().add(base);

        let x0 = _mm256_loadu_pd(x_ptr);
        let x1 = _mm256_loadu_pd(x_ptr.add(4));
        let x2 = _mm256_loadu_pd(x_ptr.add(8));
        let x3 = _mm256_loadu_pd(x_ptr.add(12));

        let y0 = _mm256_loadu_pd(y_ptr);
        let y1 = _mm256_loadu_pd(y_ptr.add(4));
        let y2 = _mm256_loadu_pd(y_ptr.add(8));
        let y3 = _mm256_loadu_pd(y_ptr.add(12));

        sum_vec0 = _mm256_fmadd_pd(x0, y0, sum_vec0);
        sum_vec1 = _mm256_fmadd_pd(x1, y1, sum_vec1);
        sum_vec2 = _mm256_fmadd_pd(x2, y2, sum_vec2);
        sum_vec3 = _mm256_fmadd_pd(x3, y3, sum_vec3);
    }

    // Combine and reduce
    let sum_01 = _mm256_add_pd(sum_vec0, sum_vec1);
    let sum_23 = _mm256_add_pd(sum_vec2, sum_vec3);
    let sum_all = _mm256_add_pd(sum_01, sum_23);

    let mut sum_arr = [0.0f64; 4];
    _mm256_storeu_pd(sum_arr.as_mut_ptr(), sum_all);
    let mut sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

    // Handle remainder
    let base = chunks * 16;
    for i in 0..remainder {
        sum = x[base + i].mul_add(y[base + i], sum);
    }

    sum
}

/// AVX2/FMA optimized dot product for f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = x.len();

    let mut sum_vec0 = _mm256_setzero_ps();
    let mut sum_vec1 = _mm256_setzero_ps();
    let mut sum_vec2 = _mm256_setzero_ps();
    let mut sum_vec3 = _mm256_setzero_ps();

    let chunks = n / 32;
    let remainder = n % 32;

    for i in 0..chunks {
        let base = i * 32;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_ptr().add(base);

        let x0 = _mm256_loadu_ps(x_ptr);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        let x3 = _mm256_loadu_ps(x_ptr.add(24));

        let y0 = _mm256_loadu_ps(y_ptr);
        let y1 = _mm256_loadu_ps(y_ptr.add(8));
        let y2 = _mm256_loadu_ps(y_ptr.add(16));
        let y3 = _mm256_loadu_ps(y_ptr.add(24));

        sum_vec0 = _mm256_fmadd_ps(x0, y0, sum_vec0);
        sum_vec1 = _mm256_fmadd_ps(x1, y1, sum_vec1);
        sum_vec2 = _mm256_fmadd_ps(x2, y2, sum_vec2);
        sum_vec3 = _mm256_fmadd_ps(x3, y3, sum_vec3);
    }

    // Combine and reduce
    let sum_01 = _mm256_add_ps(sum_vec0, sum_vec1);
    let sum_23 = _mm256_add_ps(sum_vec2, sum_vec3);
    let sum_all = _mm256_add_ps(sum_01, sum_23);

    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_all);
    let mut sum: f32 = sum_arr.iter().sum();

    // Handle remainder
    let base = chunks * 32;
    for i in 0..remainder {
        sum = x[base + i].mul_add(y[base + i], sum);
    }

    sum
}

// =============================================================================
// Complex dot product optimizations (ZDOTC, CDOTC)
// =============================================================================

/// SIMD-optimized conjugate dot product for Complex64 (ZDOTC).
///
/// Computes: `x^H · y = Σ conj(x[i]) * y[i]`
///
/// Uses NEON on aarch64 and AVX2/FMA on `x86_64` for performance.
/// For conj(x) * y where x = (a + bi), y = (c + di):
///   result = (ac + bd) + (ad - bc)i
#[inline]
#[must_use]
pub fn dotc_c64(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return Complex64::new(0.0, 0.0);
    }

    // For small vectors, use scalar implementation
    if n < 16 {
        return dotc_c64_scalar(x, y);
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dotc_c64_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dotc_c64_avx2(x, y) };
        }
    }

    #[allow(unreachable_code)]
    dotc_c64_scalar(x, y)
}

/// SIMD-optimized conjugate dot product for Complex32 (CDOTC).
///
/// Computes: `x^H · y = Σ conj(x[i]) * y[i]`
#[inline]
#[must_use]
pub fn dotc_c32(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return Complex32::new(0.0, 0.0);
    }

    // For small vectors, use scalar implementation
    if n < 32 {
        return dotc_c32_scalar(x, y);
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dotc_c32_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dotc_c32_avx2(x, y) };
        }
    }

    #[allow(unreachable_code)]
    dotc_c32_scalar(x, y)
}

/// Unconjugated dot product for Complex64 (ZDOTU).
///
/// Computes: `x · y = Σ x[i] * y[i]`
#[inline]
#[must_use]
pub fn dotu_c64(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return Complex64::new(0.0, 0.0);
    }

    // For small vectors, use scalar implementation
    if n < 16 {
        return dotu_c64_scalar(x, y);
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dotu_c64_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dotu_c64_avx2(x, y) };
        }
    }

    #[allow(unreachable_code)]
    dotu_c64_scalar(x, y)
}

/// Unconjugated dot product for Complex32 (CDOTU).
///
/// Computes: `x · y = Σ x[i] * y[i]`
#[inline]
#[must_use]
pub fn dotu_c32(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return Complex32::new(0.0, 0.0);
    }

    // For small vectors, use scalar implementation
    if n < 32 {
        return dotu_c32_scalar(x, y);
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dotu_c32_neon(x, y) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dotu_c32_avx2(x, y) };
        }
    }

    #[allow(unreachable_code)]
    dotu_c32_scalar(x, y)
}

/// Scalar implementation for Complex64 conjugate dot product.
/// Uses 4-way accumulation for pipelining.
#[inline]
fn dotc_c64_scalar(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    let n = x.len();

    // 4-way accumulation for better pipelining
    let mut acc0 = Complex64::new(0.0, 0.0);
    let mut acc1 = Complex64::new(0.0, 0.0);
    let mut acc2 = Complex64::new(0.0, 0.0);
    let mut acc3 = Complex64::new(0.0, 0.0);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        // conj(x) * y = (x.re - x.im*i) * (y.re + y.im*i)
        //             = (x.re*y.re + x.im*y.im) + (x.re*y.im - x.im*y.re)*i
        acc0 += x[base].conj() * y[base];
        acc1 += x[base + 1].conj() * y[base + 1];
        acc2 += x[base + 2].conj() * y[base + 2];
        acc3 += x[base + 3].conj() * y[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += x[base + i].conj() * y[base + i];
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Scalar implementation for Complex32 conjugate dot product.
#[inline]
fn dotc_c32_scalar(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    let n = x.len();

    let mut acc0 = Complex32::new(0.0, 0.0);
    let mut acc1 = Complex32::new(0.0, 0.0);
    let mut acc2 = Complex32::new(0.0, 0.0);
    let mut acc3 = Complex32::new(0.0, 0.0);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += x[base].conj() * y[base];
        acc1 += x[base + 1].conj() * y[base + 1];
        acc2 += x[base + 2].conj() * y[base + 2];
        acc3 += x[base + 3].conj() * y[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += x[base + i].conj() * y[base + i];
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Scalar implementation for Complex64 unconjugated dot product.
#[inline]
fn dotu_c64_scalar(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    let n = x.len();

    let mut acc0 = Complex64::new(0.0, 0.0);
    let mut acc1 = Complex64::new(0.0, 0.0);
    let mut acc2 = Complex64::new(0.0, 0.0);
    let mut acc3 = Complex64::new(0.0, 0.0);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += x[base] * y[base];
        acc1 += x[base + 1] * y[base + 1];
        acc2 += x[base + 2] * y[base + 2];
        acc3 += x[base + 3] * y[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += x[base + i] * y[base + i];
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Scalar implementation for Complex32 unconjugated dot product.
#[inline]
fn dotu_c32_scalar(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    let n = x.len();

    let mut acc0 = Complex32::new(0.0, 0.0);
    let mut acc1 = Complex32::new(0.0, 0.0);
    let mut acc2 = Complex32::new(0.0, 0.0);
    let mut acc3 = Complex32::new(0.0, 0.0);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += x[base] * y[base];
        acc1 += x[base + 1] * y[base + 1];
        acc2 += x[base + 2] * y[base + 2];
        acc3 += x[base + 3] * y[base + 3];
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += x[base + i] * y[base + i];
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// NEON optimized conjugate dot product for Complex64.
///
/// Processes 2 complex numbers per iteration using 128-bit NEON registers.
/// For conj(x) * y: real = x.re*y.re + x.im*y.im, imag = x.re*y.im - x.im*y.re
#[cfg(target_arch = "aarch64")]
unsafe fn dotc_c64_neon(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    use core::arch::aarch64::{
        vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld2q_f64,
    };

    let n = x.len();

    // Accumulators for real and imaginary parts
    let mut sum_re0 = vdupq_n_f64(0.0);
    let mut sum_im0 = vdupq_n_f64(0.0);
    let mut sum_re1 = vdupq_n_f64(0.0);
    let mut sum_im1 = vdupq_n_f64(0.0);

    // Process 4 complex numbers per iteration (8 f64 values)
    let chunks = n / 4;
    let remainder = n % 4;

    let x_ptr = x.as_ptr().cast::<f64>();
    let y_ptr = y.as_ptr().cast::<f64>();

    for i in 0..chunks {
        let base = i * 8; // 4 complex = 8 f64

        // Load x values: [re0, im0, re1, im1] x 2
        let x01 = vld2q_f64(x_ptr.add(base)); // Deinterleave: x01.0 = [re0, re1], x01.1 = [im0, im1]
        let x23 = vld2q_f64(x_ptr.add(base + 4));

        // Load y values
        let y01 = vld2q_f64(y_ptr.add(base));
        let y23 = vld2q_f64(y_ptr.add(base + 4));

        // For conj(x) * y:
        // real = x.re * y.re + x.im * y.im
        // imag = x.re * y.im - x.im * y.re

        // First pair
        sum_re0 = vfmaq_f64(sum_re0, x01.0, y01.0); // re += x.re * y.re
        sum_re0 = vfmaq_f64(sum_re0, x01.1, y01.1); // re += x.im * y.im
        sum_im0 = vfmaq_f64(sum_im0, x01.0, y01.1); // im += x.re * y.im
        sum_im0 = vfmsq_f64(sum_im0, x01.1, y01.0); // im -= x.im * y.re

        // Second pair
        sum_re1 = vfmaq_f64(sum_re1, x23.0, y23.0);
        sum_re1 = vfmaq_f64(sum_re1, x23.1, y23.1);
        sum_im1 = vfmaq_f64(sum_im1, x23.0, y23.1);
        sum_im1 = vfmsq_f64(sum_im1, x23.1, y23.0);
    }

    // Combine accumulators
    let sum_re = vaddq_f64(sum_re0, sum_re1);
    let sum_im = vaddq_f64(sum_im0, sum_im1);

    let mut re = vaddvq_f64(sum_re);
    let mut im = vaddvq_f64(sum_im);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        // conj(x) * y
        re += xi.re.mul_add(yi.re, xi.im * yi.im);
        im += xi.re.mul_add(yi.im, -(xi.im * yi.re));
    }

    Complex64::new(re, im)
}

/// NEON optimized conjugate dot product for Complex32.
///
/// Processes 4 complex numbers per iteration using 128-bit NEON registers.
#[cfg(target_arch = "aarch64")]
unsafe fn dotc_c32_neon(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vfmsq_f32, vld2q_f32,
    };

    let n = x.len();

    let mut sum_re0 = vdupq_n_f32(0.0);
    let mut sum_im0 = vdupq_n_f32(0.0);
    let mut sum_re1 = vdupq_n_f32(0.0);
    let mut sum_im1 = vdupq_n_f32(0.0);

    // Process 8 complex numbers per iteration (16 f32 values)
    let chunks = n / 8;
    let remainder = n % 8;

    let x_ptr = x.as_ptr().cast::<f32>();
    let y_ptr = y.as_ptr().cast::<f32>();

    for i in 0..chunks {
        let base = i * 16;

        // Load and deinterleave: 4 complex per vld2q
        let x03 = vld2q_f32(x_ptr.add(base)); // 4 complex
        let x47 = vld2q_f32(x_ptr.add(base + 8)); // 4 complex

        let y03 = vld2q_f32(y_ptr.add(base));
        let y47 = vld2q_f32(y_ptr.add(base + 8));

        // First group of 4
        sum_re0 = vfmaq_f32(sum_re0, x03.0, y03.0);
        sum_re0 = vfmaq_f32(sum_re0, x03.1, y03.1);
        sum_im0 = vfmaq_f32(sum_im0, x03.0, y03.1);
        sum_im0 = vfmsq_f32(sum_im0, x03.1, y03.0);

        // Second group of 4
        sum_re1 = vfmaq_f32(sum_re1, x47.0, y47.0);
        sum_re1 = vfmaq_f32(sum_re1, x47.1, y47.1);
        sum_im1 = vfmaq_f32(sum_im1, x47.0, y47.1);
        sum_im1 = vfmsq_f32(sum_im1, x47.1, y47.0);
    }

    let sum_re = vaddq_f32(sum_re0, sum_re1);
    let sum_im = vaddq_f32(sum_im0, sum_im1);

    let mut re = vaddvq_f32(sum_re);
    let mut im = vaddvq_f32(sum_im);

    let base = chunks * 8;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        re += xi.re.mul_add(yi.re, xi.im * yi.im);
        im += xi.re.mul_add(yi.im, -(xi.im * yi.re));
    }

    Complex32::new(re, im)
}

/// NEON optimized unconjugated dot product for Complex64.
///
/// For x * y: real = x.re*y.re - x.im*y.im, imag = x.re*y.im + x.im*y.re
#[cfg(target_arch = "aarch64")]
unsafe fn dotu_c64_neon(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    use core::arch::aarch64::{
        vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld2q_f64,
    };

    let n = x.len();

    let mut sum_re0 = vdupq_n_f64(0.0);
    let mut sum_im0 = vdupq_n_f64(0.0);
    let mut sum_re1 = vdupq_n_f64(0.0);
    let mut sum_im1 = vdupq_n_f64(0.0);

    let chunks = n / 4;
    let remainder = n % 4;

    let x_ptr = x.as_ptr().cast::<f64>();
    let y_ptr = y.as_ptr().cast::<f64>();

    for i in 0..chunks {
        let base = i * 8;

        let x01 = vld2q_f64(x_ptr.add(base));
        let x23 = vld2q_f64(x_ptr.add(base + 4));

        let y01 = vld2q_f64(y_ptr.add(base));
        let y23 = vld2q_f64(y_ptr.add(base + 4));

        // For x * y (unconjugated):
        // real = x.re * y.re - x.im * y.im
        // imag = x.re * y.im + x.im * y.re

        sum_re0 = vfmaq_f64(sum_re0, x01.0, y01.0);
        sum_re0 = vfmsq_f64(sum_re0, x01.1, y01.1);
        sum_im0 = vfmaq_f64(sum_im0, x01.0, y01.1);
        sum_im0 = vfmaq_f64(sum_im0, x01.1, y01.0);

        sum_re1 = vfmaq_f64(sum_re1, x23.0, y23.0);
        sum_re1 = vfmsq_f64(sum_re1, x23.1, y23.1);
        sum_im1 = vfmaq_f64(sum_im1, x23.0, y23.1);
        sum_im1 = vfmaq_f64(sum_im1, x23.1, y23.0);
    }

    let sum_re = vaddq_f64(sum_re0, sum_re1);
    let sum_im = vaddq_f64(sum_im0, sum_im1);

    let mut re = vaddvq_f64(sum_re);
    let mut im = vaddvq_f64(sum_im);

    let base = chunks * 4;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        re += xi.re.mul_add(yi.re, -(xi.im * yi.im));
        im += xi.re.mul_add(yi.im, xi.im * yi.re);
    }

    Complex64::new(re, im)
}

/// NEON optimized unconjugated dot product for Complex32.
#[cfg(target_arch = "aarch64")]
unsafe fn dotu_c32_neon(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vfmsq_f32, vld2q_f32,
    };

    let n = x.len();

    let mut sum_re0 = vdupq_n_f32(0.0);
    let mut sum_im0 = vdupq_n_f32(0.0);
    let mut sum_re1 = vdupq_n_f32(0.0);
    let mut sum_im1 = vdupq_n_f32(0.0);

    let chunks = n / 8;
    let remainder = n % 8;

    let x_ptr = x.as_ptr().cast::<f32>();
    let y_ptr = y.as_ptr().cast::<f32>();

    for i in 0..chunks {
        let base = i * 16;

        let x03 = vld2q_f32(x_ptr.add(base));
        let x47 = vld2q_f32(x_ptr.add(base + 8));

        let y03 = vld2q_f32(y_ptr.add(base));
        let y47 = vld2q_f32(y_ptr.add(base + 8));

        sum_re0 = vfmaq_f32(sum_re0, x03.0, y03.0);
        sum_re0 = vfmsq_f32(sum_re0, x03.1, y03.1);
        sum_im0 = vfmaq_f32(sum_im0, x03.0, y03.1);
        sum_im0 = vfmaq_f32(sum_im0, x03.1, y03.0);

        sum_re1 = vfmaq_f32(sum_re1, x47.0, y47.0);
        sum_re1 = vfmsq_f32(sum_re1, x47.1, y47.1);
        sum_im1 = vfmaq_f32(sum_im1, x47.0, y47.1);
        sum_im1 = vfmaq_f32(sum_im1, x47.1, y47.0);
    }

    let sum_re = vaddq_f32(sum_re0, sum_re1);
    let sum_im = vaddq_f32(sum_im0, sum_im1);

    let mut re = vaddvq_f32(sum_re);
    let mut im = vaddvq_f32(sum_im);

    let base = chunks * 8;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        re += xi.re.mul_add(yi.re, -(xi.im * yi.im));
        im += xi.re.mul_add(yi.im, xi.im * yi.re);
    }

    Complex32::new(re, im)
}

/// AVX2/FMA optimized conjugate dot product for Complex64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dotc_c64_avx2(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    use core::arch::x86_64::*;

    let n = x.len();

    // We process complex numbers as pairs of f64
    // [re0, im0, re1, im1] in each 256-bit register
    let mut sum_re = _mm256_setzero_pd();
    let mut sum_im = _mm256_setzero_pd();

    let chunks = n / 4;
    let remainder = n % 4;

    let x_ptr = x.as_ptr() as *const f64;
    let y_ptr = y.as_ptr() as *const f64;

    for i in 0..chunks {
        let base = i * 8;

        // Load 4 complex numbers (8 f64)
        let x0 = _mm256_loadu_pd(x_ptr.add(base)); // [x0.re, x0.im, x1.re, x1.im]
        let x1 = _mm256_loadu_pd(x_ptr.add(base + 4)); // [x2.re, x2.im, x3.re, x3.im]

        let y0 = _mm256_loadu_pd(y_ptr.add(base));
        let y1 = _mm256_loadu_pd(y_ptr.add(base + 4));

        // Shuffle to get separate re and im
        // For x0: [x0.re, x0.im, x1.re, x1.im]
        // x_re = [x0.re, x0.re, x1.re, x1.re] (duplicate re)
        // x_im = [x0.im, x0.im, x1.im, x1.im] (duplicate im)
        let _x0_re = _mm256_unpacklo_pd(x0, x0); // [x0.re, x0.re, x1.re, x1.re]
        let _x0_im = _mm256_unpackhi_pd(x0, x0); // [x0.im, x0.im, x1.im, x1.im]
        let _x1_re = _mm256_unpacklo_pd(x1, x1);
        let _x1_im = _mm256_unpackhi_pd(x1, x1);

        // Actually, let's use a different approach with proper interleaving
        // Separate real and imaginary parts using permute
        // x_re_im = [x0.re, x0.im, x1.re, x1.im]
        // We need: x_re = [x0.re, x1.re], x_im = [x0.im, x1.im]

        // Use shuffle with immediate to deinterleave
        // For conj(x) * y:
        // real_part = x.re*y.re + x.im*y.im
        // imag_part = x.re*y.im - x.im*y.re

        // Interleaved approach: compute products then horizontal add
        // x0 = [x0.re, x0.im, x1.re, x1.im]
        // y0 = [y0.re, y0.im, y1.re, y1.im]

        // For real: x.re*y.re + x.im*y.im
        // prod_re = x0 * y0 = [x0.re*y0.re, x0.im*y0.im, x1.re*y1.re, x1.im*y1.im]
        let prod0 = _mm256_mul_pd(x0, y0);
        let prod1 = _mm256_mul_pd(x1, y1);

        // Horizontal add pairs: [x0.re*y0.re + x0.im*y0.im, ...]
        let hadd0 = _mm256_hadd_pd(prod0, prod1); // [re0+im0, re2+im2, re1+im1, re3+im3]
        sum_re = _mm256_add_pd(sum_re, hadd0);

        // For imag: x.re*y.im - x.im*y.re
        // Shuffle y: [y0.im, y0.re, y1.im, y1.re]
        let y0_swapped = _mm256_permute_pd(y0, 0b0101);
        let y1_swapped = _mm256_permute_pd(y1, 0b0101);

        // prod_swapped = x0 * y0_swapped = [x0.re*y0.im, x0.im*y0.re, ...]
        let prod0_swapped = _mm256_mul_pd(x0, y0_swapped);
        let prod1_swapped = _mm256_mul_pd(x1, y1_swapped);

        // Horizontal sub for imag: x.re*y.im - x.im*y.re
        let hsub0 = _mm256_hsub_pd(prod0_swapped, prod1_swapped);
        sum_im = _mm256_add_pd(sum_im, hsub0);
    }

    // Reduce
    let mut re_arr = [0.0f64; 4];
    let mut im_arr = [0.0f64; 4];
    _mm256_storeu_pd(re_arr.as_mut_ptr(), sum_re);
    _mm256_storeu_pd(im_arr.as_mut_ptr(), sum_im);

    let mut re = re_arr[0] + re_arr[1] + re_arr[2] + re_arr[3];
    let mut im = im_arr[0] + im_arr[1] + im_arr[2] + im_arr[3];

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        re += xi.re * yi.re + xi.im * yi.im;
        im += xi.re * yi.im - xi.im * yi.re;
    }

    Complex64::new(re, im)
}

/// AVX2/FMA optimized conjugate dot product for Complex32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dotc_c32_avx2(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    use core::arch::x86_64::*;

    let n = x.len();

    let mut sum_re = _mm256_setzero_ps();
    let mut sum_im = _mm256_setzero_ps();

    let chunks = n / 8;
    let remainder = n % 8;

    let x_ptr = x.as_ptr() as *const f32;
    let y_ptr = y.as_ptr() as *const f32;

    for i in 0..chunks {
        let base = i * 16;

        // Load 8 complex numbers (16 f32)
        let x0 = _mm256_loadu_ps(x_ptr.add(base)); // 4 complex
        let x1 = _mm256_loadu_ps(x_ptr.add(base + 8)); // 4 complex

        let y0 = _mm256_loadu_ps(y_ptr.add(base));
        let y1 = _mm256_loadu_ps(y_ptr.add(base + 8));

        // For real part: x.re*y.re + x.im*y.im
        let prod0 = _mm256_mul_ps(x0, y0);
        let prod1 = _mm256_mul_ps(x1, y1);

        // Horizontal add pairs
        let hadd0 = _mm256_hadd_ps(prod0, prod1);
        sum_re = _mm256_add_ps(sum_re, hadd0);

        // For imag: x.re*y.im - x.im*y.re
        // Shuffle y to swap re/im
        let y0_swapped = _mm256_permute_ps(y0, 0b10_11_00_01);
        let y1_swapped = _mm256_permute_ps(y1, 0b10_11_00_01);

        let prod0_swapped = _mm256_mul_ps(x0, y0_swapped);
        let prod1_swapped = _mm256_mul_ps(x1, y1_swapped);

        let hsub0 = _mm256_hsub_ps(prod0_swapped, prod1_swapped);
        sum_im = _mm256_add_ps(sum_im, hsub0);
    }

    // Reduce
    let mut re_arr = [0.0f32; 8];
    let mut im_arr = [0.0f32; 8];
    _mm256_storeu_ps(re_arr.as_mut_ptr(), sum_re);
    _mm256_storeu_ps(im_arr.as_mut_ptr(), sum_im);

    let mut re: f32 = re_arr.iter().sum();
    let mut im: f32 = im_arr.iter().sum();

    let base = chunks * 8;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        re += xi.re * yi.re + xi.im * yi.im;
        im += xi.re * yi.im - xi.im * yi.re;
    }

    Complex32::new(re, im)
}

/// AVX2/FMA optimized unconjugated dot product for Complex64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dotu_c64_avx2(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    use core::arch::x86_64::*;

    let n = x.len();

    let mut sum_re = _mm256_setzero_pd();
    let mut sum_im = _mm256_setzero_pd();

    let chunks = n / 4;
    let remainder = n % 4;

    let x_ptr = x.as_ptr() as *const f64;
    let y_ptr = y.as_ptr() as *const f64;

    // Sign mask for negating imaginary products in real calculation
    let sign_mask = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);

    for i in 0..chunks {
        let base = i * 8;

        let x0 = _mm256_loadu_pd(x_ptr.add(base));
        let x1 = _mm256_loadu_pd(x_ptr.add(base + 4));

        let y0 = _mm256_loadu_pd(y_ptr.add(base));
        let y1 = _mm256_loadu_pd(y_ptr.add(base + 4));

        // For real: x.re*y.re - x.im*y.im
        // Multiply with sign adjustment
        let prod0 = _mm256_mul_pd(x0, y0);
        let prod1 = _mm256_mul_pd(x1, y1);

        let prod0_signed = _mm256_mul_pd(prod0, sign_mask);
        let prod1_signed = _mm256_mul_pd(prod1, sign_mask);

        let hadd0 = _mm256_hadd_pd(prod0_signed, prod1_signed);
        sum_re = _mm256_add_pd(sum_re, hadd0);

        // For imag: x.re*y.im + x.im*y.re
        let y0_swapped = _mm256_permute_pd(y0, 0b0101);
        let y1_swapped = _mm256_permute_pd(y1, 0b0101);

        let prod0_swapped = _mm256_mul_pd(x0, y0_swapped);
        let prod1_swapped = _mm256_mul_pd(x1, y1_swapped);

        let hadd0_im = _mm256_hadd_pd(prod0_swapped, prod1_swapped);
        sum_im = _mm256_add_pd(sum_im, hadd0_im);
    }

    let mut re_arr = [0.0f64; 4];
    let mut im_arr = [0.0f64; 4];
    _mm256_storeu_pd(re_arr.as_mut_ptr(), sum_re);
    _mm256_storeu_pd(im_arr.as_mut_ptr(), sum_im);

    let mut re = re_arr[0] + re_arr[1] + re_arr[2] + re_arr[3];
    let mut im = im_arr[0] + im_arr[1] + im_arr[2] + im_arr[3];

    let base = chunks * 4;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        re += xi.re * yi.re - xi.im * yi.im;
        im += xi.re * yi.im + xi.im * yi.re;
    }

    Complex64::new(re, im)
}

/// AVX2/FMA optimized unconjugated dot product for Complex32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dotu_c32_avx2(x: &[Complex32], y: &[Complex32]) -> Complex32 {
    use core::arch::x86_64::*;

    let n = x.len();

    let mut sum_re = _mm256_setzero_ps();
    let mut sum_im = _mm256_setzero_ps();

    let chunks = n / 8;
    let remainder = n % 8;

    let x_ptr = x.as_ptr() as *const f32;
    let y_ptr = y.as_ptr() as *const f32;

    let sign_mask = _mm256_set_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);

    for i in 0..chunks {
        let base = i * 16;

        let x0 = _mm256_loadu_ps(x_ptr.add(base));
        let x1 = _mm256_loadu_ps(x_ptr.add(base + 8));

        let y0 = _mm256_loadu_ps(y_ptr.add(base));
        let y1 = _mm256_loadu_ps(y_ptr.add(base + 8));

        // For real: x.re*y.re - x.im*y.im
        let prod0 = _mm256_mul_ps(x0, y0);
        let prod1 = _mm256_mul_ps(x1, y1);

        let prod0_signed = _mm256_mul_ps(prod0, sign_mask);
        let prod1_signed = _mm256_mul_ps(prod1, sign_mask);

        let hadd0 = _mm256_hadd_ps(prod0_signed, prod1_signed);
        sum_re = _mm256_add_ps(sum_re, hadd0);

        // For imag: x.re*y.im + x.im*y.re
        let y0_swapped = _mm256_permute_ps(y0, 0b10_11_00_01);
        let y1_swapped = _mm256_permute_ps(y1, 0b10_11_00_01);

        let prod0_swapped = _mm256_mul_ps(x0, y0_swapped);
        let prod1_swapped = _mm256_mul_ps(x1, y1_swapped);

        let hadd0_im = _mm256_hadd_ps(prod0_swapped, prod1_swapped);
        sum_im = _mm256_add_ps(sum_im, hadd0_im);
    }

    let mut re_arr = [0.0f32; 8];
    let mut im_arr = [0.0f32; 8];
    _mm256_storeu_ps(re_arr.as_mut_ptr(), sum_re);
    _mm256_storeu_ps(im_arr.as_mut_ptr(), sum_im);

    let mut re: f32 = re_arr.iter().sum();
    let mut im: f32 = im_arr.iter().sum();

    let base = chunks * 8;
    for i in 0..remainder {
        let xi = x[base + i];
        let yi = y[base + i];
        re += xi.re * yi.re - xi.im * yi.im;
        im += xi.re * yi.im + xi.im * yi.re;
    }

    Complex32::new(re, im)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f64() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0f64, 4.0, 3.0, 2.0, 1.0];

        // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
        let result = dot(&x, &y);
        assert!((result - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_f32() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];

        let result = dot(&x, &y);
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_empty() {
        let x: [f64; 0] = [];
        let y: [f64; 0] = [];

        let result = dot(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_single() {
        let x = [3.0f64];
        let y = [4.0f64];

        let result = dot(&x, &y);
        assert!((result - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_large() {
        let n = 1000;
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![1.0; n];

        // Sum of 1 to n = n*(n+1)/2 = 500500
        let result = dot(&x, &y);
        assert!((result - 500500.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_f64_simd() {
        let n = 1000;
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![1.0; n];

        let result = dot_f64(&x, &y);
        let expected = 500500.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_dot_f32_simd() {
        let n = 1000;
        let x: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let y: Vec<f32> = vec![1.0; n];

        let result = dot_f32(&x, &y);
        let expected = 500500.0f32;
        assert!(
            (result - expected).abs() < 1.0, // f32 has less precision
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_dot_f64_large() {
        // Test with large vector to exercise blocked path
        let n = 100000;
        let x: Vec<f64> = vec![1.0; n];
        let y: Vec<f64> = vec![1.0; n];

        let result = dot_f64(&x, &y);
        assert!(
            (result - n as f64).abs() < 1e-6,
            "Expected {}, got {}",
            n,
            result
        );
    }

    #[test]
    fn test_dot_f32_large() {
        // Test with large vector to exercise blocked path
        let n = 100000;
        let x: Vec<f32> = vec![1.0; n];
        let y: Vec<f32> = vec![1.0; n];

        let result = dot_f32(&x, &y);
        assert!(
            (result - n as f32).abs() < 100.0, // f32 accumulation error grows
            "Expected {}, got {}",
            n,
            result
        );
    }

    #[test]
    fn test_dot_f64_consistency() {
        // Test that SIMD and generic implementations give consistent results
        let n = 500;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let y: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 0.1).collect();

        let result_generic = dot(&x, &y);
        let result_simd = dot_f64(&x, &y);

        assert!(
            (result_generic - result_simd).abs() / result_generic.abs() < 1e-10,
            "Generic: {}, SIMD: {}",
            result_generic,
            result_simd
        );
    }

    #[test]
    fn test_dot_edge_cases() {
        // Empty vectors
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        assert_eq!(dot_f64(&x, &y), 0.0);

        // Single element
        let x = vec![5.0f64];
        let y = vec![3.0f64];
        assert!((dot_f64(&x, &y) - 15.0).abs() < 1e-10);

        // Odd length
        let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0f64, 1.0, 1.0, 1.0, 1.0];
        assert!((dot_f64(&x, &y) - 15.0).abs() < 1e-10);
    }

    // =============================================================================
    // Complex dot product tests
    // =============================================================================

    #[test]
    fn test_dotc_c64_basic() {
        // Test conjugate dot product
        // x = [1+2i, 3+4i], y = [5+6i, 7+8i]
        // conj(x) * y = (1-2i)(5+6i) + (3-4i)(7+8i)
        //             = (5+12) + (6-10)i + (21+32) + (24-28)i
        //             = 17 + (-4)i + 53 + (-4)i
        //             = 70 + (-8)i
        let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        let result = dotc_c64(&x, &y);
        assert!((result.re - 70.0).abs() < 1e-10, "re: {}", result.re);
        assert!((result.im - (-8.0)).abs() < 1e-10, "im: {}", result.im);
    }

    #[test]
    fn test_dotu_c64_basic() {
        // Test unconjugated dot product
        // x = [1+2i, 3+4i], y = [5+6i, 7+8i]
        // x * y = (1+2i)(5+6i) + (3+4i)(7+8i)
        //       = (5-12) + (6+10)i + (21-32) + (24+28)i
        //       = -7 + 16i + (-11) + 52i
        //       = -18 + 68i
        let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        let result = dotu_c64(&x, &y);
        assert!((result.re - (-18.0)).abs() < 1e-10, "re: {}", result.re);
        assert!((result.im - 68.0).abs() < 1e-10, "im: {}", result.im);
    }

    #[test]
    fn test_dotc_c32_basic() {
        let x = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = vec![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

        let result = dotc_c32(&x, &y);
        assert!((result.re - 70.0).abs() < 1e-5, "re: {}", result.re);
        assert!((result.im - (-8.0)).abs() < 1e-5, "im: {}", result.im);
    }

    #[test]
    fn test_dotu_c32_basic() {
        let x = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = vec![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

        let result = dotu_c32(&x, &y);
        assert!((result.re - (-18.0)).abs() < 1e-5, "re: {}", result.re);
        assert!((result.im - 68.0).abs() < 1e-5, "im: {}", result.im);
    }

    #[test]
    fn test_dotc_c64_empty() {
        let x: Vec<Complex64> = vec![];
        let y: Vec<Complex64> = vec![];

        let result = dotc_c64(&x, &y);
        assert_eq!(result, Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_dotc_c64_single() {
        let x = vec![Complex64::new(3.0, 4.0)];
        let y = vec![Complex64::new(1.0, 2.0)];

        // conj(3+4i) * (1+2i) = (3-4i)(1+2i) = 3 + 6i - 4i - 8i^2 = 3 + 2i + 8 = 11 + 2i
        let result = dotc_c64(&x, &y);
        assert!((result.re - 11.0).abs() < 1e-10);
        assert!((result.im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_dotc_c64_medium() {
        // Medium-sized test to exercise SIMD path but verify correctness
        let n = 100;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64, (i as f64) * 0.5))
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(1.0, 0.1 * i as f64))
            .collect();

        let result_simd = dotc_c64(&x, &y);

        // Compare with scalar implementation
        let result_scalar = dotc_c64_scalar(&x, &y);

        assert!(
            (result_simd.re - result_scalar.re).abs() < 1e-6,
            "re: simd={}, scalar={}",
            result_simd.re,
            result_scalar.re
        );
        assert!(
            (result_simd.im - result_scalar.im).abs() < 1e-6,
            "im: simd={}, scalar={}",
            result_simd.im,
            result_scalar.im
        );
    }

    #[test]
    fn test_dotc_c64_large() {
        // Large test to exercise SIMD path
        let n = 10000;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i % 100) as f64, ((i + 50) % 100) as f64))
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(((i + 25) % 100) as f64, ((i + 75) % 100) as f64))
            .collect();

        let result_simd = dotc_c64(&x, &y);

        // Compare with scalar implementation
        let result_scalar = dotc_c64_scalar(&x, &y);

        let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
        let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

        assert!(
            re_err < 1e-10,
            "re: simd={}, scalar={}, rel_err={}",
            result_simd.re,
            result_scalar.re,
            re_err
        );
        assert!(
            im_err < 1e-10,
            "im: simd={}, scalar={}, rel_err={}",
            result_simd.im,
            result_scalar.im,
            im_err
        );
    }

    #[test]
    fn test_dotu_c64_large() {
        let n = 10000;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i % 100) as f64, ((i + 50) % 100) as f64))
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(((i + 25) % 100) as f64, ((i + 75) % 100) as f64))
            .collect();

        let result_simd = dotu_c64(&x, &y);
        let result_scalar = dotu_c64_scalar(&x, &y);

        let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
        let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

        assert!(
            re_err < 1e-10,
            "re: simd={}, scalar={}, rel_err={}",
            result_simd.re,
            result_scalar.re,
            re_err
        );
        assert!(
            im_err < 1e-10,
            "im: simd={}, scalar={}, rel_err={}",
            result_simd.im,
            result_scalar.im,
            im_err
        );
    }

    #[test]
    fn test_dotc_c32_large() {
        let n = 10000;
        let x: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i % 100) as f32, ((i + 50) % 100) as f32))
            .collect();
        let y: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(((i + 25) % 100) as f32, ((i + 75) % 100) as f32))
            .collect();

        let result_simd = dotc_c32(&x, &y);
        let result_scalar = dotc_c32_scalar(&x, &y);

        let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
        let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

        assert!(
            re_err < 1e-4,
            "re: simd={}, scalar={}, rel_err={}",
            result_simd.re,
            result_scalar.re,
            re_err
        );
        assert!(
            im_err < 1e-4,
            "im: simd={}, scalar={}, rel_err={}",
            result_simd.im,
            result_scalar.im,
            im_err
        );
    }

    #[test]
    fn test_dotu_c32_large() {
        let n = 10000;
        let x: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i % 100) as f32, ((i + 50) % 100) as f32))
            .collect();
        let y: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(((i + 25) % 100) as f32, ((i + 75) % 100) as f32))
            .collect();

        let result_simd = dotu_c32(&x, &y);
        let result_scalar = dotu_c32_scalar(&x, &y);

        let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
        let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

        assert!(
            re_err < 1e-4,
            "re: simd={}, scalar={}, rel_err={}",
            result_simd.re,
            result_scalar.re,
            re_err
        );
        assert!(
            im_err < 1e-4,
            "im: simd={}, scalar={}, rel_err={}",
            result_simd.im,
            result_scalar.im,
            im_err
        );
    }

    #[test]
    fn test_dotc_c64_self_dot() {
        // Dot product with conjugate should give real result for x^H * x
        let n = 100;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64, (i as f64) * 0.5))
            .collect();

        let result = dotc_c64(&x, &x);

        // x^H * x should be purely real (imaginary part should be ~0)
        assert!(
            result.im.abs() < 1e-10,
            "x^H * x should be real, but im={}",
            result.im
        );

        // x^H * x should equal sum of |x[i]|^2
        let expected_re: f64 = x.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (result.re - expected_re).abs() < 1e-6,
            "re: {}, expected: {}",
            result.re,
            expected_re
        );
    }

    #[test]
    fn test_complex_dot_remainder_handling() {
        // Test various sizes to exercise remainder handling
        for n in [1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65] {
            let x: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, 1.0)).collect();
            let y: Vec<Complex64> = (0..n).map(|i| Complex64::new(1.0, i as f64)).collect();

            let result_simd = dotc_c64(&x, &y);
            let result_scalar = dotc_c64_scalar(&x, &y);

            assert!(
                (result_simd.re - result_scalar.re).abs() < 1e-10,
                "n={}: re mismatch",
                n
            );
            assert!(
                (result_simd.im - result_scalar.im).abs() < 1e-10,
                "n={}: im mismatch",
                n
            );
        }
    }
}
