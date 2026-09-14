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

    // Other architectures: the blocked driver falls back to the unrolled
    // scalar kernel per block, which bounds error growth on long vectors.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        return dot_f64_blocked(x, y);
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

    // Other architectures: the blocked driver falls back to the unrolled
    // scalar kernel per block, which bounds error growth on long vectors.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        return dot_f32_blocked(x, y);
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

    // Sign mask for the real part of x*y = re*re - im*im.
    //
    // WHY the exact literal order matters: after `_mm256_loadu_pd` of the
    // interleaved [re, im, re, im] layout the lanes are lane0=re, lane1=im,
    // lane2=re, lane3=im, so `prod = x*y` holds [re*re, im*im, re*re, im*im].
    // Only the im*im lanes (odd lanes 1 and 3) may be negated. `_mm256_set_pd`
    // fills lanes in reverse argument order (e3, e2, e1, e0), so this literal
    // yields per-lane multipliers [lane0=+1, lane1=-1, lane2=+1, lane3=-1].
    // Reversing it negates the re*re lanes instead and flips the sign of the
    // entire real part (historic ZDOTU/CDOTU correctness bug on AVX2).
    let sign_mask = _mm256_set_pd(-1.0, 1.0, -1.0, 1.0);

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

    // See dotu_c64_avx2 for the rationale: `_mm256_set_ps` fills lanes in
    // reverse argument order, so this literal yields per-lane multipliers
    // [+1, -1, +1, -1, +1, -1, +1, -1], negating exactly the odd (im*im) lanes
    // of the interleaved [re, im, ...] layout so the real part stays re*re-im*im.
    let sign_mask = _mm256_set_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

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
#[path = "dot_tests.rs"]
mod tests;
