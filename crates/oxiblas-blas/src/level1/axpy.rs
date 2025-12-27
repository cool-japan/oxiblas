//! AXPY: y = α·x + y
//!
//! One of the most fundamental BLAS operations.
//!
//! This module provides optimized implementations using SIMD instructions
//! on supported platforms (AVX2 on `x86_64`, NEON on aarch64).

use oxiblas_core::scalar::Field;

/// Computes y = α·x + y.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::axpy;
///
/// let x = [1.0f64, 2.0, 3.0];
/// let mut y = [4.0f64, 5.0, 6.0];
///
/// // y = 2*x + y = [6, 9, 12]
/// axpy(2.0, &x, &mut y);
///
/// assert!((y[0] - 6.0).abs() < 1e-10);
/// assert!((y[1] - 9.0).abs() < 1e-10);
/// assert!((y[2] - 12.0).abs() < 1e-10);
/// ```
pub fn axpy<T: Field>(alpha: T, x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 || alpha == T::zero() {
        return;
    }

    // Generic implementation - use specialized functions for SIMD
    axpy_generic(alpha, x, y);
}

/// SIMD-optimized AXPY for f64 (standalone function).
#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 || alpha == 0.0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                axpy_f64_avx2(alpha, x, y);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            axpy_f64_neon(alpha, x, y);
        }
        return;
    }

    #[allow(unreachable_code)]
    axpy_f64_scalar(alpha, x, y);
}

/// SIMD-optimized AXPY for f32 (standalone function).
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 || alpha == 0.0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                axpy_f32_avx2(alpha, x, y);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            axpy_f32_neon(alpha, x, y);
        }
        return;
    }

    #[allow(unreachable_code)]
    axpy_f32_scalar(alpha, x, y);
}

/// Generic AXPY implementation with loop unrolling.
#[inline]
fn axpy_generic<T: Field>(alpha: T, x: &[T], y: &mut [T]) {
    let n = x.len();

    // Unroll by 8 for better pipelining
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let base = i * 8;
        y[base] = alpha * x[base] + y[base];
        y[base + 1] = alpha * x[base + 1] + y[base + 1];
        y[base + 2] = alpha * x[base + 2] + y[base + 2];
        y[base + 3] = alpha * x[base + 3] + y[base + 3];
        y[base + 4] = alpha * x[base + 4] + y[base + 4];
        y[base + 5] = alpha * x[base + 5] + y[base + 5];
        y[base + 6] = alpha * x[base + 6] + y[base + 6];
        y[base + 7] = alpha * x[base + 7] + y[base + 7];
    }

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        y[base + i] = alpha * x[base + i] + y[base + i];
    }
}

/// Scalar AXPY for f64 with 8-way unrolling.
#[inline]
fn axpy_f64_scalar(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let base = i * 8;
        y[base] = alpha.mul_add(x[base], y[base]);
        y[base + 1] = alpha.mul_add(x[base + 1], y[base + 1]);
        y[base + 2] = alpha.mul_add(x[base + 2], y[base + 2]);
        y[base + 3] = alpha.mul_add(x[base + 3], y[base + 3]);
        y[base + 4] = alpha.mul_add(x[base + 4], y[base + 4]);
        y[base + 5] = alpha.mul_add(x[base + 5], y[base + 5]);
        y[base + 6] = alpha.mul_add(x[base + 6], y[base + 6]);
        y[base + 7] = alpha.mul_add(x[base + 7], y[base + 7]);
    }

    let base = chunks * 8;
    for i in 0..remainder {
        y[base + i] = alpha.mul_add(x[base + i], y[base + i]);
    }
}

/// Scalar AXPY for f32 with 8-way unrolling.
#[inline]
fn axpy_f32_scalar(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let base = i * 8;
        y[base] = alpha.mul_add(x[base], y[base]);
        y[base + 1] = alpha.mul_add(x[base + 1], y[base + 1]);
        y[base + 2] = alpha.mul_add(x[base + 2], y[base + 2]);
        y[base + 3] = alpha.mul_add(x[base + 3], y[base + 3]);
        y[base + 4] = alpha.mul_add(x[base + 4], y[base + 4]);
        y[base + 5] = alpha.mul_add(x[base + 5], y[base + 5]);
        y[base + 6] = alpha.mul_add(x[base + 6], y[base + 6]);
        y[base + 7] = alpha.mul_add(x[base + 7], y[base + 7]);
    }

    let base = chunks * 8;
    for i in 0..remainder {
        y[base + i] = alpha.mul_add(x[base + i], y[base + i]);
    }
}

/// AVX2 + FMA optimized AXPY for f64.
///
/// Processes 16 elements per iteration (4 AVX2 registers × 4 f64 each).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_f64_avx2(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::x86_64::*;

    let n = x.len();
    let alpha_vec = _mm256_set1_pd(alpha);

    // Process 16 elements per iteration (4 × 4 f64)
    let chunks = n / 16;
    let remainder = n % 16;

    for i in 0..chunks {
        let base = i * 16;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_mut_ptr().add(base);

        // Load 4 vectors
        let x0 = _mm256_loadu_pd(x_ptr);
        let x1 = _mm256_loadu_pd(x_ptr.add(4));
        let x2 = _mm256_loadu_pd(x_ptr.add(8));
        let x3 = _mm256_loadu_pd(x_ptr.add(12));

        let y0 = _mm256_loadu_pd(y_ptr);
        let y1 = _mm256_loadu_pd(y_ptr.add(4));
        let y2 = _mm256_loadu_pd(y_ptr.add(8));
        let y3 = _mm256_loadu_pd(y_ptr.add(12));

        // y = alpha * x + y using FMA
        let r0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
        let r1 = _mm256_fmadd_pd(alpha_vec, x1, y1);
        let r2 = _mm256_fmadd_pd(alpha_vec, x2, y2);
        let r3 = _mm256_fmadd_pd(alpha_vec, x3, y3);

        // Store results
        _mm256_storeu_pd(y_ptr, r0);
        _mm256_storeu_pd(y_ptr.add(4), r1);
        _mm256_storeu_pd(y_ptr.add(8), r2);
        _mm256_storeu_pd(y_ptr.add(12), r3);
    }

    // Handle remaining elements in groups of 4
    let base = chunks * 16;
    let remaining_chunks = remainder / 4;

    for i in 0..remaining_chunks {
        let idx = base + i * 4;
        let x_ptr = x.as_ptr().add(idx);
        let y_ptr = y.as_mut_ptr().add(idx);

        let x_vec = _mm256_loadu_pd(x_ptr);
        let y_vec = _mm256_loadu_pd(y_ptr);
        let r = _mm256_fmadd_pd(alpha_vec, x_vec, y_vec);
        _mm256_storeu_pd(y_ptr, r);
    }

    // Handle final scalar remainder
    let scalar_base = base + remaining_chunks * 4;
    for i in scalar_base..n {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

/// AVX2 + FMA optimized AXPY for f32.
///
/// Processes 32 elements per iteration (4 AVX2 registers × 8 f32 each).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_f32_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;

    let n = x.len();
    let alpha_vec = _mm256_set1_ps(alpha);

    // Process 32 elements per iteration (4 × 8 f32)
    let chunks = n / 32;
    let remainder = n % 32;

    for i in 0..chunks {
        let base = i * 32;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_mut_ptr().add(base);

        // Load 4 vectors
        let x0 = _mm256_loadu_ps(x_ptr);
        let x1 = _mm256_loadu_ps(x_ptr.add(8));
        let x2 = _mm256_loadu_ps(x_ptr.add(16));
        let x3 = _mm256_loadu_ps(x_ptr.add(24));

        let y0 = _mm256_loadu_ps(y_ptr);
        let y1 = _mm256_loadu_ps(y_ptr.add(8));
        let y2 = _mm256_loadu_ps(y_ptr.add(16));
        let y3 = _mm256_loadu_ps(y_ptr.add(24));

        // y = alpha * x + y using FMA
        let r0 = _mm256_fmadd_ps(alpha_vec, x0, y0);
        let r1 = _mm256_fmadd_ps(alpha_vec, x1, y1);
        let r2 = _mm256_fmadd_ps(alpha_vec, x2, y2);
        let r3 = _mm256_fmadd_ps(alpha_vec, x3, y3);

        // Store results
        _mm256_storeu_ps(y_ptr, r0);
        _mm256_storeu_ps(y_ptr.add(8), r1);
        _mm256_storeu_ps(y_ptr.add(16), r2);
        _mm256_storeu_ps(y_ptr.add(24), r3);
    }

    // Handle remaining elements in groups of 8
    let base = chunks * 32;
    let remaining_chunks = remainder / 8;

    for i in 0..remaining_chunks {
        let idx = base + i * 8;
        let x_ptr = x.as_ptr().add(idx);
        let y_ptr = y.as_mut_ptr().add(idx);

        let x_vec = _mm256_loadu_ps(x_ptr);
        let y_vec = _mm256_loadu_ps(y_ptr);
        let r = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);
        _mm256_storeu_ps(y_ptr, r);
    }

    // Handle final scalar remainder
    let scalar_base = base + remaining_chunks * 8;
    for i in scalar_base..n {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

/// NEON optimized AXPY for f64.
///
/// Processes 8 elements per iteration (4 NEON registers × 2 f64 each).
#[cfg(target_arch = "aarch64")]
unsafe fn axpy_f64_neon(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

    let n = x.len();
    let alpha_vec = vdupq_n_f64(alpha);

    // Process 8 elements per iteration (4 × 2 f64)
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let base = i * 8;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_mut_ptr().add(base);

        // Load 4 vectors
        let x0 = vld1q_f64(x_ptr);
        let x1 = vld1q_f64(x_ptr.add(2));
        let x2 = vld1q_f64(x_ptr.add(4));
        let x3 = vld1q_f64(x_ptr.add(6));

        let y0 = vld1q_f64(y_ptr);
        let y1 = vld1q_f64(y_ptr.add(2));
        let y2 = vld1q_f64(y_ptr.add(4));
        let y3 = vld1q_f64(y_ptr.add(6));

        // y = alpha * x + y using FMA
        let r0 = vfmaq_f64(y0, alpha_vec, x0);
        let r1 = vfmaq_f64(y1, alpha_vec, x1);
        let r2 = vfmaq_f64(y2, alpha_vec, x2);
        let r3 = vfmaq_f64(y3, alpha_vec, x3);

        // Store results
        vst1q_f64(y_ptr, r0);
        vst1q_f64(y_ptr.add(2), r1);
        vst1q_f64(y_ptr.add(4), r2);
        vst1q_f64(y_ptr.add(6), r3);
    }

    // Handle remaining elements in groups of 2
    let base = chunks * 8;
    let remaining_chunks = remainder / 2;

    for i in 0..remaining_chunks {
        let idx = base + i * 2;
        let x_ptr = x.as_ptr().add(idx);
        let y_ptr = y.as_mut_ptr().add(idx);

        let x_vec = vld1q_f64(x_ptr);
        let y_vec = vld1q_f64(y_ptr);
        let r = vfmaq_f64(y_vec, alpha_vec, x_vec);
        vst1q_f64(y_ptr, r);
    }

    // Handle final scalar remainder
    let scalar_base = base + remaining_chunks * 2;
    for i in scalar_base..n {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

/// NEON optimized AXPY for f32.
///
/// Processes 16 elements per iteration (4 NEON registers × 4 f32 each).
#[cfg(target_arch = "aarch64")]
unsafe fn axpy_f32_neon(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

    let n = x.len();
    let alpha_vec = vdupq_n_f32(alpha);

    // Process 16 elements per iteration (4 × 4 f32)
    let chunks = n / 16;
    let remainder = n % 16;

    for i in 0..chunks {
        let base = i * 16;
        let x_ptr = x.as_ptr().add(base);
        let y_ptr = y.as_mut_ptr().add(base);

        // Load 4 vectors
        let x0 = vld1q_f32(x_ptr);
        let x1 = vld1q_f32(x_ptr.add(4));
        let x2 = vld1q_f32(x_ptr.add(8));
        let x3 = vld1q_f32(x_ptr.add(12));

        let y0 = vld1q_f32(y_ptr);
        let y1 = vld1q_f32(y_ptr.add(4));
        let y2 = vld1q_f32(y_ptr.add(8));
        let y3 = vld1q_f32(y_ptr.add(12));

        // y = alpha * x + y using FMA
        let r0 = vfmaq_f32(y0, alpha_vec, x0);
        let r1 = vfmaq_f32(y1, alpha_vec, x1);
        let r2 = vfmaq_f32(y2, alpha_vec, x2);
        let r3 = vfmaq_f32(y3, alpha_vec, x3);

        // Store results
        vst1q_f32(y_ptr, r0);
        vst1q_f32(y_ptr.add(4), r1);
        vst1q_f32(y_ptr.add(8), r2);
        vst1q_f32(y_ptr.add(12), r3);
    }

    // Handle remaining elements in groups of 4
    let base = chunks * 16;
    let remaining_chunks = remainder / 4;

    for i in 0..remaining_chunks {
        let idx = base + i * 4;
        let x_ptr = x.as_ptr().add(idx);
        let y_ptr = y.as_mut_ptr().add(idx);

        let x_vec = vld1q_f32(x_ptr);
        let y_vec = vld1q_f32(y_ptr);
        let r = vfmaq_f32(y_vec, alpha_vec, x_vec);
        vst1q_f32(y_ptr, r);
    }

    // Handle final scalar remainder
    let scalar_base = base + remaining_chunks * 4;
    for i in scalar_base..n {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axpy_basic() {
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let mut y = [5.0f64, 6.0, 7.0, 8.0];

        axpy(2.0, &x, &mut y);

        // y = 2*[1,2,3,4] + [5,6,7,8] = [7, 10, 13, 16]
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 10.0).abs() < 1e-10);
        assert!((y[2] - 13.0).abs() < 1e-10);
        assert!((y[3] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_negative_alpha() {
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [4.0f64, 5.0, 6.0];

        axpy(-1.0, &x, &mut y);

        // y = -1*[1,2,3] + [4,5,6] = [3, 3, 3]
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_zero_alpha() {
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [4.0f64, 5.0, 6.0];
        let y_orig = y;

        axpy(0.0, &x, &mut y);

        // y should be unchanged
        assert_eq!(y, y_orig);
    }

    #[test]
    fn test_axpy_f32() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [0.0f32, 0.0, 0.0];

        axpy(1.0, &x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-5);
        assert!((y[1] - 2.0).abs() < 1e-5);
        assert!((y[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_axpy_odd_length() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut y = [0.0f64, 0.0, 0.0, 0.0, 0.0];

        axpy(3.0, &x, &mut y);

        for i in 0..5 {
            assert!((y[i] - 3.0 * x[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_axpy_f64_simd() {
        // Test with length that exercises SIMD paths
        let n = 100;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut y: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let y_expected: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) + (i * 2) as f64).collect();

        axpy_f64(2.0, &x, &mut y);

        for i in 0..n {
            assert!(
                (y[i] - y_expected[i]).abs() < 1e-10,
                "Mismatch at index {}: got {}, expected {}",
                i,
                y[i],
                y_expected[i]
            );
        }
    }

    #[test]
    fn test_axpy_f32_simd() {
        // Test with length that exercises SIMD paths
        let n = 100;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let y_expected: Vec<f32> = (0..n).map(|i| 2.0 * (i as f32) + (i * 2) as f32).collect();

        axpy_f32(2.0, &x, &mut y);

        for i in 0..n {
            assert!(
                (y[i] - y_expected[i]).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                y[i],
                y_expected[i]
            );
        }
    }

    #[test]
    fn test_axpy_large_vector() {
        // Test large vectors to ensure SIMD paths work correctly
        let n = 10000;
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![2.0; n];

        axpy_f64(3.0, &x, &mut y);

        for i in 0..n {
            assert!(
                (y[i] - 5.0).abs() < 1e-10,
                "Mismatch at index {}: got {}, expected 5.0",
                i,
                y[i]
            );
        }
    }

    #[test]
    fn test_axpy_edge_cases() {
        // Test empty vectors
        let x: Vec<f64> = vec![];
        let mut y: Vec<f64> = vec![];
        axpy_f64(1.0, &x, &mut y);
        assert!(y.is_empty());

        // Test single element
        let x = [5.0f64];
        let mut y = [3.0f64];
        axpy_f64(2.0, &x, &mut y);
        assert!((y[0] - 13.0).abs() < 1e-10);

        // Test two elements
        let x = [1.0f64, 2.0];
        let mut y = [3.0f64, 4.0];
        axpy_f64(2.0, &x, &mut y);
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 8.0).abs() < 1e-10);
    }
}
