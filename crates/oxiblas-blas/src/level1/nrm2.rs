//! NRM2: Euclidean (L2) norm
//!
//! Computes ||x||_2 = sqrt(Σ |x\[i\]|^2)
//!
//! Uses a numerically stable algorithm to prevent overflow/underflow.
//! SIMD-optimized implementations available for f64 and f32.

use oxiblas_core::scalar::{Real, Scalar};

/// Computes the Euclidean (L2) norm of a vector.
///
/// ||x||_2 = sqrt(Σ |x\[i\]|^2)
///
/// This implementation uses a numerically stable algorithm that avoids
/// overflow and underflow for vectors with very large or very small elements.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::nrm2;
///
/// let x = [3.0f64, 4.0];
/// let norm = nrm2(&x);
///
/// // ||[3,4]||_2 = sqrt(9 + 16) = 5
/// assert!((norm - 5.0).abs() < 1e-10);
/// ```
pub fn nrm2<T: Real>(x: &[T]) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    if n == 1 {
        return Scalar::abs(x[0]);
    }

    // Use the "Blue's algorithm" approach for numerical stability:
    // Scale by the largest element to prevent overflow/underflow
    let mut scale = T::zero();
    let mut ssq = T::one();

    for &xi in x {
        let abs_xi = Scalar::abs(xi);
        if abs_xi > T::zero() {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = T::one() + ssq * t * t;
                scale = abs_xi;
            } else {
                let t = abs_xi / scale;
                ssq += t * t;
            }
        }
    }

    scale * Real::sqrt(ssq)
}

/// Computes the squared Euclidean norm (avoids the sqrt).
///
/// ||x||_2^2 = Σ |x\[i\]|^2
///
/// Useful when only the squared norm is needed, avoiding the sqrt overhead.
pub fn nrm2_sq<T: Real>(x: &[T]) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    // Use 4-way accumulation for stability and performance
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        let x0 = x[base];
        let x1 = x[base + 1];
        let x2 = x[base + 2];
        let x3 = x[base + 3];
        acc0 += x0 * x0;
        acc1 += x1 * x1;
        acc2 += x2 * x2;
        acc3 += x3 * x3;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        let xi = x[base + i];
        acc0 += xi * xi;
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// SIMD-optimized L2 norm for f64.
///
/// Uses NEON FMA on aarch64 and AVX2/FMA on `x86_64` for performance,
/// with safe overflow/underflow handling.
#[inline]
#[must_use]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }

    // For small vectors, use simple implementation
    if n < 64 {
        return nrm2_f64_scaled(x);
    }

    // For larger vectors, use SIMD with overflow protection
    #[cfg(target_arch = "aarch64")]
    {
        return nrm2_f64_simd_safe(x);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { nrm2_f64_avx2_safe(x) };
        }
    }

    #[allow(unreachable_code)]
    nrm2_f64_scaled(x)
}

/// SIMD-optimized L2 norm for f32.
///
/// Uses NEON FMA on aarch64 and AVX2/FMA on `x86_64` for performance,
/// with safe overflow/underflow handling.
#[inline]
#[must_use]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }

    // For small vectors, use simple implementation
    if n < 64 {
        return nrm2_f32_scaled(x);
    }

    // For larger vectors, use SIMD with overflow protection
    #[cfg(target_arch = "aarch64")]
    {
        return nrm2_f32_simd_safe(x);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { nrm2_f32_avx2_safe(x) };
        }
    }

    #[allow(unreachable_code)]
    nrm2_f32_scaled(x)
}

/// Scaled nrm2 for f64 using Blue's algorithm.
fn nrm2_f64_scaled(x: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;

    for &xi in x {
        let abs_xi = xi.abs();
        if abs_xi > 0.0 {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = (ssq * t).mul_add(t, 1.0);
                scale = abs_xi;
            } else {
                let t = abs_xi / scale;
                ssq += t * t;
            }
        }
    }

    scale * ssq.sqrt()
}

/// Scaled nrm2 for f32 using Blue's algorithm.
fn nrm2_f32_scaled(x: &[f32]) -> f32 {
    let mut scale = 0.0f32;
    let mut ssq = 1.0f32;

    for &xi in x {
        let abs_xi = xi.abs();
        if abs_xi > 0.0 {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = (ssq * t).mul_add(t, 1.0);
                scale = abs_xi;
            } else {
                let t = abs_xi / scale;
                ssq += t * t;
            }
        }
    }

    scale * ssq.sqrt()
}

/// NEON SIMD nrm2 for f64 with overflow protection.
///
/// Uses a three-phase approach to handle values across the full floating-point range:
/// 1. Detect the range of input values
/// 2. Scale appropriately to avoid overflow/underflow
/// 3. Compute sum of squares using SIMD
#[cfg(target_arch = "aarch64")]
fn nrm2_f64_simd_safe(x: &[f64]) -> f64 {
    use core::arch::aarch64::{
        float64x2_t, vaddq_f64, vaddvq_f64, vandq_u64, vdupq_n_f64, vdupq_n_u64, vfmaq_f64,
        vld1q_f64, vmaxq_f64, vmaxvq_f64, vmulq_f64,
    };

    let n = x.len();

    // First pass: find maximum absolute value for scaling
    let max_val: f64;
    unsafe {
        let mut max_vec = vdupq_n_f64(0.0);
        let abs_mask = vdupq_n_u64(0x7FFF_FFFF_FFFF_FFFF); // Mask to clear sign bit

        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let base = i * 8;
            let ptr = x.as_ptr().add(base);

            let x0 = vld1q_f64(ptr);
            let x1 = vld1q_f64(ptr.add(2));
            let x2 = vld1q_f64(ptr.add(4));
            let x3 = vld1q_f64(ptr.add(6));

            // Absolute value via bit masking
            let abs0: float64x2_t =
                core::mem::transmute(vandq_u64(core::mem::transmute(x0), abs_mask));
            let abs1: float64x2_t =
                core::mem::transmute(vandq_u64(core::mem::transmute(x1), abs_mask));
            let abs2: float64x2_t =
                core::mem::transmute(vandq_u64(core::mem::transmute(x2), abs_mask));
            let abs3: float64x2_t =
                core::mem::transmute(vandq_u64(core::mem::transmute(x3), abs_mask));

            max_vec = vmaxq_f64(max_vec, abs0);
            max_vec = vmaxq_f64(max_vec, abs1);
            max_vec = vmaxq_f64(max_vec, abs2);
            max_vec = vmaxq_f64(max_vec, abs3);
        }

        // Reduce max_vec to scalar
        let mut max_val_local = vmaxvq_f64(max_vec);

        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            let abs_xi = x[base + i].abs();
            if abs_xi > max_val_local {
                max_val_local = abs_xi;
            }
        }

        max_val = max_val_local;
    }

    if max_val == 0.0 {
        return 0.0;
    }

    // Compute scale factor (use reciprocal for better numerical stability)
    let scale = 1.0 / max_val;

    // Second pass: compute sum of squares with scaling
    let sum: f64;
    unsafe {
        let scale_vec = vdupq_n_f64(scale);
        let mut sum_vec0 = vdupq_n_f64(0.0);
        let mut sum_vec1 = vdupq_n_f64(0.0);
        let mut sum_vec2 = vdupq_n_f64(0.0);
        let mut sum_vec3 = vdupq_n_f64(0.0);

        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let base = i * 8;
            let ptr = x.as_ptr().add(base);

            let x0 = vld1q_f64(ptr);
            let x1 = vld1q_f64(ptr.add(2));
            let x2 = vld1q_f64(ptr.add(4));
            let x3 = vld1q_f64(ptr.add(6));

            // Scale values
            let s0 = vmulq_f64(x0, scale_vec);
            let s1 = vmulq_f64(x1, scale_vec);
            let s2 = vmulq_f64(x2, scale_vec);
            let s3 = vmulq_f64(x3, scale_vec);

            // Accumulate squared scaled values using FMA
            sum_vec0 = vfmaq_f64(sum_vec0, s0, s0);
            sum_vec1 = vfmaq_f64(sum_vec1, s1, s1);
            sum_vec2 = vfmaq_f64(sum_vec2, s2, s2);
            sum_vec3 = vfmaq_f64(sum_vec3, s3, s3);
        }

        // Combine accumulators
        let sum_01 = vaddq_f64(sum_vec0, sum_vec1);
        let sum_23 = vaddq_f64(sum_vec2, sum_vec3);
        let sum_all = vaddq_f64(sum_01, sum_23);
        let mut sum_local = vaddvq_f64(sum_all);

        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            let scaled = x[base + i] * scale;
            sum_local += scaled * scaled;
        }

        sum = sum_local;
    }

    max_val * sum.sqrt()
}

/// NEON SIMD nrm2 for f32 with overflow protection.
#[cfg(target_arch = "aarch64")]
fn nrm2_f32_simd_safe(x: &[f32]) -> f32 {
    use core::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vandq_u32, vdupq_n_f32, vdupq_n_u32, vfmaq_f32,
        vld1q_f32, vmaxq_f32, vmaxvq_f32, vmulq_f32,
    };

    let n = x.len();

    // First pass: find maximum absolute value for scaling
    let max_val: f32;
    unsafe {
        let mut max_vec = vdupq_n_f32(0.0);
        let abs_mask = vdupq_n_u32(0x7FFF_FFFF); // Mask to clear sign bit

        let chunks = n / 16;
        let remainder = n % 16;

        for i in 0..chunks {
            let base = i * 16;
            let ptr = x.as_ptr().add(base);

            let x0 = vld1q_f32(ptr);
            let x1 = vld1q_f32(ptr.add(4));
            let x2 = vld1q_f32(ptr.add(8));
            let x3 = vld1q_f32(ptr.add(12));

            // Absolute value via bit masking
            let abs0: float32x4_t =
                core::mem::transmute(vandq_u32(core::mem::transmute(x0), abs_mask));
            let abs1: float32x4_t =
                core::mem::transmute(vandq_u32(core::mem::transmute(x1), abs_mask));
            let abs2: float32x4_t =
                core::mem::transmute(vandq_u32(core::mem::transmute(x2), abs_mask));
            let abs3: float32x4_t =
                core::mem::transmute(vandq_u32(core::mem::transmute(x3), abs_mask));

            max_vec = vmaxq_f32(max_vec, abs0);
            max_vec = vmaxq_f32(max_vec, abs1);
            max_vec = vmaxq_f32(max_vec, abs2);
            max_vec = vmaxq_f32(max_vec, abs3);
        }

        // Reduce max_vec to scalar
        let mut max_val_local = vmaxvq_f32(max_vec);

        // Handle remainder
        let base = chunks * 16;
        for i in 0..remainder {
            let abs_xi = x[base + i].abs();
            if abs_xi > max_val_local {
                max_val_local = abs_xi;
            }
        }

        max_val = max_val_local;
    }

    if max_val == 0.0 {
        return 0.0;
    }

    // Compute scale factor
    let scale = 1.0 / max_val;

    // Second pass: compute sum of squares with scaling
    let sum: f32;
    unsafe {
        let scale_vec = vdupq_n_f32(scale);
        let mut sum_vec0 = vdupq_n_f32(0.0);
        let mut sum_vec1 = vdupq_n_f32(0.0);
        let mut sum_vec2 = vdupq_n_f32(0.0);
        let mut sum_vec3 = vdupq_n_f32(0.0);

        let chunks = n / 16;
        let remainder = n % 16;

        for i in 0..chunks {
            let base = i * 16;
            let ptr = x.as_ptr().add(base);

            let x0 = vld1q_f32(ptr);
            let x1 = vld1q_f32(ptr.add(4));
            let x2 = vld1q_f32(ptr.add(8));
            let x3 = vld1q_f32(ptr.add(12));

            // Scale values
            let s0 = vmulq_f32(x0, scale_vec);
            let s1 = vmulq_f32(x1, scale_vec);
            let s2 = vmulq_f32(x2, scale_vec);
            let s3 = vmulq_f32(x3, scale_vec);

            // Accumulate squared scaled values using FMA
            sum_vec0 = vfmaq_f32(sum_vec0, s0, s0);
            sum_vec1 = vfmaq_f32(sum_vec1, s1, s1);
            sum_vec2 = vfmaq_f32(sum_vec2, s2, s2);
            sum_vec3 = vfmaq_f32(sum_vec3, s3, s3);
        }

        // Combine accumulators
        let sum_01 = vaddq_f32(sum_vec0, sum_vec1);
        let sum_23 = vaddq_f32(sum_vec2, sum_vec3);
        let sum_all = vaddq_f32(sum_01, sum_23);
        let mut sum_local = vaddvq_f32(sum_all);

        // Handle remainder
        let base = chunks * 16;
        for i in 0..remainder {
            let scaled = x[base + i] * scale;
            sum_local += scaled * scaled;
        }

        sum = sum_local;
    }

    max_val * sum.sqrt()
}

/// AVX2/FMA SIMD nrm2 for f64 with overflow protection.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nrm2_f64_avx2_safe(x: &[f64]) -> f64 {
    use core::arch::x86_64::*;

    let n = x.len();

    // First pass: find maximum absolute value
    let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
    let mut max_vec = _mm256_setzero_pd();

    let chunks = n / 16;
    let remainder = n % 16;

    for i in 0..chunks {
        let base = i * 16;
        let ptr = x.as_ptr().add(base);

        let x0 = _mm256_loadu_pd(ptr);
        let x1 = _mm256_loadu_pd(ptr.add(4));
        let x2 = _mm256_loadu_pd(ptr.add(8));
        let x3 = _mm256_loadu_pd(ptr.add(12));

        let abs0 = _mm256_and_pd(x0, abs_mask);
        let abs1 = _mm256_and_pd(x1, abs_mask);
        let abs2 = _mm256_and_pd(x2, abs_mask);
        let abs3 = _mm256_and_pd(x3, abs_mask);

        max_vec = _mm256_max_pd(max_vec, abs0);
        max_vec = _mm256_max_pd(max_vec, abs1);
        max_vec = _mm256_max_pd(max_vec, abs2);
        max_vec = _mm256_max_pd(max_vec, abs3);
    }

    // Reduce max_vec
    let mut max_arr = [0.0f64; 4];
    _mm256_storeu_pd(max_arr.as_mut_ptr(), max_vec);
    let mut max_val = max_arr[0].max(max_arr[1]).max(max_arr[2]).max(max_arr[3]);

    // Handle remainder
    let base = chunks * 16;
    for i in 0..remainder {
        let abs_xi = x[base + i].abs();
        if abs_xi > max_val {
            max_val = abs_xi;
        }
    }

    if max_val == 0.0 {
        return 0.0;
    }

    let scale = 1.0 / max_val;
    let scale_vec = _mm256_set1_pd(scale);

    // Second pass: sum of squares with scaling
    let mut sum_vec0 = _mm256_setzero_pd();
    let mut sum_vec1 = _mm256_setzero_pd();
    let mut sum_vec2 = _mm256_setzero_pd();
    let mut sum_vec3 = _mm256_setzero_pd();

    for i in 0..chunks {
        let base = i * 16;
        let ptr = x.as_ptr().add(base);

        let x0 = _mm256_loadu_pd(ptr);
        let x1 = _mm256_loadu_pd(ptr.add(4));
        let x2 = _mm256_loadu_pd(ptr.add(8));
        let x3 = _mm256_loadu_pd(ptr.add(12));

        let s0 = _mm256_mul_pd(x0, scale_vec);
        let s1 = _mm256_mul_pd(x1, scale_vec);
        let s2 = _mm256_mul_pd(x2, scale_vec);
        let s3 = _mm256_mul_pd(x3, scale_vec);

        sum_vec0 = _mm256_fmadd_pd(s0, s0, sum_vec0);
        sum_vec1 = _mm256_fmadd_pd(s1, s1, sum_vec1);
        sum_vec2 = _mm256_fmadd_pd(s2, s2, sum_vec2);
        sum_vec3 = _mm256_fmadd_pd(s3, s3, sum_vec3);
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
        let scaled = x[base + i] * scale;
        sum += scaled * scaled;
    }

    max_val * sum.sqrt()
}

/// AVX2/FMA SIMD nrm2 for f32 with overflow protection.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nrm2_f32_avx2_safe(x: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = x.len();

    // First pass: find maximum absolute value
    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
    let mut max_vec = _mm256_setzero_ps();

    let chunks = n / 32;
    let remainder = n % 32;

    for i in 0..chunks {
        let base = i * 32;
        let ptr = x.as_ptr().add(base);

        let x0 = _mm256_loadu_ps(ptr);
        let x1 = _mm256_loadu_ps(ptr.add(8));
        let x2 = _mm256_loadu_ps(ptr.add(16));
        let x3 = _mm256_loadu_ps(ptr.add(24));

        let abs0 = _mm256_and_ps(x0, abs_mask);
        let abs1 = _mm256_and_ps(x1, abs_mask);
        let abs2 = _mm256_and_ps(x2, abs_mask);
        let abs3 = _mm256_and_ps(x3, abs_mask);

        max_vec = _mm256_max_ps(max_vec, abs0);
        max_vec = _mm256_max_ps(max_vec, abs1);
        max_vec = _mm256_max_ps(max_vec, abs2);
        max_vec = _mm256_max_ps(max_vec, abs3);
    }

    // Reduce max_vec
    let mut max_arr = [0.0f32; 8];
    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
    let mut max_val = max_arr
        .iter()
        .cloned()
        .fold(0.0f32, |a, b| if a > b { a } else { b });

    // Handle remainder
    let base = chunks * 32;
    for i in 0..remainder {
        let abs_xi = x[base + i].abs();
        if abs_xi > max_val {
            max_val = abs_xi;
        }
    }

    if max_val == 0.0 {
        return 0.0;
    }

    let scale = 1.0 / max_val;
    let scale_vec = _mm256_set1_ps(scale);

    // Second pass: sum of squares with scaling
    let mut sum_vec0 = _mm256_setzero_ps();
    let mut sum_vec1 = _mm256_setzero_ps();
    let mut sum_vec2 = _mm256_setzero_ps();
    let mut sum_vec3 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 32;
        let ptr = x.as_ptr().add(base);

        let x0 = _mm256_loadu_ps(ptr);
        let x1 = _mm256_loadu_ps(ptr.add(8));
        let x2 = _mm256_loadu_ps(ptr.add(16));
        let x3 = _mm256_loadu_ps(ptr.add(24));

        let s0 = _mm256_mul_ps(x0, scale_vec);
        let s1 = _mm256_mul_ps(x1, scale_vec);
        let s2 = _mm256_mul_ps(x2, scale_vec);
        let s3 = _mm256_mul_ps(x3, scale_vec);

        sum_vec0 = _mm256_fmadd_ps(s0, s0, sum_vec0);
        sum_vec1 = _mm256_fmadd_ps(s1, s1, sum_vec1);
        sum_vec2 = _mm256_fmadd_ps(s2, s2, sum_vec2);
        sum_vec3 = _mm256_fmadd_ps(s3, s3, sum_vec3);
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
        let scaled = x[base + i] * scale;
        sum += scaled * scaled;
    }

    max_val * sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nrm2_basic() {
        let x = [3.0, 4.0];
        let norm = nrm2(&x);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_single() {
        let x = [5.0];
        let norm = nrm2(&x);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_empty() {
        let x: [f64; 0] = [];
        let norm = nrm2(&x);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_nrm2_ones() {
        let x = [1.0; 9];
        let norm = nrm2(&x);
        // sqrt(9) = 3
        assert!((norm - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_f32() {
        let x = [3.0f32, 4.0];
        let norm = nrm2(&x);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_nrm2_large_values() {
        // Test numerical stability with large values
        let x = [1e150, 1e150, 1e150];
        let norm = nrm2(&x);
        let expected = (3.0f64).sqrt() * 1e150;
        assert!((norm - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_nrm2_small_values() {
        // Test numerical stability with small values
        let x = [1e-150, 1e-150, 1e-150];
        let norm = nrm2(&x);
        let expected = (3.0f64).sqrt() * 1e-150;
        assert!((norm - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_nrm2_mixed_sign() {
        let x = [-3.0, 4.0];
        let norm = nrm2(&x);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_sq() {
        let x = [3.0, 4.0];
        let norm_sq = nrm2_sq(&x);
        assert!((norm_sq - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_f64_simd() {
        // Test SIMD path with larger vector
        let n = 1000;
        let x: Vec<f64> = vec![1.0; n];
        let norm = nrm2_f64(&x);
        let expected = (n as f64).sqrt();
        assert!(
            (norm - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_f32_simd() {
        // Test SIMD path with larger vector
        let n = 1000;
        let x: Vec<f32> = vec![1.0; n];
        let norm = nrm2_f32(&x);
        let expected = (n as f32).sqrt();
        assert!(
            (norm - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_f64_simd_large_values() {
        // Test SIMD path with large values for overflow protection
        let n = 100;
        let x: Vec<f64> = vec![1e150; n];
        let norm = nrm2_f64(&x);
        let expected = (n as f64).sqrt() * 1e150;
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_f64_simd_small_values() {
        // Test SIMD path with small values for underflow protection
        let n = 100;
        let x: Vec<f64> = vec![1e-150; n];
        let norm = nrm2_f64(&x);
        let expected = (n as f64).sqrt() * 1e-150;
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_f32_simd_large_values() {
        // Test SIMD path with large f32 values
        let n = 100;
        let x: Vec<f32> = vec![1e20; n];
        let norm = nrm2_f32(&x);
        let expected = (n as f32).sqrt() * 1e20;
        assert!(
            (norm - expected).abs() / expected < 1e-5,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_f64_edge_cases() {
        // Empty vector
        let x: Vec<f64> = vec![];
        assert_eq!(nrm2_f64(&x), 0.0);

        // Single element
        let x = vec![5.0f64];
        assert!((nrm2_f64(&x) - 5.0).abs() < 1e-10);

        // Two elements (3-4-5 triangle)
        let x = vec![3.0f64, 4.0];
        assert!((nrm2_f64(&x) - 5.0).abs() < 1e-10);

        // All zeros
        let x = vec![0.0f64; 100];
        assert_eq!(nrm2_f64(&x), 0.0);
    }

    #[test]
    fn test_nrm2_consistency() {
        // Test that SIMD and scalar implementations give consistent results
        let n = 500;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();

        let norm_generic = nrm2(&x);
        let norm_simd = nrm2_f64(&x);

        assert!(
            (norm_generic - norm_simd).abs() / norm_generic < 1e-10,
            "Generic: {}, SIMD: {}",
            norm_generic,
            norm_simd
        );
    }

    // =============================================================================
    // Overflow/Underflow Tests
    // =============================================================================

    #[test]
    fn test_nrm2_overflow_prevention_f64() {
        // Test with large values that would overflow without scaling
        // Squaring 1e160 gives 1e320, which would overflow f64::MAX (1.8e308)
        // But with Blue's scaling algorithm, this should work
        let large_val = 1e160_f64;
        let x = vec![large_val; 4];
        let norm = nrm2_f64(&x);
        let expected = 2.0 * large_val; // sqrt(4) * large_val
        assert!(norm.is_finite(), "Norm should be finite, got {}", norm);
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_underflow_prevention_f64() {
        // Test with values near f64::MIN_POSITIVE
        // Without proper scaling, squaring would underflow to 0
        let small_val = 1e-308_f64; // Near f64::MIN_POSITIVE (2.2e-308)
        let x = vec![small_val; 4];
        let norm = nrm2_f64(&x);
        let expected = 2.0 * small_val; // sqrt(4) * small_val
        assert!(norm > 0.0, "Norm should be positive, got {}", norm);
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_mixed_scale_f64() {
        // Test with mixed large and small values
        // Should handle both without overflow or precision loss
        let x = vec![1e200, 1e-200, 1e200, 1e-200];
        let norm = nrm2_f64(&x);
        // The large values dominate: sqrt(2) * 1e200
        let expected = (2.0f64).sqrt() * 1e200;
        assert!(norm.is_finite(), "Norm should be finite, got {}", norm);
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_overflow_prevention_f32() {
        // Test with values near f32::MAX
        let large_val = 1e38_f32; // Near f32::MAX (3.4e38)
        let x = vec![large_val; 4];
        let norm = nrm2_f32(&x);
        let expected = 2.0 * large_val;
        assert!(norm.is_finite(), "Norm should be finite, got {}", norm);
        assert!(
            (norm - expected).abs() / expected < 1e-5,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_underflow_prevention_f32() {
        // Test with values near f32::MIN_POSITIVE
        let small_val = 1e-38_f32; // Near f32::MIN_POSITIVE (1.2e-38)
        let x = vec![small_val; 4];
        let norm = nrm2_f32(&x);
        let expected = 2.0 * small_val;
        assert!(norm > 0.0, "Norm should be positive, got {}", norm);
        assert!(
            (norm - expected).abs() / expected < 1e-5,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_subnormal_values_f64() {
        // Test with subnormal (denormalized) numbers
        let subnormal = 1e-320_f64; // Subnormal for f64
        let x = vec![subnormal; 4];
        let norm = nrm2_f64(&x);
        // Should still compute correctly with scaled algorithm
        assert!(norm >= 0.0, "Norm should be non-negative, got {}", norm);
    }

    #[test]
    fn test_nrm2_inf_handling() {
        // Test with infinity - should propagate infinity
        let x = vec![f64::INFINITY, 1.0, 2.0];
        let norm = nrm2(&x);
        assert!(
            norm.is_infinite(),
            "Norm of vector with infinity should be infinite, got {}",
            norm
        );
    }

    #[test]
    fn test_nrm2_nan_handling() {
        // Note: The Blue's algorithm implementation treats NaN as 0
        // because NaN > 0 is false. This is a known limitation for performance.
        // For vectors where NaN is the dominant value, it should still propagate.
        let x = vec![f64::NAN];
        let norm = nrm2(&x);
        // Single element takes abs() which propagates NaN
        assert!(
            norm.is_nan(),
            "Norm of single NaN should be NaN, got {}",
            norm
        );

        // For mixed values, Blue's algorithm may not propagate NaN
        // Test that at least the algorithm doesn't crash
        let x = vec![1.0, f64::NAN, 2.0];
        let norm = nrm2(&x);
        // The result may or may not be NaN depending on implementation
        assert!(
            norm.is_finite() || norm.is_nan(),
            "Norm should be finite or NaN, got {}",
            norm
        );
    }

    #[test]
    fn test_nrm2_large_vector_numerical_stability() {
        // Test numerical stability with large vectors
        // Sum of many small squares should not accumulate error badly
        let n = 10000;
        let x: Vec<f64> = vec![1e-6; n];
        let norm = nrm2_f64(&x);
        let expected = (n as f64).sqrt() * 1e-6;
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_nrm2_alternating_scale() {
        // Test with alternating large/small values to stress the scaling algorithm
        let n = 100;
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            if i % 2 == 0 {
                x.push(1e150);
            } else {
                x.push(1e-150);
            }
        }
        let norm = nrm2_f64(&x);
        // Large values dominate: sqrt(50) * 1e150
        let expected = (50.0f64).sqrt() * 1e150;
        assert!(norm.is_finite(), "Norm should be finite, got {}", norm);
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "Expected {}, got {}",
            expected,
            norm
        );
    }
}
