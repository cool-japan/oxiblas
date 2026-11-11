//! NRM2: Euclidean (L2) norm
//!
//! Computes ||x||_2 = sqrt(Σ |x\[i\]|^2)
//!
//! Uses a numerically stable algorithm to prevent overflow/underflow.
//! SIMD-optimized implementations available for f64 and f32.

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Real, Scalar};

/// Shortest vector handed to the NEON kernel on aarch64. Below it the dlassq
/// loop is cheaper than the kernel's setup and lane reductions (on an Apple M3
/// the kernel was up to 2.2x slower at n = 2..4 and faster from n = 8), so short
/// vectors keep the previous code and results.
#[cfg(target_arch = "aarch64")]
const BLUE_MIN_LEN: usize = 8;

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
///
/// On aarch64, `f64` and `f32` vectors of at least eight elements use the
/// one-pass NEON kernel of [`nrm2_f64`] / [`nrm2_f32`] (LAPACK 3.12 `dnrm2.f90`
/// algorithm) unless the `force-scalar` feature is enabled; shorter vectors and
/// other types use the loop below.
pub fn nrm2<T: Real>(x: &[T]) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    if n == 1 {
        // `abs` propagates NaN and maps ±Inf to +Inf, matching reference BLAS.
        return Scalar::abs(x[0]);
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    if n >= BLUE_MIN_LEN {
        use super::typed::{same_slice, same_value};
        if let Some(xs) = same_slice::<T, f64>(x)
            && let Some(r) = same_value::<f64, T>(super::nrm2_blue::nrm2_f64(xs))
        {
            return r;
        }
        if let Some(xs) = same_slice::<T, f32>(x)
            && let Some(r) = same_value::<f32, T>(super::nrm2_blue::nrm2_f32(xs))
        {
            return r;
        }
    }

    // dlassq-style scaled sum of squares (a single accumulator, not the three-bin
    // Blue scheme of `?nrm2.f90`): carry a running `(scale, ssq)` pair whose
    // invariant is `Σ|x[i]|² == scale² · ssq`. Scaling by the running maximum
    // magnitude keeps every squared term in `[0, 1]`, which prevents both
    // overflow and underflow regardless of the dynamic range of the input.
    let mut state = (T::zero(), T::one());
    for &xi in x {
        state = nrm2_fold(state, xi);
    }
    nrm2_finalize(state)
}

/// Folds one value `xi` into a running `(scale, ssq)` accumulator using the
/// `?lassq` scaling rule. Invariant: the accumulated `Σ|x|²` equals
/// `scale² · ssq`.
///
/// WHY `!= zero` and not `> zero`: NaN satisfies `NaN != 0` yet fails every
/// ordered comparison, so a `> 0` guard would silently *drop* a NaN input and
/// return a finite norm. Entering on `!= 0` routes NaN through `abs_xi / scale`
/// (= NaN), so it propagates to the result exactly as reference BLAS requires.
///
/// WHY the `scale == abs_xi` guard in the else branch: when both are `+Inf`,
/// `abs_xi / scale` would be `Inf / Inf = NaN`; two equal-magnitude infinities
/// really contribute a ratio of exactly `1`, so the guard keeps the norm `+Inf`
/// instead of spuriously turning an all-infinite input into NaN.
#[inline]
pub(crate) fn nrm2_fold<T: Real>(state: (T, T), xi: T) -> (T, T) {
    let (scale, ssq) = state;
    let abs_xi = Scalar::abs(xi);
    if abs_xi != T::zero() {
        if scale < abs_xi {
            // New magnitude dominates: rescale the accumulated sum down to it.
            // `scale < abs_xi` guarantees `scale` is finite here, so `t` is well
            // defined (it is `0` when `abs_xi` is `+Inf`).
            let t = scale / abs_xi;
            (abs_xi, T::one() + ssq * t * t)
        } else {
            // `abs_xi <= scale`, or `abs_xi` is NaN (every comparison is false).
            let t = if scale == abs_xi {
                T::one()
            } else {
                abs_xi / scale
            };
            (scale, ssq + t * t)
        }
    } else {
        state
    }
}

/// Collapses a `(scale, ssq)` accumulator into the final norm `scale · √ssq`.
#[inline]
pub(crate) fn nrm2_finalize<T: Real>(state: (T, T)) -> T {
    let (scale, ssq) = state;
    scale * Real::sqrt(ssq)
}

/// Computes the Euclidean (L2) norm of a complex vector (`SCNRM2`/`DZNRM2`).
///
/// ||x||_2 = sqrt(Σ (Re(x\[i\])² + Im(x\[i\])²))
///
/// This is the true complex 2-norm -- unlike complex ASUM/IAMAX, NRM2 does
/// not use the `cabs1` metric. As in reference `DZNRM2`, the real and
/// imaginary parts are folded into the same overflow/underflow-safe scaled
/// sum of squares as the real [`nrm2`], so entries near the overflow
/// threshold or deep in the underflow range neither overflow nor flush to
/// zero. A NaN component yields NaN; otherwise an infinite component yields
/// `+Inf`.
///
/// For strided vectors (including negative and zero increments) see
/// [`cnrm2_strided`](crate::level1::cnrm2_strided).
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::cnrm2;
/// use oxiblas_core::scalar::c64;
///
/// let x = [c64(3.0, 4.0), c64(0.0, 12.0)];
/// // sqrt(9 + 16 + 144) = 13
/// assert!((cnrm2(&x) - 13.0).abs() < 1e-12);
/// ```
///
/// On aarch64, `Complex64` / `Complex32` vectors are evaluated as their `2n`
/// interleaved components through [`nrm2`] (unless the `force-scalar` feature
/// is enabled), so `cnrm2(x)` still equals `nrm2` of the component vector bit
/// for bit.
pub fn cnrm2<T: ComplexScalar>(x: &[T]) -> T::Real {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        use super::typed::{same_slice, same_value};
        if let Some(z) = same_slice::<T, Complex64>(x)
            && let Ok(components) = bytemuck::try_cast_slice::<Complex64, f64>(z)
            && let Some(r) = same_value::<f64, T::Real>(nrm2(components))
        {
            return r;
        }
        if let Some(z) = same_slice::<T, Complex32>(x)
            && let Ok(components) = bytemuck::try_cast_slice::<Complex32, f32>(z)
            && let Some(r) = same_value::<f32, T::Real>(nrm2(components))
        {
            return r;
        }
    }

    let mut state = (T::Real::zero(), T::Real::one());
    for &z in x {
        state = nrm2_fold(nrm2_fold(state, z.real()), z.imag());
    }
    nrm2_finalize(state)
}

/// `SCNRM2`: Euclidean norm of a single-precision complex vector.
///
/// The interleaved `[re, im, re, im, …]` storage is viewed as `2n` reals and
/// evaluated by the SIMD-accelerated, overflow-safe [`nrm2_f32`]: the
/// complex 2-norm is the real 2-norm of the component vector, so the result
/// agrees with [`cnrm2`] up to rounding (like `nrm2_f32` vs. [`nrm2`]).
#[inline]
#[must_use]
pub fn scnrm2(x: &[Complex32]) -> f32 {
    match bytemuck::try_cast_slice::<Complex32, f32>(x) {
        Ok(components) => nrm2_f32(components),
        Err(_) => cnrm2(x),
    }
}

/// `DZNRM2`: Euclidean norm of a double-precision complex vector.
///
/// The interleaved `[re, im, re, im, …]` storage is viewed as `2n` reals and
/// evaluated by the SIMD-accelerated, overflow-safe [`nrm2_f64`]: the
/// complex 2-norm is the real 2-norm of the component vector, so the result
/// agrees with [`cnrm2`] up to rounding (like `nrm2_f64` vs. [`nrm2`]).
#[inline]
#[must_use]
pub fn dznrm2(x: &[Complex64]) -> f64 {
    match bytemuck::try_cast_slice::<Complex64, f64>(x) {
        Ok(components) => nrm2_f64(components),
        Err(_) => cnrm2(x),
    }
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
/// On aarch64 this is the one-pass NEON implementation of the LAPACK 3.12
/// `dnrm2.f90` algorithm (Blue's three accumulators, no division per element)
/// from eight elements on. On `x86_64` with AVX2/FMA it is a two-pass
/// max-then-scaled-sum kernel; elsewhere (and for short vectors on both)
/// a dlassq-style scaled sum of squares. All variants are overflow/underflow
/// safe, return NaN when any element is NaN and `+Inf` when an element is
/// infinite.
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

    #[cfg(target_arch = "aarch64")]
    {
        if n < BLUE_MIN_LEN {
            nrm2_f64_scaled(x)
        } else {
            super::nrm2_blue::nrm2_f64(x)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // For small vectors, use simple implementation
        if n < 64 {
            return nrm2_f64_scaled(x);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2 and FMA were detected at run time just above, and the
                // kernel reads only `x[..n]`.
                return unsafe { nrm2_f64_avx2_safe(x) };
            }
        }

        nrm2_f64_scaled(x)
    }
}

/// SIMD-optimized L2 norm for f32.
///
/// The f32 counterpart of [`nrm2_f64`] (LAPACK 3.12 `snrm2.f90` algorithm on
/// aarch64).
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

    #[cfg(target_arch = "aarch64")]
    {
        if n < BLUE_MIN_LEN {
            nrm2_f32_scaled(x)
        } else {
            super::nrm2_blue::nrm2_f32(x)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // For small vectors, use simple implementation
        if n < 64 {
            return nrm2_f32_scaled(x);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2 and FMA were detected at run time just above, and the
                // kernel reads only `x[..n]`.
                return unsafe { nrm2_f32_avx2_safe(x) };
            }
        }

        nrm2_f32_scaled(x)
    }
}

/// Scaled nrm2 for f64 (small-vector fallback): a dlassq-style scaled sum of
/// squares with a single `(scale, ssq)` accumulator.
///
/// Delegates to the shared `nrm2_fold` accumulator so the small-vector path is
/// bit-for-bit consistent with the generic scalar path — including NaN
/// propagation and `+Inf` handling.
fn nrm2_f64_scaled(x: &[f64]) -> f64 {
    let mut state = (0.0f64, 1.0f64);
    for &xi in x {
        state = nrm2_fold(state, xi);
    }
    nrm2_finalize(state)
}

/// Scaled nrm2 for f32 (small-vector fallback): a dlassq-style scaled sum of
/// squares with a single `(scale, ssq)` accumulator.
///
/// Delegates to the shared `nrm2_fold` accumulator so the small-vector path is
/// bit-for-bit consistent with the generic scalar path — including NaN
/// propagation and `+Inf` handling.
fn nrm2_f32_scaled(x: &[f32]) -> f32 {
    let mut state = (0.0f32, 1.0f32);
    for &xi in x {
        state = nrm2_fold(state, xi);
    }
    nrm2_finalize(state)
}

/// Resolves the norm when the maximum magnitude found in the first SIMD pass is
/// non-finite.
///
/// WHY this is needed: the SIMD paths scale every element by `1 / max_val`. When
/// `max_val` is `+Inf`, that factor is `0`, so `Inf · 0 == NaN` poisons the
/// second-pass sum and the function would return NaN for a purely infinite input
/// — a distinct, wrong outcome. Resolve the exceptional case directly, matching
/// IEEE / reference DNRM2: any NaN input yields NaN; otherwise an infinity is
/// present and the Euclidean norm is `+Inf`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn nrm2_nonfinite_f64(x: &[f64]) -> f64 {
    if x.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        f64::INFINITY
    }
}

/// f32 counterpart of [`nrm2_nonfinite_f64`]; see that function for rationale.
#[cfg(target_arch = "x86_64")]
#[inline]
fn nrm2_nonfinite_f32(x: &[f32]) -> f32 {
    if x.iter().any(|v| v.is_nan()) {
        f32::NAN
    } else {
        f32::INFINITY
    }
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

    // Scaling by `1/max_val` cannot represent a non-finite maximum: `1/Inf == 0`
    // turns `Inf·0` into NaN in the second pass. Resolve Inf/NaN inputs exactly.
    if !max_val.is_finite() {
        return nrm2_nonfinite_f64(x);
    }

    if max_val == 0.0 {
        return 0.0;
    }

    let scale = 1.0 / max_val;
    // `1/max_val` overflows to `+Inf` for `max_val < 2^-1024` (deep
    // subnormals): take the scaled scalar path instead.
    if !scale.is_finite() {
        return nrm2_f64_scaled(x);
    }
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

    // Scaling by `1/max_val` cannot represent a non-finite maximum: `1/Inf == 0`
    // turns `Inf·0` into NaN in the second pass. Resolve Inf/NaN inputs exactly.
    if !max_val.is_finite() {
        return nrm2_nonfinite_f32(x);
    }

    if max_val == 0.0 {
        return 0.0;
    }

    let scale = 1.0 / max_val;
    // `1/max_val` overflows to `+Inf` for `max_val < 2^-128` (deep
    // subnormals): take the scaled scalar path instead.
    if !scale.is_finite() {
        return nrm2_f32_scaled(x);
    }
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
        // But with scaled accumulation, this should work
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
        // A NaN anywhere in the input must make the whole norm NaN, matching
        // IEEE-754 propagation and reference BLAS. The accumulation enters on
        // `abs != 0` (not `> 0`), so NaN is never silently skipped.
        let x = vec![f64::NAN];
        let norm = nrm2(&x);
        assert!(
            norm.is_nan(),
            "Norm of single NaN should be NaN, got {}",
            norm
        );

        // NaN in the middle of finite values must still propagate.
        let x = vec![1.0, f64::NAN, 2.0];
        let norm = nrm2(&x);
        assert!(
            norm.is_nan(),
            "Norm of vector containing NaN should be NaN, got {}",
            norm
        );

        // NaN as the leading element (before any scale is established).
        let x = vec![f64::NAN, 3.0, 4.0];
        let norm = nrm2(&x);
        assert!(
            norm.is_nan(),
            "Norm with leading NaN should be NaN, got {}",
            norm
        );

        // NaN alongside an infinity: NaN wins over Inf.
        let x = vec![f64::INFINITY, f64::NAN];
        let norm = nrm2(&x);
        assert!(
            norm.is_nan(),
            "Norm of {{Inf, NaN}} should be NaN, got {}",
            norm
        );
    }

    #[test]
    fn test_nrm2_nan_simd_path() {
        // Drive the SIMD entry points (n >= 64) with a NaN and require NaN out.
        // Regression for the scalar-vs-SIMD inconsistency where the scalar path
        // could drop NaN via a `> 0` guard.
        let mut x = vec![1.0f64; 128];
        x[64] = f64::NAN;
        let norm = nrm2_f64(&x);
        assert!(
            norm.is_nan(),
            "SIMD f64 nrm2 must propagate NaN, got {}",
            norm
        );

        let mut x32 = vec![1.0f32; 256];
        x32[100] = f32::NAN;
        let norm32 = nrm2_f32(&x32);
        assert!(
            norm32.is_nan(),
            "SIMD f32 nrm2 must propagate NaN, got {}",
            norm32
        );

        // The small-vector fallback (n < 64) must behave identically.
        let x_small = vec![1.0f64, f64::NAN, 2.0, 3.0];
        assert!(
            nrm2_f64(&x_small).is_nan(),
            "small-vector f64 nrm2 must propagate NaN"
        );
    }

    #[test]
    fn test_nrm2_inf_yields_positive_infinity() {
        // +Inf (not NaN) is the correct Euclidean norm of a vector with an
        // infinite element. Cover scalar, SIMD f64, and SIMD f32 paths.
        let x = vec![f64::INFINITY, 1.0, 2.0];
        assert!(
            nrm2(&x).is_infinite() && nrm2(&x) > 0.0,
            "scalar nrm2 of Inf must be +Inf, got {}",
            nrm2(&x)
        );

        // SIMD f64 path (n >= 64). Before the fix this returned NaN because the
        // second pass computed Inf * (1/Inf) == Inf * 0 == NaN.
        let mut xf = vec![1.0f64; 128];
        xf[70] = f64::INFINITY;
        let norm = nrm2_f64(&xf);
        assert!(
            norm.is_infinite() && norm > 0.0,
            "SIMD f64 nrm2 of Inf must be +Inf, got {}",
            norm
        );

        // Negative infinity has the same magnitude and must also yield +Inf.
        let mut xn = vec![1.0f64; 128];
        xn[10] = f64::NEG_INFINITY;
        let norm_neg = nrm2_f64(&xn);
        assert!(
            norm_neg.is_infinite() && norm_neg > 0.0,
            "SIMD f64 nrm2 of -Inf must be +Inf, got {}",
            norm_neg
        );

        // Multiple infinities must still give +Inf, not Inf/Inf = NaN.
        let mut xm = vec![1.0f64; 128];
        xm[5] = f64::INFINITY;
        xm[90] = f64::INFINITY;
        let norm_multi = nrm2_f64(&xm);
        assert!(
            norm_multi.is_infinite() && norm_multi > 0.0,
            "SIMD f64 nrm2 of multiple Inf must be +Inf, got {}",
            norm_multi
        );

        // SIMD f32 path (n >= 64).
        let mut xf32 = vec![1.0f32; 256];
        xf32[120] = f32::INFINITY;
        let norm32 = nrm2_f32(&xf32);
        assert!(
            norm32.is_infinite() && norm32 > 0.0,
            "SIMD f32 nrm2 of Inf must be +Inf, got {}",
            norm32
        );
    }

    #[test]
    fn test_nrm2_overflow_1e200_scalar() {
        // 1e200² == 1e400 overflows f64 (max ≈ 1.8e308), so a naive sum of
        // squares would return +Inf. The scaled accumulation must return the
        // correct finite value.
        let x = vec![1e200f64, 1.0, 1.0];
        let norm = nrm2(&x);
        assert!(norm.is_finite(), "norm must be finite, got {}", norm);
        // The 1e200 term dominates; the two unit terms are below f64 precision
        // relative to it, so the norm rounds to exactly 1e200.
        assert!(
            (norm - 1e200).abs() / 1e200 < 1e-10,
            "expected ≈1e200, got {}",
            norm
        );

        // Three equal 1e200 values: sqrt(3) * 1e200, still overflow-prone naively.
        let x3 = vec![1e200f64; 3];
        let norm3 = nrm2(&x3);
        let expected = (3.0f64).sqrt() * 1e200;
        assert!(norm3.is_finite(), "norm must be finite, got {}", norm3);
        assert!(
            (norm3 - expected).abs() / expected < 1e-10,
            "expected {}, got {}",
            expected,
            norm3
        );
    }

    #[test]
    fn test_nrm2_overflow_1e200_simd() {
        // Same overflow guard, but forced through the SIMD entry point (n >= 64).
        let mut x = vec![0.0f64; 128];
        x[0] = 1e200;
        x[64] = 1e200;
        let norm = nrm2_f64(&x);
        let expected = (2.0f64).sqrt() * 1e200;
        assert!(norm.is_finite(), "SIMD norm must be finite, got {}", norm);
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "expected {}, got {}",
            expected,
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

    // =========================================================================
    // Issue #15: native complex NRM2 (cnrm2 / scnrm2 / dznrm2)
    // =========================================================================

    #[test]
    fn test_issue_15_complex_nrm2_basic_and_empty() {
        // sqrt(9 + 16 + 144) = 13 for all three entry points.
        let x64 = [Complex64::new(3.0, 4.0), Complex64::new(0.0, 12.0)];
        let x32 = [Complex32::new(3.0, 4.0), Complex32::new(0.0, -12.0)];
        assert!((cnrm2(&x64) - 13.0).abs() < 1e-14);
        assert!((dznrm2(&x64) - 13.0).abs() < 1e-14);
        assert!((cnrm2(&x32) - 13.0).abs() < 1e-5);
        assert!((scnrm2(&x32) - 13.0).abs() < 1e-5);
        let empty64: [Complex64; 0] = [];
        let empty32: [Complex32; 0] = [];
        assert_eq!((cnrm2(&empty64), dznrm2(&empty64)), (0.0, 0.0));
        assert_eq!((cnrm2(&empty32), scnrm2(&empty32)), (0.0, 0.0));
    }

    #[test]
    fn test_issue_15_cnrm2_is_real_nrm2_of_components() {
        // The complex 2-norm is the real 2-norm of [re0, im0, re1, im1, ...];
        // `cnrm2` folds exactly that sequence, so it matches `nrm2` bit-for-bit.
        let x: Vec<Complex64> = (0..37)
            .map(|k| Complex64::new((k as f64 * 0.7).sin() * 1e3, (k as f64).cos() * 1e-3))
            .collect();
        let components: Vec<f64> = x.iter().flat_map(|z| [z.re, z.im]).collect();
        assert_eq!(cnrm2(&x), nrm2(&components));
        // The typed SIMD wrapper agrees up to rounding (2n >= 64 reals here).
        let typed = dznrm2(&x);
        assert!((typed - cnrm2(&x)).abs() <= 1e-14 * typed, "typed {typed}");
    }

    #[test]
    fn test_issue_15_complex_nrm2_no_overflow_or_underflow() {
        // Squares overflow / underflow; the scaled accumulation must not.
        let big = [Complex64::new(1e300, -1e300); 3];
        let expected = 6.0f64.sqrt() * 1e300;
        for got in [cnrm2(&big), dznrm2(&big)] {
            assert!((got - expected).abs() <= 1e-14 * expected, "got {got}");
        }
        let tiny = [Complex64::new(3e-300, 4e-300); 4];
        for got in [cnrm2(&tiny), dznrm2(&tiny)] {
            assert!((got - 1e-299).abs() <= 1e-14 * 1e-299, "got {got}");
        }
        let big32 = [Complex32::new(1e30, 1e30); 3];
        let expected32 = 6.0f32.sqrt() * 1e30;
        for got in [cnrm2(&big32), scnrm2(&big32)] {
            assert!((got - expected32).abs() <= 1e-6 * expected32, "got {got}");
        }
        // Long vector: exercises the SIMD two-pass path of the typed wrapper.
        let long = vec![Complex64::new(1e200, 1e200); 100];
        let expected = (200.0f64).sqrt() * 1e200;
        for got in [cnrm2(&long), dznrm2(&long)] {
            assert!((got - expected).abs() <= 1e-13 * expected, "got {got}");
        }
    }

    #[test]
    fn test_issue_15_complex_nrm2_nonfinite() {
        let inf = [
            Complex64::new(f64::INFINITY, f64::NEG_INFINITY),
            Complex64::new(1.0, 1.0),
        ];
        for got in [cnrm2(&inf), dznrm2(&inf)] {
            assert!(got.is_infinite() && got > 0.0, "got {got}");
        }
        let nan = [
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(1.0, f64::NAN),
        ];
        assert!(cnrm2(&nan).is_nan());
        assert!(dznrm2(&nan).is_nan());
        // Same on the long (SIMD) path of the typed wrapper.
        let mut long = vec![Complex32::new(1.0, -1.0); 80];
        long[40] = Complex32::new(f32::INFINITY, 0.0);
        assert!(scnrm2(&long).is_infinite() && scnrm2(&long) > 0.0);
        long[70] = Complex32::new(0.0, f32::NAN);
        assert!(scnrm2(&long).is_nan());
        assert!(cnrm2(&long).is_nan());
    }
}

#[cfg(test)]
#[path = "nrm2_tests.rs"]
mod subnormal_scale_tests;
