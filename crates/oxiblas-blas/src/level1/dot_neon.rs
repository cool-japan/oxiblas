//! aarch64 NEON dot-product kernels for `f64` and `f32`.
//!
//! Both kernels keep eight independent vector accumulators (16 `f64` / 32 `f32`
//! lanes) fed by fused multiply-adds, which covers the FMA latency of the
//! Apple/Neoverse cores and leaves the loop bound by the two loads per FMA.
//!
//! # Two-level summation
//!
//! The lane accumulators are flushed into eight outer accumulators every
//! [`F64_BLOCK`] / [`F32_BLOCK`] elements (64 loop iterations). That keeps the
//! longest chain of dependent roundings at `k = 64 + ceil(n / BLOCK) + 8`
//! (block FMAs, outer additions, then at most one tail FMA per lane, the
//! depth-3 tree, the lane sum and the scalar tail) instead of about
//! `n / LANES + 8`, and the worst-case error bound
//! `|fl(x·y) − x·y| ≤ γ_k · Σ|x_i·y_i|` scales accordingly: for `n = 10⁶` f64
//! elements `k` is about 1100 instead of about 62 500. The flush costs eight
//! vector additions per block; against a single-level variant it measured
//! within ±2% (no consistent sign) at 10³–10⁶ elements on an Apple M3.
//!
//! # Tail
//!
//! The `n mod 16` (`n mod 32`) trailing elements are folded into the outer
//! accumulators with progressively narrower vector steps and at most one
//! (three) scalar FMA, all through raw-pointer reads inside the checked length,
//! so no bounds checks are emitted anywhere.

use core::arch::aarch64::{
    float32x4_t, float64x2_t, vaddq_f32, vaddq_f64, vaddvq_f32, vaddvq_f64, vdupq_n_f32,
    vdupq_n_f64, vfmaq_f32, vfmaq_f64, vld1q_f32, vld1q_f64,
};

/// Elements per inner block of the f64 kernel (64 iterations of 16).
const F64_BLOCK: usize = 1024;

/// Elements per inner block of the f32 kernel (64 iterations of 32).
const F32_BLOCK: usize = 2048;

/// `Σ x[i]·y[i]` over the common prefix of `x` and `y` (callers pass equal
/// lengths).
#[inline]
pub(crate) fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    // SAFETY: the kernel reads `x[..n]` and `y[..n]` only, and both slices hold
    // at least `n` elements.
    unsafe { dot_f64_kernel(x.as_ptr(), y.as_ptr(), n) }
}

/// `Σ x[i]·y[i]` over the common prefix of `x` and `y` (callers pass equal
/// lengths).
#[inline]
pub(crate) fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    // SAFETY: the kernel reads `x[..n]` and `y[..n]` only, and both slices hold
    // at least `n` elements.
    unsafe { dot_f32_kernel(x.as_ptr(), y.as_ptr(), n) }
}

/// Pairwise tree over eight f64 vector accumulators, then the lane sum.
#[inline(always)]
unsafe fn reduce8_f64(a: [float64x2_t; 8]) -> f64 {
    let s01 = vaddq_f64(a[0], a[1]);
    let s23 = vaddq_f64(a[2], a[3]);
    let s45 = vaddq_f64(a[4], a[5]);
    let s67 = vaddq_f64(a[6], a[7]);
    vaddvq_f64(vaddq_f64(vaddq_f64(s01, s23), vaddq_f64(s45, s67)))
}

/// Pairwise tree over eight f32 vector accumulators, then the lane sum.
#[inline(always)]
unsafe fn reduce8_f32(a: [float32x4_t; 8]) -> f32 {
    let s01 = vaddq_f32(a[0], a[1]);
    let s23 = vaddq_f32(a[2], a[3]);
    let s45 = vaddq_f32(a[4], a[5]);
    let s67 = vaddq_f32(a[6], a[7]);
    vaddvq_f32(vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67)))
}

/// f64 dot-product kernel.
///
/// # Safety
///
/// `x` and `y` must be valid for reads of `n` consecutive `f64` values.
#[inline]
unsafe fn dot_f64_kernel(x: *const f64, y: *const f64, n: usize) -> f64 {
    let zero = vdupq_n_f64(0.0);
    let mut outer = [zero; 8];
    let body = n - n % 16;
    let mut i = 0;
    while i < body {
        let block_end = if body - i > F64_BLOCK {
            i + F64_BLOCK
        } else {
            body
        };
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let (mut a4, mut a5, mut a6, mut a7) = (zero, zero, zero, zero);
        while i < block_end {
            let xp = x.add(i);
            let yp = y.add(i);
            a0 = vfmaq_f64(a0, vld1q_f64(xp), vld1q_f64(yp));
            a1 = vfmaq_f64(a1, vld1q_f64(xp.add(2)), vld1q_f64(yp.add(2)));
            a2 = vfmaq_f64(a2, vld1q_f64(xp.add(4)), vld1q_f64(yp.add(4)));
            a3 = vfmaq_f64(a3, vld1q_f64(xp.add(6)), vld1q_f64(yp.add(6)));
            a4 = vfmaq_f64(a4, vld1q_f64(xp.add(8)), vld1q_f64(yp.add(8)));
            a5 = vfmaq_f64(a5, vld1q_f64(xp.add(10)), vld1q_f64(yp.add(10)));
            a6 = vfmaq_f64(a6, vld1q_f64(xp.add(12)), vld1q_f64(yp.add(12)));
            a7 = vfmaq_f64(a7, vld1q_f64(xp.add(14)), vld1q_f64(yp.add(14)));
            i += 16;
        }
        outer[0] = vaddq_f64(outer[0], a0);
        outer[1] = vaddq_f64(outer[1], a1);
        outer[2] = vaddq_f64(outer[2], a2);
        outer[3] = vaddq_f64(outer[3], a3);
        outer[4] = vaddq_f64(outer[4], a4);
        outer[5] = vaddq_f64(outer[5], a5);
        outer[6] = vaddq_f64(outer[6], a6);
        outer[7] = vaddq_f64(outer[7], a7);
    }

    // Tail: 0..=15 elements left at `i`.
    if n - i >= 8 {
        let (xp, yp) = (x.add(i), y.add(i));
        outer[0] = vfmaq_f64(outer[0], vld1q_f64(xp), vld1q_f64(yp));
        outer[1] = vfmaq_f64(outer[1], vld1q_f64(xp.add(2)), vld1q_f64(yp.add(2)));
        outer[2] = vfmaq_f64(outer[2], vld1q_f64(xp.add(4)), vld1q_f64(yp.add(4)));
        outer[3] = vfmaq_f64(outer[3], vld1q_f64(xp.add(6)), vld1q_f64(yp.add(6)));
        i += 8;
    }
    if n - i >= 4 {
        let (xp, yp) = (x.add(i), y.add(i));
        outer[4] = vfmaq_f64(outer[4], vld1q_f64(xp), vld1q_f64(yp));
        outer[5] = vfmaq_f64(outer[5], vld1q_f64(xp.add(2)), vld1q_f64(yp.add(2)));
        i += 4;
    }
    if n - i >= 2 {
        outer[6] = vfmaq_f64(outer[6], vld1q_f64(x.add(i)), vld1q_f64(y.add(i)));
        i += 2;
    }
    let mut sum = reduce8_f64(outer);
    if i < n {
        sum = (*x.add(i)).mul_add(*y.add(i), sum);
    }
    sum
}

/// f32 dot-product kernel.
///
/// # Safety
///
/// `x` and `y` must be valid for reads of `n` consecutive `f32` values.
#[inline]
unsafe fn dot_f32_kernel(x: *const f32, y: *const f32, n: usize) -> f32 {
    let zero = vdupq_n_f32(0.0);
    let mut outer = [zero; 8];
    let body = n - n % 32;
    let mut i = 0;
    while i < body {
        let block_end = if body - i > F32_BLOCK {
            i + F32_BLOCK
        } else {
            body
        };
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let (mut a4, mut a5, mut a6, mut a7) = (zero, zero, zero, zero);
        while i < block_end {
            let xp = x.add(i);
            let yp = y.add(i);
            a0 = vfmaq_f32(a0, vld1q_f32(xp), vld1q_f32(yp));
            a1 = vfmaq_f32(a1, vld1q_f32(xp.add(4)), vld1q_f32(yp.add(4)));
            a2 = vfmaq_f32(a2, vld1q_f32(xp.add(8)), vld1q_f32(yp.add(8)));
            a3 = vfmaq_f32(a3, vld1q_f32(xp.add(12)), vld1q_f32(yp.add(12)));
            a4 = vfmaq_f32(a4, vld1q_f32(xp.add(16)), vld1q_f32(yp.add(16)));
            a5 = vfmaq_f32(a5, vld1q_f32(xp.add(20)), vld1q_f32(yp.add(20)));
            a6 = vfmaq_f32(a6, vld1q_f32(xp.add(24)), vld1q_f32(yp.add(24)));
            a7 = vfmaq_f32(a7, vld1q_f32(xp.add(28)), vld1q_f32(yp.add(28)));
            i += 32;
        }
        outer[0] = vaddq_f32(outer[0], a0);
        outer[1] = vaddq_f32(outer[1], a1);
        outer[2] = vaddq_f32(outer[2], a2);
        outer[3] = vaddq_f32(outer[3], a3);
        outer[4] = vaddq_f32(outer[4], a4);
        outer[5] = vaddq_f32(outer[5], a5);
        outer[6] = vaddq_f32(outer[6], a6);
        outer[7] = vaddq_f32(outer[7], a7);
    }

    // Tail: 0..=31 elements left at `i`.
    if n - i >= 16 {
        let (xp, yp) = (x.add(i), y.add(i));
        outer[0] = vfmaq_f32(outer[0], vld1q_f32(xp), vld1q_f32(yp));
        outer[1] = vfmaq_f32(outer[1], vld1q_f32(xp.add(4)), vld1q_f32(yp.add(4)));
        outer[2] = vfmaq_f32(outer[2], vld1q_f32(xp.add(8)), vld1q_f32(yp.add(8)));
        outer[3] = vfmaq_f32(outer[3], vld1q_f32(xp.add(12)), vld1q_f32(yp.add(12)));
        i += 16;
    }
    if n - i >= 8 {
        let (xp, yp) = (x.add(i), y.add(i));
        outer[4] = vfmaq_f32(outer[4], vld1q_f32(xp), vld1q_f32(yp));
        outer[5] = vfmaq_f32(outer[5], vld1q_f32(xp.add(4)), vld1q_f32(yp.add(4)));
        i += 8;
    }
    if n - i >= 4 {
        outer[6] = vfmaq_f32(outer[6], vld1q_f32(x.add(i)), vld1q_f32(y.add(i)));
        i += 4;
    }
    let mut sum = reduce8_f32(outer);
    while i < n {
        sum = (*x.add(i)).mul_add(*y.add(i), sum);
        i += 1;
    }
    sum
}

#[cfg(test)]
#[path = "dot_neon_tests.rs"]
mod tests;
