//! One-pass, division-free Euclidean norm (LAPACK 3.12 `dnrm2.f90` / `snrm2.f90`,
//! Anderson's safe-scaling algorithm with Blue's constants) as aarch64 NEON
//! kernels for `f64` and `f32`.
//!
//! # Algorithm
//!
//! Every element is classified by its magnitude `|x|` against Blue's thresholds
//! and squared into one of three accumulators:
//!
//! - `|x| > tbig`: `abig += (|x|·sbig)²` (scaled down, cannot overflow);
//! - `|x| < tsml`: `asml += (|x|·ssml)²` (scaled up, cannot underflow);
//! - otherwise (including NaN): `amed += |x|²` (no scaling needed).
//!
//! The thresholds and scale factors are exact powers of the radix derived from
//! the type's radix, digits and exponent range exactly as in the reference, so
//! each scaled square stays in the normal range. The reference stops feeding
//! `asml` after the first big value (`notbig`); the kernels always accumulate
//! it and [`blue_combine_f64`] discards it when `abig > 0`, which is the same
//! result. The combine step is a line-by-line port of the reference.
//!
//! The NEON body classifies two (`f64`) or four (`f32`) lanes at once with
//! absolute compares (`FACGT`), masks each scaled value into its accumulator
//! with bitwise selects and squares it with an FMA: no division and no
//! data-dependent branch per element. The `amed` square is fused
//! (`fma(|x|, |x|, amed)`), so results can differ from a non-fused build of the
//! reference in the last bits. The last few elements go through the same
//! classification in scalar code. The public entry points send vectors of fewer
//! than eight elements to the dlassq loop instead (see `level1::nrm2`).
//!
//! # Fast path
//!
//! Almost all data is mid-range, where the classification is a no-op. Each
//! block of 64 (`f64`) / 128 (`f32`) elements is therefore first summed as plain
//! fused squares while tracking its largest magnitude. If that maximum lies in
//! `[FAST_MIN, tbig]` (which also rejects NaN and infinities) the block's squares
//! are kept: no element overflows, and elements below `tsml` are so small next
//! to the maximum that their squares are exact to far below one ulp of the norm
//! (see [`F64_FAST_MIN`]). Otherwise the block, still in L1, is classified
//! element by element exactly as above. The vector is still read from memory
//! once.
//!
//! NaN anywhere yields NaN (it lands in `amed`, which the combine step always
//! propagates); otherwise any infinity yields `+Inf` (`abig = +Inf`).

use core::arch::aarch64::{
    float32x4_t, float64x2_t, vabsq_f32, vabsq_f64, vaddq_f32, vaddq_f64, vaddvq_f32, vaddvq_f64,
    vandq_u32, vandq_u64, vbicq_u32, vbicq_u64, vcagtq_f32, vcagtq_f64, vcaltq_f32, vcaltq_f64,
    vdupq_n_f32, vdupq_n_f64, vfmaq_f32, vfmaq_f64, vld1q_f32, vld1q_f64, vmaxq_f32, vmaxq_f64,
    vmaxvq_f32, vmaxvq_f64, vmulq_f32, vmulq_f64, vorrq_u32, vorrq_u64, vreinterpretq_f32_u32,
    vreinterpretq_f64_u64, vreinterpretq_u32_f32, vreinterpretq_u64_f64,
};

/// `floor(k / 2)` for an integer `k`.
const fn floor_half(k: i32) -> i32 {
    k.div_euclid(2)
}

/// `ceiling(k / 2)` for an integer `k`.
const fn ceil_half(k: i32) -> i32 {
    (k + 1).div_euclid(2)
}

/// `radix^e` for f64, by repeated multiplication or division by the radix.
///
/// Every step is exact as long as the result stays in the normal range, which
/// holds for all exponents used here (`|e| <= 538`).
const fn pow_radix_f64(e: i32) -> f64 {
    let radix = f64::RADIX as f64;
    let mut r = 1.0f64;
    let mut k = 0;
    while k < e {
        r *= radix;
        k += 1;
    }
    while k > e {
        r /= radix;
        k -= 1;
    }
    r
}

/// `radix^e` for f32 (see [`pow_radix_f64`]; here `|e| <= 76`).
const fn pow_radix_f32(e: i32) -> f32 {
    let radix = f32::RADIX as f32;
    let mut r = 1.0f32;
    let mut k = 0;
    while k < e {
        r *= radix;
        k += 1;
    }
    while k > e {
        r /= radix;
        k -= 1;
    }
    r
}

/// f64 digits (`digits(0._wp)` in the reference) as a signed exponent offset.
const F64_DIGITS: i32 = f64::MANTISSA_DIGITS as i32;
/// f32 digits (`digits(0._sp)` in the reference) as a signed exponent offset.
const F32_DIGITS: i32 = f32::MANTISSA_DIGITS as i32;

/// `tsml = radix^ceiling((minexponent - 1) / 2)` (`2^-511` for f64).
pub(crate) const F64_TSML: f64 = pow_radix_f64(ceil_half(f64::MIN_EXP - 1));
/// `tbig = radix^floor((maxexponent - digits + 1) / 2)` (`2^486` for f64).
pub(crate) const F64_TBIG: f64 = pow_radix_f64(floor_half(f64::MAX_EXP - F64_DIGITS + 1));
/// `ssml = radix^(-floor((minexponent - digits) / 2))` (`2^537` for f64).
pub(crate) const F64_SSML: f64 = pow_radix_f64(-floor_half(f64::MIN_EXP - F64_DIGITS));
/// `sbig = radix^(-ceiling((maxexponent + digits - 1) / 2))` (`2^-538` for f64).
pub(crate) const F64_SBIG: f64 = pow_radix_f64(-ceil_half(f64::MAX_EXP + F64_DIGITS - 1));

/// `tsml` for f32 (`2^-63`).
pub(crate) const F32_TSML: f32 = pow_radix_f32(ceil_half(f32::MIN_EXP - 1));
/// `tbig` for f32 (`2^52`).
pub(crate) const F32_TBIG: f32 = pow_radix_f32(floor_half(f32::MAX_EXP - F32_DIGITS + 1));
/// `ssml` for f32 (`2^75`).
pub(crate) const F32_SSML: f32 = pow_radix_f32(-floor_half(f32::MIN_EXP - F32_DIGITS));
/// `sbig` for f32 (`2^-76`).
pub(crate) const F32_SBIG: f32 = pow_radix_f32(-ceil_half(f32::MAX_EXP + F32_DIGITS - 1));

/// Elements per speculative block of the f64 kernel (see [`accumulate_f64`]).
const F64_BLOCK: usize = 64;
/// Elements per speculative block of the f32 kernel.
const F32_BLOCK: usize = 128;
/// Smallest block maximum `max|x|` for which the f64 fast path keeps plain
/// squares. At most `F64_BLOCK` elements of such a block lie below `tsml`; each
/// of their fused squares is off by at most half a subnormal ulp (`2^-1075`), so
/// the block adds an absolute error `<= 2^-1069`, while the norm is at least
/// `2^-500` and one of its ulps (in `Σ|x|²`) at least `2^-1052`.
const F64_FAST_MIN: f64 = pow_radix_f64(-500);
/// f32 counterpart of [`F64_FAST_MIN`]: `F32_BLOCK · 2^-150 = 2^-143` against an
/// ulp of at least `2^-123` in `Σ|x|²`.
const F32_FAST_MIN: f32 = pow_radix_f32(-50);

/// Euclidean norm of `x` (`DNRM2` with unit increment).
#[inline]
pub(crate) fn nrm2_f64(x: &[f64]) -> f64 {
    // SAFETY: the kernel reads exactly `x[..x.len()]`.
    let (asml, amed, abig) = unsafe { accumulate_f64(x.as_ptr(), x.len()) };
    blue_combine_f64(asml, amed, abig)
}

/// Euclidean norm of `x` (`SNRM2` with unit increment).
#[inline]
pub(crate) fn nrm2_f32(x: &[f32]) -> f32 {
    // SAFETY: the kernel reads exactly `x[..x.len()]`.
    let (asml, amed, abig) = unsafe { accumulate_f32(x.as_ptr(), x.len()) };
    blue_combine_f32(asml, amed, abig)
}

/// The `(asml, amed, abig)` accumulators of `x`, for callers that split a
/// vector into chunks: the accumulators of the chunks add up, and
/// [`blue_combine_f64`] of the sums is the norm of the whole vector.
#[cfg(all(feature = "parallel", not(feature = "force-scalar")))]
#[inline]
pub(crate) fn accumulators_f64(x: &[f64]) -> (f64, f64, f64) {
    // SAFETY: the kernel reads exactly `x[..x.len()]`.
    unsafe { accumulate_f64(x.as_ptr(), x.len()) }
}

/// f32 counterpart of [`accumulators_f64`].
#[cfg(all(feature = "parallel", not(feature = "force-scalar")))]
#[inline]
pub(crate) fn accumulators_f32(x: &[f32]) -> (f32, f32, f32) {
    // SAFETY: the kernel reads exactly `x[..x.len()]`.
    unsafe { accumulate_f32(x.as_ptr(), x.len()) }
}

/// The reference combine step for f64 accumulators (`asml` may contain
/// contributions made after the first big value; they are discarded here).
#[inline]
pub(crate) fn blue_combine_f64(asml: f64, amed: f64, abig: f64) -> f64 {
    let (scl, sumsq) = if abig > 0.0 {
        // Combine abig and amed if abig > 0.
        let mut abig = abig;
        if amed > 0.0 || amed > f64::MAX || amed.is_nan() {
            abig += (amed * F64_SBIG) * F64_SBIG;
        }
        (1.0 / F64_SBIG, abig)
    } else if asml > 0.0 {
        // Combine amed and asml if asml > 0.
        if amed > 0.0 || amed > f64::MAX || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / F64_SSML;
            let (ymin, ymax) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            let ratio = ymin / ymax;
            (1.0, ymax * ymax * (1.0 + ratio * ratio))
        } else {
            (1.0 / F64_SSML, asml)
        }
    } else {
        // Otherwise all values are mid-range (or zero).
        (1.0, amed)
    };
    scl * sumsq.sqrt()
}

/// The reference combine step for f32 accumulators (see [`blue_combine_f64`]).
#[inline]
pub(crate) fn blue_combine_f32(asml: f32, amed: f32, abig: f32) -> f32 {
    let (scl, sumsq) = if abig > 0.0 {
        let mut abig = abig;
        if amed > 0.0 || amed > f32::MAX || amed.is_nan() {
            abig += (amed * F32_SBIG) * F32_SBIG;
        }
        (1.0 / F32_SBIG, abig)
    } else if asml > 0.0 {
        if amed > 0.0 || amed > f32::MAX || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / F32_SSML;
            let (ymin, ymax) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            let ratio = ymin / ymax;
            (1.0, ymax * ymax * (1.0 + ratio * ratio))
        } else {
            (1.0 / F32_SSML, asml)
        }
    } else {
        (1.0, amed)
    };
    scl * sumsq.sqrt()
}

/// The three f64 accumulators of one lane group.
#[derive(Clone, Copy)]
struct Acc64 {
    sml: float64x2_t,
    med: float64x2_t,
    big: float64x2_t,
}

/// The three f32 accumulators of one lane group.
#[derive(Clone, Copy)]
struct Acc32 {
    sml: float32x4_t,
    med: float32x4_t,
    big: float32x4_t,
}

/// Classifies the two lanes of `v` and squares each into its accumulator.
#[inline(always)]
unsafe fn step_f64(acc: &mut Acc64, v: float64x2_t, k: &[float64x2_t; 4]) {
    let [tsml, tbig, ssml, sbig] = *k;
    let is_sml = vcaltq_f64(v, tsml);
    let is_big = vcagtq_f64(v, tbig);
    let ts = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vmulq_f64(v, ssml)), is_sml));
    let tb = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vmulq_f64(v, sbig)), is_big));
    let tm = vreinterpretq_f64_u64(vbicq_u64(
        vreinterpretq_u64_f64(v),
        vorrq_u64(is_sml, is_big),
    ));
    acc.sml = vfmaq_f64(acc.sml, ts, ts);
    acc.big = vfmaq_f64(acc.big, tb, tb);
    acc.med = vfmaq_f64(acc.med, tm, tm);
}

/// Classifies the four lanes of `v` and squares each into its accumulator.
#[inline(always)]
unsafe fn step_f32(acc: &mut Acc32, v: float32x4_t, k: &[float32x4_t; 4]) {
    let [tsml, tbig, ssml, sbig] = *k;
    let is_sml = vcaltq_f32(v, tsml);
    let is_big = vcagtq_f32(v, tbig);
    let ts = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vmulq_f32(v, ssml)), is_sml));
    let tb = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vmulq_f32(v, sbig)), is_big));
    let tm = vreinterpretq_f32_u32(vbicq_u32(
        vreinterpretq_u32_f32(v),
        vorrq_u32(is_sml, is_big),
    ));
    acc.sml = vfmaq_f32(acc.sml, ts, ts);
    acc.big = vfmaq_f32(acc.big, tb, tb);
    acc.med = vfmaq_f32(acc.med, tm, tm);
}

/// Scalar classification of one f64 element (the reference loop body, with the
/// squares fused like the vector lanes).
#[inline(always)]
fn step_scalar_f64(acc: (f64, f64, f64), v: f64) -> (f64, f64, f64) {
    let (asml, amed, abig) = acc;
    let ax = v.abs();
    if ax > F64_TBIG {
        let t = ax * F64_SBIG;
        (asml, amed, t.mul_add(t, abig))
    } else if ax < F64_TSML {
        let t = ax * F64_SSML;
        (t.mul_add(t, asml), amed, abig)
    } else {
        (asml, ax.mul_add(ax, amed), abig)
    }
}

/// Scalar classification of one f32 element (see [`step_scalar_f64`]).
#[inline(always)]
fn step_scalar_f32(acc: (f32, f32, f32), v: f32) -> (f32, f32, f32) {
    let (asml, amed, abig) = acc;
    let ax = v.abs();
    if ax > F32_TBIG {
        let t = ax * F32_SBIG;
        (asml, amed, t.mul_add(t, abig))
    } else if ax < F32_TSML {
        let t = ax * F32_SSML;
        (t.mul_add(t, asml), amed, abig)
    } else {
        (asml, ax.mul_add(ax, amed), abig)
    }
}

/// Returns `(asml, amed, abig)` for `x[..n]`.
///
/// Full blocks of [`F64_BLOCK`] elements take the speculative fast path (plain
/// squares into `amed`) when their maximum allows it, and are classified
/// element by element otherwise; the remainder is always classified.
///
/// # Safety
///
/// `x` must be valid for reads of `n` consecutive `f64` values.
#[inline]
unsafe fn accumulate_f64(x: *const f64, n: usize) -> (f64, f64, f64) {
    let k = [
        vdupq_n_f64(F64_TSML),
        vdupq_n_f64(F64_TBIG),
        vdupq_n_f64(F64_SSML),
        vdupq_n_f64(F64_SBIG),
    ];
    let zero = vdupq_n_f64(0.0);
    let mut a0 = Acc64 {
        sml: zero,
        med: zero,
        big: zero,
    };
    let mut a1 = a0;
    let mut fast = zero;
    let blocks = n - n % F64_BLOCK;
    let mut i = 0;
    while i < blocks {
        let p = x.add(i);
        let mut s = [zero; 8];
        let (mut m0, mut m1) = (zero, zero);
        let mut j = 0;
        while j < F64_BLOCK {
            let q = p.add(j);
            let v = [
                vld1q_f64(q),
                vld1q_f64(q.add(2)),
                vld1q_f64(q.add(4)),
                vld1q_f64(q.add(6)),
                vld1q_f64(q.add(8)),
                vld1q_f64(q.add(10)),
                vld1q_f64(q.add(12)),
                vld1q_f64(q.add(14)),
            ];
            for (sk, &vk) in s.iter_mut().zip(&v) {
                *sk = vfmaq_f64(*sk, vk, vk);
            }
            m0 = vmaxq_f64(
                m0,
                vmaxq_f64(
                    vmaxq_f64(vabsq_f64(v[0]), vabsq_f64(v[1])),
                    vmaxq_f64(vabsq_f64(v[2]), vabsq_f64(v[3])),
                ),
            );
            m1 = vmaxq_f64(
                m1,
                vmaxq_f64(
                    vmaxq_f64(vabsq_f64(v[4]), vabsq_f64(v[5])),
                    vmaxq_f64(vabsq_f64(v[6]), vabsq_f64(v[7])),
                ),
            );
            j += 16;
        }
        // `vmaxq` propagates NaN, so a NaN block fails the range test too.
        let block_max = vmaxvq_f64(vmaxq_f64(m0, m1));
        if (F64_FAST_MIN..=F64_TBIG).contains(&block_max) {
            let s01 = vaddq_f64(s[0], s[1]);
            let s23 = vaddq_f64(s[2], s[3]);
            let s45 = vaddq_f64(s[4], s[5]);
            let s67 = vaddq_f64(s[6], s[7]);
            fast = vaddq_f64(fast, vaddq_f64(vaddq_f64(s01, s23), vaddq_f64(s45, s67)));
        } else {
            let mut j = 0;
            while j < F64_BLOCK {
                let q = p.add(j);
                step_f64(&mut a0, vld1q_f64(q), &k);
                step_f64(&mut a1, vld1q_f64(q.add(2)), &k);
                step_f64(&mut a0, vld1q_f64(q.add(4)), &k);
                step_f64(&mut a1, vld1q_f64(q.add(6)), &k);
                j += 8;
            }
        }
        i += F64_BLOCK;
    }
    while n - i >= 8 {
        let p = x.add(i);
        step_f64(&mut a0, vld1q_f64(p), &k);
        step_f64(&mut a1, vld1q_f64(p.add(2)), &k);
        step_f64(&mut a0, vld1q_f64(p.add(4)), &k);
        step_f64(&mut a1, vld1q_f64(p.add(6)), &k);
        i += 8;
    }
    while n - i >= 2 {
        step_f64(&mut a0, vld1q_f64(x.add(i)), &k);
        i += 2;
    }
    let mut acc = (
        vaddvq_f64(vaddq_f64(a0.sml, a1.sml)),
        vaddvq_f64(vaddq_f64(vaddq_f64(a0.med, a1.med), fast)),
        vaddvq_f64(vaddq_f64(a0.big, a1.big)),
    );
    if i < n {
        acc = step_scalar_f64(acc, *x.add(i));
    }
    acc
}

/// Returns `(asml, amed, abig)` for `x[..n]` (see [`accumulate_f64`]).
///
/// # Safety
///
/// `x` must be valid for reads of `n` consecutive `f32` values.
#[inline]
unsafe fn accumulate_f32(x: *const f32, n: usize) -> (f32, f32, f32) {
    let k = [
        vdupq_n_f32(F32_TSML),
        vdupq_n_f32(F32_TBIG),
        vdupq_n_f32(F32_SSML),
        vdupq_n_f32(F32_SBIG),
    ];
    let zero = vdupq_n_f32(0.0);
    let mut a0 = Acc32 {
        sml: zero,
        med: zero,
        big: zero,
    };
    let mut a1 = a0;
    let mut fast = zero;
    let blocks = n - n % F32_BLOCK;
    let mut i = 0;
    while i < blocks {
        let p = x.add(i);
        let mut s = [zero; 8];
        let (mut m0, mut m1) = (zero, zero);
        let mut j = 0;
        while j < F32_BLOCK {
            let q = p.add(j);
            let v = [
                vld1q_f32(q),
                vld1q_f32(q.add(4)),
                vld1q_f32(q.add(8)),
                vld1q_f32(q.add(12)),
                vld1q_f32(q.add(16)),
                vld1q_f32(q.add(20)),
                vld1q_f32(q.add(24)),
                vld1q_f32(q.add(28)),
            ];
            for (sk, &vk) in s.iter_mut().zip(&v) {
                *sk = vfmaq_f32(*sk, vk, vk);
            }
            m0 = vmaxq_f32(
                m0,
                vmaxq_f32(
                    vmaxq_f32(vabsq_f32(v[0]), vabsq_f32(v[1])),
                    vmaxq_f32(vabsq_f32(v[2]), vabsq_f32(v[3])),
                ),
            );
            m1 = vmaxq_f32(
                m1,
                vmaxq_f32(
                    vmaxq_f32(vabsq_f32(v[4]), vabsq_f32(v[5])),
                    vmaxq_f32(vabsq_f32(v[6]), vabsq_f32(v[7])),
                ),
            );
            j += 32;
        }
        let block_max = vmaxvq_f32(vmaxq_f32(m0, m1));
        if (F32_FAST_MIN..=F32_TBIG).contains(&block_max) {
            let s01 = vaddq_f32(s[0], s[1]);
            let s23 = vaddq_f32(s[2], s[3]);
            let s45 = vaddq_f32(s[4], s[5]);
            let s67 = vaddq_f32(s[6], s[7]);
            fast = vaddq_f32(fast, vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67)));
        } else {
            let mut j = 0;
            while j < F32_BLOCK {
                let q = p.add(j);
                step_f32(&mut a0, vld1q_f32(q), &k);
                step_f32(&mut a1, vld1q_f32(q.add(4)), &k);
                step_f32(&mut a0, vld1q_f32(q.add(8)), &k);
                step_f32(&mut a1, vld1q_f32(q.add(12)), &k);
                j += 16;
            }
        }
        i += F32_BLOCK;
    }
    while n - i >= 16 {
        let p = x.add(i);
        step_f32(&mut a0, vld1q_f32(p), &k);
        step_f32(&mut a1, vld1q_f32(p.add(4)), &k);
        step_f32(&mut a0, vld1q_f32(p.add(8)), &k);
        step_f32(&mut a1, vld1q_f32(p.add(12)), &k);
        i += 16;
    }
    while n - i >= 4 {
        step_f32(&mut a0, vld1q_f32(x.add(i)), &k);
        i += 4;
    }
    let mut acc = (
        vaddvq_f32(vaddq_f32(a0.sml, a1.sml)),
        vaddvq_f32(vaddq_f32(vaddq_f32(a0.med, a1.med), fast)),
        vaddvq_f32(vaddq_f32(a0.big, a1.big)),
    );
    while i < n {
        acc = step_scalar_f32(acc, *x.add(i));
        i += 1;
    }
    acc
}

#[cfg(test)]
#[path = "nrm2_blue_tests.rs"]
mod tests;
