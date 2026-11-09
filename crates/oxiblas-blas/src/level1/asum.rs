//! ASUM: Sum of absolute values (L1 norm)
//!
//! Computes ||x||_1 = Σ |x\[i\]|
//!
//! This module also hosts the other missing complex Level-1 BLAS routines
//! flagged by the production audit (they are grouped here, rather than in
//! `rot.rs`, per the audit's file scope for this fix):
//!
//! - [`cabs1`] / [`scabs1`] / [`dcabs1`]: the BLAS `cabs1` metric
//!   (`|Re| + |Im|`, NOT the complex modulus).
//! - [`casum`] / [`scasum`] / [`dzasum`]: complex `ASUM` using the `cabs1`
//!   metric.
//! - [`crotg`]: complex Givens rotation generator (`ZROTG`/`CROTG`).
//! - [`zdrot`]: apply a **real-valued** Givens rotation to complex vectors
//!   (`ZDROT`/`CSROT`).

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Real, Scalar};

/// Computes the sum of absolute values (L1 norm) for real vectors.
///
/// ||x||_1 = Σ |x\[i\]|
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::asum;
///
/// let x = [1.0f64, -2.0, 3.0, -4.0];
/// let sum = asum(&x);
///
/// // |1| + |-2| + |3| + |-4| = 10
/// assert!((sum - 10.0).abs() < 1e-10);
/// ```
pub fn asum<T: Real>(x: &[T]) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    // Use 4-way accumulation
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += Scalar::abs(x[base]);
        acc1 += Scalar::abs(x[base + 1]);
        acc2 += Scalar::abs(x[base + 2]);
        acc3 += Scalar::abs(x[base + 3]);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += Scalar::abs(x[base + i]);
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Computes `|Re(z)| + |Im(z)|`, the BLAS "cabs1" metric (`SCABS1`/`DCABS1`).
///
/// This is the comparison/summation metric used by the complex Level-1 BLAS
/// routines `SCASUM`/`DZASUM` (sum, see [`casum`]) and `ICAMAX`/`IZAMAX`
/// (max-index search, see [`crate::level1::iamax()`] and
/// [`crate::level1::iamax_strided`]). It is **not** the complex modulus
/// `sqrt(re^2 + im^2)`; reference BLAS deliberately uses the cheaper L1-style
/// metric, so callers implementing reference `I?AMAX` semantics need exactly
/// this function. For real scalars (`im == 0`) this reduces to plain `|re|`;
/// a NaN in either component propagates.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::cabs1;
/// use oxiblas_core::scalar::c64;
///
/// // |3| + |-4| = 7 (the modulus would be 5).
/// assert_eq!(cabs1(c64(3.0, -4.0)), 7.0);
/// // Real scalars: plain absolute value.
/// assert_eq!(cabs1(-2.5f64), 2.5);
/// ```
#[inline]
pub fn cabs1<T: Scalar>(z: T) -> T::Real {
    Scalar::abs(z.real()) + Scalar::abs(z.imag())
}

/// `SCABS1`: `|Re(z)| + |Im(z)|` for a single-precision complex scalar.
///
/// Thin concrete wrapper over [`cabs1`] matching the reference BLAS name.
#[inline]
pub fn scabs1(z: Complex32) -> f32 {
    cabs1(z)
}

/// `DCABS1`: `|Re(z)| + |Im(z)|` for a double-precision complex scalar.
///
/// Thin concrete wrapper over [`cabs1`] matching the reference BLAS name.
#[inline]
pub fn dcabs1(z: Complex64) -> f64 {
    cabs1(z)
}

/// Computes the sum of `|Re(x[i])| + |Im(x[i])|` for a complex vector
/// (the BLAS `cabs1` metric), matching reference BLAS `SCASUM`/`DZASUM`.
///
/// Unlike the real [`asum`], which sums `|x[i]|` directly, the complex
/// BLAS `ASUM` routines deliberately sum `|Re| + |Im|` per element rather
/// than the true complex modulus `sqrt(re^2 + im^2)` -- this matches
/// Netlib `scasum.f`/`dzasum.f` (via the internal `SCABS1`/`DCABS1`
/// helper) exactly.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::casum;
/// use oxiblas_core::scalar::c64;
///
/// let x = [c64(1.0, 2.0), c64(-3.0, 4.0), c64(0.0, -5.0)];
/// let sum = casum(&x);
///
/// // (|1|+|2|) + (|-3|+|4|) + (|0|+|-5|) = 3 + 7 + 5 = 15
/// assert!((sum - 15.0).abs() < 1e-10);
/// ```
pub fn casum<T: ComplexScalar>(x: &[T]) -> T::Real {
    let n = x.len();
    if n == 0 {
        return T::Real::zero();
    }

    // 4-way accumulation, mirroring the real `asum` above.
    let mut acc0 = T::Real::zero();
    let mut acc1 = T::Real::zero();
    let mut acc2 = T::Real::zero();
    let mut acc3 = T::Real::zero();

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += cabs1(x[base]);
        acc1 += cabs1(x[base + 1]);
        acc2 += cabs1(x[base + 2]);
        acc3 += cabs1(x[base + 3]);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += cabs1(x[base + i]);
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// `SCASUM`: sum of `|Re| + |Im|` for a single-precision complex vector.
///
/// Thin concrete wrapper over [`casum`] matching the reference BLAS name.
#[inline]
pub fn scasum(x: &[Complex32]) -> f32 {
    casum(x)
}

/// `DZASUM`: sum of `|Re| + |Im|` for a double-precision complex vector.
///
/// Thin concrete wrapper over [`casum`] matching the reference BLAS name.
#[inline]
pub fn dzasum(x: &[Complex64]) -> f64 {
    casum(x)
}

/// Result of the complex plane-rotation generator (`ZROTG`/`CROTG`).
///
/// `c` is always real even though the inputs/outputs are complex; `s` is
/// complex. This matches reference BLAS `ZROTG`/`CROTG` exactly.
#[derive(Debug, Clone, Copy)]
pub struct ComplexRotgResult<T: Scalar> {
    /// The rotated value `r = sgn(a) * sqrt(|a|^2 + |b|^2)` (complex), which
    /// carries the phase of `a`. When `a == 0` (and `b != 0`) it is the real,
    /// non-negative `|b|`; when `b == 0` it is `a` itself.
    pub r: T,
    /// Cosine of the rotation (always real, in `[0, 1]`).
    pub c: T::Real,
    /// Sine of the rotation (complex).
    pub s: T,
}

/// Divides a complex scalar by a **real** scalar, component-wise.
///
/// The reference `zrotg.f90` only ever divides by real quantities. Forming
/// `T::from_real(d)` and using complex/complex division instead squares the
/// divisor's modulus internally (`(a·conj(b)) / |b|^2`), which overflows
/// once `|d|` exceeds `sqrt(MAX)` (≈1e19 in `f32`, ≈1e154 in `f64`) and
/// underflows for tiny divisors — the root cause of issue #12.
#[inline]
fn div_real<T: ComplexScalar>(z: T, d: T::Real) -> T {
    T::from_real_imag(z.real() / d, z.imag() / d)
}

/// Multiplies a complex scalar by a **real** scalar, component-wise (no
/// `0·∞` cross terms, matching how gfortran evaluates real×complex products).
#[inline]
fn mul_real<T: ComplexScalar>(z: T, d: T::Real) -> T {
    T::from_real_imag(z.real() * d, z.imag() * d)
}

/// `max(|Re(z)|, |Im(z)|)`, the magnitude proxy `zrotg.f90` range-checks.
#[inline]
fn max_abs_component<T: ComplexScalar>(z: T) -> T::Real {
    num_traits::Float::max(Scalar::abs(z.real()), Scalar::abs(z.imag()))
}

/// Shared tail of `zrotg.f90`'s unscaled and scaled general branches.
///
/// `f`/`g` are the (possibly scaled) inputs, `f2 = |f|^2` and `h2` the
/// (possibly `w^2`-weighted) `|f|^2 + |g|^2`; `rtmax` is the branch's
/// `sqrt(safmax / 4)`. Returns `(c, r, s)` before the scaled branch's final
/// `c·w`, `r·u` rescaling. Every division is by a real scalar.
#[inline]
fn crotg_general_tail<T: ComplexScalar>(
    f: T,
    g: T,
    f2: T::Real,
    h2: T::Real,
    rtmin: T::Real,
    rtmax: T::Real,
    safmin: T::Real,
) -> (T::Real, T, T) {
    if f2 >= h2 * safmin {
        // safmin <= f2/h2 <= 1, and h2/f2 is finite.
        let c = Real::sqrt(f2 / h2);
        let r = div_real(f, c);
        let rtmax = rtmax + rtmax;
        let s = if f2 > rtmin && h2 < rtmax {
            // safmin <= sqrt(f2*h2) <= safmax.
            g.conj() * div_real(f, Real::sqrt(f2 * h2))
        } else {
            g.conj() * div_real(r, h2)
        };
        (c, r, s)
    } else {
        // f2/h2 <= safmin may be subnormal and h2/f2 may overflow. Here
        // g2 >> f2, so h2 == g2 and sqrt(safmin) <= sqrt(f2*h2) <= sqrt(safmax).
        let d = Real::sqrt(f2 * h2);
        let c = f2 / d;
        let r = if c >= safmin {
            div_real(f, c)
        } else {
            // f2 / sqrt(f2*h2) < safmin: h2/d stays representable.
            mul_real(f, h2 / d)
        };
        let s = g.conj() * div_real(f, d);
        (c, r, s)
    }
}

/// Generates a Givens plane rotation for complex scalars (`ZROTG`/`CROTG`).
///
/// Given complex scalars `a` and `b`, computes a real cosine `c` and a
/// complex sine `s` such that
/// ```text
/// [  c         s  ] [ a ]   [ r ]
/// [ -conj(s)   c  ] [ b ] = [ 0 ]
/// ```
/// with `c^2 + |s|^2 = 1`, using the reference formulas
/// ```text
/// c = |a| / sqrt(|a|^2 + |b|^2)
/// s = sgn(a) * conj(b) / sqrt(|a|^2 + |b|^2)
/// r = sgn(a) * sqrt(|a|^2 + |b|^2)        (sgn(a) = a/|a|, or 1 if a == 0)
/// ```
///
/// This is a translation of the LAPACK >= 3.10 reference `zrotg.f90` /
/// `crotg.f90` (Netlib BLAS 3.12; E. Anderson, "Algorithm 978: Safe Scaling
/// in the Level 1 BLAS", ACM TOMS 44, 2017). Its safe-scaling branches
/// (`safmin` = smallest normal, `safmax = 2/safmin`) and real-only divisions
/// keep every intermediate representable, so inputs anywhere from the
/// subnormal range up to about `MAX/4` per component neither overflow nor
/// underflow. Special cases follow the current reference convention:
///
/// * `b == 0`: `c = 1`, `s = 0`, `r = a` (this includes `a == b == 0`).
/// * `a == 0`, `b != 0`: `c = 0`, `r = |b|` (real), `s = conj(b) / |b|`.
///
/// (The pre-3.10 `zrotg.f` used `c = 0, s = 1, r = b` for `a == 0`; both
/// are valid rotations, but only the above matches current Netlib.)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::crotg;
/// use oxiblas_core::scalar::c64;
///
/// let a = c64(3.0, 4.0);
/// let b = c64(1.0, -2.0);
/// let result = crotg(a, b);
///
/// let c = c64(result.c, 0.0);
/// let lhs = c * a + result.s * b;
/// let residual = -result.s.conj() * a + c * b;
///
/// // c*a + s*b == r, and the rotation zeroes the second component.
/// assert!((lhs - result.r).norm() < 1e-10);
/// assert!(residual.norm() < 1e-10);
/// ```
pub fn crotg<T: ComplexScalar>(a: T, b: T) -> ComplexRotgResult<T> {
    let zero = T::Real::zero();
    let one = T::Real::one();
    let two = one + one;

    // `zrotg.f90` scaling constants: safmin = radix**max(minexp-1, 1-maxexp)
    // is the smallest positive normal (2^-1022 / 2^-126), and
    // safmax = radix**max(1-minexp, maxexp-1) = 2^1023 / 2^127 = 2 / safmin.
    let safmin = T::min_positive();
    let safmax = two / safmin;
    let rtmin = Real::sqrt(safmin);

    let f = a;
    let g = b;

    if g == T::zero() {
        return ComplexRotgResult {
            r: f,
            c: one,
            s: T::zero(),
        };
    }

    if f == T::zero() {
        // c = 0, r = |g| (real), s = conj(g) / |g|.
        let (gr, gi) = (g.real(), g.imag());
        let (r, s) = if gr == zero {
            let r = Scalar::abs(gi);
            (r, div_real(g.conj(), r))
        } else if gi == zero {
            let r = Scalar::abs(gr);
            (r, div_real(g.conj(), r))
        } else {
            let g1 = max_abs_component(g);
            let rtmax = Real::sqrt(safmax / two);
            if g1 > rtmin && g1 < rtmax {
                // Unscaled algorithm.
                let d = Real::sqrt(Scalar::abs_sq(g));
                (d, div_real(g.conj(), d))
            } else {
                // Scaled algorithm.
                let u = num_traits::Float::min(safmax, num_traits::Float::max(safmin, g1));
                let gs = div_real(g, u);
                let d = Real::sqrt(Scalar::abs_sq(gs));
                (d * u, div_real(gs.conj(), d))
            }
        };
        return ComplexRotgResult {
            r: T::from_real(r),
            c: zero,
            s,
        };
    }

    let f1 = max_abs_component(f);
    let g1 = max_abs_component(g);
    let rtmax = Real::sqrt(safmax / (two + two));
    if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        // Unscaled algorithm: safmin <= f2 <= h2 <= safmax.
        let f2 = Scalar::abs_sq(f);
        let h2 = f2 + Scalar::abs_sq(g);
        let (c, r, s) = crotg_general_tail(f, g, f2, h2, rtmin, rtmax, safmin);
        ComplexRotgResult { r, c, s }
    } else {
        // Scaled algorithm.
        let u = num_traits::Float::min(
            safmax,
            num_traits::Float::max(safmin, num_traits::Float::max(f1, g1)),
        );
        let gs = div_real(g, u);
        let g2 = Scalar::abs_sq(gs);
        let (w, fs, f2, h2) = if f1 / u < rtmin {
            // f is not well-scaled when scaled by g1: use a separate scale.
            let v = num_traits::Float::min(safmax, num_traits::Float::max(safmin, f1));
            let w = v / u;
            let fs = div_real(f, v);
            let f2 = Scalar::abs_sq(fs);
            (w, fs, f2, f2 * (w * w) + g2)
        } else {
            // Otherwise use the same scaling for f and g.
            let fs = div_real(f, u);
            let f2 = Scalar::abs_sq(fs);
            (one, fs, f2, f2 + g2)
        };
        let (c, r, s) = crotg_general_tail(fs, gs, f2, h2, rtmin, rtmax, safmin);
        // Rescale c and r.
        ComplexRotgResult {
            r: mul_real(r, u),
            c: c * w,
            s,
        }
    }
}

/// Applies the real rotation `[c s; -s c]` to one complex pair, scaling each
/// component by the real `c`/`s` exactly as reference `ZDROT`/`CSROT`
/// (compiled by gfortran) does: `x' = c·x + s·y`, `y' = c·y - s·x`.
///
/// Unlike multiplying by `T::from_real(c)` (a complex product with `0·Im`
/// cross terms), this never turns a finite rotation of an infinite component
/// into a NaN. Shared by [`zdrot`] and the strided `zdrot_strided`.
#[inline]
pub(crate) fn zdrot_pair<T: ComplexScalar>(c: T::Real, s: T::Real, x: T, y: T) -> (T, T) {
    let (xr, xi) = (x.real(), x.imag());
    let (yr, yi) = (y.real(), y.imag());
    (
        T::from_real_imag(c * xr + s * yr, c * xi + s * yi),
        T::from_real_imag(c * yr - s * xr, c * yi - s * xi),
    )
}

/// Applies a **real-valued** Givens plane rotation to complex vectors
/// (`ZDROT`/`CSROT`).
///
/// Given a real cosine `c` and real sine `s`, rotates each pair
/// `(x[i], y[i])` of complex numbers:
/// ```text
/// [ x[i] ]     [ c  s ] [ x[i] ]
/// [ y[i] ]  =  [-s  c ] [ y[i] ]
/// ```
/// Note this is distinct from [`crotg`]'s rotation, whose `s` is complex --
/// `ZDROT` (Netlib `zdrot.f`) is a separate BLAS extension that applies a
/// *real* rotation angle to complex data (e.g. as produced by the real
/// `rotg` acting on the magnitudes of a complex pair).
///
/// `c` and `s` scale the real and imaginary parts separately, exactly like
/// the reference (`c·x` is `(c·Re x, c·Im x)`), so infinite components are
/// rotated without spurious `0·∞ = NaN` cross terms. For increments
/// (including zero and negative ones) see
/// [`zdrot_strided`](crate::level1::zdrot_strided).
///
/// # Panics
///
/// Panics if `x` and `y` have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::{rotg, zdrot};
/// use oxiblas_core::scalar::c64;
///
/// // Real rotation that zeroes (3, 4) -> (5, 0)
/// let rg = rotg(3.0f64, 4.0);
///
/// // Apply the same real (c, s) to complex vectors sharing a common phase.
/// let phase = c64(0.6, 0.8);
/// let mut x = [phase * c64(3.0, 0.0)];
/// let mut y = [phase * c64(4.0, 0.0)];
///
/// zdrot(rg.c, rg.s, &mut x, &mut y);
///
/// assert!((x[0] - phase * c64(5.0, 0.0)).norm() < 1e-10);
/// assert!(y[0].norm() < 1e-10);
/// ```
pub fn zdrot<T: ComplexScalar>(c: T::Real, s: T::Real, x: &mut [T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    for i in 0..x.len() {
        let (xi, yi) = zdrot_pair(c, s, x[i], y[i]);
        x[i] = xi;
        y[i] = yi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_core::scalar::c64;

    #[test]
    fn test_asum_basic() {
        let x = [1.0, -2.0, 3.0, -4.0];
        let sum = asum(&x);
        assert!((sum - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = asum(&x);
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_empty() {
        let x: [f64; 0] = [];
        let sum = asum(&x);
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_asum_single() {
        let x = [-5.0];
        let sum = asum(&x);
        assert!((sum - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_f32() {
        let x = [1.0f32, -2.0, 3.0];
        let sum = asum(&x);
        assert!((sum - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_asum_zeros() {
        let x = [0.0; 5];
        let sum = asum(&x);
        assert_eq!(sum, 0.0);
    }

    // =========================================================================
    // Complex ASUM (casum / scasum / dzasum) -- cabs1 metric
    // =========================================================================

    #[test]
    fn test_casum_hand_verified() {
        // (|1|+|2|) + (|-3|+|4|) + (|0|+|-5|) = 3 + 7 + 5 = 15
        // The true L1 norm using the modulus instead would be
        // sqrt(5) + 5 + 5 = 12.236..., a materially different number,
        // confirming this exercises the cabs1 metric, not the modulus.
        let x = [c64(1.0, 2.0), c64(-3.0, 4.0), c64(0.0, -5.0)];
        let sum = casum(&x);
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_casum_empty() {
        let x: [Complex64; 0] = [];
        assert_eq!(casum(&x), 0.0);
    }

    #[test]
    fn test_casum_real_reduces_to_asum() {
        // For purely-real complex entries, cabs1 == |re|, matching `asum`.
        let x = [c64(3.0, 0.0), c64(-4.0, 0.0)];
        assert!((casum(&x) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_scasum_and_dzasum() {
        let x32 = [Complex32::new(1.0, 2.0), Complex32::new(-3.0, 4.0)];
        // (1+2) + (3+4) = 10
        assert!((scasum(&x32) - 10.0).abs() < 1e-5);

        let x64 = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        assert!((dzasum(&x64) - 10.0).abs() < 1e-10);
    }

    // =========================================================================
    // Complex rotg (crotg / zrotg)
    // =========================================================================

    #[test]
    fn test_crotg_basic_round_trip() {
        // Hand-verified against the classic Netlib zrotg.f algorithm.
        let a = c64(3.0, 4.0);
        let b = c64(1.0, -2.0);
        let result = crotg(a, b);

        // |r| == sqrt(|a|^2 + |b|^2)
        let expected_r_mag = (25.0f64 + 5.0).sqrt();
        assert!((result.r.norm() - expected_r_mag).abs() < 1e-9);

        // The defining rotation property: c*a + s*b == r
        let c = c64(result.c, 0.0);
        let lhs = c * a + result.s * b;
        assert!((lhs - result.r).norm() < 1e-9);

        // ...and it truly zeroes the second component:
        // -conj(s)*a + c*b == 0
        let residual = -result.s.conj() * a + c * b;
        assert!(
            residual.norm() < 1e-9,
            "residual should vanish, got {residual:?}"
        );
    }

    #[test]
    fn test_crotg_a_zero() {
        // Reference zrotg.f90 (LAPACK >= 3.10) special case: a == 0 =>
        // c = 0, r = |b| (real), s = conj(b) / |b|.
        let a = c64(0.0, 0.0);
        let b = c64(3.0, 4.0);
        let result = crotg(a, b);

        assert_eq!(result.c, 0.0);
        assert!((result.s - c64(0.6, -0.8)).norm() < 1e-12);
        assert!((result.r - c64(5.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_crotg_both_zero() {
        // b == 0 is checked first by the reference: c = 1, s = 0, r = a.
        let a = c64(0.0, 0.0);
        let b = c64(0.0, 0.0);
        let result = crotg(a, b);

        assert_eq!(result.c, 1.0);
        assert_eq!(result.s, c64(0.0, 0.0));
        assert_eq!(result.r, c64(0.0, 0.0));
    }

    #[test]
    fn test_crotg_b_zero() {
        // a purely real/nonzero, b == 0: rotation should be identity-like.
        let a = c64(5.0, 0.0);
        let b = c64(0.0, 0.0);
        let result = crotg(a, b);

        assert!((result.c - 1.0).abs() < 1e-10);
        assert!(result.s.norm() < 1e-10);
        assert!((result.r - a).norm() < 1e-10);
    }

    // =========================================================================
    // Issue #12: crotg overflow + LAPACK >= 3.10 (`zrotg.f90`) conventions
    // =========================================================================

    /// Checks the defining ZROTG/CROTG identities for `crotg(a, b)`: every
    /// output finite, `c` real in `[0, 1]`, `c^2 + |s|^2 = 1`,
    /// `[c s; -conj(s) c] [a; b] = [r; 0]`, `|r| = hypot(|a|, |b|)` and, for
    /// `a != 0`, `r` carries the phase of `a`. Residuals are measured with the
    /// hypot-based modulus relative to `max(|a|, |b|)` -- never via `|z|^2` --
    /// so the check itself cannot overflow or underflow.
    fn assert_crotg_identities<T: ComplexScalar>(a: T, b: T, tol: T::Real) {
        let res = crotg(a, b);
        let (r, c, s) = (res.r, res.c, res.s);
        let zero = T::Real::zero();
        let one = T::Real::one();
        let finite = |v: T::Real| num_traits::Float::is_finite(v);
        assert!(
            finite(c)
                && finite(s.real())
                && finite(s.imag())
                && finite(r.real())
                && finite(r.imag()),
            "non-finite output for a={a:?}, b={b:?}: r={r:?}, c={c:?}, s={s:?}"
        );
        assert!(
            c >= zero && c <= one,
            "c={c:?} outside [0, 1] for a={a:?}, b={b:?}"
        );

        let abs_s = Scalar::abs(s);
        let unit = c * c + abs_s * abs_s;
        assert!(
            Scalar::abs(unit - one) <= tol,
            "c^2 + |s|^2 = {unit:?} for a={a:?}, b={b:?}"
        );

        let abs_a = Scalar::abs(a);
        let abs_b = Scalar::abs(b);
        let scale = num_traits::Float::max(abs_a, abs_b);
        let c_t = T::from_real(c);
        let top = c_t * a + s * b;
        let bottom = -s.conj() * a + c_t * b;
        assert!(
            Scalar::abs(top - r) <= tol * scale,
            "c*a + s*b != r for a={a:?}, b={b:?}: got {top:?}, r={r:?}"
        );
        assert!(
            Scalar::abs(bottom) <= tol * scale,
            "-conj(s)*a + c*b != 0 for a={a:?}, b={b:?}: got {bottom:?}"
        );

        let abs_r = Scalar::abs(r);
        let expected = Real::hypot(abs_a, abs_b);
        assert!(
            Scalar::abs(abs_r - expected) <= tol * expected,
            "|r| = {abs_r:?}, expected {expected:?} for a={a:?}, b={b:?}"
        );

        if abs_a != zero {
            // sgn(r) == sgn(a), compared on unit-modulus representatives.
            let unit_r = T::from_real_imag(r.real() / abs_r, r.imag() / abs_r);
            let unit_a = T::from_real_imag(a.real() / abs_a, a.imag() / abs_a);
            assert!(
                Scalar::abs(unit_r - unit_a) <= tol,
                "phase of r differs from phase of a for a={a:?}, b={b:?}: r={r:?}"
            );
        }
    }

    #[test]
    fn test_issue_12_crotg_both_zero_returns_identity_rotation() {
        // Netlib 3.12 checks `g == 0` first: c = 1, s = 0, r = f.
        let r64 = crotg(c64(0.0, 0.0), c64(0.0, 0.0));
        assert_eq!((r64.c, r64.s, r64.r), (1.0, c64(0.0, 0.0), c64(0.0, 0.0)));

        let z32 = Complex32::new(0.0, 0.0);
        let r32 = crotg(z32, z32);
        assert_eq!((r32.c, r32.s, r32.r), (1.0f32, z32, z32));
    }

    #[test]
    fn test_issue_12_crotg_a_zero_uses_modulus_and_conjugate() {
        // a = 0, b = 3+4i: Netlib 3.12 gives r = 5 (real), c = 0,
        // s = conj(b)/|b| = 0.6 - 0.8i (the pre-3.10 code returned r = b,
        // s = 1).
        let r64 = crotg(c64(0.0, 0.0), c64(3.0, 4.0));
        assert_eq!(r64.c, 0.0);
        assert_eq!(r64.r, c64(5.0, 0.0));
        assert_eq!(r64.s, c64(0.6, -0.8));

        let r32 = crotg(Complex32::new(0.0, 0.0), Complex32::new(3.0, 4.0));
        assert_eq!(r32.c, 0.0f32);
        assert_eq!(r32.r, Complex32::new(5.0, 0.0));
        assert_eq!(r32.s, Complex32::new(0.6, -0.8));

        // Purely imaginary / purely real b take the exact |Im| / |Re| route.
        let imag = crotg(c64(0.0, 0.0), c64(0.0, -2.0));
        assert_eq!(
            (imag.c, imag.r, imag.s),
            (0.0, c64(2.0, 0.0), c64(0.0, 1.0))
        );
        let real = crotg(c64(0.0, 0.0), c64(-4.0, 0.0));
        assert_eq!(
            (real.c, real.r, real.s),
            (0.0, c64(4.0, 0.0), c64(-1.0, 0.0))
        );

        // Huge b with both components non-zero takes the scaled branch.
        let big = crotg(c64(0.0, 0.0), c64(1e300, 1e300));
        assert_eq!(big.c, 0.0);
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!((big.r.re - sqrt2 * 1e300).abs() <= 4.0 * f64::EPSILON * sqrt2 * 1e300);
        assert_eq!(big.r.im, 0.0);
        assert!((big.s - c64(1.0 / sqrt2, -1.0 / sqrt2)).norm() < 1e-15);
    }

    #[test]
    fn test_issue_12_crotg_b_zero_returns_a_exactly() {
        let a = c64(-2.5, 7.0);
        let res = crotg(a, c64(0.0, 0.0));
        assert_eq!((res.c, res.s, res.r), (1.0, c64(0.0, 0.0), a));
    }

    #[test]
    fn test_issue_12_crotg_c32_large_inputs_do_not_overflow() {
        // Reporter's case: c32, a = 1e20, b = 1e20(1+i). The pre-fix code
        // produced r = c = s = NaN (complex/complex division squares |divisor|).
        let inv_sqrt3 = 1.0f32 / 3.0f32.sqrt();
        let res = crotg(Complex32::new(1e20, 0.0), Complex32::new(1e20, 1e20));
        assert!(
            (res.r.re - 1.732_050_8e20).abs() <= 1e-6 * 1.732_050_8e20,
            "r={:?}",
            res.r
        );
        assert!(res.r.im.abs() <= 1e-6 * 1e20, "r={:?}", res.r);
        assert!((res.c - inv_sqrt3).abs() <= 1e-6, "c={}", res.c);
        assert!((res.s.re - inv_sqrt3).abs() <= 1e-6, "s={:?}", res.s);
        assert!((res.s.im + inv_sqrt3).abs() <= 1e-6, "s={:?}", res.s);

        // At 1e19 the pre-fix code returned r = 0, c = inf.
        let tol = 32.0 * f32::EPSILON;
        assert_crotg_identities(Complex32::new(1e19, 0.0), Complex32::new(1e19, 1e19), tol);
        assert_crotg_identities(Complex32::new(1e20, 0.0), Complex32::new(1e20, 1e20), tol);
        // No overflow up to ~f32::MAX / 4 per component.
        let quarter = f32::MAX / 4.0;
        assert_crotg_identities(
            Complex32::new(quarter, 0.0),
            Complex32::new(0.0, quarter),
            tol,
        );
        assert_crotg_identities(
            Complex32::new(-quarter, quarter),
            Complex32::new(quarter, -quarter),
            tol,
        );
    }

    #[test]
    fn test_issue_12_crotg_c64_large_inputs_do_not_overflow() {
        // The pre-fix code broke from ~1e154 upwards in double precision.
        let tol = 32.0 * f64::EPSILON;
        for &m in &[1e154, 1e200, 1e300] {
            assert_crotg_identities(c64(m, 0.0), c64(m, m), tol);
            assert_crotg_identities(c64(-m, m), c64(m, -m), tol);
        }
        let inv_sqrt3 = 1.0 / 3.0f64.sqrt();
        let res = crotg(c64(1e300, 0.0), c64(1e300, 1e300));
        assert!((res.c - inv_sqrt3).abs() <= 1e-15);
        assert!((res.r.re - 3.0f64.sqrt() * 1e300).abs() <= 1e-15 * 3.0f64.sqrt() * 1e300);
        assert!((res.s - c64(inv_sqrt3, -inv_sqrt3)).norm() <= 1e-15);
    }

    /// Unit-modulus phases used by the magnitude sweeps below.
    const SWEEP_PHASES: [(f64, f64); 5] = [
        (1.0, 0.0),
        (0.0, 1.0),
        (-0.6, 0.8),
        (0.28, -0.96),
        (-1.0, 0.0),
    ];

    #[test]
    fn test_issue_12_crotg_identity_sweep_c32() {
        let tol = 32.0 * f32::EPSILON;
        let mags: Vec<f32> = (-6..=6).map(|k| 10f32.powi(5 * k)).collect(); // 1e-30 ..= 1e30
        for &ma in &mags {
            for &mb in &mags {
                for &(pa_re, pa_im) in &SWEEP_PHASES {
                    for &(pb_re, pb_im) in SWEEP_PHASES.iter().rev() {
                        let a = Complex32::new(ma * pa_re as f32, ma * pa_im as f32);
                        let b = Complex32::new(mb * pb_re as f32, mb * pb_im as f32);
                        assert_crotg_identities(a, b, tol);
                    }
                }
            }
        }
    }

    #[test]
    fn test_issue_12_crotg_identity_sweep_c64() {
        let tol = 32.0 * f64::EPSILON;
        let mags: Vec<f64> = (-6..=6).map(|k| 10f64.powi(50 * k)).collect(); // 1e-300 ..= 1e300
        for &ma in &mags {
            for &mb in &mags {
                for &(pa_re, pa_im) in &SWEEP_PHASES {
                    for &(pb_re, pb_im) in SWEEP_PHASES.iter().rev() {
                        let a = c64(ma * pa_re, ma * pa_im);
                        let b = c64(mb * pb_re, mb * pb_im);
                        assert_crotg_identities(a, b, tol);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Complex rot (zdrot) -- real-valued rotation applied to complex data
    // =========================================================================

    #[test]
    fn test_zdrot_round_trip_zeroes_second_component() {
        use crate::level1::rotg;

        // Generate a REAL rotation that eliminates the pair (3, 4) -> (5, 0).
        let rg = rotg(3.0f64, 4.0);

        // Apply the identical real (c, s) to complex vectors that share a
        // common phase factor -- the rotation must still zero the second
        // component, scaled by that phase.
        let phase = c64(0.6, 0.8);
        let mut x = [phase * c64(3.0, 0.0)];
        let mut y = [phase * c64(4.0, 0.0)];

        zdrot(rg.c, rg.s, &mut x, &mut y);

        let expected_x = phase * c64(5.0, 0.0);
        assert!((x[0] - expected_x).norm() < 1e-9);
        assert!(y[0].norm() < 1e-9, "y[0] should vanish, got {:?}", y[0]);
    }

    #[test]
    fn test_zdrot_identity() {
        let mut x = [c64(1.0, 2.0), c64(3.0, -1.0)];
        let mut y = [c64(4.0, 0.0), c64(-2.0, 5.0)];
        let x_orig = x;
        let y_orig = y;

        zdrot(1.0, 0.0, &mut x, &mut y);

        for i in 0..2 {
            assert!((x[i] - x_orig[i]).norm() < 1e-12);
            assert!((y[i] - y_orig[i]).norm() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "Vector lengths must match")]
    fn test_zdrot_length_mismatch_panics() {
        let mut x = [c64(1.0, 0.0)];
        let mut y = [c64(1.0, 0.0), c64(2.0, 0.0)];
        zdrot(1.0, 0.0, &mut x, &mut y);
    }

    // =========================================================================
    // Issue #15: public cabs1 / scabs1 / dcabs1, component-wise zdrot
    // =========================================================================

    #[test]
    fn test_issue_15_cabs1_matches_reference_definition() {
        // DCABS1 = ABS(DBLE(Z)) + ABS(DIMAG(Z)) -- not the modulus.
        assert_eq!(cabs1(c64(3.0, -4.0)), 7.0);
        assert_eq!(dcabs1(c64(-1.0, 2.0)), 3.0);
        assert_eq!(scabs1(Complex32::new(1.5, -2.5)), 4.0f32);
        assert_eq!(cabs1(c64(-0.0, 0.0)), 0.0);
        assert!(cabs1(c64(f64::NAN, 1.0)).is_nan());
        assert_eq!(cabs1(c64(f64::NEG_INFINITY, 1.0)), f64::INFINITY);
        // Real scalars reduce to |x|.
        assert_eq!(cabs1(-2.5f64), 2.5);
        assert_eq!(cabs1(4.0f32), 4.0);
    }

    #[test]
    fn test_issue_15_zdrot_scales_components_without_nan_cross_terms() {
        // zdrot.f computes C*ZX + S*ZY with real C, S, i.e. per component:
        // x' = (0.6*inf + 0.8*1, 0.6*0 + 0.8*1) = (inf, 0.8),
        // y' = (0.6*1 - 0.8*inf, 0.6*1 - 0.8*0) = (-inf, 0.6).
        // Multiplying by the complex (0.6 + 0i) would form 0*inf = NaN.
        let mut x = [c64(f64::INFINITY, 0.0)];
        let mut y = [c64(1.0, 1.0)];
        zdrot(0.6, 0.8, &mut x, &mut y);
        assert_eq!(x[0], c64(f64::INFINITY, 0.8));
        assert_eq!(y[0], c64(f64::NEG_INFINITY, 0.6));

        // Finite data: identical to the textbook complex formula.
        let x0 = [c64(1.0, 2.0), c64(-3.0, 0.5)];
        let y0 = [c64(0.25, -1.0), c64(4.0, 4.0)];
        let (mut x, mut y) = (x0, y0);
        zdrot(0.28, -0.96, &mut x, &mut y);
        for i in 0..2 {
            let ex = c64(0.28, 0.0) * x0[i] + c64(-0.96, 0.0) * y0[i];
            let ey = c64(0.28, 0.0) * y0[i] - c64(-0.96, 0.0) * x0[i];
            assert!((x[i] - ex).norm() < 1e-15 && (y[i] - ey).norm() < 1e-15);
        }
    }
}
