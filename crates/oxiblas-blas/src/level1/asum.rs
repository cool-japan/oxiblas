//! ASUM: Sum of absolute values (L1 norm)
//!
//! Computes ||x||_1 = Σ |x\[i\]|
//!
//! This module also hosts the other missing complex Level-1 BLAS routines
//! flagged by the production audit (they are grouped here, rather than in
//! `rot.rs`, per the audit's file scope for this fix):
//!
//! - [`casum`] / [`scasum`] / [`dzasum`]: complex `ASUM` using the BLAS
//!   `cabs1` metric (`|Re| + |Im|`, NOT the complex modulus).
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

/// Computes `|Re(z)| + |Im(z)|`, the BLAS "cabs1" metric.
///
/// This is the comparison/summation metric used internally by the complex
/// Level-1 BLAS routines `SCASUM`/`DZASUM` (sum, see [`casum`]) and
/// `ICAMAX`/`IZAMAX` (max-index search, see [`crate::level1::iamax`]).
/// It is **not** the complex modulus `sqrt(re^2 + im^2)`; reference BLAS
/// deliberately uses the cheaper L1-style metric. For real scalars
/// (`im == 0`) this reduces to plain `|re|`.
pub(crate) fn cabs1<T: Scalar>(z: T) -> T::Real {
    Scalar::abs(z.real()) + Scalar::abs(z.imag())
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
    /// The rotated value r (complex), carrying the phase of the dominant
    /// input.
    pub r: T,
    /// Cosine of the rotation (always real).
    pub c: T::Real,
    /// Sine of the rotation (complex).
    pub s: T,
}

/// Generates a Givens plane rotation for complex scalars (`ZROTG`/`CROTG`).
///
/// Given complex scalars `a` and `b`, computes a real cosine `c` and a
/// complex sine `s` such that
/// ```text
/// [  c         s  ] [ a ]   [ r ]
/// [ -conj(s)   c  ] [ b ] = [ 0 ]
/// ```
/// This is a direct translation of the classic reference BLAS `zrotg.f` /
/// `crotg.f` algorithm (Netlib), including the `a == 0` special case:
/// `c = 0`, `s = 1 + 0i`, `r = b`.
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
    let abs_a = Scalar::abs(a);

    if abs_a.is_zero() {
        return ComplexRotgResult {
            r: b,
            c: T::Real::zero(),
            s: T::from_real(T::Real::one()),
        };
    }

    let abs_b = Scalar::abs(b);
    let scale = abs_a + abs_b;
    let scale_c = T::from_real(scale);

    // |r| = scale * sqrt(|a/scale|^2 + |b/scale|^2), computed in a scaled
    // fashion to avoid premature overflow, matching Netlib zrotg.f.
    let abs_a_scaled = Scalar::abs(a / scale_c);
    let abs_b_scaled = Scalar::abs(b / scale_c);
    let norm = scale * (abs_a_scaled * abs_a_scaled + abs_b_scaled * abs_b_scaled).sqrt();

    let alpha = a / T::from_real(abs_a);
    let c = abs_a / norm;
    let s = (alpha * b.conj()) / T::from_real(norm);
    let r = alpha * T::from_real(norm);

    ComplexRotgResult { r, c, s }
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

    let c_t = T::from_real(c);
    let s_t = T::from_real(s);

    for i in 0..x.len() {
        let xi = x[i];
        let yi = y[i];
        x[i] = c_t * xi + s_t * yi;
        y[i] = c_t * yi - s_t * xi;
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
        // Reference zrotg.f special case: a == 0 => c=0, s=1, r=b.
        let a = c64(0.0, 0.0);
        let b = c64(3.0, 4.0);
        let result = crotg(a, b);

        assert_eq!(result.c, 0.0);
        assert!((result.s - c64(1.0, 0.0)).norm() < 1e-12);
        assert!((result.r - b).norm() < 1e-12);
    }

    #[test]
    fn test_crotg_both_zero() {
        let a = c64(0.0, 0.0);
        let b = c64(0.0, 0.0);
        let result = crotg(a, b);

        assert_eq!(result.c, 0.0);
        assert!((result.s - c64(1.0, 0.0)).norm() < 1e-12);
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
}
