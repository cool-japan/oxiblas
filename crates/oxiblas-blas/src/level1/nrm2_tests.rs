//! Regression tests for the f64/f32 SIMD `nrm2` kernels (reached for `n >= 64`)
//! on vectors whose largest magnitude is so small that `1 / max|x|` overflows to
//! `+Inf`: `max|x| < 2^-1024 ≈ 5.6e-309` for f64 and `max|x| < 2^-128 ≈ 2.9e-39`
//! for f32, i.e. deep in the subnormal range. The kernels scale every element by
//! `1 / max|x|`; with an infinite scale they returned `+Inf` (every element
//! non-zero) or NaN (`0 * Inf` for a zero element) instead of the tiny norm.
//!
//! Expected values are computed independently of the kernels:
//!
//! * f64 data are integers `m_i` times an exact power of two `2^k`, so the norm
//!   is `sqrt(Σ m_i²) · 2^k`: the integer sum of squares is exact, and rescaling
//!   by `2^k` is exact up to one final rounding into the subnormal range.
//! * f32 subnormals are exact (normal) f64 numbers, so the f32 reference is an
//!   f64 sum of squares rounded once to f32.

use super::*;
use crate::complex_interleaved::{nrm2_interleaved_f32, nrm2_interleaved_f64};

/// `2^e` for every exponent of the f64 range, subnormal ones (`e < -1022`)
/// included, built from its bit pattern (exact).
fn pow2_f64(e: i32) -> f64 {
    assert!((-1074..=1023).contains(&e), "2^{e} is not an f64");
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// `2^e` for every exponent of the f32 range, subnormal ones (`e < -126`)
/// included, built from its bit pattern (exact).
fn pow2_f32(e: i32) -> f32 {
    assert!((-149..=127).contains(&e), "2^{e} is not an f32");
    if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (e + 149))
    }
}

/// `got` is finite and within `rel·|expected|` plus two subnormal ulps.
fn assert_close_f64(got: f64, expected: f64, rel: f64, what: &str) {
    let tol = rel * expected.abs() + 2.0 * f64::from_bits(1);
    assert!(
        got.is_finite() && (got - expected).abs() <= tol,
        "{what}: got {got:e}, expected {expected:e}"
    );
}

/// f32 counterpart of [`assert_close_f64`].
fn assert_close_f32(got: f32, expected: f32, rel: f32, what: &str) {
    let tol = rel * expected.abs() + 2.0 * f32::from_bits(1);
    assert!(
        got.is_finite() && (got - expected).abs() <= tol,
        "{what}: got {got:e}, expected {expected:e}"
    );
}

/// Integer magnitudes below `bound` with alternating signs and a zero at every
/// seventh position. With `bound <= 2^20` and `n <= 100`, `Σ m_i²` is an exact
/// f64 integer.
fn mixed_integers(n: usize, bound: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i % 7 == 3 {
                0.0
            } else {
                let m = ((i * 7919 + 13) % (bound - 1) + 1) as f64;
                if i % 2 == 0 { m } else { -m }
            }
        })
        .collect()
}

/// `sqrt(Σ m_i²) · 2^k` for the integer vector `m`, rounded once at the end.
fn scaled_reference_norm(m: &[f64], k: i32) -> f64 {
    let sum_sq: f64 = m.iter().map(|v| v * v).sum();
    assert!(
        sum_sq < 9.0e15,
        "the integer sum of squares must stay exact"
    );
    sum_sq.sqrt() * pow2_f64(k)
}

/// Reference norm of an f32 vector, evaluated in f64 (every f32, subnormals
/// included, is an exact normal f64) and rounded once to f32.
fn widened_reference_norm(x: &[f32]) -> f32 {
    let sum_sq: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    sum_sq.sqrt() as f32
}

/// `n` copies of one subnormal `v`: the norm is exactly `v · sqrt(n)` for a
/// square `n`.
#[test]
fn test_nrm2_f64_simd_all_subnormal_vector() {
    let v = 1e-310f64;
    assert!(v < f64::MIN_POSITIVE && (1.0 / v).is_infinite());
    for (n, root) in [(64usize, 8.0f64), (100, 10.0), (256, 16.0)] {
        let x = vec![v; n];
        let expected = v * root;
        assert_close_f64(nrm2_f64(&x), expected, 0.0, &format!("n={n}"));
        let negated: Vec<f64> = x.iter().map(|&e| -e).collect();
        assert_close_f64(nrm2_f64(&negated), expected, 0.0, &format!("-x n={n}"));
    }
}

/// Powers of two around the overflow threshold of `1 / max|x|`: `2^-1023` still
/// has a finite reciprocal (the SIMD scaling path), `2^-1024` and below do not.
/// The norm of 100 copies of `2^e` is exactly `10 · 2^e`.
#[test]
fn test_nrm2_f64_simd_threshold_powers_of_two() {
    for e in [-1022, -1023, -1024, -1025, -1040, -1060, -1074] {
        let v = pow2_f64(e);
        let x = vec![v; 100];
        assert_close_f64(nrm2_f64(&x), 10.0 * v, 0.0, &format!("100 x 2^{e}"));
    }
}

/// Mixed magnitudes, signs and zeros, all below `2^-1030`. Before the fix the
/// zeros turned into `0 · Inf = NaN`.
#[test]
fn test_nrm2_f64_simd_mixed_tiny_vector() {
    let k = -1050;
    for n in [64usize, 100] {
        let m = mixed_integers(n, 1 << 20);
        let x: Vec<f64> = m.iter().map(|&v| v * pow2_f64(k)).collect();
        let max_abs = x.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(max_abs < pow2_f64(-1030) && (1.0 / max_abs).is_infinite());
        let expected = scaled_reference_norm(&m, k);
        assert_close_f64(nrm2_f64(&x), expected, 1e-13, &format!("n={n}"));
        // The generic scalar `nrm2` agrees (control).
        assert_close_f64(nrm2(&x), expected, 1e-13, &format!("generic n={n}"));
    }

    // Pythagorean pairs: 16 x (3t, 4t) padded with zeros has the exact norm 20t.
    let t = pow2_f64(-1040);
    let mut x = vec![0.0f64; 100];
    for pair in 0..16 {
        x[4 * pair + 1] = 3.0 * t;
        x[4 * pair + 2] = -4.0 * t;
    }
    assert_close_f64(nrm2_f64(&x), 20.0 * t, 0.0, "pythagorean pairs");
}

/// f32 counterpart of the all-subnormal case (`1e-40 < 2^-128`).
#[test]
fn test_nrm2_f32_simd_all_subnormal_vector() {
    let v = 1e-40f32;
    assert!(v < f32::MIN_POSITIVE && (1.0 / v).is_infinite());
    for (n, root) in [(64usize, 8.0f32), (100, 10.0), (256, 16.0)] {
        let x = vec![v; n];
        let expected = v * root;
        assert_close_f32(nrm2_f32(&x), expected, 0.0, &format!("n={n}"));
        let negated: Vec<f32> = x.iter().map(|&e| -e).collect();
        assert_close_f32(nrm2_f32(&negated), expected, 0.0, &format!("-x n={n}"));
    }
}

/// f32 powers of two around the threshold: `2^-127` has a finite reciprocal,
/// `2^-128` and below do not.
#[test]
fn test_nrm2_f32_simd_threshold_powers_of_two() {
    for e in [-126, -127, -128, -129, -140, -149] {
        let v = pow2_f32(e);
        let x = vec![v; 100];
        assert_close_f32(nrm2_f32(&x), 10.0 * v, 0.0, &format!("100 x 2^{e}"));
    }
}

/// f32 mixed magnitudes, signs and zeros, all below `2^-130`.
#[test]
fn test_nrm2_f32_simd_mixed_tiny_vector() {
    for n in [64usize, 100] {
        let x: Vec<f32> = mixed_integers(n, 1 << 10)
            .iter()
            .map(|&v| v as f32 * pow2_f32(-140))
            .collect();
        let max_abs = x.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(max_abs < pow2_f32(-130) && (1.0 / max_abs).is_infinite());
        let expected = widened_reference_norm(&x);
        assert_close_f32(nrm2_f32(&x), expected, 3e-5, &format!("n={n}"));
        assert_close_f32(nrm2(&x), expected, 3e-5, &format!("generic n={n}"));
    }
}

/// The complex entry points view interleaved storage as `2n` reals and reach
/// the same kernels: `dznrm2` / `nrm2_interleaved_f64` (f64) and `scnrm2` /
/// `nrm2_interleaved_f32` (f32), with 50 complex elements (100 components).
#[test]
fn test_nrm2_complex_simd_tiny_vector() {
    let k = -1050;
    let m = mixed_integers(100, 1 << 20);
    let flat: Vec<f64> = m.iter().map(|&v| v * pow2_f64(k)).collect();
    let z: Vec<Complex64> = flat
        .chunks_exact(2)
        .map(|p| Complex64::new(p[0], p[1]))
        .collect();
    let expected = scaled_reference_norm(&m, k);
    assert_close_f64(dznrm2(&z), expected, 1e-13, "dznrm2");
    assert_close_f64(
        nrm2_interleaved_f64(&flat).expect("even length"),
        expected,
        1e-13,
        "nrm2_interleaved_f64",
    );
    assert_close_f64(cnrm2(&z), expected, 1e-13, "cnrm2 (control)");

    let flat32: Vec<f32> = mixed_integers(100, 1 << 10)
        .iter()
        .map(|&v| v as f32 * pow2_f32(-140))
        .collect();
    let z32: Vec<Complex32> = flat32
        .chunks_exact(2)
        .map(|p| Complex32::new(p[0], p[1]))
        .collect();
    let expected32 = widened_reference_norm(&flat32);
    assert_close_f32(scnrm2(&z32), expected32, 3e-5, "scnrm2");
    assert_close_f32(
        nrm2_interleaved_f32(&flat32).expect("even length"),
        expected32,
        3e-5,
        "nrm2_interleaved_f32",
    );
    assert_close_f32(cnrm2(&z32), expected32, 3e-5, "cnrm2 (control)");

    // An all-subnormal complex vector of identical entries: |z|·sqrt(64).
    let w = vec![Complex64::new(3e-310, -4e-310); 64];
    assert_close_f64(dznrm2(&w), 5e-310 * 8.0, 1e-13, "dznrm2 constant");
    let w32 = vec![Complex32::new(3e-40, -4e-40); 64];
    assert_close_f32(scnrm2(&w32), 5e-40 * 8.0, 3e-5, "scnrm2 constant");
}

/// Control: NaN, infinities and all-zero vectors keep their results when the
/// rest of the vector is tiny (NaN in a SIMD chunk and in the scalar remainder,
/// where the first pass does not see it).
#[test]
fn test_nrm2_simd_tiny_vector_keeps_nan_inf_zero() {
    for pos in [5usize, 99] {
        let mut x = vec![1e-310f64; 100];
        x[pos] = f64::NAN;
        assert!(nrm2_f64(&x).is_nan(), "f64 NaN at {pos}");
        x[pos] = f64::NEG_INFINITY;
        assert_eq!(nrm2_f64(&x), f64::INFINITY, "f64 -Inf at {pos}");

        let mut x = vec![1e-40f32; 100];
        x[pos] = f32::NAN;
        assert!(nrm2_f32(&x).is_nan(), "f32 NaN at {pos}");
        x[pos] = f32::INFINITY;
        assert_eq!(nrm2_f32(&x), f32::INFINITY, "f32 Inf at {pos}");
    }
    assert_eq!(nrm2_f64(&[0.0f64; 100]).to_bits(), 0.0f64.to_bits());
    assert_eq!(nrm2_f32(&[-0.0f32; 100]).to_bits(), 0.0f32.to_bits());
}
