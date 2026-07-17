//! Tests for scalar traits and implementations.

#[allow(unused_imports)]
use super::*;

#[test]
fn test_f32_scalar() {
    let x: f32 = 3.0;
    assert_eq!(x.abs(), 3.0);
    assert_eq!(x.conj(), 3.0);
    assert!(f32::is_real());
    assert_eq!(x.real(), 3.0);
    assert_eq!(x.imag(), 0.0);
    assert_eq!(x.abs_sq(), 9.0);
}

#[test]
fn test_f64_scalar() {
    let x: f64 = -4.0;
    assert_eq!(x.abs(), 4.0);
    assert_eq!(x.conj(), -4.0);
    assert!(f64::is_real());
    assert_eq!(x.real(), -4.0);
    assert_eq!(x.imag(), 0.0);
    assert_eq!(x.abs_sq(), 16.0);
}

#[test]
fn test_complex32_scalar() {
    use num_complex::Complex32;
    let z = Complex32::new(3.0, 4.0);
    assert!((z.abs() - 5.0).abs() < 1e-6);
    assert_eq!(z.conj(), Complex32::new(3.0, -4.0));
    assert!(!Complex32::is_real());
    assert_eq!(z.real(), 3.0);
    assert_eq!(z.imag(), 4.0);
    assert!((z.abs_sq() - 25.0).abs() < 1e-6);
}

#[test]
fn test_complex64_scalar() {
    use num_complex::Complex64;
    let z = Complex64::new(3.0, 4.0);
    assert!((z.abs() - 5.0).abs() < 1e-12);
    assert_eq!(z.conj(), Complex64::new(3.0, -4.0));
    assert!(!Complex64::is_real());
    assert_eq!(z.real(), 3.0);
    assert_eq!(z.imag(), 4.0);
    assert!((z.abs_sq() - 25.0).abs() < 1e-12);
}

#[test]
fn test_field_operations() {
    let a: f64 = 2.0;
    let b: f64 = 3.0;
    assert_eq!(a.mul_conj(b), 6.0);
    assert_eq!(a.conj_mul(b), 6.0);
    assert!((a.recip() - 0.5).abs() < 1e-12);
    assert!((a.powi(3) - 8.0).abs() < 1e-12);
}

#[test]
fn test_complex_field_operations() {
    use num_complex::Complex64;
    let a = Complex64::new(1.0, 2.0);
    let b = Complex64::new(3.0, 4.0);

    // mul_conj: a * conj(b) = (1+2i) * (3-4i) = 3 - 4i + 6i - 8i^2 = 3 + 2i + 8 = 11 + 2i
    let mc = a.mul_conj(b);
    assert!((mc.re - 11.0).abs() < 1e-12);
    assert!((mc.im - 2.0).abs() < 1e-12);

    // conj_mul: conj(a) * b = (1-2i) * (3+4i) = 3 + 4i - 6i - 8i^2 = 3 - 2i + 8 = 11 - 2i
    let cm = a.conj_mul(b);
    assert!((cm.re - 11.0).abs() < 1e-12);
    assert!((cm.im - (-2.0)).abs() < 1e-12);
}

#[test]
#[cfg(feature = "f16")]
fn test_f16_scalar() {
    use half::f16;
    let x = f16::from_f32(3.0);
    let y = f16::from_f32(4.0);
    let result = x + y;
    assert!((result.to_f32() - 7.0).abs() < 0.01);
}

#[test]
#[cfg(feature = "f128")]
fn test_f128_scalar() {
    use crate::scalar::Real as ScalarReal;

    // Test basic arithmetic
    let x = QuadFloat::from(3.0);
    let y = QuadFloat::from(4.0);
    let result = x + y;
    assert!(Scalar::abs(result - QuadFloat::from(7.0)) < QuadFloat::from(1e-28));

    // Test sqrt operation
    let a = QuadFloat::from(2.0);
    let epsilon = QuadFloat::from(1e-28);
    let sqrt_a = ScalarReal::sqrt(a);
    assert!(Scalar::abs(sqrt_a * sqrt_a - a) < epsilon);

    // Test other operations
    let b = QuadFloat::from(5.0);
    let c = QuadFloat::from(3.0);
    assert!(Scalar::abs(b / c - QuadFloat::from(5.0 / 3.0)) < QuadFloat::from(1e-15));
}

#[cfg(feature = "f128")]
#[test]
fn test_f128_rem_truncated() {
    // `%` on QuadFloat must follow Rust's truncated-division semantics (result
    // carries the sign of the dividend), NOT floored modulo (sign of divisor).
    use num_traits::Float as NumF;
    let qabs = |x: QuadFloat| QuadFloat::from(x.inner().abs());
    let tol = QuadFloat::from(1e-28);

    // Operands exactly representable => remainder is exact at DD precision.
    let cases = [
        (7.0_f64, 2.5_f64, 2.0_f64),
        (-7.0, 2.5, -2.0), // truncated: trunc(-2.8) = -2 => -7 - (-5) = -2
        (7.0, -2.5, 2.0),
        (-7.0, -2.5, -2.0),
        (5.0, 2.0, 1.0),
        (-5.0, 2.0, -1.0), // floored modulo would (wrongly) give +1.0
        (5.0, -2.0, 1.0),
        (-5.0, -2.0, -1.0),
    ];
    for (a, b, expected) in cases {
        let r = QuadFloat::from(a) % QuadFloat::from(b);
        assert!(
            qabs(r - QuadFloat::from(expected)) < tol,
            "{a} % {b} = {r} (expected {expected})"
        );
        // Sign of the result must match the dividend.
        if r != QuadFloat::from(0.0) {
            assert_eq!(
                r.inner().is_sign_negative(),
                a < 0.0,
                "{a} % {b} = {r} has wrong sign"
            );
        }
    }

    // Defining invariants of the truncated remainder for a value whose quotient
    // is not exactly representable (1/0.1) — exercises the boundary correction:
    // sign(r) == sign(a), |r| < |b|, and r agrees with f64's exact fmod.
    let a = 1.0_f64;
    let b = 0.1_f64;
    let r = QuadFloat::from(a) % QuadFloat::from(b);
    assert!(
        qabs(r) < QuadFloat::from(QuadFloat::from(b).inner().abs()),
        "|1.0 % 0.1| = {r} must be < 0.1"
    );
    assert!(!r.inner().is_sign_negative(), "1.0 % 0.1 must be positive");
    assert!(
        (r.inner().hi() - (a % b)).abs() < 1e-13,
        "1.0 % 0.1 = {r} disagrees with f64 fmod {}",
        a % b
    );

    // Special values, matching f64 `%`.
    let inf: QuadFloat = NumF::infinity();
    assert!((QuadFloat::from(1.0) % QuadFloat::from(0.0)).is_nan());
    assert!((inf % QuadFloat::from(2.0)).is_nan());
    let finite_mod_inf = QuadFloat::from(3.0) % inf;
    assert!(qabs(finite_mod_inf - QuadFloat::from(3.0)) < tol);
}

#[cfg(feature = "f128")]
#[test]
fn test_f128_rounding_double_double() {
    // floor/ceil/round/trunc/fract must respect the low (correction) word, not
    // just the high f64 component. We build values that straddle an integer or
    // half-integer boundary by an amount far below one f64 ULP, so the high word
    // alone gives the wrong answer.
    use num_traits::Float as NumF;
    let qabs = |x: QuadFloat| QuadFloat::from(x.inner().abs());
    let tol = QuadFloat::from(1e-28);
    let tiny = QuadFloat::from(1e-30);
    let int_of = |x: QuadFloat| x.inner().hi(); // integer results have lo == 0

    // Value just BELOW the integer 3 (hi == 3.0, lo < 0): true value is < 3.
    let below_3 = QuadFloat::from(3.0) - tiny;
    assert_eq!(int_of(NumF::floor(below_3)), 2.0, "floor(3-eps) must be 2");
    assert_eq!(int_of(NumF::ceil(below_3)), 3.0, "ceil(3-eps) must be 3");
    assert_eq!(int_of(NumF::round(below_3)), 3.0, "round(3-eps) must be 3");
    assert_eq!(int_of(NumF::trunc(below_3)), 2.0, "trunc(3-eps) must be 2");
    // fract(3-eps) = (3-eps) - 2 = 1 - eps.
    assert!(qabs(NumF::fract(below_3) - (QuadFloat::from(1.0) - tiny)) < tol);

    // Value just ABOVE the integer 3 (hi == 3.0, lo > 0).
    let above_3 = QuadFloat::from(3.0) + tiny;
    assert_eq!(int_of(NumF::floor(above_3)), 3.0);
    assert_eq!(int_of(NumF::ceil(above_3)), 4.0);
    assert_eq!(int_of(NumF::round(above_3)), 3.0);
    assert_eq!(int_of(NumF::trunc(above_3)), 3.0);
    assert!(qabs(NumF::fract(above_3) - tiny) < tol);

    // Negative value just BELOW -3 in magnitude (hi == -3.0, lo < 0): < -3.
    let below_neg3 = QuadFloat::from(-3.0) - tiny;
    assert_eq!(int_of(NumF::floor(below_neg3)), -4.0);
    assert_eq!(int_of(NumF::ceil(below_neg3)), -3.0);
    assert_eq!(int_of(NumF::round(below_neg3)), -3.0);
    assert_eq!(int_of(NumF::trunc(below_neg3)), -3.0, "trunc toward zero");

    // Negative value just ABOVE -3 (hi == -3.0, lo > 0): value is -2.999...
    let above_neg3 = QuadFloat::from(-3.0) + tiny;
    assert_eq!(int_of(NumF::floor(above_neg3)), -3.0);
    assert_eq!(int_of(NumF::ceil(above_neg3)), -2.0);
    assert_eq!(int_of(NumF::round(above_neg3)), -3.0);
    assert_eq!(int_of(NumF::trunc(above_neg3)), -2.0, "trunc toward zero");

    // Half-integer ties broken by the low word (hi is the half-integer 2.5).
    assert_eq!(int_of(NumF::round(QuadFloat::from(2.5))), 3.0, "tie away from 0");
    assert_eq!(
        int_of(NumF::round(QuadFloat::from(2.5) - tiny)),
        2.0,
        "2.5-eps rounds down"
    );
    assert_eq!(
        int_of(NumF::round(QuadFloat::from(2.5) + tiny)),
        3.0,
        "2.5+eps rounds up"
    );
    assert_eq!(
        int_of(NumF::round(QuadFloat::from(-2.5) + tiny)),
        -2.0,
        "-2.5+eps rounds toward zero"
    );
    assert_eq!(
        int_of(NumF::round(QuadFloat::from(-2.5) - tiny)),
        -3.0,
        "-2.5-eps rounds away"
    );

    // Large-magnitude ties where the fractional 0.5 lives in the LOW word and
    // the integer part is in the HIGH word (verified normalizations below).
    let two_pow_52 = QuadFloat::from((2.0_f64).powi(52));
    let two_pow_53 = QuadFloat::from((2.0_f64).powi(53));

    // 2^52 + 0.5 => (hi=2^52, lo=+0.5): low word's sign matches the whole,
    // so the tie rounds away from zero to 2^52 + 1.
    let tie_high = two_pow_52 + QuadFloat::from(0.5);
    assert!(
        qabs(NumF::round(tie_high) - (two_pow_52 + QuadFloat::from(1.0))) < QuadFloat::from(0.5),
        "round(2^52 + 0.5) must be 2^52 + 1"
    );

    // 2^53 - 0.5 => (hi=2^53, lo=-0.5): the low word's local away-from-zero
    // direction (down) DISAGREES with the positive whole's (up). The whole value
    // is the genuine tie 2^53 - 0.5 and must round away from zero to 2^53, NOT
    // down to 2^53 - 1. This exercises the sign-flip tie correction.
    let tie_flip = two_pow_53 - QuadFloat::from(0.5);
    assert!(
        qabs(NumF::round(tie_flip) - two_pow_53) < QuadFloat::from(0.5),
        "round(2^53 - 0.5) must be 2^53"
    );

    // 2^52 - 0.5 is exactly representable, so it normalizes to a half-integer
    // high word (lo == 0); the non-integer-hi tie path must round it away to
    // 2^52.
    let tie_half_hi = two_pow_52 - QuadFloat::from(0.5);
    assert!(
        qabs(NumF::round(tie_half_hi) - two_pow_52) < QuadFloat::from(0.5),
        "round(2^52 - 0.5) must be 2^52"
    );
}

#[cfg(feature = "f128")]
#[test]
fn test_f128_hypot_no_overflow() {
    // hypot must not overflow when the true result is finite (scale-then-sqrt).
    use num_traits::Float as NumF;
    let qabs = |x: QuadFloat| QuadFloat::from(x.inner().abs());

    // 3-4-5 sanity check.
    let h = NumF::hypot(QuadFloat::from(3.0), QuadFloat::from(4.0));
    assert!(qabs(h - QuadFloat::from(5.0)) < QuadFloat::from(1e-28));

    // Both operands square to +inf individually (1e200^2 = 1e400) yet the true
    // hypot ~ 1.414e200 is finite. The naive sqrt(a*a + b*b) would return inf.
    let big = QuadFloat::from(1e200);
    let h_big = NumF::hypot(big, big);
    assert!(NumF::is_finite(h_big), "hypot(1e200, 1e200) overflowed");
    let expected = 1e200_f64.hypot(1e200_f64);
    assert!(
        ((h_big.inner().hi() - expected) / expected).abs() < 1e-13,
        "hypot(1e200,1e200) = {h_big} (expected ~{expected})"
    );

    // Degenerate legs are handled without spurious NaN/overflow.
    let h_zero = NumF::hypot(big, QuadFloat::from(0.0));
    assert!(qabs(h_zero - big) < QuadFloat::from(1e185));
    let h_zero2 = NumF::hypot(QuadFloat::from(0.0), big);
    assert!(qabs(h_zero2 - big) < QuadFloat::from(1e185));
    assert!(NumF::hypot(QuadFloat::from(0.0), QuadFloat::from(0.0)) == QuadFloat::from(0.0));
}

#[cfg(feature = "f128")]
#[test]
fn test_f128_cbrt_negative() {
    // cbrt of a negative real is a well-defined negative real, not NaN.
    use num_traits::Float as NumF;
    let qabs = |x: QuadFloat| QuadFloat::from(x.inner().abs());
    let tol = QuadFloat::from(1e-28);

    assert!(qabs(NumF::cbrt(QuadFloat::from(27.0)) - QuadFloat::from(3.0)) < tol);
    assert!(qabs(NumF::cbrt(QuadFloat::from(8.0)) - QuadFloat::from(2.0)) < tol);

    let cbrt_neg27 = NumF::cbrt(QuadFloat::from(-27.0));
    assert!(!cbrt_neg27.is_nan(), "cbrt(-27) must not be NaN");
    assert!(qabs(cbrt_neg27 - QuadFloat::from(-3.0)) < tol);
    assert!(cbrt_neg27.inner().is_sign_negative(), "cbrt(-27) is negative");

    assert!(qabs(NumF::cbrt(QuadFloat::from(-8.0)) - QuadFloat::from(-2.0)) < tol);

    // cbrt(-x) == -cbrt(x), and the result cubes back to the input at DD
    // precision (validates the Newton refinement, not just an f64 seed).
    for v in [2.0_f64, 10.0, 1234.5, 1e100, 1e-100] {
        let pos = NumF::cbrt(QuadFloat::from(v));
        let neg = NumF::cbrt(QuadFloat::from(-v));
        assert!(qabs(neg + pos) < qabs(pos) * QuadFloat::from(1e-28));
        let cubed = pos * pos * pos;
        assert!(
            qabs(cubed - QuadFloat::from(v)) < qabs(QuadFloat::from(v)) * QuadFloat::from(1e-28),
            "cbrt({v})^3 = {cubed}"
        );
    }
}

// Complex ergonomics tests
#[test]
fn test_complex_constructors() {
    // c64 and c32 constructors
    let z64 = c64(3.0, 4.0);
    assert_eq!(z64.re, 3.0);
    assert_eq!(z64.im, 4.0);

    let z32 = c32(1.0, 2.0);
    assert_eq!(z32.re, 1.0);
    assert_eq!(z32.im, 2.0);

    // imag() and real() constructors
    let i = imag(5.0);
    assert_eq!(i.re, 0.0);
    assert_eq!(i.im, 5.0);

    let r = real(3.0);
    assert_eq!(r.re, 3.0);
    assert_eq!(r.im, 0.0);

    // I64 constant
    assert_eq!(I64.re, 0.0);
    assert_eq!(I64.im, 1.0);
}

#[test]
fn test_complex_polar() {
    use std::f64::consts::PI;

    let z = from_polar(1.0, PI / 2.0);
    assert!((z.re - 0.0).abs() < 1e-10);
    assert!((z.im - 1.0).abs() < 1e-10);

    let z2 = from_polar(2.0, 0.0);
    assert!((z2.re - 2.0).abs() < 1e-10);
    assert!((z2.im - 0.0).abs() < 1e-10);
}

#[test]
fn test_complex_ext() {
    use std::f64::consts::PI;

    let z = c64(3.0, 4.0);

    // normalize
    let n = z.normalize();
    assert!((n.norm() - 1.0).abs() < 1e-10);

    // is_purely_real / is_purely_imaginary
    assert!(c64(3.0, 0.0).is_purely_real(1e-10));
    assert!(!c64(3.0, 1.0).is_purely_real(1e-10));
    assert!(c64(0.0, 4.0).is_purely_imaginary(1e-10));
    assert!(!c64(1.0, 4.0).is_purely_imaginary(1e-10));

    // rotate
    let r = c64(1.0, 0.0).rotate(PI / 2.0);
    assert!((r.re - 0.0).abs() < 1e-10);
    assert!((r.im - 1.0).abs() < 1e-10);

    // distance
    let a = c64(0.0, 0.0);
    let b = c64(3.0, 4.0);
    assert!((a.distance(b) - 5.0).abs() < 1e-10);

    // reflect_imag
    let reflected = c64(1.0, 2.0).reflect_imag();
    assert_eq!(reflected.re, -1.0);
    assert_eq!(reflected.im, 2.0);
}

#[test]
fn test_to_complex() {
    let x: f64 = 3.0;
    let z = x.to_complex();
    assert_eq!(z.re, 3.0);
    assert_eq!(z.im, 0.0);

    let z2 = (2.0f64).with_imag(5.0);
    assert_eq!(z2.re, 2.0);
    assert_eq!(z2.im, 5.0);

    // f32 version
    let x32: f32 = 4.0;
    let z32 = x32.to_complex();
    assert_eq!(z32.re, 4.0);
    assert_eq!(z32.im, 0.0);
}

// Scalar specialization tests
#[test]
fn test_simd_compatible() {
    // Test SIMD width constants

    // Test use_simd_for
    assert!(!f32::use_simd_for(4));
    assert!(f32::use_simd_for(32));
}

#[test]
fn test_scalar_batch_f64() {
    let x = [1.0f64, 2.0, 3.0, 4.0];
    let y = [5.0f64, 6.0, 7.0, 8.0];

    // dot_batch
    let dot = f64::dot_batch(&x, &y);
    assert!((dot - 70.0).abs() < 1e-10); // 1*5 + 2*6 + 3*7 + 4*8 = 70

    // sum_batch
    let sum = f64::sum_batch(&x);
    assert!((sum - 10.0).abs() < 1e-10);

    // asum_batch
    let x_neg = [-1.0f64, 2.0, -3.0, 4.0];
    let asum = f64::asum_batch(&x_neg);
    assert!((asum - 10.0).abs() < 1e-10);

    // iamax_batch
    let x_mixed = [1.0f64, -5.0, 3.0, 2.0];
    let iamax = f64::iamax_batch(&x_mixed);
    assert_eq!(iamax, 1); // index of -5.0

    // scale_batch
    let mut x_scale = [1.0f64, 2.0, 3.0];
    f64::scale_batch(2.0, &mut x_scale);
    assert!((x_scale[0] - 2.0).abs() < 1e-10);
    assert!((x_scale[1] - 4.0).abs() < 1e-10);
    assert!((x_scale[2] - 6.0).abs() < 1e-10);

    // axpy_batch
    let x_axpy = [1.0f64, 2.0, 3.0];
    let mut y_axpy = [1.0f64, 1.0, 1.0];
    f64::axpy_batch(2.0, &x_axpy, &mut y_axpy);
    assert!((y_axpy[0] - 3.0).abs() < 1e-10); // 2*1 + 1
    assert!((y_axpy[1] - 5.0).abs() < 1e-10); // 2*2 + 1
    assert!((y_axpy[2] - 7.0).abs() < 1e-10); // 2*3 + 1

    // fma_batch
    let a = [1.0f64, 2.0, 3.0];
    let b = [2.0f64, 3.0, 4.0];
    let c = [1.0f64, 1.0, 1.0];
    let mut out = [0.0f64; 3];
    f64::fma_batch(&a, &b, &c, &mut out);
    assert!((out[0] - 3.0).abs() < 1e-10); // 1*2 + 1
    assert!((out[1] - 7.0).abs() < 1e-10); // 2*3 + 1
    assert!((out[2] - 13.0).abs() < 1e-10); // 3*4 + 1
}

#[test]
fn test_scalar_batch_complex64() {
    use num_complex::Complex64;
    let x = [c64(1.0, 1.0), c64(2.0, 2.0)];
    let y = [c64(1.0, -1.0), c64(2.0, -2.0)];

    // dot_batch: (1+i)*(1-i) + (2+2i)*(2-2i) = 2 + 8 = 10
    let dot = Complex64::dot_batch(&x, &y);
    assert!((dot.re - 10.0).abs() < 1e-10);
    assert!(dot.im.abs() < 1e-10);

    // sum_batch
    let sum = Complex64::sum_batch(&x);
    assert!((sum.re - 3.0).abs() < 1e-10);
    assert!((sum.im - 3.0).abs() < 1e-10);

    // asum_batch (BLAS-style: sum of |re| + |im|)
    let asum = Complex64::asum_batch(&x);
    assert!((asum - 6.0).abs() < 1e-10); // (1+1) + (2+2)

    // iamax_batch
    let iamax = Complex64::iamax_batch(&x);
    assert_eq!(iamax, 1); // index of (2+2i) has larger |re|+|im|
}

#[test]
fn test_scalar_classify() {
    use num_complex::Complex64;
    assert_eq!(f32::CLASS, ScalarClass::RealF32);
    assert_eq!(f64::CLASS, ScalarClass::RealF64);
    assert_eq!(num_complex::Complex32::CLASS, ScalarClass::ComplexF32);
    assert_eq!(Complex64::CLASS, ScalarClass::ComplexF64);

    assert_eq!(f32::PRECISION_LEVEL, 2);
    assert_eq!(f64::PRECISION_LEVEL, 3);

    assert_eq!(f32::STORAGE_BYTES, 4);
    assert_eq!(f64::STORAGE_BYTES, 8);
    assert_eq!(Complex64::STORAGE_BYTES, 16);
}

#[test]
fn test_unroll_hints() {
    assert_eq!(f32::UNROLL_FACTOR, 8);
    assert_eq!(f64::UNROLL_FACTOR, 4);
    assert_eq!(num_complex::Complex64::UNROLL_FACTOR, 2);
}

#[test]
fn test_extended_precision() {
    // f32 -> f64 accumulation
    let x: f32 = 1.5;
    let acc: f64 = x.to_accumulator();
    assert!((acc - 1.5).abs() < 1e-10);

    let back: f32 = f32::from_accumulator(acc);
    assert!((back - 1.5).abs() < 1e-6);

    // Complex32 -> Complex64
    let z = c32(1.0, 2.0);
    let z_acc: num_complex::Complex64 = z.to_accumulator();
    assert!((z_acc.re - 1.0).abs() < 1e-10);
    assert!((z_acc.im - 2.0).abs() < 1e-10);
}

#[test]
fn test_kahan_sum() {
    let mut kahan = KahanSum::<f64>::new();
    for i in 0..1000 {
        kahan.add(0.1);
        let _ = i; // suppress warning
    }
    // Should be close to 100.0
    let result = kahan.sum();
    assert!((result - 100.0).abs() < 1e-10);
}

#[test]
fn test_pairwise_sum() {
    let values: Vec<f64> = (0..1000).map(|_| 0.1).collect();
    let result = pairwise_sum(&values);
    assert!((result - 100.0).abs() < 1e-10);

    // Empty case
    let empty: Vec<f64> = vec![];
    assert_eq!(pairwise_sum(&empty), 0.0);

    // Small case
    let small = [1.0, 2.0, 3.0];
    assert!(Scalar::abs(pairwise_sum(&small) - 6.0) < 1e-10);
}

#[test]
fn test_kbk_sum() {
    let mut kbk = KBKSum::<f64>::new();
    for _ in 0..10000 {
        kbk.add(0.1);
    }
    let result = kbk.sum();
    // KBK should give very accurate results
    assert!((result - 1000.0).abs() < 1e-8);
}
// =============================================================================
// Regression tests for TODO.md audit findings
// =============================================================================

/// Finding: `Field::powi` for `Complex32` returned ~1.0 for every negative
/// exponent because it multiplied `self.powu(|n|)` by
/// `self.recip().powu(|n|)` instead of choosing one or the other based on
/// the sign of `n`.
///
/// Note: `num_complex::Complex<T>` also defines its *own* inherent
/// `powi(&self, i32)` method (already correct, backed by `num_traits::Pow`),
/// and inherent methods always win over trait methods in dot-call method
/// resolution for a concrete type. So `Complex32::new(..).powi(-2)` was
/// silently calling the correct inherent method all along and would NOT
/// have caught this bug. The only way to reach the buggy code is through
/// the `Field` trait itself — either via explicit UFCS, or (the realistic
/// failure mode) through code generic over `T: Field` calling `x.powi(n)`,
/// where the compiler can only resolve to the trait method. This test
/// exercises both paths explicitly.
#[test]
fn test_complex32_powi_negative_exponent() {
    use num_complex::Complex32;

    let z = Complex32::new(2.0, 1.0);
    let z_squared = z * z;
    let expected = Complex32::new(1.0, 0.0) / z_squared;

    // Exercise the trait method directly via UFCS -- this is what was
    // actually broken, and is bypassed by ordinary dot-call syntax on a
    // concrete Complex32 due to num_complex's own inherent `powi`.
    let actual = Field::powi(z, -2);
    assert!(
        (actual.re - expected.re).abs() < 1e-5,
        "re mismatch: actual={actual:?} expected={expected:?}"
    );
    assert!(
        (actual.im - expected.im).abs() < 1e-5,
        "im mismatch: actual={actual:?} expected={expected:?}"
    );

    // Regression guard against the historical bug: the buggy
    // implementation collapsed to ~(1.0, 0.0) for every negative exponent.
    assert!(
        (actual.re - 1.0).abs() > 1e-3 || actual.im.abs() > 1e-3,
        "Field::powi(-2) collapsed to ~1.0, the historical bug: {actual:?}"
    );

    // Exercise the realistic failure mode: a function generic over
    // `T: Field`, which can only resolve `.powi()` to the trait method.
    fn generic_powi<T: Field>(x: T, n: i32) -> T {
        x.powi(n)
    }
    let generic_actual = generic_powi(z, -2);
    assert!((generic_actual.re - expected.re).abs() < 1e-5);
    assert!((generic_actual.im - expected.im).abs() < 1e-5);

    // Field::powi(-1) must equal Field::recip().
    let inv = Field::powi(z, -1);
    let recip = Field::recip(z);
    assert!((inv.re - recip.re).abs() < 1e-6);
    assert!((inv.im - recip.im).abs() < 1e-6);

    // Field::powi(0) == 1, Field::powi(positive) unaffected by the fix.
    assert_eq!(Field::powi(z, 0), Complex32::new(1.0, 0.0));
    let cubed = Field::powi(z, 3);
    let cubed_expected = z * z * z;
    assert!((cubed.re - cubed_expected.re).abs() < 1e-4);
    assert!((cubed.im - cubed_expected.im).abs() < 1e-4);

    // Complex32 must now match the (already-correct) Complex64 behavior
    // for the same exponent, up to the precision difference.
    use num_complex::Complex64;
    let z64 = Complex64::new(2.0, 1.0);
    let actual64 = Field::powi(z64, -2);
    assert!((actual.re as f64 - actual64.re).abs() < 1e-5);
    assert!((actual.im as f64 - actual64.im).abs() < 1e-5);
}

/// Finding: `Real::signum` doc claimed `0.0 if zero`, which does not match
/// Rust's f32/f64 `signum` (IEEE-754 style: sign of zero is preserved as
/// +/-1.0, never a literal 0.0, and NaN propagates). This test pins down
/// the corrected, documented semantics for the f32/f64 impls.
#[test]
fn test_real_signum_matches_ieee754_semantics() {
    // f32
    assert_eq!(<f32 as Real>::signum(3.5), 1.0);
    assert_eq!(<f32 as Real>::signum(-3.5), -1.0);
    assert_eq!(<f32 as Real>::signum(0.0), 1.0);
    assert_eq!(<f32 as Real>::signum(-0.0), -1.0);
    assert!(<f32 as Real>::signum(-0.0_f32).is_sign_negative());
    assert_eq!(<f32 as Real>::signum(f32::INFINITY), 1.0);
    assert_eq!(<f32 as Real>::signum(f32::NEG_INFINITY), -1.0);
    assert!(<f32 as Real>::signum(f32::NAN).is_nan());

    // f64
    assert_eq!(<f64 as Real>::signum(3.5), 1.0);
    assert_eq!(<f64 as Real>::signum(-3.5), -1.0);
    assert_eq!(<f64 as Real>::signum(0.0), 1.0);
    assert_eq!(<f64 as Real>::signum(-0.0), -1.0);
    assert!(<f64 as Real>::signum(-0.0_f64).is_sign_negative());
    assert_eq!(<f64 as Real>::signum(f64::INFINITY), 1.0);
    assert_eq!(<f64 as Real>::signum(f64::NEG_INFINITY), -1.0);
    assert!(<f64 as Real>::signum(f64::NAN).is_nan());
}

/// Finding: `iamax_batch` used `Iterator::max_by`, which returns the LAST
/// maximum on ties and treats incomparable (NaN) values as `Equal`. This
/// diverges from reference BLAS IxAMAX, which keeps the FIRST index on
/// ties (via a strict `>` comparison) and never lets a NaN comparison
/// displace the running maximum.
#[test]
fn test_iamax_batch_first_index_wins_ties() {
    use num_complex::{Complex32, Complex64};

    // f32: three-way tie at abs value 3.0 -> first index (0) wins.
    let x = [3.0f32, -3.0, 1.0, 3.0];
    assert_eq!(f32::iamax_batch(&x), 0);

    // f64: tie at index 1 and 2 -> first of the tied indices (1) wins.
    let y = [1.0f64, 5.0, 5.0, 2.0];
    assert_eq!(f64::iamax_batch(&y), 1);

    // Complex32: |1|+|2| == |-2|+|-1| == 3.0 -> first index (0) wins.
    let zc32 = [
        Complex32::new(1.0, 2.0),
        Complex32::new(-2.0, -1.0),
        Complex32::new(0.5, 0.5),
    ];
    assert_eq!(Complex32::iamax_batch(&zc32), 0);

    // Complex64: |2|+|0| == |0|+|2| == 2.0 -> first index (0) wins.
    let zc64 = [
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 2.0),
        Complex64::new(1.0, 0.5),
    ];
    assert_eq!(Complex64::iamax_batch(&zc64), 0);
}

#[test]
fn test_iamax_batch_nan_matches_reference_blas() {
    // Reference BLAS IxAMAX only updates the running maximum on a strict
    // `>` comparison, which is always false against NaN. If the FIRST
    // element is NaN, nothing can ever beat it, so index 0 sticks even
    // though later elements have a well-defined, larger magnitude.
    let x = [f32::NAN, 10.0, 20.0];
    assert_eq!(f32::iamax_batch(&x), 0);

    // A NaN appearing after a genuine finite maximum must not be selected.
    let y = [1.0f64, 9.0, f64::NAN, 3.0];
    assert_eq!(f64::iamax_batch(&y), 1);

    // A NaN appearing before the true maximum must not suppress it either.
    let z = [1.0f64, f64::NAN, 9.0, 3.0];
    assert_eq!(f64::iamax_batch(&z), 2);
}

/// Finding: `ExtendedPrecision for f64` always used `Accumulator = f64`
/// even when the crate's `f128` (double-double) type was available,
/// silently discarding the opportunity for genuinely higher-precision
/// accumulation. With the `f128` feature enabled, the accumulator must
/// now be `QuadFloat` and must be materially more accurate than plain
/// `f64` summation for a case with significant f64 rounding error.
#[test]
#[cfg(feature = "f128")]
fn test_extended_precision_f64_uses_quadfloat_with_f128_feature() {
    // Lossless round trip for a value that is not exactly representable
    // in binary floating point.
    let x: f64 = 1.0 / 3.0;
    let acc: QuadFloat = x.to_accumulator();
    let back: f64 = f64::from_accumulator(acc);
    assert_eq!(back, x);

    // Summing 0.1 ten thousand times accumulates significant f64 rounding
    // error; accumulating in QuadFloat space and rounding back to f64 at
    // the end must be at least as accurate as summing directly in f64.
    let mut extended_sum = QuadFloat::from(0.0);
    let mut naive_sum = 0.0f64;
    let term = 0.1f64;
    for _ in 0..10_000 {
        extended_sum += QuadFloat::from(term);
        naive_sum += term;
    }
    let extended_result = f64::from_accumulator(extended_sum);
    let exact = 1000.0f64; // 10_000 * 0.1
    let extended_error = (extended_result - exact).abs();
    let naive_error = (naive_sum - exact).abs();
    assert!(
        extended_error <= naive_error,
        "extended accumulation ({extended_result}, err={extended_error}) was not at least \
         as accurate as naive f64 summation ({naive_sum}, err={naive_error})"
    );
}
