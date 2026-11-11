//! Unit tests for the Level-1 dot product kernels.

use super::*;

#[test]
fn test_dot_f64() {
    let x = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let y = [5.0f64, 4.0, 3.0, 2.0, 1.0];

    // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
    let result = dot(&x, &y);
    assert!((result - 35.0).abs() < 1e-10);
}

#[test]
fn test_dot_f32() {
    let x = [1.0f32, 2.0, 3.0];
    let y = [4.0f32, 5.0, 6.0];

    let result = dot(&x, &y);
    assert!((result - 32.0).abs() < 1e-5);
}

#[test]
fn test_dot_empty() {
    let x: [f64; 0] = [];
    let y: [f64; 0] = [];

    let result = dot(&x, &y);
    assert_eq!(result, 0.0);
}

#[test]
fn test_dot_single() {
    let x = [3.0f64];
    let y = [4.0f64];

    let result = dot(&x, &y);
    assert!((result - 12.0).abs() < 1e-10);
}

#[test]
fn test_dot_large() {
    let n = 1000;
    let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let y: Vec<f64> = vec![1.0; n];

    // Sum of 1 to n = n*(n+1)/2 = 500500
    let result = dot(&x, &y);
    assert!((result - 500500.0).abs() < 1e-6);
}

#[test]
fn test_dot_f64_simd() {
    let n = 1000;
    let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let y: Vec<f64> = vec![1.0; n];

    let result = dot_f64(&x, &y);
    let expected = 500500.0;
    assert!(
        (result - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_dot_f32_simd() {
    let n = 1000;
    let x: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let y: Vec<f32> = vec![1.0; n];

    let result = dot_f32(&x, &y);
    let expected = 500500.0f32;
    assert!(
        (result - expected).abs() < 1.0, // f32 has less precision
        "Expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_dot_f64_large() {
    // Test with large vector to exercise blocked path
    let n = 100000;
    let x: Vec<f64> = vec![1.0; n];
    let y: Vec<f64> = vec![1.0; n];

    let result = dot_f64(&x, &y);
    assert!(
        (result - n as f64).abs() < 1e-6,
        "Expected {}, got {}",
        n,
        result
    );
}

#[test]
fn test_dot_f32_large() {
    // Test with large vector to exercise blocked path
    let n = 100000;
    let x: Vec<f32> = vec![1.0; n];
    let y: Vec<f32> = vec![1.0; n];

    let result = dot_f32(&x, &y);
    assert!(
        (result - n as f32).abs() < 100.0, // f32 accumulation error grows
        "Expected {}, got {}",
        n,
        result
    );
}

#[test]
fn test_dot_f64_consistency() {
    // Test that SIMD and generic implementations give consistent results
    let n = 500;
    let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let y: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 0.1).collect();

    let result_generic = dot(&x, &y);
    let result_simd = dot_f64(&x, &y);

    assert!(
        (result_generic - result_simd).abs() / result_generic.abs() < 1e-10,
        "Generic: {}, SIMD: {}",
        result_generic,
        result_simd
    );
}

#[test]
fn test_dot_edge_cases() {
    // Empty vectors
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];
    assert_eq!(dot_f64(&x, &y), 0.0);

    // Single element
    let x = vec![5.0f64];
    let y = vec![3.0f64];
    assert!((dot_f64(&x, &y) - 15.0).abs() < 1e-10);

    // Odd length
    let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0f64, 1.0, 1.0, 1.0, 1.0];
    assert!((dot_f64(&x, &y) - 15.0).abs() < 1e-10);
}

// =============================================================================
// Complex dot product tests
// =============================================================================

#[test]
fn test_dotc_c64_basic() {
    // Test conjugate dot product
    // x = [1+2i, 3+4i], y = [5+6i, 7+8i]
    // conj(x) * y = (1-2i)(5+6i) + (3-4i)(7+8i)
    //             = (5+12) + (6-10)i + (21+32) + (24-28)i
    //             = 17 + (-4)i + 53 + (-4)i
    //             = 70 + (-8)i
    let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let y = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

    let result = dotc_c64(&x, &y);
    assert!((result.re - 70.0).abs() < 1e-10, "re: {}", result.re);
    assert!((result.im - (-8.0)).abs() < 1e-10, "im: {}", result.im);
}

#[test]
fn test_dotu_c64_basic() {
    // Test unconjugated dot product
    // x = [1+2i, 3+4i], y = [5+6i, 7+8i]
    // x * y = (1+2i)(5+6i) + (3+4i)(7+8i)
    //       = (5-12) + (6+10)i + (21-32) + (24+28)i
    //       = -7 + 16i + (-11) + 52i
    //       = -18 + 68i
    let x = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let y = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

    let result = dotu_c64(&x, &y);
    assert!((result.re - (-18.0)).abs() < 1e-10, "re: {}", result.re);
    assert!((result.im - 68.0).abs() < 1e-10, "im: {}", result.im);
}

#[test]
fn test_dotc_c32_basic() {
    let x = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    let y = vec![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

    let result = dotc_c32(&x, &y);
    assert!((result.re - 70.0).abs() < 1e-5, "re: {}", result.re);
    assert!((result.im - (-8.0)).abs() < 1e-5, "im: {}", result.im);
}

#[test]
fn test_dotu_c32_basic() {
    let x = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    let y = vec![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

    let result = dotu_c32(&x, &y);
    assert!((result.re - (-18.0)).abs() < 1e-5, "re: {}", result.re);
    assert!((result.im - 68.0).abs() < 1e-5, "im: {}", result.im);
}

#[test]
fn test_dotc_c64_empty() {
    let x: Vec<Complex64> = vec![];
    let y: Vec<Complex64> = vec![];

    let result = dotc_c64(&x, &y);
    assert_eq!(result, Complex64::new(0.0, 0.0));
}

#[test]
fn test_dotc_c64_single() {
    let x = vec![Complex64::new(3.0, 4.0)];
    let y = vec![Complex64::new(1.0, 2.0)];

    // conj(3+4i) * (1+2i) = (3-4i)(1+2i) = 3 + 6i - 4i - 8i^2 = 3 + 2i + 8 = 11 + 2i
    let result = dotc_c64(&x, &y);
    assert!((result.re - 11.0).abs() < 1e-10);
    assert!((result.im - 2.0).abs() < 1e-10);
}

#[test]
fn test_dotc_c64_medium() {
    // Medium-sized test to exercise SIMD path but verify correctness
    let n = 100;
    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, (i as f64) * 0.5))
        .collect();
    let y: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(1.0, 0.1 * i as f64))
        .collect();

    let result_simd = dotc_c64(&x, &y);

    // Compare with scalar implementation
    let result_scalar = dotc_c64_scalar(&x, &y);

    assert!(
        (result_simd.re - result_scalar.re).abs() < 1e-6,
        "re: simd={}, scalar={}",
        result_simd.re,
        result_scalar.re
    );
    assert!(
        (result_simd.im - result_scalar.im).abs() < 1e-6,
        "im: simd={}, scalar={}",
        result_simd.im,
        result_scalar.im
    );
}

#[test]
fn test_dotc_c64_large() {
    // Large test to exercise SIMD path
    let n = 10000;
    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i % 100) as f64, ((i + 50) % 100) as f64))
        .collect();
    let y: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(((i + 25) % 100) as f64, ((i + 75) % 100) as f64))
        .collect();

    let result_simd = dotc_c64(&x, &y);

    // Compare with scalar implementation
    let result_scalar = dotc_c64_scalar(&x, &y);

    let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
    let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

    assert!(
        re_err < 1e-10,
        "re: simd={}, scalar={}, rel_err={}",
        result_simd.re,
        result_scalar.re,
        re_err
    );
    assert!(
        im_err < 1e-10,
        "im: simd={}, scalar={}, rel_err={}",
        result_simd.im,
        result_scalar.im,
        im_err
    );
}

#[test]
fn test_dotu_c64_large() {
    let n = 10000;
    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i % 100) as f64, ((i + 50) % 100) as f64))
        .collect();
    let y: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(((i + 25) % 100) as f64, ((i + 75) % 100) as f64))
        .collect();

    let result_simd = dotu_c64(&x, &y);
    let result_scalar = dotu_c64_scalar(&x, &y);

    let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
    let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

    assert!(
        re_err < 1e-10,
        "re: simd={}, scalar={}, rel_err={}",
        result_simd.re,
        result_scalar.re,
        re_err
    );
    assert!(
        im_err < 1e-10,
        "im: simd={}, scalar={}, rel_err={}",
        result_simd.im,
        result_scalar.im,
        im_err
    );
}

#[test]
fn test_dotc_c32_large() {
    let n = 10000;
    let x: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i % 100) as f32, ((i + 50) % 100) as f32))
        .collect();
    let y: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(((i + 25) % 100) as f32, ((i + 75) % 100) as f32))
        .collect();

    let result_simd = dotc_c32(&x, &y);
    let result_scalar = dotc_c32_scalar(&x, &y);

    let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
    let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

    assert!(
        re_err < 1e-4,
        "re: simd={}, scalar={}, rel_err={}",
        result_simd.re,
        result_scalar.re,
        re_err
    );
    assert!(
        im_err < 1e-4,
        "im: simd={}, scalar={}, rel_err={}",
        result_simd.im,
        result_scalar.im,
        im_err
    );
}

#[test]
fn test_dotu_c32_large() {
    let n = 10000;
    let x: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i % 100) as f32, ((i + 50) % 100) as f32))
        .collect();
    let y: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(((i + 25) % 100) as f32, ((i + 75) % 100) as f32))
        .collect();

    let result_simd = dotu_c32(&x, &y);
    let result_scalar = dotu_c32_scalar(&x, &y);

    let re_err = (result_simd.re - result_scalar.re).abs() / result_scalar.re.abs().max(1.0);
    let im_err = (result_simd.im - result_scalar.im).abs() / result_scalar.im.abs().max(1.0);

    assert!(
        re_err < 1e-4,
        "re: simd={}, scalar={}, rel_err={}",
        result_simd.re,
        result_scalar.re,
        re_err
    );
    assert!(
        im_err < 1e-4,
        "im: simd={}, scalar={}, rel_err={}",
        result_simd.im,
        result_scalar.im,
        im_err
    );
}

#[test]
fn test_dotc_c64_self_dot() {
    // Dot product with conjugate should give real result for x^H * x
    let n = 100;
    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64, (i as f64) * 0.5))
        .collect();

    let result = dotc_c64(&x, &x);

    // x^H * x should be purely real (imaginary part should be ~0)
    assert!(
        result.im.abs() < 1e-10,
        "x^H * x should be real, but im={}",
        result.im
    );

    // x^H * x should equal sum of |x[i]|^2
    let expected_re: f64 = x.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (result.re - expected_re).abs() < 1e-6,
        "re: {}, expected: {}",
        result.re,
        expected_re
    );
}

#[test]
fn test_complex_dot_remainder_handling() {
    // Test various sizes to exercise remainder handling
    for n in [1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65] {
        let x: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, 1.0)).collect();
        let y: Vec<Complex64> = (0..n).map(|i| Complex64::new(1.0, i as f64)).collect();

        let result_simd = dotc_c64(&x, &y);
        let result_scalar = dotc_c64_scalar(&x, &y);

        assert!(
            (result_simd.re - result_scalar.re).abs() < 1e-10,
            "n={}: re mismatch",
            n
        );
        assert!(
            (result_simd.im - result_scalar.im).abs() < 1e-10,
            "n={}: im mismatch",
            n
        );
    }
}

// -------------------------------------------------------------------------
// Regression: reversed sign-mask lane order in the AVX2 ZDOTU/CDOTU kernels
// negated the real part of the unconjugated complex dot product. These tests
// pin the kernels against an independent reference using GENUINELY ASYMMETRIC
// complex vectors (re and im differ in sign and magnitude, with negatives).
// A symmetric or real-only test vector would NOT expose a flipped real sign,
// which is why the historic bug went unnoticed.
// -------------------------------------------------------------------------

/// Independent, straightforward reference for the unconjugated complex dot
/// product: `Σ x[i] * y[i]` with re = re*re - im*im, im = re*im + im*re.
fn dotu_c64_reference(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (a, b) in x.iter().zip(y.iter()) {
        re += a.re * b.re - a.im * b.im;
        im += a.re * b.im + a.im * b.re;
    }
    Complex64::new(re, im)
}

/// Independent reference for the conjugated complex dot product:
/// `Σ conj(x[i]) * y[i]` with re = re*re + im*im, im = re*im - im*re.
fn dotc_c64_reference(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (a, b) in x.iter().zip(y.iter()) {
        re += a.re * b.re + a.im * b.im;
        im += a.re * b.im - a.im * b.re;
    }
    Complex64::new(re, im)
}

/// Build asymmetric Complex64 test vectors of length `n`: components mix
/// positive and negative values and re/im magnitudes differ per element.
fn asymmetric_c64(n: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let x: Vec<Complex64> = (0..n)
        .map(|i| {
            let ii = i as f64;
            let re = if i % 2 == 0 { ii + 1.0 } else { -(ii + 1.0) };
            let im = ii - 13.0; // spans negative and positive
            Complex64::new(re, im)
        })
        .collect();
    let y: Vec<Complex64> = (0..n)
        .map(|i| {
            let ii = i as f64;
            let re = ii - 7.0; // spans negative and positive
            let im = if i % 3 == 0 {
                -(2.0 * ii + 1.0)
            } else {
                2.0 * ii + 1.0
            };
            Complex64::new(re, im)
        })
        .collect();
    (x, y)
}

#[test]
fn test_dotu_c64_avx2_asymmetric_signs() {
    // n odd and > 16 so the AVX2 path runs (chunks + scalar remainder).
    let (x, y) = asymmetric_c64(37);

    let expected = dotu_c64_reference(&x, &y);
    let simd = dotu_c64(&x, &y);
    let scalar = dotu_c64_scalar(&x, &y);

    // Guard the test itself: the real part must be large and non-zero so a
    // flipped sign (the historic bug) is unambiguously observable.
    assert!(
        expected.re.abs() > 100.0,
        "test data too weak to catch a sign flip: re={}",
        expected.re
    );

    assert!(
        (simd.re - expected.re).abs() < 1e-8,
        "re: simd={}, expected={}",
        simd.re,
        expected.re
    );
    assert!(
        (simd.im - expected.im).abs() < 1e-8,
        "im: simd={}, expected={}",
        simd.im,
        expected.im
    );
    assert!(
        (scalar.re - expected.re).abs() < 1e-8,
        "scalar re: {} vs expected {}",
        scalar.re,
        expected.re
    );
    assert!(
        (scalar.im - expected.im).abs() < 1e-8,
        "scalar im: {} vs expected {}",
        scalar.im,
        expected.im
    );
}

#[test]
fn test_dotc_c64_avx2_asymmetric_signs() {
    // Independently guard the conjugated kernel against the same bug class.
    let (x, y) = asymmetric_c64(37);

    let expected = dotc_c64_reference(&x, &y);
    let simd = dotc_c64(&x, &y);

    assert!(
        expected.im.abs() > 100.0,
        "test data too weak to catch an imag sign flip: im={}",
        expected.im
    );

    assert!(
        (simd.re - expected.re).abs() < 1e-8,
        "re: simd={}, expected={}",
        simd.re,
        expected.re
    );
    assert!(
        (simd.im - expected.im).abs() < 1e-8,
        "im: simd={}, expected={}",
        simd.im,
        expected.im
    );
}

#[test]
fn test_dotu_c32_avx2_asymmetric_signs() {
    // n > 32 so the CDOTU AVX2 path runs. Keep magnitudes modest and use a
    // relative tolerance because f32 accumulation order differs across the
    // SIMD, scalar, and reference paths; the historic sign flip produces a
    // ~200% error, far above this tolerance.
    let n = 41usize;
    let x: Vec<Complex32> = (0..n)
        .map(|i| {
            let ii = i as f32;
            let re = if i % 2 == 0 { ii + 1.0 } else { -(ii + 1.0) };
            Complex32::new(re, ii - 13.0)
        })
        .collect();
    let y: Vec<Complex32> = (0..n)
        .map(|i| {
            let ii = i as f32;
            let im = if i % 3 == 0 {
                -(2.0 * ii + 1.0)
            } else {
                2.0 * ii + 1.0
            };
            Complex32::new(ii - 7.0, im)
        })
        .collect();

    // f64 reference for accuracy, then compare in f32 magnitudes.
    let x64: Vec<Complex64> = x
        .iter()
        .map(|c| Complex64::new(c.re as f64, c.im as f64))
        .collect();
    let y64: Vec<Complex64> = y
        .iter()
        .map(|c| Complex64::new(c.re as f64, c.im as f64))
        .collect();
    let expected = dotu_c64_reference(&x64, &y64);

    let simd = dotu_c32(&x, &y);

    assert!(
        expected.re.abs() > 100.0,
        "test data too weak to catch a sign flip: re={}",
        expected.re
    );

    let re_err = (simd.re as f64 - expected.re).abs() / expected.re.abs();
    let im_err = (simd.im as f64 - expected.im).abs() / expected.im.abs().max(1.0);
    assert!(
        re_err < 1e-3,
        "re: simd={}, expected={}, rel_err={}",
        simd.re,
        expected.re,
        re_err
    );
    assert!(
        im_err < 1e-3,
        "im: simd={}, expected={}, rel_err={}",
        simd.im,
        expected.im,
        im_err
    );
}
