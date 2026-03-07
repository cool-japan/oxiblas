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
