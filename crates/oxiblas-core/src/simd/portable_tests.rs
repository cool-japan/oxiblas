//! Regression tests for the portable (generic-architecture) SIMD fallback.
//!
//! The portable registers are the `SimdScalar` backend on every architecture
//! without a native backend (riscv64, powerpc64, s390x, loongarch64, armv7,
//! i686, ...). Those targets are not exercised by the host test run, so these
//! tests drive the portable implementation directly on the host and compare it
//! lane-by-lane against (a) a plain scalar reference and (b) the host's native
//! `SimdScalar` registers on random inputs.

use super::*;

/// Deterministic xorshift64* generator (no external RNG dependency).
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform-ish value in `[-100, 100)`, never exactly zero.
    fn next_f64(&mut self) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let v = unit * 200.0 - 100.0;
        if v == 0.0 { 0.5 } else { v }
    }

    fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }
}

/// Checks every lane-wise op of a portable register against a scalar
/// reference, bit-for-bit (IEEE-754 ops are correctly rounded per lane), and
/// the horizontal reductions against a tolerance.
macro_rules! lanewise_reference_test {
    ($test_name:ident, $reg:ty, $scalar:ty, $gen:ident, $tol:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = <$reg as SimdRegister>::LANES;
            let mut rng = XorShift(0x9E37_79B9_7F4A_7C15 ^ (N as u64));
            for _ in 0..500 {
                let mut xa = [0.0 as $scalar; N];
                let mut xb = [0.0 as $scalar; N];
                let mut xc = [0.0 as $scalar; N];
                for i in 0..N {
                    xa[i] = rng.$gen();
                    xb[i] = rng.$gen();
                    xc[i] = rng.$gen();
                }
                // SAFETY: each array holds exactly `N == LANES` elements.
                let (a, b, c) = unsafe {
                    (
                        <$reg>::load_unaligned(xa.as_ptr()),
                        <$reg>::load_aligned(xb.as_ptr()),
                        <$reg>::load_unaligned(xc.as_ptr()),
                    )
                };

                let ops: [(&str, $reg, fn($scalar, $scalar, $scalar) -> $scalar); 7] = [
                    ("add", a.add(b), |x, y, _| x + y),
                    ("sub", a.sub(b), |x, y, _| x - y),
                    ("mul", a.mul(b), |x, y, _| x * y),
                    ("div", a.div(b), |x, y, _| x / y),
                    ("mul_add", a.mul_add(b, c), |x, y, z| x.mul_add(y, z)),
                    ("mul_sub", a.mul_sub(b, c), |x, y, z| x.mul_add(y, -z)),
                    ("neg_mul_add", a.neg_mul_add(b, c), |x, y, z| {
                        (-x).mul_add(y, z)
                    }),
                ];
                for (name, got, reference) in ops {
                    let mut out = [0.0 as $scalar; N];
                    // SAFETY: `out` holds exactly `LANES` writable elements.
                    unsafe { got.store_unaligned(out.as_mut_ptr()) };
                    for i in 0..N {
                        let want = reference(xa[i], xb[i], xc[i]);
                        assert_eq!(
                            out[i].to_bits(),
                            want.to_bits(),
                            "{name} lane {i}: {} vs {want}",
                            out[i]
                        );
                        assert_eq!(got.extract(i).to_bits(), want.to_bits());
                    }
                }

                let sum: $scalar = xa.iter().sum();
                let scale: $scalar = xa.iter().map(|v| v.abs()).sum::<$scalar>().max(1.0);
                assert!((a.reduce_sum() - sum).abs() <= $tol * scale);
                let max = xa
                    .iter()
                    .copied()
                    .fold(<$scalar>::NEG_INFINITY, <$scalar>::max);
                let min = xa.iter().copied().fold(<$scalar>::INFINITY, <$scalar>::min);
                assert_eq!(a.reduce_max(), max);
                assert_eq!(a.reduce_min(), min);

                // insert / extract / splat / zero
                let idx = (rng.next_u64() as usize) % N;
                let ins = a.insert(idx, 7.0);
                for i in 0..N {
                    let want = if i == idx { 7.0 } else { xa[i] };
                    assert_eq!(ins.extract(i), want);
                }
                assert_eq!(<$reg>::splat(3.0).reduce_sum(), 3.0 * N as $scalar);
                assert_eq!(<$reg>::zero().reduce_sum(), 0.0);
            }
        }
    };
}

lanewise_reference_test!(
    portable_f64x4_matches_scalar_reference,
    PortableF64x4,
    f64,
    next_f64,
    1e-13
);
lanewise_reference_test!(
    portable_f64x8_matches_scalar_reference,
    PortableF64x8,
    f64,
    next_f64,
    1e-13
);
lanewise_reference_test!(
    portable_f32x8_matches_scalar_reference,
    PortableF32x8,
    f32,
    next_f32,
    1e-5
);
lanewise_reference_test!(
    portable_f32x16_matches_scalar_reference,
    PortableF32x16,
    f32,
    next_f32,
    1e-5
);

/// Lane counts must agree with the nominal widths the `SimdScalar` trait
/// advertises, since the portable types back `Simd256`/`Simd512` there.
#[test]
fn portable_lane_counts_match_nominal_widths() {
    assert_eq!(PortableF64x4::LANES, 32 / core::mem::size_of::<f64>());
    assert_eq!(PortableF64x8::LANES, 64 / core::mem::size_of::<f64>());
    assert_eq!(PortableF32x8::LANES, 32 / core::mem::size_of::<f32>());
    assert_eq!(PortableF32x16::LANES, 64 / core::mem::size_of::<f32>());
}

/// On fallback architectures the `SimdScalar` associated registers must be
/// the portable ones, with lane counts consistent with `LANES_256/512`.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
#[test]
fn fallback_simd_scalar_uses_portable_registers() {
    use crate::simd::SimdScalar;
    assert_eq!(
        <f64 as SimdScalar>::Simd256::LANES,
        <f64 as SimdScalar>::LANES_256
    );
    assert_eq!(
        <f64 as SimdScalar>::Simd512::LANES,
        <f64 as SimdScalar>::LANES_512
    );
    assert_eq!(
        <f32 as SimdScalar>::Simd256::LANES,
        <f32 as SimdScalar>::LANES_256
    );
    assert_eq!(
        <f32 as SimdScalar>::Simd512::LANES,
        <f32 as SimdScalar>::LANES_512
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn portable_extract_out_of_range_panics() {
    let _ = PortableF64x4::splat(1.0).extract(4);
}

#[test]
#[should_panic(expected = "out of range")]
fn portable_insert_out_of_range_panics() {
    let _ = PortableF32x8::splat(1.0).insert(8, 0.0);
}

/// Generic Level-1 kernels written once against `SimdRegister`, used to run
/// the same algorithm through the portable and the native registers.
fn generic_dot<R: SimdRegister<Scalar = f64>>(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let lanes = R::LANES;
    let full = len / lanes * lanes;
    let mut acc = R::zero();
    let mut i = 0;
    while i < full {
        // SAFETY: `i + lanes <= full <= len`, so both loads are in bounds.
        let (va, vb) = unsafe {
            (
                R::load_unaligned(a.as_ptr().add(i)),
                R::load_unaligned(b.as_ptr().add(i)),
            )
        };
        acc = va.mul_add(vb, acc);
        i += lanes;
    }
    let mut sum = acc.reduce_sum();
    for k in full..len {
        sum += a[k] * b[k];
    }
    sum
}

fn generic_axpy<R: SimdRegister<Scalar = f32>>(alpha: f32, x: &[f32], y: &mut [f32]) {
    let len = x.len().min(y.len());
    let lanes = R::LANES;
    let full = len / lanes * lanes;
    let va = R::splat(alpha);
    let mut i = 0;
    while i < full {
        // SAFETY: `i + lanes <= full <= len` for both slices.
        unsafe {
            let vx = R::load_unaligned(x.as_ptr().add(i));
            let vy = R::load_unaligned(y.as_ptr().add(i));
            va.mul_add(vx, vy).store_unaligned(y.as_mut_ptr().add(i));
        }
        i += lanes;
    }
    for k in full..len {
        y[k] = alpha.mul_add(x[k], y[k]);
    }
}

/// Whether the host's native `SimdScalar::Simd256` registers may be executed
/// (the x86_64 AVX2 registers have no internal runtime guard).
fn native_simd256_usable() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    {
        cfg!(all(target_feature = "avx2", target_feature = "fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// The portable fallback must reproduce the native SIMD path on random data,
/// both lane-wise (axpy, bit-exact since both use fused multiply-add) and for
/// a reduction (dot, within a tight tolerance for summation order).
#[test]
fn portable_matches_native_simd_on_random_inputs() {
    use crate::simd::SimdScalar;
    if !native_simd256_usable() {
        return;
    }
    type NativeF64 = <f64 as SimdScalar>::Simd256;
    type NativeF32 = <f32 as SimdScalar>::Simd256;

    let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
    for len in [0usize, 1, 3, 7, 8, 15, 16, 33, 100, 257] {
        let mut a = [0.0f64; 257];
        let mut b = [0.0f64; 257];
        for i in 0..len {
            a[i] = rng.next_f64();
            b[i] = rng.next_f64();
        }
        let native = generic_dot::<NativeF64>(&a[..len], &b[..len]);
        let portable = generic_dot::<PortableF64x4>(&a[..len], &b[..len]);
        let portable512 = generic_dot::<PortableF64x8>(&a[..len], &b[..len]);
        let scale: f64 = a[..len]
            .iter()
            .zip(&b[..len])
            .map(|(x, y)| (x * y).abs())
            .sum::<f64>()
            .max(1.0);
        assert!(
            (native - portable).abs() <= 1e-12 * scale,
            "len {len}: {native} vs {portable}"
        );
        assert!(
            (native - portable512).abs() <= 1e-12 * scale,
            "len {len}: {native} vs {portable512}"
        );

        let alpha = rng.next_f32();
        let mut x = [0.0f32; 257];
        let mut y0 = [0.0f32; 257];
        for i in 0..len {
            x[i] = rng.next_f32();
            y0[i] = rng.next_f32();
        }
        let mut y_native = y0;
        let mut y_portable = y0;
        let mut y_portable512 = y0;
        generic_axpy::<NativeF32>(alpha, &x[..len], &mut y_native[..len]);
        generic_axpy::<PortableF32x8>(alpha, &x[..len], &mut y_portable[..len]);
        generic_axpy::<PortableF32x16>(alpha, &x[..len], &mut y_portable512[..len]);
        for i in 0..len {
            assert_eq!(
                y_native[i].to_bits(),
                y_portable[i].to_bits(),
                "axpy lane {i}"
            );
            assert_eq!(
                y_native[i].to_bits(),
                y_portable512[i].to_bits(),
                "axpy lane {i}"
            );
        }
    }
}
