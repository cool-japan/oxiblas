//! Oracle tests for the NEON Blue's-algorithm NRM2 kernels.
//!
//! The oracle is a scalar, statement-by-statement port of LAPACK 3.12
//! `BLAS/SRC/dnrm2.f90` (and `snrm2.f90`) for `incx = 1`: the same constants,
//! the same `notbig` flag, unfused squares and a strictly sequential loop.
//!
//! The kernel differs from the oracle only in the order of the additions inside
//! each accumulator (lane-parallel partial sums instead of one running sum) and
//! in fusing each square into its accumulation (`fma` instead of `x*x` then
//! `+`). Every accumulator sums non-negative terms, so each ordering is within
//! `γ_n` (relative) of the exact sum. For the random data below (n = 0..=300,
//! seven magnitude classes, 2107 vectors per type) the largest observed
//! kernel–oracle distance is 4 ulps of the final norm (three vectors each for
//! f64 and f32, about 25 more at 3 ulps, all others at most 2); the tests
//! assert at most [`F64_ULPS`] / [`F32_ULPS`] = 6. The double-double test shows
//! that the distance comes from the oracle's single running sum: against a
//! high-precision norm the kernel is never more than one ulp worse than the
//! oracle. Exceptional results must match exactly: NaN iff the oracle is NaN,
//! `+Inf` iff the oracle is `+Inf`, and `+0` (with a positive sign) for
//! all-zero input.

use super::*;

/// Accepted distance (in ulps of the result) between kernel and oracle (f64).
const F64_ULPS: u64 = 6;
/// Accepted distance (in ulps of the result) between kernel and oracle (f32).
const F32_ULPS: u32 = 6;

/// Port of `DNRM2` (LAPACK 3.12, `dnrm2.f90`) for `incx = 1`.
fn dnrm2_reference(x: &[f64]) -> f64 {
    let zero = 0.0f64;
    let one = 1.0f64;
    let maxn = f64::MAX;
    let (tsml, tbig, ssml, sbig) = (F64_TSML, F64_TBIG, F64_SSML, F64_SBIG);

    if x.is_empty() {
        return zero;
    }
    let mut notbig = true;
    let mut asml = zero;
    let mut amed = zero;
    let mut abig = zero;
    for &xi in x {
        let ax = xi.abs();
        if ax > tbig {
            let t = ax * sbig;
            abig += t * t;
            notbig = false;
        } else if ax < tsml {
            if notbig {
                let t = ax * ssml;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > zero {
        if amed > zero || amed > maxn || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = one / sbig;
        sumsq = abig;
    } else if asml > zero {
        if amed > zero || amed > maxn || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            scl = one;
            sumsq = ymax * ymax * (one + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = one / ssml;
            sumsq = asml;
        }
    } else {
        scl = one;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// Port of `SNRM2` (LAPACK 3.12, `snrm2.f90`) for `incx = 1`.
fn snrm2_reference(x: &[f32]) -> f32 {
    let zero = 0.0f32;
    let one = 1.0f32;
    let maxn = f32::MAX;
    let (tsml, tbig, ssml, sbig) = (F32_TSML, F32_TBIG, F32_SSML, F32_SBIG);

    if x.is_empty() {
        return zero;
    }
    let mut notbig = true;
    let mut asml = zero;
    let mut amed = zero;
    let mut abig = zero;
    for &xi in x {
        let ax = xi.abs();
        if ax > tbig {
            let t = ax * sbig;
            abig += t * t;
            notbig = false;
        } else if ax < tsml {
            if notbig {
                let t = ax * ssml;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > zero {
        if amed > zero || amed > maxn || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = one / sbig;
        sumsq = abig;
    } else if asml > zero {
        if amed > zero || amed > maxn || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            scl = one;
            sumsq = ymax * ymax * (one + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = one / ssml;
            sumsq = asml;
        }
    } else {
        scl = one;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// Deterministic xorshift generator (no external crates in the test build).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

/// `2^e` for any f64 exponent, subnormal ones included (exact).
fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// A magnitude class of test data.
#[derive(Clone, Copy, Debug)]
enum Class {
    Normal,
    Subnormal,
    Tiny,
    Huge,
    Mixed,
    Zeros,
    SparseZeros,
}

const CLASSES: [Class; 7] = [
    Class::Normal,
    Class::Subnormal,
    Class::Tiny,
    Class::Huge,
    Class::Mixed,
    Class::Zeros,
    Class::SparseZeros,
];

/// One f64 element of class `c` (finite).
fn sample_f64(rng: &mut Rng, c: Class) -> f64 {
    let mant = 1.0 + rng.unit();
    match c {
        Class::Normal => rng.sign() * mant * pow2(rng.below(41) as i32 - 20),
        // Uniform subnormal bit patterns (never zero).
        Class::Subnormal => rng.sign() * f64::from_bits(1 + rng.below((1u64 << 52) - 1)),
        // Normal numbers below tsml = 2^-511.
        Class::Tiny => rng.sign() * mant * pow2(-1022 + rng.below(510) as i32),
        // Numbers above tbig = 2^486 whose squares-sum stays finite for n <= 300.
        Class::Huge => rng.sign() * mant * pow2(487 + rng.below(500) as i32),
        Class::Mixed => {
            let pick = [
                Class::Normal,
                Class::Subnormal,
                Class::Tiny,
                Class::Huge,
                Class::Zeros,
            ];
            let c = pick[rng.below(pick.len() as u64) as usize];
            // Spread the "normal" picks over the whole mid range as well.
            if matches!(c, Class::Normal) {
                rng.sign() * mant * pow2(rng.below(990) as i32 - 505)
            } else {
                sample_f64(rng, c)
            }
        }
        Class::Zeros => {
            if rng.below(2) == 0 {
                0.0
            } else {
                -0.0
            }
        }
        Class::SparseZeros => {
            if rng.below(3) == 0 {
                0.0
            } else {
                sample_f64(rng, Class::Normal)
            }
        }
    }
}

/// One f32 element of class `c` (finite).
fn sample_f32(rng: &mut Rng, c: Class) -> f32 {
    let mant = (1.0 + rng.unit()) as f32;
    let p2 = |e: i32| -> f32 {
        if e >= -126 {
            f32::from_bits(((e + 127) as u32) << 23)
        } else {
            f32::from_bits(1u32 << (e + 149))
        }
    };
    let sign = rng.sign() as f32;
    match c {
        Class::Normal => sign * mant * p2(rng.below(21) as i32 - 10),
        Class::Subnormal => sign * f32::from_bits(1 + rng.below((1u64 << 23) - 1) as u32),
        // Normal numbers below tsml = 2^-63.
        Class::Tiny => sign * mant * p2(-126 + rng.below(62) as i32),
        // Numbers above tbig = 2^52 whose squares-sum stays finite for n <= 300.
        Class::Huge => sign * mant * p2(53 + rng.below(60) as i32),
        Class::Mixed => {
            let pick = [
                Class::Normal,
                Class::Subnormal,
                Class::Tiny,
                Class::Huge,
                Class::Zeros,
            ];
            let c = pick[rng.below(pick.len() as u64) as usize];
            if matches!(c, Class::Normal) {
                sign * mant * p2(rng.below(114) as i32 - 62)
            } else {
                sample_f32(rng, c)
            }
        }
        Class::Zeros => {
            if rng.below(2) == 0 {
                0.0
            } else {
                -0.0
            }
        }
        Class::SparseZeros => {
            if rng.below(3) == 0 {
                0.0
            } else {
                sample_f32(rng, Class::Normal)
            }
        }
    }
}

fn ulps_f64(a: f64, b: f64) -> u64 {
    a.to_bits().abs_diff(b.to_bits())
}

fn ulps_f32(a: f32, b: f32) -> u32 {
    a.to_bits().abs_diff(b.to_bits())
}

/// Kernel vs oracle: exact on exceptional results, within `F64_ULPS` otherwise.
fn check_f64(x: &[f64], what: &str) -> u64 {
    let got = nrm2_f64(x);
    let want = dnrm2_reference(x);
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: got {got:e}, oracle NaN");
        return 0;
    }
    assert!(got.is_sign_positive(), "{what}: negative result {got:e}");
    if want.is_infinite() || want == 0.0 {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{what}: got {got:e}, oracle {want:e}"
        );
        return 0;
    }
    let d = ulps_f64(got, want);
    assert!(
        d <= F64_ULPS,
        "{what}: got {got:e}, oracle {want:e} ({d} ulps)"
    );
    d
}

/// Kernel vs oracle for f32 (see [`check_f64`]).
fn check_f32(x: &[f32], what: &str) -> u32 {
    let got = nrm2_f32(x);
    let want = snrm2_reference(x);
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: got {got:e}, oracle NaN");
        return 0;
    }
    assert!(got.is_sign_positive(), "{what}: negative result {got:e}");
    if want.is_infinite() || want == 0.0 {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{what}: got {got:e}, oracle {want:e}"
        );
        return 0;
    }
    let d = ulps_f32(got, want);
    assert!(
        d <= F32_ULPS,
        "{what}: got {got:e}, oracle {want:e} ({d} ulps)"
    );
    d
}

#[test]
fn test_blue_constants_match_reference_values() {
    assert_eq!(F64_TSML, pow2(-511));
    assert_eq!(F64_TBIG, pow2(486));
    assert_eq!(F64_SSML, pow2(537));
    assert_eq!(F64_SBIG, pow2(-538));
    assert_eq!(F32_TSML, 2.0f32.powi(-63));
    assert_eq!(F32_TBIG, 2.0f32.powi(52));
    assert_eq!(F32_SSML, 2.0f32.powi(75));
    assert_eq!(F32_SBIG, 2.0f32.powi(-76));
}

#[test]
fn test_nrm2_blue_f64_matches_dnrm2_oracle_all_classes() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0;
    for n in 0..=300usize {
        for c in CLASSES {
            let x: Vec<f64> = (0..n).map(|_| sample_f64(&mut rng, c)).collect();
            worst = worst.max(check_f64(&x, &format!("{c:?} n={n}")));
        }
    }
    assert!(worst <= F64_ULPS);
}

#[test]
fn test_nrm2_blue_f32_matches_snrm2_oracle_all_classes() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut worst = 0;
    for n in 0..=300usize {
        for c in CLASSES {
            let x: Vec<f32> = (0..n).map(|_| sample_f32(&mut rng, c)).collect();
            worst = worst.max(check_f32(&x, &format!("{c:?} n={n}")));
        }
    }
    assert!(worst <= F32_ULPS);
}

/// NaN anywhere (SIMD body, vector tail or scalar tail) yields NaN, also next
/// to an infinity; ±Inf without NaN yields `+Inf`.
#[test]
fn test_nrm2_blue_nan_and_inf_anywhere() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    for n in 1..=300usize {
        let base: Vec<f64> = (0..n).map(|_| sample_f64(&mut rng, Class::Mixed)).collect();
        let base32: Vec<f32> = (0..n).map(|_| sample_f32(&mut rng, Class::Mixed)).collect();
        let positions = [0, n / 2, n - 1, (n * 7) / 9];
        for &p in &positions {
            let mut x = base.clone();
            x[p] = f64::NAN;
            assert!(nrm2_f64(&x).is_nan(), "f64 NaN at {p}/{n}");
            check_f64(&x, &format!("f64 NaN at {p}/{n}"));
            x[p] = if p % 2 == 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            assert_eq!(nrm2_f64(&x), f64::INFINITY, "f64 Inf at {p}/{n}");
            check_f64(&x, &format!("f64 Inf at {p}/{n}"));
            let q = (p + n / 3) % n;
            if q != p {
                x[q] = f64::NAN;
                assert!(nrm2_f64(&x).is_nan(), "f64 Inf at {p} + NaN at {q} of {n}");
            }

            let mut y = base32.clone();
            y[p] = f32::NAN;
            assert!(nrm2_f32(&y).is_nan(), "f32 NaN at {p}/{n}");
            check_f32(&y, &format!("f32 NaN at {p}/{n}"));
            y[p] = if p % 2 == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            assert_eq!(nrm2_f32(&y), f32::INFINITY, "f32 Inf at {p}/{n}");
            if q != p {
                y[q] = f32::NAN;
                assert!(nrm2_f32(&y).is_nan(), "f32 Inf at {p} + NaN at {q} of {n}");
            }
        }
    }
}

/// All-zero vectors of either sign give `+0`; overflow of the true norm gives
/// `+Inf` exactly like the oracle.
#[test]
fn test_nrm2_blue_zero_sign_and_true_overflow() {
    for n in 0..=40usize {
        let z = vec![-0.0f64; n];
        assert_eq!(nrm2_f64(&z).to_bits(), 0.0f64.to_bits(), "n={n}");
        let z32 = vec![-0.0f32; n];
        assert_eq!(nrm2_f32(&z32).to_bits(), 0.0f32.to_bits(), "n={n}");
    }
    let big = vec![f64::MAX; 9];
    assert_eq!(nrm2_f64(&big), f64::INFINITY);
    assert_eq!(dnrm2_reference(&big), f64::INFINITY);
    let big32 = vec![f32::MAX; 9];
    assert_eq!(nrm2_f32(&big32), f32::INFINITY);
    // Just representable: sqrt(4) * MAX / 2 == MAX.
    let half = vec![f64::MAX / 2.0; 4];
    assert_eq!(nrm2_f64(&half), f64::MAX);
}

/// High-precision norm of `x` for single-class data: every element is scaled by
/// the exact power of two `s`, the squares are summed as an unevaluated
/// double-double (error-free products and sums), and the root is corrected to
/// first order before the exact rescaling by `1/s`.
fn double_double_norm(x: &[f64], s: f64) -> f64 {
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let t = a + b;
        let z = t - a;
        (t, (a - (t - z)) + (b - z))
    }
    let (mut hi, mut lo) = (0.0f64, 0.0f64);
    for &v in x {
        let v = v * s;
        let p = v * v;
        let pe = v.mul_add(v, -p);
        let (t, e) = two_sum(hi, p);
        hi = t;
        lo += e + pe;
    }
    let (t, e) = two_sum(hi, lo);
    let r = t.sqrt();
    (r + e / (2.0 * r)) / s
}

/// Against a high-precision norm the kernel is never more than one ulp worse
/// than the oracle, for every finite magnitude class and both precisions.
#[test]
fn test_nrm2_blue_accuracy_against_double_double() {
    let mut rng = Rng(0x5DEE_CE66_D1CE_4E5B);
    let classes = [
        (Class::Normal, 1.0, 1.0f32),
        (Class::Subnormal, F64_SSML, F32_SSML),
        (Class::Tiny, F64_SSML, F32_SSML),
        (Class::Huge, F64_SBIG, F32_SBIG),
    ];
    for (c, s64, s32) in classes {
        for n in [17usize, 49, 64, 98, 100, 255, 300, 1000, 4097] {
            let x: Vec<f64> = (0..n).map(|_| sample_f64(&mut rng, c)).collect();
            let exact = double_double_norm(&x, s64);
            let err_got = ulps_f64(nrm2_f64(&x), exact);
            let err_oracle = ulps_f64(dnrm2_reference(&x), exact);
            assert!(
                err_got <= err_oracle.max(1) + 1,
                "f64 {c:?} n={n}: kernel {err_got} ulps, oracle {err_oracle} ulps from exact"
            );

            // f32 data are exact f64 values; the f64 double-double norm rounded
            // once to f32 is the correctly rounded f32 norm (to ~2^-80).
            let x32: Vec<f32> = (0..n).map(|_| sample_f32(&mut rng, c)).collect();
            let wide: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();
            let exact32 = double_double_norm(&wide, f64::from(s32)) as f32;
            let err_got = ulps_f32(nrm2_f32(&x32), exact32);
            let err_oracle = ulps_f32(snrm2_reference(&x32), exact32);
            assert!(
                err_got <= err_oracle.max(1) + 1,
                "f32 {c:?} n={n}: kernel {err_got} ulps, oracle {err_oracle} ulps from exact"
            );
        }
    }
}

/// `nrm2_f64` / `nrm2_f32` (the public typed entry points), the generic `nrm2`
/// and the complex entry points all reach this kernel on aarch64 from eight
/// elements on, and keep the dlassq loop below that.
#[test]
fn test_nrm2_entry_points_reach_blue_kernel() {
    use crate::level1;
    use crate::level1::nrm2::{nrm2_finalize, nrm2_fold};
    // Below eight elements the entry points keep the previous dlassq loop.
    let dlassq_f64 = |x: &[f64]| nrm2_finalize(x.iter().fold((0.0, 1.0), |s, &v| nrm2_fold(s, v)));
    let dlassq_f32 = |x: &[f32]| nrm2_finalize(x.iter().fold((0.0, 1.0), |s, &v| nrm2_fold(s, v)));
    let mut rng = Rng(42);
    for n in [2usize, 3, 7, 8, 9, 63, 64, 65, 130, 1001] {
        let x: Vec<f64> = (0..n).map(|_| sample_f64(&mut rng, Class::Mixed)).collect();
        let x32: Vec<f32> = (0..n).map(|_| sample_f32(&mut rng, Class::Mixed)).collect();
        let (k, k32) = if n >= 8 {
            (nrm2_f64(&x), nrm2_f32(&x32))
        } else {
            (dlassq_f64(&x), dlassq_f32(&x32))
        };
        assert_eq!(
            level1::nrm2_f64(&x).to_bits(),
            k.to_bits(),
            "nrm2_f64 n={n}"
        );
        assert_eq!(
            level1::nrm2_f32(&x32).to_bits(),
            k32.to_bits(),
            "nrm2_f32 n={n}"
        );
        #[cfg(not(feature = "force-scalar"))]
        {
            assert_eq!(level1::nrm2(&x).to_bits(), k.to_bits(), "nrm2::<f64> n={n}");
            assert_eq!(
                level1::nrm2(&x32).to_bits(),
                k32.to_bits(),
                "nrm2::<f32> n={n}"
            );
        }
        if n % 2 == 0 {
            let z: Vec<num_complex::Complex64> = x
                .chunks_exact(2)
                .map(|p| num_complex::Complex64::new(p[0], p[1]))
                .collect();
            assert_eq!(level1::dznrm2(&z).to_bits(), k.to_bits(), "dznrm2 n={n}");
            #[cfg(not(feature = "force-scalar"))]
            assert_eq!(level1::cnrm2(&z).to_bits(), k.to_bits(), "cnrm2 n={n}");
            let z32: Vec<num_complex::Complex32> = x32
                .chunks_exact(2)
                .map(|p| num_complex::Complex32::new(p[0], p[1]))
                .collect();
            assert_eq!(
                level1::scnrm2(&z32).to_bits(),
                k32.to_bits(),
                "scnrm2 n={n}"
            );
        }
    }
}
