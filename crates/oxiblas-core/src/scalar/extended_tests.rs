//! Tests of the double-double (`QuadFloat`) division and of the operations
//! built on it.
//!
//! Accuracy is measured against references that do not use the division
//! under test: the exact residual `a - q·b` (an error-free expansion of
//! FMA products, see [`exact_residual`]) and correctly rounded double-double
//! constants computed with 60-digit arithmetic (mpmath).

use num_traits::Float;

use super::QuadFloat;
use crate::scalar::{Field, Real};

/// `2^-106`, the unit roundoff squared of `f64` (`u²`).
const U2: f64 = 1.232_595_164_407_831e-32;

/// `a·b + c` with one rounding.
fn fma(a: f64, b: f64, c: f64) -> f64 {
    <f64 as Float>::mul_add(a, b, c)
}

/// Error-free sum: `a + b = s + e` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Error-free product: `a·b = p + e` exactly (no underflow).
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, fma(a, b, -p))
}

/// The two words of a `QuadFloat`.
fn words(x: QuadFloat) -> (f64, f64) {
    (x.inner().hi(), x.inner().lo())
}

/// A `QuadFloat` from a normalized pair of words.
fn dd(hi: f64, lo: f64) -> QuadFloat {
    QuadFloat::from(twofloat::TwoFloat::new_add(hi, lo))
}

/// A `QuadFloat` from the bit patterns of its words.
fn dd_bits(hi: u64, lo: u64) -> QuadFloat {
    dd(f64::from_bits(hi), f64::from_bits(lo))
}

/// The exact sum of `terms`, rounded once at the end (to within a few ulps).
///
/// Shewchuk's GROW-EXPANSION keeps the components nonoverlapping, in
/// increasing magnitude, and summing exactly to the terms added so far;
/// summing them smallest first then rounds only at the end.
fn expansion_sum(terms: &[f64]) -> f64 {
    let mut expansion: Vec<f64> = Vec::new();
    for &t in terms {
        let mut carry = t;
        for c in expansion.iter_mut() {
            let (s, e) = two_sum(carry, *c);
            *c = e;
            carry = s;
        }
        expansion.push(carry);
    }
    expansion.iter().fold(0.0, |acc, &c| acc + c)
}

/// `a - q·b`, evaluated exactly from the ten error-free terms
/// `a_hi, a_lo, -(q_i·b_j)` and rounded once at the end.
///
/// `(a - q·b) / a = (a/b - q) / (a/b)`, so this divided by `|a|` is exactly
/// the relative error of `q` as an approximation of `a/b`.
fn exact_residual(a: QuadFloat, b: QuadFloat, q: QuadFloat) -> f64 {
    let (ah, al) = words(a);
    let (bh, bl) = words(b);
    let (qh, ql) = words(q);
    let mut terms = vec![ah, al];
    for (x, y) in [(qh, bh), (qh, bl), (ql, bh), (ql, bl)] {
        let (p, e) = two_prod(x, y);
        terms.push(-p);
        terms.push(-e);
    }
    expansion_sum(&terms)
}

/// Relative error of `h` as `sqrt(x² + y²)`: the exact residual
/// `h² - x² - y²` is `2·h²` times it (to first order).
fn hypot_error(x: QuadFloat, y: QuadFloat, h: QuadFloat) -> f64 {
    let mut terms = Vec::new();
    for (v, sign) in [(h, 1.0), (x, -1.0), (y, -1.0)] {
        let (vh, vl) = words(v);
        for (p, q) in [(vh, vh), (vh, vl), (vl, vh), (vl, vl)] {
            let (prod, err) = two_prod(p, q);
            terms.push(sign * prod);
            terms.push(sign * err);
        }
    }
    let hh = words(h).0;
    (expansion_sum(&terms) / (2.0 * hh * hh)).abs()
}

/// Relative error of `q` as an approximation of `a / b`.
fn quotient_error(a: QuadFloat, b: QuadFloat, q: QuadFloat) -> f64 {
    (exact_residual(a, b, q) / words(a).0).abs()
}

/// Relative error of `x` against the correctly rounded double-double `r`.
fn rel_err(x: QuadFloat, r: QuadFloat) -> f64 {
    let (xh, xl) = words(x);
    let (rh, rl) = words(r);
    // (xh - rh) is exact when xh and rh are close (Sterbenz).
    (((xh - rh) + (xl - rl)) / rh).abs()
}

/// Deterministic split-mix PRNG (no external dependencies).
struct Prng(u64);

impl Prng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// A random double-double with magnitude around `2^±max_exp`, and a low
    /// word anywhere in `[-ulp/2, ulp/2]` of the high word.
    fn next_dd(&mut self, max_exp: i32) -> QuadFloat {
        let e = (self.next_f64() * f64::from(max_exp)).round() as i32;
        let mut hi = self.next_f64() * 2f64.powi(e);
        if hi == 0.0 {
            hi = 1.0;
        }
        let ulp = f64::from_bits(hi.abs().to_bits() + 1) - hi.abs();
        dd(hi, 0.5 * ulp * self.next_f64())
    }
}

#[test]
fn div_matches_correctly_rounded_constants() {
    // Correctly rounded double-double quotients (mpmath): q = hi + lo.
    let cases: [(f64, f64, u64, u64); 3] = [
        (1.0, 3.0, 0x3FD5_5555_5555_5555, 0x3C75_5555_5555_5555),
        (2.0, 3.0, 0x3FE5_5555_5555_5555, 0x3C85_5555_5555_5555),
        (1.0, 7.0, 0x3FC2_4924_9249_2492, 0x3C62_4924_9249_2492),
    ];
    for (a, b, hi, lo) in cases {
        let q = QuadFloat::from(a) / QuadFloat::from(b);
        assert_eq!(
            words(q),
            (f64::from_bits(hi), f64::from_bits(lo)),
            "{a}/{b} = {:?}, expected the correctly rounded double-double",
            words(q)
        );
        let err = quotient_error(QuadFloat::from(a), QuadFloat::from(b), q);
        assert!(err <= 0.5 * U2, "{a}/{b}: relative error {err:e}");
    }
}

#[test]
fn div_random_pairs_are_double_double_accurate() {
    let mut rng = Prng(0x00D1_7D1D);
    let mut worst = 0.0f64;
    for i in 0..4000 {
        let (a, b) = match i % 3 {
            // plain doubles (low words zero)
            0 => (
                QuadFloat::from(rng.next_f64() * 1e3),
                QuadFloat::from(rng.next_f64() * 7.0 + 8.0),
            ),
            // genuine double-doubles over a wide range of magnitudes
            1 => (rng.next_dd(60), rng.next_dd(60)),
            // divisor close to the dividend (quotient near 1)
            _ => {
                let a = rng.next_dd(20);
                let eps = QuadFloat::from(rng.next_f64() * 2f64.powi(-30));
                (a, a + a * eps)
            }
        };
        let q = a / b;
        let err = quotient_error(a, b, q);
        worst = worst.max(err);
        assert!(
            err <= 2.0 * U2,
            "{:?}/{:?} = {:?}: relative error {err:e} = {} u²",
            words(a),
            words(b),
            words(q),
            err / U2
        );
    }
    // Measured worst over these 4000 pairs: below 0.5 u² (Apple M3).
    assert!(worst > 0.0);
}

#[test]
fn div_by_self_is_exactly_one() {
    let mut rng = Prng(0x5E1F);
    for _ in 0..500 {
        let x = rng.next_dd(100);
        assert_eq!(words(x / x), (1.0, 0.0), "{:?} / itself", words(x));
    }
    assert_eq!(
        words(QuadFloat::from(3.0) / QuadFloat::from(3.0)),
        (1.0, 0.0)
    );
}

#[test]
fn div_is_scale_invariant_at_extreme_exponents() {
    // Powers of two scale the quotient exactly, so the extreme-exponent
    // quotients must be the exact scalings of the moderate one.
    let mut rng = Prng(0xE0E0);
    for _ in 0..200 {
        let a = rng.next_dd(4);
        let b = rng.next_dd(4);
        let q = a / b;
        for (ka, kb) in [
            (900, 0),
            (0, 900),
            (-900, 0),
            (0, -900),
            (500, -400),
            (-400, 500),
        ] {
            let sa = QuadFloat::from(2f64.powi(ka));
            let sb = QuadFloat::from(2f64.powi(kb));
            let qs = (a * sa) / (b * sb);
            let expected = q * QuadFloat::from(2f64.powi(ka - kb));
            assert_eq!(words(qs), words(expected), "scale 2^{ka}/2^{kb}");
        }
    }
    // A dividend at the top of the range with a divisor just below 1: the
    // quotient exceeds f64::MAX but is a finite double-double.
    let big = QuadFloat::from(f64::MAX);
    let below_one = dd(1.0, -(2f64.powi(-60)));
    let q = big / below_one;
    assert!(Float::is_finite(q), "{:?}", words(q));
    assert!(quotient_error(big, below_one, q) <= 2.0 * U2);
}

#[test]
fn div_assign_recip_and_powi_use_the_accurate_division() {
    let mut rng = Prng(0xA55);
    for _ in 0..300 {
        let a = rng.next_dd(30);
        let b = rng.next_dd(30);
        let mut c = a;
        c /= b;
        assert_eq!(words(c), words(a / b), "DivAssign differs from Div");

        let one = QuadFloat::from(1.0);
        let inv = one / b;
        assert_eq!(words(Float::recip(b)), words(inv), "Float::recip");
        assert_eq!(words(Field::recip(b)), words(inv), "Field::recip");
        assert_eq!(words(Float::powi(b, -1)), words(inv), "Float::powi(-1)");
        assert_eq!(words(Field::powi(b, -1)), words(inv), "Field::powi(-1)");
        assert!(quotient_error(one, b, inv) <= 2.0 * U2);
    }
    // 3^-2 = 1/9 to double-double accuracy.
    let ninth = Float::powi(QuadFloat::from(3.0), -2);
    assert!(quotient_error(QuadFloat::from(1.0), QuadFloat::from(9.0), ninth) <= 2.0 * U2);
    // i32::MIN must not overflow the exponent negation (twofloat's `powi`
    // negates it and panics with overflow checks on).
    assert_eq!(
        words(Float::powi(QuadFloat::from(1.0), i32::MIN)),
        (1.0, 0.0)
    );
    assert_eq!(
        words(Float::powi(QuadFloat::from(-1.0), i32::MIN)),
        (1.0, 0.0)
    );
    assert_eq!(
        words(Field::powi(QuadFloat::from(1.0), i32::MIN)),
        (1.0, 0.0)
    );
}

/// Expected IEEE-754 class of a special-value result.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Ieee {
    Nan,
    PosInf,
    NegInf,
    PosZero,
    NegZero,
}

fn class(x: QuadFloat) -> Option<Ieee> {
    let hi = x.inner().hi();
    if Float::is_nan(x) {
        Some(Ieee::Nan)
    } else if hi == f64::INFINITY {
        Some(Ieee::PosInf)
    } else if hi == f64::NEG_INFINITY {
        Some(Ieee::NegInf)
    } else if hi == 0.0 && x.inner().lo() == 0.0 {
        Some(if hi.is_sign_negative() {
            Ieee::NegZero
        } else {
            Ieee::PosZero
        })
    } else {
        None
    }
}

#[test]
fn div_special_values_follow_ieee() {
    use Ieee::{Nan, NegInf, NegZero, PosInf, PosZero};
    let inf = f64::INFINITY;
    let nan = f64::NAN;
    let cases: [(f64, f64, Ieee); 23] = [
        (1.0, 0.0, PosInf),
        (1.0, -0.0, NegInf),
        (-1.0, 0.0, NegInf),
        (-1.0, -0.0, PosInf),
        (0.0, 0.0, Nan),
        (-0.0, 0.0, Nan),
        (inf, 2.0, PosInf),
        (inf, -2.0, NegInf),
        (-inf, 2.0, NegInf),
        (2.0, inf, PosZero),
        (2.0, -inf, NegZero),
        (-2.0, inf, NegZero),
        (inf, inf, Nan),
        (-inf, inf, Nan),
        (nan, 2.0, Nan),
        (2.0, nan, Nan),
        (0.0, 3.0, PosZero),
        (-0.0, 3.0, NegZero),
        (0.0, -3.0, NegZero),
        (-0.0, -3.0, PosZero),
        (0.0, inf, PosZero),
        (f64::MAX, 0.5, PosInf),
        (f64::MIN_POSITIVE, 1e300, PosZero),
    ];
    for (a, b, expected) in cases {
        let q = QuadFloat::from(a) / QuadFloat::from(b);
        assert_eq!(class(q), Some(expected), "{a} / {b} = {:?}", words(q));
        let mut c = QuadFloat::from(a);
        c /= QuadFloat::from(b);
        assert_eq!(class(c), Some(expected), "{a} /= {b} = {:?}", words(c));
    }
    let recips: [(f64, Ieee); 5] = [
        (0.0, PosInf),
        (-0.0, NegInf),
        (inf, PosZero),
        (-inf, NegZero),
        (nan, Nan),
    ];
    for (x, expected) in recips {
        let q = QuadFloat::from(x);
        assert_eq!(class(Float::recip(q)), Some(expected), "Float::recip({x})");
        assert_eq!(class(Field::recip(q)), Some(expected), "Field::recip({x})");
        assert_eq!(class(Float::powi(q, -1)), Some(expected), "powi({x}, -1)");
    }
    assert_eq!(class(Float::powi(QuadFloat::from(-0.0), -2)), Some(PosInf));
    assert_eq!(class(Float::powi(QuadFloat::from(0.0), -3)), Some(PosInf));
}

#[test]
fn hypot_scaled_path_is_double_double_accurate() {
    // Above the direct-formula window hypot divides min/max. The operands are
    // scaled by 2^600, so the result scaled back by 2^-600 (exactly) must be
    // hypot of the unscaled operands, checked with the exact residual.
    let mut rng = Prng(0x4B07);
    let up = QuadFloat::from(2f64.powi(600));
    let down = QuadFloat::from(2f64.powi(-600));
    let mut worst = (0.0f64, 0.0f64);
    for _ in 0..500 {
        let x = rng.next_dd(3);
        let y = rng.next_dd(3);
        let scaled = Float::hypot(x * up, y * up) * down;
        let direct = Float::hypot(x, y);
        let err_scaled = hypot_error(x, y, scaled);
        let err_direct = hypot_error(x, y, direct);
        worst = (worst.0.max(err_scaled), worst.1.max(err_direct));
        assert!(
            err_scaled <= 8.0 * U2,
            "hypot scaled path: relative error {err_scaled:e} = {} u²",
            err_scaled / U2
        );
        assert!(err_direct <= 8.0 * U2, "hypot direct path: {err_direct:e}");
    }
}

#[test]
fn tan_atan_atan2_are_not_limited_by_the_division() {
    // Correctly rounded double-double values (mpmath, 60 digits).
    let tan_cases: [(f64, u64, u64); 3] = [
        (1.0, 0x3FF8_EB24_5CBE_E3A6, 0xBC91_D4CE_0AFB_373B),
        (-2.5, 0x3FE7_E79B_4E00_BB15, 0xBC84_D92D_DFF3_1690),
        (0.3, 0x3FD3_CC2A_44E2_9998, 0xBC7F_AD79_4F03_F7B9),
    ];
    for (x, hi, lo) in tan_cases {
        let t = Float::tan(QuadFloat::from(x));
        let err = rel_err(t, dd_bits(hi, lo));
        // Bounded by twofloat's sin/cos kernels (~1e-21), not the division.
        assert!(err <= 1e-19, "tan({x}): relative error {err:e}");
    }
    let atan_cases: [(f64, u64, u64); 5] = [
        (0.3, 0x3FD2_A73A_661E_AF06, 0xBC72_F6C1_B5C5_F02C),
        (0.6, 0x3FE1_4B1D_D5F9_0CE1, 0xBC7B_DC3D_8B25_F2A4),
        (0.9, 0x3FE7_7338_A806_03BE, 0x3C8E_95CD_918F_CF3E),
        (2.0, 0x3FF1_B6E1_92EB_BE44, 0x3C9B_1B46_6A88_828E),
        (-10.0, 0xBFF7_89BD_2C16_0054, 0x3C9F_4550_3CCA_D255),
    ];
    for (x, hi, lo) in atan_cases {
        let a = Float::atan(QuadFloat::from(x));
        let err = rel_err(a, dd_bits(hi, lo));
        assert!(err <= 1e-21, "atan({x}): relative error {err:e}");
    }
    let atan2_cases: [(f64, f64, u64, u64); 3] = [
        (1.0, 3.0, 0x3FD4_978F_A326_9EE1, 0x3C62_419A_87F2_A458),
        (-2.0, -0.5, 0xBFFD_0D6A_1369_BD34, 0x3C6A_2360_2A65_700C),
        (5.0, -7.0, 0x4004_2BB5_F776_EB80, 0x3C77_5C72_B745_CAB4),
    ];
    for (y, x, hi, lo) in atan2_cases {
        let r = dd_bits(hi, lo);
        let a = Float::atan2(QuadFloat::from(y), QuadFloat::from(x));
        let b = Real::atan2(QuadFloat::from(y), QuadFloat::from(x));
        assert!(
            rel_err(a, r) <= 1e-21,
            "Float::atan2({y}, {x}) = {:?}",
            words(a)
        );
        assert!(
            rel_err(b, r) <= 1e-21,
            "Real::atan2({y}, {x}) = {:?}",
            words(b)
        );
    }
    // atan of an infinite argument is ±π/2, and atan2's axis cases are kept.
    let half_pi = Float::atan(QuadFloat::from(f64::INFINITY));
    assert!((half_pi.inner().hi() - core::f64::consts::FRAC_PI_2).abs() <= f64::EPSILON);
    assert!(Float::atan(QuadFloat::from(f64::NAN)).is_nan());
    assert!(Float::tan(QuadFloat::from(f64::NAN)).is_nan());
    let zero = Float::atan2(QuadFloat::from(0.0), QuadFloat::from(2.0));
    assert_eq!(words(zero), (0.0, 0.0));
    let up = Float::atan2(QuadFloat::from(3.0), QuadFloat::from(0.0));
    assert_eq!(up.inner().hi(), core::f64::consts::FRAC_PI_2);
}

#[test]
fn log_family_divides_accurately() {
    // log(x, x) and log2(2) are exact ratios of identical logarithms.
    let mut rng = Prng(0x1061);
    for _ in 0..100 {
        let x = QuadFloat::from(rng.next_f64().abs() * 50.0 + 1.5);
        assert_eq!(words(Float::log(x, x)), (1.0, 0.0), "log(x, x)");
    }
    assert_eq!(words(Float::log2(QuadFloat::from(2.0))), (1.0, 0.0));
}
