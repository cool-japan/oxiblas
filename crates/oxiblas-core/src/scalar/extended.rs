//! Extended precision type implementations (f16 and QuadFloat/f128).

#[cfg(any(feature = "f16", feature = "f128"))]
use super::traits::{Field, Real, Scalar};

// =============================================================================
// f16 support (half-precision)
// =============================================================================

#[cfg(feature = "f16")]
use half::f16;

#[cfg(feature = "f16")]
impl Scalar for f16 {
    type Real = f16;

    #[inline]
    fn abs(self) -> Self::Real {
        if self < f16::ZERO { -self } else { self }
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn is_real() -> bool {
        true
    }

    #[inline]
    fn real(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag(self) -> Self::Real {
        f16::ZERO
    }

    #[inline]
    fn from_real_imag(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline]
    fn abs_sq(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn epsilon() -> Self::Real {
        f16::EPSILON
    }

    #[inline]
    fn min_positive() -> Self::Real {
        f16::MIN_POSITIVE
    }

    #[inline]
    fn max_value() -> Self::Real {
        f16::MAX
    }
}

#[cfg(feature = "f16")]
impl Real for f16 {
    #[inline]
    fn sqrt(self) -> Self {
        f16::from_f32(Real::sqrt(self.to_f32()))
    }

    #[inline]
    fn ln(self) -> Self {
        f16::from_f32(Real::ln(self.to_f32()))
    }

    #[inline]
    fn exp(self) -> Self {
        f16::from_f32(Real::exp(self.to_f32()))
    }

    #[inline]
    fn sin(self) -> Self {
        f16::from_f32(Real::sin(self.to_f32()))
    }

    #[inline]
    fn cos(self) -> Self {
        f16::from_f32(Real::cos(self.to_f32()))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        f16::from_f32(Real::atan2(self.to_f32(), other.to_f32()))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        f16::from_f32(Real::powf(self.to_f32(), n.to_f32()))
    }

    #[inline]
    fn signum(self) -> Self {
        // Inherent IEEE-754 style signum: +/-1 by sign bit (zeros included), NaN stays NaN.
        f16::signum(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f16::from_f32(Real::mul_add(self.to_f32(), a.to_f32(), b.to_f32()))
    }

    #[inline]
    fn floor(self) -> Self {
        f16::from_f32(Real::floor(self.to_f32()))
    }

    #[inline]
    fn ceil(self) -> Self {
        f16::from_f32(Real::ceil(self.to_f32()))
    }

    #[inline]
    fn round(self) -> Self {
        f16::from_f32(Real::round(self.to_f32()))
    }

    #[inline]
    fn trunc(self) -> Self {
        f16::from_f32(Real::trunc(self.to_f32()))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        f16::from_f32(Real::hypot(self.to_f32(), other.to_f32()))
    }
}

#[cfg(feature = "f16")]
impl Field for f16 {
    #[inline]
    fn mul_conj(self, other: Self) -> Self {
        self * other
    }

    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self * other
    }

    #[inline]
    fn recip(self) -> Self {
        f16::ONE / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        f16::from_f32(Field::powi(self.to_f32(), n))
    }
}

// =============================================================================
// QuadFloat (f128 / double-double) support
// =============================================================================

#[cfg(feature = "f128")]
use core::fmt::Display;
#[cfg(feature = "f128")]
use core::iter::Sum;
#[cfg(feature = "f128")]
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
#[cfg(feature = "f128")]
use num_traits::{Float, FromPrimitive, One, Zero};
#[cfg(feature = "f128")]
use twofloat::TwoFloat;
#[cfg(feature = "f128")]
use twofloat::consts::{FRAC_PI_2, FRAC_PI_4, LN_10, PI};

/// Quad-precision floating-point type using double-double arithmetic.
///
/// This newtype wraps `TwoFloat` from the `twofloat` crate, which provides
/// approximately 106 bits of mantissa precision (31 decimal digits) using
/// double-double arithmetic. This gives quadruple precision (similar to IEEE 754
/// binary128) without requiring platform-specific quadmath libraries.
///
/// # Features
///
/// - Cross-platform pure Rust implementation
/// - ~31 decimal digits of precision
/// - All standard mathematical operations (sin, cos, exp, ln, etc.)
/// - Compatible with OxiBLAS scalar traits
///
/// # Accuracy
///
/// `+`, `-`, `*`, `/`, `recip` and `sqrt` are accurate to a few units of
/// `2^-106` relative (`powi(n)` to about `|n|` units, as repeated squaring
/// compounds the rounding of each product). Division (and everything built
/// on it) uses FMA-based exact residuals; it does not use
/// `TwoFloat / TwoFloat`, which is only double precision in twofloat 0.8
/// (see the `Div` implementation).
/// The elementary functions (`exp`, `ln`, `sin`, `cos`, …) are twofloat's
/// own kernels and are in general considerably less accurate than the
/// arithmetic. Special values of the division follow IEEE 754:
/// `x / ±0 = ±∞`, `x / ±∞ = ±0`, `0 / 0 = ∞ / ∞ = NaN`, with the IEEE sign
/// rules.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "f128")] {
/// use num_traits::Float;
/// use oxiblas_core::scalar::QuadFloat;
///
/// let x = QuadFloat::from(2.0);
/// let y = x.sqrt();
/// assert!((y * y - x).abs() < QuadFloat::from(1e-30));
/// # }
/// ```
#[cfg(feature = "f128")]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[repr(transparent)]
pub struct QuadFloat(TwoFloat);

#[cfg(feature = "f128")]
impl QuadFloat {
    /// Create a new QuadFloat from a f64
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(TwoFloat::from_f64(value))
    }

    /// Get the underlying TwoFloat
    #[inline]
    pub const fn inner(self) -> TwoFloat {
        self.0
    }
}

// -----------------------------------------------------------------------------
// Double-double-aware integer rounding and helpers
// -----------------------------------------------------------------------------
//
// A normalized `TwoFloat` stores a value as two non-overlapping `f64` words
// `hi + lo`, where `hi` is the correctly rounded `f64` nearest the true value
// and `lo` is the exact rounding error, so `|lo| <= 0.5 * ulp(hi)`. Rounding the
// value to an integer therefore cannot be done on `hi` alone: when `hi` is
// itself an integer the entire fractional part lives in `lo`, and when `hi` is a
// half-integer the sign of `lo` decides how a tie falls. Each helper below
// derives the case analysis from that invariant and renormalizes the resulting
// pair with an error-free `two_sum` (`TwoFloat::new_add`).
#[cfg(feature = "f128")]
impl QuadFloat {
    /// Largest integer `<= self`, at double-double precision (`f64::floor`).
    ///
    /// WHY the split: if `hi` is not an integer, the true value stays on the
    /// same side of every integer as `hi` — the low word is at most half a ULP,
    /// and had it been able to cross an integer boundary `hi` would already have
    /// rounded to that integer — so `floor(hi+lo) == floor(hi)`. If `hi` is an
    /// integer, `floor(hi+lo) = hi + floor(lo)`, which we renormalize.
    #[inline]
    fn dd_floor(self) -> Self {
        let hi = self.0.hi();
        let floored_hi = Real::floor(hi);
        if floored_hi != hi {
            QuadFloat(TwoFloat::from_f64(floored_hi))
        } else {
            QuadFloat(TwoFloat::new_add(hi, Real::floor(self.0.lo())))
        }
    }

    /// Smallest integer `>= self`, at double-double precision (`f64::ceil`).
    ///
    /// Mirror image of [`dd_floor`](Self::dd_floor).
    #[inline]
    fn dd_ceil(self) -> Self {
        let hi = self.0.hi();
        let ceiled_hi = Real::ceil(hi);
        if ceiled_hi != hi {
            QuadFloat(TwoFloat::from_f64(ceiled_hi))
        } else {
            QuadFloat(TwoFloat::new_add(hi, Real::ceil(self.0.lo())))
        }
    }

    /// Sign of the value with `f32::signum` semantics: NaN for NaN, otherwise
    /// +1 or -1 by the sign of the high word (zeros and infinities included).
    ///
    /// Comparisons on `TwoFloat` mis-order invalid values (e.g. -inf), so the
    /// sign is read from the high word directly.
    #[inline]
    fn dd_signum(self) -> Self {
        let hi = self.0.hi();
        if hi.is_nan() {
            QuadFloat(TwoFloat::from_f64(f64::NAN))
        } else if hi.is_sign_negative() {
            QuadFloat(TwoFloat::from_f64(-1.0))
        } else {
            QuadFloat(TwoFloat::from_f64(1.0))
        }
    }

    /// Truncation toward zero, at double-double precision (`f64::trunc`).
    ///
    /// Truncation is floor for non-negative values and ceil for negative ones;
    /// the sign of a normalized double-double equals the sign of its high word.
    #[inline]
    fn dd_trunc(self) -> Self {
        if self.0.is_sign_negative() {
            self.dd_ceil()
        } else {
            self.dd_floor()
        }
    }

    /// Round half away from zero, at double-double precision (`f64::round`).
    ///
    /// WHY the tie handling: a genuine tie (value exactly `k + 0.5`) can arise
    /// either from `hi` being a half-integer with `lo == 0`, or from `hi` being
    /// an integer with `lo` a half-integer. Rounding `hi` (or `lo`) in isolation
    /// with `f64::round` breaks the tie away from *that word's* zero, which is
    /// the wrong direction whenever the word's sign differs from the whole
    /// value's sign; those cases are corrected explicitly.
    #[inline]
    fn dd_round(self) -> Self {
        let hi = self.0.hi();
        let lo = self.0.lo();
        let rounded_hi = Real::round(hi);
        if rounded_hi != hi {
            // `hi` is not an integer, so `round(hi+lo) == round(hi)` unless `hi`
            // is exactly a half-integer (its own tie), in which case the low
            // word decides which side of the half-way point the value lies on.
            if (hi - rounded_hi).abs() == 0.5 {
                if hi > 0.0 && lo < 0.0 {
                    QuadFloat(TwoFloat::from_f64(rounded_hi - 1.0))
                } else if hi < 0.0 && lo > 0.0 {
                    QuadFloat(TwoFloat::from_f64(rounded_hi + 1.0))
                } else {
                    QuadFloat(TwoFloat::from_f64(rounded_hi))
                }
            } else {
                QuadFloat(TwoFloat::from_f64(rounded_hi))
            }
        } else {
            // `hi` is an integer; the fractional part is entirely in `lo`.
            let mut rounded_lo = Real::round(lo);
            // On a tie in `lo` whose away-from-zero direction disagrees with the
            // whole value's away-from-zero direction (opposite signs), pick the
            // neighbor lying on the whole value's side, i.e. `trunc(lo)`.
            if (lo - Real::trunc(lo)).abs() == 0.5 && (lo < 0.0) != (hi < 0.0) {
                rounded_lo = Real::trunc(lo);
            }
            QuadFloat(TwoFloat::new_add(hi, rounded_lo))
        }
    }

    /// Fractional part `self - trunc(self)`, at double-double precision.
    ///
    /// Matches `f64::fract`: the result carries the sign of `self`. The integer
    /// part is exact, so the subtraction is a well-conditioned double-double
    /// operation.
    #[inline]
    fn dd_fract(self) -> Self {
        self - self.dd_trunc()
    }

    /// Overflow-safe Euclidean length `sqrt(self^2 + other^2)`.
    ///
    /// WHY the two paths: the direct form `sqrt(a^2 + b^2)` is correctly rounded
    /// at full double-double precision but overflows to infinity once the larger
    /// operand exceeds `sqrt(MAX)` (and loses precision to subnormals when both
    /// are tiny), even when the true result is finite. When the larger magnitude
    /// is in the safe window we therefore use the direct form; outside it we fall
    /// back to the scaled form `max * sqrt(1 + (min/max)^2)`, whose squared term
    /// is bounded to `[0, 1]` so the only overflow that can occur is a genuine
    /// one. (The scaled form adds a division and a multiplication, each
    /// rounding at the `2^-106` level, so it is used only when unavoidable.)
    #[inline]
    fn dd_hypot(self, other: Self) -> Self {
        // `sqrt(f64::MAX / 2)`, with headroom so `a*a + b*b` cannot overflow.
        const HYPOT_MAX_SAFE: f64 = 2.9e153;
        // Below this the squares would start degrading into the subnormal range.
        const HYPOT_MIN_SAFE: f64 = 1e-150;
        let a = QuadFloat(self.0.abs());
        let b = QuadFloat(other.0.abs());
        // IEEE-754 hypot special cases: an infinity dominates (even over a NaN),
        // then a NaN propagates.
        if a.0.is_infinite() || b.0.is_infinite() {
            return QuadFloat(TwoFloat::INFINITY);
        }
        if a.0.is_nan() || b.0.is_nan() {
            return QuadFloat(TwoFloat::NAN);
        }
        let (max, min) = if a >= b { (a, b) } else { (b, a) };
        let max_hi = max.0.hi();
        if max_hi == 0.0 {
            return QuadFloat::from(0.0);
        }
        if (HYPOT_MIN_SAFE..HYPOT_MAX_SAFE).contains(&max_hi) {
            let sum = self * self + other * other;
            QuadFloat(sum.0.sqrt())
        } else {
            let ratio = min / max;
            let radicand = QuadFloat::from(1.0) + ratio * ratio;
            max * QuadFloat(radicand.0.sqrt())
        }
    }

    /// Real cube root, at double-double precision (`f64::cbrt`).
    ///
    /// WHY not `powf(1/3)`: raising a negative base to a fractional power is
    /// NaN, yet the real cube root of a negative number is a well-defined
    /// negative real. We reduce to `|self|` and restore the sign via
    /// `cbrt(-x) = -cbrt(x)`.
    ///
    /// WHY a division-free Newton: twofloat's `TwoFloat / TwoFloat` division and
    /// its `powf`/`exp`/`ln` are not full double-double precision, so we refine
    /// the *inverse* cube root `r = x^(-1/3)` with the iteration
    /// `r <- r * (4 - x*r^3) / 3`, which uses only multiplications and an exact
    /// `TwoFloat / f64` division by 3. Two quadratically convergent steps lift
    /// the ~2^-53 `f64` seed to full ~2^-106 precision; then `cbrt(x) = x * r^2`.
    #[inline]
    fn dd_cbrt(self) -> Self {
        if !self.0.is_finite() {
            // cbrt(+-inf) = +-inf, cbrt(NaN) = NaN.
            return self;
        }
        if self == QuadFloat::from(0.0) {
            // Preserve the sign of zero (cbrt(-0.0) = -0.0).
            return self;
        }
        let negative = self.0.is_sign_negative();
        let x = self.0.abs();
        let four = TwoFloat::from_f64(4.0);
        let mut r = TwoFloat::from_f64((1.0 / x.hi()).cbrt());
        r = r * ((four - x * (r * r * r)) / 3.0_f64);
        r = r * ((four - x * (r * r * r)) / 3.0_f64);
        let result = QuadFloat(x * r * r);
        if negative { -result } else { result }
    }
}

// -----------------------------------------------------------------------------
// Double-double division with FMA residuals
// -----------------------------------------------------------------------------
//
// WHY not twofloat's division: twofloat 0.8.4 implements `TwoFloat / TwoFloat`
// (and `f64 / TwoFloat`, hence `TwoFloat::recip` and negative `powi`) as
// Algorithm 18 of Joldes, Muller & Popescu (2017). That algorithm starts from
// the residual `1 - b_hi·fl(1/b_hi)`, which it requires to be EXACT (an FMA);
// twofloat forms it as `1.0 - rhs.hi * th` with a rounded product. For
// `b = 3`, `3·fl(1/3)` rounds to exactly 1, the residual vanishes and
// `1/3` comes out as `fl(1/3) + 0`: every quotient is only double precision
// (relative error up to 2^-53), and `3/3` is not even 1.
//
// [`dd_div`] instead computes the quotient digit by digit (as the QD library's
// `accurate_div`), with every residual formed exactly through `f64::mul_add`:
// the remainder `a - q·b` of a correctly rounded quotient `q = fl(a/b)` is
// representable, and an FMA delivers it without rounding.

/// `a·b + c` with a single rounding.
///
/// `num_traits` maps this to `f64::mul_add` with `std` (a hardware FMA on
/// aarch64 and FMA-capable x86_64, a correctly rounded software `fma`
/// elsewhere) and to `libm::fma` without `std`; both are exact up to the one
/// final rounding, which every error-free transformation below relies on.
#[cfg(feature = "f128")]
#[inline]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    <f64 as Float>::mul_add(a, b, c)
}

/// Knuth's two-sum: `a + b = s + e` exactly, for any magnitudes (barring
/// overflow).
#[cfg(feature = "f128")]
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// FMA two-product: `a·b = p + e` exactly (barring underflow).
#[cfg(feature = "f128")]
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, fma(a, b, -p))
}

/// Double-double quotient `a / b`, accurate to about `2^-107` relative
/// (below half a unit of the low word for every pair measured).
///
/// The quotient is built from three `f64` digits:
///
/// 1. `q1 = fl(a_hi / b_hi)`; its remainder `a - q1·b` is assembled from the
///    exact FMA remainder `a_hi - q1·b_hi`, `a_lo` and the exact product
///    `q1·b_lo`, so it is known to `O(2^-159·|a|)`;
/// 2. `q2 = fl(r_hi / b_hi)` with the remainder `r - q2·b` formed the same
///    way (its terms are all `O(2^-106·|a|)`, plain sums suffice);
/// 3. `q3` corrects the `O(2^-106)` error left by `q2`.
///
/// `q1 + q2 + q3` is then renormalized. The digits fold in everything the
/// one-step algorithms drop (the low words of `a` and `b` and the rounding
/// of `q2`), so the only error left is the final rounding into two words.
///
/// Special values follow IEEE 754 through the leading digit: when `q1` is not
/// finite or is zero (`x / ±0`, `±∞ / x`, `x / ±∞`, `0 / x`, `0 / 0`,
/// `∞ / ∞`, NaN operands, overflow or underflow of the quotient) it is the
/// IEEE quotient of the high words, which is the IEEE quotient of the values,
/// and it is returned as is (with its sign of zero; a NaN as twofloat's
/// canonical `TwoFloat::NAN`).
#[cfg(feature = "f128")]
fn dd_div(a: TwoFloat, b: TwoFloat) -> TwoFloat {
    let (ah, al) = (a.hi(), a.lo());
    let (bh, bl) = (b.hi(), b.lo());
    let q1 = ah / bh;
    if q1.is_nan() {
        return TwoFloat::NAN;
    }
    if q1.is_infinite() || q1 == 0.0 {
        return TwoFloat::from_f64(q1);
    }
    // r = a - q1·b = (ah - q1·bh) + al - q1·bl. The first term is the exact
    // remainder of the correctly rounded q1; two-sum/two-product keep the
    // O(u·|a|) terms exact and leave only O(u²·|a|) terms to round.
    let rem = fma(-q1, bh, ah);
    let (p, pe) = two_prod(q1, bl);
    let (s1, t1) = two_sum(rem, al);
    let (rh, t2) = two_sum(s1, -p);
    let rl = (t1 + t2) - pe;
    // r - q2·b = (rh - q2·bh) + rl - q2·bl; the first term is again an exact
    // FMA remainder and the other two are O(u²·|a|).
    let q2 = rh / bh;
    let r2 = fma(-q2, bh, rh) + (rl - q2 * bl);
    let q3 = r2 / bh;
    let q = TwoFloat::new_add(q1, q2) + q3;
    if q.hi().is_finite() {
        q
    } else {
        // The renormalization carried past f64::MAX: overflow to ±∞ (the
        // error-free sums would turn it into NaN).
        TwoFloat::from_f64(q.hi())
    }
}

/// Twofloat's `atan(1/2)` and `atan(3/2)` reduction constants (correctly
/// rounded double-doubles; private in twofloat).
#[cfg(feature = "f128")]
#[inline]
fn atan_reduction_constants() -> (TwoFloat, TwoFloat) {
    (
        TwoFloat::new_add(
            f64::from_bits(0x3FDD_AC67_0561_BB4F),
            f64::from_bits(0x3C7A_2B7F_222F_65E2),
        ),
        TwoFloat::new_add(
            f64::from_bits(0x3FEF_730B_D281_F69B),
            f64::from_bits(0x3C70_0788_7AF0_CBBD),
        ),
    )
}

#[cfg(feature = "f128")]
impl QuadFloat {
    /// `1 / self` through [`dd_div`] (twofloat's `recip` is `1.0 / TwoFloat`,
    /// which has the inexact residual).
    #[inline]
    fn dd_recip(self) -> Self {
        QuadFloat(dd_div(TwoFloat::from_f64(1.0), self.0))
    }

    /// `self^n`; a negative `n` is the reciprocal of `self^|n|` through
    /// [`dd_div`]. Non-negative powers are twofloat's square-and-multiply,
    /// unchanged.
    ///
    /// `|n|` is formed as `-(n + 1)` plus one more factor, so `i32::MIN` (whose
    /// negation overflows, as it does inside twofloat's `powi`) is fine.
    #[inline]
    fn dd_powi(self, n: i32) -> Self {
        if n >= 0 {
            return QuadFloat(self.0.powi(n));
        }
        let m = -(n + 1);
        let magnitude = if m == 0 {
            self.0
        } else {
            self.0.powi(m) * self.0
        };
        QuadFloat(dd_div(TwoFloat::from_f64(1.0), magnitude))
    }

    /// Tangent as `sin / cos` through [`dd_div`].
    ///
    /// twofloat's `tan` takes `-1 / tan(reduced)` in odd quadrants with its
    /// inexact `f64 / TwoFloat` division (errors around 1e-16). `sin` and
    /// `cos` come from one quadrant reduction and keep their relative
    /// accuracy near the poles, so the quotient is as accurate as they are.
    /// Non-finite arguments pass through as in twofloat.
    #[inline]
    fn dd_tan(self) -> Self {
        if !self.0.is_valid() {
            return self;
        }
        let (s, c) = self.0.sin_cos();
        QuadFloat(dd_div(s, c))
    }

    /// Arctangent with twofloat's argument reduction, but [`dd_div`] for the
    /// reduced arguments (twofloat's own divisions there are only double
    /// precision).
    ///
    /// For `|x| <= 7/16` twofloat's `atan` evaluates its polynomial directly,
    /// so it is used unchanged; every reduced argument below lies in that
    /// range as well (`|t| <= 16/39`). `atan(±∞) = ±π/2` (twofloat returns NaN
    /// there because its validity check precedes its infinity branch).
    fn dd_atan(self) -> Self {
        let x = self.0;
        if x.hi().is_nan() || x.lo().is_nan() {
            return QuadFloat(TwoFloat::NAN);
        }
        if x.hi().is_infinite() {
            return QuadFloat(if x.hi() > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 });
        }
        let ax = x.abs();
        let k = 4.0 * ax + 0.25;
        if k <= 2.0 {
            return QuadFloat(x.atan());
        }
        let (atan_half, atan_three_halves) = atan_reduction_constants();
        let one = TwoFloat::from_f64(1.0);
        let result = if k < 3.0 {
            atan_half + dd_div(ax - 0.5, 1.0 + 0.5 * ax).atan()
        } else if k < 5.0 {
            FRAC_PI_4 + dd_div(ax - 1.0, 1.0 + ax).atan()
        } else if k < 10.0 {
            atan_three_halves + dd_div(ax - 1.5, 1.0 + 1.5 * ax).atan()
        } else {
            FRAC_PI_2 - dd_div(one, ax).atan()
        };
        QuadFloat(if x.is_sign_positive() {
            result
        } else {
            -result
        })
    }

    /// Four-quadrant arctangent of `self / other`: twofloat's case analysis
    /// with [`dd_div`] and [`Self::dd_atan`].
    fn dd_atan2(self, other: Self) -> Self {
        let (y, x) = (self.0, other.0);
        let angle = if y.hi() == 0.0 {
            if x.hi().is_sign_positive() {
                TwoFloat::from_f64(0.0)
            } else if y.hi().is_sign_positive() {
                PI
            } else {
                -PI
            }
        } else if x.hi() == 0.0 {
            if y.hi().is_sign_positive() {
                FRAC_PI_2
            } else {
                -FRAC_PI_2
            }
        } else {
            let a = QuadFloat(dd_div(y, x)).dd_atan().0;
            if x.hi().is_sign_positive() {
                a
            } else if y.hi().is_sign_positive() {
                a + PI
            } else {
                a - PI
            }
        };
        QuadFloat(angle)
    }
}

#[cfg(feature = "f128")]
impl From<f64> for QuadFloat {
    #[inline]
    fn from(value: f64) -> Self {
        Self(TwoFloat::from_f64(value))
    }
}

#[cfg(feature = "f128")]
impl From<TwoFloat> for QuadFloat {
    #[inline]
    fn from(value: TwoFloat) -> Self {
        Self(value)
    }
}

// Implement arithmetic operations by delegating to TwoFloat
#[cfg(feature = "f128")]
impl Add for QuadFloat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[cfg(feature = "f128")]
impl Sub for QuadFloat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

#[cfg(feature = "f128")]
impl Mul for QuadFloat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

/// Double-double division, accurate to about `2^-107` relative.
///
/// It does not delegate to `TwoFloat / TwoFloat`: in twofloat 0.8 that
/// division forms its residual without an FMA and is only double precision
/// (`1/3` came out as `fl(1/3) + 0`). Special values follow IEEE 754
/// (`x / ±0 = ±∞`, `x / ±∞ = ±0`, `0 / 0` and `∞ / ∞` are NaN).
#[cfg(feature = "f128")]
impl Div for QuadFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(dd_div(self.0, rhs.0))
    }
}

#[cfg(feature = "f128")]
impl Neg for QuadFloat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

#[cfg(feature = "f128")]
impl AddAssign for QuadFloat {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

#[cfg(feature = "f128")]
impl SubAssign for QuadFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
    }
}

#[cfg(feature = "f128")]
impl MulAssign for QuadFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

#[cfg(feature = "f128")]
impl DivAssign for QuadFloat {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 = dd_div(self.0, rhs.0);
    }
}

#[cfg(feature = "f128")]
impl Rem for QuadFloat {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        // Truncated remainder, matching Rust's `%` for primitive floats: the
        // result carries the sign of the dividend and `|result| < |rhs|`.
        //   r = self - trunc(self / rhs) * rhs
        // (NOT floored division, which would give the sign of the divisor.) The
        // quotient is truncated at double-double precision, and a single
        // boundary correction repairs the at-most-one-ULP error the division can
        // introduce into the truncated quotient so the two defining properties
        // above hold exactly. For `|self / rhs|` beyond the ~2^106 integer range
        // of double-double the quotient can no longer be represented exactly and
        // the result degrades gracefully, as it does for any single-step fmod.

        // twofloat does not define operations on non-finite values, so match
        // f64 `%` on the special cases explicitly:
        //   x % 0 = NaN, inf % y = NaN, x % inf = x, NaN propagates.
        if self.0.is_nan() || rhs.0.is_nan() || self.0.is_infinite() || rhs == QuadFloat::from(0.0)
        {
            return QuadFloat(TwoFloat::NAN);
        }
        if rhs.0.is_infinite() {
            return self;
        }

        let quotient = self / rhs;
        let mut n = quotient.dd_trunc();
        let mut remainder = self - n * rhs;

        if remainder != QuadFloat::from(0.0) {
            let step = if quotient > QuadFloat::from(0.0) {
                QuadFloat::from(1.0)
            } else {
                QuadFloat::from(-1.0)
            };
            if remainder.0.is_sign_negative() != self.0.is_sign_negative() {
                // Truncated quotient overshot: step it one unit toward zero.
                n -= step;
                remainder = self - n * rhs;
            } else if QuadFloat(remainder.0.abs()) >= QuadFloat(rhs.0.abs()) {
                // Truncated quotient fell short: step it one unit outward.
                n += step;
                remainder = self - n * rhs;
            }
        }
        remainder
    }
}

#[cfg(feature = "f128")]
impl RemAssign for QuadFloat {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

#[cfg(feature = "f128")]
impl Sum for QuadFloat {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(QuadFloat::from(0.0), |acc, x| acc + x)
    }
}

#[cfg(feature = "f128")]
impl<'a> Sum<&'a QuadFloat> for QuadFloat {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().fold(QuadFloat::from(0.0), |acc, x| acc + x)
    }
}

#[cfg(feature = "f128")]
impl Zero for QuadFloat {
    #[inline]
    fn zero() -> Self {
        QuadFloat::from(0.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == TwoFloat::from_f64(0.0)
    }
}

#[cfg(feature = "f128")]
impl One for QuadFloat {
    #[inline]
    fn one() -> Self {
        QuadFloat::from(1.0)
    }
}

// NumAssign is automatically derived from Num + NumAssignOps

#[cfg(feature = "f128")]
impl Display for QuadFloat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Implement Float trait for QuadFloat by delegating to TwoFloat
#[cfg(feature = "f128")]
impl Float for QuadFloat {
    fn nan() -> Self {
        QuadFloat(TwoFloat::NAN)
    }

    fn infinity() -> Self {
        QuadFloat(TwoFloat::INFINITY)
    }

    fn neg_infinity() -> Self {
        QuadFloat(TwoFloat::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        QuadFloat(-TwoFloat::from_f64(0.0))
    }

    fn min_value() -> Self {
        QuadFloat(TwoFloat::MIN)
    }

    fn min_positive_value() -> Self {
        QuadFloat(TwoFloat::MIN_POSITIVE)
    }

    fn max_value() -> Self {
        QuadFloat(TwoFloat::MAX)
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> core::num::FpCategory {
        self.0.classify()
    }

    fn floor(self) -> Self {
        self.dd_floor()
    }

    fn ceil(self) -> Self {
        self.dd_ceil()
    }

    fn round(self) -> Self {
        self.dd_round()
    }

    fn trunc(self) -> Self {
        self.dd_trunc()
    }

    fn fract(self) -> Self {
        self.dd_fract()
    }

    fn abs(self) -> Self {
        QuadFloat(self.0.abs())
    }

    fn signum(self) -> Self {
        self.dd_signum()
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    fn recip(self) -> Self {
        self.dd_recip()
    }

    fn powi(self, n: i32) -> Self {
        self.dd_powi(n)
    }

    fn powf(self, n: Self) -> Self {
        QuadFloat(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        QuadFloat(self.0.sqrt())
    }

    fn exp(self) -> Self {
        QuadFloat(self.0.exp())
    }

    fn exp2(self) -> Self {
        QuadFloat(TwoFloat::from_f64(2.0).powf(self.0))
    }

    fn ln(self) -> Self {
        QuadFloat(self.0.ln())
    }

    // The logarithms divide through `dd_div`; their accuracy is still bounded
    // by twofloat's `ln`. `log2` keeps the computed `ln(2)` as the divisor so
    // that `log2(2)` stays exactly 1.
    fn log(self, base: Self) -> Self {
        QuadFloat(dd_div(self.0.ln(), base.0.ln()))
    }

    fn log2(self) -> Self {
        QuadFloat(dd_div(self.0.ln(), TwoFloat::from_f64(2.0).ln()))
    }

    fn log10(self) -> Self {
        QuadFloat(dd_div(self.0.ln(), LN_10))
    }

    fn max(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self < other { self } else { other }
    }

    fn abs_sub(self, other: Self) -> Self {
        if self > other {
            self - other
        } else {
            QuadFloat::from(0.0)
        }
    }

    fn cbrt(self) -> Self {
        self.dd_cbrt()
    }

    fn hypot(self, other: Self) -> Self {
        self.dd_hypot(other)
    }

    fn sin(self) -> Self {
        QuadFloat(self.0.sin())
    }

    fn cos(self) -> Self {
        QuadFloat(self.0.cos())
    }

    fn tan(self) -> Self {
        self.dd_tan()
    }

    fn asin(self) -> Self {
        QuadFloat(self.0.asin())
    }

    fn acos(self) -> Self {
        QuadFloat(self.0.acos())
    }

    fn atan(self) -> Self {
        self.dd_atan()
    }

    fn atan2(self, other: Self) -> Self {
        self.dd_atan2(other)
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.0.sin_cos();
        (QuadFloat(sin), QuadFloat(cos))
    }

    fn exp_m1(self) -> Self {
        QuadFloat(self.0.exp() - TwoFloat::from_f64(1.0))
    }

    fn ln_1p(self) -> Self {
        QuadFloat((self.0 + TwoFloat::from_f64(1.0)).ln())
    }

    fn sinh(self) -> Self {
        QuadFloat(self.0.sinh())
    }

    fn cosh(self) -> Self {
        QuadFloat(self.0.cosh())
    }

    fn tanh(self) -> Self {
        QuadFloat(self.0.tanh())
    }

    fn asinh(self) -> Self {
        QuadFloat(self.0.asinh())
    }

    fn acosh(self) -> Self {
        QuadFloat(self.0.acosh())
    }

    fn atanh(self) -> Self {
        QuadFloat(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        // For double-double, we decode the high part
        self.0.hi().integer_decode()
    }

    fn epsilon() -> Self {
        QuadFloat::from(f64::EPSILON) * QuadFloat::from(f64::EPSILON)
    }

    fn to_degrees(self) -> Self {
        const FACTOR: f64 = 180.0 / core::f64::consts::PI;
        self * QuadFloat::from(FACTOR)
    }

    fn to_radians(self) -> Self {
        const FACTOR: f64 = core::f64::consts::PI / 180.0;
        self * QuadFloat::from(FACTOR)
    }
}

#[cfg(feature = "f128")]
impl FromPrimitive for QuadFloat {
    fn from_i64(n: i64) -> Option<Self> {
        Some(QuadFloat::from(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(QuadFloat::from(n as f64))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(QuadFloat::from(n))
    }
}

#[cfg(feature = "f128")]
impl num_traits::Num for QuadFloat {
    type FromStrRadixErr = num_traits::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix)
            .map(QuadFloat::from)
            .map_err(|_| num_traits::ParseFloatError {
                kind: num_traits::FloatErrorKind::Invalid,
            })
    }
}

#[cfg(feature = "f128")]
impl num_traits::NumCast for QuadFloat {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(<QuadFloat as From<f64>>::from)
    }
}

#[cfg(feature = "f128")]
impl num_traits::ToPrimitive for QuadFloat {
    fn to_i64(&self) -> Option<i64> {
        self.0.hi().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.hi().to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0.hi())
    }
}

// =============================================================================
// Scalar/Real/Field implementations for QuadFloat
// =============================================================================

#[cfg(feature = "f128")]
impl Scalar for QuadFloat {
    type Real = QuadFloat;

    #[inline]
    fn abs(self) -> Self::Real {
        QuadFloat(self.0.abs())
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn is_real() -> bool {
        true
    }

    #[inline]
    fn real(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag(self) -> Self::Real {
        QuadFloat::from(0.0)
    }

    #[inline]
    fn from_real_imag(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline]
    fn abs_sq(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn epsilon() -> Self::Real {
        // Double-double epsilon is approximately 2^-106
        QuadFloat::from(f64::EPSILON) * QuadFloat::from(f64::EPSILON)
    }

    #[inline]
    fn min_positive() -> Self::Real {
        QuadFloat::from(f64::MIN_POSITIVE)
    }

    #[inline]
    fn max_value() -> Self::Real {
        QuadFloat::from(f64::MAX)
    }
}

#[cfg(feature = "f128")]
impl Real for QuadFloat {
    #[inline]
    fn sqrt(self) -> Self {
        QuadFloat(self.0.sqrt())
    }

    #[inline]
    fn ln(self) -> Self {
        QuadFloat(self.0.ln())
    }

    #[inline]
    fn exp(self) -> Self {
        QuadFloat(self.0.exp())
    }

    #[inline]
    fn sin(self) -> Self {
        QuadFloat(self.0.sin())
    }

    #[inline]
    fn cos(self) -> Self {
        QuadFloat(self.0.cos())
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        self.dd_atan2(other)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        QuadFloat(self.0.powf(n.0))
    }

    #[inline]
    fn signum(self) -> Self {
        self.dd_signum()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // TwoFloat doesn't have mul_add, so implement manually
        self * a + b
    }

    #[inline]
    fn floor(self) -> Self {
        self.dd_floor()
    }

    #[inline]
    fn ceil(self) -> Self {
        self.dd_ceil()
    }

    #[inline]
    fn round(self) -> Self {
        self.dd_round()
    }

    #[inline]
    fn trunc(self) -> Self {
        self.dd_trunc()
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        self.dd_hypot(other)
    }
}

#[cfg(feature = "f128")]
impl Field for QuadFloat {
    #[inline]
    fn mul_conj(self, other: Self) -> Self {
        self * other
    }

    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self * other
    }

    #[inline]
    fn recip(self) -> Self {
        self.dd_recip()
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        self.dd_powi(n)
    }
}

#[cfg(all(test, feature = "f128"))]
#[path = "extended_tests.rs"]
mod tests;
