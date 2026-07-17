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
        f16::from_f32(self.to_f32().sqrt())
    }

    #[inline]
    fn ln(self) -> Self {
        f16::from_f32(self.to_f32().ln())
    }

    #[inline]
    fn exp(self) -> Self {
        f16::from_f32(self.to_f32().exp())
    }

    #[inline]
    fn sin(self) -> Self {
        f16::from_f32(self.to_f32().sin())
    }

    #[inline]
    fn cos(self) -> Self {
        f16::from_f32(self.to_f32().cos())
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        f16::from_f32(self.to_f32().atan2(other.to_f32()))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        f16::from_f32(self.to_f32().powf(n.to_f32()))
    }

    #[inline]
    fn signum(self) -> Self {
        if self > f16::ZERO {
            f16::ONE
        } else if self < f16::ZERO {
            -f16::ONE
        } else {
            f16::ZERO
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f16::from_f32(self.to_f32().mul_add(a.to_f32(), b.to_f32()))
    }

    #[inline]
    fn floor(self) -> Self {
        f16::from_f32(self.to_f32().floor())
    }

    #[inline]
    fn ceil(self) -> Self {
        f16::from_f32(self.to_f32().ceil())
    }

    #[inline]
    fn round(self) -> Self {
        f16::from_f32(self.to_f32().round())
    }

    #[inline]
    fn trunc(self) -> Self {
        f16::from_f32(self.to_f32().trunc())
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        f16::from_f32(self.to_f32().hypot(other.to_f32()))
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
        f16::from_f32(self.to_f32().powi(n))
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
/// # Example
///
/// ```ignore
/// use oxiblas_core::scalar::QuadFloat;
///
/// let x = QuadFloat::from(2.0);
/// let y = x.sqrt();
/// assert!((y * y - x).abs() < QuadFloat::from(1e-30));
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
        let floored_hi = hi.floor();
        if floored_hi != hi {
            QuadFloat(TwoFloat::from_f64(floored_hi))
        } else {
            QuadFloat(TwoFloat::new_add(hi, self.0.lo().floor()))
        }
    }

    /// Smallest integer `>= self`, at double-double precision (`f64::ceil`).
    ///
    /// Mirror image of [`dd_floor`](Self::dd_floor).
    #[inline]
    fn dd_ceil(self) -> Self {
        let hi = self.0.hi();
        let ceiled_hi = hi.ceil();
        if ceiled_hi != hi {
            QuadFloat(TwoFloat::from_f64(ceiled_hi))
        } else {
            QuadFloat(TwoFloat::new_add(hi, self.0.lo().ceil()))
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
        let rounded_hi = hi.round();
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
            let mut rounded_lo = lo.round();
            // On a tie in `lo` whose away-from-zero direction disagrees with the
            // whole value's away-from-zero direction (opposite signs), pick the
            // neighbor lying on the whole value's side, i.e. `trunc(lo)`.
            if (lo - lo.trunc()).abs() == 0.5 && (lo < 0.0) != (hi < 0.0) {
                rounded_lo = lo.trunc();
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
    /// one. (twofloat's `TwoFloat / TwoFloat` division is not full precision, so
    /// the scaled path is slightly less accurate — hence it is used only when
    /// unavoidable.)
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

#[cfg(feature = "f128")]
impl Div for QuadFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
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
        self.0 = self.0 / rhs.0;
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
        let zero = QuadFloat::from(0.0);
        let one = QuadFloat::from(1.0);
        if self > zero {
            one
        } else if self < zero {
            -one
        } else {
            zero
        }
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
        QuadFloat(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        QuadFloat(self.0.powi(n))
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

    fn log(self, base: Self) -> Self {
        QuadFloat(self.0.ln() / base.0.ln())
    }

    fn log2(self) -> Self {
        QuadFloat(self.0.ln() / TwoFloat::from_f64(2.0).ln())
    }

    fn log10(self) -> Self {
        QuadFloat(self.0.log10())
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
        QuadFloat(self.0.tan())
    }

    fn asin(self) -> Self {
        QuadFloat(self.0.asin())
    }

    fn acos(self) -> Self {
        QuadFloat(self.0.acos())
    }

    fn atan(self) -> Self {
        QuadFloat(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        QuadFloat(self.0.atan2(other.0))
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
        QuadFloat(self.0.atan2(other.0))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        QuadFloat(self.0.powf(n.0))
    }

    #[inline]
    fn signum(self) -> Self {
        let zero = QuadFloat::from(0.0);
        let one = QuadFloat::from(1.0);
        if self > zero {
            one
        } else if self < zero {
            -one
        } else {
            zero
        }
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
        QuadFloat(self.0.recip())
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        QuadFloat(self.0.powi(n))
    }
}
