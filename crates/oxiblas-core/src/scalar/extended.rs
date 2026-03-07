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
        // Implement remainder using floor division
        let quotient = QuadFloat::from((self.0 / rhs.0).hi().floor());
        self - quotient * rhs
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
        QuadFloat::from(self.0.hi().floor())
    }

    fn ceil(self) -> Self {
        QuadFloat::from(self.0.hi().ceil())
    }

    fn round(self) -> Self {
        QuadFloat::from(self.0.hi().round())
    }

    fn trunc(self) -> Self {
        QuadFloat::from(self.0.hi().trunc())
    }

    fn fract(self) -> Self {
        QuadFloat::from(self.0.hi().fract())
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
        QuadFloat(self.0.powf(TwoFloat::from_f64(1.0 / 3.0)))
    }

    fn hypot(self, other: Self) -> Self {
        Float::sqrt(self * self + other * other)
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
        QuadFloat::from(self.0.hi().floor())
    }

    #[inline]
    fn ceil(self) -> Self {
        QuadFloat::from(self.0.hi().ceil())
    }

    #[inline]
    fn round(self) -> Self {
        QuadFloat::from(self.0.hi().round())
    }

    #[inline]
    fn trunc(self) -> Self {
        QuadFloat::from(self.0.hi().trunc())
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Float::sqrt(self * self + other * other)
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
