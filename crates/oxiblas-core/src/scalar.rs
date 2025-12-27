//! Scalar traits for numeric types used in OxiBLAS.
//!
//! This module defines the trait hierarchy for numeric types:
//! - `Scalar`: Base trait for all scalar types
//! - `Real`: Real number types (f32, f64, f16 with feature, QuadFloat with f128 feature)
//! - `ComplexScalar`: Complex number types
//! - `Field`: Field operations (complete algebraic structure)

use core::fmt::{Debug, Display};
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, NumAssign, One, Zero};

// =============================================================================
// Type aliases for convenience
// =============================================================================

/// 32-bit complex type alias (same as `Complex32`).
pub type C32 = Complex32;

/// 64-bit complex type alias (same as `Complex64`).
pub type C64 = Complex64;

// =============================================================================
// Complex number constructors and utilities
// =============================================================================

/// Creates a complex number from real and imaginary parts.
///
/// # Examples
///
/// ```
/// use oxiblas_core::scalar::c64;
/// let z = c64(3.0, 4.0);
/// assert_eq!(z.re, 3.0);
/// assert_eq!(z.im, 4.0);
/// ```
#[inline]
pub const fn c64(re: f64, im: f64) -> C64 {
    Complex64::new(re, im)
}

/// Creates a complex number from real and imaginary parts (f32).
///
/// # Examples
///
/// ```
/// use oxiblas_core::scalar::c32;
/// let z = c32(3.0, 4.0);
/// assert_eq!(z.re, 3.0);
/// assert_eq!(z.im, 4.0);
/// ```
#[inline]
pub const fn c32(re: f32, im: f32) -> C32 {
    Complex32::new(re, im)
}

/// The imaginary unit i (64-bit).
pub const I64: C64 = Complex64::new(0.0, 1.0);

/// The imaginary unit i (32-bit).
pub const I32: C32 = Complex32::new(0.0, 1.0);

/// Returns the imaginary unit as a 64-bit complex number.
#[inline]
pub const fn imag_unit() -> C64 {
    I64
}

/// Returns the imaginary unit as a 32-bit complex number.
#[inline]
pub const fn imag_unit32() -> C32 {
    I32
}

/// Creates a purely imaginary number (64-bit).
///
/// # Examples
///
/// ```
/// use oxiblas_core::scalar::imag;
/// let z = imag(2.0);
/// assert_eq!(z.re, 0.0);
/// assert_eq!(z.im, 2.0);
/// ```
#[inline]
pub const fn imag(im: f64) -> C64 {
    Complex64::new(0.0, im)
}

/// Creates a purely imaginary number (32-bit).
#[inline]
pub const fn imag32(im: f32) -> C32 {
    Complex32::new(0.0, im)
}

/// Creates a real number as complex (64-bit).
///
/// # Examples
///
/// ```
/// use oxiblas_core::scalar::real;
/// let z = real(3.0);
/// assert_eq!(z.re, 3.0);
/// assert_eq!(z.im, 0.0);
/// ```
#[inline]
pub const fn real(re: f64) -> C64 {
    Complex64::new(re, 0.0)
}

/// Creates a real number as complex (32-bit).
#[inline]
pub const fn real32(re: f32) -> C32 {
    Complex32::new(re, 0.0)
}

/// Creates a complex number from polar coordinates (64-bit).
///
/// # Arguments
/// * `r` - The magnitude (radius)
/// * `theta` - The angle in radians
///
/// # Examples
///
/// ```
/// use oxiblas_core::scalar::from_polar;
/// use std::f64::consts::PI;
/// let z = from_polar(1.0, PI / 2.0);
/// assert!((z.re - 0.0).abs() < 1e-10);
/// assert!((z.im - 1.0).abs() < 1e-10);
/// ```
#[inline]
pub fn from_polar(r: f64, theta: f64) -> C64 {
    Complex64::from_polar(r, theta)
}

/// Creates a complex number from polar coordinates (32-bit).
#[inline]
pub fn from_polar32(r: f32, theta: f32) -> C32 {
    Complex32::from_polar(r, theta)
}

/// Extension trait for more ergonomic complex number operations.
pub trait ComplexExt: Sized {
    /// The real component type.
    type Real;

    /// Returns true if this complex number is purely real (imaginary part ≈ 0).
    #[allow(clippy::wrong_self_convention)] // Complex numbers are Copy, self by value is efficient
    fn is_purely_real(self, tolerance: Self::Real) -> bool;

    /// Returns true if this complex number is purely imaginary (real part ≈ 0).
    #[allow(clippy::wrong_self_convention)] // Complex numbers are Copy, self by value is efficient
    fn is_purely_imaginary(self, tolerance: Self::Real) -> bool;

    /// Rotates the complex number by the given angle (in radians).
    fn rotate(self, angle: Self::Real) -> Self;

    /// Scales the magnitude while keeping the phase.
    fn scale_magnitude(self, factor: Self::Real) -> Self;

    /// Returns the complex number normalized to unit magnitude.
    fn normalize(self) -> Self;

    /// Reflects across the real axis (same as conjugate).
    fn reflect_real(self) -> Self;

    /// Reflects across the imaginary axis.
    fn reflect_imag(self) -> Self;

    /// Returns the distance to another complex number.
    fn distance(self, other: Self) -> Self::Real;
}

impl ComplexExt for C64 {
    type Real = f64;

    #[inline]
    fn is_purely_real(self, tolerance: f64) -> bool {
        self.im.abs() <= tolerance
    }

    #[inline]
    fn is_purely_imaginary(self, tolerance: f64) -> bool {
        self.re.abs() <= tolerance
    }

    #[inline]
    fn rotate(self, angle: f64) -> Self {
        self * Complex64::from_polar(1.0, angle)
    }

    #[inline]
    fn scale_magnitude(self, factor: f64) -> Self {
        let (r, theta) = self.to_polar();
        Complex64::from_polar(r * factor, theta)
    }

    #[inline]
    fn normalize(self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            Complex64::new(0.0, 0.0)
        } else {
            self / norm
        }
    }

    #[inline]
    fn reflect_real(self) -> Self {
        self.conj()
    }

    #[inline]
    fn reflect_imag(self) -> Self {
        Complex64::new(-self.re, self.im)
    }

    #[inline]
    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

impl ComplexExt for C32 {
    type Real = f32;

    #[inline]
    fn is_purely_real(self, tolerance: f32) -> bool {
        self.im.abs() <= tolerance
    }

    #[inline]
    fn is_purely_imaginary(self, tolerance: f32) -> bool {
        self.re.abs() <= tolerance
    }

    #[inline]
    fn rotate(self, angle: f32) -> Self {
        self * Complex32::from_polar(1.0, angle)
    }

    #[inline]
    fn scale_magnitude(self, factor: f32) -> Self {
        let (r, theta) = self.to_polar();
        Complex32::from_polar(r * factor, theta)
    }

    #[inline]
    fn normalize(self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            Complex32::new(0.0, 0.0)
        } else {
            self / norm
        }
    }

    #[inline]
    fn reflect_real(self) -> Self {
        self.conj()
    }

    #[inline]
    fn reflect_imag(self) -> Self {
        Complex32::new(-self.re, self.im)
    }

    #[inline]
    fn distance(self, other: Self) -> f32 {
        (self - other).norm()
    }
}

/// Trait for converting real numbers to complex.
pub trait ToComplex<C> {
    /// Converts to complex with zero imaginary part.
    fn to_complex(self) -> C;

    /// Converts to complex with given imaginary part.
    fn with_imag(self, im: Self) -> C;
}

impl ToComplex<C64> for f64 {
    #[inline]
    fn to_complex(self) -> C64 {
        Complex64::new(self, 0.0)
    }

    #[inline]
    fn with_imag(self, im: f64) -> C64 {
        Complex64::new(self, im)
    }
}

impl ToComplex<C32> for f32 {
    #[inline]
    fn to_complex(self) -> C32 {
        Complex32::new(self, 0.0)
    }

    #[inline]
    fn with_imag(self, im: f32) -> C32 {
        Complex32::new(self, im)
    }
}

#[cfg(feature = "f128")]
use core::ops::{Rem, RemAssign};

#[cfg(feature = "f16")]
use half::f16;

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

/// Base trait for all scalar types used in OxiBLAS.
///
/// This trait provides the fundamental requirements for any numeric type
/// that can be used in matrix operations.
pub trait Scalar:
    Copy
    + Clone
    + Debug
    + Display
    + Default
    + Send
    + Sync
    + PartialEq
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Neg<Output = Self>
    + Sum
    + NumAssign
    + FromPrimitive
    + 'static
{
    /// The real component type (for complex numbers, this is the component type).
    type Real: Real;

    /// Returns the absolute value (modulus for complex numbers).
    fn abs(self) -> Self::Real;

    /// Returns the complex conjugate. For real numbers, returns self.
    fn conj(self) -> Self;

    /// Returns true if this is a real type (not complex).
    fn is_real() -> bool;

    /// Returns the real part.
    fn real(self) -> Self::Real;

    /// Returns the imaginary part (zero for real types).
    fn imag(self) -> Self::Real;

    /// Creates a scalar from real and imaginary parts.
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self;

    /// Creates a scalar from just the real part (imaginary = 0).
    fn from_real(re: Self::Real) -> Self {
        Self::from_real_imag(re, Self::Real::zero())
    }

    /// Square of the absolute value (more efficient than abs().powi(2)).
    fn abs_sq(self) -> Self::Real {
        let re = self.real();
        let im = self.imag();
        re * re + im * im
    }

    /// Machine epsilon for this type.
    fn epsilon() -> Self::Real;

    /// Smallest positive normal value.
    fn min_positive() -> Self::Real;

    /// Largest finite value.
    fn max_value() -> Self::Real;

    /// Size of the type in bytes.
    const SIZE: usize = core::mem::size_of::<Self>();

    /// Alignment requirement.
    const ALIGN: usize = core::mem::align_of::<Self>();
}

/// Trait for real number types (f32, f64).
pub trait Real: Scalar<Real = Self> + Float + PartialOrd {
    /// Square root.
    fn sqrt(self) -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// Exponential function.
    fn exp(self) -> Self;

    /// Sine.
    fn sin(self) -> Self;

    /// Cosine.
    fn cos(self) -> Self;

    /// Arctangent of y/x with correct quadrant.
    fn atan2(self, other: Self) -> Self;

    /// Power function.
    fn powf(self, n: Self) -> Self;

    /// Sign function: 1.0 if positive, -1.0 if negative, 0.0 if zero.
    fn signum(self) -> Self;

    /// Fused multiply-add: self * a + b
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Floor function.
    fn floor(self) -> Self;

    /// Ceiling function.
    fn ceil(self) -> Self;

    /// Round to nearest integer.
    fn round(self) -> Self;

    /// Truncate toward zero.
    fn trunc(self) -> Self;

    /// Safe reciprocal (returns None if self is zero or would overflow).
    fn safe_recip(self) -> Option<Self> {
        if Scalar::abs(self) < Self::min_positive() {
            None
        } else {
            Some(Self::one() / self)
        }
    }

    /// Hypot: sqrt(self^2 + other^2) computed without overflow.
    fn hypot(self, other: Self) -> Self;
}

/// Trait for complex scalar types.
pub trait ComplexScalar: Scalar {
    /// Creates a complex number from real and imaginary parts.
    fn new(re: Self::Real, im: Self::Real) -> Self;

    /// Returns the argument (phase angle) of the complex number.
    fn arg(self) -> Self::Real;

    /// Returns the polar form (r, theta) where self = r * e^(i*theta).
    fn to_polar(self) -> (Self::Real, Self::Real) {
        (self.abs(), self.arg())
    }

    /// Creates a complex number from polar form.
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;

    /// Complex exponential.
    fn cexp(self) -> Self;

    /// Complex logarithm (principal branch).
    fn cln(self) -> Self;

    /// Complex square root (principal branch).
    fn csqrt(self) -> Self;
}

/// Field trait - complete algebraic structure with all operations.
///
/// This is the main trait used throughout OxiBLAS for generic programming
/// over numeric types.
pub trait Field: Scalar {
    /// Computes self * alpha + other * beta
    #[inline]
    fn scale_add(self, alpha: Self, other: Self, beta: Self) -> Self {
        self * alpha + other * beta
    }

    /// Computes self * conj(other) for complex, self * other for real.
    fn mul_conj(self, other: Self) -> Self;

    /// Computes conj(self) * other for complex, self * other for real.
    fn conj_mul(self, other: Self) -> Self;

    /// Reciprocal (1/self).
    fn recip(self) -> Self;

    /// Integer power.
    fn powi(self, n: i32) -> Self;
}

// =============================================================================
// Implementations for f32
// =============================================================================

impl Scalar for f32 {
    type Real = f32;

    #[inline]
    fn abs(self) -> Self::Real {
        <f32 as Float>::abs(self)
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
        0.0
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
        f32::EPSILON
    }

    #[inline]
    fn min_positive() -> Self::Real {
        f32::MIN_POSITIVE
    }

    #[inline]
    fn max_value() -> Self::Real {
        f32::MAX
    }
}

impl Real for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        <f32 as Float>::sqrt(self)
    }

    #[inline]
    fn ln(self) -> Self {
        <f32 as Float>::ln(self)
    }

    #[inline]
    fn exp(self) -> Self {
        <f32 as Float>::exp(self)
    }

    #[inline]
    fn sin(self) -> Self {
        <f32 as Float>::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        <f32 as Float>::cos(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        <f32 as Float>::atan2(self, other)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        <f32 as Float>::powf(self, n)
    }

    #[inline]
    fn signum(self) -> Self {
        <f32 as Float>::signum(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        <f32 as Float>::mul_add(self, a, b)
    }

    #[inline]
    fn floor(self) -> Self {
        <f32 as Float>::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        <f32 as Float>::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        <f32 as Float>::round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        <f32 as Float>::trunc(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        <f32 as Float>::hypot(self, other)
    }
}

impl Field for f32 {
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
        1.0 / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        <f32 as Float>::powi(self, n)
    }
}

// =============================================================================
// Implementations for f64
// =============================================================================

impl Scalar for f64 {
    type Real = f64;

    #[inline]
    fn abs(self) -> Self::Real {
        <f64 as Float>::abs(self)
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
        0.0
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
        f64::EPSILON
    }

    #[inline]
    fn min_positive() -> Self::Real {
        f64::MIN_POSITIVE
    }

    #[inline]
    fn max_value() -> Self::Real {
        f64::MAX
    }
}

impl Real for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        <f64 as Float>::sqrt(self)
    }

    #[inline]
    fn ln(self) -> Self {
        <f64 as Float>::ln(self)
    }

    #[inline]
    fn exp(self) -> Self {
        <f64 as Float>::exp(self)
    }

    #[inline]
    fn sin(self) -> Self {
        <f64 as Float>::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        <f64 as Float>::cos(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        <f64 as Float>::atan2(self, other)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        <f64 as Float>::powf(self, n)
    }

    #[inline]
    fn signum(self) -> Self {
        <f64 as Float>::signum(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        <f64 as Float>::mul_add(self, a, b)
    }

    #[inline]
    fn floor(self) -> Self {
        <f64 as Float>::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        <f64 as Float>::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        <f64 as Float>::round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        <f64 as Float>::trunc(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        <f64 as Float>::hypot(self, other)
    }
}

impl Field for f64 {
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
        1.0 / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        <f64 as Float>::powi(self, n)
    }
}

// =============================================================================
// Implementations for Complex32
// =============================================================================

impl Scalar for Complex32 {
    type Real = f32;

    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }

    #[inline]
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }

    #[inline]
    fn is_real() -> bool {
        false
    }

    #[inline]
    fn real(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        Complex32::new(re, im)
    }

    #[inline]
    fn abs_sq(self) -> Self::Real {
        self.norm_sqr()
    }

    #[inline]
    fn epsilon() -> Self::Real {
        f32::EPSILON
    }

    #[inline]
    fn min_positive() -> Self::Real {
        f32::MIN_POSITIVE
    }

    #[inline]
    fn max_value() -> Self::Real {
        f32::MAX
    }
}

impl ComplexScalar for Complex32 {
    #[inline]
    fn new(re: Self::Real, im: Self::Real) -> Self {
        Complex32::new(re, im)
    }

    #[inline]
    fn arg(self) -> Self::Real {
        self.arg()
    }

    #[inline]
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self {
        Complex32::from_polar(r, theta)
    }

    #[inline]
    fn cexp(self) -> Self {
        self.exp()
    }

    #[inline]
    fn cln(self) -> Self {
        self.ln()
    }

    #[inline]
    fn csqrt(self) -> Self {
        self.sqrt()
    }
}

impl Field for Complex32 {
    #[inline]
    fn mul_conj(self, other: Self) -> Self {
        self * other.conj()
    }

    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self.conj() * other
    }

    #[inline]
    fn recip(self) -> Self {
        Complex32::new(1.0, 0.0) / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        self.powu(n.unsigned_abs())
            * if n < 0 {
                self.recip().powu(n.unsigned_abs())
            } else {
                Complex32::new(1.0, 0.0)
            }
    }
}

// =============================================================================
// Implementations for Complex64
// =============================================================================

impl Scalar for Complex64 {
    type Real = f64;

    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }

    #[inline]
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }

    #[inline]
    fn is_real() -> bool {
        false
    }

    #[inline]
    fn real(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        Complex64::new(re, im)
    }

    #[inline]
    fn abs_sq(self) -> Self::Real {
        self.norm_sqr()
    }

    #[inline]
    fn epsilon() -> Self::Real {
        f64::EPSILON
    }

    #[inline]
    fn min_positive() -> Self::Real {
        f64::MIN_POSITIVE
    }

    #[inline]
    fn max_value() -> Self::Real {
        f64::MAX
    }
}

impl ComplexScalar for Complex64 {
    #[inline]
    fn new(re: Self::Real, im: Self::Real) -> Self {
        Complex64::new(re, im)
    }

    #[inline]
    fn arg(self) -> Self::Real {
        self.arg()
    }

    #[inline]
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self {
        Complex64::from_polar(r, theta)
    }

    #[inline]
    fn cexp(self) -> Self {
        self.exp()
    }

    #[inline]
    fn cln(self) -> Self {
        self.ln()
    }

    #[inline]
    fn csqrt(self) -> Self {
        self.sqrt()
    }
}

impl Field for Complex64 {
    #[inline]
    fn mul_conj(self, other: Self) -> Self {
        self * other.conj()
    }

    #[inline]
    fn conj_mul(self, other: Self) -> Self {
        self.conj() * other
    }

    #[inline]
    fn recip(self) -> Self {
        Complex64::new(1.0, 0.0) / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        if n >= 0 {
            self.powu(n as u32)
        } else {
            self.recip().powu((-n) as u32)
        }
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(f32::CLASS, ScalarClass::RealF32);
        assert_eq!(f64::CLASS, ScalarClass::RealF64);
        assert_eq!(Complex32::CLASS, ScalarClass::ComplexF32);
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
        assert_eq!(Complex64::UNROLL_FACTOR, 2);
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
        let z_acc: Complex64 = z.to_accumulator();
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
}

// =============================================================================
// Half-precision (f16) support
// =============================================================================

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
// Quad-precision (double-double) support using QuadFloat wrapper
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

// =============================================================================
// Scalar trait specialization for performance
// =============================================================================

/// Marker trait for types with hardware FMA (fused multiply-add) support.
///
/// Types implementing this trait have efficient hardware FMA instructions,
/// enabling optimized implementations of algorithms like dot products and
/// matrix multiplications.
pub trait HasFastFma: Scalar {}

impl HasFastFma for f32 {}
impl HasFastFma for f64 {}
impl HasFastFma for Complex32 {}
impl HasFastFma for Complex64 {}

/// Marker trait for types that can be efficiently vectorized with SIMD.
///
/// This trait indicates that the type has a natural mapping to SIMD registers
/// and operations.
pub trait SimdCompatible: Scalar {
    /// The preferred SIMD width (number of elements) for this type.
    const SIMD_WIDTH: usize;

    /// Returns true if SIMD operations are beneficial for the given length.
    #[inline]
    fn use_simd_for(len: usize) -> bool {
        len >= Self::SIMD_WIDTH * 2
    }
}

impl SimdCompatible for f32 {
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 8; // AVX2: 256-bit / 32-bit = 8

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 4; // NEON: 128-bit / 32-bit = 4

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 4;
}

impl SimdCompatible for f64 {
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 4; // AVX2: 256-bit / 64-bit = 4

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 2; // NEON: 128-bit / 64-bit = 2

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 2;
}

impl SimdCompatible for Complex32 {
    // Complex types have half the SIMD width due to doubled storage
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 4;

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 2;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 2;
}

impl SimdCompatible for Complex64 {
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 2;

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 1;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 1;
}

/// Batch operations on scalar arrays for performance-critical code.
///
/// This trait provides optimized implementations of common operations on
/// contiguous arrays of scalars, leveraging SIMD where available.
pub trait ScalarBatch: Scalar + SimdCompatible {
    /// Computes the dot product of two slices.
    ///
    /// # Safety
    /// Both slices must have the same length.
    fn dot_batch(x: &[Self], y: &[Self]) -> Self;

    /// Computes the sum of all elements.
    fn sum_batch(x: &[Self]) -> Self;

    /// Computes the sum of absolute values (L1 norm).
    fn asum_batch(x: &[Self]) -> Self::Real;

    /// Finds the index of the element with maximum absolute value.
    fn iamax_batch(x: &[Self]) -> usize;

    /// Scales a vector: x = alpha * x
    fn scale_batch(alpha: Self, x: &mut [Self]);

    /// AXPY operation: y = alpha * x + y
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]);

    /// Fused multiply-add on arrays: `z[i] = a[i] * b[i] + c[i]`
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]);
}

impl ScalarBatch for f32 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = 0.0f32;
        for i in 0..x.len() {
            sum = x[i].mul_add(y[i], sum);
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|&v| v.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] = alpha.mul_add(x[i], y[i]);
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

impl ScalarBatch for f64 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = 0.0f64;
        for i in 0..x.len() {
            sum = x[i].mul_add(y[i], sum);
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|&v| v.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] = alpha.mul_add(x[i], y[i]);
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

impl ScalarBatch for Complex32 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = Complex32::new(0.0, 0.0);
        for i in 0..x.len() {
            sum += x[i] * y[i];
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|z| z.re.abs() + z.im.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                (a.re.abs() + a.im.abs())
                    .partial_cmp(&(b.re.abs() + b.im.abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] += alpha * x[i];
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

impl ScalarBatch for Complex64 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..x.len() {
            sum += x[i] * y[i];
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|z| z.re.abs() + z.im.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                (a.re.abs() + a.im.abs())
                    .partial_cmp(&(b.re.abs() + b.im.abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] += alpha * x[i];
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

/// Type-level scalar classification for compile-time dispatch.
///
/// This enum enables algorithms to specialize at compile time based on
/// the scalar type's properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarClass {
    /// Single-precision real (f32)
    RealF32,
    /// Double-precision real (f64)
    RealF64,
    /// Single-precision complex
    ComplexF32,
    /// Double-precision complex
    ComplexF64,
    /// Half-precision real (f16)
    RealF16,
    /// Quad-precision real (f128)
    RealF128,
    /// Unknown/other type
    Other,
}

/// Trait for compile-time scalar classification.
pub trait ScalarClassify: Scalar {
    /// The compile-time class of this scalar type.
    const CLASS: ScalarClass;

    /// Returns the precision level (1 = lowest, 4 = highest).
    const PRECISION_LEVEL: u8;

    /// Returns the storage size in bytes.
    const STORAGE_BYTES: usize = core::mem::size_of::<Self>();
}

impl ScalarClassify for f32 {
    const CLASS: ScalarClass = ScalarClass::RealF32;
    const PRECISION_LEVEL: u8 = 2;
}

impl ScalarClassify for f64 {
    const CLASS: ScalarClass = ScalarClass::RealF64;
    const PRECISION_LEVEL: u8 = 3;
}

impl ScalarClassify for Complex32 {
    const CLASS: ScalarClass = ScalarClass::ComplexF32;
    const PRECISION_LEVEL: u8 = 2;
}

impl ScalarClassify for Complex64 {
    const CLASS: ScalarClass = ScalarClass::ComplexF64;
    const PRECISION_LEVEL: u8 = 3;
}

#[cfg(feature = "f16")]
impl ScalarClassify for f16 {
    const CLASS: ScalarClass = ScalarClass::RealF16;
    const PRECISION_LEVEL: u8 = 1;
}

#[cfg(feature = "f128")]
impl ScalarClassify for QuadFloat {
    const CLASS: ScalarClass = ScalarClass::RealF128;
    const PRECISION_LEVEL: u8 = 4;
}

/// Unrolling hints for vectorized loops.
///
/// These constants help the compiler make better unrolling decisions
/// for different scalar types.
pub trait UnrollHints: Scalar {
    /// Recommended unroll factor for tight loops.
    const UNROLL_FACTOR: usize;

    /// Recommended chunk size for blocked algorithms.
    const BLOCK_SIZE: usize;

    /// Whether to prefer streaming stores (for large writes).
    const PREFER_STREAMING: bool;
}

impl UnrollHints for f32 {
    const UNROLL_FACTOR: usize = 8;
    const BLOCK_SIZE: usize = 64;
    const PREFER_STREAMING: bool = true;
}

impl UnrollHints for f64 {
    const UNROLL_FACTOR: usize = 4;
    const BLOCK_SIZE: usize = 32;
    const PREFER_STREAMING: bool = true;
}

impl UnrollHints for Complex32 {
    const UNROLL_FACTOR: usize = 4;
    const BLOCK_SIZE: usize = 32;
    const PREFER_STREAMING: bool = true;
}

impl UnrollHints for Complex64 {
    const UNROLL_FACTOR: usize = 2;
    const BLOCK_SIZE: usize = 16;
    const PREFER_STREAMING: bool = true;
}

/// Extended precision accumulation support.
///
/// For algorithms requiring higher precision during intermediate calculations,
/// this trait provides access to an extended precision accumulator type.
pub trait ExtendedPrecision: Scalar {
    /// The type used for extended precision accumulation.
    type Accumulator: Scalar;

    /// Converts a value to the accumulator type.
    fn to_accumulator(self) -> Self::Accumulator;

    /// Converts from the accumulator type back to this type.
    fn from_accumulator(acc: Self::Accumulator) -> Self;
}

impl ExtendedPrecision for f32 {
    type Accumulator = f64;

    #[inline]
    fn to_accumulator(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_accumulator(acc: f64) -> f32 {
        acc as f32
    }
}

impl ExtendedPrecision for f64 {
    // For f64, we use the same type (or could use f128 if available)
    type Accumulator = f64;

    #[inline]
    fn to_accumulator(self) -> f64 {
        self
    }

    #[inline]
    fn from_accumulator(acc: f64) -> f64 {
        acc
    }
}

impl ExtendedPrecision for Complex32 {
    type Accumulator = Complex64;

    #[inline]
    fn to_accumulator(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }

    #[inline]
    fn from_accumulator(acc: Complex64) -> Complex32 {
        Complex32::new(acc.re as f32, acc.im as f32)
    }
}

impl ExtendedPrecision for Complex64 {
    type Accumulator = Complex64;

    #[inline]
    fn to_accumulator(self) -> Complex64 {
        self
    }

    #[inline]
    fn from_accumulator(acc: Complex64) -> Complex64 {
        acc
    }
}

/// Kahan summation for improved accuracy.
///
/// Uses compensated summation to reduce floating-point errors.
#[derive(Debug, Clone, Copy)]
pub struct KahanSum<T: Scalar> {
    sum: T,
    compensation: T,
}

impl<T: Scalar> Default for KahanSum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> KahanSum<T> {
    /// Creates a new Kahan sum accumulator initialized to zero.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            compensation: T::zero(),
        }
    }

    /// Adds a value to the sum with compensation.
    #[inline]
    pub fn add(&mut self, value: T) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Returns the current sum.
    #[inline]
    pub fn sum(self) -> T {
        self.sum
    }
}

/// Pairwise summation for reduced error accumulation.
///
/// Recursively splits the array and sums pairs, reducing error from O(n) to O(log n).
#[inline]
pub fn pairwise_sum<T: Scalar>(values: &[T]) -> T {
    const THRESHOLD: usize = 32;

    if values.is_empty() {
        return T::zero();
    }
    if values.len() <= THRESHOLD {
        return values.iter().copied().fold(T::zero(), |acc, x| acc + x);
    }

    let mid = values.len() / 2;
    pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
}

/// Kahan-Babuska-Klein summation (improved compensated summation).
///
/// Provides even better error bounds than standard Kahan summation.
#[derive(Debug, Clone, Copy)]
pub struct KBKSum<T: Scalar> {
    sum: T,
    cs: T,
    ccs: T,
}

impl<T: Scalar> Default for KBKSum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> KBKSum<T> {
    /// Creates a new KBK sum accumulator.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            cs: T::zero(),
            ccs: T::zero(),
        }
    }

    /// Adds a value with double compensation.
    #[inline]
    pub fn add(&mut self, value: T) {
        let t = self.sum + value;
        let c = if Scalar::abs(self.sum) >= Scalar::abs(value) {
            (self.sum - t) + value
        } else {
            (value - t) + self.sum
        };
        self.sum = t;

        let t2 = self.cs + c;
        let cc = if Scalar::abs(self.cs) >= Scalar::abs(c) {
            (self.cs - t2) + c
        } else {
            (c - t2) + self.cs
        };
        self.cs = t2;
        self.ccs += cc;
    }

    /// Returns the compensated sum.
    #[inline]
    pub fn sum(self) -> T {
        self.sum + self.cs + self.ccs
    }
}
