//! Core scalar trait definitions: Scalar, Real, ComplexScalar, and Field.

use core::fmt::{Debug, Display};
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{FromPrimitive, NumAssign, One, Zero};

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
pub trait Real: Scalar<Real = Self> + num_traits::Float + PartialOrd {
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
