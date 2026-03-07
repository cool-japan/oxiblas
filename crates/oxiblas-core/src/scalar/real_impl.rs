//! Scalar, Real, and Field implementations for f32 and f64.

use num_traits::Float;

use super::traits::{Field, Real, Scalar};

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
