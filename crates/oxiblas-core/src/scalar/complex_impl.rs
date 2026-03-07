//! Scalar, ComplexScalar, and Field implementations for Complex32 and Complex64.

use num_complex::{Complex32, Complex64};

use super::traits::{ComplexScalar, Field, Scalar};

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
