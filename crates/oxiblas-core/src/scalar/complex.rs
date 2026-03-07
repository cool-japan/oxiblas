//! Complex number constructors, utilities, and extension traits.

use num_complex::{Complex32, Complex64};

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

    /// Returns true if this complex number is purely real (imaginary part is approximately 0).
    #[allow(clippy::wrong_self_convention)] // Complex numbers are Copy, self by value is efficient
    fn is_purely_real(self, tolerance: Self::Real) -> bool;

    /// Returns true if this complex number is purely imaginary (real part is approximately 0).
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
