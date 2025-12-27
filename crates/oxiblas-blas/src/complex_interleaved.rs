//! Interleaved complex storage support.
//!
//! This module provides support for interleaved complex number storage format,
//! where real and imaginary parts are stored alternately: [re0, im0, re1, im1, ...].
//!
//! This format is commonly used in BLAS libraries and can be more efficient for
//! certain SIMD operations.
//!
//! # Storage Formats
//!
//! **Interleaved (packed):**
//! ```text
//! [re0, im0, re1, im1, re2, im2, ...]
//! ```
//!
//! **Split (separate arrays):**
//! ```text
//! real: [re0, re1, re2, ...]
//! imag: [im0, im1, im2, ...]
//! ```
//!
//! # Example
//!
//! ```
//! use oxiblas_blas::complex_interleaved::{InterleavedComplex, split_to_interleaved, interleaved_to_split};
//!
//! let real = [1.0, 2.0, 3.0];
//! let imag = [4.0, 5.0, 6.0];
//!
//! let interleaved = split_to_interleaved(&real, &imag);
//! assert_eq!(interleaved, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
//!
//! let (re, im) = interleaved_to_split(&interleaved);
//! assert_eq!(re, real);
//! assert_eq!(im, imag);
//! ```

use num_complex::{Complex32, Complex64};
use num_traits::Float;
use oxiblas_core::scalar::Field;

// =============================================================================
// Type Definitions
// =============================================================================

/// Trait for scalar types that can be used in interleaved complex format.
pub trait InterleavedComplex: Field + Float {
    /// The complex type using this scalar.
    type Complex: Field;

    /// Create a complex number from real and imaginary parts.
    fn to_complex(re: Self, im: Self) -> Self::Complex;

    /// Extract real and imaginary parts from a complex number.
    fn from_complex(c: Self::Complex) -> (Self, Self);
}

impl InterleavedComplex for f32 {
    type Complex = Complex32;

    #[inline]
    fn to_complex(re: Self, im: Self) -> Self::Complex {
        Complex32::new(re, im)
    }

    #[inline]
    fn from_complex(c: Self::Complex) -> (Self, Self) {
        (c.re, c.im)
    }
}

impl InterleavedComplex for f64 {
    type Complex = Complex64;

    #[inline]
    fn to_complex(re: Self, im: Self) -> Self::Complex {
        Complex64::new(re, im)
    }

    #[inline]
    fn from_complex(c: Self::Complex) -> (Self, Self) {
        (c.re, c.im)
    }
}

// =============================================================================
// Conversion Functions
// =============================================================================

/// Convert split complex arrays to interleaved format.
///
/// # Arguments
///
/// * `real` - Array of real parts
/// * `imag` - Array of imaginary parts
///
/// # Returns
///
/// Interleaved array of size 2*n.
///
/// # Panics
///
/// Panics if arrays have different lengths.
#[inline]
pub fn split_to_interleaved<T: InterleavedComplex>(real: &[T], imag: &[T]) -> Vec<T> {
    assert_eq!(real.len(), imag.len(), "Arrays must have same length");

    let n = real.len();
    let mut result = Vec::with_capacity(2 * n);

    // Use SIMD-friendly 4-way unrolling
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        result.push(real[base]);
        result.push(imag[base]);
        result.push(real[base + 1]);
        result.push(imag[base + 1]);
        result.push(real[base + 2]);
        result.push(imag[base + 2]);
        result.push(real[base + 3]);
        result.push(imag[base + 3]);
    }

    // Handle remainder
    for i in (n - remainder)..n {
        result.push(real[i]);
        result.push(imag[i]);
    }

    result
}

/// Convert interleaved complex array to split format.
///
/// # Arguments
///
/// * `interleaved` - Interleaved array [re0, im0, re1, im1, ...]
///
/// # Returns
///
/// Tuple of (real array, imaginary array).
///
/// # Panics
///
/// Panics if array length is not even.
#[inline]
pub fn interleaved_to_split<T: InterleavedComplex>(interleaved: &[T]) -> (Vec<T>, Vec<T>) {
    assert!(
        interleaved.len() % 2 == 0,
        "Interleaved array must have even length"
    );

    let n = interleaved.len() / 2;
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);

    // Use 4-way unrolling for better performance
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;
        real.push(interleaved[base]);
        imag.push(interleaved[base + 1]);
        real.push(interleaved[base + 2]);
        imag.push(interleaved[base + 3]);
        real.push(interleaved[base + 4]);
        imag.push(interleaved[base + 5]);
        real.push(interleaved[base + 6]);
        imag.push(interleaved[base + 7]);
    }

    // Handle remainder
    for i in (n - remainder)..n {
        let base = i * 2;
        real.push(interleaved[base]);
        imag.push(interleaved[base + 1]);
    }

    (real, imag)
}

/// Convert `num_complex` slice to interleaved format.
///
/// # Arguments
///
/// * `complex` - Slice of complex numbers
///
/// # Returns
///
/// Interleaved array of size 2*n.
#[inline]
#[must_use]
pub fn complex_to_interleaved_f64(complex: &[Complex64]) -> Vec<f64> {
    let n = complex.len();
    let mut result = Vec::with_capacity(2 * n);

    // 4-way unrolling
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        result.push(complex[base].re);
        result.push(complex[base].im);
        result.push(complex[base + 1].re);
        result.push(complex[base + 1].im);
        result.push(complex[base + 2].re);
        result.push(complex[base + 2].im);
        result.push(complex[base + 3].re);
        result.push(complex[base + 3].im);
    }

    for i in (n - remainder)..n {
        result.push(complex[i].re);
        result.push(complex[i].im);
    }

    result
}

/// Convert `num_complex` slice to interleaved format (f32).
#[inline]
#[must_use]
pub fn complex_to_interleaved_f32(complex: &[Complex32]) -> Vec<f32> {
    let n = complex.len();
    let mut result = Vec::with_capacity(2 * n);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        result.push(complex[base].re);
        result.push(complex[base].im);
        result.push(complex[base + 1].re);
        result.push(complex[base + 1].im);
        result.push(complex[base + 2].re);
        result.push(complex[base + 2].im);
        result.push(complex[base + 3].re);
        result.push(complex[base + 3].im);
    }

    for i in (n - remainder)..n {
        result.push(complex[i].re);
        result.push(complex[i].im);
    }

    result
}

/// Convert interleaved format to `num_complex` slice.
///
/// # Arguments
///
/// * `interleaved` - Interleaved array [re0, im0, re1, im1, ...]
///
/// # Returns
///
/// Vector of complex numbers.
///
/// # Panics
///
/// Panics if array length is not even.
#[inline]
#[must_use]
pub fn interleaved_to_complex_f64(interleaved: &[f64]) -> Vec<Complex64> {
    assert!(
        interleaved.len() % 2 == 0,
        "Interleaved array must have even length"
    );

    let n = interleaved.len() / 2;
    let mut result = Vec::with_capacity(n);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;
        result.push(Complex64::new(interleaved[base], interleaved[base + 1]));
        result.push(Complex64::new(interleaved[base + 2], interleaved[base + 3]));
        result.push(Complex64::new(interleaved[base + 4], interleaved[base + 5]));
        result.push(Complex64::new(interleaved[base + 6], interleaved[base + 7]));
    }

    for i in (n - remainder)..n {
        let base = i * 2;
        result.push(Complex64::new(interleaved[base], interleaved[base + 1]));
    }

    result
}

/// Convert interleaved format to `num_complex` slice (f32).
#[inline]
#[must_use]
pub fn interleaved_to_complex_f32(interleaved: &[f32]) -> Vec<Complex32> {
    assert!(
        interleaved.len() % 2 == 0,
        "Interleaved array must have even length"
    );

    let n = interleaved.len() / 2;
    let mut result = Vec::with_capacity(n);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;
        result.push(Complex32::new(interleaved[base], interleaved[base + 1]));
        result.push(Complex32::new(interleaved[base + 2], interleaved[base + 3]));
        result.push(Complex32::new(interleaved[base + 4], interleaved[base + 5]));
        result.push(Complex32::new(interleaved[base + 6], interleaved[base + 7]));
    }

    for i in (n - remainder)..n {
        let base = i * 2;
        result.push(Complex32::new(interleaved[base], interleaved[base + 1]));
    }

    result
}

// =============================================================================
// In-place Conversion Functions
// =============================================================================

/// Convert interleaved data to split format in place.
///
/// This is more efficient than allocating new buffers when the data is already
/// contiguous and can be rearranged.
///
/// # Arguments
///
/// * `data` - Mutable slice containing interleaved data. Will be rearranged to
///   contain all real parts in the first half and all imaginary parts in the second half.
///
/// # Panics
///
/// Panics if array length is not even.
pub fn interleaved_to_split_inplace<T: InterleavedComplex>(data: &mut [T]) {
    assert!(
        data.len() % 2 == 0,
        "Interleaved array must have even length"
    );

    let n = data.len() / 2;
    if n <= 1 {
        return;
    }

    // Use auxiliary buffer for efficient in-place conversion
    // This is O(n) time and O(n) space, but more cache-friendly than the
    // O(1) space algorithm which has poor locality
    let mut temp = Vec::with_capacity(n);

    // Extract imaginary parts to temp
    for i in 0..n {
        temp.push(data[i * 2 + 1]);
    }

    // Move real parts to first half
    for i in 0..n {
        data[i] = data[i * 2];
    }

    // Copy imaginary parts to second half
    for i in 0..n {
        data[n + i] = temp[i];
    }
}

/// Convert split data to interleaved format in place.
///
/// # Arguments
///
/// * `data` - Mutable slice containing split data (real parts in first half,
///   imaginary parts in second half). Will be rearranged to interleaved format.
///
/// # Panics
///
/// Panics if array length is not even.
pub fn split_to_interleaved_inplace<T: InterleavedComplex>(data: &mut [T]) {
    assert!(data.len() % 2 == 0, "Array must have even length");

    let n = data.len() / 2;
    if n <= 1 {
        return;
    }

    // Use auxiliary buffers for efficient in-place conversion
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);

    // Copy real and imaginary parts
    for i in 0..n {
        real.push(data[i]);
        imag.push(data[n + i]);
    }

    // Interleave
    for i in 0..n {
        data[i * 2] = real[i];
        data[i * 2 + 1] = imag[i];
    }
}

// =============================================================================
// SIMD-Optimized Operations for Interleaved Complex
// =============================================================================

/// Dot product of two interleaved complex vectors.
///
/// Computes conj(x) · y = Σ `conj(x_i)` * `y_i`
///
/// # Arguments
///
/// * `x` - First interleaved complex vector
/// * `y` - Second interleaved complex vector
///
/// # Returns
///
/// Complex dot product as (real, imaginary) tuple.
#[inline]
#[must_use]
pub fn dotc_interleaved_f64(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert!(x.len() % 2 == 0, "x must have even length");
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    let n = x.len() / 2;
    if n == 0 {
        return (0.0, 0.0);
    }

    // Use 4-way accumulation for better numerical stability and performance
    let mut re0 = 0.0;
    let mut re1 = 0.0;
    let mut re2 = 0.0;
    let mut re3 = 0.0;
    let mut im0 = 0.0;
    let mut im1 = 0.0;
    let mut im2 = 0.0;
    let mut im3 = 0.0;

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;

        // x0 = x_re0 - i*x_im0 (conjugate)
        // y0 = y_re0 + i*y_im0
        // conj(x0) * y0 = (x_re0*y_re0 + x_im0*y_im0) + i*(x_re0*y_im0 - x_im0*y_re0)
        let x_re0 = x[base];
        let x_im0 = x[base + 1];
        let y_re0 = y[base];
        let y_im0 = y[base + 1];
        re0 += x_re0.mul_add(y_re0, x_im0 * y_im0);
        im0 += x_re0.mul_add(y_im0, -(x_im0 * y_re0));

        let x_re1 = x[base + 2];
        let x_im1 = x[base + 3];
        let y_re1 = y[base + 2];
        let y_im1 = y[base + 3];
        re1 += x_re1.mul_add(y_re1, x_im1 * y_im1);
        im1 += x_re1.mul_add(y_im1, -(x_im1 * y_re1));

        let x_re2 = x[base + 4];
        let x_im2 = x[base + 5];
        let y_re2 = y[base + 4];
        let y_im2 = y[base + 5];
        re2 += x_re2.mul_add(y_re2, x_im2 * y_im2);
        im2 += x_re2.mul_add(y_im2, -(x_im2 * y_re2));

        let x_re3 = x[base + 6];
        let x_im3 = x[base + 7];
        let y_re3 = y[base + 6];
        let y_im3 = y[base + 7];
        re3 += x_re3.mul_add(y_re3, x_im3 * y_im3);
        im3 += x_re3.mul_add(y_im3, -(x_im3 * y_re3));
    }

    // Handle remainder
    for i in (n - remainder)..n {
        let base = i * 2;
        let x_re = x[base];
        let x_im = x[base + 1];
        let y_re = y[base];
        let y_im = y[base + 1];
        re0 += x_re.mul_add(y_re, x_im * y_im);
        im0 += x_re.mul_add(y_im, -(x_im * y_re));
    }

    ((re0 + re1) + (re2 + re3), (im0 + im1) + (im2 + im3))
}

/// Dot product of two interleaved complex vectors (f32).
#[inline]
#[must_use]
pub fn dotc_interleaved_f32(x: &[f32], y: &[f32]) -> (f32, f32) {
    assert!(x.len() % 2 == 0, "x must have even length");
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    let n = x.len() / 2;
    if n == 0 {
        return (0.0, 0.0);
    }

    let mut re0 = 0.0_f32;
    let mut re1 = 0.0_f32;
    let mut re2 = 0.0_f32;
    let mut re3 = 0.0_f32;
    let mut im0 = 0.0_f32;
    let mut im1 = 0.0_f32;
    let mut im2 = 0.0_f32;
    let mut im3 = 0.0_f32;

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;

        let x_re0 = x[base];
        let x_im0 = x[base + 1];
        let y_re0 = y[base];
        let y_im0 = y[base + 1];
        re0 += x_re0.mul_add(y_re0, x_im0 * y_im0);
        im0 += x_re0.mul_add(y_im0, -(x_im0 * y_re0));

        let x_re1 = x[base + 2];
        let x_im1 = x[base + 3];
        let y_re1 = y[base + 2];
        let y_im1 = y[base + 3];
        re1 += x_re1.mul_add(y_re1, x_im1 * y_im1);
        im1 += x_re1.mul_add(y_im1, -(x_im1 * y_re1));

        let x_re2 = x[base + 4];
        let x_im2 = x[base + 5];
        let y_re2 = y[base + 4];
        let y_im2 = y[base + 5];
        re2 += x_re2.mul_add(y_re2, x_im2 * y_im2);
        im2 += x_re2.mul_add(y_im2, -(x_im2 * y_re2));

        let x_re3 = x[base + 6];
        let x_im3 = x[base + 7];
        let y_re3 = y[base + 6];
        let y_im3 = y[base + 7];
        re3 += x_re3.mul_add(y_re3, x_im3 * y_im3);
        im3 += x_re3.mul_add(y_im3, -(x_im3 * y_re3));
    }

    for i in (n - remainder)..n {
        let base = i * 2;
        let x_re = x[base];
        let x_im = x[base + 1];
        let y_re = y[base];
        let y_im = y[base + 1];
        re0 += x_re.mul_add(y_re, x_im * y_im);
        im0 += x_re.mul_add(y_im, -(x_im * y_re));
    }

    ((re0 + re1) + (re2 + re3), (im0 + im1) + (im2 + im3))
}

/// AXPY operation on interleaved complex vectors: y = alpha * x + y
///
/// # Arguments
///
/// * `alpha_re` - Real part of scalar
/// * `alpha_im` - Imaginary part of scalar
/// * `x` - Source interleaved complex vector
/// * `y` - Destination interleaved complex vector (modified in place)
#[inline]
pub fn axpy_interleaved_f64(alpha_re: f64, alpha_im: f64, x: &[f64], y: &mut [f64]) {
    assert!(x.len() % 2 == 0, "x must have even length");
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    let n = x.len() / 2;
    if n == 0 {
        return;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;

        // y = alpha * x + y
        // alpha * x = (alpha_re + i*alpha_im) * (x_re + i*x_im)
        //           = (alpha_re*x_re - alpha_im*x_im) + i*(alpha_re*x_im + alpha_im*x_re)
        let x_re0 = x[base];
        let x_im0 = x[base + 1];
        y[base] += alpha_re.mul_add(x_re0, -(alpha_im * x_im0));
        y[base + 1] += alpha_re.mul_add(x_im0, alpha_im * x_re0);

        let x_re1 = x[base + 2];
        let x_im1 = x[base + 3];
        y[base + 2] += alpha_re.mul_add(x_re1, -(alpha_im * x_im1));
        y[base + 3] += alpha_re.mul_add(x_im1, alpha_im * x_re1);

        let x_re2 = x[base + 4];
        let x_im2 = x[base + 5];
        y[base + 4] += alpha_re.mul_add(x_re2, -(alpha_im * x_im2));
        y[base + 5] += alpha_re.mul_add(x_im2, alpha_im * x_re2);

        let x_re3 = x[base + 6];
        let x_im3 = x[base + 7];
        y[base + 6] += alpha_re.mul_add(x_re3, -(alpha_im * x_im3));
        y[base + 7] += alpha_re.mul_add(x_im3, alpha_im * x_re3);
    }

    for i in (n - remainder)..n {
        let base = i * 2;
        let x_re = x[base];
        let x_im = x[base + 1];
        y[base] += alpha_re.mul_add(x_re, -(alpha_im * x_im));
        y[base + 1] += alpha_re.mul_add(x_im, alpha_im * x_re);
    }
}

/// SCAL operation on interleaved complex vector: x = alpha * x
///
/// # Arguments
///
/// * `alpha_re` - Real part of scalar
/// * `alpha_im` - Imaginary part of scalar
/// * `x` - Interleaved complex vector (modified in place)
#[inline]
pub fn scal_interleaved_f64(alpha_re: f64, alpha_im: f64, x: &mut [f64]) {
    assert!(x.len() % 2 == 0, "x must have even length");

    let n = x.len() / 2;
    if n == 0 {
        return;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;

        let x_re0 = x[base];
        let x_im0 = x[base + 1];
        x[base] = alpha_re.mul_add(x_re0, -(alpha_im * x_im0));
        x[base + 1] = alpha_re.mul_add(x_im0, alpha_im * x_re0);

        let x_re1 = x[base + 2];
        let x_im1 = x[base + 3];
        x[base + 2] = alpha_re.mul_add(x_re1, -(alpha_im * x_im1));
        x[base + 3] = alpha_re.mul_add(x_im1, alpha_im * x_re1);

        let x_re2 = x[base + 4];
        let x_im2 = x[base + 5];
        x[base + 4] = alpha_re.mul_add(x_re2, -(alpha_im * x_im2));
        x[base + 5] = alpha_re.mul_add(x_im2, alpha_im * x_re2);

        let x_re3 = x[base + 6];
        let x_im3 = x[base + 7];
        x[base + 6] = alpha_re.mul_add(x_re3, -(alpha_im * x_im3));
        x[base + 7] = alpha_re.mul_add(x_im3, alpha_im * x_re3);
    }

    for i in (n - remainder)..n {
        let base = i * 2;
        let x_re = x[base];
        let x_im = x[base + 1];
        x[base] = alpha_re.mul_add(x_re, -(alpha_im * x_im));
        x[base + 1] = alpha_re.mul_add(x_im, alpha_im * x_re);
    }
}

/// Compute the Euclidean norm of an interleaved complex vector.
///
/// ||x|| = sqrt(Σ |`x_i|^2`) = sqrt(Σ (`re_i^2` + `im_i^2`))
#[inline]
#[must_use]
pub fn nrm2_interleaved_f64(x: &[f64]) -> f64 {
    assert!(x.len() % 2 == 0, "x must have even length");

    let n = x.len() / 2;
    if n == 0 {
        return 0.0;
    }

    // Use 4-way accumulation
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;

        let re0 = x[base];
        let im0 = x[base + 1];
        sum0 += re0.mul_add(re0, im0 * im0);

        let re1 = x[base + 2];
        let im1 = x[base + 3];
        sum1 += re1.mul_add(re1, im1 * im1);

        let re2 = x[base + 4];
        let im2 = x[base + 5];
        sum2 += re2.mul_add(re2, im2 * im2);

        let re3 = x[base + 6];
        let im3 = x[base + 7];
        sum3 += re3.mul_add(re3, im3 * im3);
    }

    for i in (n - remainder)..n {
        let base = i * 2;
        let re = x[base];
        let im = x[base + 1];
        sum0 += re.mul_add(re, im * im);
    }

    ((sum0 + sum1) + (sum2 + sum3)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_to_interleaved() {
        let real = [1.0, 2.0, 3.0, 4.0];
        let imag = [5.0, 6.0, 7.0, 8.0];

        let interleaved = split_to_interleaved(&real, &imag);
        assert_eq!(interleaved, [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn test_interleaved_to_split() {
        let interleaved = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];

        let (real, imag) = interleaved_to_split(&interleaved);
        assert_eq!(real, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(imag, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_complex_to_interleaved_f64() {
        let complex = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ];

        let interleaved = complex_to_interleaved_f64(&complex);
        assert_eq!(interleaved, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_interleaved_to_complex_f64() {
        let interleaved = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let complex = interleaved_to_complex_f64(&interleaved);
        assert_eq!(complex[0], Complex64::new(1.0, 2.0));
        assert_eq!(complex[1], Complex64::new(3.0, 4.0));
        assert_eq!(complex[2], Complex64::new(5.0, 6.0));
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = [
            Complex64::new(1.5, 2.5),
            Complex64::new(3.5, 4.5),
            Complex64::new(5.5, 6.5),
            Complex64::new(7.5, 8.5),
            Complex64::new(9.5, 10.5),
        ];

        let interleaved = complex_to_interleaved_f64(&original);
        let recovered = interleaved_to_complex_f64(&interleaved);

        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o.re - r.re).abs() < 1e-10);
            assert!((o.im - r.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inplace_conversion() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Convert interleaved to split
        interleaved_to_split_inplace(&mut data);
        assert_eq!(data, [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);

        // Convert back to interleaved
        split_to_interleaved_inplace(&mut data);
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_dotc_interleaved_f64() {
        // x = [1+2i, 3+4i], y = [5+6i, 7+8i]
        // conj(x) * y = (1-2i)(5+6i) + (3-4i)(7+8i)
        //             = (5+6i-10i+12) + (21+24i-28i+32)
        //             = (17-4i) + (53-4i)
        //             = 70 - 8i
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0, 7.0, 8.0];

        let (re, im) = dotc_interleaved_f64(&x, &y);
        assert!((re - 70.0).abs() < 1e-10);
        assert!((im - (-8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_interleaved_f64() {
        // y = alpha * x + y
        // alpha = 2+1i, x = [1+2i, 3+4i], y = [0, 0]
        // alpha * x[0] = (2+i)(1+2i) = 2+4i+i-2 = 5i
        // alpha * x[1] = (2+i)(3+4i) = 6+8i+3i-4 = 2+11i
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0, 0.0, 0.0, 0.0];

        axpy_interleaved_f64(2.0, 1.0, &x, &mut y);

        assert!((y[0] - 0.0).abs() < 1e-10); // re of (2+i)(1+2i)
        assert!((y[1] - 5.0).abs() < 1e-10); // im of (2+i)(1+2i)
        assert!((y[2] - 2.0).abs() < 1e-10); // re of (2+i)(3+4i)
        assert!((y[3] - 11.0).abs() < 1e-10); // im of (2+i)(3+4i)
    }

    #[test]
    fn test_scal_interleaved_f64() {
        // x = alpha * x
        // alpha = 2+1i, x = [1+2i, 3+4i]
        let mut x = [1.0, 2.0, 3.0, 4.0];

        scal_interleaved_f64(2.0, 1.0, &mut x);

        assert!((x[0] - 0.0).abs() < 1e-10); // re of (2+i)(1+2i)
        assert!((x[1] - 5.0).abs() < 1e-10); // im of (2+i)(1+2i)
        assert!((x[2] - 2.0).abs() < 1e-10); // re of (2+i)(3+4i)
        assert!((x[3] - 11.0).abs() < 1e-10); // im of (2+i)(3+4i)
    }

    #[test]
    fn test_nrm2_interleaved_f64() {
        // x = [3+4i] => ||x|| = sqrt(9+16) = 5
        let x = [3.0, 4.0];
        let norm = nrm2_interleaved_f64(&x);
        assert!((norm - 5.0).abs() < 1e-10);

        // x = [1+0i, 0+1i] => ||x|| = sqrt(1+1) = sqrt(2)
        let x2 = [1.0, 0.0, 0.0, 1.0];
        let norm2 = nrm2_interleaved_f64(&x2);
        assert!((norm2 - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_empty_vectors() {
        let empty: [f64; 0] = [];

        let (re, im) = dotc_interleaved_f64(&empty, &empty);
        assert_eq!(re, 0.0);
        assert_eq!(im, 0.0);

        assert_eq!(nrm2_interleaved_f64(&empty), 0.0);

        let mut empty_mut: [f64; 0] = [];
        axpy_interleaved_f64(1.0, 0.0, &empty, &mut empty_mut);
        scal_interleaved_f64(2.0, 0.0, &mut empty_mut);
    }

    #[test]
    fn test_large_vector() {
        let n = 1000;
        let x: Vec<f64> = (0..2 * n).map(|i| (i % 100) as f64 * 0.01).collect();
        let y: Vec<f64> = (0..2 * n).map(|i| ((i + 1) % 100) as f64 * 0.01).collect();

        // Just verify it runs without panicking
        let (_re, _im) = dotc_interleaved_f64(&x, &y);
        let _norm = nrm2_interleaved_f64(&x);

        let mut y_copy = y.clone();
        axpy_interleaved_f64(1.0, 1.0, &x, &mut y_copy);

        let mut x_copy = x.clone();
        scal_interleaved_f64(2.0, 0.0, &mut x_copy);
    }

    #[test]
    fn test_odd_element_count() {
        // Test with odd number of complex elements (5 elements = 10 reals)
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect();

        let (_re, _im) = dotc_interleaved_f64(&x, &y);
        let _norm = nrm2_interleaved_f64(&x);
    }
}
