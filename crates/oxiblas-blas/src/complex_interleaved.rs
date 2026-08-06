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
//! let interleaved = split_to_interleaved(&real, &imag).expect("real and imag have equal length");
//! assert_eq!(interleaved, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
//!
//! let (re, im) = interleaved_to_split(&interleaved).expect("interleaved length is even");
//! assert_eq!(re, real);
//! assert_eq!(im, imag);
//! ```

use num_complex::{Complex32, Complex64};
use num_traits::Float;
use oxiblas_core::scalar::Field;

// =============================================================================
// Error Type
// =============================================================================

/// Errors that can occur when working with interleaved complex data.
///
/// All of the fallible functions in this module take caller-supplied slice
/// lengths, so violations are reported as a typed [`Result`] error instead of
/// panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterleavedComplexError {
    /// An interleaved array does not have an even number of elements, so it
    /// cannot be decomposed into `(re, im)` pairs.
    OddLength {
        /// The actual (odd) length that was supplied.
        len: usize,
    },
    /// The real-part and imaginary-part arrays passed to a split/interleave
    /// conversion do not have the same length.
    SplitLengthMismatch {
        /// Length of the real-part array.
        real_len: usize,
        /// Length of the imaginary-part array.
        imag_len: usize,
    },
    /// Two interleaved vectors passed to a binary vector operation (dotc,
    /// axpy, ...) do not have the same length.
    VectorLengthMismatch {
        /// Length of the first vector.
        x_len: usize,
        /// Length of the second vector.
        y_len: usize,
    },
}

impl core::fmt::Display for InterleavedComplexError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OddLength { len } => write!(
                f,
                "interleaved complex array must have an even length, got {len}"
            ),
            Self::SplitLengthMismatch { real_len, imag_len } => write!(
                f,
                "real and imaginary arrays must have the same length (real: {real_len}, imag: {imag_len})"
            ),
            Self::VectorLengthMismatch { x_len, y_len } => write!(
                f,
                "interleaved vectors must have the same length (x: {x_len}, y: {y_len})"
            ),
        }
    }
}

impl std::error::Error for InterleavedComplexError {}

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
/// # Errors
///
/// Returns [`InterleavedComplexError::SplitLengthMismatch`] if `real` and
/// `imag` have different lengths.
#[inline]
pub fn split_to_interleaved<T: InterleavedComplex>(
    real: &[T],
    imag: &[T],
) -> Result<Vec<T>, InterleavedComplexError> {
    if real.len() != imag.len() {
        return Err(InterleavedComplexError::SplitLengthMismatch {
            real_len: real.len(),
            imag_len: imag.len(),
        });
    }

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

    Ok(result)
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
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `interleaved.len()` is odd.
#[inline]
pub fn interleaved_to_split<T: InterleavedComplex>(
    interleaved: &[T],
) -> Result<(Vec<T>, Vec<T>), InterleavedComplexError> {
    if interleaved.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength {
            len: interleaved.len(),
        });
    }

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

    Ok((real, imag))
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
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `interleaved.len()` is odd.
#[inline]
pub fn interleaved_to_complex_f64(
    interleaved: &[f64],
) -> Result<Vec<Complex64>, InterleavedComplexError> {
    if interleaved.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength {
            len: interleaved.len(),
        });
    }

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

    Ok(result)
}

/// Convert interleaved format to `num_complex` slice (f32).
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `interleaved.len()` is odd.
#[inline]
pub fn interleaved_to_complex_f32(
    interleaved: &[f32],
) -> Result<Vec<Complex32>, InterleavedComplexError> {
    if interleaved.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength {
            len: interleaved.len(),
        });
    }

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

    Ok(result)
}

// =============================================================================
// In-place Conversion Functions
// =============================================================================

/// Minimal fixed-size bit-vector used only for cycle-visitation bookkeeping by
/// [`permute_in_place`]. This keeps the auxiliary memory used by the
/// "genuinely in-place" conversions below at `O(n)` *bits* (`n / 64` `u64`
/// words), rather than an `O(n)`-element buffer of `T` values.
struct VisitedBits {
    words: Vec<u64>,
}

impl VisitedBits {
    #[inline]
    fn new(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)],
        }
    }

    #[inline]
    fn is_set(&self, index: usize) -> bool {
        (self.words[index / 64] >> (index % 64)) & 1 != 0
    }

    #[inline]
    fn set(&mut self, index: usize) {
        self.words[index / 64] |= 1u64 << (index % 64);
    }
}

/// Applies the permutation described by `dest` to `data` in place by
/// following permutation cycles: every element is moved directly to its
/// final resting place exactly once, so no full-size auxiliary buffer of `T`
/// is required (only the small [`VisitedBits`] bitmap above).
///
/// `dest(i)` must describe a bijection of `0..data.len()` onto itself: the
/// element currently stored at index `i` belongs at index `dest(i)` in the
/// final layout.
fn permute_in_place<T: Copy>(data: &mut [T], dest: impl Fn(usize) -> usize) {
    let total = data.len();
    let mut visited = VisitedBits::new(total);

    for start in 0..total {
        if visited.is_set(start) {
            continue;
        }

        let mut cur = start;
        let mut carry = data[start];
        loop {
            visited.set(cur);
            let d = dest(cur);
            if d == start {
                data[start] = carry;
                break;
            }
            carry = core::mem::replace(&mut data[d], carry);
            cur = d;
        }
    }
}

/// Convert interleaved data to split format in place.
///
/// Real/imaginary de-interleaving is equivalent to transposing a conceptual
/// `n x 2` matrix (rows = complex elements, columns = `[re, im]`) into a
/// `2 x n` matrix (rows = `[all re, all im]`). This is performed with a
/// genuine in-place cycle-following permutation (`permute_in_place`):
/// every element is written to its final position exactly once, and the
/// only auxiliary memory used is a small bit-vector for cycle-visitation
/// bookkeeping -- not a full `O(n)`-element buffer of `T`.
///
/// # Arguments
///
/// * `data` - Mutable slice containing interleaved data. Will be rearranged to
///   contain all real parts in the first half and all imaginary parts in the second half.
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `data.len()` is odd.
pub fn interleaved_to_split_inplace<T: InterleavedComplex>(
    data: &mut [T],
) -> Result<(), InterleavedComplexError> {
    if data.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: data.len() });
    }

    let n = data.len() / 2;
    if n <= 1 {
        return Ok(());
    }

    // Source index `i` holds `re_{i/2}` when `i` is even, or `im_{i/2}` when
    // `i` is odd. In the split layout, `re_k` belongs at index `k` and
    // `im_k` belongs at index `n + k`.
    permute_in_place(data, |i| (i % 2) * n + i / 2);
    Ok(())
}

/// Convert split data to interleaved format in place.
///
/// This is the inverse permutation of [`interleaved_to_split_inplace`]
/// (transposing the conceptual `2 x n` matrix back into `n x 2`), performed
/// with the same in-place cycle-following algorithm and the same `O(n)`-bit
/// (not `O(n)`-element) auxiliary memory footprint.
///
/// # Arguments
///
/// * `data` - Mutable slice containing split data (real parts in first half,
///   imaginary parts in second half). Will be rearranged to interleaved format.
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `data.len()` is odd.
pub fn split_to_interleaved_inplace<T: InterleavedComplex>(
    data: &mut [T],
) -> Result<(), InterleavedComplexError> {
    if data.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: data.len() });
    }

    let n = data.len() / 2;
    if n <= 1 {
        return Ok(());
    }

    // Source index `k` holds `re_k` when `k < n`, or `im_{k-n}` when `k >= n`.
    // In interleaved layout, `re_k` belongs at index `2k` and `im_k` belongs
    // at index `2k + 1`.
    permute_in_place(data, |k| (k % n) * 2 + k / n);
    Ok(())
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
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd, or
/// [`InterleavedComplexError::VectorLengthMismatch`] if `x` and `y` have
/// different lengths.
#[inline]
pub fn dotc_interleaved_f64(x: &[f64], y: &[f64]) -> Result<(f64, f64), InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }
    if x.len() != y.len() {
        return Err(InterleavedComplexError::VectorLengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }

    let n = x.len() / 2;
    if n == 0 {
        return Ok((0.0, 0.0));
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

    Ok(((re0 + re1) + (re2 + re3), (im0 + im1) + (im2 + im3)))
}

/// Dot product of two interleaved complex vectors (f32).
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd, or
/// [`InterleavedComplexError::VectorLengthMismatch`] if `x` and `y` have
/// different lengths.
#[inline]
pub fn dotc_interleaved_f32(x: &[f32], y: &[f32]) -> Result<(f32, f32), InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }
    if x.len() != y.len() {
        return Err(InterleavedComplexError::VectorLengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }

    let n = x.len() / 2;
    if n == 0 {
        return Ok((0.0, 0.0));
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

    Ok(((re0 + re1) + (re2 + re3), (im0 + im1) + (im2 + im3)))
}

/// AXPY operation on interleaved complex vectors: y = alpha * x + y
///
/// # Arguments
///
/// * `alpha_re` - Real part of scalar
/// * `alpha_im` - Imaginary part of scalar
/// * `x` - Source interleaved complex vector
/// * `y` - Destination interleaved complex vector (modified in place)
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd, or
/// [`InterleavedComplexError::VectorLengthMismatch`] if `x` and `y` have
/// different lengths.
#[inline]
pub fn axpy_interleaved_f64(
    alpha_re: f64,
    alpha_im: f64,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }
    if x.len() != y.len() {
        return Err(InterleavedComplexError::VectorLengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }

    let n = x.len() / 2;
    if n == 0 {
        return Ok(());
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

    Ok(())
}

/// AXPY operation on interleaved complex vectors (f32): y = alpha * x + y
///
/// # Arguments
///
/// * `alpha_re` - Real part of scalar
/// * `alpha_im` - Imaginary part of scalar
/// * `x` - Source interleaved complex vector
/// * `y` - Destination interleaved complex vector (modified in place)
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd, or
/// [`InterleavedComplexError::VectorLengthMismatch`] if `x` and `y` have
/// different lengths.
#[inline]
pub fn axpy_interleaved_f32(
    alpha_re: f32,
    alpha_im: f32,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }
    if x.len() != y.len() {
        return Err(InterleavedComplexError::VectorLengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }

    let n = x.len() / 2;
    if n == 0 {
        return Ok(());
    }

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 8;

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

    Ok(())
}

/// SCAL operation on interleaved complex vector: x = alpha * x
///
/// # Arguments
///
/// * `alpha_re` - Real part of scalar
/// * `alpha_im` - Imaginary part of scalar
/// * `x` - Interleaved complex vector (modified in place)
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd.
#[inline]
pub fn scal_interleaved_f64(
    alpha_re: f64,
    alpha_im: f64,
    x: &mut [f64],
) -> Result<(), InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }

    let n = x.len() / 2;
    if n == 0 {
        return Ok(());
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

    Ok(())
}

/// SCAL operation on interleaved complex vector (f32): x = alpha * x
///
/// # Arguments
///
/// * `alpha_re` - Real part of scalar
/// * `alpha_im` - Imaginary part of scalar
/// * `x` - Interleaved complex vector (modified in place)
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd.
#[inline]
pub fn scal_interleaved_f32(
    alpha_re: f32,
    alpha_im: f32,
    x: &mut [f32],
) -> Result<(), InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }

    let n = x.len() / 2;
    if n == 0 {
        return Ok(());
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

    Ok(())
}

/// Compute the Euclidean norm of an interleaved complex vector.
///
/// ||x|| = sqrt(Σ |`x_i|^2`) = sqrt(Σ (`re_i^2` + `im_i^2`))
///
/// `Σ (re_i^2 + im_i^2)` over all complex elements is exactly the sum of
/// squares of every individual real/imaginary component stored in the flat
/// interleaved buffer, so this reuses the crate's numerically stable,
/// scaled-accumulation (Blue's-algorithm-style) real `nrm2` kernel
/// ([`crate::level1::nrm2_f64`]) directly on `x`. This gives interleaved
/// complex vectors with very large or very small magnitude components the
/// same overflow/underflow protection as the real `nrm2` kernels.
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd.
#[inline]
pub fn nrm2_interleaved_f64(x: &[f64]) -> Result<f64, InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }

    Ok(crate::level1::nrm2_f64(x))
}

/// Compute the Euclidean norm of an interleaved complex vector (f32).
///
/// See [`nrm2_interleaved_f64`] for the numerical-stability rationale; this
/// delegates to [`crate::level1::nrm2_f32`] the same way.
///
/// # Errors
///
/// Returns [`InterleavedComplexError::OddLength`] if `x.len()` is odd.
#[inline]
pub fn nrm2_interleaved_f32(x: &[f32]) -> Result<f32, InterleavedComplexError> {
    if x.len() % 2 != 0 {
        return Err(InterleavedComplexError::OddLength { len: x.len() });
    }

    Ok(crate::level1::nrm2_f32(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_to_interleaved() {
        let real = [1.0, 2.0, 3.0, 4.0];
        let imag = [5.0, 6.0, 7.0, 8.0];

        let interleaved =
            split_to_interleaved(&real, &imag).expect("real and imag have equal length");
        assert_eq!(interleaved, [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn test_split_to_interleaved_length_mismatch() {
        let real = [1.0, 2.0, 3.0];
        let imag = [5.0, 6.0];

        let err = split_to_interleaved(&real, &imag).expect_err("lengths differ");
        assert_eq!(
            err,
            InterleavedComplexError::SplitLengthMismatch {
                real_len: 3,
                imag_len: 2
            }
        );
    }

    #[test]
    fn test_interleaved_to_split() {
        let interleaved = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];

        let (real, imag) = interleaved_to_split(&interleaved).expect("interleaved length is even");
        assert_eq!(real, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(imag, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_interleaved_to_split_odd_length() {
        let interleaved = [1.0, 5.0, 2.0];
        let err = interleaved_to_split(&interleaved).expect_err("odd length");
        assert_eq!(err, InterleavedComplexError::OddLength { len: 3 });
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

        let complex = interleaved_to_complex_f64(&interleaved).expect("interleaved length is even");
        assert_eq!(complex[0], Complex64::new(1.0, 2.0));
        assert_eq!(complex[1], Complex64::new(3.0, 4.0));
        assert_eq!(complex[2], Complex64::new(5.0, 6.0));
    }

    #[test]
    fn test_interleaved_to_complex_f64_odd_length() {
        let interleaved = [1.0, 2.0, 3.0];
        let err = interleaved_to_complex_f64(&interleaved).expect_err("odd length");
        assert_eq!(err, InterleavedComplexError::OddLength { len: 3 });
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
        let recovered =
            interleaved_to_complex_f64(&interleaved).expect("interleaved length is even");

        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o.re - r.re).abs() < 1e-10);
            assert!((o.im - r.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inplace_conversion() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Convert interleaved to split
        interleaved_to_split_inplace(&mut data).expect("even length");
        assert_eq!(data, [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);

        // Convert back to interleaved
        split_to_interleaved_inplace(&mut data).expect("even length");
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_inplace_conversion_odd_length_error() {
        let mut data = [1.0, 2.0, 3.0];
        let err = interleaved_to_split_inplace(&mut data).expect_err("odd length");
        assert_eq!(err, InterleavedComplexError::OddLength { len: 3 });

        let mut data2 = [1.0, 2.0, 3.0];
        let err2 = split_to_interleaved_inplace(&mut data2).expect_err("odd length");
        assert_eq!(err2, InterleavedComplexError::OddLength { len: 3 });
    }

    /// The cycle-following in-place algorithm's correctness depends on the
    /// permutation formula holding for every `n`, not just powers of two.
    /// This regression test sweeps a range of sizes -- including odd `n`
    /// and `n` with various small prime factors -- verifying the round trip
    /// `interleaved -> split -> interleaved` reproduces the original data
    /// exactly. This is exactly the kind of bug a subtly-wrong cycle
    /// formula (e.g. one that only works for power-of-two sizes) would have
    /// been caught by.
    #[test]
    fn test_inplace_conversion_many_sizes() {
        for n in 0..=64usize {
            let original: Vec<f64> = (0..2 * n).map(|i| i as f64).collect();
            let mut data = original.clone();

            interleaved_to_split_inplace(&mut data).expect("even length");
            if n > 0 {
                let expected_real: Vec<f64> = (0..n).map(|k| (2 * k) as f64).collect();
                let expected_imag: Vec<f64> = (0..n).map(|k| (2 * k + 1) as f64).collect();
                assert_eq!(&data[..n], expected_real.as_slice(), "n={n}");
                assert_eq!(&data[n..], expected_imag.as_slice(), "n={n}");
            }

            split_to_interleaved_inplace(&mut data).expect("even length");
            assert_eq!(data, original, "round trip failed for n={n}");
        }
    }

    /// Confirms the split->interleaved direction independently (not just as
    /// the second half of a round trip), across a range of sizes.
    #[test]
    fn test_split_to_interleaved_inplace_matches_out_of_place() {
        for n in 0..=40usize {
            let real: Vec<f64> = (0..n).map(|k| k as f64).collect();
            let imag: Vec<f64> = (0..n).map(|k| (k as f64) + 1000.0).collect();

            let mut data = real.clone();
            data.extend_from_slice(&imag);
            split_to_interleaved_inplace(&mut data).expect("even length");

            let expected = split_to_interleaved(&real, &imag).expect("equal length");
            assert_eq!(data, expected, "n={n}");
        }
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

        let (re, im) = dotc_interleaved_f64(&x, &y).expect("even, matching lengths");
        assert!((re - 70.0).abs() < 1e-10);
        assert!((im - (-8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dotc_interleaved_f64_length_mismatch() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0];
        let err = dotc_interleaved_f64(&x, &y).expect_err("length mismatch");
        assert_eq!(
            err,
            InterleavedComplexError::VectorLengthMismatch { x_len: 4, y_len: 2 }
        );
    }

    #[test]
    fn test_axpy_interleaved_f64() {
        // y = alpha * x + y
        // alpha = 2+1i, x = [1+2i, 3+4i], y = [0, 0]
        // alpha * x[0] = (2+i)(1+2i) = 2+4i+i-2 = 5i
        // alpha * x[1] = (2+i)(3+4i) = 6+8i+3i-4 = 2+11i
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0, 0.0, 0.0, 0.0];

        axpy_interleaved_f64(2.0, 1.0, &x, &mut y).expect("even, matching lengths");

        assert!((y[0] - 0.0).abs() < 1e-10); // re of (2+i)(1+2i)
        assert!((y[1] - 5.0).abs() < 1e-10); // im of (2+i)(1+2i)
        assert!((y[2] - 2.0).abs() < 1e-10); // re of (2+i)(3+4i)
        assert!((y[3] - 11.0).abs() < 1e-10); // im of (2+i)(3+4i)
    }

    #[test]
    fn test_axpy_interleaved_f32() {
        // Same hand-computed case as the f64 test, in f32.
        // alpha = 2+1i, x = [1+2i, 3+4i], y = [0, 0]
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0f32, 0.0, 0.0, 0.0];

        axpy_interleaved_f32(2.0, 1.0, &x, &mut y).expect("even, matching lengths");

        assert!((y[0] - 0.0).abs() < 1e-5);
        assert!((y[1] - 5.0).abs() < 1e-5);
        assert!((y[2] - 2.0).abs() < 1e-5);
        assert!((y[3] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_axpy_interleaved_f64_length_mismatch() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0, 0.0];
        let err = axpy_interleaved_f64(1.0, 0.0, &x, &mut y).expect_err("length mismatch");
        assert_eq!(
            err,
            InterleavedComplexError::VectorLengthMismatch { x_len: 4, y_len: 2 }
        );
    }

    #[test]
    fn test_scal_interleaved_f64() {
        // x = alpha * x
        // alpha = 2+1i, x = [1+2i, 3+4i]
        let mut x = [1.0, 2.0, 3.0, 4.0];

        scal_interleaved_f64(2.0, 1.0, &mut x).expect("even length");

        assert!((x[0] - 0.0).abs() < 1e-10); // re of (2+i)(1+2i)
        assert!((x[1] - 5.0).abs() < 1e-10); // im of (2+i)(1+2i)
        assert!((x[2] - 2.0).abs() < 1e-10); // re of (2+i)(3+4i)
        assert!((x[3] - 11.0).abs() < 1e-10); // im of (2+i)(3+4i)
    }

    #[test]
    fn test_scal_interleaved_f32() {
        // Same hand-computed case as the f64 test, in f32.
        let mut x = [1.0f32, 2.0, 3.0, 4.0];

        scal_interleaved_f32(2.0, 1.0, &mut x).expect("even length");

        assert!((x[0] - 0.0).abs() < 1e-5);
        assert!((x[1] - 5.0).abs() < 1e-5);
        assert!((x[2] - 2.0).abs() < 1e-5);
        assert!((x[3] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_scal_interleaved_f64_odd_length() {
        let mut x = [1.0, 2.0, 3.0];
        let err = scal_interleaved_f64(1.0, 0.0, &mut x).expect_err("odd length");
        assert_eq!(err, InterleavedComplexError::OddLength { len: 3 });
    }

    #[test]
    fn test_nrm2_interleaved_f64() {
        // x = [3+4i] => ||x|| = sqrt(9+16) = 5
        let x = [3.0, 4.0];
        let norm = nrm2_interleaved_f64(&x).expect("even length");
        assert!((norm - 5.0).abs() < 1e-10);

        // x = [1+0i, 0+1i] => ||x|| = sqrt(1+1) = sqrt(2)
        let x2 = [1.0, 0.0, 0.0, 1.0];
        let norm2 = nrm2_interleaved_f64(&x2).expect("even length");
        assert!((norm2 - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_interleaved_f32() {
        // Same hand-computed cases as the f64 test, in f32.
        let x = [3.0f32, 4.0];
        let norm = nrm2_interleaved_f32(&x).expect("even length");
        assert!((norm - 5.0).abs() < 1e-5);

        let x2 = [1.0f32, 0.0, 0.0, 1.0];
        let norm2 = nrm2_interleaved_f32(&x2).expect("even length");
        assert!((norm2 - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_nrm2_interleaved_f64_odd_length() {
        let x = [1.0, 2.0, 3.0];
        let err = nrm2_interleaved_f64(&x).expect_err("odd length");
        assert_eq!(err, InterleavedComplexError::OddLength { len: 3 });
    }

    /// Mirrors `level1::nrm2`'s `test_nrm2_overflow_prevention_f64`: values
    /// large enough that naive `re*re + im*im` accumulation would overflow
    /// `f64::MAX` (~1.8e308), which the scaled-accumulation kernel must
    /// avoid.
    #[test]
    fn test_nrm2_interleaved_f64_overflow_prevention() {
        // Two complex elements, each (1e160 + 1e160i).
        // Naive sum of squares: 4 * (1e160)^2 = 4e320, which overflows f64.
        let large_val = 1e160_f64;
        let x = vec![large_val; 4];
        let norm = nrm2_interleaved_f64(&x).expect("even length");
        let expected = 2.0 * large_val; // sqrt(4) * large_val
        assert!(norm.is_finite(), "norm should be finite, got {norm}");
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "expected {expected}, got {norm}"
        );
    }

    /// Mirrors `level1::nrm2`'s underflow-prevention test: values near
    /// `f64::MIN_POSITIVE` should not silently underflow to zero when
    /// squared.
    #[test]
    fn test_nrm2_interleaved_f64_underflow_prevention() {
        let small_val = 1e-308_f64;
        let x = vec![small_val; 4];
        let norm = nrm2_interleaved_f64(&x).expect("even length");
        let expected = 2.0 * small_val;
        assert!(norm > 0.0, "norm should be positive, got {norm}");
        assert!(
            (norm - expected).abs() / expected < 1e-10,
            "expected {expected}, got {norm}"
        );
    }

    #[test]
    fn test_empty_vectors() {
        let empty: [f64; 0] = [];

        let (re, im) = dotc_interleaved_f64(&empty, &empty).expect("empty is valid");
        assert_eq!(re, 0.0);
        assert_eq!(im, 0.0);

        assert_eq!(nrm2_interleaved_f64(&empty).expect("empty is valid"), 0.0);

        let mut empty_mut: [f64; 0] = [];
        axpy_interleaved_f64(1.0, 0.0, &empty, &mut empty_mut).expect("empty is valid");
        scal_interleaved_f64(2.0, 0.0, &mut empty_mut).expect("empty is valid");
    }

    #[test]
    fn test_large_vector() {
        let n = 1000;
        let x: Vec<f64> = (0..2 * n).map(|i| (i % 100) as f64 * 0.01).collect();
        let y: Vec<f64> = (0..2 * n).map(|i| ((i + 1) % 100) as f64 * 0.01).collect();

        // Just verify it runs without panicking
        let (_re, _im) = dotc_interleaved_f64(&x, &y).expect("even, matching lengths");
        let _norm = nrm2_interleaved_f64(&x).expect("even length");

        let mut y_copy = y.clone();
        axpy_interleaved_f64(1.0, 1.0, &x, &mut y_copy).expect("even, matching lengths");

        let mut x_copy = x.clone();
        scal_interleaved_f64(2.0, 0.0, &mut x_copy).expect("even length");
    }

    #[test]
    fn test_odd_element_count_error() {
        // Odd total length (not a multiple of 2) must now be a typed error,
        // not a silently-truncated computation or a panic.
        let x: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..9).map(|i| (i + 1) as f64).collect();

        assert_eq!(
            dotc_interleaved_f64(&x, &y).expect_err("odd length"),
            InterleavedComplexError::OddLength { len: 9 }
        );
        assert_eq!(
            nrm2_interleaved_f64(&x).expect_err("odd length"),
            InterleavedComplexError::OddLength { len: 9 }
        );
    }

    #[test]
    fn test_error_display() {
        let odd = InterleavedComplexError::OddLength { len: 3 };
        assert!(odd.to_string().contains('3'));

        let split_mismatch = InterleavedComplexError::SplitLengthMismatch {
            real_len: 2,
            imag_len: 5,
        };
        assert!(split_mismatch.to_string().contains('2'));
        assert!(split_mismatch.to_string().contains('5'));

        let vec_mismatch = InterleavedComplexError::VectorLengthMismatch { x_len: 4, y_len: 6 };
        assert!(vec_mismatch.to_string().contains('4'));
        assert!(vec_mismatch.to_string().contains('6'));
    }
}
