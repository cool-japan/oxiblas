//! BLAS Level 1 FFI - Vector-Vector operations.

use std::ffi::c_int;
use std::slice;

use crate::types::{OblasComplex32, OblasComplex64};
use num_complex::{Complex32, Complex64};

// CDOTU - Complex single precision dot product (unconjugated)
// =============================================================================

/// Computes the unconjugated dot product of two complex single-precision vectors.
///
/// CDOTU = x^T * y = Σ x\[i\] * y\[i\]
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cdotu(
    n: c_int,
    x: *const OblasComplex32,
    incx: c_int,
    y: *const OblasComplex32,
    incy: c_int,
) -> OblasComplex32 {
    if n <= 0 || x.is_null() || y.is_null() {
        return OblasComplex32 { re: 0.0, im: 0.0 };
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let mut sum_re = 0.0f32;
    let mut sum_im = 0.0f32;

    for i in 0..n {
        let xi = &x_slice[i * incx];
        let yi = &y_slice[i * incy];
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        sum_re += xi.re * yi.re - xi.im * yi.im;
        sum_im += xi.re * yi.im + xi.im * yi.re;
    }

    OblasComplex32 {
        re: sum_re,
        im: sum_im,
    }
}

// =============================================================================
// ZDOTU - Complex double precision dot product (unconjugated)
// =============================================================================

/// Computes the unconjugated dot product of two complex double-precision vectors.
///
/// ZDOTU = x^T * y = Σ x\[i\] * y\[i\]
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zdotu(
    n: c_int,
    x: *const OblasComplex64,
    incx: c_int,
    y: *const OblasComplex64,
    incy: c_int,
) -> OblasComplex64 {
    if n <= 0 || x.is_null() || y.is_null() {
        return OblasComplex64 { re: 0.0, im: 0.0 };
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;

    for i in 0..n {
        let xi = &x_slice[i * incx];
        let yi = &y_slice[i * incy];
        sum_re += xi.re * yi.re - xi.im * yi.im;
        sum_im += xi.re * yi.im + xi.im * yi.re;
    }

    OblasComplex64 {
        re: sum_re,
        im: sum_im,
    }
}

// =============================================================================
// CDOTC - Complex single precision dot product (conjugated)
// =============================================================================

/// Computes the conjugated dot product of two complex single-precision vectors.
///
/// CDOTC = x^H * y = Σ conj(x\[i\]) * y\[i\]
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cdotc(
    n: c_int,
    x: *const OblasComplex32,
    incx: c_int,
    y: *const OblasComplex32,
    incy: c_int,
) -> OblasComplex32 {
    if n <= 0 || x.is_null() || y.is_null() {
        return OblasComplex32 { re: 0.0, im: 0.0 };
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let mut sum_re = 0.0f32;
    let mut sum_im = 0.0f32;

    for i in 0..n {
        let xi = &x_slice[i * incx];
        let yi = &y_slice[i * incy];
        // conj(a + bi) * (c + di) = (a - bi) * (c + di) = (ac + bd) + (ad - bc)i
        sum_re += xi.re * yi.re + xi.im * yi.im;
        sum_im += xi.re * yi.im - xi.im * yi.re;
    }

    OblasComplex32 {
        re: sum_re,
        im: sum_im,
    }
}

// =============================================================================
// ZDOTC - Complex double precision dot product (conjugated)
// =============================================================================

/// Computes the conjugated dot product of two complex double-precision vectors.
///
/// ZDOTC = x^H * y = Σ conj(x\[i\]) * y\[i\]
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zdotc(
    n: c_int,
    x: *const OblasComplex64,
    incx: c_int,
    y: *const OblasComplex64,
    incy: c_int,
) -> OblasComplex64 {
    if n <= 0 || x.is_null() || y.is_null() {
        return OblasComplex64 { re: 0.0, im: 0.0 };
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;

    for i in 0..n {
        let xi = &x_slice[i * incx];
        let yi = &y_slice[i * incy];
        sum_re += xi.re * yi.re + xi.im * yi.im;
        sum_im += xi.re * yi.im - xi.im * yi.re;
    }

    OblasComplex64 {
        re: sum_re,
        im: sum_im,
    }
}

// =============================================================================
// SCNRM2 - Complex single precision Euclidean norm
// =============================================================================

/// Computes the Euclidean norm of a complex single-precision vector.
///
/// ||x||_2 = sqrt(Σ |x\[i\]|^2) = sqrt(Σ (re\[i\]^2 + im\[i\]^2))
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_scnrm2(n: c_int, x: *const OblasComplex32, incx: c_int) -> f32 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    // Use Blue's algorithm for numerical stability
    let mut scale = 0.0f32;
    let mut ssq = 1.0f32;

    for i in 0..n {
        let xi = &x_slice[i * incx];

        // Process real part
        let abs_re = xi.re.abs();
        if abs_re > 0.0 {
            if scale < abs_re {
                let t = scale / abs_re;
                ssq = 1.0 + ssq * t * t;
                scale = abs_re;
            } else {
                let t = abs_re / scale;
                ssq += t * t;
            }
        }

        // Process imaginary part
        let abs_im = xi.im.abs();
        if abs_im > 0.0 {
            if scale < abs_im {
                let t = scale / abs_im;
                ssq = 1.0 + ssq * t * t;
                scale = abs_im;
            } else {
                let t = abs_im / scale;
                ssq += t * t;
            }
        }
    }

    scale * ssq.sqrt()
}

// =============================================================================
// DZNRM2 - Complex double precision Euclidean norm
// =============================================================================

/// Computes the Euclidean norm of a complex double-precision vector.
///
/// ||x||_2 = sqrt(Σ |x\[i\]|^2) = sqrt(Σ (re\[i\]^2 + im\[i\]^2))
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dznrm2(n: c_int, x: *const OblasComplex64, incx: c_int) -> f64 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    // Use Blue's algorithm for numerical stability
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;

    for i in 0..n {
        let xi = &x_slice[i * incx];

        let abs_re = xi.re.abs();
        if abs_re > 0.0 {
            if scale < abs_re {
                let t = scale / abs_re;
                ssq = 1.0 + ssq * t * t;
                scale = abs_re;
            } else {
                let t = abs_re / scale;
                ssq += t * t;
            }
        }

        let abs_im = xi.im.abs();
        if abs_im > 0.0 {
            if scale < abs_im {
                let t = scale / abs_im;
                ssq = 1.0 + ssq * t * t;
                scale = abs_im;
            } else {
                let t = abs_im / scale;
                ssq += t * t;
            }
        }
    }

    scale * ssq.sqrt()
}

// =============================================================================
// SCASUM - Complex single precision sum of absolute values
// =============================================================================

/// Computes the sum of absolute values of a complex single-precision vector.
///
/// For complex: Σ (|Re(x\[i\])| + |Im(x\[i\])|)
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_scasum(n: c_int, x: *const OblasComplex32, incx: c_int) -> f32 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut sum = 0.0f32;
    for i in 0..n {
        let xi = &x_slice[i * incx];
        sum += xi.re.abs() + xi.im.abs();
    }
    sum
}

// =============================================================================
// DZASUM - Complex double precision sum of absolute values
// =============================================================================

/// Computes the sum of absolute values of a complex double-precision vector.
///
/// For complex: Σ (|Re(x\[i\])| + |Im(x\[i\])|)
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dzasum(n: c_int, x: *const OblasComplex64, incx: c_int) -> f64 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut sum = 0.0f64;
    for i in 0..n {
        let xi = &x_slice[i * incx];
        sum += xi.re.abs() + xi.im.abs();
    }
    sum
}

// =============================================================================
// ICAMAX - Index of maximum absolute value (complex single precision)
// =============================================================================

/// Finds the index of the element with maximum absolute value (|re| + |im|).
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_icamax(n: c_int, x: *const OblasComplex32, incx: c_int) -> c_int {
    if n <= 0 || x.is_null() {
        return 0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let x0 = &x_slice[0];
    let mut max_idx = 0;
    let mut max_val = x0.re.abs() + x0.im.abs();

    for i in 1..n {
        let xi = &x_slice[i * incx];
        let val = xi.re.abs() + xi.im.abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx as c_int
}

// =============================================================================
// IZAMAX - Index of maximum absolute value (complex double precision)
// =============================================================================

/// Finds the index of the element with maximum absolute value (|re| + |im|).
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_izamax(n: c_int, x: *const OblasComplex64, incx: c_int) -> c_int {
    if n <= 0 || x.is_null() {
        return 0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let x0 = &x_slice[0];
    let mut max_idx = 0;
    let mut max_val = x0.re.abs() + x0.im.abs();

    for i in 1..n {
        let xi = &x_slice[i * incx];
        let val = xi.re.abs() + xi.im.abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx as c_int
}

// =============================================================================
// CSCAL - Scale vector by complex scalar (single precision)
// =============================================================================

/// Scales a complex single-precision vector by a complex constant.
///
/// x = alpha * x
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cscal(
    n: c_int,
    alpha: OblasComplex32,
    x: *mut OblasComplex32,
    incx: c_int,
) {
    if n <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    for i in 0..n {
        let xi = &mut x_slice[i * incx];
        let new_re = alpha.re * xi.re - alpha.im * xi.im;
        let new_im = alpha.re * xi.im + alpha.im * xi.re;
        xi.re = new_re;
        xi.im = new_im;
    }
}

// =============================================================================
// ZSCAL - Scale vector by complex scalar (double precision)
// =============================================================================

/// Scales a complex double-precision vector by a complex constant.
///
/// x = alpha * x
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zscal(
    n: c_int,
    alpha: OblasComplex64,
    x: *mut OblasComplex64,
    incx: c_int,
) {
    if n <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    for i in 0..n {
        let xi = &mut x_slice[i * incx];
        let new_re = alpha.re * xi.re - alpha.im * xi.im;
        let new_im = alpha.re * xi.im + alpha.im * xi.re;
        xi.re = new_re;
        xi.im = new_im;
    }
}

// =============================================================================
// CSSCAL - Scale complex vector by real scalar (single precision)
// =============================================================================

/// Scales a complex single-precision vector by a real constant.
///
/// x = alpha * x (where alpha is real)
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_csscal(n: c_int, alpha: f32, x: *mut OblasComplex32, incx: c_int) {
    if n <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    for i in 0..n {
        let xi = &mut x_slice[i * incx];
        xi.re *= alpha;
        xi.im *= alpha;
    }
}

// =============================================================================
// ZDSCAL - Scale complex vector by real scalar (double precision)
// =============================================================================

/// Scales a complex double-precision vector by a real constant.
///
/// x = alpha * x (where alpha is real)
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zdscal(n: c_int, alpha: f64, x: *mut OblasComplex64, incx: c_int) {
    if n <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    for i in 0..n {
        let xi = &mut x_slice[i * incx];
        xi.re *= alpha;
        xi.im *= alpha;
    }
}

// =============================================================================
// CAXPY - Complex single precision y = alpha*x + y
// =============================================================================

/// Computes y = alpha*x + y for complex single-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_caxpy(
    n: c_int,
    alpha: OblasComplex32,
    x: *const OblasComplex32,
    incx: c_int,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    // Skip if alpha is zero
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        let xi = &x_slice[i * incx];
        let yi = &mut y_slice[i * incy];
        // alpha * x + y
        yi.re += alpha.re * xi.re - alpha.im * xi.im;
        yi.im += alpha.re * xi.im + alpha.im * xi.re;
    }
}

// =============================================================================
// ZAXPY - Complex double precision y = alpha*x + y
// =============================================================================

/// Computes y = alpha*x + y for complex double-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zaxpy(
    n: c_int,
    alpha: OblasComplex64,
    x: *const OblasComplex64,
    incx: c_int,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        let xi = &x_slice[i * incx];
        let yi = &mut y_slice[i * incy];
        yi.re += alpha.re * xi.re - alpha.im * xi.im;
        yi.im += alpha.re * xi.im + alpha.im * xi.re;
    }
}

// =============================================================================
// CCOPY - Copy complex vector (single precision)
// =============================================================================

/// Copies a complex single-precision vector to another.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ccopy(
    n: c_int,
    x: *const OblasComplex32,
    incx: c_int,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        y_slice[i * incy] = x_slice[i * incx];
    }
}

// =============================================================================
// ZCOPY - Copy complex vector (double precision)
// =============================================================================

/// Copies a complex double-precision vector to another.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zcopy(
    n: c_int,
    x: *const OblasComplex64,
    incx: c_int,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        y_slice[i * incy] = x_slice[i * incx];
    }
}

// =============================================================================
// CSWAP - Swap complex vectors (single precision)
// =============================================================================

/// Swaps two complex single-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cswap(
    n: c_int,
    x: *mut OblasComplex32,
    incx: c_int,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        std::mem::swap(&mut x_slice[i * incx], &mut y_slice[i * incy]);
    }
}

// =============================================================================
// ZSWAP - Swap complex vectors (double precision)
// =============================================================================

/// Swaps two complex double-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zswap(
    n: c_int,
    x: *mut OblasComplex64,
    incx: c_int,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        std::mem::swap(&mut x_slice[i * incx], &mut y_slice[i * incy]);
    }
}

// =============================================================================
// CROTG - Generate complex Givens rotation (single precision)
// =============================================================================

/// Generates a complex Givens plane rotation.
///
/// Given ca and cb, computes c (real), s (complex), r such that:
/// ```text
/// [ c        s ] [ ca ]   [ r ]
/// [-conj(s)  c ] [ cb ] = [ 0 ]
/// ```
///
/// # Safety
/// - All pointers must be valid and non-null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_crotg(
    ca: *mut OblasComplex32,
    cb: OblasComplex32,
    c: *mut f32,
    s: *mut OblasComplex32,
) {
    if ca.is_null() || c.is_null() || s.is_null() {
        return;
    }

    let a = Complex32::new((*ca).re, (*ca).im);
    let b = Complex32::new(cb.re, cb.im);

    if a.norm() == 0.0 {
        *c = 0.0;
        *s = OblasComplex32 { re: 1.0, im: 0.0 };
        *ca = OblasComplex32 { re: b.re, im: b.im };
        return;
    }

    let scale = a.norm() + b.norm();
    let norm = scale * ((a / scale).norm_sqr() + (b / scale).norm_sqr()).sqrt();

    // alpha = a / |a|
    let alpha = a / a.norm();

    *c = a.norm() / norm;
    let s_val = alpha * b.conj() / norm;
    *s = OblasComplex32 {
        re: s_val.re,
        im: s_val.im,
    };

    let r = alpha * norm;
    *ca = OblasComplex32 { re: r.re, im: r.im };
}

// =============================================================================
// ZROTG - Generate complex Givens rotation (double precision)
// =============================================================================

/// Generates a complex Givens plane rotation.
///
/// Given ca and cb, computes c (real), s (complex), r such that:
/// ```text
/// [ c        s ] [ ca ]   [ r ]
/// [-conj(s)  c ] [ cb ] = [ 0 ]
/// ```
///
/// # Safety
/// - All pointers must be valid and non-null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zrotg(
    ca: *mut OblasComplex64,
    cb: OblasComplex64,
    c: *mut f64,
    s: *mut OblasComplex64,
) {
    if ca.is_null() || c.is_null() || s.is_null() {
        return;
    }

    let a = Complex64::new((*ca).re, (*ca).im);
    let b = Complex64::new(cb.re, cb.im);

    if a.norm() == 0.0 {
        *c = 0.0;
        *s = OblasComplex64 { re: 1.0, im: 0.0 };
        *ca = OblasComplex64 { re: b.re, im: b.im };
        return;
    }

    let scale = a.norm() + b.norm();
    let norm = scale * ((a / scale).norm_sqr() + (b / scale).norm_sqr()).sqrt();

    let alpha = a / a.norm();

    *c = a.norm() / norm;
    let s_val = alpha * b.conj() / norm;
    *s = OblasComplex64 {
        re: s_val.re,
        im: s_val.im,
    };

    let r = alpha * norm;
    *ca = OblasComplex64 { re: r.re, im: r.im };
}

// =============================================================================
// CSROT - Apply real Givens rotation to complex vectors (single precision)
// =============================================================================

/// Applies a real Givens plane rotation to complex single-precision vectors.
///
/// ```text
/// [ x[i] ]   [ c   s ] [ x[i] ]
/// [ y[i] ] = [-s   c ] [ y[i] ]
/// ```
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_csrot(
    n: c_int,
    x: *mut OblasComplex32,
    incx: c_int,
    y: *mut OblasComplex32,
    incy: c_int,
    c: f32,
    s: f32,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    for i in 0..n {
        let xi = &mut *x.add(i * incx);
        let yi = &mut *y.add(i * incy);

        let temp_re = c * xi.re + s * yi.re;
        let temp_im = c * xi.im + s * yi.im;

        yi.re = c * yi.re - s * xi.re;
        yi.im = c * yi.im - s * xi.im;

        xi.re = temp_re;
        xi.im = temp_im;
    }
}

// =============================================================================
// ZDROT - Apply real Givens rotation to complex vectors (double precision)
// =============================================================================

/// Applies a real Givens plane rotation to complex double-precision vectors.
///
/// ```text
/// [ x[i] ]   [ c   s ] [ x[i] ]
/// [ y[i] ] = [-s   c ] [ y[i] ]
/// ```
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zdrot(
    n: c_int,
    x: *mut OblasComplex64,
    incx: c_int,
    y: *mut OblasComplex64,
    incy: c_int,
    c: f64,
    s: f64,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    for i in 0..n {
        let xi = &mut *x.add(i * incx);
        let yi = &mut *y.add(i * incy);

        let temp_re = c * xi.re + s * yi.re;
        let temp_im = c * xi.im + s * yi.im;

        yi.re = c * yi.re - s * xi.re;
        yi.im = c * yi.im - s * xi.im;

        xi.re = temp_re;
        xi.im = temp_im;
    }
}

// =============================================================================
// CROT - Apply complex Givens rotation (single precision)
// =============================================================================

/// Applies a complex Givens rotation to complex single-precision vectors.
///
/// ```text
/// [ x[i] ]   [  c        s ] [ x[i] ]
/// [ y[i] ] = [-conj(s)   c ] [ y[i] ]
/// ```
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_crot(
    n: c_int,
    x: *mut OblasComplex32,
    incx: c_int,
    y: *mut OblasComplex32,
    incy: c_int,
    c: f32,
    s: OblasComplex32,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let sc = Complex32::new(s.re, s.im);
    let sc_conj = sc.conj();

    for i in 0..n {
        let xi = &mut *x.add(i * incx);
        let yi = &mut *y.add(i * incy);

        let x_val = Complex32::new(xi.re, xi.im);
        let y_val = Complex32::new(yi.re, yi.im);

        // x' = c*x + s*y
        let new_x = x_val * c + sc * y_val;
        // y' = -conj(s)*x + c*y
        let new_y = y_val * c - sc_conj * x_val;

        xi.re = new_x.re;
        xi.im = new_x.im;
        yi.re = new_y.re;
        yi.im = new_y.im;
    }
}

// =============================================================================
// ZROT - Apply complex Givens rotation (double precision)
// =============================================================================

/// Applies a complex Givens rotation to complex double-precision vectors.
///
/// ```text
/// [ x[i] ]   [  c        s ] [ x[i] ]
/// [ y[i] ] = [-conj(s)   c ] [ y[i] ]
/// ```
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zrot(
    n: c_int,
    x: *mut OblasComplex64,
    incx: c_int,
    y: *mut OblasComplex64,
    incy: c_int,
    c: f64,
    s: OblasComplex64,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let sc = Complex64::new(s.re, s.im);
    let sc_conj = sc.conj();

    for i in 0..n {
        let xi = &mut *x.add(i * incx);
        let yi = &mut *y.add(i * incy);

        let x_val = Complex64::new(xi.re, xi.im);
        let y_val = Complex64::new(yi.re, yi.im);

        let new_x = x_val * c + sc * y_val;
        let new_y = y_val * c - sc_conj * x_val;

        xi.re = new_x.re;
        xi.im = new_x.im;
        yi.re = new_y.re;
        yi.im = new_y.im;
    }
}

#[cfg(test)]
mod complex_tests {
    use super::*;

    #[test]
    fn test_zdotu() {
        let x = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 4.0 },
        ];
        let y = [
            OblasComplex64 { re: 5.0, im: 6.0 },
            OblasComplex64 { re: 7.0, im: 8.0 },
        ];

        unsafe {
            let result = oblas_zdotu(2, x.as_ptr(), 1, y.as_ptr(), 1);
            // (1+2i)(5+6i) + (3+4i)(7+8i)
            // = (5-12) + (6+10)i + (21-32) + (24+28)i
            // = -7 + 16i + (-11 + 52i)
            // = -18 + 68i
            assert!((result.re - (-18.0)).abs() < 1e-10);
            assert!((result.im - 68.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zdotc() {
        let x = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 4.0 },
        ];
        let y = [
            OblasComplex64 { re: 5.0, im: 6.0 },
            OblasComplex64 { re: 7.0, im: 8.0 },
        ];

        unsafe {
            let result = oblas_zdotc(2, x.as_ptr(), 1, y.as_ptr(), 1);
            // conj(1+2i)(5+6i) + conj(3+4i)(7+8i)
            // = (1-2i)(5+6i) + (3-4i)(7+8i)
            // = (5+12) + (6-10)i + (21+32) + (24-28)i
            // = 17 - 4i + 53 - 4i
            // = 70 - 8i
            assert!((result.re - 70.0).abs() < 1e-10);
            assert!((result.im - (-8.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dznrm2() {
        let x = [
            OblasComplex64 { re: 3.0, im: 4.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_dznrm2(2, x.as_ptr(), 1);
            // sqrt(3^2 + 4^2 + 0 + 0) = sqrt(25) = 5
            assert!((result - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dzasum() {
        let x = [
            OblasComplex64 { re: 1.0, im: -2.0 },
            OblasComplex64 { re: -3.0, im: 4.0 },
        ];

        unsafe {
            let result = oblas_dzasum(2, x.as_ptr(), 1);
            // |1| + |-2| + |-3| + |4| = 1 + 2 + 3 + 4 = 10
            assert!((result - 10.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_izamax() {
        let x = [
            OblasComplex64 { re: 1.0, im: 1.0 },  // |1| + |1| = 2
            OblasComplex64 { re: 3.0, im: -4.0 }, // |3| + |-4| = 7
            OblasComplex64 { re: -2.0, im: 2.0 }, // |-2| + |2| = 4
        ];

        unsafe {
            let result = oblas_izamax(3, x.as_ptr(), 1);
            assert_eq!(result, 1); // index 1 has max |re| + |im|
        }
    }

    #[test]
    fn test_zscal() {
        let mut x = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 4.0 },
        ];
        let alpha = OblasComplex64 { re: 2.0, im: 1.0 };

        unsafe {
            oblas_zscal(2, alpha, x.as_mut_ptr(), 1);
        }

        // (2+i)(1+2i) = 2 + 4i + i + 2i^2 = 2 + 5i - 2 = 0 + 5i
        assert!((x[0].re - 0.0).abs() < 1e-10);
        assert!((x[0].im - 5.0).abs() < 1e-10);

        // (2+i)(3+4i) = 6 + 8i + 3i + 4i^2 = 6 + 11i - 4 = 2 + 11i
        assert!((x[1].re - 2.0).abs() < 1e-10);
        assert!((x[1].im - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_zdscal() {
        let mut x = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 4.0 },
        ];

        unsafe {
            oblas_zdscal(2, 2.0, x.as_mut_ptr(), 1);
        }

        assert!((x[0].re - 2.0).abs() < 1e-10);
        assert!((x[0].im - 4.0).abs() < 1e-10);
        assert!((x[1].re - 6.0).abs() < 1e-10);
        assert!((x[1].im - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_zaxpy() {
        let x = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 4.0 },
        ];
        let mut y = [
            OblasComplex64 { re: 5.0, im: 6.0 },
            OblasComplex64 { re: 7.0, im: 8.0 },
        ];
        let alpha = OblasComplex64 { re: 2.0, im: 0.0 };

        unsafe {
            oblas_zaxpy(2, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }

        // y = 2*x + y
        // y[0] = 2*(1+2i) + (5+6i) = (2+4i) + (5+6i) = 7+10i
        assert!((y[0].re - 7.0).abs() < 1e-10);
        assert!((y[0].im - 10.0).abs() < 1e-10);

        // y[1] = 2*(3+4i) + (7+8i) = (6+8i) + (7+8i) = 13+16i
        assert!((y[1].re - 13.0).abs() < 1e-10);
        assert!((y[1].im - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_zcopy() {
        let x = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 4.0 },
        ];
        let mut y = [
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
        ];

        unsafe {
            oblas_zcopy(2, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }

        assert_eq!(y[0].re, x[0].re);
        assert_eq!(y[0].im, x[0].im);
        assert_eq!(y[1].re, x[1].re);
        assert_eq!(y[1].im, x[1].im);
    }

    #[test]
    fn test_zswap() {
        let mut x = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 4.0 },
        ];
        let mut y = [
            OblasComplex64 { re: 5.0, im: 6.0 },
            OblasComplex64 { re: 7.0, im: 8.0 },
        ];

        unsafe {
            oblas_zswap(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);
        }

        assert!((x[0].re - 5.0).abs() < 1e-10);
        assert!((x[0].im - 6.0).abs() < 1e-10);
        assert!((y[0].re - 1.0).abs() < 1e-10);
        assert!((y[0].im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_zrotg() {
        let mut ca = OblasComplex64 { re: 3.0, im: 4.0 };
        let cb = OblasComplex64 { re: 0.0, im: 0.0 };
        let mut c = 0.0f64;
        let mut s = OblasComplex64 { re: 0.0, im: 0.0 };

        unsafe {
            oblas_zrotg(&mut ca, cb, &mut c, &mut s);
        }

        // When cb = 0, c = 1 and s = 0, r = ca
        assert!((c - 1.0).abs() < 1e-10);
        assert!((s.re).abs() < 1e-10);
        assert!((s.im).abs() < 1e-10);
    }

    #[test]
    fn test_zdrot() {
        let mut x = [OblasComplex64 { re: 1.0, im: 0.0 }];
        let mut y = [OblasComplex64 { re: 0.0, im: 1.0 }];
        let c = (2.0f64).sqrt() / 2.0;
        let s = (2.0f64).sqrt() / 2.0;

        unsafe {
            oblas_zdrot(1, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, c, s);
        }

        // x' = c*x + s*y = c*(1+0i) + s*(0+1i) = c + si
        let expected_x_re = c;
        let expected_x_im = s;
        // y' = c*y - s*x = c*(0+1i) - s*(1+0i) = ci - s = -s + ci
        let expected_y_re = -s;
        let expected_y_im = c;

        assert!((x[0].re - expected_x_re).abs() < 1e-10);
        assert!((x[0].im - expected_x_im).abs() < 1e-10);
        assert!((y[0].re - expected_y_re).abs() < 1e-10);
        assert!((y[0].im - expected_y_im).abs() < 1e-10);
    }
}
