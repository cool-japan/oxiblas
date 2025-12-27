//! BLAS Level 1 FFI - Vector-Vector operations.

use std::ffi::c_int;
use std::slice;

// =============================================================================
// SDOT - Single precision dot product
// =============================================================================

/// Computes the dot product of two single-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sdot(
    n: c_int,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
) -> f32 {
    if n <= 0 || x.is_null() || y.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x_slice[i * incx] * y_slice[i * incy];
    }
    sum
}

// =============================================================================
// DDOT - Double precision dot product
// =============================================================================

/// Computes the dot product of two double-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ddot(
    n: c_int,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
) -> f64 {
    if n <= 0 || x.is_null() || y.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let mut sum = 0.0f64;
    for i in 0..n {
        sum += x_slice[i * incx] * y_slice[i * incy];
    }
    sum
}

// =============================================================================
// SNRM2 - Single precision Euclidean norm
// =============================================================================

/// Computes the Euclidean norm of a single-precision vector.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_snrm2(n: c_int, x: *const f32, incx: c_int) -> f32 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut sum = 0.0f32;
    for i in 0..n {
        let val = x_slice[i * incx];
        sum += val * val;
    }
    sum.sqrt()
}

// =============================================================================
// DNRM2 - Double precision Euclidean norm
// =============================================================================

/// Computes the Euclidean norm of a double-precision vector.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dnrm2(n: c_int, x: *const f64, incx: c_int) -> f64 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut sum = 0.0f64;
    for i in 0..n {
        let val = x_slice[i * incx];
        sum += val * val;
    }
    sum.sqrt()
}

// =============================================================================
// SASUM - Single precision sum of absolute values
// =============================================================================

/// Computes the sum of absolute values of a single-precision vector.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sasum(n: c_int, x: *const f32, incx: c_int) -> f32 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x_slice[i * incx].abs();
    }
    sum
}

// =============================================================================
// DASUM - Double precision sum of absolute values
// =============================================================================

/// Computes the sum of absolute values of a double-precision vector.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dasum(n: c_int, x: *const f64, incx: c_int) -> f64 {
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut sum = 0.0f64;
    for i in 0..n {
        sum += x_slice[i * incx].abs();
    }
    sum
}

// =============================================================================
// ISAMAX - Index of maximum absolute value (single precision)
// =============================================================================

/// Finds the index of the element with maximum absolute value.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_isamax(n: c_int, x: *const f32, incx: c_int) -> c_int {
    if n <= 0 || x.is_null() {
        return 0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut max_idx = 0;
    let mut max_val = x_slice[0].abs();

    for i in 1..n {
        let val = x_slice[i * incx].abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx as c_int
}

// =============================================================================
// IDAMAX - Index of maximum absolute value (double precision)
// =============================================================================

/// Finds the index of the element with maximum absolute value.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_idamax(n: c_int, x: *const f64, incx: c_int) -> c_int {
    if n <= 0 || x.is_null() {
        return 0;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts(x, n * incx);

    let mut max_idx = 0;
    let mut max_val = x_slice[0].abs();

    for i in 1..n {
        let val = x_slice[i * incx].abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx as c_int
}

// =============================================================================
// SSCAL - Scale vector (single precision)
// =============================================================================

/// Scales a single-precision vector by a constant.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sscal(n: c_int, alpha: f32, x: *mut f32, incx: c_int) {
    if n <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    for i in 0..n {
        x_slice[i * incx] *= alpha;
    }
}

// =============================================================================
// DSCAL - Scale vector (double precision)
// =============================================================================

/// Scales a double-precision vector by a constant.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dscal(n: c_int, alpha: f64, x: *mut f64, incx: c_int) {
    if n <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    for i in 0..n {
        x_slice[i * incx] *= alpha;
    }
}

// =============================================================================
// SAXPY - Vector addition y = alpha*x + y (single precision)
// =============================================================================

/// Computes y = alpha*x + y for single-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_saxpy(
    n: c_int,
    alpha: f32,
    x: *const f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        y_slice[i * incy] += alpha * x_slice[i * incx];
    }
}

// =============================================================================
// DAXPY - Vector addition y = alpha*x + y (double precision)
// =============================================================================

/// Computes y = alpha*x + y for double-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_daxpy(
    n: c_int,
    alpha: f64,
    x: *const f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
) {
    if n <= 0 || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    for i in 0..n {
        y_slice[i * incy] += alpha * x_slice[i * incx];
    }
}

// =============================================================================
// SCOPY - Copy vector (single precision)
// =============================================================================

/// Copies a single-precision vector to another.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_scopy(
    n: c_int,
    x: *const f32,
    incx: c_int,
    y: *mut f32,
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
// DCOPY - Copy vector (double precision)
// =============================================================================

/// Copies a double-precision vector to another.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dcopy(
    n: c_int,
    x: *const f64,
    incx: c_int,
    y: *mut f64,
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
// SSWAP - Swap vectors (single precision)
// =============================================================================

/// Swaps two single-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sswap(n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int) {
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
// DSWAP - Swap vectors (double precision)
// =============================================================================

/// Swaps two double-precision vectors.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dswap(n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int) {
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
// SROTG - Generate Givens rotation (single precision)
// =============================================================================

/// Generates a Givens plane rotation.
///
/// Given a and b, computes c, s, r, z such that:
/// ```text
/// [ c  s ] [ a ]   [ r ]
/// [-s  c ] [ b ] = [ 0 ]
/// ```
///
/// # Safety
/// - All pointers must be valid and non-null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_srotg(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32) {
    use oxiblas_blas::level1::rotg;

    if a.is_null() || b.is_null() || c.is_null() || s.is_null() {
        return;
    }

    let result = rotg(*a, *b);
    *a = result.r;
    *b = result.z;
    *c = result.c;
    *s = result.s;
}

// =============================================================================
// DROTG - Generate Givens rotation (double precision)
// =============================================================================

/// Generates a Givens plane rotation.
///
/// Given a and b, computes c, s, r, z such that:
/// ```text
/// [ c  s ] [ a ]   [ r ]
/// [-s  c ] [ b ] = [ 0 ]
/// ```
///
/// # Safety
/// - All pointers must be valid and non-null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_drotg(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64) {
    use oxiblas_blas::level1::rotg;

    if a.is_null() || b.is_null() || c.is_null() || s.is_null() {
        return;
    }

    let result = rotg(*a, *b);
    *a = result.r;
    *b = result.z;
    *c = result.c;
    *s = result.s;
}

// =============================================================================
// SROT - Apply Givens rotation (single precision)
// =============================================================================

/// Applies a Givens plane rotation to vectors x and y.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_srot(
    n: c_int,
    x: *mut f32,
    incx: c_int,
    y: *mut f32,
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
        let xi = *x.add(i * incx);
        let yi = *y.add(i * incy);
        *x.add(i * incx) = c * xi + s * yi;
        *y.add(i * incy) = c * yi - s * xi;
    }
}

// =============================================================================
// DROT - Apply Givens rotation (double precision)
// =============================================================================

/// Applies a Givens plane rotation to vectors x and y.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_drot(
    n: c_int,
    x: *mut f64,
    incx: c_int,
    y: *mut f64,
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
        let xi = *x.add(i * incx);
        let yi = *y.add(i * incy);
        *x.add(i * incx) = c * xi + s * yi;
        *y.add(i * incy) = c * yi - s * xi;
    }
}

// =============================================================================
// SROTMG - Generate modified Givens rotation (single precision)
// =============================================================================

/// Generates a modified Givens rotation.
///
/// # Safety
/// - All pointers must be valid and non-null
/// - `param` must point to an array of 5 elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_srotmg(
    d1: *mut f32,
    d2: *mut f32,
    x1: *mut f32,
    y1: f32,
    param: *mut f32,
) {
    use oxiblas_blas::level1::rotmg;

    if d1.is_null() || d2.is_null() || x1.is_null() || param.is_null() {
        return;
    }

    let result = rotmg(&mut *d1, &mut *d2, &mut *x1, y1);

    // Store parameters in BLAS format: [flag, h11, h21, h12, h22]
    *param.add(0) = result.flag;
    *param.add(1) = result.h11;
    *param.add(2) = result.h21;
    *param.add(3) = result.h12;
    *param.add(4) = result.h22;
}

// =============================================================================
// DROTMG - Generate modified Givens rotation (double precision)
// =============================================================================

/// Generates a modified Givens rotation.
///
/// # Safety
/// - All pointers must be valid and non-null
/// - `param` must point to an array of 5 elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_drotmg(
    d1: *mut f64,
    d2: *mut f64,
    x1: *mut f64,
    y1: f64,
    param: *mut f64,
) {
    use oxiblas_blas::level1::rotmg;

    if d1.is_null() || d2.is_null() || x1.is_null() || param.is_null() {
        return;
    }

    let result = rotmg(&mut *d1, &mut *d2, &mut *x1, y1);

    // Store parameters in BLAS format: [flag, h11, h21, h12, h22]
    *param.add(0) = result.flag;
    *param.add(1) = result.h11;
    *param.add(2) = result.h21;
    *param.add(3) = result.h12;
    *param.add(4) = result.h22;
}

// =============================================================================
// SROTM - Apply modified Givens rotation (single precision)
// =============================================================================

/// Applies a modified Givens rotation to vectors x and y.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
/// - `param` must point to an array of 5 elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_srotm(
    n: c_int,
    x: *mut f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
    param: *const f32,
) {
    use oxiblas_blas::level1::{RotmParams, rotm};

    if n <= 0 || x.is_null() || y.is_null() || param.is_null() {
        return;
    }

    let params = RotmParams {
        flag: *param.add(0),
        h11: *param.add(1),
        h12: *param.add(3),
        h21: *param.add(2),
        h22: *param.add(4),
    };

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    // For strided access, apply element by element
    if incx == 1 && incy == 1 {
        let x_slice = slice::from_raw_parts_mut(x, n);
        let y_slice = slice::from_raw_parts_mut(y, n);
        rotm(&params, x_slice, y_slice);
    } else {
        // Handle strided case manually
        let zero = 0.0f32;
        let one = 1.0f32;
        let minus_one = -one;
        let minus_two = -2.0f32;

        if params.flag == minus_two {
            return; // Identity
        }

        for i in 0..n {
            let xi = *x.add(i * incx);
            let yi = *y.add(i * incy);

            if params.flag == minus_one {
                *x.add(i * incx) = params.h11 * xi + params.h12 * yi;
                *y.add(i * incy) = params.h21 * xi + params.h22 * yi;
            } else if params.flag == zero {
                *x.add(i * incx) = xi + params.h12 * yi;
                *y.add(i * incy) = params.h21 * xi + yi;
            } else if params.flag == one {
                *x.add(i * incx) = params.h11 * xi + yi;
                *y.add(i * incy) = -xi + params.h22 * yi;
            }
        }
    }
}

// =============================================================================
// DROTM - Apply modified Givens rotation (double precision)
// =============================================================================

/// Applies a modified Givens rotation to vectors x and y.
///
/// # Safety
/// - `x` must point to at least `n` elements spaced by `incx`
/// - `y` must point to at least `n` elements spaced by `incy`
/// - `param` must point to an array of 5 elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_drotm(
    n: c_int,
    x: *mut f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
    param: *const f64,
) {
    use oxiblas_blas::level1::{RotmParams, rotm};

    if n <= 0 || x.is_null() || y.is_null() || param.is_null() {
        return;
    }

    let params = RotmParams {
        flag: *param.add(0),
        h11: *param.add(1),
        h12: *param.add(3),
        h21: *param.add(2),
        h22: *param.add(4),
    };

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    // For strided access, apply element by element
    if incx == 1 && incy == 1 {
        let x_slice = slice::from_raw_parts_mut(x, n);
        let y_slice = slice::from_raw_parts_mut(y, n);
        rotm(&params, x_slice, y_slice);
    } else {
        // Handle strided case manually
        let zero = 0.0f64;
        let one = 1.0f64;
        let minus_one = -one;
        let minus_two = -2.0f64;

        if params.flag == minus_two {
            return; // Identity
        }

        for i in 0..n {
            let xi = *x.add(i * incx);
            let yi = *y.add(i * incy);

            if params.flag == minus_one {
                *x.add(i * incx) = params.h11 * xi + params.h12 * yi;
                *y.add(i * incy) = params.h21 * xi + params.h22 * yi;
            } else if params.flag == zero {
                *x.add(i * incx) = xi + params.h12 * yi;
                *y.add(i * incy) = params.h21 * xi + yi;
            } else if params.flag == one {
                *x.add(i * incx) = params.h11 * xi + yi;
                *y.add(i * incy) = -xi + params.h22 * yi;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddot() {
        let x = [1.0f64, 2.0, 3.0];
        let y = [4.0f64, 5.0, 6.0];

        unsafe {
            let result = oblas_ddot(3, x.as_ptr(), 1, y.as_ptr(), 1);
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assert!((result - 32.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dnrm2() {
        let x = [3.0f64, 4.0];

        unsafe {
            let result = oblas_dnrm2(2, x.as_ptr(), 1);
            // sqrt(9 + 16) = 5
            assert!((result - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dscal() {
        let mut x = [1.0f64, 2.0, 3.0];

        unsafe {
            oblas_dscal(3, 2.0, x.as_mut_ptr(), 1);
        }

        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
        assert!((x[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_daxpy() {
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [4.0f64, 5.0, 6.0];

        unsafe {
            oblas_daxpy(3, 2.0, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }

        // y = 2*x + y = [2+4, 4+5, 6+6] = [6, 9, 12]
        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 9.0).abs() < 1e-10);
        assert!((y[2] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_drotg() {
        let mut a = 3.0f64;
        let mut b = 4.0f64;
        let mut c = 0.0f64;
        let mut s = 0.0f64;

        unsafe {
            oblas_drotg(&mut a, &mut b, &mut c, &mut s);
        }

        // a should be r = 5, c = 0.6, s = 0.8
        assert!((a - 5.0).abs() < 1e-10, "r = {}", a);
        assert!((c - 0.6).abs() < 1e-10, "c = {}", c);
        assert!((s - 0.8).abs() < 1e-10, "s = {}", s);
    }

    #[test]
    fn test_drot() {
        let mut x = [3.0f64];
        let mut y = [4.0f64];
        let c = 0.6;
        let s = 0.8;

        unsafe {
            oblas_drot(1, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, c, s);
        }

        // x = c*3 + s*4 = 1.8 + 3.2 = 5.0
        // y = c*4 - s*3 = 2.4 - 2.4 = 0.0
        assert!((x[0] - 5.0).abs() < 1e-10, "x = {}", x[0]);
        assert!(y[0].abs() < 1e-10, "y = {}", y[0]);
    }

    #[test]
    fn test_drotg_and_drot_combined() {
        // Generate rotation to eliminate 4 from (3, 4)
        let mut a = 3.0f64;
        let mut b = 4.0f64;
        let mut c = 0.0f64;
        let mut s = 0.0f64;

        unsafe {
            oblas_drotg(&mut a, &mut b, &mut c, &mut s);
        }

        let r = a; // r = 5

        // Apply rotation to same vector
        let mut x = [3.0f64];
        let mut y = [4.0f64];

        unsafe {
            oblas_drot(1, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, c, s);
        }

        assert!((x[0] - r).abs() < 1e-10, "x = {}, r = {}", x[0], r);
        assert!(y[0].abs() < 1e-10, "y = {}", y[0]);
    }

    #[test]
    fn test_drotm() {
        // Test with explicit H matrix: flag = -1, H = [[0.6, 0.8], [-0.8, 0.6]]
        let param = [-1.0f64, 0.6, -0.8, 0.8, 0.6]; // [flag, h11, h21, h12, h22]

        let mut x = [1.0f64, 0.0];
        let mut y = [0.0f64, 1.0];

        unsafe {
            oblas_drotm(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, param.as_ptr());
        }

        // x = 0.6*1 + 0.8*0 = 0.6
        // y = -0.8*1 + 0.6*0 = -0.8
        assert!((x[0] - 0.6).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((y[0] - (-0.8)).abs() < 1e-10, "y[0] = {}", y[0]);

        // x = 0.6*0 + 0.8*1 = 0.8
        // y = -0.8*0 + 0.6*1 = 0.6
        assert!((x[1] - 0.8).abs() < 1e-10, "x[1] = {}", x[1]);
        assert!((y[1] - 0.6).abs() < 1e-10, "y[1] = {}", y[1]);
    }
}

// =============================================================================
