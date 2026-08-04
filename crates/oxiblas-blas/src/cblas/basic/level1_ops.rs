//! CBLAS Level 1 (vector-vector) real operations.

use super::super::validate::inc_valid;
use super::vector_start_offset;
use crate::level1;

/// Double precision dot product.
///
/// Computes: x · y
///
/// Uses optimized SIMD implementation for unit stride vectors.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ddot(
    n: i32,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
) -> f64 {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // return the DDOT identity (0) instead. A zero increment aliases every
    // logical element onto element 0 (meaningless for a reduction), and a
    // null operand would be dereferenced unconditionally by the loop below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts(y, n);
        return level1::dot(x_slice, y_slice);
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = 0.0;
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
    }
    result
}

/// Single precision dot product.
///
/// Uses optimized SIMD implementation for unit stride vectors.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sdot(
    n: i32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
) -> f32 {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // return the SDOT identity (0) instead. A zero increment aliases every
    // logical element onto element 0 (meaningless for a reduction), and a
    // null operand would be dereferenced unconditionally by the loop below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts(y, n);
        return level1::dot(x_slice, y_slice);
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = 0.0f32;
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
    }
    result
}

/// Double precision Euclidean norm.
///
/// Uses Blue's algorithm for numerical stability with unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64 {
    // Reference DNRM2 is defined only for incx >= 1; anything else returns 0.
    // A null `x` would be dereferenced unconditionally by the loop below.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation with Blue's algorithm
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: strided. Use Blue's scaled sum-of-squares (mirroring the core
    // `level1::nrm2` implementation) instead of a naive Σx² so that vectors with
    // very large or very small magnitudes do not overflow/underflow to inf/0.
    let incx = incx as isize;
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    let mut ix = 0isize;
    for _ in 0..n {
        let abs_xi = (*x.offset(ix)).abs();
        // `!= 0.0` (not `> 0.0`): a NaN element must still enter this branch
        // and poison the accumulator, matching `nrm2_fold` in
        // `level1/nrm2.rs` (`>` is always false for NaN).
        if abs_xi != 0.0 {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = (ssq * t).mul_add(t, 1.0);
                scale = abs_xi;
            } else {
                let t = if scale == abs_xi { 1.0 } else { abs_xi / scale };
                ssq += t * t;
            }
        }
        ix += incx;
    }
    scale * ssq.sqrt()
}

/// Single precision Euclidean norm.
///
/// Uses Blue's algorithm for numerical stability with unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32 {
    // Reference SNRM2 is defined only for incx >= 1; anything else returns 0.
    // A null `x` would be dereferenced unconditionally by the loop below.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation with Blue's algorithm
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: strided. Use Blue's scaled sum-of-squares (mirroring the core
    // `level1::nrm2` implementation) instead of a naive Σx² so that vectors with
    // very large or very small magnitudes do not overflow/underflow to inf/0.
    let incx = incx as isize;
    let mut scale = 0.0f32;
    let mut ssq = 1.0f32;
    let mut ix = 0isize;
    for _ in 0..n {
        let abs_xi = (*x.offset(ix)).abs();
        // `!= 0.0` (not `> 0.0`): a NaN element must still enter this branch
        // and poison the accumulator, matching `nrm2_fold` in
        // `level1/nrm2.rs` (`>` is always false for NaN).
        if abs_xi != 0.0 {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = (ssq * t).mul_add(t, 1.0);
                scale = abs_xi;
            } else {
                let t = if scale == abs_xi { 1.0 } else { abs_xi / scale };
                ssq += t * t;
            }
        }
        ix += incx;
    }
    scale * ssq.sqrt()
}

/// Double precision sum of absolute values.
///
/// Uses SIMD-optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64 {
    // Reference DASUM is a no-op returning 0 for incx <= 0. A null `x` would
    // be dereferenced unconditionally by the loop below.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::asum(x_slice);
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut result = 0.0;
    let mut ix = 0isize;
    for _ in 0..n {
        result += (*x.offset(ix)).abs();
        ix += incx;
    }
    result
}

/// Single precision sum of absolute values.
///
/// Uses SIMD-optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sasum(n: i32, x: *const f32, incx: i32) -> f32 {
    // Reference SASUM is a no-op returning 0 for incx <= 0. A null `x` would
    // be dereferenced unconditionally by the loop below.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::asum(x_slice);
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut result = 0.0f32;
    let mut ix = 0isize;
    for _ in 0..n {
        result += (*x.offset(ix)).abs();
        ix += incx;
    }
    result
}

/// Double precision index of maximum absolute value.
///
/// Uses optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32 {
    // Reference IDAMAX is a no-op returning 0 for incx <= 0. The fallback
    // below dereferences `*x` unconditionally to seed the running maximum,
    // so a null `x` must be rejected here rather than inside the loop.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::iamax(x_slice) as i32;
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut max_idx = 0;
    let mut max_val = (*x).abs();
    let mut ix = incx;
    for i in 1..n {
        let val = (*x.offset(ix)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
        ix += incx;
    }
    max_idx as i32
}

/// Single precision index of maximum absolute value.
///
/// Uses optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_isamax(n: i32, x: *const f32, incx: i32) -> i32 {
    // Reference ISAMAX is a no-op returning 0 for incx <= 0. The fallback
    // below dereferences `*x` unconditionally to seed the running maximum,
    // so a null `x` must be rejected here rather than inside the loop.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::iamax(x_slice) as i32;
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut max_idx = 0;
    let mut max_val = (*x).abs();
    let mut ix = incx;
    for i in 1..n {
        let val = (*x.offset(ix)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
        ix += incx;
    }
    max_idx as i32
}

/// Double precision vector scaling: x = alpha * x.
///
/// Uses SIMD-optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32) {
    // Reference DSCAL is a no-op for incx <= 0. A null `x` would be
    // dereferenced unconditionally by the loop below.
    if n <= 0 || incx <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        level1::scal(alpha, x_slice);
        return;
    }

    // Fallback: positive non-unit stride (incx <= 0 rejected above).
    let incx = incx as isize;
    let mut ix = 0isize;
    for _ in 0..n {
        *x.offset(ix) *= alpha;
        ix += incx;
    }
}

/// Single precision vector scaling: x = alpha * x.
///
/// Uses SIMD-optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32) {
    // Reference SSCAL is a no-op for incx <= 0. A null `x` would be
    // dereferenced unconditionally by the loop below.
    if n <= 0 || incx <= 0 || x.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        level1::scal(alpha, x_slice);
        return;
    }

    // Fallback: positive non-unit stride (incx <= 0 rejected above).
    let incx = incx as isize;
    let mut ix = 0isize;
    for _ in 0..n {
        *x.offset(ix) *= alpha;
        ix += incx;
    }
}

/// Double precision axpy: y = alpha * x + y.
///
/// Uses SIMD-optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_daxpy(
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
) {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. A zero increment aliases every logical element onto element 0
    // (meaningless for a mutating routine), and a null operand would be
    // dereferenced unconditionally by the loops below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::axpy(alpha, x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) += alpha * *x.offset(ix);
        ix += incx;
        iy += incy;
    }
}

/// Single precision axpy: y = alpha * x + y.
///
/// Uses SIMD-optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_saxpy(
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
) {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. A zero increment aliases every logical element onto element 0
    // (meaningless for a mutating routine), and a null operand would be
    // dereferenced unconditionally by the loops below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::axpy(alpha, x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) += alpha * *x.offset(ix);
        ix += incx;
        iy += incy;
    }
}

/// Double precision vector copy: y = x.
///
/// Uses optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32) {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. A zero increment aliases every logical element onto element 0
    // (meaningless for a mutating routine), and a null operand would be
    // dereferenced unconditionally by the loops below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::copy(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) = *x.offset(ix);
        ix += incx;
        iy += incy;
    }
}

/// Single precision vector copy: y = x.
///
/// Uses optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_scopy(n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. A zero increment aliases every logical element onto element 0
    // (meaningless for a mutating routine), and a null operand would be
    // dereferenced unconditionally by the loops below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::copy(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) = *x.offset(ix);
        ix += incx;
        iy += incy;
    }
}

/// Double precision vector swap.
///
/// Uses optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dswap(n: i32, x: *mut f64, incx: i32, y: *mut f64, incy: i32) {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. A zero increment aliases every logical element onto element 0
    // (meaningless for a mutating routine), and a null operand would be
    // dereferenced unconditionally by the loops below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::swap(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        std::ptr::swap(x.offset(ix), y.offset(iy));
        ix += incx;
        iy += incy;
    }
}

/// Single precision vector swap.
///
/// Uses optimized implementation for unit stride.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sswap(n: i32, x: *mut f32, incx: i32, y: *mut f32, incy: i32) {
    // Reference BLAS calls xerbla and returns on invalid arguments; a C ABI
    // cannot surface an error code and unwinding across it would be UB, so we
    // no-op. A zero increment aliases every logical element onto element 0
    // (meaningless for a mutating routine), and a null operand would be
    // dereferenced unconditionally by the loops below.
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::swap(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        std::ptr::swap(x.offset(ix), y.offset(iy));
        ix += incx;
        iy += incy;
    }
}
