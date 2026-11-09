//! CBLAS Level 1 (vector-vector) real operations.

use super::vector_start_offset;
use crate::level1;

/// Double precision dot product.
///
/// Computes: x · y
///
/// Unit stride vectors go through `level1::dot`, which runs a NEON kernel on
/// aarch64 (unless the `force-scalar` feature is enabled); on other targets it
/// is a portable four-accumulator loop. Other increments use the reference
/// strided loop.
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
    // `n <= 0` yields the DDOT identity (0), as in the reference. A null
    // operand would be dereferenced unconditionally by the loop below and a C
    // ABI cannot surface an error (a panic reaching it would abort the
    // process), so it yields 0 too. A zero increment is valid (Netlib 3.12):
    // the reference loop simply re-reads element 0 on every iteration.
    if n <= 0 || x.is_null() || y.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation
    if incx == 1 && incy == 1 {
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return 0.0;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above) and each
        // holds `n` readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts(y, n);
        return level1::dot(x_slice, y_slice);
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
/// Unit stride vectors go through `level1::dot`, which runs a NEON kernel on
/// aarch64 (unless the `force-scalar` feature is enabled); on other targets it
/// is a portable four-accumulator loop. Other increments use the reference
/// strided loop.
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
    // `n <= 0` yields the SDOT identity (0), as in the reference. A null
    // operand would be dereferenced unconditionally by the loop below and a C
    // ABI cannot surface an error (a panic reaching it would abort the
    // process), so it yields 0 too. A zero increment is valid (Netlib 3.12):
    // the reference loop simply re-reads element 0 on every iteration.
    if n <= 0 || x.is_null() || y.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation
    if incx == 1 && incy == 1 {
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return 0.0;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above) and each
        // holds `n` readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts(y, n);
        return level1::dot(x_slice, y_slice);
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
/// Never overflows or underflows for very large or very small elements. Unit
/// stride goes through `level1::nrm2`: on aarch64 (unless the `force-scalar`
/// feature is enabled) vectors of 8 or more elements run the one-pass LAPACK
/// 3.12 `dnrm2.f90` algorithm (Blue's three scaled accumulators) as a NEON
/// kernel; shorter vectors, other targets and other increments (inline here)
/// use a dlassq-style scaled sum of squares (a single `(scale, ssq)`
/// accumulator).
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
    // Netlib >= 3.10 DNRM2 (`dnrm2.f90`) accepts every increment: a
    // negative `incx` walks the vector backwards from `(1 - n) * incx`, and
    // `incx == 0` reads x[0] on every iteration (sqrt(n) * |x[0]|). Only
    // `n <= 0` returns 0 (the pre-3.10 rule also returned 0 for incx <= 0).
    // A null `x` would be dereferenced unconditionally by the loop below.
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - `level1::nrm2` (Blue's algorithm on aarch64 for
    // n >= 8, the dlassq-style scaled sum of squares otherwise)
    if incx == 1 {
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return 0.0;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: strided (any sign of incx, including zero, starting at the
    // reference offset). Use the dlassq-style scaled sum-of-squares (mirroring
    // the portable `level1::nrm2` loop) instead of a naive Σx² so that
    // vectors with very large or very small magnitudes do not overflow/underflow
    // to inf/0.
    let incx = incx as isize;
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    let mut ix = vector_start_offset(n, incx);
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
/// Never overflows or underflows for very large or very small elements. Unit
/// stride goes through `level1::nrm2`: on aarch64 (unless the `force-scalar`
/// feature is enabled) vectors of 8 or more elements run the one-pass LAPACK
/// 3.12 `snrm2.f90` algorithm (Blue's three scaled accumulators) as a NEON
/// kernel; shorter vectors, other targets and other increments (inline here)
/// use a dlassq-style scaled sum of squares (a single `(scale, ssq)`
/// accumulator).
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
    // Netlib >= 3.10 SNRM2 (`snrm2.f90`) accepts every increment: a
    // negative `incx` walks the vector backwards from `(1 - n) * incx`, and
    // `incx == 0` reads x[0] on every iteration (sqrt(n) * |x[0]|). Only
    // `n <= 0` returns 0 (the pre-3.10 rule also returned 0 for incx <= 0).
    // A null `x` would be dereferenced unconditionally by the loop below.
    if n <= 0 || x.is_null() {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - `level1::nrm2` (Blue's algorithm on aarch64 for
    // n >= 8, the dlassq-style scaled sum of squares otherwise)
    if incx == 1 {
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return 0.0;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: strided (any sign of incx, including zero, starting at the
    // reference offset). Use the dlassq-style scaled sum-of-squares (mirroring
    // the portable `level1::nrm2` loop) instead of a naive Σx² so that
    // vectors with very large or very small magnitudes do not overflow/underflow
    // to inf/0.
    let incx = incx as isize;
    let mut scale = 0.0f32;
    let mut ssq = 1.0f32;
    let mut ix = vector_start_offset(n, incx);
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
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return 0.0;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
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
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return 0.0;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
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
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return 0;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
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
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: return the identity (0) instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return 0;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable elements (caller contract), and `n` elements fit in
        // `isize::MAX` bytes (the `fits_isize_bytes` guard above).
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
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts_mut`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable and writable elements (caller contract), and `n` elements fit
        // in `isize::MAX` bytes (the `fits_isize_bytes` guard above).
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
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts_mut`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return;
        }
        // SAFETY: unit stride; `x` is non-null (checked above) and holds `n`
        // readable and writable elements (caller contract), and `n` elements fit
        // in `isize::MAX` bytes (the `fits_isize_bytes` guard above).
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
    // Netlib ?AXPY: `IF (N.LE.0) RETURN` then `IF (DA.EQ.0) RETURN`, so `y` is
    // never touched for alpha == 0 on any stride (NaN/Inf in `x` cannot leak
    // in and signed zeros in `y` survive). A null operand would be
    // dereferenced unconditionally by the loops below; a C ABI cannot
    // surface an error (a panic reaching it would abort the process), so it
    // no-ops. A zero increment is valid (Netlib 3.12): the reference loop
    // re-uses x[0] (incx == 0) or accumulates every term into y[0] (incy == 0).
    if n <= 0 || alpha == 0.0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above), `x`
        // holds `n` readable and `y` `n` readable and writable elements that do
        // not overlap (caller contract), and `n` elements fit in `isize::MAX`
        // bytes (the `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::axpy(alpha, x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
    // Netlib ?AXPY: `IF (N.LE.0) RETURN` then `IF (DA.EQ.0) RETURN`, so `y` is
    // never touched for alpha == 0 on any stride (NaN/Inf in `x` cannot leak
    // in and signed zeros in `y` survive). A null operand would be
    // dereferenced unconditionally by the loops below; a C ABI cannot
    // surface an error (a panic reaching it would abort the process), so it
    // no-ops. A zero increment is valid (Netlib 3.12): the reference loop
    // re-uses x[0] (incx == 0) or accumulates every term into y[0] (incy == 0).
    if n <= 0 || alpha == 0.0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above), `x`
        // holds `n` readable and `y` `n` readable and writable elements that do
        // not overlap (caller contract), and `n` elements fit in `isize::MAX`
        // bytes (the `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::axpy(alpha, x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (a panic reaching it would abort the process), so it
    // no-ops too. A zero increment is valid (Netlib 3.12): the reference loop
    // re-uses element 0 on every iteration (e.g. LAPACK's
    // `dcopy(n, zero, 0, x, 1)` broadcast).
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above), `x`
        // holds `n` readable and `y` `n` writable elements that do not overlap
        // (caller contract), and `n` elements fit in `isize::MAX` bytes (the
        // `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::copy(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (a panic reaching it would abort the process), so it
    // no-ops too. A zero increment is valid (Netlib 3.12): the reference loop
    // re-uses element 0 on every iteration (e.g. LAPACK's
    // `dcopy(n, zero, 0, x, 1)` broadcast).
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above), `x`
        // holds `n` readable and `y` `n` writable elements that do not overlap
        // (caller contract), and `n` elements fit in `isize::MAX` bytes (the
        // `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::copy(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (a panic reaching it would abort the process), so it
    // no-ops too. A zero increment is valid (Netlib 3.12): the reference loop
    // swaps against element 0 on every iteration, and the final values follow
    // the reference loop order.
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        // `n` f64s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts_mut`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f64>(n) {
            return;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above) and each
        // holds `n` readable and writable elements, without overlap (caller
        // contract), and `n` elements fit in `isize::MAX` bytes (the
        // `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::swap(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (a panic reaching it would abort the process), so it
    // no-ops too. A zero increment is valid (Netlib 3.12): the reference loop
    // swaps against element 0 on every iteration, and the final values follow
    // the reference loop order.
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        // `n` f32s spanning more than `isize::MAX` bytes (possible on 32-bit
        // targets only) cannot be a real buffer, and `slice::from_raw_parts_mut`
        // forbids such a length: no-op instead.
        if !level1::fits_isize_bytes::<f32>(n) {
            return;
        }
        // SAFETY: unit strides; `x` and `y` are non-null (checked above) and each
        // holds `n` readable and writable elements, without overlap (caller
        // contract), and `n` elements fit in `isize::MAX` bytes (the
        // `fits_isize_bytes` guard above).
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::swap(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride. incx/incy may be negative (back-to-front walk
    // from `vector_start_offset`) or zero (element 0 re-used every iteration);
    // both are spec-valid and follow the reference loop exactly.
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
