//! CBLAS-compatible complex Level-1 routines.
//!
//! Rounds out the complex vector operations: scaling (`cscal`/`zscal`,
//! `csscal`/`zdscal`), `caxpy`/`zaxpy`, `ccopy`/`zcopy`, `cswap`/`zswap`,
//! the real-valued reductions `scnrm2`/`dznrm2`, `scasum`/`dzasum`, and the
//! index reductions `icamax`/`izamax`.
//!
//! Unit-stride `copy`/`swap` defer to the native kernels (`level1::copy`/
//! `swap`) and the norms forward to the native `level1::cnrm2_strided`;
//! `scal`/`axpy` (generic over the component type) and every other strided
//! path use scalar loops.
//!
//! Increments follow Netlib BLAS 3.12 exactly: AXPY/COPY/SWAP and NRM2 accept
//! negative increments (walked back-to-front, see `vec_offset`) and zero
//! increments (element 0 re-used on every iteration), while SCAL/ASUM/IAMAX
//! treat every `incx <= 0` as a quick return.

use crate::level1;
use num_complex::{Complex, Complex32, Complex64};
use num_traits::Float;

/// Pointer offset (in elements) of logical index `i` for a length-`n` vector
/// with increment `inc`, honoring the reference BLAS negative-increment rule.
#[inline]
fn vec_offset(i: usize, n: usize, inc: isize) -> isize {
    if inc >= 0 {
        i as isize * inc
    } else {
        (i as isize - (n as isize - 1)) * inc
    }
}

// =============================================================================
// SCAL — x = alpha * x  (complex alpha)
// =============================================================================

/// Complex single precision scaling by a complex scalar.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cscal(
    n: i32,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: i32,
) {
    // Netlib CSCAL/ZSCAL (3.12): `IF (N.LE.0 .OR. INCX.LE.0 .OR. ZA.EQ.ONE)
    // RETURN`. A non-positive increment is a quick return (x is left
    // untouched, not scaled back-to-front), and so is alpha == 1: a literal
    // complex multiply by 1 + 0i is not an identity for non-finite input
    // ((inf + 0i)(1 + 0i) = inf + NaN i). A null operand would be
    // dereferenced unconditionally by the loop below; a C ABI cannot surface
    // an error (unwinding across it would be UB), so it no-ops.
    if n <= 0 || incx <= 0 || alpha.is_null() || x.is_null() {
        return;
    }
    let alpha = *alpha;
    if alpha == Complex32::new(1.0, 0.0) {
        return;
    }
    scal_c(n as usize, alpha, x, incx as isize);
}

/// Complex double precision scaling by a complex scalar.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zscal(
    n: i32,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: i32,
) {
    // Netlib CSCAL/ZSCAL (3.12): `IF (N.LE.0 .OR. INCX.LE.0 .OR. ZA.EQ.ONE)
    // RETURN`. A non-positive increment is a quick return (x is left
    // untouched, not scaled back-to-front), and so is alpha == 1: a literal
    // complex multiply by 1 + 0i is not an identity for non-finite input
    // ((inf + 0i)(1 + 0i) = inf + NaN i). A null operand would be
    // dereferenced unconditionally by the loop below; a C ABI cannot surface
    // an error (unwinding across it would be UB), so it no-ops.
    if n <= 0 || incx <= 0 || alpha.is_null() || x.is_null() {
        return;
    }
    let alpha = *alpha;
    if alpha == Complex64::new(1.0, 0.0) {
        return;
    }
    scal_c(n as usize, alpha, x, incx as isize);
}

unsafe fn scal_c<F: Float>(n: usize, alpha: Complex<F>, x: *mut Complex<F>, inc: isize) {
    // Generic over `F`, so we cannot name the `Complex<F>: Field` bound the SIMD
    // kernel needs; a scalar loop is used for both unit and strided access.
    // Callers guarantee `inc > 0` (Netlib's INCX.LE.0 quick return).
    for i in 0..n {
        let p = x.offset(vec_offset(i, n, inc));
        *p = *p * alpha;
    }
}

// =============================================================================
// (D)SCAL — x = alpha * x  (real alpha, complex x): csscal / zdscal
// =============================================================================

/// Complex single precision scaling by a real scalar.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_csscal(n: i32, alpha: f32, x: *mut Complex32, incx: i32) {
    // Netlib CSSCAL/ZDSCAL (3.12): `IF (N.LE.0 .OR. INCX.LE.0 .OR. DA.EQ.ONE)
    // RETURN` -- a non-positive increment leaves x untouched (it is not
    // scaled back-to-front). A null `x` would be dereferenced unconditionally
    // by the loop below; a C ABI cannot surface an error, so it no-ops.
    if n <= 0 || incx <= 0 || alpha == 1.0 || x.is_null() {
        return;
    }
    dscal_c(n as usize, alpha, x, incx as isize);
}

/// Complex double precision scaling by a real scalar.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zdscal(n: i32, alpha: f64, x: *mut Complex64, incx: i32) {
    // Netlib CSSCAL/ZDSCAL (3.12): `IF (N.LE.0 .OR. INCX.LE.0 .OR. DA.EQ.ONE)
    // RETURN` -- a non-positive increment leaves x untouched (it is not
    // scaled back-to-front). A null `x` would be dereferenced unconditionally
    // by the loop below; a C ABI cannot surface an error, so it no-ops.
    if n <= 0 || incx <= 0 || alpha == 1.0 || x.is_null() {
        return;
    }
    dscal_c(n as usize, alpha, x, incx as isize);
}

unsafe fn dscal_c<F: Float>(n: usize, alpha: F, x: *mut Complex<F>, inc: isize) {
    // Component-wise `(alpha * re, alpha * im)`, exactly like the reference
    // `CMPLX(SA*REAL(X), SA*AIMAG(X))`. Callers guarantee `inc > 0`.
    for i in 0..n {
        let p = x.offset(vec_offset(i, n, inc));
        *p = (*p).scale(alpha);
    }
}

// =============================================================================
// AXPY — y = alpha * x + y  (complex alpha)
// =============================================================================

/// Complex single precision AXPY.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_caxpy(
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    // Netlib CAXPY/ZAXPY: `IF (N.LE.0) RETURN` then `IF (DCABS1(ZA).EQ.0)
    // RETURN` -- y is never touched for alpha == 0 on any stride (NaN/Inf in
    // x cannot leak in and signed zeros in y survive). A null operand would
    // be dereferenced unconditionally by the loop below; a C ABI cannot
    // surface an error (unwinding across it would be UB), so it no-ops. A
    // zero increment is valid (Netlib 3.12): the reference loop re-uses x[0]
    // (incx == 0) or accumulates every term into y[0] (incy == 0).
    if n <= 0 || alpha.is_null() || x.is_null() || y.is_null() {
        return;
    }
    let alpha = *alpha;
    if level1::scabs1(alpha) == 0.0 {
        return;
    }
    axpy_c(n as usize, alpha, x, incx as isize, y, incy as isize);
}

/// Complex double precision AXPY.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `alpha` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zaxpy(
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    // Netlib CAXPY/ZAXPY: `IF (N.LE.0) RETURN` then `IF (DCABS1(ZA).EQ.0)
    // RETURN` -- y is never touched for alpha == 0 on any stride (NaN/Inf in
    // x cannot leak in and signed zeros in y survive). A null operand would
    // be dereferenced unconditionally by the loop below; a C ABI cannot
    // surface an error (unwinding across it would be UB), so it no-ops. A
    // zero increment is valid (Netlib 3.12): the reference loop re-uses x[0]
    // (incx == 0) or accumulates every term into y[0] (incy == 0).
    if n <= 0 || alpha.is_null() || x.is_null() || y.is_null() {
        return;
    }
    let alpha = *alpha;
    if level1::dcabs1(alpha) == 0.0 {
        return;
    }
    axpy_c(n as usize, alpha, x, incx as isize, y, incy as isize);
}

unsafe fn axpy_c<F: Float>(
    n: usize,
    alpha: Complex<F>,
    x: *const Complex<F>,
    incx: isize,
    y: *mut Complex<F>,
    incy: isize,
) {
    // Scalar loop for both unit and strided access (see `scal_c`).
    for i in 0..n {
        let xi = *x.offset(vec_offset(i, n, incx));
        let yp = y.offset(vec_offset(i, n, incy));
        *yp = *yp + alpha * xi;
    }
}

// =============================================================================
// COPY — y = x
// =============================================================================

/// Complex single precision copy.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ccopy(
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (unwinding across it would be UB), so it no-ops too.
    // A zero increment is valid (Netlib 3.12): the reference loop re-uses
    // element 0 on every iteration (e.g. LAPACK's `zcopy(n, zero, 0, x, 1)`
    // broadcast).
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    copy_c(n as usize, x, incx as isize, y, incy as isize);
}

/// Complex double precision copy.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zcopy(
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (unwinding across it would be UB), so it no-ops too.
    // A zero increment is valid (Netlib 3.12): the reference loop re-uses
    // element 0 on every iteration (e.g. LAPACK's `zcopy(n, zero, 0, x, 1)`
    // broadcast).
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    copy_c(n as usize, x, incx as isize, y, incy as isize);
}

unsafe fn copy_c<T: Copy>(n: usize, x: *const T, incx: isize, y: *mut T, incy: isize) {
    if incx == 1 && incy == 1 {
        let xs = std::slice::from_raw_parts(x, n);
        let ys = std::slice::from_raw_parts_mut(y, n);
        level1::copy(xs, ys);
        return;
    }
    for i in 0..n {
        *y.offset(vec_offset(i, n, incy)) = *x.offset(vec_offset(i, n, incx));
    }
}

// =============================================================================
// SWAP — x <-> y
// =============================================================================

/// Complex single precision swap.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cswap(
    n: i32,
    x: *mut Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (unwinding across it would be UB), so it no-ops too.
    // A zero increment is valid (Netlib 3.12): with inc = 0 the reference
    // loop swaps against element 0 on every iteration, and the final values
    // follow the reference loop order.
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    swap_c(n as usize, x, incx as isize, y, incy as isize);
}

/// Complex double precision swap.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zswap(
    n: i32,
    x: *mut Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    // `n <= 0` is a no-op, as in the reference. A null operand would be
    // dereferenced unconditionally by the loops below and a C ABI cannot
    // surface an error (unwinding across it would be UB), so it no-ops too.
    // A zero increment is valid (Netlib 3.12): with inc = 0 the reference
    // loop swaps against element 0 on every iteration, and the final values
    // follow the reference loop order.
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    swap_c(n as usize, x, incx as isize, y, incy as isize);
}

unsafe fn swap_c<T: Copy>(n: usize, x: *mut T, incx: isize, y: *mut T, incy: isize) {
    if incx == 1 && incy == 1 {
        let xs = std::slice::from_raw_parts_mut(x, n);
        let ys = std::slice::from_raw_parts_mut(y, n);
        level1::swap(xs, ys);
        return;
    }
    for i in 0..n {
        let xp = x.offset(vec_offset(i, n, incx));
        let yp = y.offset(vec_offset(i, n, incy));
        std::ptr::swap(xp, yp);
    }
}

// =============================================================================
// NRM2 — Euclidean norm of a complex vector (real result)
// =============================================================================

/// Complex single precision Euclidean norm (`scnrm2`).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_scnrm2(n: i32, x: *const Complex32, incx: i32) -> f32 {
    // Netlib >= 3.10 SCNRM2 (`scnrm2.f90`) accepts every increment: a
    // negative `incx` walks the vector backwards from `(1 - n) * incx` and
    // `incx == 0` reads x[0] on every iteration (sqrt(n) * |x[0]|); only
    // `n <= 0` returns 0. A null `x` would be dereferenced by the kernel.
    if n <= 0 || x.is_null() {
        return 0.0;
    }
    // The caller guarantees (see `# Safety`) that `x` -- the lowest address
    // of the vector for either sign of `incx` -- is valid for the whole
    // strided walk, i.e. `(n - 1) * |incx| + 1` elements, exactly the extent
    // `from_raw_parts` views. An extent no buffer can have (one that overflows
    // `usize`, possible on 32-bit targets, or spans more than `isize::MAX`
    // bytes, possible on 64-bit targets too) yields an empty view, whose norm
    // is 0, without touching `x`.
    level1::cnrm2_strided(level1::StridedSlice::from_raw_parts(
        x,
        n as usize,
        incx as isize,
    ))
}

/// Complex double precision Euclidean norm (`dznrm2`).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dznrm2(n: i32, x: *const Complex64, incx: i32) -> f64 {
    // Netlib >= 3.10 DZNRM2 (`dznrm2.f90`) accepts every increment: a
    // negative `incx` walks the vector backwards from `(1 - n) * incx` and
    // `incx == 0` reads x[0] on every iteration (sqrt(n) * |x[0]|); only
    // `n <= 0` returns 0. A null `x` would be dereferenced by the kernel.
    if n <= 0 || x.is_null() {
        return 0.0;
    }
    // The caller guarantees (see `# Safety`) that `x` -- the lowest address
    // of the vector for either sign of `incx` -- is valid for the whole
    // strided walk, i.e. `(n - 1) * |incx| + 1` elements, exactly the extent
    // `from_raw_parts` views. An extent no buffer can have (one that overflows
    // `usize`, possible on 32-bit targets, or spans more than `isize::MAX`
    // bytes, possible on 64-bit targets too) yields an empty view, whose norm
    // is 0, without touching `x`.
    level1::cnrm2_strided(level1::StridedSlice::from_raw_parts(
        x,
        n as usize,
        incx as isize,
    ))
}

// =============================================================================
// ASUM — sum of |re| + |im| over the vector (real result)
// =============================================================================

/// Complex single precision `scasum`.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_scasum(n: i32, x: *const Complex32, incx: i32) -> f32 {
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0.0;
    }
    asum_c(n as usize, x, incx as isize)
}

/// Complex double precision `dzasum`.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dzasum(n: i32, x: *const Complex64, incx: i32) -> f64 {
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0.0;
    }
    asum_c(n as usize, x, incx as isize)
}

unsafe fn asum_c<F: Float>(n: usize, x: *const Complex<F>, inc: isize) -> F {
    let mut acc = F::zero();
    for i in 0..n {
        let z = *x.offset(vec_offset(i, n, inc));
        acc = acc + z.re.abs() + z.im.abs();
    }
    acc
}

// =============================================================================
// IAMAX — index of element with largest |re| + |im| (0-based)
// =============================================================================

/// Complex single precision `icamax` (0-based index).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_icamax(n: i32, x: *const Complex32, incx: i32) -> i32 {
    // `iamax_c` dereferences `*x` unconditionally to seed the running maximum,
    // so a null `x` must be rejected here rather than inside the loop.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0;
    }
    iamax_c(n as usize, x, incx as isize) as i32
}

/// Complex double precision `izamax` (0-based index).
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `Complex64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_izamax(n: i32, x: *const Complex64, incx: i32) -> i32 {
    // `iamax_c` dereferences `*x` unconditionally to seed the running maximum,
    // so a null `x` must be rejected here rather than inside the loop.
    if n <= 0 || incx <= 0 || x.is_null() {
        return 0;
    }
    iamax_c(n as usize, x, incx as isize) as i32
}

unsafe fn iamax_c<F: Float>(n: usize, x: *const Complex<F>, inc: isize) -> usize {
    let mut max_idx = 0usize;
    let z0 = *x;
    let mut max_val = z0.re.abs() + z0.im.abs();
    for i in 1..n {
        let z = *x.offset(vec_offset(i, n, inc));
        let v = z.re.abs() + z.im.abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cblas_zscal() {
        let alpha = Complex64::new(2.0, 1.0);
        let mut x = [Complex64::new(1.0, 1.0), Complex64::new(0.0, -2.0)];
        unsafe {
            cblas_zscal(2, &alpha, x.as_mut_ptr(), 1);
        }
        // (2+i)(1+i) = 2 + 2i + i - 1 = 1 + 3i
        // (2+i)(0-2i) = -4i - 2i^2 = 2 - 4i
        assert!((x[0].re - 1.0).abs() < 1e-12 && (x[0].im - 3.0).abs() < 1e-12);
        assert!((x[1].re - 2.0).abs() < 1e-12 && (x[1].im + 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zdscal() {
        let mut x = [Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)];
        unsafe {
            cblas_zdscal(2, 0.5, x.as_mut_ptr(), 1);
        }
        assert!((x[0].re - 0.5).abs() < 1e-12 && (x[0].im + 1.0).abs() < 1e-12);
        assert!((x[1].re - 1.5).abs() < 1e-12 && (x[1].im - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zaxpy_strided() {
        let alpha = Complex64::new(1.0, 0.0);
        let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)];
        // y strided incy=2
        let mut y = [
            Complex64::new(0.0, 0.0),
            Complex64::new(9.0, 9.0),
            Complex64::new(0.0, 0.0),
        ];
        unsafe {
            cblas_zaxpy(2, &alpha, x.as_ptr(), 1, y.as_mut_ptr(), 2);
        }
        assert!((y[0].re - 1.0).abs() < 1e-12 && (y[0].im - 1.0).abs() < 1e-12);
        assert!((y[2].re - 2.0).abs() < 1e-12 && (y[2].im - 2.0).abs() < 1e-12);
        // untouched
        assert!((y[1].re - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zaxpy_matches_native_unit() {
        let alpha = Complex64::new(1.5, -0.5);
        let x = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 3.0),
        ];
        let mut y = [
            Complex64::new(0.5, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(1.0, -1.0),
        ];
        let mut y_ref = y;
        unsafe {
            cblas_zaxpy(3, &alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        level1::axpy(alpha, &x, &mut y_ref);
        for (g, r) in y.iter().zip(y_ref.iter()) {
            assert!((g.re - r.re).abs() < 1e-12 && (g.im - r.im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_zcopy_zswap() {
        let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let mut y = [Complex64::new(0.0, 0.0); 2];
        unsafe {
            cblas_zcopy(2, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert_eq!(y, x);
        let mut a = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let mut b = [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)];
        unsafe {
            cblas_zswap(2, a.as_mut_ptr(), 1, b.as_mut_ptr(), 1);
        }
        assert_eq!(a, [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]);
        assert_eq!(b, [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
    }

    #[test]
    fn test_cblas_dznrm2() {
        // z = [3+4i] -> |z| = 5 ; [3+4i, 0] -> 5
        let x = [Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)];
        let r = unsafe { cblas_dznrm2(2, x.as_ptr(), 1) };
        assert!((r - 5.0).abs() < 1e-12);
        // [1+1i, 1+1i] -> sqrt(1+1+1+1) = 2
        let y = [Complex64::new(1.0, 1.0), Complex64::new(1.0, 1.0)];
        let r2 = unsafe { cblas_dznrm2(2, y.as_ptr(), 1) };
        assert!((r2 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_dzasum() {
        let x = [Complex64::new(1.0, -2.0), Complex64::new(-3.0, 4.0)];
        let r = unsafe { cblas_dzasum(2, x.as_ptr(), 1) };
        // |1|+|−2| + |−3|+|4| = 1+2+3+4 = 10
        assert!((r - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_izamax() {
        let x = [
            Complex64::new(1.0, 1.0),  // 2
            Complex64::new(3.0, -3.0), // 6
            Complex64::new(0.0, 5.0),  // 5
        ];
        let idx = unsafe { cblas_izamax(3, x.as_ptr(), 1) };
        assert_eq!(idx, 1);
    }

    // =========================================================================
    // Issue #10: Netlib 3.12 increment semantics of the complex wrappers
    // =========================================================================

    /// Bit pattern of every component, so `-0.0`/NaN payloads are compared
    /// exactly rather than through `==`.
    fn bits64(z: &[Complex64]) -> Vec<(u64, u64)> {
        z.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
    }

    fn bits32(z: &[Complex32]) -> Vec<(u32, u32)> {
        z.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
    }

    #[test]
    fn test_issue_10_complex_scal_nonpositive_incx_is_noop() {
        // Netlib CSCAL/ZSCAL/CSSCAL/ZDSCAL: IF (N.LE.0 .OR. INCX.LE.0) RETURN.
        // The wrappers used to scale the vector back-to-front for incx < 0.
        let x32 = [Complex32::new(1.0, 2.0), Complex32::new(-3.0, 4.0)];
        let x64 = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        let a32 = Complex32::new(2.0, -1.0);
        let a64 = Complex64::new(2.0, -1.0);
        for incx in [-1, 0] {
            let mut c = x32;
            // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
            unsafe { cblas_cscal(2, &a32, c.as_mut_ptr(), incx) };
            assert_eq!(c, x32, "cscal incx={incx}");
            let mut z = x64;
            // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
            unsafe { cblas_zscal(2, &a64, z.as_mut_ptr(), incx) };
            assert_eq!(z, x64, "zscal incx={incx}");
            let mut cs = x32;
            // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
            unsafe { cblas_csscal(2, 3.0, cs.as_mut_ptr(), incx) };
            assert_eq!(cs, x32, "csscal incx={incx}");
            let mut zd = x64;
            // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
            unsafe { cblas_zdscal(2, 3.0, zd.as_mut_ptr(), incx) };
            assert_eq!(zd, x64, "zdscal incx={incx}");
        }
        // Positive strides still scale (only the touched elements).
        let mut z = [x64[0], Complex64::new(7.0, 7.0), x64[1]];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_zdscal(2, 2.0, z.as_mut_ptr(), 2) };
        assert_eq!(
            z,
            [
                Complex64::new(2.0, 4.0),
                Complex64::new(7.0, 7.0),
                Complex64::new(-6.0, 8.0)
            ]
        );
    }

    #[test]
    fn test_issue_10_complex_scal_alpha_one_is_quick_return() {
        // The same Netlib 3.12 IF line also returns for alpha == 1. A literal
        // complex multiply by (1, 0) is not an identity for non-finite input:
        // (inf + 0i)(1 + 0i) = inf + NaN i.
        let x64 = [
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(1.0, f64::NAN),
            Complex64::new(-0.0, f64::NEG_INFINITY),
        ];
        let mut z = x64;
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_zscal(3, &Complex64::new(1.0, 0.0), z.as_mut_ptr(), 1) };
        assert_eq!(bits64(&z), bits64(&x64));
        let mut z = x64;
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_zdscal(3, 1.0, z.as_mut_ptr(), 1) };
        assert_eq!(bits64(&z), bits64(&x64));

        let x32 = [
            Complex32::new(f32::INFINITY, 0.0),
            Complex32::new(1.0, f32::NAN),
        ];
        let mut c = x32;
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_cscal(2, &Complex32::new(1.0, 0.0), c.as_mut_ptr(), 1) };
        assert_eq!(bits32(&c), bits32(&x32));
        let mut c = x32;
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_csscal(2, 1.0, c.as_mut_ptr(), 1) };
        assert_eq!(bits32(&c), bits32(&x32));
    }

    #[test]
    fn test_issue_10_complex_axpy_zero_increments_follow_reference_loop() {
        let alpha = Complex64::new(1.0, 1.0);
        let x = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -1.0),
        ];
        // incy == 0: y[0] accumulates alpha * x[i] for every i.
        let mut y = [Complex64::new(5.0, 5.0), Complex64::new(7.0, 7.0)];
        let mut expected = y;
        for xi in x {
            expected[0] = expected[0] + alpha * xi;
        }
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_zaxpy(3, &alpha, x.as_ptr(), 1, y.as_mut_ptr(), 0) };
        assert_eq!(y, expected);

        // incx == 0: every y[i] receives alpha * x[0].
        let xs = [Complex32::new(2.0, -1.0), Complex32::new(99.0, 99.0)];
        let a32 = Complex32::new(0.5, 2.0);
        let mut ys = [Complex32::new(1.0, 1.0), Complex32::new(-1.0, 3.0)];
        let expected = [ys[0] + a32 * xs[0], ys[1] + a32 * xs[0]];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_caxpy(2, &a32, xs.as_ptr(), 0, ys.as_mut_ptr(), 1) };
        assert_eq!(ys, expected);
    }

    #[test]
    fn test_issue_10_complex_axpy_alpha_zero_is_noop_for_every_stride() {
        // Netlib ZAXPY/CAXPY: IF (DCABS1(ZA).EQ.0) RETURN -- y is never
        // touched, so NaN/Inf in x and signed zeros in y survive untouched.
        let x = [
            Complex64::new(f64::NAN, f64::NAN),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(1.0, 1.0),
        ];
        let y0 = [
            Complex64::new(-0.0, -0.0),
            Complex64::new(5.0, 5.0),
            Complex64::new(f64::NEG_INFINITY, 1.0),
        ];
        let alpha = Complex64::new(0.0, -0.0);
        for &(n, incx, incy) in &[(3, 1, 1), (2, 2, 1), (3, 0, 1), (3, 1, 0), (3, -1, -1)] {
            let mut y = y0;
            // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
            unsafe { cblas_zaxpy(n, &alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy) };
            assert_eq!(bits64(&y), bits64(&y0), "zaxpy incx={incx}, incy={incy}");
        }
        let xs = [
            Complex32::new(f32::NAN, 0.0),
            Complex32::new(f32::INFINITY, 1.0),
        ];
        let ys0 = [Complex32::new(-0.0, 2.0), Complex32::new(3.0, -0.0)];
        let mut ys = ys0;
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe {
            cblas_caxpy(
                2,
                &Complex32::new(0.0, 0.0),
                xs.as_ptr(),
                1,
                ys.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(bits32(&ys), bits32(&ys0));
    }

    #[test]
    fn test_issue_10_complex_copy_zero_increments_follow_reference_loop() {
        // incx == 0 broadcasts x[0] (LAPACK's `zcopy(n, zero, 0, v, 1)`).
        let x = [Complex64::new(1.0, -2.0), Complex64::new(9.0, 9.0)];
        let mut y = [Complex64::new(0.0, 0.0); 3];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_zcopy(3, x.as_ptr(), 0, y.as_mut_ptr(), 1) };
        assert_eq!(y, [x[0]; 3]);
        // incy == 0 leaves y[0] holding the last x.
        let xs = [
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 2.0),
            Complex32::new(3.0, 3.0),
        ];
        let mut ys = [Complex32::new(9.0, 9.0); 2];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_ccopy(3, xs.as_ptr(), 1, ys.as_mut_ptr(), 0) };
        assert_eq!(ys, [xs[2], Complex32::new(9.0, 9.0)]);
    }

    #[test]
    fn test_issue_10_complex_swap_zero_increments_follow_reference_loop() {
        // Reference loop: tmp = x[ix]; x[ix] = y[iy]; y[iy] = tmp.
        let (a, b) = (Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0));
        let (p, q, r) = (
            Complex64::new(10.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(30.0, 0.0),
        );
        let mut x = [a, b];
        let mut y = [p, q, r];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_zswap(3, x.as_mut_ptr(), 0, y.as_mut_ptr(), 1) };
        assert_eq!(x, [r, b]);
        assert_eq!(y, [a, p, q]);

        let mut xs = [
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
        ];
        let mut ys = [Complex32::new(9.0, 0.0), Complex32::new(8.0, 0.0)];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        unsafe { cblas_cswap(3, xs.as_mut_ptr(), 1, ys.as_mut_ptr(), 0) };
        assert_eq!(
            xs,
            [
                Complex32::new(9.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0)
            ]
        );
        assert_eq!(ys, [Complex32::new(3.0, 0.0), Complex32::new(8.0, 0.0)]);
    }

    #[test]
    fn test_issue_10_complex_nrm2_negative_and_zero_incx() {
        // ||[3+4i, 12i]|| = sqrt(9 + 16 + 144) = 13; incx = -1 walks it
        // backwards (Netlib >= 3.10 `dznrm2.f90`), it used to return 0.
        let x = [Complex64::new(3.0, 4.0), Complex64::new(0.0, 12.0)];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        let got = unsafe { cblas_dznrm2(2, x.as_ptr(), -1) };
        assert!((got - 13.0).abs() <= 1e-14 * 13.0, "got {got}");
        // incx = 0 reads x[0] n times: sqrt(4) * |3+4i| = 10.
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        assert_eq!(unsafe { cblas_dznrm2(4, x.as_ptr(), 0) }, 10.0);

        let xs = [Complex32::new(3.0, 4.0), Complex32::new(0.0, 12.0)];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        let got = unsafe { cblas_scnrm2(2, xs.as_ptr(), -1) };
        assert!((got - 13.0).abs() <= 1e-6 * 13.0, "got {got}");
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        assert_eq!(unsafe { cblas_scnrm2(4, xs.as_ptr(), 0) }, 10.0);

        // Large magnitudes stay finite on the strided paths too.
        let big = [Complex64::new(1e300, -1e300), Complex64::new(1e300, 1e300)];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        let got = unsafe { cblas_dznrm2(2, big.as_ptr(), -1) };
        assert!((got - 2e300).abs() <= 1e-14 * 2e300, "got {got}");
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        let got = unsafe { cblas_dznrm2(3, big.as_ptr(), 0) };
        let expected = 6.0f64.sqrt() * 1e300;
        assert!((got - expected).abs() <= 1e-14 * expected, "got {got}");
    }

    #[test]
    fn test_issue_10_complex_nrm2_infinite_components_give_inf() {
        // Two equal infinite components used to form inf/inf = NaN in the
        // scaled accumulation; the norm of an infinite vector is +inf.
        let x = [Complex64::new(f64::INFINITY, f64::NEG_INFINITY)];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        let got = unsafe { cblas_dznrm2(1, x.as_ptr(), 1) };
        assert!(got.is_infinite() && got > 0.0, "got {got}");
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        let got = unsafe { cblas_dznrm2(2, x.as_ptr(), 0) };
        assert!(got.is_infinite() && got > 0.0, "got {got}");
        let xs = [
            Complex32::new(f32::INFINITY, 1.0),
            Complex32::new(2.0, f32::INFINITY),
        ];
        // SAFETY: pointers are to live local buffers covering this (n, inc) walk.
        let got = unsafe { cblas_scnrm2(2, xs.as_ptr(), -1) };
        assert!(got.is_infinite() && got > 0.0, "got {got}");
    }
}
