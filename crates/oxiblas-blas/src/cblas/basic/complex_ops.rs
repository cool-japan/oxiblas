//! CBLAS complex-valued operations: dot products and GEMM.

use super::super::types::*;
use super::super::validate::{gemm_params_valid, inc_valid};
use super::vector_start_offset;
use num_complex::{Complex32, Complex64};
/// Complex double precision dot product.
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
///   to be read from at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
/// - `dotu` must be non-null, properly aligned for `Complex64`, and valid
///   for a write of one `Complex64` value (this function overwrites it
///   unconditionally and never reads its prior contents).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zdotu_sub(
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    dotu: *mut Complex64,
) {
    // `dotu` is dereferenced for the write on every path, including the
    // n<=0 early return below, so it must be checked before anything else —
    // there is no way to signal failure back to the caller other than a
    // silent no-op. Once `dotu` is known valid, every other rejected input
    // (n<=0, a zero increment, or a null `x`/`y`) still writes the ZDOTU
    // identity, matching this function's own documented "overwrites `dotu`
    // unconditionally" contract instead of leaving it uninitialized.
    if dotu.is_null() {
        return;
    }
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        *dotu = Complex64::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex64::new(0.0, 0.0);
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
    }
    *dotu = result;
}

/// Complex double precision conjugate dot product.
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
///   to be read from at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
/// - `dotc` must be non-null, properly aligned for `Complex64`, and valid
///   for a write of one `Complex64` value (this function overwrites it
///   unconditionally and never reads its prior contents).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zdotc_sub(
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    dotc: *mut Complex64,
) {
    // `dotc` is dereferenced for the write on every path, including the
    // n<=0 early return below, so it must be checked before anything else —
    // there is no way to signal failure back to the caller other than a
    // silent no-op. Once `dotc` is known valid, every other rejected input
    // (n<=0, a zero increment, or a null `x`/`y`) still writes the ZDOTC
    // identity, matching this function's own documented "overwrites `dotc`
    // unconditionally" contract instead of leaving it uninitialized.
    if dotc.is_null() {
        return;
    }
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        *dotc = Complex64::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex64::new(0.0, 0.0);
    for _ in 0..n {
        result += (*x.offset(ix)).conj() * *y.offset(iy);
        ix += incx;
        iy += incy;
    }
    *dotc = result;
}

/// Complex single precision dot product.
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
///   to be read from at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
/// - `dotu` must be non-null, properly aligned for `Complex32`, and valid
///   for a write of one `Complex32` value (this function overwrites it
///   unconditionally and never reads its prior contents).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cdotu_sub(
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    dotu: *mut Complex32,
) {
    // `dotu` is dereferenced for the write on every path, including the
    // n<=0 early return below, so it must be checked before anything else —
    // there is no way to signal failure back to the caller other than a
    // silent no-op. Once `dotu` is known valid, every other rejected input
    // (n<=0, a zero increment, or a null `x`/`y`) still writes the CDOTU
    // identity, matching this function's own documented "overwrites `dotu`
    // unconditionally" contract instead of leaving it uninitialized.
    if dotu.is_null() {
        return;
    }
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        *dotu = Complex32::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex32::new(0.0, 0.0);
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
    }
    *dotu = result;
}

/// Complex single precision conjugate dot product.
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
///   to be read from at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
/// - `dotc` must be non-null, properly aligned for `Complex32`, and valid
///   for a write of one `Complex32` value (this function overwrites it
///   unconditionally and never reads its prior contents).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cdotc_sub(
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    dotc: *mut Complex32,
) {
    // `dotc` is dereferenced for the write on every path, including the
    // n<=0 early return below, so it must be checked before anything else —
    // there is no way to signal failure back to the caller other than a
    // silent no-op. Once `dotc` is known valid, every other rejected input
    // (n<=0, a zero increment, or a null `x`/`y`) still writes the CDOTC
    // identity, matching this function's own documented "overwrites `dotc`
    // unconditionally" contract instead of leaving it uninitialized.
    if dotc.is_null() {
        return;
    }
    if n <= 0 || !inc_valid(incx) || !inc_valid(incy) || x.is_null() || y.is_null() {
        *dotc = Complex32::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex32::new(0.0, 0.0);
    for _ in 0..n {
        result += (*x.offset(ix)).conj() * *y.offset(iy);
        ix += incx;
        iy += incy;
    }
    *dotc = result;
}

/// Complex double precision GEMM.
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
/// - `a` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `beta` must be non-null and point to one valid, properly aligned,
///   initialized `Complex64` value.
/// - `c` must be non-null, properly aligned for `Complex64`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zgemm(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *const Complex64,
    ldb: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
        || alpha.is_null()
        || beta.is_null()
    {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;
    let alpha = *alpha;
    let beta = *beta;

    match layout {
        CblasLayout::ColMajor => {
            zgemm_raw_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        CblasLayout::RowMajor => {
            let new_transa = transb;
            let new_transb = transa;
            zgemm_raw_colmajor(
                new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc,
            );
        }
    }
}

/// Internal ZGEMM for column-major layout.
unsafe fn zgemm_raw_colmajor(
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: *const Complex64,
    lda: usize,
    b: *const Complex64,
    ldb: usize,
    beta: Complex64,
    c: *mut Complex64,
    ldc: usize,
) {
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite with 0 rather than compute 0*C (which would
    // turn any NaN/Inf in uninitialized C into NaN).
    let zero = Complex64::new(0.0, 0.0);
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == zero {
                *cp = zero;
            } else {
                *cp *= beta;
            }
        }
    }

    if k == 0 {
        return;
    }

    // Compute C += alpha * op(A) * op(B)
    for j in 0..n {
        for p in 0..k {
            let b_val = match transb {
                CblasTranspose::NoTrans => *b.add(p + j * ldb),
                CblasTranspose::Trans => *b.add(j + p * ldb),
                CblasTranspose::ConjTrans => (*b.add(j + p * ldb)).conj(),
            };
            let temp = alpha * b_val;

            for i in 0..m {
                let a_val = match transa {
                    CblasTranspose::NoTrans => *a.add(i + p * lda),
                    CblasTranspose::Trans => *a.add(p + i * lda),
                    CblasTranspose::ConjTrans => (*a.add(p + i * lda)).conj(),
                };
                let cp = c.add(i + j * ldc);
                *cp += a_val * temp;
            }
        }
    }
}

/// Complex single precision GEMM.
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
/// - `a` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `beta` must be non-null and point to one valid, properly aligned,
///   initialized `Complex32` value.
/// - `c` must be non-null, properly aligned for `Complex32`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cgemm(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *const Complex32,
    ldb: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
        || alpha.is_null()
        || beta.is_null()
    {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;
    let alpha = *alpha;
    let beta = *beta;

    match layout {
        CblasLayout::ColMajor => {
            cgemm_raw_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        CblasLayout::RowMajor => {
            let new_transa = transb;
            let new_transb = transa;
            cgemm_raw_colmajor(
                new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc,
            );
        }
    }
}

/// Internal CGEMM for column-major layout.
unsafe fn cgemm_raw_colmajor(
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: *const Complex32,
    lda: usize,
    b: *const Complex32,
    ldb: usize,
    beta: Complex32,
    c: *mut Complex32,
    ldc: usize,
) {
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite with 0 rather than compute 0*C (which would
    // turn any NaN/Inf in uninitialized C into NaN).
    let zero = Complex32::new(0.0, 0.0);
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == zero {
                *cp = zero;
            } else {
                *cp *= beta;
            }
        }
    }

    if k == 0 {
        return;
    }

    for j in 0..n {
        for p in 0..k {
            let b_val = match transb {
                CblasTranspose::NoTrans => *b.add(p + j * ldb),
                CblasTranspose::Trans => *b.add(j + p * ldb),
                CblasTranspose::ConjTrans => (*b.add(j + p * ldb)).conj(),
            };
            let temp = alpha * b_val;

            for i in 0..m {
                let a_val = match transa {
                    CblasTranspose::NoTrans => *a.add(i + p * lda),
                    CblasTranspose::Trans => *a.add(p + i * lda),
                    CblasTranspose::ConjTrans => (*a.add(p + i * lda)).conj(),
                };
                let cp = c.add(i + j * ldc);
                *cp += a_val * temp;
            }
        }
    }
}
