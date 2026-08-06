//! CBLAS Level 2 (matrix-vector) real operations: GEMV.

use super::super::types::*;
use super::super::validate::gemv_params_valid;
use super::vector_start_offset;
/// Double precision GEMV: y = alpha * op(A) * x + beta * y.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
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
pub unsafe extern "C" fn cblas_dgemv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemv_params_valid(layout, m, n, lda) || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    // Determine effective dimensions based on transpose. `x` has `cols`
    // elements, `y` has `rows` elements.
    let (rows, cols) = match trans {
        CblasTranspose::NoTrans => (m, n),
        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
    };

    // Reference start offsets so that negative incx/incy walk each vector
    // back-to-front (see `vector_start_offset`) instead of reading OOB.
    let x_start = vector_start_offset(cols, incx);
    let y_start = vector_start_offset(rows, incy);

    // Scale y by beta. When beta == 0 the BLAS contract says y need not be
    // initialized, so we overwrite rather than multiply (avoids 0*NaN = NaN).
    for i in 0..rows {
        let yp = y.offset(y_start + i as isize * incy);
        if beta == 0.0 {
            *yp = 0.0;
        } else {
            *yp *= beta;
        }
    }

    // Compute matrix-vector product
    match layout {
        CblasLayout::ColMajor => {
            for i in 0..rows {
                let yp = y.offset(y_start + i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai + aj * lda);
                    let x_val = *x.offset(x_start + j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
        CblasLayout::RowMajor => {
            for i in 0..rows {
                let yp = y.offset(y_start + i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai * lda + aj);
                    let x_val = *x.offset(x_start + j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
    }
}

/// Single precision GEMV: y = alpha * op(A) * x + beta * y.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
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
pub unsafe extern "C" fn cblas_sgemv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemv_params_valid(layout, m, n, lda) || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    let (rows, cols) = match trans {
        CblasTranspose::NoTrans => (m, n),
        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
    };

    let x_start = vector_start_offset(cols, incx);
    let y_start = vector_start_offset(rows, incy);

    // beta == 0 overwrites y (uninitialized y allowed by the BLAS contract).
    for i in 0..rows {
        let yp = y.offset(y_start + i as isize * incy);
        if beta == 0.0 {
            *yp = 0.0;
        } else {
            *yp *= beta;
        }
    }

    match layout {
        CblasLayout::ColMajor => {
            for i in 0..rows {
                let yp = y.offset(y_start + i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai + aj * lda);
                    let x_val = *x.offset(x_start + j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
        CblasLayout::RowMajor => {
            for i in 0..rows {
                let yp = y.offset(y_start + i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai * lda + aj);
                    let x_val = *x.offset(x_start + j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
    }
}
