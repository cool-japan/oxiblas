//! BLAS Level 2 FFI - Matrix-Vector operations.

use crate::types::*;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGEMV - Single precision general matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y or y = alpha*A'*x + beta*y for single precision.
///
/// # Safety
/// - `a` must point to a valid matrix of size `lda * n` (col major) or `lda * m` (row major)
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgemv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    alpha: f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: f32,
    y: *mut f32,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

    // Scale y by beta
    let y_slice = slice::from_raw_parts_mut(y, rows * incy);
    if beta == 0.0 {
        for i in 0..rows {
            y_slice[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..rows {
            y_slice[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    let x_slice = slice::from_raw_parts(x, cols * incx);

    // Perform matrix-vector multiply
    let row_major = layout == OblasLayout::RowMajor;
    let do_trans = trans != OblasTranspose::NoTrans;

    for i in 0..rows {
        let mut sum = 0.0f32;
        for j in 0..cols {
            let a_idx = if row_major {
                if do_trans { j * lda + i } else { i * lda + j }
            } else {
                if do_trans { i * lda + j } else { j * lda + i }
            };
            sum += *a.add(a_idx) * x_slice[j * incx];
        }
        y_slice[i * incy] += alpha * sum;
    }
}

// =============================================================================
// DGEMV - Double precision general matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y or y = alpha*A'*x + beta*y for double precision.
///
/// # Safety
/// - `a` must point to a valid matrix of size `lda * n` (col major) or `lda * m` (row major)
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgemv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: f64,
    y: *mut f64,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

    // Scale y by beta
    let y_slice = slice::from_raw_parts_mut(y, rows * incy);
    if beta == 0.0 {
        for i in 0..rows {
            y_slice[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..rows {
            y_slice[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    let x_slice = slice::from_raw_parts(x, cols * incx);

    // Perform matrix-vector multiply
    let row_major = layout == OblasLayout::RowMajor;
    let do_trans = trans != OblasTranspose::NoTrans;

    for i in 0..rows {
        let mut sum = 0.0f64;
        for j in 0..cols {
            let a_idx = if row_major {
                if do_trans { j * lda + i } else { i * lda + j }
            } else {
                if do_trans { i * lda + j } else { j * lda + i }
            };
            sum += *a.add(a_idx) * x_slice[j * incx];
        }
        y_slice[i * incy] += alpha * sum;
    }
}

// =============================================================================
// STRSV - Triangular solve (single precision)
// =============================================================================

/// Solves A*x = b or A'*x = b where A is triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector (input b, output x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_strsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    a: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    // Effective upper/lower after accounting for layout and transpose
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        // Back substitution
        for i in (0..n).rev() {
            let mut sum = x_slice[i * incx];
            for j in (i + 1)..n {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }
            if !unit_diag {
                let a_ii = if row_major {
                    *a.add(i * lda + i)
                } else {
                    *a.add(i * lda + i)
                };
                sum /= a_ii;
            }
            x_slice[i * incx] = sum;
        }
    } else {
        // Forward substitution
        for i in 0..n {
            let mut sum = x_slice[i * incx];
            for j in 0..i {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }
            if !unit_diag {
                let a_ii = if row_major {
                    *a.add(i * lda + i)
                } else {
                    *a.add(i * lda + i)
                };
                sum /= a_ii;
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// DTRSV - Triangular solve (double precision)
// =============================================================================

/// Solves A*x = b or A'*x = b where A is triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector (input b, output x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtrsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    a: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let x_slice = slice::from_raw_parts_mut(x, n * incx);

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    // Effective upper/lower after accounting for layout and transpose
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        // Back substitution
        for i in (0..n).rev() {
            let mut sum = x_slice[i * incx];
            for j in (i + 1)..n {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }
            if !unit_diag {
                let a_ii = if row_major {
                    *a.add(i * lda + i)
                } else {
                    *a.add(i * lda + i)
                };
                sum /= a_ii;
            }
            x_slice[i * incx] = sum;
        }
    } else {
        // Forward substitution
        for i in 0..n {
            let mut sum = x_slice[i * incx];
            for j in 0..i {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }
            if !unit_diag {
                let a_ii = if row_major {
                    *a.add(i * lda + i)
                } else {
                    *a.add(i * lda + i)
                };
                sum /= a_ii;
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// SGER - Rank-1 update (single precision)
// =============================================================================

/// Performs the rank-1 update A = alpha*x*y' + A.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `x` must point to a vector of length m
/// - `y` must point to a vector of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sger(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    alpha: f32,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
    a: *mut f32,
    lda: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, m * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let row_major = layout == OblasLayout::RowMajor;

    for i in 0..m {
        for j in 0..n {
            let a_idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(a_idx) += alpha * x_slice[i * incx] * y_slice[j * incy];
        }
    }
}

// =============================================================================
// DGER - Rank-1 update (double precision)
// =============================================================================

/// Performs the rank-1 update A = alpha*x*y' + A.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `x` must point to a vector of length m
/// - `y` must point to a vector of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dger(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    alpha: f64,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    a: *mut f64,
    lda: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, m * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);

    let row_major = layout == OblasLayout::RowMajor;

    for i in 0..m {
        for j in 0..n {
            let a_idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(a_idx) += alpha * x_slice[i * incx] * y_slice[j * incy];
        }
    }
}

// =============================================================================
// CGEMV - Complex single precision general matrix-vector multiply
// =============================================================================

/// Computes y = alpha*op(A)*x + beta*y for complex single precision.
///
/// # Safety
/// - `a` must point to a valid matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgemv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    x: *const OblasComplex32,
    incx: c_int,
    beta: OblasComplex32,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

    let conj_trans = trans == OblasTranspose::ConjTrans;
    let do_trans = trans != OblasTranspose::NoTrans;
    let row_major = layout == OblasLayout::RowMajor;

    // Scale y by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..rows {
        let idx = i * incy;
        if beta_zero {
            (*y.add(idx)).re = 0.0;
            (*y.add(idx)).im = 0.0;
        } else if !beta_one {
            let y_val = *y.add(idx);
            (*y.add(idx)).re = beta.re * y_val.re - beta.im * y_val.im;
            (*y.add(idx)).im = beta.re * y_val.im + beta.im * y_val.re;
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    // Perform matrix-vector multiply
    for i in 0..rows {
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        for j in 0..cols {
            let a_idx = if row_major {
                if do_trans { j * lda + i } else { i * lda + j }
            } else {
                if do_trans { i * lda + j } else { j * lda + i }
            };
            let a_val = *a.add(a_idx);
            let x_val = *x.add(j * incx);

            let a_re = a_val.re;
            let a_im = if conj_trans { -a_val.im } else { a_val.im };

            sum_re += a_re * x_val.re - a_im * x_val.im;
            sum_im += a_re * x_val.im + a_im * x_val.re;
        }
        let idx = i * incy;
        (*y.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
        (*y.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
    }
}

// =============================================================================
// ZGEMV - Complex double precision general matrix-vector multiply
// =============================================================================

/// Computes y = alpha*op(A)*x + beta*y for complex double precision.
///
/// # Safety
/// - `a` must point to a valid matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgemv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    x: *const OblasComplex64,
    incx: c_int,
    beta: OblasComplex64,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

    let conj_trans = trans == OblasTranspose::ConjTrans;
    let do_trans = trans != OblasTranspose::NoTrans;
    let row_major = layout == OblasLayout::RowMajor;

    // Scale y by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..rows {
        let idx = i * incy;
        if beta_zero {
            (*y.add(idx)).re = 0.0;
            (*y.add(idx)).im = 0.0;
        } else if !beta_one {
            let y_val = *y.add(idx);
            (*y.add(idx)).re = beta.re * y_val.re - beta.im * y_val.im;
            (*y.add(idx)).im = beta.re * y_val.im + beta.im * y_val.re;
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    // Perform matrix-vector multiply
    for i in 0..rows {
        let mut sum_re = 0.0f64;
        let mut sum_im = 0.0f64;
        for j in 0..cols {
            let a_idx = if row_major {
                if do_trans { j * lda + i } else { i * lda + j }
            } else {
                if do_trans { i * lda + j } else { j * lda + i }
            };
            let a_val = *a.add(a_idx);
            let x_val = *x.add(j * incx);

            let a_re = a_val.re;
            let a_im = if conj_trans { -a_val.im } else { a_val.im };

            sum_re += a_re * x_val.re - a_im * x_val.im;
            sum_im += a_re * x_val.im + a_im * x_val.re;
        }
        let idx = i * incy;
        (*y.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
        (*y.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
    }
}

// =============================================================================
// CTRSV - Complex single precision triangular solve
// =============================================================================

/// Solves op(A)*x = b where A is complex triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctrsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    a: *const OblasComplex32,
    lda: c_int,
    x: *mut OblasComplex32,
    incx: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let unit_diag = diag == OblasDiag::Unit;

    // For triangular solve (trsv), upper means back substitution, lower means forward
    let access_upper = upper != do_trans;

    // Complex division helper
    let cdiv = |a_re: f32, a_im: f32, b_re: f32, b_im: f32| -> (f32, f32) {
        let denom = b_re * b_re + b_im * b_im;
        (
            (a_re * b_re + a_im * b_im) / denom,
            (a_im * b_re - a_re * b_im) / denom,
        )
    };

    if access_upper {
        // Upper: back substitution (i from n-1 to 0)
        for i in (0..n).rev() {
            let mut sum_re = (*x.add(i * incx)).re;
            let mut sum_im = (*x.add(i * incx)).im;
            for j in (i + 1)..n {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                let a_val = *a.add(a_idx);
                let x_val = *x.add(j * incx);
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_val.re - a_im * x_val.im;
                sum_im -= a_re * x_val.im + a_im * x_val.re;
            }
            if !unit_diag {
                let a_ii = *a.add(i * lda + i);
                let a_re = a_ii.re;
                let a_im = if conj_trans { -a_ii.im } else { a_ii.im };
                let (res_re, res_im) = cdiv(sum_re, sum_im, a_re, a_im);
                sum_re = res_re;
                sum_im = res_im;
            }
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    } else {
        // Lower: forward substitution (i from 0 to n-1)
        for i in 0..n {
            let mut sum_re = (*x.add(i * incx)).re;
            let mut sum_im = (*x.add(i * incx)).im;
            for j in 0..i {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                let a_val = *a.add(a_idx);
                let x_val = *x.add(j * incx);
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_val.re - a_im * x_val.im;
                sum_im -= a_re * x_val.im + a_im * x_val.re;
            }
            if !unit_diag {
                let a_ii = *a.add(i * lda + i);
                let a_re = a_ii.re;
                let a_im = if conj_trans { -a_ii.im } else { a_ii.im };
                let (res_re, res_im) = cdiv(sum_re, sum_im, a_re, a_im);
                sum_re = res_re;
                sum_im = res_im;
            }
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    }
}

// =============================================================================
// ZTRSV - Complex double precision triangular solve
// =============================================================================

/// Solves op(A)*x = b where A is complex triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztrsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    a: *const OblasComplex64,
    lda: c_int,
    x: *mut OblasComplex64,
    incx: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let access_upper = upper != do_trans;

    // Complex division helper
    let cdiv = |a_re: f64, a_im: f64, b_re: f64, b_im: f64| -> (f64, f64) {
        let denom = b_re * b_re + b_im * b_im;
        (
            (a_re * b_re + a_im * b_im) / denom,
            (a_im * b_re - a_re * b_im) / denom,
        )
    };

    if access_upper {
        // Upper: back substitution (i from n-1 to 0)
        for i in (0..n).rev() {
            let mut sum_re = (*x.add(i * incx)).re;
            let mut sum_im = (*x.add(i * incx)).im;
            for j in (i + 1)..n {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                let a_val = *a.add(a_idx);
                let x_val = *x.add(j * incx);
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_val.re - a_im * x_val.im;
                sum_im -= a_re * x_val.im + a_im * x_val.re;
            }
            if !unit_diag {
                let a_ii = *a.add(i * lda + i);
                let a_re = a_ii.re;
                let a_im = if conj_trans { -a_ii.im } else { a_ii.im };
                let (res_re, res_im) = cdiv(sum_re, sum_im, a_re, a_im);
                sum_re = res_re;
                sum_im = res_im;
            }
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    } else {
        // Lower: forward substitution (i from 0 to n-1)
        for i in 0..n {
            let mut sum_re = (*x.add(i * incx)).re;
            let mut sum_im = (*x.add(i * incx)).im;
            for j in 0..i {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                let a_val = *a.add(a_idx);
                let x_val = *x.add(j * incx);
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_val.re - a_im * x_val.im;
                sum_im -= a_re * x_val.im + a_im * x_val.re;
            }
            if !unit_diag {
                let a_ii = *a.add(i * lda + i);
                let a_re = a_ii.re;
                let a_im = if conj_trans { -a_ii.im } else { a_ii.im };
                let (res_re, res_im) = cdiv(sum_re, sum_im, a_re, a_im);
                sum_re = res_re;
                sum_im = res_im;
            }
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    }
}

// =============================================================================
// CGERU - Complex single precision rank-1 update (unconjugated)
// =============================================================================

/// Performs A = alpha*x*y^T + A (unconjugated).
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgeru(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    alpha: OblasComplex32,
    x: *const OblasComplex32,
    incx: c_int,
    y: *const OblasComplex32,
    incy: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;

    for i in 0..m {
        let x_val = *x.add(i * incx);
        // alpha * x[i]
        let ax_re = alpha.re * x_val.re - alpha.im * x_val.im;
        let ax_im = alpha.re * x_val.im + alpha.im * x_val.re;
        for j in 0..n {
            let y_val = *y.add(j * incy);
            let a_idx = if row_major { i * lda + j } else { j * lda + i };
            // A[i,j] += alpha*x[i]*y[j]
            (*a.add(a_idx)).re += ax_re * y_val.re - ax_im * y_val.im;
            (*a.add(a_idx)).im += ax_re * y_val.im + ax_im * y_val.re;
        }
    }
}

// =============================================================================
// ZGERU - Complex double precision rank-1 update (unconjugated)
// =============================================================================

/// Performs A = alpha*x*y^T + A (unconjugated).
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgeru(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    alpha: OblasComplex64,
    x: *const OblasComplex64,
    incx: c_int,
    y: *const OblasComplex64,
    incy: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;

    for i in 0..m {
        let x_val = *x.add(i * incx);
        let ax_re = alpha.re * x_val.re - alpha.im * x_val.im;
        let ax_im = alpha.re * x_val.im + alpha.im * x_val.re;
        for j in 0..n {
            let y_val = *y.add(j * incy);
            let a_idx = if row_major { i * lda + j } else { j * lda + i };
            (*a.add(a_idx)).re += ax_re * y_val.re - ax_im * y_val.im;
            (*a.add(a_idx)).im += ax_re * y_val.im + ax_im * y_val.re;
        }
    }
}

// =============================================================================
// CGERC - Complex single precision rank-1 update (conjugated)
// =============================================================================

/// Performs A = alpha*x*conj(y)^T + A (conjugated).
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgerc(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    alpha: OblasComplex32,
    x: *const OblasComplex32,
    incx: c_int,
    y: *const OblasComplex32,
    incy: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;

    for i in 0..m {
        let x_val = *x.add(i * incx);
        let ax_re = alpha.re * x_val.re - alpha.im * x_val.im;
        let ax_im = alpha.re * x_val.im + alpha.im * x_val.re;
        for j in 0..n {
            let y_val = *y.add(j * incy);
            // Conjugate y
            let y_re = y_val.re;
            let y_im = -y_val.im;
            let a_idx = if row_major { i * lda + j } else { j * lda + i };
            (*a.add(a_idx)).re += ax_re * y_re - ax_im * y_im;
            (*a.add(a_idx)).im += ax_re * y_im + ax_im * y_re;
        }
    }
}

// =============================================================================
// ZGERC - Complex double precision rank-1 update (conjugated)
// =============================================================================

/// Performs A = alpha*x*conj(y)^T + A (conjugated).
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgerc(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    alpha: OblasComplex64,
    x: *const OblasComplex64,
    incx: c_int,
    y: *const OblasComplex64,
    incy: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;

    for i in 0..m {
        let x_val = *x.add(i * incx);
        let ax_re = alpha.re * x_val.re - alpha.im * x_val.im;
        let ax_im = alpha.re * x_val.im + alpha.im * x_val.re;
        for j in 0..n {
            let y_val = *y.add(j * incy);
            let y_re = y_val.re;
            let y_im = -y_val.im;
            let a_idx = if row_major { i * lda + j } else { j * lda + i };
            (*a.add(a_idx)).re += ax_re * y_re - ax_im * y_im;
            (*a.add(a_idx)).im += ax_re * y_im + ax_im * y_re;
        }
    }
}

// =============================================================================
// CHEMV - Complex single precision Hermitian matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y where A is Hermitian.
///
/// # Safety
/// - `a` must point to a valid Hermitian matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_chemv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    x: *const OblasComplex32,
    incx: c_int,
    beta: OblasComplex32,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Scale y by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..n {
        let idx = i * incy;
        if beta_zero {
            (*y.add(idx)).re = 0.0;
            (*y.add(idx)).im = 0.0;
        } else if !beta_one {
            let y_val = *y.add(idx);
            (*y.add(idx)).re = beta.re * y_val.re - beta.im * y_val.im;
            (*y.add(idx)).im = beta.re * y_val.im + beta.im * y_val.re;
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    // Helper to get Hermitian element
    let get_a = |i: usize, j: usize| -> (f32, f32) {
        if i == j {
            // Diagonal: must be real
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let a_val = *a.add(idx);
            (a_val.re, 0.0)
        } else if (upper && j > i) || (!upper && i > j) {
            // In stored region
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let a_val = *a.add(idx);
            (a_val.re, a_val.im)
        } else {
            // Mirror with conjugate
            let idx = if row_major { j * lda + i } else { i * lda + j };
            let a_val = *a.add(idx);
            (a_val.re, -a_val.im)
        }
    };

    for i in 0..n {
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        for j in 0..n {
            let (a_re, a_im) = get_a(i, j);
            let x_val = *x.add(j * incx);
            sum_re += a_re * x_val.re - a_im * x_val.im;
            sum_im += a_re * x_val.im + a_im * x_val.re;
        }
        let idx = i * incy;
        (*y.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
        (*y.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
    }
}

// =============================================================================
// ZHEMV - Complex double precision Hermitian matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y where A is Hermitian.
///
/// # Safety
/// - `a` must point to a valid Hermitian matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zhemv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    x: *const OblasComplex64,
    incx: c_int,
    beta: OblasComplex64,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Scale y by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..n {
        let idx = i * incy;
        if beta_zero {
            (*y.add(idx)).re = 0.0;
            (*y.add(idx)).im = 0.0;
        } else if !beta_one {
            let y_val = *y.add(idx);
            (*y.add(idx)).re = beta.re * y_val.re - beta.im * y_val.im;
            (*y.add(idx)).im = beta.re * y_val.im + beta.im * y_val.re;
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let get_a = |i: usize, j: usize| -> (f64, f64) {
        if i == j {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let a_val = *a.add(idx);
            (a_val.re, 0.0)
        } else if (upper && j > i) || (!upper && i > j) {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let a_val = *a.add(idx);
            (a_val.re, a_val.im)
        } else {
            let idx = if row_major { j * lda + i } else { i * lda + j };
            let a_val = *a.add(idx);
            (a_val.re, -a_val.im)
        }
    };

    for i in 0..n {
        let mut sum_re = 0.0f64;
        let mut sum_im = 0.0f64;
        for j in 0..n {
            let (a_re, a_im) = get_a(i, j);
            let x_val = *x.add(j * incx);
            sum_re += a_re * x_val.re - a_im * x_val.im;
            sum_im += a_re * x_val.im + a_im * x_val.re;
        }
        let idx = i * incy;
        (*y.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
        (*y.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
    }
}

// =============================================================================
// CHER - Complex single precision Hermitian rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^H + A where A is Hermitian.
///
/// # Safety
/// - `a` must point to a valid Hermitian matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cher(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f32,
    x: *const OblasComplex32,
    incx: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    for i in 0..n {
        let x_i = *x.add(i * incx);
        // Diagonal: A[i,i] += alpha * |x[i]|^2 (real)
        let diag_idx = if row_major { i * lda + i } else { i * lda + i };
        (*a.add(diag_idx)).re += alpha * (x_i.re * x_i.re + x_i.im * x_i.im);
        (*a.add(diag_idx)).im = 0.0; // Ensure diagonal is real

        // Off-diagonal
        let (start, end) = if upper { (i + 1, n) } else { (0, i) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            // alpha * x[i] * conj(x[j])
            let update_re = alpha * (x_i.re * x_j.re + x_i.im * x_j.im);
            let update_im = alpha * (x_i.im * x_j.re - x_i.re * x_j.im);
            let idx = if row_major { i * lda + j } else { j * lda + i };
            (*a.add(idx)).re += update_re;
            (*a.add(idx)).im += update_im;
        }
    }
}

// =============================================================================
// ZHER - Complex double precision Hermitian rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^H + A where A is Hermitian.
///
/// # Safety
/// - `a` must point to a valid Hermitian matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zher(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f64,
    x: *const OblasComplex64,
    incx: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    for i in 0..n {
        let x_i = *x.add(i * incx);
        let diag_idx = if row_major { i * lda + i } else { i * lda + i };
        (*a.add(diag_idx)).re += alpha * (x_i.re * x_i.re + x_i.im * x_i.im);
        (*a.add(diag_idx)).im = 0.0;

        let (start, end) = if upper { (i + 1, n) } else { (0, i) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            let update_re = alpha * (x_i.re * x_j.re + x_i.im * x_j.im);
            let update_im = alpha * (x_i.im * x_j.re - x_i.re * x_j.im);
            let idx = if row_major { i * lda + j } else { j * lda + i };
            (*a.add(idx)).re += update_re;
            (*a.add(idx)).im += update_im;
        }
    }
}

// =============================================================================
// CHER2 - Complex single precision Hermitian rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^H + conj(alpha)*y*x^H + A where A is Hermitian.
///
/// # Safety
/// - `a` must point to a valid Hermitian matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cher2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex32,
    x: *const OblasComplex32,
    incx: c_int,
    y: *const OblasComplex32,
    incy: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // conj(alpha)
    let alpha_conj_re = alpha.re;
    let alpha_conj_im = -alpha.im;

    for i in 0..n {
        let x_i = *x.add(i * incx);
        let y_i = *y.add(i * incy);

        // Diagonal: update must be real
        // alpha * x[i] * conj(y[i]) + conj(alpha) * y[i] * conj(x[i])
        // Both terms are conjugates of each other, so sum is 2*Re(alpha*x[i]*conj(y[i]))
        let xy_re = x_i.re * y_i.re + x_i.im * y_i.im;
        let xy_im = x_i.im * y_i.re - x_i.re * y_i.im;
        let diag_update = 2.0 * (alpha.re * xy_re - alpha.im * xy_im);
        let diag_idx = if row_major { i * lda + i } else { i * lda + i };
        (*a.add(diag_idx)).re += diag_update;
        (*a.add(diag_idx)).im = 0.0;

        // Off-diagonal
        let (start, end) = if upper { (i + 1, n) } else { (0, i) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            let y_j = *y.add(j * incy);

            // alpha * x[i] * conj(y[j])
            let xy_re1 = x_i.re * y_j.re + x_i.im * y_j.im;
            let xy_im1 = x_i.im * y_j.re - x_i.re * y_j.im;
            let term1_re = alpha.re * xy_re1 - alpha.im * xy_im1;
            let term1_im = alpha.re * xy_im1 + alpha.im * xy_re1;

            // conj(alpha) * y[i] * conj(x[j])
            let yx_re = y_i.re * x_j.re + y_i.im * x_j.im;
            let yx_im = y_i.im * x_j.re - y_i.re * x_j.im;
            let term2_re = alpha_conj_re * yx_re - alpha_conj_im * yx_im;
            let term2_im = alpha_conj_re * yx_im + alpha_conj_im * yx_re;

            let idx = if row_major { i * lda + j } else { j * lda + i };
            (*a.add(idx)).re += term1_re + term2_re;
            (*a.add(idx)).im += term1_im + term2_im;
        }
    }
}

// =============================================================================
// ZHER2 - Complex double precision Hermitian rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^H + conj(alpha)*y*x^H + A where A is Hermitian.
///
/// # Safety
/// - `a` must point to a valid Hermitian matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zher2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex64,
    x: *const OblasComplex64,
    incx: c_int,
    y: *const OblasComplex64,
    incy: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    let alpha_conj_re = alpha.re;
    let alpha_conj_im = -alpha.im;

    for i in 0..n {
        let x_i = *x.add(i * incx);
        let y_i = *y.add(i * incy);

        let xy_re = x_i.re * y_i.re + x_i.im * y_i.im;
        let xy_im = x_i.im * y_i.re - x_i.re * y_i.im;
        let diag_update = 2.0 * (alpha.re * xy_re - alpha.im * xy_im);
        let diag_idx = if row_major { i * lda + i } else { i * lda + i };
        (*a.add(diag_idx)).re += diag_update;
        (*a.add(diag_idx)).im = 0.0;

        let (start, end) = if upper { (i + 1, n) } else { (0, i) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            let y_j = *y.add(j * incy);

            let xy_re1 = x_i.re * y_j.re + x_i.im * y_j.im;
            let xy_im1 = x_i.im * y_j.re - x_i.re * y_j.im;
            let term1_re = alpha.re * xy_re1 - alpha.im * xy_im1;
            let term1_im = alpha.re * xy_im1 + alpha.im * xy_re1;

            let yx_re = y_i.re * x_j.re + y_i.im * x_j.im;
            let yx_im = y_i.im * x_j.re - y_i.re * x_j.im;
            let term2_re = alpha_conj_re * yx_re - alpha_conj_im * yx_im;
            let term2_im = alpha_conj_re * yx_im + alpha_conj_im * yx_re;

            let idx = if row_major { i * lda + j } else { j * lda + i };
            (*a.add(idx)).re += term1_re + term2_re;
            (*a.add(idx)).im += term1_im + term2_im;
        }
    }
}

// =============================================================================
