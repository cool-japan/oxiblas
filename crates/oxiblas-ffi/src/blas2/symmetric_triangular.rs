//! BLAS Level 2 FFI - Matrix-Vector operations.

use crate::types::*;
use std::ffi::c_int;

// SSYMV - Single precision symmetric matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y where A is symmetric.
///
/// # Safety
/// - `a` must point to a valid symmetric matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssymv(
    layout: OblasLayout,
    uplo: OblasUplo,
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
    for i in 0..n {
        if beta == 0.0 {
            *y.add(i * incy) = 0.0;
        } else if beta != 1.0 {
            *y.add(i * incy) *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    let get_a = |i: usize, j: usize| -> f32 {
        let (ii, jj) = if (upper && j >= i) || (!upper && i >= j) {
            (i, j)
        } else {
            (j, i)
        };
        let idx = if row_major {
            ii * lda + jj
        } else {
            jj * lda + ii
        };
        *a.add(idx)
    };

    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += get_a(i, j) * *x.add(j * incx);
        }
        *y.add(i * incy) += alpha * sum;
    }
}

// =============================================================================
// DSYMV - Double precision symmetric matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y where A is symmetric.
///
/// # Safety
/// - `a` must point to a valid symmetric matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsymv(
    layout: OblasLayout,
    uplo: OblasUplo,
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
    if n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    for i in 0..n {
        if beta == 0.0 {
            *y.add(i * incy) = 0.0;
        } else if beta != 1.0 {
            *y.add(i * incy) *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    let get_a = |i: usize, j: usize| -> f64 {
        let (ii, jj) = if (upper && j >= i) || (!upper && i >= j) {
            (i, j)
        } else {
            (j, i)
        };
        let idx = if row_major {
            ii * lda + jj
        } else {
            jj * lda + ii
        };
        *a.add(idx)
    };

    for i in 0..n {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += get_a(i, j) * *x.add(j * incx);
        }
        *y.add(i * incy) += alpha * sum;
    }
}

// =============================================================================
// SSYR - Single precision symmetric rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^T + A where A is symmetric.
///
/// # Safety
/// - `a` must point to a valid symmetric matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssyr(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f32,
    x: *const f32,
    incx: c_int,
    a: *mut f32,
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
        let (start, end) = if upper { (i, n) } else { (0, i + 1) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) += alpha * x_i * x_j;
        }
    }
}

// =============================================================================
// DSYR - Double precision symmetric rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^T + A where A is symmetric.
///
/// # Safety
/// - `a` must point to a valid symmetric matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsyr(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f64,
    x: *const f64,
    incx: c_int,
    a: *mut f64,
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
        let (start, end) = if upper { (i, n) } else { (0, i + 1) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) += alpha * x_i * x_j;
        }
    }
}

// =============================================================================
// SSYR2 - Single precision symmetric rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^T + alpha*y*x^T + A where A is symmetric.
///
/// # Safety
/// - `a` must point to a valid symmetric matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssyr2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f32,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
    a: *mut f32,
    lda: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    for i in 0..n {
        let x_i = *x.add(i * incx);
        let y_i = *y.add(i * incy);
        let (start, end) = if upper { (i, n) } else { (0, i + 1) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            let y_j = *y.add(j * incy);
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) += alpha * (x_i * y_j + y_i * x_j);
        }
    }
}

// =============================================================================
// DSYR2 - Double precision symmetric rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^T + alpha*y*x^T + A where A is symmetric.
///
/// # Safety
/// - `a` must point to a valid symmetric matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsyr2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f64,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    a: *mut f64,
    lda: c_int,
) {
    if n <= 0 || a.is_null() || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    for i in 0..n {
        let x_i = *x.add(i * incx);
        let y_i = *y.add(i * incy);
        let (start, end) = if upper { (i, n) } else { (0, i + 1) };
        for j in start..end {
            let x_j = *x.add(j * incx);
            let y_j = *y.add(j * incy);
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) += alpha * (x_i * y_j + y_i * x_j);
        }
    }
}

// =============================================================================
// STRMV - Single precision triangular matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x where A is triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_strmv(
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
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    // Determine access pattern: upper means we access j >= i
    // upper + notrans => access upper (j >= i)
    // upper + trans => access lower (j <= i)
    // lower + notrans => access lower (j <= i)
    // lower + trans => access upper (j >= i)
    let access_upper = upper != do_trans;

    if access_upper {
        // Upper: process top to bottom, access j >= i
        for i in 0..n {
            let mut sum = if unit_diag { *x.add(i * incx) } else { 0.0f32 };
            let start = if unit_diag { i + 1 } else { i };
            for j in start..n {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum += *a.add(a_idx) * *x.add(j * incx);
            }
            *x.add(i * incx) = sum;
        }
    } else {
        // Lower: process bottom to top, access j <= i
        for i in (0..n).rev() {
            let mut sum = if unit_diag { *x.add(i * incx) } else { 0.0f32 };
            let end = if unit_diag { i } else { i + 1 };
            for j in 0..end {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum += *a.add(a_idx) * *x.add(j * incx);
            }
            *x.add(i * incx) = sum;
        }
    }
}

// =============================================================================
// DTRMV - Double precision triangular matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x where A is triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtrmv(
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
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let access_upper = upper != do_trans;

    if access_upper {
        for i in 0..n {
            let mut sum = if unit_diag { *x.add(i * incx) } else { 0.0f64 };
            let start = if unit_diag { i + 1 } else { i };
            for j in start..n {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum += *a.add(a_idx) * *x.add(j * incx);
            }
            *x.add(i * incx) = sum;
        }
    } else {
        for i in (0..n).rev() {
            let mut sum = if unit_diag { *x.add(i * incx) } else { 0.0f64 };
            let end = if unit_diag { i } else { i + 1 };
            for j in 0..end {
                let a_idx = if row_major {
                    if do_trans { j * lda + i } else { i * lda + j }
                } else {
                    if do_trans { i * lda + j } else { j * lda + i }
                };
                sum += *a.add(a_idx) * *x.add(j * incx);
            }
            *x.add(i * incx) = sum;
        }
    }
}

// =============================================================================
// CTRMV - Complex single precision triangular matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x where A is complex triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctrmv(
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

    let access_upper = upper != do_trans;

    if access_upper {
        for i in 0..n {
            let x_i = *x.add(i * incx);
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f32 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f32 };
            let start = if unit_diag { i + 1 } else { i };
            for j in start..n {
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
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    } else {
        for i in (0..n).rev() {
            let x_i = *x.add(i * incx);
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f32 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f32 };
            let end = if unit_diag { i } else { i + 1 };
            for j in 0..end {
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
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    }
}

// =============================================================================
// ZTRMV - Complex double precision triangular matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x where A is complex triangular.
///
/// # Safety
/// - `a` must point to a valid triangular matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztrmv(
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

    if access_upper {
        for i in 0..n {
            let x_i = *x.add(i * incx);
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f64 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f64 };
            let start = if unit_diag { i + 1 } else { i };
            for j in start..n {
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
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    } else {
        for i in (0..n).rev() {
            let x_i = *x.add(i * incx);
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f64 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f64 };
            let end = if unit_diag { i } else { i + 1 };
            for j in 0..end {
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
            (*x.add(i * incx)).re = sum_re;
            (*x.add(i * incx)).im = sum_im;
        }
    }
}

// =============================================================================
