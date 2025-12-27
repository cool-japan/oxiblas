//! BLAS Level 2 FFI - Matrix-Vector operations.

use crate::types::*;
use std::ffi::c_int;
use std::slice;

use super::helpers::*;

// SSPMV - Single precision symmetric packed matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a symmetric packed matrix (single precision).
///
/// # Safety
/// - `ap` must point to a valid packed array of size n*(n+1)/2
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sspmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f32,
    ap: *const f32,
    x: *const f32,
    incx: c_int,
    beta: f32,
    y: *mut f32,
    incy: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    if beta == 0.0 {
        for i in 0..n {
            y_slice[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..n {
            y_slice[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    for i in 0..n {
        let mut sum = 0.0f32;
        let x_i = x_slice[i * incx];

        if eff_upper {
            // Diagonal
            let ap_idx = upper_packed_index(i, i);
            sum += *ap.add(ap_idx) * x_i;

            // Off-diagonal elements (j > i)
            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                sum += a_val * x_slice[j * incx];
                y_slice[j * incy] += alpha * a_val * x_i;
            }
        } else {
            // Off-diagonal elements (j < i)
            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                sum += a_val * x_slice[j * incx];
                y_slice[j * incy] += alpha * a_val * x_i;
            }

            // Diagonal
            let ap_idx = lower_packed_index(i, i, n);
            sum += *ap.add(ap_idx) * x_i;
        }
        y_slice[i * incy] += alpha * sum;
    }
}

// =============================================================================
// DSPMV - Double precision symmetric packed matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a symmetric packed matrix (double precision).
///
/// # Safety
/// - `ap` must point to a valid packed array of size n*(n+1)/2
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dspmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f64,
    ap: *const f64,
    x: *const f64,
    incx: c_int,
    beta: f64,
    y: *mut f64,
    incy: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let y_slice = slice::from_raw_parts_mut(y, n * incy);

    if beta == 0.0 {
        for i in 0..n {
            y_slice[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..n {
            y_slice[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    for i in 0..n {
        let mut sum = 0.0f64;
        let x_i = x_slice[i * incx];

        if eff_upper {
            let ap_idx = upper_packed_index(i, i);
            sum += *ap.add(ap_idx) * x_i;

            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                sum += a_val * x_slice[j * incx];
                y_slice[j * incy] += alpha * a_val * x_i;
            }
        } else {
            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                sum += a_val * x_slice[j * incx];
                y_slice[j * incy] += alpha * a_val * x_i;
            }

            let ap_idx = lower_packed_index(i, i, n);
            sum += *ap.add(ap_idx) * x_i;
        }
        y_slice[i * incy] += alpha * sum;
    }
}

// =============================================================================
// CHPMV - Single precision complex Hermitian packed matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a Hermitian packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_chpmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex32,
    ap: *const OblasComplex32,
    x: *const OblasComplex32,
    incx: c_int,
    beta: OblasComplex32,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let y_slice = slice::from_raw_parts_mut(y, n * incy);
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    if beta_zero {
        for i in 0..n {
            y_slice[i * incy] = OblasComplex32 { re: 0.0, im: 0.0 };
        }
    } else if !beta_one {
        for i in 0..n {
            let y_i = y_slice[i * incy];
            y_slice[i * incy] = OblasComplex32 {
                re: beta.re * y_i.re - beta.im * y_i.im,
                im: beta.re * y_i.im + beta.im * y_i.re,
            };
        }
    }

    let alpha_zero = alpha.re == 0.0 && alpha.im == 0.0;
    if alpha_zero {
        return;
    }

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    for i in 0..n {
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        let x_i = x_slice[i * incx];

        if eff_upper {
            // Diagonal (real)
            let ap_idx = upper_packed_index(i, i);
            let a_diag = (*ap.add(ap_idx)).re;
            sum_re += a_diag * x_i.re;
            sum_im += a_diag * x_i.im;

            // Off-diagonal
            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];

                // A[i,j] * x[j]
                sum_re += a_val.re * x_j.re - a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im + a_val.im * x_j.re;

                // conj(A[i,j]) * x[i] for y[j]
                let conj_a_x_re = a_val.re * x_i.re + a_val.im * x_i.im;
                let conj_a_x_im = a_val.re * x_i.im - a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * conj_a_x_re - alpha.im * conj_a_x_im;
                y_slice[j * incy].im += alpha.re * conj_a_x_im + alpha.im * conj_a_x_re;
            }
        } else {
            // Off-diagonal
            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];

                // conj(A[i,j]) * x[j]
                sum_re += a_val.re * x_j.re + a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im - a_val.im * x_j.re;

                // A[i,j] * x[i] for y[j]
                let a_x_re = a_val.re * x_i.re - a_val.im * x_i.im;
                let a_x_im = a_val.re * x_i.im + a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * a_x_re - alpha.im * a_x_im;
                y_slice[j * incy].im += alpha.re * a_x_im + alpha.im * a_x_re;
            }

            // Diagonal (real)
            let ap_idx = lower_packed_index(i, i, n);
            let a_diag = (*ap.add(ap_idx)).re;
            sum_re += a_diag * x_i.re;
            sum_im += a_diag * x_i.im;
        }

        let prod_re = alpha.re * sum_re - alpha.im * sum_im;
        let prod_im = alpha.re * sum_im + alpha.im * sum_re;
        y_slice[i * incy].re += prod_re;
        y_slice[i * incy].im += prod_im;
    }
}

// =============================================================================
// ZHPMV - Double precision complex Hermitian packed matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a Hermitian packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zhpmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex64,
    ap: *const OblasComplex64,
    x: *const OblasComplex64,
    incx: c_int,
    beta: OblasComplex64,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let y_slice = slice::from_raw_parts_mut(y, n * incy);
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    if beta_zero {
        for i in 0..n {
            y_slice[i * incy] = OblasComplex64 { re: 0.0, im: 0.0 };
        }
    } else if !beta_one {
        for i in 0..n {
            let y_i = y_slice[i * incy];
            y_slice[i * incy] = OblasComplex64 {
                re: beta.re * y_i.re - beta.im * y_i.im,
                im: beta.re * y_i.im + beta.im * y_i.re,
            };
        }
    }

    let alpha_zero = alpha.re == 0.0 && alpha.im == 0.0;
    if alpha_zero {
        return;
    }

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    for i in 0..n {
        let mut sum_re = 0.0f64;
        let mut sum_im = 0.0f64;
        let x_i = x_slice[i * incx];

        if eff_upper {
            let ap_idx = upper_packed_index(i, i);
            let a_diag = (*ap.add(ap_idx)).re;
            sum_re += a_diag * x_i.re;
            sum_im += a_diag * x_i.im;

            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];

                sum_re += a_val.re * x_j.re - a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im + a_val.im * x_j.re;

                let conj_a_x_re = a_val.re * x_i.re + a_val.im * x_i.im;
                let conj_a_x_im = a_val.re * x_i.im - a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * conj_a_x_re - alpha.im * conj_a_x_im;
                y_slice[j * incy].im += alpha.re * conj_a_x_im + alpha.im * conj_a_x_re;
            }
        } else {
            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];

                sum_re += a_val.re * x_j.re + a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im - a_val.im * x_j.re;

                let a_x_re = a_val.re * x_i.re - a_val.im * x_i.im;
                let a_x_im = a_val.re * x_i.im + a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * a_x_re - alpha.im * a_x_im;
                y_slice[j * incy].im += alpha.re * a_x_im + alpha.im * a_x_re;
            }

            let ap_idx = lower_packed_index(i, i, n);
            let a_diag = (*ap.add(ap_idx)).re;
            sum_re += a_diag * x_i.re;
            sum_im += a_diag * x_i.im;
        }

        let prod_re = alpha.re * sum_re - alpha.im * sum_im;
        let prod_im = alpha.re * sum_im + alpha.im * sum_re;
        y_slice[i * incy].re += prod_re;
        y_slice[i * incy].im += prod_im;
    }
}

// =============================================================================
// STPMV - Single precision triangular packed matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a triangular packed matrix (single precision).
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_stpmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const f32,
    x: *mut f32,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        for i in (0..n).rev() {
            let x_i = x_slice[i * incx];
            let mut sum = if unit_diag { x_i } else { 0.0f32 };

            let j_start = if unit_diag { i + 1 } else { i };
            for j in j_start..n {
                let ap_idx = upper_packed_index(i, j);
                sum += *ap.add(ap_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    } else {
        for i in 0..n {
            let x_i = x_slice[i * incx];
            let mut sum = if unit_diag { x_i } else { 0.0f32 };

            let j_end = if unit_diag { i } else { i + 1 };
            for j in 0..j_end {
                let ap_idx = lower_packed_index(i, j, n);
                sum += *ap.add(ap_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// DTPMV - Double precision triangular packed matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a triangular packed matrix (double precision).
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtpmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const f64,
    x: *mut f64,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        for i in (0..n).rev() {
            let x_i = x_slice[i * incx];
            let mut sum = if unit_diag { x_i } else { 0.0f64 };

            let j_start = if unit_diag { i + 1 } else { i };
            for j in j_start..n {
                let ap_idx = upper_packed_index(i, j);
                sum += *ap.add(ap_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    } else {
        for i in 0..n {
            let x_i = x_slice[i * incx];
            let mut sum = if unit_diag { x_i } else { 0.0f64 };

            let j_end = if unit_diag { i } else { i + 1 };
            for j in 0..j_end {
                let ap_idx = lower_packed_index(i, j, n);
                sum += *ap.add(ap_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// CTPMV - Single precision complex triangular packed matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a complex triangular packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctpmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const OblasComplex32,
    x: *mut OblasComplex32,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        for i in (0..n).rev() {
            let x_i = x_slice[i * incx];
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f32 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f32 };

            let j_start = if unit_diag { i + 1 } else { i };
            for j in j_start..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re += a_re * x_j.re - a_im * x_j.im;
                sum_im += a_re * x_j.im + a_im * x_j.re;
            }
            x_slice[i * incx] = OblasComplex32 {
                re: sum_re,
                im: sum_im,
            };
        }
    } else {
        for i in 0..n {
            let x_i = x_slice[i * incx];
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f32 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f32 };

            let j_end = if unit_diag { i } else { i + 1 };
            for j in 0..j_end {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re += a_re * x_j.re - a_im * x_j.im;
                sum_im += a_re * x_j.im + a_im * x_j.re;
            }
            x_slice[i * incx] = OblasComplex32 {
                re: sum_re,
                im: sum_im,
            };
        }
    }
}

// =============================================================================
// ZTPMV - Double precision complex triangular packed matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a complex triangular packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztpmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const OblasComplex64,
    x: *mut OblasComplex64,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        for i in (0..n).rev() {
            let x_i = x_slice[i * incx];
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f64 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f64 };

            let j_start = if unit_diag { i + 1 } else { i };
            for j in j_start..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re += a_re * x_j.re - a_im * x_j.im;
                sum_im += a_re * x_j.im + a_im * x_j.re;
            }
            x_slice[i * incx] = OblasComplex64 {
                re: sum_re,
                im: sum_im,
            };
        }
    } else {
        for i in 0..n {
            let x_i = x_slice[i * incx];
            let mut sum_re = if unit_diag { x_i.re } else { 0.0f64 };
            let mut sum_im = if unit_diag { x_i.im } else { 0.0f64 };

            let j_end = if unit_diag { i } else { i + 1 };
            for j in 0..j_end {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re += a_re * x_j.re - a_im * x_j.im;
                sum_im += a_re * x_j.im + a_im * x_j.re;
            }
            x_slice[i * incx] = OblasComplex64 {
                re: sum_re,
                im: sum_im,
            };
        }
    }
}

// =============================================================================
// STPSV - Single precision triangular packed solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a triangular packed matrix (single precision).
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_stpsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const f32,
    x: *mut f32,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        // Back substitution
        for i in (0..n).rev() {
            let mut sum = x_slice[i * incx];

            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                sum -= *ap.add(ap_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let ap_idx = upper_packed_index(i, i);
                sum /= *ap.add(ap_idx);
            }
            x_slice[i * incx] = sum;
        }
    } else {
        // Forward substitution
        for i in 0..n {
            let mut sum = x_slice[i * incx];

            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                sum -= *ap.add(ap_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let ap_idx = lower_packed_index(i, i, n);
                sum /= *ap.add(ap_idx);
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// DTPSV - Double precision triangular packed solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a triangular packed matrix (double precision).
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtpsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const f64,
    x: *mut f64,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        for i in (0..n).rev() {
            let mut sum = x_slice[i * incx];

            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                sum -= *ap.add(ap_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let ap_idx = upper_packed_index(i, i);
                sum /= *ap.add(ap_idx);
            }
            x_slice[i * incx] = sum;
        }
    } else {
        for i in 0..n {
            let mut sum = x_slice[i * incx];

            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                sum -= *ap.add(ap_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let ap_idx = lower_packed_index(i, i, n);
                sum /= *ap.add(ap_idx);
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// CTPSV - Single precision complex triangular packed solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a complex triangular packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctpsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const OblasComplex32,
    x: *mut OblasComplex32,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        for i in (0..n).rev() {
            let mut sum_re = x_slice[i * incx].re;
            let mut sum_im = x_slice[i * incx].im;

            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let ap_idx = upper_packed_index(i, i);
                let a_diag = *ap.add(ap_idx);
                let a_re = a_diag.re;
                let a_im = if conj_trans { -a_diag.im } else { a_diag.im };
                let denom = a_re * a_re + a_im * a_im;
                let new_re = (sum_re * a_re + sum_im * a_im) / denom;
                let new_im = (sum_im * a_re - sum_re * a_im) / denom;
                sum_re = new_re;
                sum_im = new_im;
            }
            x_slice[i * incx] = OblasComplex32 {
                re: sum_re,
                im: sum_im,
            };
        }
    } else {
        for i in 0..n {
            let mut sum_re = x_slice[i * incx].re;
            let mut sum_im = x_slice[i * incx].im;

            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let ap_idx = lower_packed_index(i, i, n);
                let a_diag = *ap.add(ap_idx);
                let a_re = a_diag.re;
                let a_im = if conj_trans { -a_diag.im } else { a_diag.im };
                let denom = a_re * a_re + a_im * a_im;
                let new_re = (sum_re * a_re + sum_im * a_im) / denom;
                let new_im = (sum_im * a_re - sum_re * a_im) / denom;
                sum_re = new_re;
                sum_im = new_im;
            }
            x_slice[i * incx] = OblasComplex32 {
                re: sum_re,
                im: sum_im,
            };
        }
    }
}

// =============================================================================
// ZTPSV - Double precision complex triangular packed solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a complex triangular packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztpsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    ap: *const OblasComplex64,
    x: *mut OblasComplex64,
    incx: c_int,
) {
    if n <= 0 || ap.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let unit_diag = diag == OblasDiag::Unit;
    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        for i in (0..n).rev() {
            let mut sum_re = x_slice[i * incx].re;
            let mut sum_im = x_slice[i * incx].im;

            for j in (i + 1)..n {
                let ap_idx = upper_packed_index(i, j);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let ap_idx = upper_packed_index(i, i);
                let a_diag = *ap.add(ap_idx);
                let a_re = a_diag.re;
                let a_im = if conj_trans { -a_diag.im } else { a_diag.im };
                let denom = a_re * a_re + a_im * a_im;
                let new_re = (sum_re * a_re + sum_im * a_im) / denom;
                let new_im = (sum_im * a_re - sum_re * a_im) / denom;
                sum_re = new_re;
                sum_im = new_im;
            }
            x_slice[i * incx] = OblasComplex64 {
                re: sum_re,
                im: sum_im,
            };
        }
    } else {
        for i in 0..n {
            let mut sum_re = x_slice[i * incx].re;
            let mut sum_im = x_slice[i * incx].im;

            for j in 0..i {
                let ap_idx = lower_packed_index(i, j, n);
                let a_val = *ap.add(ap_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let ap_idx = lower_packed_index(i, i, n);
                let a_diag = *ap.add(ap_idx);
                let a_re = a_diag.re;
                let a_im = if conj_trans { -a_diag.im } else { a_diag.im };
                let denom = a_re * a_re + a_im * a_im;
                let new_re = (sum_re * a_re + sum_im * a_im) / denom;
                let new_im = (sum_im * a_re - sum_re * a_im) / denom;
                sum_re = new_re;
                sum_im = new_im;
            }
            x_slice[i * incx] = OblasComplex64 {
                re: sum_re,
                im: sum_im,
            };
        }
    }
}

// =============================================================================
// SSPR - Single precision symmetric packed rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^T + A for a symmetric packed matrix (single precision).
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sspr(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f32,
    x: *const f32,
    incx: c_int,
    ap: *mut f32,
) {
    if n <= 0 || ap.is_null() || x.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j = alpha * x_j;
            for i in 0..=j {
                let ap_idx = upper_packed_index(i, j);
                *ap.add(ap_idx) += alpha_x_j * x_slice[i * incx];
            }
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j = alpha * x_j;
            for i in j..n {
                let ap_idx = lower_packed_index(i, j, n);
                *ap.add(ap_idx) += alpha_x_j * x_slice[i * incx];
            }
        }
    }
}

// =============================================================================
// DSPR - Double precision symmetric packed rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^T + A for a symmetric packed matrix (double precision).
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dspr(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f64,
    x: *const f64,
    incx: c_int,
    ap: *mut f64,
) {
    if n <= 0 || ap.is_null() || x.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j = alpha * x_j;
            for i in 0..=j {
                let ap_idx = upper_packed_index(i, j);
                *ap.add(ap_idx) += alpha_x_j * x_slice[i * incx];
            }
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j = alpha * x_j;
            for i in j..n {
                let ap_idx = lower_packed_index(i, j, n);
                *ap.add(ap_idx) += alpha_x_j * x_slice[i * incx];
            }
        }
    }
}

// =============================================================================
// CHPR - Single precision complex Hermitian packed rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^H + A for a Hermitian packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_chpr(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f32,
    x: *const OblasComplex32,
    incx: c_int,
    ap: *mut OblasComplex32,
) {
    if n <= 0 || ap.is_null() || x.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j_re = alpha * x_j.re;
            let alpha_x_j_im = alpha * x_j.im;

            for i in 0..j {
                let ap_idx = upper_packed_index(i, j);
                let x_i = x_slice[i * incx];
                // A[i,j] += alpha * x[i] * conj(x[j])
                (*ap.add(ap_idx)).re += x_i.re * alpha_x_j_re + x_i.im * alpha_x_j_im;
                (*ap.add(ap_idx)).im += x_i.im * alpha_x_j_re - x_i.re * alpha_x_j_im;
            }

            // Diagonal (must be real)
            let ap_idx = upper_packed_index(j, j);
            (*ap.add(ap_idx)).re += alpha * (x_j.re * x_j.re + x_j.im * x_j.im);
            (*ap.add(ap_idx)).im = 0.0;
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j_re = alpha * x_j.re;
            let alpha_x_j_im = alpha * x_j.im;

            // Diagonal
            let ap_idx = lower_packed_index(j, j, n);
            (*ap.add(ap_idx)).re += alpha * (x_j.re * x_j.re + x_j.im * x_j.im);
            (*ap.add(ap_idx)).im = 0.0;

            for i in (j + 1)..n {
                let ap_idx = lower_packed_index(i, j, n);
                let x_i = x_slice[i * incx];
                // A[i,j] += alpha * x[i] * conj(x[j])
                (*ap.add(ap_idx)).re += x_i.re * alpha_x_j_re + x_i.im * alpha_x_j_im;
                (*ap.add(ap_idx)).im += x_i.im * alpha_x_j_re - x_i.re * alpha_x_j_im;
            }
        }
    }
}

// =============================================================================
// ZHPR - Double precision complex Hermitian packed rank-1 update
// =============================================================================

/// Performs A = alpha*x*x^H + A for a Hermitian packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zhpr(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f64,
    x: *const OblasComplex64,
    incx: c_int,
    ap: *mut OblasComplex64,
) {
    if n <= 0 || ap.is_null() || x.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j_re = alpha * x_j.re;
            let alpha_x_j_im = alpha * x_j.im;

            for i in 0..j {
                let ap_idx = upper_packed_index(i, j);
                let x_i = x_slice[i * incx];
                (*ap.add(ap_idx)).re += x_i.re * alpha_x_j_re + x_i.im * alpha_x_j_im;
                (*ap.add(ap_idx)).im += x_i.im * alpha_x_j_re - x_i.re * alpha_x_j_im;
            }

            let ap_idx = upper_packed_index(j, j);
            (*ap.add(ap_idx)).re += alpha * (x_j.re * x_j.re + x_j.im * x_j.im);
            (*ap.add(ap_idx)).im = 0.0;
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let alpha_x_j_re = alpha * x_j.re;
            let alpha_x_j_im = alpha * x_j.im;

            let ap_idx = lower_packed_index(j, j, n);
            (*ap.add(ap_idx)).re += alpha * (x_j.re * x_j.re + x_j.im * x_j.im);
            (*ap.add(ap_idx)).im = 0.0;

            for i in (j + 1)..n {
                let ap_idx = lower_packed_index(i, j, n);
                let x_i = x_slice[i * incx];
                (*ap.add(ap_idx)).re += x_i.re * alpha_x_j_re + x_i.im * alpha_x_j_im;
                (*ap.add(ap_idx)).im += x_i.im * alpha_x_j_re - x_i.re * alpha_x_j_im;
            }
        }
    }
}

// =============================================================================
// SSPR2 - Single precision symmetric packed rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^T + alpha*y*x^T + A for a symmetric packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sspr2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f32,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
    ap: *mut f32,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];
            for i in 0..=j {
                let ap_idx = upper_packed_index(i, j);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];
                *ap.add(ap_idx) += alpha * (x_i * y_j + y_i * x_j);
            }
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];
            for i in j..n {
                let ap_idx = lower_packed_index(i, j, n);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];
                *ap.add(ap_idx) += alpha * (x_i * y_j + y_i * x_j);
            }
        }
    }
}

// =============================================================================
// DSPR2 - Double precision symmetric packed rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^T + alpha*y*x^T + A for a symmetric packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dspr2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: f64,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    ap: *mut f64,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() || alpha == 0.0 {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];
            for i in 0..=j {
                let ap_idx = upper_packed_index(i, j);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];
                *ap.add(ap_idx) += alpha * (x_i * y_j + y_i * x_j);
            }
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];
            for i in j..n {
                let ap_idx = lower_packed_index(i, j, n);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];
                *ap.add(ap_idx) += alpha * (x_i * y_j + y_i * x_j);
            }
        }
    }
}

// =============================================================================
// CHPR2 - Single precision complex Hermitian packed rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^H + conj(alpha)*y*x^H + A for a Hermitian packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_chpr2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex32,
    x: *const OblasComplex32,
    incx: c_int,
    y: *const OblasComplex32,
    incy: c_int,
    ap: *mut OblasComplex32,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let alpha_zero = alpha.re == 0.0 && alpha.im == 0.0;
    if alpha_zero {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];

            // alpha * conj(y_j)
            let alpha_conj_y_re = alpha.re * y_j.re + alpha.im * y_j.im;
            let alpha_conj_y_im = alpha.im * y_j.re - alpha.re * y_j.im;
            // conj(alpha) * conj(x_j)
            let conj_alpha_conj_x_re = alpha.re * x_j.re + alpha.im * x_j.im;
            let conj_alpha_conj_x_im = -alpha.im * x_j.re + alpha.re * x_j.im;

            for i in 0..j {
                let ap_idx = upper_packed_index(i, j);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];

                // x_i * alpha * conj(y_j)
                let term1_re = x_i.re * alpha_conj_y_re - x_i.im * alpha_conj_y_im;
                let term1_im = x_i.re * alpha_conj_y_im + x_i.im * alpha_conj_y_re;
                // y_i * conj(alpha) * conj(x_j)
                let term2_re = y_i.re * conj_alpha_conj_x_re - y_i.im * conj_alpha_conj_x_im;
                let term2_im = y_i.re * conj_alpha_conj_x_im + y_i.im * conj_alpha_conj_x_re;

                (*ap.add(ap_idx)).re += term1_re + term2_re;
                (*ap.add(ap_idx)).im += term1_im + term2_im;
            }

            // Diagonal (result must be real)
            let ap_idx = upper_packed_index(j, j);
            let x_conj_y_re = x_j.re * y_j.re + x_j.im * y_j.im;
            let x_conj_y_im = x_j.im * y_j.re - x_j.re * y_j.im;
            let update_re = 2.0 * (alpha.re * x_conj_y_re - alpha.im * x_conj_y_im);
            (*ap.add(ap_idx)).re += update_re;
            (*ap.add(ap_idx)).im = 0.0;
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];

            let alpha_conj_y_re = alpha.re * y_j.re + alpha.im * y_j.im;
            let alpha_conj_y_im = alpha.im * y_j.re - alpha.re * y_j.im;
            let conj_alpha_conj_x_re = alpha.re * x_j.re + alpha.im * x_j.im;
            let conj_alpha_conj_x_im = -alpha.im * x_j.re + alpha.re * x_j.im;

            // Diagonal
            let ap_idx = lower_packed_index(j, j, n);
            let x_conj_y_re = x_j.re * y_j.re + x_j.im * y_j.im;
            let x_conj_y_im = x_j.im * y_j.re - x_j.re * y_j.im;
            let update_re = 2.0 * (alpha.re * x_conj_y_re - alpha.im * x_conj_y_im);
            (*ap.add(ap_idx)).re += update_re;
            (*ap.add(ap_idx)).im = 0.0;

            for i in (j + 1)..n {
                let ap_idx = lower_packed_index(i, j, n);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];

                let term1_re = x_i.re * alpha_conj_y_re - x_i.im * alpha_conj_y_im;
                let term1_im = x_i.re * alpha_conj_y_im + x_i.im * alpha_conj_y_re;
                let term2_re = y_i.re * conj_alpha_conj_x_re - y_i.im * conj_alpha_conj_x_im;
                let term2_im = y_i.re * conj_alpha_conj_x_im + y_i.im * conj_alpha_conj_x_re;

                (*ap.add(ap_idx)).re += term1_re + term2_re;
                (*ap.add(ap_idx)).im += term1_im + term2_im;
            }
        }
    }
}

// =============================================================================
// ZHPR2 - Double precision complex Hermitian packed rank-2 update
// =============================================================================

/// Performs A = alpha*x*y^H + conj(alpha)*y*x^H + A for a Hermitian packed matrix.
///
/// # Safety
/// - `ap` must point to a valid packed array
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zhpr2(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    alpha: OblasComplex64,
    x: *const OblasComplex64,
    incx: c_int,
    y: *const OblasComplex64,
    incy: c_int,
    ap: *mut OblasComplex64,
) {
    if n <= 0 || ap.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let alpha_zero = alpha.re == 0.0 && alpha.im == 0.0;
    if alpha_zero {
        return;
    }

    let n = n as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let x_slice = slice::from_raw_parts(x, n * incx);
    let y_slice = slice::from_raw_parts(y, n * incy);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let eff_upper = upper != row_major;

    if eff_upper {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];

            let alpha_conj_y_re = alpha.re * y_j.re + alpha.im * y_j.im;
            let alpha_conj_y_im = alpha.im * y_j.re - alpha.re * y_j.im;
            let conj_alpha_conj_x_re = alpha.re * x_j.re + alpha.im * x_j.im;
            let conj_alpha_conj_x_im = -alpha.im * x_j.re + alpha.re * x_j.im;

            for i in 0..j {
                let ap_idx = upper_packed_index(i, j);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];

                let term1_re = x_i.re * alpha_conj_y_re - x_i.im * alpha_conj_y_im;
                let term1_im = x_i.re * alpha_conj_y_im + x_i.im * alpha_conj_y_re;
                let term2_re = y_i.re * conj_alpha_conj_x_re - y_i.im * conj_alpha_conj_x_im;
                let term2_im = y_i.re * conj_alpha_conj_x_im + y_i.im * conj_alpha_conj_x_re;

                (*ap.add(ap_idx)).re += term1_re + term2_re;
                (*ap.add(ap_idx)).im += term1_im + term2_im;
            }

            let ap_idx = upper_packed_index(j, j);
            let x_conj_y_re = x_j.re * y_j.re + x_j.im * y_j.im;
            let x_conj_y_im = x_j.im * y_j.re - x_j.re * y_j.im;
            let update_re = 2.0 * (alpha.re * x_conj_y_re - alpha.im * x_conj_y_im);
            (*ap.add(ap_idx)).re += update_re;
            (*ap.add(ap_idx)).im = 0.0;
        }
    } else {
        for j in 0..n {
            let x_j = x_slice[j * incx];
            let y_j = y_slice[j * incy];

            let alpha_conj_y_re = alpha.re * y_j.re + alpha.im * y_j.im;
            let alpha_conj_y_im = alpha.im * y_j.re - alpha.re * y_j.im;
            let conj_alpha_conj_x_re = alpha.re * x_j.re + alpha.im * x_j.im;
            let conj_alpha_conj_x_im = -alpha.im * x_j.re + alpha.re * x_j.im;

            let ap_idx = lower_packed_index(j, j, n);
            let x_conj_y_re = x_j.re * y_j.re + x_j.im * y_j.im;
            let x_conj_y_im = x_j.im * y_j.re - x_j.re * y_j.im;
            let update_re = 2.0 * (alpha.re * x_conj_y_re - alpha.im * x_conj_y_im);
            (*ap.add(ap_idx)).re += update_re;
            (*ap.add(ap_idx)).im = 0.0;

            for i in (j + 1)..n {
                let ap_idx = lower_packed_index(i, j, n);
                let x_i = x_slice[i * incx];
                let y_i = y_slice[i * incy];

                let term1_re = x_i.re * alpha_conj_y_re - x_i.im * alpha_conj_y_im;
                let term1_im = x_i.re * alpha_conj_y_im + x_i.im * alpha_conj_y_re;
                let term2_re = y_i.re * conj_alpha_conj_x_re - y_i.im * conj_alpha_conj_x_im;
                let term2_im = y_i.re * conj_alpha_conj_x_im + y_i.im * conj_alpha_conj_x_re;

                (*ap.add(ap_idx)).re += term1_re + term2_re;
                (*ap.add(ap_idx)).im += term1_im + term2_im;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::blas2::{
        oblas_dgemv, oblas_dsymv, oblas_dtrmv, oblas_zgemv, oblas_zgerc, oblas_zhemv, oblas_zher,
        oblas_ztrsv,
    };

    use super::*;

    #[test]
    fn test_dgemv() {
        // A = [[1, 2], [3, 4], [5, 6]] (3x2 column-major)
        let a = [1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0]; // Column-major
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64, 0.0, 0.0];

        unsafe {
            oblas_dgemv(
                OblasLayout::ColMajor,
                OblasTranspose::NoTrans,
                3,
                2,
                1.0,
                a.as_ptr(),
                3,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }

        // y = A * x = [[1, 2], [3, 4], [5, 6]] * [1, 2] = [1+4, 3+8, 5+12] = [5, 11, 17]
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
        assert!((y[2] - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_zgemv() {
        // A = [[1+i, 2], [3, 4-i]] (2x2 row-major)
        // x = [1, 1]
        // y = A * x = [3+i, 7-i]
        let a = [
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
            OblasComplex64 { re: 4.0, im: -1.0 },
        ];
        let x = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
        ];
        let mut y = [OblasComplex64 { re: 0.0, im: 0.0 }; 2];

        unsafe {
            oblas_zgemv(
                OblasLayout::RowMajor,
                OblasTranspose::NoTrans,
                2,
                2,
                OblasComplex64 { re: 1.0, im: 0.0 },
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                OblasComplex64 { re: 0.0, im: 0.0 },
                y.as_mut_ptr(),
                1,
            );
        }

        assert!((y[0].re - 3.0).abs() < 1e-10);
        assert!((y[0].im - 1.0).abs() < 1e-10);
        assert!((y[1].re - 7.0).abs() < 1e-10);
        assert!((y[1].im - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_zhemv() {
        // A = [[2, 1+i], [1-i, 3]] Hermitian (upper stored, row-major)
        // x = [1, 1]
        // y = A * x = [3+i, 4-i]
        let a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 0.0, im: 0.0 }, // Not used (lower)
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];
        let x = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
        ];
        let mut y = [OblasComplex64 { re: 0.0, im: 0.0 }; 2];

        unsafe {
            oblas_zhemv(
                OblasLayout::RowMajor,
                OblasUplo::Upper,
                2,
                OblasComplex64 { re: 1.0, im: 0.0 },
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                OblasComplex64 { re: 0.0, im: 0.0 },
                y.as_mut_ptr(),
                1,
            );
        }

        // y[0] = 2*1 + (1+i)*1 = 3+i
        // y[1] = (1-i)*1 + 3*1 = 4-i
        assert!((y[0].re - 3.0).abs() < 1e-10);
        assert!((y[0].im - 1.0).abs() < 1e-10);
        assert!((y[1].re - 4.0).abs() < 1e-10);
        assert!((y[1].im - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dsymv() {
        // A = [[1, 2], [2, 3]] symmetric (upper stored)
        // x = [1, 1]
        // y = A * x = [3, 5]
        let a = [1.0f64, 2.0, 0.0, 3.0]; // Upper triangle
        let x = [1.0f64, 1.0];
        let mut y = [0.0f64; 2];

        unsafe {
            oblas_dsymv(
                OblasLayout::RowMajor,
                OblasUplo::Upper,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }

        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_zgerc() {
        // A = 0, x = [1+i, 2], y = [1, 1-i]
        // A = x * conj(y)^T = [[1+i, 2+2i], [2, 2+2i]]
        let mut a = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];
        let x = [
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];
        let y = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: -1.0 },
        ];

        unsafe {
            oblas_zgerc(
                OblasLayout::RowMajor,
                2,
                2,
                OblasComplex64 { re: 1.0, im: 0.0 },
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                2,
            );
        }

        // A[0,0] = (1+i)*conj(1) = 1+i
        assert!((a[0].re - 1.0).abs() < 1e-10);
        assert!((a[0].im - 1.0).abs() < 1e-10);
        // A[0,1] = (1+i)*conj(1-i) = (1+i)*(1+i) = 2i
        assert!((a[1].re - 0.0).abs() < 1e-10);
        assert!((a[1].im - 2.0).abs() < 1e-10);
        // A[1,0] = 2*conj(1) = 2
        assert!((a[2].re - 2.0).abs() < 1e-10);
        assert!((a[2].im - 0.0).abs() < 1e-10);
        // A[1,1] = 2*conj(1-i) = 2*(1+i) = 2+2i
        assert!((a[3].re - 2.0).abs() < 1e-10);
        assert!((a[3].im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_zher() {
        // A = 0 (Hermitian), x = [1+i, 2]
        // A = x*x^H = [[2, 2-2i], [2+2i, 4]]
        let mut a = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];
        let x = [
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];

        unsafe {
            oblas_zher(
                OblasLayout::RowMajor,
                OblasUplo::Upper,
                2,
                1.0,
                x.as_ptr(),
                1,
                a.as_mut_ptr(),
                2,
            );
        }

        // A[0,0] = |1+i|^2 = 2
        assert!((a[0].re - 2.0).abs() < 1e-10);
        assert!(a[0].im.abs() < 1e-10);
        // A[0,1] = (1+i)*conj(2) = 2+2i
        assert!((a[1].re - 2.0).abs() < 1e-10);
        assert!((a[1].im - 2.0).abs() < 1e-10);
        // A[1,1] = |2|^2 = 4
        assert!((a[3].re - 4.0).abs() < 1e-10);
        assert!(a[3].im.abs() < 1e-10);
    }

    #[test]
    fn test_dtrmv() {
        // A = [[2, 1], [0, 3]] upper triangular (row-major)
        // x = [1, 2]
        // A*x = [4, 6]
        let a = [2.0f64, 1.0, 0.0, 3.0];
        let mut x = [1.0f64, 2.0];

        unsafe {
            oblas_dtrmv(
                OblasLayout::RowMajor,
                OblasUplo::Upper,
                OblasTranspose::NoTrans,
                OblasDiag::NonUnit,
                2,
                a.as_ptr(),
                2,
                x.as_mut_ptr(),
                1,
            );
        }

        assert!((x[0] - 4.0).abs() < 1e-10);
        assert!((x[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ztrsv() {
        // A = [[2+i, 1], [0, 3-i]] upper triangular
        // Solve A*x = [3+i, 3-i]
        // x[1] = (3-i)/(3-i) = 1
        // x[0] = (3+i - 1)/(2+i) = (2+i)/(2+i) = 1
        let a = [
            OblasComplex64 { re: 2.0, im: 1.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 3.0, im: -1.0 },
        ];
        let mut x = [
            OblasComplex64 { re: 3.0, im: 1.0 },
            OblasComplex64 { re: 3.0, im: -1.0 },
        ];

        unsafe {
            oblas_ztrsv(
                OblasLayout::RowMajor,
                OblasUplo::Upper,
                OblasTranspose::NoTrans,
                OblasDiag::NonUnit,
                2,
                a.as_ptr(),
                2,
                x.as_mut_ptr(),
                1,
            );
        }

        assert!((x[0].re - 1.0).abs() < 1e-10);
        assert!(x[0].im.abs() < 1e-10);
        assert!((x[1].re - 1.0).abs() < 1e-10);
        assert!(x[1].im.abs() < 1e-10);
    }
}
