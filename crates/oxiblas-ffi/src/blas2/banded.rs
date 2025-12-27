//! BLAS Level 2 FFI - Matrix-Vector operations.

use crate::types::*;
use std::ffi::c_int;
use std::slice;

// SGBMV - Single precision general band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a general band matrix (single precision).
///
/// # Safety
/// - `a` must point to a valid band matrix of size `lda * n`
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgbmv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    alpha: f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: f32,
    y: *mut f32,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || kl < 0 || ku < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

    let y_slice = slice::from_raw_parts_mut(y, rows * incy);

    // Scale y by beta
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
    let row_major = layout == OblasLayout::RowMajor;

    // For band storage:
    // Column-major: element A[i,j] is at a[ku + i - j + j * lda]
    // Row-major: element A[i,j] is at a[kl + j - i + i * lda]

    match trans {
        OblasTranspose::NoTrans => {
            for i in 0..m {
                let j_start = i.saturating_sub(kl);
                let j_end = (i + ku + 1).min(n);
                let mut sum = 0.0f32;

                for j in j_start..j_end {
                    let a_idx = if row_major {
                        kl + j - i + i * lda
                    } else {
                        ku + i - j + j * lda
                    };
                    sum += *a.add(a_idx) * x_slice[j * incx];
                }
                y_slice[i * incy] += alpha * sum;
            }
        }
        _ => {
            // Transpose or ConjTrans (same for real)
            for j in 0..n {
                let alpha_xj = alpha * x_slice[j * incx];
                let i_start = j.saturating_sub(ku);
                let i_end = (j + kl + 1).min(m);

                for i in i_start..i_end {
                    let a_idx = if row_major {
                        kl + j - i + i * lda
                    } else {
                        ku + i - j + j * lda
                    };
                    y_slice[i * incy] += alpha_xj * *a.add(a_idx);
                }
            }
        }
    }
}

// =============================================================================
// DGBMV - Double precision general band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a general band matrix (double precision).
///
/// # Safety
/// - `a` must point to a valid band matrix of size `lda * n`
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgbmv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: f64,
    y: *mut f64,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || kl < 0 || ku < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

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
    let row_major = layout == OblasLayout::RowMajor;

    match trans {
        OblasTranspose::NoTrans => {
            for i in 0..m {
                let j_start = i.saturating_sub(kl);
                let j_end = (i + ku + 1).min(n);
                let mut sum = 0.0f64;

                for j in j_start..j_end {
                    let a_idx = if row_major {
                        kl + j - i + i * lda
                    } else {
                        ku + i - j + j * lda
                    };
                    sum += *a.add(a_idx) * x_slice[j * incx];
                }
                y_slice[i * incy] += alpha * sum;
            }
        }
        _ => {
            for j in 0..n {
                let alpha_xj = alpha * x_slice[j * incx];
                let i_start = j.saturating_sub(ku);
                let i_end = (j + kl + 1).min(m);

                for i in i_start..i_end {
                    let a_idx = if row_major {
                        kl + j - i + i * lda
                    } else {
                        ku + i - j + j * lda
                    };
                    y_slice[i * incy] += alpha_xj * *a.add(a_idx);
                }
            }
        }
    }
}

// =============================================================================
// CGBMV - Single precision complex general band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*op(A)*x + beta*y for a complex general band matrix.
///
/// # Safety
/// - `a` must point to a valid band matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgbmv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    x: *const OblasComplex32,
    incx: c_int,
    beta: OblasComplex32,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || kl < 0 || ku < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

    let y_slice = slice::from_raw_parts_mut(y, rows * incy);
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    if beta_zero {
        for i in 0..rows {
            y_slice[i * incy] = OblasComplex32 { re: 0.0, im: 0.0 };
        }
    } else if !beta_one {
        for i in 0..rows {
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

    let x_slice = slice::from_raw_parts(x, cols * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let do_trans = trans != OblasTranspose::NoTrans;

    if !do_trans {
        for i in 0..m {
            let j_start = i.saturating_sub(kl);
            let j_end = (i + ku + 1).min(n);
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;

            for j in j_start..j_end {
                let a_idx = if row_major {
                    kl + j - i + i * lda
                } else {
                    ku + i - j + j * lda
                };
                let a_val = *a.add(a_idx);
                let x_val = x_slice[j * incx];
                sum_re += a_val.re * x_val.re - a_val.im * x_val.im;
                sum_im += a_val.re * x_val.im + a_val.im * x_val.re;
            }
            let prod_re = alpha.re * sum_re - alpha.im * sum_im;
            let prod_im = alpha.re * sum_im + alpha.im * sum_re;
            y_slice[i * incy].re += prod_re;
            y_slice[i * incy].im += prod_im;
        }
    } else {
        for j in 0..n {
            let x_val = x_slice[j * incx];
            let ax_re = alpha.re * x_val.re - alpha.im * x_val.im;
            let ax_im = alpha.re * x_val.im + alpha.im * x_val.re;

            let i_start = j.saturating_sub(ku);
            let i_end = (j + kl + 1).min(m);

            for i in i_start..i_end {
                let a_idx = if row_major {
                    kl + j - i + i * lda
                } else {
                    ku + i - j + j * lda
                };
                let a_val = *a.add(a_idx);
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                y_slice[i * incy].re += a_re * ax_re - a_im * ax_im;
                y_slice[i * incy].im += a_re * ax_im + a_im * ax_re;
            }
        }
    }
}

// =============================================================================
// ZGBMV - Double precision complex general band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*op(A)*x + beta*y for a complex general band matrix.
///
/// # Safety
/// - `a` must point to a valid band matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgbmv(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    x: *const OblasComplex64,
    incx: c_int,
    beta: OblasComplex64,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if m <= 0 || n <= 0 || kl < 0 || ku < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;
    let incy = incy.max(1) as usize;

    let (rows, cols) = match trans {
        OblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };

    let y_slice = slice::from_raw_parts_mut(y, rows * incy);
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    if beta_zero {
        for i in 0..rows {
            y_slice[i * incy] = OblasComplex64 { re: 0.0, im: 0.0 };
        }
    } else if !beta_one {
        for i in 0..rows {
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

    let x_slice = slice::from_raw_parts(x, cols * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let conj_trans = trans == OblasTranspose::ConjTrans;
    let do_trans = trans != OblasTranspose::NoTrans;

    if !do_trans {
        for i in 0..m {
            let j_start = i.saturating_sub(kl);
            let j_end = (i + ku + 1).min(n);
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;

            for j in j_start..j_end {
                let a_idx = if row_major {
                    kl + j - i + i * lda
                } else {
                    ku + i - j + j * lda
                };
                let a_val = *a.add(a_idx);
                let x_val = x_slice[j * incx];
                sum_re += a_val.re * x_val.re - a_val.im * x_val.im;
                sum_im += a_val.re * x_val.im + a_val.im * x_val.re;
            }
            let prod_re = alpha.re * sum_re - alpha.im * sum_im;
            let prod_im = alpha.re * sum_im + alpha.im * sum_re;
            y_slice[i * incy].re += prod_re;
            y_slice[i * incy].im += prod_im;
        }
    } else {
        for j in 0..n {
            let x_val = x_slice[j * incx];
            let ax_re = alpha.re * x_val.re - alpha.im * x_val.im;
            let ax_im = alpha.re * x_val.im + alpha.im * x_val.re;

            let i_start = j.saturating_sub(ku);
            let i_end = (j + kl + 1).min(m);

            for i in i_start..i_end {
                let a_idx = if row_major {
                    kl + j - i + i * lda
                } else {
                    ku + i - j + j * lda
                };
                let a_val = *a.add(a_idx);
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                y_slice[i * incy].re += a_re * ax_re - a_im * ax_im;
                y_slice[i * incy].im += a_re * ax_im + a_im * ax_re;
            }
        }
    }
}

// =============================================================================
// SSBMV - Single precision symmetric band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a symmetric band matrix (single precision).
///
/// # Safety
/// - `a` must point to a valid symmetric band matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: f32,
    y: *mut f32,
    incy: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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

    // Effective upper storage after accounting for layout
    let eff_upper = upper != row_major;

    for i in 0..n {
        let mut sum = 0.0f32;

        if eff_upper {
            // Upper triangle stored: access elements A[i,j] for j >= i
            let j_end = (i + k + 1).min(n);
            for j in i..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
                if i == j {
                    sum += a_val * x_slice[j * incx];
                } else {
                    sum += a_val * x_slice[j * incx];
                    // Symmetric: also add contribution to y[j]
                    y_slice[j * incy] += alpha * a_val * x_slice[i * incx];
                }
            }
        } else {
            // Lower triangle stored
            let j_start = i.saturating_sub(k);
            for j in j_start..=i {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
                if i == j {
                    sum += a_val * x_slice[j * incx];
                } else {
                    sum += a_val * x_slice[j * incx];
                    y_slice[j * incy] += alpha * a_val * x_slice[i * incx];
                }
            }
        }
        y_slice[i * incy] += alpha * sum;
    }
}

// =============================================================================
// DSBMV - Double precision symmetric band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a symmetric band matrix (double precision).
///
/// # Safety
/// - `a` must point to a valid symmetric band matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: f64,
    y: *mut f64,
    incy: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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

        if eff_upper {
            let j_end = (i + k + 1).min(n);
            for j in i..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
                if i == j {
                    sum += a_val * x_slice[j * incx];
                } else {
                    sum += a_val * x_slice[j * incx];
                    y_slice[j * incy] += alpha * a_val * x_slice[i * incx];
                }
            }
        } else {
            let j_start = i.saturating_sub(k);
            for j in j_start..=i {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
                if i == j {
                    sum += a_val * x_slice[j * incx];
                } else {
                    sum += a_val * x_slice[j * incx];
                    y_slice[j * incy] += alpha * a_val * x_slice[i * incx];
                }
            }
        }
        y_slice[i * incy] += alpha * sum;
    }
}

// =============================================================================
// CHBMV - Single precision complex Hermitian band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a Hermitian band matrix.
///
/// # Safety
/// - `a` must point to a valid Hermitian band matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_chbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    k: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    x: *const OblasComplex32,
    incx: c_int,
    beta: OblasComplex32,
    y: *mut OblasComplex32,
    incy: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            // Diagonal element (real)
            let a_diag = *a.add(k + i * lda);
            sum_re += a_diag.re * x_i.re;
            sum_im += a_diag.re * x_i.im;

            // Off-diagonal elements
            let j_end = (i + k + 1).min(n);
            for j in (i + 1)..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];

                // A[i,j] * x[j]
                sum_re += a_val.re * x_j.re - a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im + a_val.im * x_j.re;

                // conj(A[i,j]) * x[i] added to y[j]
                let conj_a_x_re = a_val.re * x_i.re + a_val.im * x_i.im;
                let conj_a_x_im = a_val.re * x_i.im - a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * conj_a_x_re - alpha.im * conj_a_x_im;
                y_slice[j * incy].im += alpha.re * conj_a_x_im + alpha.im * conj_a_x_re;
            }
        } else {
            // Lower triangle stored
            let j_start = i.saturating_sub(k);
            for j in j_start..i {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];

                // conj(A[i,j]) * x[j] (lower stored means we access conj)
                sum_re += a_val.re * x_j.re + a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im - a_val.im * x_j.re;

                // A[i,j] * x[i] added to y[j]
                let a_x_re = a_val.re * x_i.re - a_val.im * x_i.im;
                let a_x_im = a_val.re * x_i.im + a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * a_x_re - alpha.im * a_x_im;
                y_slice[j * incy].im += alpha.re * a_x_im + alpha.im * a_x_re;
            }

            // Diagonal element (real)
            let a_diag = *a.add(i * lda);
            sum_re += a_diag.re * x_i.re;
            sum_im += a_diag.re * x_i.im;
        }

        let prod_re = alpha.re * sum_re - alpha.im * sum_im;
        let prod_im = alpha.re * sum_im + alpha.im * sum_re;
        y_slice[i * incy].re += prod_re;
        y_slice[i * incy].im += prod_im;
    }
}

// =============================================================================
// ZHBMV - Double precision complex Hermitian band matrix-vector multiply
// =============================================================================

/// Computes y = alpha*A*x + beta*y for a Hermitian band matrix.
///
/// # Safety
/// - `a` must point to a valid Hermitian band matrix
/// - `x` and `y` must point to valid vectors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zhbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    k: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    x: *const OblasComplex64,
    incx: c_int,
    beta: OblasComplex64,
    y: *mut OblasComplex64,
    incy: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            let a_diag = *a.add(k + i * lda);
            sum_re += a_diag.re * x_i.re;
            sum_im += a_diag.re * x_i.im;

            let j_end = (i + k + 1).min(n);
            for j in (i + 1)..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];

                sum_re += a_val.re * x_j.re - a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im + a_val.im * x_j.re;

                let conj_a_x_re = a_val.re * x_i.re + a_val.im * x_i.im;
                let conj_a_x_im = a_val.re * x_i.im - a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * conj_a_x_re - alpha.im * conj_a_x_im;
                y_slice[j * incy].im += alpha.re * conj_a_x_im + alpha.im * conj_a_x_re;
            }
        } else {
            let j_start = i.saturating_sub(k);
            for j in j_start..i {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];

                sum_re += a_val.re * x_j.re + a_val.im * x_j.im;
                sum_im += a_val.re * x_j.im - a_val.im * x_j.re;

                let a_x_re = a_val.re * x_i.re - a_val.im * x_i.im;
                let a_x_im = a_val.re * x_i.im + a_val.im * x_i.re;
                y_slice[j * incy].re += alpha.re * a_x_re - alpha.im * a_x_im;
                y_slice[j * incy].im += alpha.re * a_x_im + alpha.im * a_x_re;
            }

            let a_diag = *a.add(i * lda);
            sum_re += a_diag.re * x_i.re;
            sum_im += a_diag.re * x_i.im;
        }

        let prod_re = alpha.re * sum_re - alpha.im * sum_im;
        let prod_im = alpha.re * sum_im + alpha.im * sum_re;
        y_slice[i * incy].re += prod_re;
        y_slice[i * incy].im += prod_im;
    }
}

// =============================================================================
// STBMV - Single precision triangular band matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a triangular band matrix (single precision).
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_stbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        // Process from end to start for upper
        for i in (0..n).rev() {
            let x_i = x_slice[i * incx];
            let mut sum = if unit_diag { x_i } else { 0.0f32 };

            let j_start = if unit_diag { i + 1 } else { i };
            let j_end = (i + k + 1).min(n);

            for j in j_start..j_end {
                let a_idx = k + i - j + j * lda;
                sum += *a.add(a_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    } else {
        // Process from start to end for lower
        for i in 0..n {
            let x_i = x_slice[i * incx];
            let mut sum = if unit_diag { x_i } else { 0.0f32 };

            let j_start = i.saturating_sub(k);
            let j_end = if unit_diag { i } else { i + 1 };

            for j in j_start..j_end {
                let a_idx = i - j + j * lda;
                sum += *a.add(a_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// DTBMV - Double precision triangular band matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a triangular band matrix (double precision).
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            let j_end = (i + k + 1).min(n);

            for j in j_start..j_end {
                let a_idx = k + i - j + j * lda;
                sum += *a.add(a_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    } else {
        for i in 0..n {
            let x_i = x_slice[i * incx];
            let mut sum = if unit_diag { x_i } else { 0.0f64 };

            let j_start = i.saturating_sub(k);
            let j_end = if unit_diag { i } else { i + 1 };

            for j in j_start..j_end {
                let a_idx = i - j + j * lda;
                sum += *a.add(a_idx) * x_slice[j * incx];
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// CTBMV - Single precision complex triangular band matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a complex triangular band matrix.
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const OblasComplex32,
    lda: c_int,
    x: *mut OblasComplex32,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            let j_end = (i + k + 1).min(n);

            for j in j_start..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
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

            let j_start = i.saturating_sub(k);
            let j_end = if unit_diag { i } else { i + 1 };

            for j in j_start..j_end {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
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
// ZTBMV - Double precision complex triangular band matrix-vector multiply
// =============================================================================

/// Computes x = op(A)*x for a complex triangular band matrix.
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztbmv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const OblasComplex64,
    lda: c_int,
    x: *mut OblasComplex64,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            let j_end = (i + k + 1).min(n);

            for j in j_start..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
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

            let j_start = i.saturating_sub(k);
            let j_end = if unit_diag { i } else { i + 1 };

            for j in j_start..j_end {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
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
// STBSV - Single precision triangular band solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a triangular band matrix (single precision).
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector (input b, output x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_stbsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let incx = incx.max(1) as usize;

    let x_slice = slice::from_raw_parts_mut(x, n * incx);
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let do_trans = trans != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let eff_upper = upper != (row_major != do_trans);

    if eff_upper {
        // Back substitution for upper triangular
        for i in (0..n).rev() {
            let mut sum = x_slice[i * incx];
            let j_end = (i + k + 1).min(n);

            for j in (i + 1)..j_end {
                let a_idx = k + i - j + j * lda;
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let a_diag = *a.add(k + i * lda);
                sum /= a_diag;
            }
            x_slice[i * incx] = sum;
        }
    } else {
        // Forward substitution for lower triangular
        for i in 0..n {
            let mut sum = x_slice[i * incx];
            let j_start = i.saturating_sub(k);

            for j in j_start..i {
                let a_idx = i - j + j * lda;
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let a_diag = *a.add(i * lda);
                sum /= a_diag;
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// DTBSV - Double precision triangular band solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a triangular band matrix (double precision).
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector (input b, output x)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtbsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            let j_end = (i + k + 1).min(n);

            for j in (i + 1)..j_end {
                let a_idx = k + i - j + j * lda;
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let a_diag = *a.add(k + i * lda);
                sum /= a_diag;
            }
            x_slice[i * incx] = sum;
        }
    } else {
        for i in 0..n {
            let mut sum = x_slice[i * incx];
            let j_start = i.saturating_sub(k);

            for j in j_start..i {
                let a_idx = i - j + j * lda;
                sum -= *a.add(a_idx) * x_slice[j * incx];
            }

            if !unit_diag {
                let a_diag = *a.add(i * lda);
                sum /= a_diag;
            }
            x_slice[i * incx] = sum;
        }
    }
}

// =============================================================================
// CTBSV - Single precision complex triangular band solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a complex triangular band matrix.
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctbsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const OblasComplex32,
    lda: c_int,
    x: *mut OblasComplex32,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            let j_end = (i + k + 1).min(n);

            for j in (i + 1)..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let a_diag = *a.add(k + i * lda);
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
            let j_start = i.saturating_sub(k);

            for j in j_start..i {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let a_diag = *a.add(i * lda);
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
// ZTBSV - Double precision complex triangular band solve
// =============================================================================

/// Solves op(A)*x = b for x where A is a complex triangular band matrix.
///
/// # Safety
/// - `a` must point to a valid triangular band matrix
/// - `x` must point to a valid vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztbsv(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    diag: OblasDiag,
    n: c_int,
    k: c_int,
    a: *const OblasComplex64,
    lda: c_int,
    x: *mut OblasComplex64,
    incx: c_int,
) {
    if n <= 0 || k < 0 || a.is_null() || x.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
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
            let j_end = (i + k + 1).min(n);

            for j in (i + 1)..j_end {
                let a_idx = k + i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let a_diag = *a.add(k + i * lda);
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
            let j_start = i.saturating_sub(k);

            for j in j_start..i {
                let a_idx = i - j + j * lda;
                let a_val = *a.add(a_idx);
                let x_j = x_slice[j * incx];
                let a_re = a_val.re;
                let a_im = if conj_trans { -a_val.im } else { a_val.im };
                sum_re -= a_re * x_j.re - a_im * x_j.im;
                sum_im -= a_re * x_j.im + a_im * x_j.re;
            }

            if !unit_diag {
                let a_diag = *a.add(i * lda);
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
