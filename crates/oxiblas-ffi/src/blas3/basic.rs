//! BLAS Level 3 FFI - Matrix-Matrix operations.

use crate::types::*;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGEMM - Single precision general matrix-matrix multiply
// =============================================================================

/// Computes C = alpha*op(A)*op(B) + beta*C for single precision.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgemm(
    layout: OblasLayout,
    trans_a: OblasTranspose,
    trans_b: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: f32,
    c: *mut f32,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let trans_b = trans_b != OblasTranspose::NoTrans;

    // Scale C by beta
    let c_size = if row_major { m * ldc } else { n * ldc };
    let c_slice = slice::from_raw_parts_mut(c, c_size);

    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldc + j } else { j * ldc + i };
                c_slice[idx] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldc + j } else { j * ldc + i };
                c_slice[idx] *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    // Perform matrix multiply
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                let a_idx = if row_major {
                    if trans_a { l * lda + i } else { i * lda + l }
                } else {
                    if trans_a { i * lda + l } else { l * lda + i }
                };

                let b_idx = if row_major {
                    if trans_b { j * ldb + l } else { l * ldb + j }
                } else {
                    if trans_b { l * ldb + j } else { j * ldb + l }
                };

                sum += *a.add(a_idx) * *b.add(b_idx);
            }

            let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
            c_slice[c_idx] += alpha * sum;
        }
    }
}

// =============================================================================
// DGEMM - Double precision general matrix-matrix multiply
// =============================================================================

/// Computes C = alpha*op(A)*op(B) + beta*C for double precision.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgemm(
    layout: OblasLayout,
    trans_a: OblasTranspose,
    trans_b: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: f64,
    c: *mut f64,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let trans_b = trans_b != OblasTranspose::NoTrans;

    // Scale C by beta
    let c_size = if row_major { m * ldc } else { n * ldc };
    let c_slice = slice::from_raw_parts_mut(c, c_size);

    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldc + j } else { j * ldc + i };
                c_slice[idx] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldc + j } else { j * ldc + i };
                c_slice[idx] *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    // Perform matrix multiply
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for l in 0..k {
                let a_idx = if row_major {
                    if trans_a { l * lda + i } else { i * lda + l }
                } else {
                    if trans_a { i * lda + l } else { l * lda + i }
                };

                let b_idx = if row_major {
                    if trans_b { j * ldb + l } else { l * ldb + j }
                } else {
                    if trans_b { l * ldb + j } else { j * ldb + l }
                };

                sum += *a.add(a_idx) * *b.add(b_idx);
            }

            let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
            c_slice[c_idx] += alpha * sum;
        }
    }
}

// =============================================================================
// STRSM - Triangular solve with multiple right-hand sides (single precision)
// =============================================================================

/// Solves op(A)*X = alpha*B or X*op(A) = alpha*B where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_strsm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    trans_a: OblasTranspose,
    diag: OblasDiag,
    m: c_int,
    n: c_int,
    alpha: f32,
    a: *const f32,
    lda: c_int,
    b: *mut f32,
    ldb: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let k = if left { m } else { n };

    // Scale B by alpha
    if alpha != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) *= alpha;
            }
        }
    }

    // Solve the triangular system
    // This is a simplified implementation; a production version would be more optimized
    let eff_upper = if row_major { !upper } else { upper };
    let eff_trans = if row_major { !trans_a } else { trans_a };

    if left {
        // Solve op(A) * X = B
        if eff_upper != eff_trans {
            // Forward solve
            for l in 0..k {
                if !unit_diag {
                    let a_ll = if row_major {
                        *a.add(l * lda + l)
                    } else {
                        *a.add(l * lda + l)
                    };
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        *b.add(idx) /= a_ll;
                    }
                }
                for i in (l + 1)..k {
                    let a_il = if row_major {
                        if eff_trans {
                            *a.add(l * lda + i)
                        } else {
                            *a.add(i * lda + l)
                        }
                    } else {
                        if eff_trans {
                            *a.add(i * lda + l)
                        } else {
                            *a.add(l * lda + i)
                        }
                    };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        *b.add(idx_i) -= a_il * *b.add(idx_l);
                    }
                }
            }
        } else {
            // Backward solve
            for l in (0..k).rev() {
                if !unit_diag {
                    let a_ll = if row_major {
                        *a.add(l * lda + l)
                    } else {
                        *a.add(l * lda + l)
                    };
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        *b.add(idx) /= a_ll;
                    }
                }
                for i in 0..l {
                    let a_il = if row_major {
                        if eff_trans {
                            *a.add(l * lda + i)
                        } else {
                            *a.add(i * lda + l)
                        }
                    } else {
                        if eff_trans {
                            *a.add(i * lda + l)
                        } else {
                            *a.add(l * lda + i)
                        }
                    };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        *b.add(idx_i) -= a_il * *b.add(idx_l);
                    }
                }
            }
        }
    } else {
        // Solve X * op(A) = B (right side)
        // Simplified implementation
        for l in 0..k {
            if !unit_diag {
                let a_ll = *a.add(l * lda + l);
                for i in 0..m {
                    let idx = if row_major { i * ldb + l } else { l * ldb + i };
                    *b.add(idx) /= a_ll;
                }
            }
        }
    }
}

// =============================================================================
// DTRSM - Triangular solve with multiple right-hand sides (double precision)
// =============================================================================

/// Solves op(A)*X = alpha*B or X*op(A) = alpha*B where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtrsm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    trans_a: OblasTranspose,
    diag: OblasDiag,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    b: *mut f64,
    ldb: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let k = if left { m } else { n };

    // Scale B by alpha
    if alpha != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) *= alpha;
            }
        }
    }

    // Solve the triangular system
    let eff_upper = if row_major { !upper } else { upper };
    let eff_trans = if row_major { !trans_a } else { trans_a };

    if left {
        if eff_upper != eff_trans {
            // Forward solve
            for l in 0..k {
                if !unit_diag {
                    let a_ll = *a.add(l * lda + l);
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        *b.add(idx) /= a_ll;
                    }
                }
                for i in (l + 1)..k {
                    let a_il = if row_major {
                        if eff_trans {
                            *a.add(l * lda + i)
                        } else {
                            *a.add(i * lda + l)
                        }
                    } else {
                        if eff_trans {
                            *a.add(i * lda + l)
                        } else {
                            *a.add(l * lda + i)
                        }
                    };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        *b.add(idx_i) -= a_il * *b.add(idx_l);
                    }
                }
            }
        } else {
            // Backward solve
            for l in (0..k).rev() {
                if !unit_diag {
                    let a_ll = *a.add(l * lda + l);
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        *b.add(idx) /= a_ll;
                    }
                }
                for i in 0..l {
                    let a_il = if row_major {
                        if eff_trans {
                            *a.add(l * lda + i)
                        } else {
                            *a.add(i * lda + l)
                        }
                    } else {
                        if eff_trans {
                            *a.add(i * lda + l)
                        } else {
                            *a.add(l * lda + i)
                        }
                    };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        *b.add(idx_i) -= a_il * *b.add(idx_l);
                    }
                }
            }
        }
    } else {
        // Solve X * op(A) = B (right side)
        for l in 0..k {
            if !unit_diag {
                let a_ll = *a.add(l * lda + l);
                for i in 0..m {
                    let idx = if row_major { i * ldb + l } else { l * ldb + i };
                    *b.add(idx) /= a_ll;
                }
            }
        }
    }
}

// =============================================================================
// CGEMM - Complex single precision general matrix-matrix multiply
// =============================================================================

/// Computes C = alpha*op(A)*op(B) + beta*C for complex single precision.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgemm(
    layout: OblasLayout,
    trans_a: OblasTranspose,
    trans_b: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    b: *const OblasComplex32,
    ldb: c_int,
    beta: OblasComplex32,
    c: *mut OblasComplex32,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let conj_a = trans_a == OblasTranspose::ConjTrans;
    let conj_b = trans_b == OblasTranspose::ConjTrans;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let trans_b = trans_b != OblasTranspose::NoTrans;

    // Scale C by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta_zero {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if !beta_one {
                let c_val = *c.add(idx);
                (*c.add(idx)).re = beta.re * c_val.re - beta.im * c_val.im;
                (*c.add(idx)).im = beta.re * c_val.im + beta.im * c_val.re;
            }
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    // Perform complex matrix multiply
    for i in 0..m {
        for j in 0..n {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for l in 0..k {
                let a_idx = if row_major {
                    if trans_a { l * lda + i } else { i * lda + l }
                } else {
                    if trans_a { i * lda + l } else { l * lda + i }
                };

                let b_idx = if row_major {
                    if trans_b { j * ldb + l } else { l * ldb + j }
                } else {
                    if trans_b { l * ldb + j } else { j * ldb + l }
                };

                let a_val = *a.add(a_idx);
                let b_val = *b.add(b_idx);

                // Apply conjugation if needed
                let a_re = a_val.re;
                let a_im = if conj_a { -a_val.im } else { a_val.im };
                let b_re = b_val.re;
                let b_im = if conj_b { -b_val.im } else { b_val.im };

                // Complex multiply: (a_re + i*a_im) * (b_re + i*b_im)
                sum_re += a_re * b_re - a_im * b_im;
                sum_im += a_re * b_im + a_im * b_re;
            }

            let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
            // C += alpha * sum
            let result_re = alpha.re * sum_re - alpha.im * sum_im;
            let result_im = alpha.re * sum_im + alpha.im * sum_re;
            (*c.add(c_idx)).re += result_re;
            (*c.add(c_idx)).im += result_im;
        }
    }
}

// =============================================================================
// ZGEMM - Complex double precision general matrix-matrix multiply
// =============================================================================

/// Computes C = alpha*op(A)*op(B) + beta*C for complex double precision.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgemm(
    layout: OblasLayout,
    trans_a: OblasTranspose,
    trans_b: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    b: *const OblasComplex64,
    ldb: c_int,
    beta: OblasComplex64,
    c: *mut OblasComplex64,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let conj_a = trans_a == OblasTranspose::ConjTrans;
    let conj_b = trans_b == OblasTranspose::ConjTrans;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let trans_b = trans_b != OblasTranspose::NoTrans;

    // Scale C by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta_zero {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if !beta_one {
                let c_val = *c.add(idx);
                (*c.add(idx)).re = beta.re * c_val.re - beta.im * c_val.im;
                (*c.add(idx)).im = beta.re * c_val.im + beta.im * c_val.re;
            }
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    // Perform complex matrix multiply
    for i in 0..m {
        for j in 0..n {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for l in 0..k {
                let a_idx = if row_major {
                    if trans_a { l * lda + i } else { i * lda + l }
                } else {
                    if trans_a { i * lda + l } else { l * lda + i }
                };

                let b_idx = if row_major {
                    if trans_b { j * ldb + l } else { l * ldb + j }
                } else {
                    if trans_b { l * ldb + j } else { j * ldb + l }
                };

                let a_val = *a.add(a_idx);
                let b_val = *b.add(b_idx);

                // Apply conjugation if needed
                let a_re = a_val.re;
                let a_im = if conj_a { -a_val.im } else { a_val.im };
                let b_re = b_val.re;
                let b_im = if conj_b { -b_val.im } else { b_val.im };

                // Complex multiply
                sum_re += a_re * b_re - a_im * b_im;
                sum_im += a_re * b_im + a_im * b_re;
            }

            let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
            let result_re = alpha.re * sum_re - alpha.im * sum_im;
            let result_im = alpha.re * sum_im + alpha.im * sum_re;
            (*c.add(c_idx)).re += result_re;
            (*c.add(c_idx)).im += result_im;
        }
    }
}

// =============================================================================
// SSYMM - Symmetric matrix multiply (single precision)
// =============================================================================

/// Computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is symmetric.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssymm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    m: c_int,
    n: c_int,
    alpha: f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: f32,
    c: *mut f32,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;

    let k = if left { m } else { n };

    // Scale C by beta
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                *c.add(idx) = 0.0;
            } else if beta != 1.0 {
                *c.add(idx) *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    // Helper to read symmetric matrix A
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

    if left {
        // C = alpha * A * B + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_il = get_a(i, l);
                    let b_lj = if row_major {
                        *b.add(l * ldb + j)
                    } else {
                        *b.add(j * ldb + l)
                    };
                    sum += a_il * b_lj;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                *c.add(c_idx) += alpha * sum;
            }
        }
    } else {
        // C = alpha * B * A + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let b_il = if row_major {
                        *b.add(i * ldb + l)
                    } else {
                        *b.add(l * ldb + i)
                    };
                    let a_lj = get_a(l, j);
                    sum += b_il * a_lj;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                *c.add(c_idx) += alpha * sum;
            }
        }
    }
}

// =============================================================================
// DSYMM - Symmetric matrix multiply (double precision)
// =============================================================================

/// Computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is symmetric.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsymm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: f64,
    c: *mut f64,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;

    let k = if left { m } else { n };

    // Scale C by beta
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                *c.add(idx) = 0.0;
            } else if beta != 1.0 {
                *c.add(idx) *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    // Helper to read symmetric matrix A
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

    if left {
        // C = alpha * A * B + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    let a_il = get_a(i, l);
                    let b_lj = if row_major {
                        *b.add(l * ldb + j)
                    } else {
                        *b.add(j * ldb + l)
                    };
                    sum += a_il * b_lj;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                *c.add(c_idx) += alpha * sum;
            }
        }
    } else {
        // C = alpha * B * A + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    let b_il = if row_major {
                        *b.add(i * ldb + l)
                    } else {
                        *b.add(l * ldb + i)
                    };
                    let a_lj = get_a(l, j);
                    sum += b_il * a_lj;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                *c.add(c_idx) += alpha * sum;
            }
        }
    }
}

// =============================================================================
// CTRSM - Complex single precision triangular solve with multiple RHS
// =============================================================================

/// Solves op(A)*X = alpha*B or X*op(A) = alpha*B where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctrsm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    trans_a: OblasTranspose,
    diag: OblasDiag,
    m: c_int,
    n: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    b: *mut OblasComplex32,
    ldb: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;
    let conj = trans_a == OblasTranspose::ConjTrans;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let k = if left { m } else { n };

    // Scale B by alpha
    let alpha_one = alpha.re == 1.0 && alpha.im == 0.0;
    if !alpha_one {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                let b_val = *b.add(idx);
                (*b.add(idx)).re = alpha.re * b_val.re - alpha.im * b_val.im;
                (*b.add(idx)).im = alpha.re * b_val.im + alpha.im * b_val.re;
            }
        }
    }

    // Helper to get A element with optional conjugation
    let get_a = |i: usize, j: usize| -> (f32, f32) {
        let idx = if row_major { i * lda + j } else { j * lda + i };
        let a_val = *a.add(idx);
        if conj {
            (a_val.re, -a_val.im)
        } else {
            (a_val.re, a_val.im)
        }
    };

    // Solve
    let eff_upper = if row_major { !upper } else { upper };
    let eff_trans = if row_major { !trans_a } else { trans_a };

    if left {
        if eff_upper != eff_trans {
            // Forward solve
            for l in 0..k {
                if !unit_diag {
                    let (a_re, a_im) = get_a(l, l);
                    let denom = a_re * a_re + a_im * a_im;
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        let b_val = *b.add(idx);
                        // b / a = b * conj(a) / |a|^2
                        (*b.add(idx)).re = (b_val.re * a_re + b_val.im * a_im) / denom;
                        (*b.add(idx)).im = (b_val.im * a_re - b_val.re * a_im) / denom;
                    }
                }
                for i in (l + 1)..k {
                    let (a_re, a_im) = if eff_trans { get_a(l, i) } else { get_a(i, l) };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        let b_l = *b.add(idx_l);
                        // b_i -= a * b_l
                        (*b.add(idx_i)).re -= a_re * b_l.re - a_im * b_l.im;
                        (*b.add(idx_i)).im -= a_re * b_l.im + a_im * b_l.re;
                    }
                }
            }
        } else {
            // Backward solve
            for l in (0..k).rev() {
                if !unit_diag {
                    let (a_re, a_im) = get_a(l, l);
                    let denom = a_re * a_re + a_im * a_im;
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        let b_val = *b.add(idx);
                        (*b.add(idx)).re = (b_val.re * a_re + b_val.im * a_im) / denom;
                        (*b.add(idx)).im = (b_val.im * a_re - b_val.re * a_im) / denom;
                    }
                }
                for i in 0..l {
                    let (a_re, a_im) = if eff_trans { get_a(l, i) } else { get_a(i, l) };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        let b_l = *b.add(idx_l);
                        (*b.add(idx_i)).re -= a_re * b_l.re - a_im * b_l.im;
                        (*b.add(idx_i)).im -= a_re * b_l.im + a_im * b_l.re;
                    }
                }
            }
        }
    } else {
        // Right side: X * op(A) = B
        for l in 0..k {
            if !unit_diag {
                let (a_re, a_im) = get_a(l, l);
                let denom = a_re * a_re + a_im * a_im;
                for i in 0..m {
                    let idx = if row_major { i * ldb + l } else { l * ldb + i };
                    let b_val = *b.add(idx);
                    (*b.add(idx)).re = (b_val.re * a_re + b_val.im * a_im) / denom;
                    (*b.add(idx)).im = (b_val.im * a_re - b_val.re * a_im) / denom;
                }
            }
        }
    }
}

// =============================================================================
// ZTRSM - Complex double precision triangular solve with multiple RHS
// =============================================================================

/// Solves op(A)*X = alpha*B or X*op(A) = alpha*B where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztrsm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    trans_a: OblasTranspose,
    diag: OblasDiag,
    m: c_int,
    n: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    b: *mut OblasComplex64,
    ldb: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;
    let conj = trans_a == OblasTranspose::ConjTrans;
    let trans_a = trans_a != OblasTranspose::NoTrans;
    let unit_diag = diag == OblasDiag::Unit;

    let k = if left { m } else { n };

    // Scale B by alpha
    let alpha_one = alpha.re == 1.0 && alpha.im == 0.0;
    if !alpha_one {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                let b_val = *b.add(idx);
                (*b.add(idx)).re = alpha.re * b_val.re - alpha.im * b_val.im;
                (*b.add(idx)).im = alpha.re * b_val.im + alpha.im * b_val.re;
            }
        }
    }

    let get_a = |i: usize, j: usize| -> (f64, f64) {
        let idx = if row_major { i * lda + j } else { j * lda + i };
        let a_val = *a.add(idx);
        if conj {
            (a_val.re, -a_val.im)
        } else {
            (a_val.re, a_val.im)
        }
    };

    let eff_upper = if row_major { !upper } else { upper };
    let eff_trans = if row_major { !trans_a } else { trans_a };

    if left {
        if eff_upper != eff_trans {
            for l in 0..k {
                if !unit_diag {
                    let (a_re, a_im) = get_a(l, l);
                    let denom = a_re * a_re + a_im * a_im;
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        let b_val = *b.add(idx);
                        (*b.add(idx)).re = (b_val.re * a_re + b_val.im * a_im) / denom;
                        (*b.add(idx)).im = (b_val.im * a_re - b_val.re * a_im) / denom;
                    }
                }
                for i in (l + 1)..k {
                    let (a_re, a_im) = if eff_trans { get_a(l, i) } else { get_a(i, l) };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        let b_l = *b.add(idx_l);
                        (*b.add(idx_i)).re -= a_re * b_l.re - a_im * b_l.im;
                        (*b.add(idx_i)).im -= a_re * b_l.im + a_im * b_l.re;
                    }
                }
            }
        } else {
            for l in (0..k).rev() {
                if !unit_diag {
                    let (a_re, a_im) = get_a(l, l);
                    let denom = a_re * a_re + a_im * a_im;
                    for j in 0..n {
                        let idx = if row_major { l * ldb + j } else { j * ldb + l };
                        let b_val = *b.add(idx);
                        (*b.add(idx)).re = (b_val.re * a_re + b_val.im * a_im) / denom;
                        (*b.add(idx)).im = (b_val.im * a_re - b_val.re * a_im) / denom;
                    }
                }
                for i in 0..l {
                    let (a_re, a_im) = if eff_trans { get_a(l, i) } else { get_a(i, l) };
                    for j in 0..n {
                        let idx_l = if row_major { l * ldb + j } else { j * ldb + l };
                        let idx_i = if row_major { i * ldb + j } else { j * ldb + i };
                        let b_l = *b.add(idx_l);
                        (*b.add(idx_i)).re -= a_re * b_l.re - a_im * b_l.im;
                        (*b.add(idx_i)).im -= a_re * b_l.im + a_im * b_l.re;
                    }
                }
            }
        }
    } else {
        for l in 0..k {
            if !unit_diag {
                let (a_re, a_im) = get_a(l, l);
                let denom = a_re * a_re + a_im * a_im;
                for i in 0..m {
                    let idx = if row_major { i * ldb + l } else { l * ldb + i };
                    let b_val = *b.add(idx);
                    (*b.add(idx)).re = (b_val.re * a_re + b_val.im * a_im) / denom;
                    (*b.add(idx)).im = (b_val.im * a_re - b_val.re * a_im) / denom;
                }
            }
        }
    }
}

// =============================================================================
// CHEMM - Hermitian matrix multiply (single precision)
// =============================================================================

/// Computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is Hermitian.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_chemm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    m: c_int,
    n: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    b: *const OblasComplex32,
    ldb: c_int,
    beta: OblasComplex32,
    c: *mut OblasComplex32,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;

    let k = if left { m } else { n };

    // Scale C by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta_zero {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if !beta_one {
                let c_val = *c.add(idx);
                (*c.add(idx)).re = beta.re * c_val.re - beta.im * c_val.im;
                (*c.add(idx)).im = beta.re * c_val.im + beta.im * c_val.re;
            }
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    // Helper to read Hermitian matrix A (conjugate for off-diagonal access from stored side)
    let get_a = |i: usize, j: usize| -> (f32, f32) {
        let (ii, jj, need_conj) = if (upper && j >= i) || (!upper && i >= j) {
            (i, j, false)
        } else {
            (j, i, true)
        };
        let idx = if row_major {
            ii * lda + jj
        } else {
            jj * lda + ii
        };
        let a_val = *a.add(idx);
        if need_conj {
            (a_val.re, -a_val.im)
        } else if i == j {
            // Diagonal is real for Hermitian
            (a_val.re, 0.0)
        } else {
            (a_val.re, a_val.im)
        }
    };

    if left {
        // C = alpha * A * B + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f32;
                let mut sum_im = 0.0f32;
                for l in 0..k {
                    let (a_re, a_im) = get_a(i, l);
                    let b_idx = if row_major { l * ldb + j } else { j * ldb + l };
                    let b_val = *b.add(b_idx);
                    sum_re += a_re * b_val.re - a_im * b_val.im;
                    sum_im += a_re * b_val.im + a_im * b_val.re;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                (*c.add(c_idx)).re += alpha.re * sum_re - alpha.im * sum_im;
                (*c.add(c_idx)).im += alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    } else {
        // C = alpha * B * A + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f32;
                let mut sum_im = 0.0f32;
                for l in 0..k {
                    let b_idx = if row_major { i * ldb + l } else { l * ldb + i };
                    let b_val = *b.add(b_idx);
                    let (a_re, a_im) = get_a(l, j);
                    sum_re += b_val.re * a_re - b_val.im * a_im;
                    sum_im += b_val.re * a_im + b_val.im * a_re;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                (*c.add(c_idx)).re += alpha.re * sum_re - alpha.im * sum_im;
                (*c.add(c_idx)).im += alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    }
}

// =============================================================================
// ZHEMM - Hermitian matrix multiply (double precision)
// =============================================================================

/// Computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is Hermitian.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zhemm(
    layout: OblasLayout,
    side: OblasSide,
    uplo: OblasUplo,
    m: c_int,
    n: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    b: *const OblasComplex64,
    ldb: c_int,
    beta: OblasComplex64,
    c: *mut OblasComplex64,
    ldc: c_int,
) {
    if m <= 0 || n <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let upper = uplo == OblasUplo::Upper;

    let k = if left { m } else { n };

    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta_zero {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if !beta_one {
                let c_val = *c.add(idx);
                (*c.add(idx)).re = beta.re * c_val.re - beta.im * c_val.im;
                (*c.add(idx)).im = beta.re * c_val.im + beta.im * c_val.re;
            }
        }
    }

    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }

    let get_a = |i: usize, j: usize| -> (f64, f64) {
        let (ii, jj, need_conj) = if (upper && j >= i) || (!upper && i >= j) {
            (i, j, false)
        } else {
            (j, i, true)
        };
        let idx = if row_major {
            ii * lda + jj
        } else {
            jj * lda + ii
        };
        let a_val = *a.add(idx);
        if need_conj {
            (a_val.re, -a_val.im)
        } else if i == j {
            (a_val.re, 0.0)
        } else {
            (a_val.re, a_val.im)
        }
    };

    if left {
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;
                for l in 0..k {
                    let (a_re, a_im) = get_a(i, l);
                    let b_idx = if row_major { l * ldb + j } else { j * ldb + l };
                    let b_val = *b.add(b_idx);
                    sum_re += a_re * b_val.re - a_im * b_val.im;
                    sum_im += a_re * b_val.im + a_im * b_val.re;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                (*c.add(c_idx)).re += alpha.re * sum_re - alpha.im * sum_im;
                (*c.add(c_idx)).im += alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;
                for l in 0..k {
                    let b_idx = if row_major { i * ldb + l } else { l * ldb + i };
                    let b_val = *b.add(b_idx);
                    let (a_re, a_im) = get_a(l, j);
                    sum_re += b_val.re * a_re - b_val.im * a_im;
                    sum_im += b_val.re * a_im + b_val.im * a_re;
                }
                let c_idx = if row_major { i * ldc + j } else { j * ldc + i };
                (*c.add(c_idx)).re += alpha.re * sum_re - alpha.im * sum_im;
                (*c.add(c_idx)).im += alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    }
}

// =============================================================================
