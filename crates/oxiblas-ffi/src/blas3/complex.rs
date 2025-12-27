//! BLAS Level 3 FFI - Matrix-Matrix operations.

use crate::types::*;
use std::ffi::c_int;
// CTRMM - Complex triangular matrix multiply (single precision)
// =============================================================================

/// Computes B = alpha*op(A)*B or B = alpha*B*op(A) where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ctrmm(
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

    if alpha.re == 0.0 && alpha.im == 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                (*b.add(idx)).re = 0.0;
                (*b.add(idx)).im = 0.0;
            }
        }
        return;
    }

    // Helper to get A element with conjugation
    let get_a = |i: usize, j: usize| -> (f32, f32) {
        let idx = if row_major { i * lda + j } else { j * lda + i };
        let a_val = *a.add(idx);
        if conj {
            (a_val.re, -a_val.im)
        } else {
            (a_val.re, a_val.im)
        }
    };

    if left {
        let k = m;
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f32;
                let mut sum_im = 0.0f32;
                for l in 0..k {
                    let (ai, aj) = if trans_a { (l, i) } else { (i, l) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let (a_re, a_im) = if !in_triangle {
                        (0.0, 0.0)
                    } else if ai == aj {
                        if unit_diag { (1.0, 0.0) } else { get_a(ai, aj) }
                    } else {
                        get_a(ai, aj)
                    };
                    let b_idx = if row_major { l * ldb + j } else { j * ldb + l };
                    let b_val = *b.add(b_idx);
                    sum_re += a_re * b_val.re - a_im * b_val.im;
                    sum_im += a_re * b_val.im + a_im * b_val.re;
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                (*b.add(b_idx)).re = alpha.re * sum_re - alpha.im * sum_im;
                (*b.add(b_idx)).im = alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    } else {
        let k = n;
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f32;
                let mut sum_im = 0.0f32;
                for l in 0..k {
                    let (ai, aj) = if trans_a { (j, l) } else { (l, j) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let (a_re, a_im) = if !in_triangle {
                        (0.0, 0.0)
                    } else if ai == aj {
                        if unit_diag { (1.0, 0.0) } else { get_a(ai, aj) }
                    } else {
                        get_a(ai, aj)
                    };
                    let b_idx = if row_major { i * ldb + l } else { l * ldb + i };
                    let b_val = *b.add(b_idx);
                    sum_re += b_val.re * a_re - b_val.im * a_im;
                    sum_im += b_val.re * a_im + b_val.im * a_re;
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                (*b.add(b_idx)).re = alpha.re * sum_re - alpha.im * sum_im;
                (*b.add(b_idx)).im = alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    }
}

// =============================================================================
// ZTRMM - Complex triangular matrix multiply (double precision)
// =============================================================================

/// Computes B = alpha*op(A)*B or B = alpha*B*op(A) where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ztrmm(
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

    if alpha.re == 0.0 && alpha.im == 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                (*b.add(idx)).re = 0.0;
                (*b.add(idx)).im = 0.0;
            }
        }
        return;
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

    if left {
        let k = m;
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;
                for l in 0..k {
                    let (ai, aj) = if trans_a { (l, i) } else { (i, l) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let (a_re, a_im) = if !in_triangle {
                        (0.0, 0.0)
                    } else if ai == aj {
                        if unit_diag { (1.0, 0.0) } else { get_a(ai, aj) }
                    } else {
                        get_a(ai, aj)
                    };
                    let b_idx = if row_major { l * ldb + j } else { j * ldb + l };
                    let b_val = *b.add(b_idx);
                    sum_re += a_re * b_val.re - a_im * b_val.im;
                    sum_im += a_re * b_val.im + a_im * b_val.re;
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                (*b.add(b_idx)).re = alpha.re * sum_re - alpha.im * sum_im;
                (*b.add(b_idx)).im = alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    } else {
        let k = n;
        for i in 0..m {
            for j in 0..n {
                let mut sum_re = 0.0f64;
                let mut sum_im = 0.0f64;
                for l in 0..k {
                    let (ai, aj) = if trans_a { (j, l) } else { (l, j) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let (a_re, a_im) = if !in_triangle {
                        (0.0, 0.0)
                    } else if ai == aj {
                        if unit_diag { (1.0, 0.0) } else { get_a(ai, aj) }
                    } else {
                        get_a(ai, aj)
                    };
                    let b_idx = if row_major { i * ldb + l } else { l * ldb + i };
                    let b_val = *b.add(b_idx);
                    sum_re += b_val.re * a_re - b_val.im * a_im;
                    sum_im += b_val.re * a_im + b_val.im * a_re;
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                (*b.add(b_idx)).re = alpha.re * sum_re - alpha.im * sum_im;
                (*b.add(b_idx)).im = alpha.re * sum_im + alpha.im * sum_re;
            }
        }
    }
}

// =============================================================================
// CSYMM - Complex symmetric matrix multiply (single precision)
// =============================================================================

/// Computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is symmetric.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_csymm(
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

    // Helper to read symmetric matrix A (no conjugation - symmetric, not Hermitian)
    let get_a = |i: usize, j: usize| -> (f32, f32) {
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
        let a_val = *a.add(idx);
        (a_val.re, a_val.im)
    };

    if left {
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
// ZSYMM - Complex symmetric matrix multiply (double precision)
// =============================================================================

/// Computes C = alpha*A*B + beta*C or C = alpha*B*A + beta*C where A is symmetric.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zsymm(
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
        let a_val = *a.add(idx);
        (a_val.re, a_val.im)
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
// CSYRK - Complex symmetric rank-k update (single precision)
// =============================================================================

/// Computes C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_csyrk(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    n: c_int,
    k: c_int,
    alpha: OblasComplex32,
    a: *const OblasComplex32,
    lda: c_int,
    beta: OblasComplex32,
    c: *mut OblasComplex32,
    ldc: c_int,
) {
    if n <= 0 || a.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let trans_a = trans != OblasTranspose::NoTrans;

    // Scale C by beta
    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
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

    if (alpha.re == 0.0 && alpha.im == 0.0) || k == 0 {
        return;
    }

    // C = alpha * A * A^T (symmetric, not Hermitian - no conjugation)
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for l in 0..k {
                let a_il_idx = if trans_a {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if trans_a {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);

                // No conjugation for symmetric
                sum_re += a_il.re * a_jl.re - a_il.im * a_jl.im;
                sum_im += a_il.re * a_jl.im + a_il.im * a_jl.re;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
            (*c.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
        }
    }
}

// =============================================================================
// ZSYRK - Complex symmetric rank-k update (double precision)
// =============================================================================

/// Computes C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zsyrk(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    n: c_int,
    k: c_int,
    alpha: OblasComplex64,
    a: *const OblasComplex64,
    lda: c_int,
    beta: OblasComplex64,
    c: *mut OblasComplex64,
    ldc: c_int,
) {
    if n <= 0 || a.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let trans_a = trans != OblasTranspose::NoTrans;

    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
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

    if (alpha.re == 0.0 && alpha.im == 0.0) || k == 0 {
        return;
    }

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for l in 0..k {
                let a_il_idx = if trans_a {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if trans_a {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);

                sum_re += a_il.re * a_jl.re - a_il.im * a_jl.im;
                sum_im += a_il.re * a_jl.im + a_il.im * a_jl.re;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
            (*c.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
        }
    }
}

// =============================================================================
// CSYR2K - Complex symmetric rank-2k update (single precision)
// =============================================================================

/// Computes C = alpha*A*B^T + alpha*B*A^T + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_csyr2k(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
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
    if n <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let trans_a = trans != OblasTranspose::NoTrans;

    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
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

    if (alpha.re == 0.0 && alpha.im == 0.0) || k == 0 {
        return;
    }

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for l in 0..k {
                let a_il_idx = if trans_a {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if trans_a {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };
                let b_il_idx = if trans_a {
                    if row_major { l * ldb + i } else { i * ldb + l }
                } else {
                    if row_major { i * ldb + l } else { l * ldb + i }
                };
                let b_jl_idx = if trans_a {
                    if row_major { l * ldb + j } else { j * ldb + l }
                } else {
                    if row_major { j * ldb + l } else { l * ldb + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);
                let b_il = *b.add(b_il_idx);
                let b_jl = *b.add(b_jl_idx);

                // A*B^T (no conjugation)
                let prod1_re = a_il.re * b_jl.re - a_il.im * b_jl.im;
                let prod1_im = a_il.re * b_jl.im + a_il.im * b_jl.re;
                // B*A^T (no conjugation)
                let prod2_re = b_il.re * a_jl.re - b_il.im * a_jl.im;
                let prod2_im = b_il.re * a_jl.im + b_il.im * a_jl.re;

                sum_re += prod1_re + prod2_re;
                sum_im += prod1_im + prod2_im;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
            (*c.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
        }
    }
}

// =============================================================================
// ZSYR2K - Complex symmetric rank-2k update (double precision)
// =============================================================================

/// Computes C = alpha*A*B^T + alpha*B*A^T + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zsyr2k(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
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
    if n <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let trans_a = trans != OblasTranspose::NoTrans;

    let beta_zero = beta.re == 0.0 && beta.im == 0.0;
    let beta_one = beta.re == 1.0 && beta.im == 0.0;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
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

    if (alpha.re == 0.0 && alpha.im == 0.0) || k == 0 {
        return;
    }

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for l in 0..k {
                let a_il_idx = if trans_a {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if trans_a {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };
                let b_il_idx = if trans_a {
                    if row_major { l * ldb + i } else { i * ldb + l }
                } else {
                    if row_major { i * ldb + l } else { l * ldb + i }
                };
                let b_jl_idx = if trans_a {
                    if row_major { l * ldb + j } else { j * ldb + l }
                } else {
                    if row_major { j * ldb + l } else { l * ldb + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);
                let b_il = *b.add(b_il_idx);
                let b_jl = *b.add(b_jl_idx);

                let prod1_re = a_il.re * b_jl.re - a_il.im * b_jl.im;
                let prod1_im = a_il.re * b_jl.im + a_il.im * b_jl.re;
                let prod2_re = b_il.re * a_jl.re - b_il.im * a_jl.im;
                let prod2_im = b_il.re * a_jl.im + b_il.im * a_jl.re;

                sum_re += prod1_re + prod2_re;
                sum_im += prod1_im + prod2_im;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += alpha.re * sum_re - alpha.im * sum_im;
            (*c.add(idx)).im += alpha.re * sum_im + alpha.im * sum_re;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas3::{oblas_dgemm, oblas_dsymm, oblas_zgemm};

    #[test]
    fn test_dgemm_row_major() {
        // A = [[1, 2], [3, 4]] (2x2 row-major)
        // B = [[5, 6], [7, 8]] (2x2 row-major)
        // C = A * B = [[19, 22], [43, 50]]
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];

        unsafe {
            oblas_dgemm(
                OblasLayout::RowMajor,
                OblasTranspose::NoTrans,
                OblasTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );
        }

        assert!((c[0] - 19.0).abs() < 1e-10);
        assert!((c[1] - 22.0).abs() < 1e-10);
        assert!((c[2] - 43.0).abs() < 1e-10);
        assert!((c[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_dgemm_col_major() {
        // A = [[1, 2], [3, 4]] stored column-major: [1, 3, 2, 4]
        // B = [[5, 6], [7, 8]] stored column-major: [5, 7, 6, 8]
        // C = A * B = [[19, 22], [43, 50]]
        let a = [1.0f64, 3.0, 2.0, 4.0]; // Column-major
        let b = [5.0f64, 7.0, 6.0, 8.0]; // Column-major
        let mut c = [0.0f64; 4];

        unsafe {
            oblas_dgemm(
                OblasLayout::ColMajor,
                OblasTranspose::NoTrans,
                OblasTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );
        }

        // Result in column-major: [19, 43, 22, 50]
        assert!((c[0] - 19.0).abs() < 1e-10);
        assert!((c[1] - 43.0).abs() < 1e-10);
        assert!((c[2] - 22.0).abs() < 1e-10);
        assert!((c[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_zgemm() {
        // A = [[1+i, 2], [3, 4-i]] (2x2)
        // B = [[1, 0], [0, 1]] = I
        // C = A * I = A
        let a = [
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
            OblasComplex64 { re: 4.0, im: -1.0 },
        ];
        let b = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
        ];
        let mut c = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];

        unsafe {
            oblas_zgemm(
                OblasLayout::RowMajor,
                OblasTranspose::NoTrans,
                OblasTranspose::NoTrans,
                2,
                2,
                2,
                OblasComplex64 { re: 1.0, im: 0.0 },
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                OblasComplex64 { re: 0.0, im: 0.0 },
                c.as_mut_ptr(),
                2,
            );
        }

        assert!((c[0].re - 1.0).abs() < 1e-10);
        assert!((c[0].im - 1.0).abs() < 1e-10);
        assert!((c[1].re - 2.0).abs() < 1e-10);
        assert!((c[3].re - 4.0).abs() < 1e-10);
        assert!((c[3].im - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dsymm() {
        // A = [[1, 2], [2, 3]] symmetric (upper stored)
        // B = [[1, 0], [0, 1]] = I
        // C = A * I = A
        let a = [1.0f64, 2.0, 0.0, 3.0]; // Upper triangle: [1, 2; _, 3]
        let b = [1.0f64, 0.0, 0.0, 1.0];
        let mut c = [0.0f64; 4];

        unsafe {
            oblas_dsymm(
                OblasLayout::RowMajor,
                OblasSide::Left,
                OblasUplo::Upper,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );
        }

        // C = [[1, 2], [2, 3]]
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 2.0).abs() < 1e-10);
        assert!((c[2] - 2.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }
}
