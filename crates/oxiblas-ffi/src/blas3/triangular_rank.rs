//! BLAS Level 3 FFI - Matrix-Matrix operations.

use crate::types::*;
use std::ffi::c_int;
// STRMM - Triangular matrix multiply (single precision)
// =============================================================================

/// Computes B = alpha*op(A)*B or B = alpha*B*op(A) where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_strmm(
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

    if alpha == 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) = 0.0;
            }
        }
        return;
    }

    if left {
        // B = alpha * op(A) * B
        let k = m;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    // Check if A[i,l] or A[l,i] (if transposed) is in the triangle
                    let (ai, aj) = if trans_a { (l, i) } else { (i, l) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let a_val = if !in_triangle {
                        0.0
                    } else if ai == aj {
                        if unit_diag {
                            1.0
                        } else {
                            let idx = if row_major {
                                ai * lda + aj
                            } else {
                                aj * lda + ai
                            };
                            *a.add(idx)
                        }
                    } else {
                        let idx = if row_major {
                            ai * lda + aj
                        } else {
                            aj * lda + ai
                        };
                        *a.add(idx)
                    };
                    let b_idx = if row_major { l * ldb + j } else { j * ldb + l };
                    sum += a_val * *b.add(b_idx);
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(b_idx) = alpha * sum;
            }
        }
    } else {
        // B = alpha * B * op(A)
        let k = n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let (ai, aj) = if trans_a { (j, l) } else { (l, j) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let a_val = if !in_triangle {
                        0.0
                    } else if ai == aj {
                        if unit_diag {
                            1.0
                        } else {
                            let idx = if row_major {
                                ai * lda + aj
                            } else {
                                aj * lda + ai
                            };
                            *a.add(idx)
                        }
                    } else {
                        let idx = if row_major {
                            ai * lda + aj
                        } else {
                            aj * lda + ai
                        };
                        *a.add(idx)
                    };
                    let b_idx = if row_major { i * ldb + l } else { l * ldb + i };
                    sum += *b.add(b_idx) * a_val;
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(b_idx) = alpha * sum;
            }
        }
    }
}

// =============================================================================
// DTRMM - Triangular matrix multiply (double precision)
// =============================================================================

/// Computes B = alpha*op(A)*B or B = alpha*B*op(A) where A is triangular.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtrmm(
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

    if alpha == 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) = 0.0;
            }
        }
        return;
    }

    if left {
        let k = m;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    let (ai, aj) = if trans_a { (l, i) } else { (i, l) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let a_val = if !in_triangle {
                        0.0
                    } else if ai == aj {
                        if unit_diag {
                            1.0
                        } else {
                            let idx = if row_major {
                                ai * lda + aj
                            } else {
                                aj * lda + ai
                            };
                            *a.add(idx)
                        }
                    } else {
                        let idx = if row_major {
                            ai * lda + aj
                        } else {
                            aj * lda + ai
                        };
                        *a.add(idx)
                    };
                    let b_idx = if row_major { l * ldb + j } else { j * ldb + l };
                    sum += a_val * *b.add(b_idx);
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(b_idx) = alpha * sum;
            }
        }
    } else {
        let k = n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    let (ai, aj) = if trans_a { (j, l) } else { (l, j) };
                    let in_triangle = if upper { aj >= ai } else { ai >= aj };
                    let a_val = if !in_triangle {
                        0.0
                    } else if ai == aj {
                        if unit_diag {
                            1.0
                        } else {
                            let idx = if row_major {
                                ai * lda + aj
                            } else {
                                aj * lda + ai
                            };
                            *a.add(idx)
                        }
                    } else {
                        let idx = if row_major {
                            ai * lda + aj
                        } else {
                            aj * lda + ai
                        };
                        *a.add(idx)
                    };
                    let b_idx = if row_major { i * ldb + l } else { l * ldb + i };
                    sum += *b.add(b_idx) * a_val;
                }
                let b_idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(b_idx) = alpha * sum;
            }
        }
    }
}

// =============================================================================
// DSYRK - Symmetric rank-k update (double precision)
// =============================================================================

/// Computes C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsyrk(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    beta: f64,
    c: *mut f64,
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

    // Scale C by beta (only the triangle being updated)
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                *c.add(idx) = 0.0;
            } else if beta != 1.0 {
                *c.add(idx) *= beta;
            }
        }
    }

    if alpha == 0.0 || k == 0 {
        return;
    }

    // C = alpha * A * A^T + C (if no trans, row-major)
    // or C = alpha * A^T * A + C (if trans)
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum = 0.0f64;
            for l in 0..k {
                let a_il = if trans_a {
                    if row_major {
                        *a.add(l * lda + i)
                    } else {
                        *a.add(i * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(i * lda + l)
                    } else {
                        *a.add(l * lda + i)
                    }
                };
                let a_jl = if trans_a {
                    if row_major {
                        *a.add(l * lda + j)
                    } else {
                        *a.add(j * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(j * lda + l)
                    } else {
                        *a.add(l * lda + j)
                    }
                };
                sum += a_il * a_jl;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) += alpha * sum;
        }
    }
}

// =============================================================================
// SSYRK - Symmetric rank-k update (single precision)
// =============================================================================

/// Computes C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssyrk(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: *const f32,
    lda: c_int,
    beta: f32,
    c: *mut f32,
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

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                *c.add(idx) = 0.0;
            } else if beta != 1.0 {
                *c.add(idx) *= beta;
            }
        }
    }

    if alpha == 0.0 || k == 0 {
        return;
    }

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum = 0.0f32;
            for l in 0..k {
                let a_il = if trans_a {
                    if row_major {
                        *a.add(l * lda + i)
                    } else {
                        *a.add(i * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(i * lda + l)
                    } else {
                        *a.add(l * lda + i)
                    }
                };
                let a_jl = if trans_a {
                    if row_major {
                        *a.add(l * lda + j)
                    } else {
                        *a.add(j * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(j * lda + l)
                    } else {
                        *a.add(l * lda + j)
                    }
                };
                sum += a_il * a_jl;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) += alpha * sum;
        }
    }
}

// =============================================================================
// ZHERK - Hermitian rank-k update (double precision)
// =============================================================================

/// Computes C = alpha*A*A^H + beta*C or C = alpha*A^H*A + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zherk(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: *const OblasComplex64,
    lda: c_int,
    beta: f64,
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
    let conj_trans = trans == OblasTranspose::ConjTrans;

    // Scale C by beta
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if beta != 1.0 {
                (*c.add(idx)).re *= beta;
                (*c.add(idx)).im *= beta;
            }
            // Ensure diagonal is real
            if i == j {
                (*c.add(idx)).im = 0.0;
            }
        }
    }

    if alpha == 0.0 || k == 0 {
        return;
    }

    // C = alpha * A * A^H or C = alpha * A^H * A
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for l in 0..k {
                // Get A[i,l] or A[l,i] depending on trans
                let a_il_idx = if conj_trans {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if conj_trans {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);

                // For conj_trans: A^H * A means conj(A[l,i]) * A[l,j]
                // For no trans: A * A^H means A[i,l] * conj(A[j,l])
                let (a1_re, a1_im, a2_re, a2_im) = if conj_trans {
                    // conj(a_il) * a_jl
                    (a_il.re, -a_il.im, a_jl.re, a_jl.im)
                } else {
                    // a_il * conj(a_jl)
                    (a_il.re, a_il.im, a_jl.re, -a_jl.im)
                };

                sum_re += a1_re * a2_re - a1_im * a2_im;
                sum_im += a1_re * a2_im + a1_im * a2_re;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += alpha * sum_re;
            if i != j {
                (*c.add(idx)).im += alpha * sum_im;
            }
        }
    }
}

// =============================================================================
// CHERK - Hermitian rank-k update (single precision)
// =============================================================================

/// Computes C = alpha*A*A^H + beta*C or C = alpha*A^H*A + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cherk(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: *const OblasComplex32,
    lda: c_int,
    beta: f32,
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
    let conj_trans = trans == OblasTranspose::ConjTrans;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if beta != 1.0 {
                (*c.add(idx)).re *= beta;
                (*c.add(idx)).im *= beta;
            }
            if i == j {
                (*c.add(idx)).im = 0.0;
            }
        }
    }

    if alpha == 0.0 || k == 0 {
        return;
    }

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for l in 0..k {
                let a_il_idx = if conj_trans {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if conj_trans {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);

                let (a1_re, a1_im, a2_re, a2_im) = if conj_trans {
                    (a_il.re, -a_il.im, a_jl.re, a_jl.im)
                } else {
                    (a_il.re, a_il.im, a_jl.re, -a_jl.im)
                };

                sum_re += a1_re * a2_re - a1_im * a2_im;
                sum_im += a1_re * a2_im + a1_im * a2_re;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += alpha * sum_re;
            if i != j {
                (*c.add(idx)).im += alpha * sum_im;
            }
        }
    }
}

// =============================================================================
// DSYR2K - Symmetric rank-2k update (double precision)
// =============================================================================

/// Computes C = alpha*A*B^T + alpha*B*A^T + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsyr2k(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
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

    // Scale C
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                *c.add(idx) = 0.0;
            } else if beta != 1.0 {
                *c.add(idx) *= beta;
            }
        }
    }

    if alpha == 0.0 || k == 0 {
        return;
    }

    // C = alpha * A * B^T + alpha * B * A^T + C
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum = 0.0f64;
            for l in 0..k {
                let a_il = if trans_a {
                    if row_major {
                        *a.add(l * lda + i)
                    } else {
                        *a.add(i * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(i * lda + l)
                    } else {
                        *a.add(l * lda + i)
                    }
                };
                let a_jl = if trans_a {
                    if row_major {
                        *a.add(l * lda + j)
                    } else {
                        *a.add(j * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(j * lda + l)
                    } else {
                        *a.add(l * lda + j)
                    }
                };
                let b_il = if trans_a {
                    if row_major {
                        *b.add(l * ldb + i)
                    } else {
                        *b.add(i * ldb + l)
                    }
                } else {
                    if row_major {
                        *b.add(i * ldb + l)
                    } else {
                        *b.add(l * ldb + i)
                    }
                };
                let b_jl = if trans_a {
                    if row_major {
                        *b.add(l * ldb + j)
                    } else {
                        *b.add(j * ldb + l)
                    }
                } else {
                    if row_major {
                        *b.add(j * ldb + l)
                    } else {
                        *b.add(l * ldb + j)
                    }
                };
                sum += a_il * b_jl + b_il * a_jl;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) += alpha * sum;
        }
    }
}

// =============================================================================
// SSYR2K - Symmetric rank-2k update (single precision)
// =============================================================================

/// Computes C = alpha*A*B^T + alpha*B*A^T + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssyr2k(
    layout: OblasLayout,
    uplo: OblasUplo,
    trans: OblasTranspose,
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

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                *c.add(idx) = 0.0;
            } else if beta != 1.0 {
                *c.add(idx) *= beta;
            }
        }
    }

    if alpha == 0.0 || k == 0 {
        return;
    }

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum = 0.0f32;
            for l in 0..k {
                let a_il = if trans_a {
                    if row_major {
                        *a.add(l * lda + i)
                    } else {
                        *a.add(i * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(i * lda + l)
                    } else {
                        *a.add(l * lda + i)
                    }
                };
                let a_jl = if trans_a {
                    if row_major {
                        *a.add(l * lda + j)
                    } else {
                        *a.add(j * lda + l)
                    }
                } else {
                    if row_major {
                        *a.add(j * lda + l)
                    } else {
                        *a.add(l * lda + j)
                    }
                };
                let b_il = if trans_a {
                    if row_major {
                        *b.add(l * ldb + i)
                    } else {
                        *b.add(i * ldb + l)
                    }
                } else {
                    if row_major {
                        *b.add(i * ldb + l)
                    } else {
                        *b.add(l * ldb + i)
                    }
                };
                let b_jl = if trans_a {
                    if row_major {
                        *b.add(l * ldb + j)
                    } else {
                        *b.add(j * ldb + l)
                    }
                } else {
                    if row_major {
                        *b.add(j * ldb + l)
                    } else {
                        *b.add(l * ldb + j)
                    }
                };
                sum += a_il * b_jl + b_il * a_jl;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) += alpha * sum;
        }
    }
}

// =============================================================================
// ZHER2K - Hermitian rank-2k update (double precision)
// =============================================================================

/// Computes C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zher2k(
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
    beta: f64,
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
    let conj_trans = trans == OblasTranspose::ConjTrans;

    // Scale C by beta
    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if beta != 1.0 {
                (*c.add(idx)).re *= beta;
                (*c.add(idx)).im *= beta;
            }
            if i == j {
                (*c.add(idx)).im = 0.0;
            }
        }
    }

    if (alpha.re == 0.0 && alpha.im == 0.0) || k == 0 {
        return;
    }

    let alpha_conj_re = alpha.re;
    let alpha_conj_im = -alpha.im;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for l in 0..k {
                let a_il_idx = if conj_trans {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if conj_trans {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };
                let b_il_idx = if conj_trans {
                    if row_major { l * ldb + i } else { i * ldb + l }
                } else {
                    if row_major { i * ldb + l } else { l * ldb + i }
                };
                let b_jl_idx = if conj_trans {
                    if row_major { l * ldb + j } else { j * ldb + l }
                } else {
                    if row_major { j * ldb + l } else { l * ldb + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);
                let b_il = *b.add(b_il_idx);
                let b_jl = *b.add(b_jl_idx);

                // A*B^H: a_il * conj(b_jl)
                // B*A^H: b_il * conj(a_jl)
                let (a1_re, a1_im, b1_re, b1_im) = if conj_trans {
                    // A^H * B: conj(a[l,i]) * b[l,j]
                    (a_il.re, -a_il.im, b_jl.re, b_jl.im)
                } else {
                    // A * B^H: a[i,l] * conj(b[j,l])
                    (a_il.re, a_il.im, b_jl.re, -b_jl.im)
                };

                let (a2_re, a2_im, b2_re, b2_im) = if conj_trans {
                    // B^H * A: conj(b[l,i]) * a[l,j]
                    (b_il.re, -b_il.im, a_jl.re, a_jl.im)
                } else {
                    // B * A^H: b[i,l] * conj(a[j,l])
                    (b_il.re, b_il.im, a_jl.re, -a_jl.im)
                };

                // alpha * (a1 * b1)
                let prod1_re = a1_re * b1_re - a1_im * b1_im;
                let prod1_im = a1_re * b1_im + a1_im * b1_re;
                let term1_re = alpha.re * prod1_re - alpha.im * prod1_im;
                let term1_im = alpha.re * prod1_im + alpha.im * prod1_re;

                // conj(alpha) * (a2 * b2)
                let prod2_re = a2_re * b2_re - a2_im * b2_im;
                let prod2_im = a2_re * b2_im + a2_im * b2_re;
                let term2_re = alpha_conj_re * prod2_re - alpha_conj_im * prod2_im;
                let term2_im = alpha_conj_re * prod2_im + alpha_conj_im * prod2_re;

                sum_re += term1_re + term2_re;
                sum_im += term1_im + term2_im;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += sum_re;
            if i != j {
                (*c.add(idx)).im += sum_im;
            }
        }
    }
}

// =============================================================================
// CHER2K - Hermitian rank-2k update (single precision)
// =============================================================================

/// Computes C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C.
///
/// # Safety
/// - All matrix pointers must be valid and properly sized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cher2k(
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
    beta: f32,
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
    let conj_trans = trans == OblasTranspose::ConjTrans;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            if beta == 0.0 {
                (*c.add(idx)).re = 0.0;
                (*c.add(idx)).im = 0.0;
            } else if beta != 1.0 {
                (*c.add(idx)).re *= beta;
                (*c.add(idx)).im *= beta;
            }
            if i == j {
                (*c.add(idx)).im = 0.0;
            }
        }
    }

    if (alpha.re == 0.0 && alpha.im == 0.0) || k == 0 {
        return;
    }

    let alpha_conj_re = alpha.re;
    let alpha_conj_im = -alpha.im;

    for i in 0..n {
        let j_start = if upper { i } else { 0 };
        let j_end = if upper { n } else { i + 1 };
        for j in j_start..j_end {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for l in 0..k {
                let a_il_idx = if conj_trans {
                    if row_major { l * lda + i } else { i * lda + l }
                } else {
                    if row_major { i * lda + l } else { l * lda + i }
                };
                let a_jl_idx = if conj_trans {
                    if row_major { l * lda + j } else { j * lda + l }
                } else {
                    if row_major { j * lda + l } else { l * lda + j }
                };
                let b_il_idx = if conj_trans {
                    if row_major { l * ldb + i } else { i * ldb + l }
                } else {
                    if row_major { i * ldb + l } else { l * ldb + i }
                };
                let b_jl_idx = if conj_trans {
                    if row_major { l * ldb + j } else { j * ldb + l }
                } else {
                    if row_major { j * ldb + l } else { l * ldb + j }
                };

                let a_il = *a.add(a_il_idx);
                let a_jl = *a.add(a_jl_idx);
                let b_il = *b.add(b_il_idx);
                let b_jl = *b.add(b_jl_idx);

                let (a1_re, a1_im, b1_re, b1_im) = if conj_trans {
                    (a_il.re, -a_il.im, b_jl.re, b_jl.im)
                } else {
                    (a_il.re, a_il.im, b_jl.re, -b_jl.im)
                };

                let (a2_re, a2_im, b2_re, b2_im) = if conj_trans {
                    (b_il.re, -b_il.im, a_jl.re, a_jl.im)
                } else {
                    (b_il.re, b_il.im, a_jl.re, -a_jl.im)
                };

                let prod1_re = a1_re * b1_re - a1_im * b1_im;
                let prod1_im = a1_re * b1_im + a1_im * b1_re;
                let term1_re = alpha.re * prod1_re - alpha.im * prod1_im;
                let term1_im = alpha.re * prod1_im + alpha.im * prod1_re;

                let prod2_re = a2_re * b2_re - a2_im * b2_im;
                let prod2_im = a2_re * b2_im + a2_im * b2_re;
                let term2_re = alpha_conj_re * prod2_re - alpha_conj_im * prod2_im;
                let term2_im = alpha_conj_re * prod2_im + alpha_conj_im * prod2_re;

                sum_re += term1_re + term2_re;
                sum_im += term1_im + term2_im;
            }
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            (*c.add(idx)).re += sum_re;
            if i != j {
                (*c.add(idx)).im += sum_im;
            }
        }
    }
}

// =============================================================================
