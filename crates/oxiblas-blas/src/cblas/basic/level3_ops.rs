//! CBLAS Level 3 (matrix-matrix) real operations: GEMM.

use super::super::types::*;
use super::super::validate::gemm_params_valid;
use crate::level3;
/// Double precision GEMM: C = alpha * op(A) * op(B) + beta * C.
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
/// - `b` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f64`, and point to
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
pub unsafe extern "C" fn cblas_dgemm(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
    {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    // Create matrices from raw pointers and call our internal GEMM
    match layout {
        CblasLayout::ColMajor => {
            gemm_raw_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        CblasLayout::RowMajor => {
            // For row-major: C = A * B in row-major is equivalent to C^T = B^T * A^T in col-major
            // So we swap A and B and swap their transposes
            let new_transa = transb;
            let new_transb = transa;
            gemm_raw_colmajor(
                new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc,
            );
        }
    }
}

/// Internal GEMM for column-major layout.
///
/// Uses optimized SIMD implementation for NoTrans/NoTrans case.
unsafe fn gemm_raw_colmajor(
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    // Fast path: NoTrans/NoTrans - use optimized SIMD implementation
    if matches!(transa, CblasTranspose::NoTrans) && matches!(transb, CblasTranspose::NoTrans) {
        use oxiblas_matrix::{MatMut, MatRef};

        // A is m×k with leading dimension lda
        let a_ref = MatRef::<f64>::new(a, m, k, lda);
        // B is k×n with leading dimension ldb
        let b_ref = MatRef::<f64>::new(b, k, n, ldb);
        // C is m×n with leading dimension ldc
        let c_mut = MatMut::<f64>::new(c.cast::<f64>(), m, n, ldc);

        level3::gemm(alpha, a_ref, b_ref, beta, c_mut);
        return;
    }

    // Fallback: handle transpose cases with scalar implementation.
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite rather than multiply (0*NaN would be NaN).
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == 0.0 {
                *cp = 0.0;
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
            // Get B element based on transpose
            let b_val = match transb {
                CblasTranspose::NoTrans => *b.add(p + j * ldb),
                CblasTranspose::Trans | CblasTranspose::ConjTrans => *b.add(j + p * ldb),
            };
            let temp = alpha * b_val;

            for i in 0..m {
                // Get A element based on transpose
                let a_val = match transa {
                    CblasTranspose::NoTrans => *a.add(i + p * lda),
                    CblasTranspose::Trans | CblasTranspose::ConjTrans => *a.add(p + i * lda),
                };
                let cp = c.add(i + j * ldc);
                *cp += a_val * temp;
            }
        }
    }
}

/// Single precision GEMM: C = alpha * op(A) * op(B) + beta * C.
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
/// - `b` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f32`, and point to
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
pub unsafe extern "C" fn cblas_sgemm(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
    {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    match layout {
        CblasLayout::ColMajor => {
            sgemm_raw_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        CblasLayout::RowMajor => {
            let new_transa = transb;
            let new_transb = transa;
            sgemm_raw_colmajor(
                new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc,
            );
        }
    }
}

/// Internal SGEMM for column-major layout.
///
/// Uses optimized SIMD implementation for NoTrans/NoTrans case.
unsafe fn sgemm_raw_colmajor(
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    // Fast path: NoTrans/NoTrans - use optimized SIMD implementation
    if matches!(transa, CblasTranspose::NoTrans) && matches!(transb, CblasTranspose::NoTrans) {
        use oxiblas_matrix::{MatMut, MatRef};

        // A is m×k with leading dimension lda
        let a_ref = MatRef::<f32>::new(a, m, k, lda);
        // B is k×n with leading dimension ldb
        let b_ref = MatRef::<f32>::new(b, k, n, ldb);
        // C is m×n with leading dimension ldc
        let c_mut = MatMut::<f32>::new(c.cast::<f32>(), m, n, ldc);

        level3::gemm(alpha, a_ref, b_ref, beta, c_mut);
        return;
    }

    // Fallback: handle transpose cases with scalar implementation.
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite rather than multiply (0*NaN would be NaN).
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == 0.0 {
                *cp = 0.0;
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
                CblasTranspose::Trans | CblasTranspose::ConjTrans => *b.add(j + p * ldb),
            };
            let temp = alpha * b_val;

            for i in 0..m {
                let a_val = match transa {
                    CblasTranspose::NoTrans => *a.add(i + p * lda),
                    CblasTranspose::Trans | CblasTranspose::ConjTrans => *a.add(p + i * lda),
                };
                let cp = c.add(i + j * ldc);
                *cp += a_val * temp;
            }
        }
    }
}
