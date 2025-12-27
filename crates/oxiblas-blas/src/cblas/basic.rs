//! CBLAS-compatible interface for BLAS-TESTER compatibility.
//!
//! This module provides a C-compatible interface that follows the standard
//! CBLAS specification, enabling interoperability with BLAS test suites
//! like BLAS-TESTER.
//!
//! # Layout
//!
//! CBLAS supports both row-major and column-major layouts. This library
//! uses column-major (Fortran) layout internally, so row-major operations
//! are converted using the identity: op(A) in row-major = op(A^T) in column-major.
//!
//! # Performance
//!
//! For unit stride vectors (incx=1, incy=1), this module uses the optimized
//! internal BLAS implementations with SIMD acceleration. For non-unit strides,
//! scalar fallbacks are used.

use super::types::*;
use crate::level1;
use crate::level3;
use num_complex::{Complex32, Complex64};

// Level 1 BLAS - Vector operations
// =============================================================================

/// Double precision dot product.
///
/// Computes: x · y
///
/// Uses optimized SIMD implementation for unit stride vectors.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ddot(
    n: i32,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
) -> f64 {
    if n <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts(y, n);
        return level1::dot(x_slice, y_slice);
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;

    let mut result = 0.0;
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        let yi = *y.offset(i as isize * incy);
        result += xi * yi;
    }
    result
}

/// Single precision dot product.
///
/// Uses optimized SIMD implementation for unit stride vectors.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sdot(
    n: i32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
) -> f32 {
    if n <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts(y, n);
        return level1::dot(x_slice, y_slice);
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;

    let mut result = 0.0f32;
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        let yi = *y.offset(i as isize * incy);
        result += xi * yi;
    }
    result
}

/// Double precision Euclidean norm.
///
/// Uses Blue's algorithm for numerical stability with unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64 {
    if n <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation with Blue's algorithm
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let mut sum_sq = 0.0;
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        sum_sq += xi * xi;
    }
    sum_sq.sqrt()
}

/// Single precision Euclidean norm.
///
/// Uses Blue's algorithm for numerical stability with unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32 {
    if n <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation with Blue's algorithm
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        sum_sq += xi * xi;
    }
    sum_sq.sqrt()
}

/// Double precision sum of absolute values.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64 {
    if n <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::asum(x_slice);
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let mut result = 0.0;
    for i in 0..n {
        result += (*x.offset(i as isize * incx)).abs();
    }
    result
}

/// Single precision sum of absolute values.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sasum(n: i32, x: *const f32, incx: i32) -> f32 {
    if n <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::asum(x_slice);
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let mut result = 0.0f32;
    for i in 0..n {
        result += (*x.offset(i as isize * incx)).abs();
    }
    result
}

/// Double precision index of maximum absolute value.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32 {
    if n <= 0 {
        return 0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::iamax(x_slice) as i32;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let mut max_idx = 0;
    let mut max_val = (*x).abs();

    for i in 1..n {
        let val = (*x.offset(i as isize * incx)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx as i32
}

/// Single precision index of maximum absolute value.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_isamax(n: i32, x: *const f32, incx: i32) -> i32 {
    if n <= 0 {
        return 0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::iamax(x_slice) as i32;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let mut max_idx = 0;
    let mut max_val = (*x).abs();

    for i in 1..n {
        let val = (*x.offset(i as isize * incx)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx as i32
}

/// Double precision vector scaling: x = alpha * x.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        level1::scal(alpha, x_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    for i in 0..n {
        let ptr = x.offset(i as isize * incx);
        *ptr *= alpha;
    }
}

/// Single precision vector scaling: x = alpha * x.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        level1::scal(alpha, x_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    for i in 0..n {
        let ptr = x.offset(i as isize * incx);
        *ptr *= alpha;
    }
}

/// Double precision axpy: y = alpha * x + y.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_daxpy(
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::axpy(alpha, x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        let yi_ptr = y.offset(i as isize * incy);
        *yi_ptr += alpha * xi;
    }
}

/// Single precision axpy: y = alpha * x + y.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_saxpy(
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::axpy(alpha, x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        let yi_ptr = y.offset(i as isize * incy);
        *yi_ptr += alpha * xi;
    }
}

/// Double precision vector copy: y = x.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::copy(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;

    for i in 0..n {
        *y.offset(i as isize * incy) = *x.offset(i as isize * incx);
    }
}

/// Single precision vector copy: y = x.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_scopy(n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::copy(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;

    for i in 0..n {
        *y.offset(i as isize * incy) = *x.offset(i as isize * incx);
    }
}

/// Double precision vector swap.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dswap(n: i32, x: *mut f64, incx: i32, y: *mut f64, incy: i32) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::swap(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;

    for i in 0..n {
        let xp = x.offset(i as isize * incx);
        let yp = y.offset(i as isize * incy);
        std::ptr::swap(xp, yp);
    }
}

/// Single precision vector swap.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sswap(n: i32, x: *mut f32, incx: i32, y: *mut f32, incy: i32) {
    if n <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 && incy == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        let y_slice = std::slice::from_raw_parts_mut(y, n);
        level1::swap(x_slice, y_slice);
        return;
    }

    // Fallback: non-unit stride
    let incx = incx as isize;
    let incy = incy as isize;

    for i in 0..n {
        let xp = x.offset(i as isize * incx);
        let yp = y.offset(i as isize * incy);
        std::ptr::swap(xp, yp);
    }
}

// =============================================================================
// Level 2 BLAS - Matrix-Vector operations
// =============================================================================

/// Double precision GEMV: y = alpha * op(A) * x + beta * y.
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

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    // Determine effective dimensions based on transpose
    let (rows, cols) = match trans {
        CblasTranspose::NoTrans => (m, n),
        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
    };

    // Scale y by beta
    for i in 0..rows {
        let yp = y.offset(i as isize * incy);
        *yp *= beta;
    }

    // Compute matrix-vector product
    match layout {
        CblasLayout::ColMajor => {
            for i in 0..rows {
                let yp = y.offset(i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai + aj * lda);
                    let x_val = *x.offset(j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
        CblasLayout::RowMajor => {
            for i in 0..rows {
                let yp = y.offset(i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai * lda + aj);
                    let x_val = *x.offset(j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
    }
}

/// Single precision GEMV: y = alpha * op(A) * x + beta * y.
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

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    let (rows, cols) = match trans {
        CblasTranspose::NoTrans => (m, n),
        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
    };

    for i in 0..rows {
        let yp = y.offset(i as isize * incy);
        *yp *= beta;
    }

    match layout {
        CblasLayout::ColMajor => {
            for i in 0..rows {
                let yp = y.offset(i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai + aj * lda);
                    let x_val = *x.offset(j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
        CblasLayout::RowMajor => {
            for i in 0..rows {
                let yp = y.offset(i as isize * incy);
                for j in 0..cols {
                    let (ai, aj) = match trans {
                        CblasTranspose::NoTrans => (i, j),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
                    };
                    let a_val = *a.add(ai * lda + aj);
                    let x_val = *x.offset(j as isize * incx);
                    *yp += alpha * a_val * x_val;
                }
            }
        }
    }
}

// =============================================================================
// Level 3 BLAS - Matrix-Matrix operations
// =============================================================================

/// Double precision GEMM: C = alpha * op(A) * op(B) + beta * C.
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

    // Fallback: handle transpose cases with scalar implementation
    // Scale C by beta
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            *cp *= beta;
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

    // Fallback: handle transpose cases with scalar implementation
    // Scale C by beta
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            *cp *= beta;
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

// =============================================================================
// Complex BLAS operations
// =============================================================================

/// Complex double precision dot product.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zdotu_sub(
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    dotu: *mut Complex64,
) {
    if n <= 0 {
        *dotu = Complex64::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    let mut result = Complex64::new(0.0, 0.0);
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        let yi = *y.offset(i as isize * incy);
        result += xi * yi;
    }
    *dotu = result;
}

/// Complex double precision conjugate dot product.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zdotc_sub(
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    dotc: *mut Complex64,
) {
    if n <= 0 {
        *dotc = Complex64::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    let mut result = Complex64::new(0.0, 0.0);
    for i in 0..n {
        let xi = (*x.offset(i as isize * incx)).conj();
        let yi = *y.offset(i as isize * incy);
        result += xi * yi;
    }
    *dotc = result;
}

/// Complex single precision dot product.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cdotu_sub(
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    dotu: *mut Complex32,
) {
    if n <= 0 {
        *dotu = Complex32::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    let mut result = Complex32::new(0.0, 0.0);
    for i in 0..n {
        let xi = *x.offset(i as isize * incx);
        let yi = *y.offset(i as isize * incy);
        result += xi * yi;
    }
    *dotu = result;
}

/// Complex single precision conjugate dot product.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cdotc_sub(
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    dotc: *mut Complex32,
) {
    if n <= 0 {
        *dotc = Complex32::new(0.0, 0.0);
        return;
    }

    let n = n as usize;
    let incx = incx as isize;
    let incy = incy as isize;

    let mut result = Complex32::new(0.0, 0.0);
    for i in 0..n {
        let xi = (*x.offset(i as isize * incx)).conj();
        let yi = *y.offset(i as isize * incy);
        result += xi * yi;
    }
    *dotc = result;
}

/// Complex double precision GEMM.
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
    // Scale C by beta
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            *cp *= beta;
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
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            *cp *= beta;
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
