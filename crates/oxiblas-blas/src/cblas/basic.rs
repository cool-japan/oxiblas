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

/// Reference-BLAS start offset (in elements) for a strided vector of length `n`.
///
/// The BLAS contract lets an increment be negative, which means the vector is
/// traversed physically back-to-front: logical element `i` (`0..n`) lives at
/// byte offset `((1 - n) + i) * inc`. The walk therefore starts at the
/// physically-last element `(n - 1) * |inc|` and steps by `inc` (negative) each
/// iteration. For `inc >= 0` the start offset is simply `0`.
///
/// # Why isize
///
/// The whole computation stays in `isize` so that `(1 - n) * inc` (a large
/// non-negative value when `inc < 0`) never underflows the way the naive
/// `i * inc as usize` pattern does — that pattern computes an astronomically
/// large `usize` and reads far out of bounds, which is the UB this helper
/// exists to prevent.
#[inline]
fn vector_start_offset(n: usize, inc: isize) -> isize {
    if inc < 0 { (1 - n as isize) * inc } else { 0 }
}

/// Validates the leading dimension of the `m`×`n` matrix in a GEMV call.
///
/// Reference (x)GEMV requires `lda >= max(1, m)` for column-major storage and
/// `lda >= max(1, n)` for row-major storage (the leading dimension must span a
/// full column/row). A too-small `lda` makes the internal `a.add(row + col*lda)`
/// indexing alias/overrun the buffer, so we reject it. Callers treat `false` as
/// a no-op, mirroring reference BLAS's xerbla-then-return behavior (a void C ABI
/// cannot surface an error code, and unwinding across `extern "C"` is UB).
#[inline]
fn gemv_params_valid(layout: CblasLayout, m: i32, n: i32, lda: i32) -> bool {
    let min_lda = match layout {
        CblasLayout::ColMajor => m.max(1),
        CblasLayout::RowMajor => n.max(1),
    };
    lda >= min_lda
}

/// Validates the shape parameters of a GEMM call.
///
/// Returns `false` (⇒ the caller must no-op, per the xerbla-then-return
/// convention) when any of these reference-BLAS constraints is violated:
/// * `k < 0` — a negative contraction length, which otherwise becomes a huge
///   `usize` and drives an unbounded / out-of-bounds accumulation loop.
/// * `lda`/`ldb`/`ldc` smaller than the minimum leading dimension the storage
///   order requires. For column-major the leading dimension is the number of
///   rows of the *stored* operand; for row-major it is the number of columns.
///
/// `m`/`n` are assumed already validated (`> 0`) by the caller.
#[inline]
fn gemm_params_valid(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
) -> bool {
    if k < 0 {
        return false;
    }
    let is_notrans = |t: CblasTranspose| matches!(t, CblasTranspose::NoTrans);
    let (nrowa, nrowb, nrowc) = match layout {
        CblasLayout::ColMajor => {
            // Stored A is m×k (NoTrans) or k×m (Trans); B is k×n or n×k; C is m×n.
            let nrowa = if is_notrans(transa) { m } else { k };
            let nrowb = if is_notrans(transb) { k } else { n };
            (nrowa, nrowb, m)
        }
        CblasLayout::RowMajor => {
            // Row-major leading dim is the row length: A is m×k (NoTrans) or
            // k×m (Trans) ⇒ length k or m; B is k×n or n×k ⇒ length n or k;
            // C is m×n ⇒ length n.
            let nrowa = if is_notrans(transa) { k } else { m };
            let nrowb = if is_notrans(transb) { n } else { k };
            (nrowa, nrowb, n)
        }
    };
    lda >= nrowa.max(1) && ldb >= nrowb.max(1) && ldc >= nrowc.max(1)
}

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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = 0.0;
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = 0.0f32;
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
    }
    result
}

/// Double precision Euclidean norm.
///
/// Uses Blue's algorithm for numerical stability with unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64 {
    // Reference DNRM2 is defined only for incx >= 1; anything else returns 0.
    if n <= 0 || incx <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation with Blue's algorithm
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: strided. Use Blue's scaled sum-of-squares (mirroring the core
    // `level1::nrm2` implementation) instead of a naive Σx² so that vectors with
    // very large or very small magnitudes do not overflow/underflow to inf/0.
    let incx = incx as isize;
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    let mut ix = 0isize;
    for _ in 0..n {
        let abs_xi = (*x.offset(ix)).abs();
        // `!= 0.0` (not `> 0.0`): a NaN element must still enter this branch
        // and poison the accumulator, matching `nrm2_fold` in
        // `level1/nrm2.rs` (`>` is always false for NaN).
        if abs_xi != 0.0 {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = (ssq * t).mul_add(t, 1.0);
                scale = abs_xi;
            } else {
                let t = if scale == abs_xi { 1.0 } else { abs_xi / scale };
                ssq += t * t;
            }
        }
        ix += incx;
    }
    scale * ssq.sqrt()
}

/// Single precision Euclidean norm.
///
/// Uses Blue's algorithm for numerical stability with unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32 {
    // Reference SNRM2 is defined only for incx >= 1; anything else returns 0.
    if n <= 0 || incx <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride - use optimized implementation with Blue's algorithm
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::nrm2(x_slice);
    }

    // Fallback: strided. Use Blue's scaled sum-of-squares (mirroring the core
    // `level1::nrm2` implementation) instead of a naive Σx² so that vectors with
    // very large or very small magnitudes do not overflow/underflow to inf/0.
    let incx = incx as isize;
    let mut scale = 0.0f32;
    let mut ssq = 1.0f32;
    let mut ix = 0isize;
    for _ in 0..n {
        let abs_xi = (*x.offset(ix)).abs();
        // `!= 0.0` (not `> 0.0`): a NaN element must still enter this branch
        // and poison the accumulator, matching `nrm2_fold` in
        // `level1/nrm2.rs` (`>` is always false for NaN).
        if abs_xi != 0.0 {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = (ssq * t).mul_add(t, 1.0);
                scale = abs_xi;
            } else {
                let t = if scale == abs_xi { 1.0 } else { abs_xi / scale };
                ssq += t * t;
            }
        }
        ix += incx;
    }
    scale * ssq.sqrt()
}

/// Double precision sum of absolute values.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64 {
    // Reference DASUM is a no-op returning 0 for incx <= 0.
    if n <= 0 || incx <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::asum(x_slice);
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut result = 0.0;
    let mut ix = 0isize;
    for _ in 0..n {
        result += (*x.offset(ix)).abs();
        ix += incx;
    }
    result
}

/// Single precision sum of absolute values.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sasum(n: i32, x: *const f32, incx: i32) -> f32 {
    // Reference SASUM is a no-op returning 0 for incx <= 0.
    if n <= 0 || incx <= 0 {
        return 0.0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::asum(x_slice);
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut result = 0.0f32;
    let mut ix = 0isize;
    for _ in 0..n {
        result += (*x.offset(ix)).abs();
        ix += incx;
    }
    result
}

/// Double precision index of maximum absolute value.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32 {
    // Reference IDAMAX is a no-op returning 0 for incx <= 0.
    if n <= 0 || incx <= 0 {
        return 0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::iamax(x_slice) as i32;
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut max_idx = 0;
    let mut max_val = (*x).abs();
    let mut ix = incx;
    for i in 1..n {
        let val = (*x.offset(ix)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
        ix += incx;
    }
    max_idx as i32
}

/// Single precision index of maximum absolute value.
///
/// Uses optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_isamax(n: i32, x: *const f32, incx: i32) -> i32 {
    // Reference ISAMAX is a no-op returning 0 for incx <= 0.
    if n <= 0 || incx <= 0 {
        return 0;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts(x, n);
        return level1::iamax(x_slice) as i32;
    }

    // Fallback: positive non-unit stride (forward traversal is correct here
    // because incx <= 0 was rejected above).
    let incx = incx as isize;
    let mut max_idx = 0;
    let mut max_val = (*x).abs();
    let mut ix = incx;
    for i in 1..n {
        let val = (*x.offset(ix)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
        ix += incx;
    }
    max_idx as i32
}

/// Double precision vector scaling: x = alpha * x.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32) {
    // Reference DSCAL is a no-op for incx <= 0.
    if n <= 0 || incx <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        level1::scal(alpha, x_slice);
        return;
    }

    // Fallback: positive non-unit stride (incx <= 0 rejected above).
    let incx = incx as isize;
    let mut ix = 0isize;
    for _ in 0..n {
        *x.offset(ix) *= alpha;
        ix += incx;
    }
}

/// Single precision vector scaling: x = alpha * x.
///
/// Uses SIMD-optimized implementation for unit stride.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32) {
    // Reference SSCAL is a no-op for incx <= 0.
    if n <= 0 || incx <= 0 {
        return;
    }

    let n = n as usize;

    // Fast path: unit stride
    if incx == 1 {
        let x_slice = std::slice::from_raw_parts_mut(x, n);
        level1::scal(alpha, x_slice);
        return;
    }

    // Fallback: positive non-unit stride (incx <= 0 rejected above).
    let incx = incx as isize;
    let mut ix = 0isize;
    for _ in 0..n {
        *x.offset(ix) *= alpha;
        ix += incx;
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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) += alpha * *x.offset(ix);
        ix += incx;
        iy += incy;
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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) += alpha * *x.offset(ix);
        ix += incx;
        iy += incy;
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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) = *x.offset(ix);
        ix += incx;
        iy += incy;
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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        *y.offset(iy) = *x.offset(ix);
        ix += incx;
        iy += incy;
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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        std::ptr::swap(x.offset(ix), y.offset(iy));
        ix += incx;
        iy += incy;
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

    // Fallback: non-unit stride (incx/incy may be negative — a spec-valid input).
    let incx = incx as isize;
    let incy = incy as isize;
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);
    for _ in 0..n {
        std::ptr::swap(x.offset(ix), y.offset(iy));
        ix += incx;
        iy += incy;
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
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex64::new(0.0, 0.0);
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
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
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex64::new(0.0, 0.0);
    for _ in 0..n {
        result += (*x.offset(ix)).conj() * *y.offset(iy);
        ix += incx;
        iy += incy;
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
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex32::new(0.0, 0.0);
    for _ in 0..n {
        result += *x.offset(ix) * *y.offset(iy);
        ix += incx;
        iy += incy;
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
    let mut ix = vector_start_offset(n, incx);
    let mut iy = vector_start_offset(n, incy);

    let mut result = Complex32::new(0.0, 0.0);
    for _ in 0..n {
        result += (*x.offset(ix)).conj() * *y.offset(iy);
        ix += incx;
        iy += incy;
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
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
        || alpha.is_null()
        || beta.is_null()
    {
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
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite with 0 rather than compute 0*C (which would
    // turn any NaN/Inf in uninitialized C into NaN).
    let zero = Complex64::new(0.0, 0.0);
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == zero {
                *cp = zero;
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
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
        || alpha.is_null()
        || beta.is_null()
    {
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
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite with 0 rather than compute 0*C (which would
    // turn any NaN/Inf in uninitialized C into NaN).
    let zero = Complex32::new(0.0, 0.0);
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == zero {
                *cp = zero;
            } else {
                *cp *= beta;
            }
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

#[cfg(test)]
mod tests {
    //! Regression tests for the negative-increment / beta==0 / parameter-
    //! validation hardening. Every strided routine family is exercised with a
    //! negative increment on a NON-palindromic vector so that forward vs.
    //! reference back-to-front traversal produce observably different results,
    //! and each is checked against an independent, manually reverse-indexed
    //! reference (not a copy of the implementation under test).
    use super::*;

    const EPS: f64 = 1e-12;
    const EPS_F32: f32 = 1e-6;

    // ---- Level 1: two-vector routines (ddot/sdot/axpy/copy/swap) ----------

    #[test]
    fn test_ddot_negative_incx() {
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let y = [10.0f64, 20.0, 30.0, 40.0];
        // Manual reference: incx=-1 reverses x, incy=+1 forward y.
        let expected: f64 = (0..4).map(|i| x[3 - i] * y[i]).sum();
        let got = unsafe { cblas_ddot(4, x.as_ptr(), -1, y.as_ptr(), 1) };
        assert!(
            (got - expected).abs() < EPS,
            "got {got}, expected {expected}"
        );
        // Sanity: differs from the naive forward dot (proves the fix matters).
        let forward: f64 = (0..4).map(|i| x[i] * y[i]).sum();
        assert!((expected - forward).abs() > 1.0);
    }

    #[test]
    fn test_ddot_negative_both_mixed_stride() {
        // incx=-1, incy=-2 on independent, non-symmetric data.
        let x = [1.0f64, 2.0, 3.0]; // logical: x[2], x[1], x[0]
        let y = [5.0f64, 6.0, 7.0, 8.0, 9.0]; // incy=-2 ⇒ y[4], y[2], y[0]
        let expected = x[2] * y[4] + x[1] * y[2] + x[0] * y[0];
        let got = unsafe { cblas_ddot(3, x.as_ptr(), -1, y.as_ptr(), -2) };
        assert!(
            (got - expected).abs() < EPS,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_sdot_negative_incx() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let y = [10.0f32, 20.0, 30.0, 40.0];
        let expected: f32 = (0..4).map(|i| x[3 - i] * y[i]).sum();
        let got = unsafe { cblas_sdot(4, x.as_ptr(), -1, y.as_ptr(), 1) };
        assert!((got - expected).abs() < EPS_F32);
    }

    #[test]
    fn test_daxpy_negative_incx() {
        let alpha = 2.0f64;
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [10.0f64, 20.0, 30.0];
        let mut expected = y;
        // incx=-1 reverses x; incy=+1 forward y.
        for i in 0..3 {
            expected[i] += alpha * x[2 - i];
        }
        unsafe { cblas_daxpy(3, alpha, x.as_ptr(), -1, y.as_mut_ptr(), 1) };
        for i in 0..3 {
            assert!((y[i] - expected[i]).abs() < EPS, "index {i}");
        }
        // Observably different from forward axpy at both ends.
        assert!((y[0] - 16.0).abs() < EPS && (y[2] - 32.0).abs() < EPS);
    }

    #[test]
    fn test_saxpy_negative_incy() {
        let alpha = 3.0f32;
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [10.0f32, 20.0, 30.0];
        let mut expected = y;
        // incx=+1 forward x; incy=-1 reverses y writes.
        for i in 0..3 {
            expected[2 - i] += alpha * x[i];
        }
        unsafe { cblas_saxpy(3, alpha, x.as_ptr(), 1, y.as_mut_ptr(), -1) };
        for i in 0..3 {
            assert!((y[i] - expected[i]).abs() < EPS_F32, "index {i}");
        }
    }

    #[test]
    fn test_dcopy_negative_incx_reverses() {
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let mut y = [0.0f64; 4];
        unsafe { cblas_dcopy(4, x.as_ptr(), -1, y.as_mut_ptr(), 1) };
        // y = reverse(x)
        assert_eq!(y, [4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_scopy_negative_incx_reverses() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [0.0f32; 3];
        unsafe { cblas_scopy(3, x.as_ptr(), -1, y.as_mut_ptr(), 1) };
        assert_eq!(y, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_dswap_negative_incx() {
        let mut x = [1.0f64, 2.0, 3.0];
        let mut y = [10.0f64, 20.0, 30.0];
        // Reference: swap x[2-i] <-> y[i] for i = 0,1,2.
        unsafe { cblas_dswap(3, x.as_mut_ptr(), -1, y.as_mut_ptr(), 1) };
        assert_eq!(x, [30.0, 20.0, 10.0]);
        assert_eq!(y, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_sswap_negative_incx() {
        let mut x = [1.0f32, 2.0, 3.0];
        let mut y = [10.0f32, 20.0, 30.0];
        unsafe { cblas_sswap(3, x.as_mut_ptr(), -1, y.as_mut_ptr(), 1) };
        assert_eq!(x, [30.0, 20.0, 10.0]);
        assert_eq!(y, [3.0, 2.0, 1.0]);
    }

    // ---- Level 1: single-vector routines (scal/nrm2/asum/iamax) -----------

    #[test]
    fn test_dscal_negative_incx_is_noop() {
        // Reference DSCAL is a no-op for incx <= 0.
        let mut x = [1.0f64, 2.0, 3.0];
        unsafe { cblas_dscal(3, 5.0, x.as_mut_ptr(), -1) };
        assert_eq!(x, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dscal_positive_strided() {
        // Only elements at stride-2 positions are scaled.
        let mut x = [1.0f64, 9.0, 2.0, 9.0];
        unsafe { cblas_dscal(2, 10.0, x.as_mut_ptr(), 2) };
        assert_eq!(x, [10.0, 9.0, 20.0, 9.0]);
    }

    #[test]
    fn test_dnrm2_negative_incx_is_zero() {
        let x = [3.0f64, 4.0];
        let got = unsafe { cblas_dnrm2(2, x.as_ptr(), -1) };
        assert_eq!(got, 0.0);
    }

    #[test]
    fn test_dnrm2_strided() {
        let x = [3.0f64, 99.0, 4.0, 99.0];
        let got = unsafe { cblas_dnrm2(2, x.as_ptr(), 2) };
        assert!((got - 5.0).abs() < EPS);
    }

    #[test]
    fn test_dnrm2_strided_no_overflow() {
        // Blue's scaled accumulation must not overflow where a naive Σx² would.
        // Squaring 1e200 overflows f64, but ||[1e200, 1e200]||_2 = √2·1e200.
        let x = [1e200f64, 0.0, 1e200, 0.0];
        let got = unsafe { cblas_dnrm2(2, x.as_ptr(), 2) };
        let expected = std::f64::consts::SQRT_2 * 1e200;
        assert!(got.is_finite(), "norm overflowed to {got}");
        assert!((got - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn test_snrm2_strided_no_overflow() {
        let x = [1e20f32, 0.0, 1e20, 0.0];
        let got = unsafe { cblas_snrm2(2, x.as_ptr(), 2) };
        let expected = std::f32::consts::SQRT_2 * 1e20;
        assert!(got.is_finite());
        assert!((got - expected).abs() / expected < 1e-5);
    }

    #[test]
    fn test_dasum_negative_incx_is_zero() {
        let x = [1.0f64, -2.0, 3.0];
        let got = unsafe { cblas_dasum(3, x.as_ptr(), -1) };
        assert_eq!(got, 0.0);
    }

    #[test]
    fn test_dasum_strided() {
        let x = [1.0f64, 9.0, -2.0, 9.0, 3.0, 9.0];
        let got = unsafe { cblas_dasum(3, x.as_ptr(), 2) };
        assert!((got - 6.0).abs() < EPS);
    }

    #[test]
    fn test_idamax_negative_incx_is_zero() {
        let x = [1.0f64, -5.0, 3.0];
        let got = unsafe { cblas_idamax(3, x.as_ptr(), -1) };
        assert_eq!(got, 0);
    }

    #[test]
    fn test_idamax_strided() {
        // Strided view over [1, -5, 3]; max |.| is at logical index 1.
        let x = [1.0f64, 9.0, -5.0, 9.0, 3.0, 9.0];
        let got = unsafe { cblas_idamax(3, x.as_ptr(), 2) };
        assert_eq!(got, 1);
    }

    // ---- Level 2: GEMV ----------------------------------------------------

    #[test]
    fn test_dgemv_positive_stride_sanity() {
        // A = [[1,2],[3,4]] col-major; y = A*x with x = [1,1] ⇒ [3,7].
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let x = [1.0f64, 1.0];
        let mut y = [0.0f64; 2];
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                2,
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
        assert!((y[0] - 3.0).abs() < EPS && (y[1] - 7.0).abs() < EPS);
    }

    #[test]
    fn test_dgemv_negative_inc_and_beta_zero() {
        // A = [[1,2],[3,4]] col-major. incx=-1 ⇒ x is read reversed; incy=-1 ⇒
        // y is written reversed. beta=0 must overwrite the NaN garbage in y.
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let x = [10.0f64, 20.0]; // logical (reversed): [20, 10]
        let mut y = [f64::NAN; 2];
        // Independent reference.
        let x_log = [x[1], x[0]];
        let y_log = [
            a[0] * x_log[0] + a[2] * x_log[1], // row 0 · x
            a[1] * x_log[0] + a[3] * x_log[1], // row 1 · x
        ];
        // incy=-1 stores y_log[0] at y[1], y_log[1] at y[0].
        let expected = [y_log[1], y_log[0]];
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                -1,
                0.0,
                y.as_mut_ptr(),
                -1,
            );
        }
        assert!(y[0].is_finite() && y[1].is_finite(), "beta=0 left NaN in y");
        assert!((y[0] - expected[0]).abs() < EPS);
        assert!((y[1] - expected[1]).abs() < EPS);
    }

    #[test]
    fn test_sgemv_negative_incx() {
        let a = [1.0f32, 3.0, 2.0, 4.0];
        let x = [10.0f32, 20.0];
        let mut y = [0.0f32; 2];
        let x_log = [x[1], x[0]];
        let expected = [
            a[0] * x_log[0] + a[2] * x_log[1],
            a[1] * x_log[0] + a[3] * x_log[1],
        ];
        unsafe {
            cblas_sgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                -1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert!((y[0] - expected[0]).abs() < EPS_F32);
        assert!((y[1] - expected[1]).abs() < EPS_F32);
    }

    // ---- Complex dot products (zdotu/zdotc/cdotu/cdotc) -------------------

    #[test]
    fn test_zdotu_negative_incx() {
        let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
        // incx=-1 reverses x, incy=+1 forward y.
        let expected = x[1] * y[0] + x[0] * y[1];
        let mut got = Complex64::new(0.0, 0.0);
        unsafe { cblas_zdotu_sub(2, x.as_ptr(), -1, y.as_ptr(), 1, &mut got) };
        assert!((got.re - expected.re).abs() < EPS);
        assert!((got.im - expected.im).abs() < EPS);
        // Different from forward dot.
        let forward = x[0] * y[0] + x[1] * y[1];
        assert!((expected.im - forward.im).abs() > 1.0);
    }

    #[test]
    fn test_zdotc_negative_incx() {
        let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
        // conj(x) reversed dotted with forward y.
        let expected = x[1].conj() * y[0] + x[0].conj() * y[1];
        let mut got = Complex64::new(0.0, 0.0);
        unsafe { cblas_zdotc_sub(2, x.as_ptr(), -1, y.as_ptr(), 1, &mut got) };
        assert!((got.re - expected.re).abs() < EPS);
        assert!((got.im - expected.im).abs() < EPS);
    }

    #[test]
    fn test_cdotu_negative_incy() {
        let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];
        // incx=+1 forward x, incy=-1 reverses y.
        let expected = x[0] * y[1] + x[1] * y[0];
        let mut got = Complex32::new(0.0, 0.0);
        unsafe { cblas_cdotu_sub(2, x.as_ptr(), 1, y.as_ptr(), -1, &mut got) };
        assert!((got.re - expected.re).abs() < EPS_F32);
        assert!((got.im - expected.im).abs() < EPS_F32);
    }

    #[test]
    fn test_cdotc_negative_incx() {
        let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];
        let expected = x[1].conj() * y[0] + x[0].conj() * y[1];
        let mut got = Complex32::new(0.0, 0.0);
        unsafe { cblas_cdotc_sub(2, x.as_ptr(), -1, y.as_ptr(), 1, &mut got) };
        assert!((got.re - expected.re).abs() < EPS_F32);
        assert!((got.im - expected.im).abs() < EPS_F32);
    }

    // ---- Level 3: beta==0 with uninitialized C ---------------------------

    #[test]
    fn test_dgemm_beta_zero_ignores_uninit_c() {
        // Hit the scalar transpose fallback (transa=Trans) with NaN in C and
        // beta=0: the NaN must be overwritten, not propagated via 0*NaN.
        let a = [1.0f64, 3.0, 2.0, 4.0]; // stored col-major 2x2
        let b = [1.0f64, 0.0, 0.0, 1.0]; // identity
        let mut c = [f64::NAN; 4];
        unsafe {
            cblas_dgemm(
                CblasLayout::ColMajor,
                CblasTranspose::Trans,
                CblasTranspose::NoTrans,
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
        // C = A^T (stored) = [[1,3],[2,4]] col-major ⇒ [1, 2, 3, 4].
        assert!(
            c.iter().all(|v| v.is_finite()),
            "beta=0 propagated NaN: {c:?}"
        );
        assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zgemm_beta_zero_ignores_uninit_c() {
        let a = [Complex64::new(2.0, 0.0)];
        let b = [Complex64::new(3.0, 0.0)];
        let mut c = [Complex64::new(f64::NAN, f64::NAN)];
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(0.0, 0.0);
        unsafe {
            cblas_zgemm(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                1,
                1,
                1,
                &alpha,
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                &beta,
                c.as_mut_ptr(),
                1,
            );
        }
        assert!(c[0].re.is_finite() && c[0].im.is_finite());
        assert!((c[0].re - 6.0).abs() < EPS && c[0].im.abs() < EPS);
    }

    // ---- Level 3: parameter validation (negative k / bad ld / null) -------

    #[test]
    fn test_dgemm_negative_k_is_noop() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [1.0f64, 2.0, 3.0, 4.0];
        let mut c = [7.0f64; 4];
        unsafe {
            cblas_dgemm(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                2,
                2,
                -1, // invalid k
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                1.0,
                c.as_mut_ptr(),
                2,
            );
        }
        assert_eq!(c, [7.0; 4]);
    }

    #[test]
    fn test_dgemm_undersized_ldc_is_noop() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [1.0f64, 2.0, 3.0, 4.0];
        let mut c = [7.0f64; 4];
        unsafe {
            cblas_dgemm(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                1.0,
                c.as_mut_ptr(),
                1, // ldc < max(1, m) = 2
            );
        }
        assert_eq!(c, [7.0; 4]);
    }

    #[test]
    fn test_dgemm_null_pointer_is_noop() {
        let b = [1.0f64, 2.0, 3.0, 4.0];
        let mut c = [7.0f64; 4];
        unsafe {
            cblas_dgemm(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                std::ptr::null(), // null A
                2,
                b.as_ptr(),
                2,
                1.0,
                c.as_mut_ptr(),
                2,
            );
        }
        assert_eq!(c, [7.0; 4]);
    }

    #[test]
    fn test_dgemv_undersized_lda_is_noop() {
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let x = [1.0f64, 1.0];
        let mut y = [7.0f64; 2];
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                2,
                2,
                1.0,
                a.as_ptr(),
                1, // lda < max(1, m) = 2
                x.as_ptr(),
                1,
                1.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(y, [7.0; 2]);
    }
}
