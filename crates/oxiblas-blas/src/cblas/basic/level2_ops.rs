//! CBLAS Level 2 (matrix-vector) real operations: GEMV.

use super::super::types::*;
use super::super::validate::{gemv_params_valid, inc_valid};
use super::vector_start_offset;
/// Double precision GEMV: y = alpha * op(A) * x + beta * y.
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
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f64`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
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
    // Reference ?GEMV rejects a zero increment (xerbla) and returns without
    // touching y: `inc == 0` would alias every element of `x`/`y` onto element 0.
    if !gemv_params_valid(layout, m, n, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    // Reference quick return: nothing to do for alpha == 0 && beta == 1.
    if alpha == 0.0 && beta == 1.0 {
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

    // Scale y by beta (reference: skipped entirely for beta == 1). When beta == 0
    // the BLAS contract says y need not be initialized, so we overwrite rather
    // than multiply (avoids 0*NaN = NaN).
    if beta != 1.0 {
        for i in 0..rows {
            let yp = y.offset(y_start + i as isize * incy);
            if beta == 0.0 {
                *yp = 0.0;
            } else {
                *yp *= beta;
            }
        }
    }

    // alpha == 0: y := beta * y is all that is left. A and x are never read, so
    // NaN/Inf in them must not reach y.
    if alpha == 0.0 {
        return;
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
/// - `x` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from at every offset the strided walk implied by `incx`
///   reaches (see this module's vector start-offset helper — a negative
///   `incx` walks the vector back-to-front rather than out of bounds).
/// - `y` must be non-null whenever the dimension it indexes is positive,
///   properly aligned for `f32`, and point to a buffer large enough
///   to be read from and written to at every offset the strided walk implied by `incy`
///   reaches (see this module's vector start-offset helper — a negative
///   `incy` walks the vector back-to-front rather than out of bounds).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
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
    // Zero increments are rejected, as in `cblas_dgemv`.
    if !gemv_params_valid(layout, m, n, lda) || !inc_valid(incx) || !inc_valid(incy) {
        return;
    }
    if a.is_null() || x.is_null() || y.is_null() {
        return;
    }

    // Reference quick return: nothing to do for alpha == 0 && beta == 1.
    if alpha == 0.0 && beta == 1.0 {
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

    // beta == 0 overwrites y (uninitialized y allowed by the BLAS contract);
    // beta == 1 skips the scaling pass, as in the reference.
    if beta != 1.0 {
        for i in 0..rows {
            let yp = y.offset(y_start + i as isize * incy);
            if beta == 0.0 {
                *yp = 0.0;
            } else {
                *yp *= beta;
            }
        }
    }

    // alpha == 0: A and x are never read (see `cblas_dgemv`).
    if alpha == 0.0 {
        return;
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

#[cfg(test)]
mod tests {
    use super::*;

    const LAYOUTS: [CblasLayout; 2] = [CblasLayout::ColMajor, CblasLayout::RowMajor];
    const TRANSES: [CblasTranspose; 3] = [
        CblasTranspose::NoTrans,
        CblasTranspose::Trans,
        CblasTranspose::ConjTrans,
    ];

    // ---- Issue #9: alpha == 0 must not read A or x -----------------------------

    /// Reporter's repro: `cblas_dgemv(Col, N, 1, 1, 0.0, A=[NaN], 1, x=[1], 1, 1.0, y=[2], 1)`
    /// must leave `y = [2]` (also for A = Inf): reference ?GEMV returns before touching A.
    #[test]
    fn test_issue_9_dgemv_alpha_zero_does_not_read_a() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for layout in LAYOUTS {
                for trans in TRANSES {
                    let a = [bad];
                    let x = [1.0f64];
                    let mut y = [2.0f64];
                    // SAFETY: 1x1 operands with lda = 1 and unit increments.
                    unsafe {
                        cblas_dgemv(
                            layout,
                            trans,
                            1,
                            1,
                            0.0,
                            a.as_ptr(),
                            1,
                            x.as_ptr(),
                            1,
                            1.0,
                            y.as_mut_ptr(),
                            1,
                        );
                    }
                    assert_eq!(y, [2.0], "{layout:?}/{trans:?} with A = {bad}");
                }
            }
        }
    }

    #[test]
    fn test_issue_9_dgemv_alpha_zero_does_not_read_x() {
        let a = [3.0f64];
        let x = [f64::NAN];
        let mut y = [2.0f64];
        // SAFETY: 1x1 operands with lda = 1 and unit increments.
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                1,
                1,
                0.0,
                a.as_ptr(),
                1,
                x.as_ptr(),
                1,
                1.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(y, [2.0]);
    }

    /// alpha == 0 still applies `y := beta * y` (A and x untouched): the early
    /// exit must come AFTER the beta scaling, not before it.
    #[test]
    fn test_issue_9_dgemv_alpha_zero_still_scales_y_by_beta() {
        let a = [f64::NAN];
        let x = [f64::NAN];
        let mut y = [3.0f64];
        // SAFETY: 1x1 operands with lda = 1 and unit increments.
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                1,
                1,
                0.0,
                a.as_ptr(),
                1,
                x.as_ptr(),
                1,
                2.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(y, [6.0]);
    }

    /// alpha == 0 and beta == 0 zero-fills y without reading it (NaN in y is discarded).
    #[test]
    fn test_issue_9_dgemv_alpha_zero_beta_zero_zero_fills_nan_y() {
        let a = [f64::NAN; 6];
        let x = [f64::INFINITY; 2];
        let mut y = [f64::NAN; 3];
        // SAFETY: A is 3x2 col-major (lda = 3), x has 2 and y has 3 elements.
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                3,
                2,
                0.0,
                a.as_ptr(),
                3,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(y, [0.0; 3]);
    }

    /// Strided / reversed vectors: beta scales every logical element of y once.
    #[test]
    fn test_issue_9_dgemv_alpha_zero_strided_vectors() {
        let a = [f64::NAN; 6];
        let x = [f64::NAN; 5]; // 2 logical elements at incx = -4 span 1 + 4 = 5 slots
        let mut y = [1.0f64, 2.0, 3.0];
        // SAFETY: A is 3x2 col-major (lda = 3); x spans 5 slots; y has 3 elements.
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                3,
                2,
                0.0,
                a.as_ptr(),
                3,
                x.as_ptr(),
                -4,
                3.0,
                y.as_mut_ptr(),
                -1,
            );
        }
        assert_eq!(y, [3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_issue_9_sgemv_alpha_zero_does_not_read_a_or_x() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for layout in LAYOUTS {
                for trans in TRANSES {
                    let a = [bad];
                    let x = [bad];
                    let mut y = [2.0f32];
                    // SAFETY: 1x1 operands with lda = 1 and unit increments.
                    unsafe {
                        cblas_sgemv(
                            layout,
                            trans,
                            1,
                            1,
                            0.0,
                            a.as_ptr(),
                            1,
                            x.as_ptr(),
                            1,
                            1.0,
                            y.as_mut_ptr(),
                            1,
                        );
                    }
                    assert_eq!(y, [2.0], "{layout:?}/{trans:?} with A = x = {bad}");
                }
            }
        }
    }

    #[test]
    fn test_issue_9_sgemv_alpha_zero_still_scales_y_by_beta() {
        let a = [f32::NAN];
        let x = [f32::NAN];
        let mut y = [3.0f32];
        // SAFETY: 1x1 operands with lda = 1 and unit increments.
        unsafe {
            cblas_sgemv(
                CblasLayout::RowMajor,
                CblasTranspose::Trans,
                1,
                1,
                0.0,
                a.as_ptr(),
                1,
                x.as_ptr(),
                1,
                2.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(y, [6.0]);
    }

    #[test]
    fn test_issue_9_sgemv_alpha_zero_beta_zero_zero_fills_nan_y() {
        let a = [f32::NAN; 6];
        let x = [f32::INFINITY; 3];
        let mut y = [f32::NAN; 2];
        // SAFETY: A is 2x3 row-major (lda = 3), x has 3 and y has 2 elements.
        unsafe {
            cblas_sgemv(
                CblasLayout::RowMajor,
                CblasTranspose::NoTrans,
                2,
                3,
                0.0,
                a.as_ptr(),
                3,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(y, [0.0; 2]);
    }

    // ---- Issue #10: zero increments are rejected (y untouched) -----------------

    /// Reporter's repro: m=3, n=1, A=[1,2,3], x=[5], incx=0. Reference CBLAS
    /// rejects it (xerbla) and leaves y untouched; the old code computed [5,10,15].
    #[test]
    fn test_issue_10_dgemv_zero_incx_is_rejected() {
        let a = [1.0f64, 2.0, 3.0];
        let x = [5.0f64];
        let mut y = [7.0f64, 8.0, 9.0];
        // SAFETY: A is 3x1 col-major (lda = 3); x has 1 element; y has 3.
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                3,
                1,
                1.0,
                a.as_ptr(),
                3,
                x.as_ptr(),
                0,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(y, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_issue_10_dgemv_zero_incy_is_rejected() {
        let a = [1.0f64, 2.0, 3.0];
        let x = [5.0f64];
        let mut y = [7.0f64, 8.0, 9.0];
        // SAFETY: A is 3x1 col-major (lda = 3); x has 1 element; y has 3.
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                3,
                1,
                1.0,
                a.as_ptr(),
                3,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                0,
            );
        }
        assert_eq!(y, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_issue_10_sgemv_zero_incx_and_incy_are_rejected() {
        let a = [1.0f32, 2.0, 3.0];
        let x = [5.0f32];
        for (incx, incy) in [(0, 1), (1, 0), (0, 0)] {
            let mut y = [7.0f32, 8.0, 9.0];
            // SAFETY: A is 3x1 col-major (lda = 3); x has 1 element; y has 3.
            unsafe {
                cblas_sgemv(
                    CblasLayout::ColMajor,
                    CblasTranspose::NoTrans,
                    3,
                    1,
                    1.0,
                    a.as_ptr(),
                    3,
                    x.as_ptr(),
                    incx,
                    0.0,
                    y.as_mut_ptr(),
                    incy,
                );
            }
            assert_eq!(y, [7.0, 8.0, 9.0], "incx = {incx}, incy = {incy}");
        }
    }

    /// Control: the same call with valid unit increments still computes [5, 10, 15].
    #[test]
    fn test_issue_10_gemv_unit_increments_still_compute() {
        let a = [1.0f64, 2.0, 3.0];
        let x = [5.0f64];
        let mut y = [7.0f64, 8.0, 9.0];
        // SAFETY: A is 3x1 col-major (lda = 3); x has 1 element; y has 3.
        unsafe {
            cblas_dgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                3,
                1,
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
        assert_eq!(y, [5.0, 10.0, 15.0]);
    }
}
