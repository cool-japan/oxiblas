//! CBLAS Level 2 (matrix-vector) real operations: GEMV.
//!
//! On aarch64 (without the `force-scalar` feature), calls with unit increments
//! are forwarded to [`crate::level2::gemv`](fn@crate::level2::gemv) and its
//! NEON kernels: a RowMajor `m x n` matrix with leading dimension `lda` is the
//! ColMajor `n x m` matrix `Aᵀ` with the same `lda`, so RowMajor `NoTrans`
//! runs as ColMajor `Trans` and RowMajor `Trans` / `ConjTrans` as ColMajor
//! `NoTrans`. Other increments and other targets keep the strided reference
//! loops.

use super::super::types::*;
use super::super::validate::{gemv_params_valid, inc_valid};
use super::vector_start_offset;
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
use crate::level1::fits_isize_bytes;
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
use crate::level2::{GemvTrans, gemv};
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
use oxiblas_core::scalar::Field;
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
use oxiblas_matrix::MatRef;

/// The column-major view and transpose that evaluate a CBLAS GEMV call:
/// `(view_rows, view_cols, op)` for a stored `m x n` matrix in `layout`.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
fn colmajor_gemv_form(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: usize,
    n: usize,
) -> (usize, usize, GemvTrans) {
    match (layout, trans) {
        (CblasLayout::ColMajor, CblasTranspose::NoTrans) => (m, n, GemvTrans::NoTrans),
        (CblasLayout::ColMajor, CblasTranspose::Trans | CblasTranspose::ConjTrans) => {
            (m, n, GemvTrans::Trans)
        }
        (CblasLayout::RowMajor, CblasTranspose::NoTrans) => (n, m, GemvTrans::Trans),
        (CblasLayout::RowMajor, CblasTranspose::Trans | CblasTranspose::ConjTrans) => {
            (n, m, GemvTrans::NoTrans)
        }
    }
}

/// Unit-increment GEMV of `cblas_dgemv` / `cblas_sgemv`: forwards to
/// [`crate::level2::gemv`](fn@crate::level2::gemv) (its NEON kernels) on the
/// column-major view given by [`colmajor_gemv_form`].
///
/// Kept out of line (`#[inline(never)]`) so that `cblas_dgemv` stays near its
/// size from before this fast path existed. The strided reference loops after
/// it compile to the same instructions either way; their timing is sensitive
/// to where those loops end up in the binary (a few percent).
///
/// # Safety
///
/// The CBLAS parameters must have been validated (`gemv_params_valid`, so
/// `lda` is a valid leading dimension of the `view_rows x view_cols` view),
/// `a` must hold every element of that view, `x` must hold `cols` and `y`
/// `rows` elements at unit stride, `y` must not alias `a` or `x`, and both
/// lengths must fit in `isize::MAX` bytes.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[inline(never)]
unsafe fn gemv_unit_stride<T: Field>(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: usize,
    n: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    x: *const T,
    beta: T,
    y: *mut T,
    rows: usize,
    cols: usize,
) {
    let (view_rows, view_cols, op) = colmajor_gemv_form(layout, trans, m, n);
    // SAFETY: guaranteed by the caller (see `# Safety`): `lda` is a valid
    // leading dimension of the view and `a` holds all of it, `x` holds `cols`
    // and `y` holds `rows` elements at unit stride, `y` aliases neither `a`
    // nor `x`, and both lengths fit in `isize::MAX` bytes.
    let (a_view, xs, ys) = (
        MatRef::new(a, view_rows, view_cols, lda),
        std::slice::from_raw_parts(x, cols),
        std::slice::from_raw_parts_mut(y, rows),
    );
    gemv(op, alpha, a_view, xs, beta, ys);
}

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

    // Determine effective dimensions based on transpose. `x` has `cols`
    // elements, `y` has `rows` elements.
    let (rows, cols) = match trans {
        CblasTranspose::NoTrans => (m, n),
        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
    };

    // Unit increments: the optimised driver, which applies the same beta rule
    // (`beta == 0` overwrites y without reading it, `beta == 1` leaves it) and
    // never reads A or x when alpha == 0. It runs out of line (see
    // `gemv_unit_stride`).
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    if incx == 1 && incy == 1 && fits_isize_bytes::<f64>(rows) && fits_isize_bytes::<f64>(cols) {
        // SAFETY: the parameters were validated above (`lda` covers the stored
        // rows of the layout, so it is a valid leading dimension of the
        // column-major view) and the caller guarantees that `a` holds every
        // element of that view, that `x` holds `cols` and `y` holds `rows`
        // elements at unit stride, and that `y` does not alias `a` or `x`;
        // both lengths fit in `isize::MAX` bytes (checked above).
        gemv_unit_stride(layout, trans, m, n, alpha, a, lda, x, beta, y, rows, cols);
        return;
    }

    let incx = incx as isize;
    let incy = incy as isize;

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

    let (rows, cols) = match trans {
        CblasTranspose::NoTrans => (m, n),
        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
    };

    // Unit increments: the optimised driver, out of line (see `cblas_dgemv`).
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    if incx == 1 && incy == 1 && fits_isize_bytes::<f32>(rows) && fits_isize_bytes::<f32>(cols) {
        // SAFETY: as in `cblas_dgemv`: validated parameters, caller-guaranteed
        // extents of `a`, `x` (`cols`) and `y` (`rows`) at unit stride, no
        // aliasing of `y`, and both lengths fit in `isize::MAX` bytes.
        gemv_unit_stride(layout, trans, m, n, alpha, a, lda, x, beta, y, rows, cols);
        return;
    }

    let incx = incx as isize;
    let incy = incy as isize;

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

    // ---- Unit-stride fast path (level2::gemv) vs the CBLAS definition ----------

    /// `(alpha, beta)` pairs including the special values of the BLAS contract.
    const ALPHA_BETA: [(f64, f64); 7] = [
        (1.0, 0.0),
        (1.5, -0.5),
        (-2.0, 1.0),
        (0.0, 0.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ];

    /// Small-integer test data (exact in any summation order).
    fn int_data(len: usize, seed: usize, k: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i * 7 + seed * 13 + 3) % (2 * k + 1)) as f64 - k as f64)
            .collect()
    }

    /// The CBLAS definition evaluated naively: element `(i, j)` of the stored
    /// `m x n` matrix is `a[i + j*lda]` (ColMajor) or `a[i*lda + j]` (RowMajor);
    /// `beta == 0` discards `y`, and `alpha == 0` skips the product entirely.
    #[allow(clippy::too_many_arguments)]
    fn cblas_gemv_definition(
        layout: CblasLayout,
        trans: CblasTranspose,
        m: usize,
        n: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        x: &[f64],
        beta: f64,
        y0: &[f64],
    ) -> Vec<f64> {
        let at = |i: usize, j: usize| match layout {
            CblasLayout::ColMajor => a[i + j * lda],
            CblasLayout::RowMajor => a[i * lda + j],
        };
        let (rows, cols) = match trans {
            CblasTranspose::NoTrans => (m, n),
            CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
        };
        (0..rows)
            .map(|r| {
                if alpha == 0.0 && beta == 1.0 {
                    return y0[r];
                }
                let base = if beta == 0.0 { 0.0 } else { beta * y0[r] };
                if alpha == 0.0 {
                    return base;
                }
                let s: f64 = (0..cols)
                    .map(|c| {
                        let v = match trans {
                            CblasTranspose::NoTrans => at(r, c),
                            CblasTranspose::Trans | CblasTranspose::ConjTrans => at(c, r),
                        };
                        v * x[c]
                    })
                    .sum();
                base + alpha * s
            })
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Every layout/transpose/alpha/beta combination, unit increments (the
    /// `level2::gemv` path) and increments 2 / -1 (the strided loops), with NaN
    /// in `y` whenever `beta == 0`: all must equal the definition bit for bit.
    #[test]
    fn test_dgemv_all_layouts_and_transposes_match_definition() {
        for (m, n) in [(1usize, 1usize), (3, 5), (5, 3), (17, 9), (9, 17), (40, 33)] {
            for layout in LAYOUTS {
                let lda = match layout {
                    CblasLayout::ColMajor => m + 2,
                    CblasLayout::RowMajor => n + 3,
                };
                let stored = match layout {
                    CblasLayout::ColMajor => lda * n,
                    CblasLayout::RowMajor => lda * m,
                };
                let a = int_data(stored, m + n, 6);
                for trans in TRANSES {
                    let (rows, cols) = match trans {
                        CblasTranspose::NoTrans => (m, n),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
                    };
                    let x = int_data(cols, 1, 4);
                    for (alpha, beta) in ALPHA_BETA {
                        let y0 = if beta == 0.0 {
                            vec![f64::NAN; rows]
                        } else {
                            int_data(rows, 2, 5)
                        };
                        let want = cblas_gemv_definition(
                            layout, trans, m, n, alpha, &a, lda, &x, beta, &y0,
                        );
                        let what = format!("{layout:?}/{trans:?} {m}x{n} a={alpha} b={beta}");

                        let mut y = y0.clone();
                        // SAFETY: `a` holds the stored matrix with leading
                        // dimension `lda`; x has `cols`, y has `rows` elements.
                        unsafe {
                            cblas_dgemv(
                                layout,
                                trans,
                                m as i32,
                                n as i32,
                                alpha,
                                a.as_ptr(),
                                lda as i32,
                                x.as_ptr(),
                                1,
                                beta,
                                y.as_mut_ptr(),
                                1,
                            );
                        }
                        assert_eq!(bits(&y), bits(&want), "unit stride {what}");

                        // Strided: x at increment 2, y at increment -1.
                        let mut x2 = vec![f64::NAN; 2 * cols];
                        for (c, &v) in x.iter().enumerate() {
                            x2[2 * c] = v;
                        }
                        let mut yr: Vec<f64> = y0.iter().rev().copied().collect();
                        // SAFETY: x2 spans 2*cols slots (incx = 2); yr holds
                        // `rows` elements walked back to front (incy = -1).
                        unsafe {
                            cblas_dgemv(
                                layout,
                                trans,
                                m as i32,
                                n as i32,
                                alpha,
                                a.as_ptr(),
                                lda as i32,
                                x2.as_ptr(),
                                2,
                                beta,
                                yr.as_mut_ptr(),
                                -1,
                            );
                        }
                        yr.reverse();
                        assert_eq!(bits(&yr), bits(&want), "strided {what}");
                    }
                }
            }
        }
    }

    /// f32 counterpart: unit stride and strided, against the definition
    /// (evaluated in f64 on exactly representable data, then rounded).
    #[test]
    fn test_sgemv_all_layouts_and_transposes_match_definition() {
        for (m, n) in [(1usize, 1usize), (3, 5), (5, 3), (17, 9), (9, 17), (40, 33)] {
            for layout in LAYOUTS {
                let lda = match layout {
                    CblasLayout::ColMajor => m + 1,
                    CblasLayout::RowMajor => n + 2,
                };
                let stored = match layout {
                    CblasLayout::ColMajor => lda * n,
                    CblasLayout::RowMajor => lda * m,
                };
                let a = int_data(stored, m * n, 6);
                let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
                for trans in TRANSES {
                    let (rows, cols) = match trans {
                        CblasTranspose::NoTrans => (m, n),
                        CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
                    };
                    let x = int_data(cols, 3, 4);
                    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
                    for (alpha, beta) in ALPHA_BETA {
                        let y0 = if beta == 0.0 {
                            vec![f64::NAN; rows]
                        } else {
                            int_data(rows, 4, 5)
                        };
                        let want: Vec<u32> = cblas_gemv_definition(
                            layout, trans, m, n, alpha, &a, lda, &x, beta, &y0,
                        )
                        .iter()
                        .map(|&v| (v as f32).to_bits())
                        .collect();
                        let what = format!("{layout:?}/{trans:?} {m}x{n} a={alpha} b={beta}");
                        for inc in [1i32, -2] {
                            let span = |len: usize| (len - 1) * inc.unsigned_abs() as usize + 1;
                            let mut yb = vec![f32::NAN; span(rows)];
                            let mut xb = vec![f32::NAN; span(cols)];
                            // Logical element k lives at slot `start + k*inc`.
                            let slot = |len: usize, k: usize| {
                                if inc > 0 {
                                    k * inc as usize
                                } else {
                                    (len - 1 - k) * inc.unsigned_abs() as usize
                                }
                            };
                            for (k, &v) in y0.iter().enumerate() {
                                yb[slot(rows, k)] = v as f32;
                            }
                            for (k, &v) in x32.iter().enumerate() {
                                xb[slot(cols, k)] = v;
                            }
                            // SAFETY: `a32` holds the stored matrix; xb / yb span
                            // the strided extents of `cols` / `rows` elements.
                            unsafe {
                                cblas_sgemv(
                                    layout,
                                    trans,
                                    m as i32,
                                    n as i32,
                                    alpha as f32,
                                    a32.as_ptr(),
                                    lda as i32,
                                    xb.as_ptr(),
                                    inc,
                                    beta as f32,
                                    yb.as_mut_ptr(),
                                    inc,
                                );
                            }
                            let got: Vec<u32> =
                                (0..rows).map(|k| yb[slot(rows, k)].to_bits()).collect();
                            assert_eq!(got, want, "inc={inc} {what}");
                        }
                    }
                }
            }
        }
    }

    /// alpha == 0 on the unit-stride path never reads A or x (NaN/Inf there
    /// cannot reach y), for every layout and transpose and beta in {0, 1, 2}.
    #[test]
    fn test_gemv_unit_stride_alpha_zero_never_reads_a_or_x() {
        let (m, n) = (7usize, 5usize);
        let a = vec![f64::NAN; m * n];
        let a32 = vec![f32::INFINITY; m * n];
        for layout in LAYOUTS {
            let lda = match layout {
                CblasLayout::ColMajor => m,
                CblasLayout::RowMajor => n,
            };
            for trans in TRANSES {
                let (rows, cols) = match trans {
                    CblasTranspose::NoTrans => (m, n),
                    CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
                };
                let x = vec![f64::NAN; cols];
                let x32 = vec![f32::NAN; cols];
                for beta in [0.0f64, 1.0, 2.0] {
                    let mut y = vec![if beta == 0.0 { f64::NAN } else { 3.0 }; rows];
                    let mut y32 = vec![if beta == 0.0 { f32::NAN } else { 3.0 }; rows];
                    // SAFETY: A is m x n with leading dimension `lda`; x has
                    // `cols` and y has `rows` elements at unit stride.
                    unsafe {
                        cblas_dgemv(
                            layout,
                            trans,
                            m as i32,
                            n as i32,
                            0.0,
                            a.as_ptr(),
                            lda as i32,
                            x.as_ptr(),
                            1,
                            beta,
                            y.as_mut_ptr(),
                            1,
                        );
                        cblas_sgemv(
                            layout,
                            trans,
                            m as i32,
                            n as i32,
                            0.0,
                            a32.as_ptr(),
                            lda as i32,
                            x32.as_ptr(),
                            1,
                            beta as f32,
                            y32.as_mut_ptr(),
                            1,
                        );
                    }
                    let want = if beta == 0.0 { 0.0 } else { 3.0 * beta };
                    assert!(
                        y.iter().all(|&v| v == want),
                        "{layout:?}/{trans:?} beta={beta}: {y:?}"
                    );
                    assert!(
                        y32.iter().all(|&v| v == want as f32),
                        "{layout:?}/{trans:?} beta={beta}: {y32:?}"
                    );
                }
            }
        }
    }

    /// Random data on the unit-stride path agrees with the strided loops to
    /// within rounding (the loops evaluate `(alpha·a)·x` per term, the kernels
    /// `fma(alpha·x, a, y)` / `alpha·Σ a·x`).
    #[test]
    fn test_dgemv_unit_stride_matches_strided_loops_on_random_data() {
        let (m, n) = (37usize, 29usize);
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        let a: Vec<f64> = (0..m * n).map(|_| next()).collect();
        let xbuf: Vec<f64> = (0..2 * m.max(n)).map(|_| next()).collect();
        let y0: Vec<f64> = (0..2 * m.max(n)).map(|_| next()).collect();
        for layout in LAYOUTS {
            let lda = match layout {
                CblasLayout::ColMajor => m,
                CblasLayout::RowMajor => n,
            };
            for trans in TRANSES {
                let (rows, cols) = match trans {
                    CblasTranspose::NoTrans => (m, n),
                    CblasTranspose::Trans | CblasTranspose::ConjTrans => (n, m),
                };
                let x: Vec<f64> = (0..cols).map(|k| xbuf[2 * k]).collect();
                let mut y_fast: Vec<f64> = (0..rows).map(|k| y0[2 * k]).collect();
                let mut y_loop = y0[..2 * rows].to_vec();
                // SAFETY: A is m x n with leading dimension `lda`; the unit-stride
                // call uses `cols`/`rows` elements, the strided one spans
                // 2*cols / 2*rows slots at increment 2.
                unsafe {
                    cblas_dgemv(
                        layout,
                        trans,
                        m as i32,
                        n as i32,
                        0.75,
                        a.as_ptr(),
                        lda as i32,
                        x.as_ptr(),
                        1,
                        -0.5,
                        y_fast.as_mut_ptr(),
                        1,
                    );
                    cblas_dgemv(
                        layout,
                        trans,
                        m as i32,
                        n as i32,
                        0.75,
                        a.as_ptr(),
                        lda as i32,
                        xbuf.as_ptr(),
                        2,
                        -0.5,
                        y_loop.as_mut_ptr(),
                        2,
                    );
                }
                for (k, (&f, &l)) in y_fast.iter().zip(y_loop.iter().step_by(2)).enumerate() {
                    assert!(
                        (f - l).abs() <= 64.0 * f64::EPSILON * (cols as f64),
                        "{layout:?}/{trans:?} row {k}: fast {f} vs loop {l}"
                    );
                }
            }
        }
    }
}
