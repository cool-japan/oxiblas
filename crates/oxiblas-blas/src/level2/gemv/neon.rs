//! aarch64 NEON GEMV kernels for column-major `f64` / `f32` matrices
//! (`col_stride() == 1`; the leading dimension is `row_stride()`).
//!
//! # NoTrans: `y += alpha·A·x` (axpy form)
//!
//! Columns are processed eight at a time (then four, then one):
//! `y[i] += t0·A[i,j] + t1·A[i,j+1] + … + t7·A[i,j+7]` with `t = alpha·x[j..]`
//! formed once per column group, streaming each column contiguously while a
//! strip of `y` stays in registers for the eight FMAs, so `y` is read and
//! written once per eight columns.
//!
//! The columns are not cut into row blocks. On an Apple M3, row blocks sized to
//! keep the `y` strip in L1 (4096 or 1024 rows) made 5000² and 10000² f64
//! problems 18–100% slower, because every block restarts all `n` column streams;
//! without blocking the `y` strip (80 KB at m = 10000) stays in L1/L2 anyway.
//!
//! Every `y[i]` receives exactly the reference DGEMV sequence
//! `y[i] = fma(alpha·x[j], A[i,j], y[i])` for `j = 0, 1, …, n−1` (vector lanes,
//! the 2/4-row step and the scalar row tail of every column group perform the
//! same FMAs in the same column order), so the result does not depend on the
//! position of a row or on how rows are split between threads.
//!
//! # Trans: `y += alpha·Aᵀ·x` (dot form)
//!
//! `y[j] += alpha·(A[:,j]·x)`, four columns at a time so each load of `x`
//! feeds four columns, with four vector accumulators per column (16 chains).
//! The per-column arithmetic of the four-column step and of the single-column
//! tail is identical, so `y[j]` does not depend on how columns are grouped or
//! split between threads either.

use core::arch::aarch64::{
    float32x4_t, float64x2_t, vaddq_f32, vaddq_f64, vaddvq_f32, vaddvq_f64, vdupq_n_f32,
    vdupq_n_f64, vfmaq_f32, vfmaq_f64, vfmaq_n_f32, vfmaq_n_f64, vld1q_f32, vld1q_f64, vst1q_f32,
    vst1q_f64,
};
use oxiblas_matrix::MatRef;

/// `y += alpha·A·x` for a column-major f64 view.
///
/// # Panics
///
/// Panics unless `a.col_stride() == 1`, `x.len() == a.ncols()` and
/// `y.len() == a.nrows()` (the GEMV drivers guarantee all three).
pub(crate) fn notrans_f64(alpha: f64, a: MatRef<'_, f64>, x: &[f64], y: &mut [f64]) {
    assert!(
        a.col_stride() == 1 && x.len() == a.ncols() && y.len() == a.nrows(),
        "notrans_f64: operands do not match a column-major m x n view"
    );
    let (m, n, lda) = (a.nrows(), a.ncols(), a.row_stride());
    // SAFETY: with `col_stride() == 1` element `(i, j)` of the view is at
    // `a.as_ptr() + i + j·lda` for `i < m`, `j < n`; the kernel reads exactly
    // those elements and `x[..n]` and updates `y[..m]`, all in bounds by the
    // assertion above.
    unsafe { notrans_rows_f64(m, n, alpha, a.as_ptr(), lda, x.as_ptr(), y.as_mut_ptr()) };
}

/// `y += alpha·A·x` for a column-major f32 view.
///
/// # Panics
///
/// Panics unless `a.col_stride() == 1`, `x.len() == a.ncols()` and
/// `y.len() == a.nrows()`.
pub(crate) fn notrans_f32(alpha: f32, a: MatRef<'_, f32>, x: &[f32], y: &mut [f32]) {
    assert!(
        a.col_stride() == 1 && x.len() == a.ncols() && y.len() == a.nrows(),
        "notrans_f32: operands do not match a column-major m x n view"
    );
    let (m, n, lda) = (a.nrows(), a.ncols(), a.row_stride());
    // SAFETY: as in `notrans_f64`.
    unsafe { notrans_rows_f32(m, n, alpha, a.as_ptr(), lda, x.as_ptr(), y.as_mut_ptr()) };
}

/// `y += alpha·Aᵀ·x` for a column-major f64 view (`ConjTrans` is the same for
/// real data).
///
/// # Panics
///
/// Panics unless `a.col_stride() == 1`, `x.len() == a.nrows()` and
/// `y.len() == a.ncols()`.
pub(crate) fn trans_f64(alpha: f64, a: MatRef<'_, f64>, x: &[f64], y: &mut [f64]) {
    assert!(
        a.col_stride() == 1 && x.len() == a.nrows() && y.len() == a.ncols(),
        "trans_f64: operands do not match a column-major m x n view"
    );
    let (m, n, lda) = (a.nrows(), a.ncols(), a.row_stride());
    let (ap, xp) = (a.as_ptr(), x.as_ptr());
    let mut j = 0;
    for yq in y.chunks_exact_mut(4) {
        // SAFETY: columns `j..j + 4` are inside the view (`yq` is a full chunk
        // of `y`, whose length is `n`) and the kernel reads rows `..m` of them
        // and `x[..m]` only.
        let d = unsafe { dot4_f64(m, ap.add(j * lda), lda, xp) };
        for (yv, dv) in yq.iter_mut().zip(d) {
            *yv += alpha * dv;
        }
        j += 4;
    }
    for yv in y.chunks_exact_mut(4).into_remainder() {
        // SAFETY: column `j < n` is inside the view; rows `..m` and `x[..m]`.
        let d = unsafe { dot1_f64(m, ap.add(j * lda), xp) };
        *yv += alpha * d;
        j += 1;
    }
    debug_assert_eq!(j, n);
}

/// `y += alpha·Aᵀ·x` for a column-major f32 view.
///
/// # Panics
///
/// Panics unless `a.col_stride() == 1`, `x.len() == a.nrows()` and
/// `y.len() == a.ncols()`.
pub(crate) fn trans_f32(alpha: f32, a: MatRef<'_, f32>, x: &[f32], y: &mut [f32]) {
    assert!(
        a.col_stride() == 1 && x.len() == a.nrows() && y.len() == a.ncols(),
        "trans_f32: operands do not match a column-major m x n view"
    );
    let (m, n, lda) = (a.nrows(), a.ncols(), a.row_stride());
    let (ap, xp) = (a.as_ptr(), x.as_ptr());
    let mut j = 0;
    for yq in y.chunks_exact_mut(4) {
        // SAFETY: as in `trans_f64`.
        let d = unsafe { dot4_f32(m, ap.add(j * lda), lda, xp) };
        for (yv, dv) in yq.iter_mut().zip(d) {
            *yv += alpha * dv;
        }
        j += 4;
    }
    for yv in y.chunks_exact_mut(4).into_remainder() {
        // SAFETY: as in `trans_f64`.
        let d = unsafe { dot1_f32(m, ap.add(j * lda), xp) };
        *yv += alpha * d;
        j += 1;
    }
    debug_assert_eq!(j, n);
}

// ---------------------------------------------------------------------------
// NoTrans kernels
// ---------------------------------------------------------------------------

/// `y[..mb] += Σ_j (alpha·x[j])·A[..mb, j]` for `j < n`, `A[i, j] = a[i + j·lda]`.
///
/// # Safety
///
/// `a` must be valid for reads at `i + j·lda` for `i < mb`, `j < n`; `x` for
/// reads of `n` values; `y` for reads and writes of `mb` values.
#[inline]
unsafe fn notrans_rows_f64(
    mb: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    x: *const f64,
    y: *mut f64,
) {
    let mut j = 0;
    while j + 8 <= n {
        let c0 = a.add(j * lda);
        let cs = [
            c0,
            c0.add(lda),
            c0.add(2 * lda),
            c0.add(3 * lda),
            c0.add(4 * lda),
            c0.add(5 * lda),
            c0.add(6 * lda),
            c0.add(7 * lda),
        ];
        let mut ts = [0.0f64; 8];
        for (k, t) in ts.iter_mut().enumerate() {
            *t = alpha * *x.add(j + k);
        }
        let mut i = 0;
        while i + 8 <= mb {
            let yp = y.add(i);
            let mut y0 = vld1q_f64(yp);
            let mut y1 = vld1q_f64(yp.add(2));
            let mut y2 = vld1q_f64(yp.add(4));
            let mut y3 = vld1q_f64(yp.add(6));
            for (c, &t) in cs.iter().zip(&ts) {
                let cp = c.add(i);
                y0 = vfmaq_n_f64(y0, vld1q_f64(cp), t);
                y1 = vfmaq_n_f64(y1, vld1q_f64(cp.add(2)), t);
                y2 = vfmaq_n_f64(y2, vld1q_f64(cp.add(4)), t);
                y3 = vfmaq_n_f64(y3, vld1q_f64(cp.add(6)), t);
            }
            vst1q_f64(yp, y0);
            vst1q_f64(yp.add(2), y1);
            vst1q_f64(yp.add(4), y2);
            vst1q_f64(yp.add(6), y3);
            i += 8;
        }
        while i + 2 <= mb {
            let yp = y.add(i);
            let mut v = vld1q_f64(yp);
            for (c, &t) in cs.iter().zip(&ts) {
                v = vfmaq_n_f64(v, vld1q_f64(c.add(i)), t);
            }
            vst1q_f64(yp, v);
            i += 2;
        }
        if i < mb {
            let yp = y.add(i);
            let mut acc = *yp;
            for (c, &t) in cs.iter().zip(&ts) {
                acc = (*c.add(i)).mul_add(t, acc);
            }
            *yp = acc;
        }
        j += 8;
    }
    while j + 4 <= n {
        let c0 = a.add(j * lda);
        let c1 = c0.add(lda);
        let c2 = c1.add(lda);
        let c3 = c2.add(lda);
        let t0 = alpha * *x.add(j);
        let t1 = alpha * *x.add(j + 1);
        let t2 = alpha * *x.add(j + 2);
        let t3 = alpha * *x.add(j + 3);
        let mut i = 0;
        while i + 8 <= mb {
            let yp = y.add(i);
            let mut y0 = vld1q_f64(yp);
            let mut y1 = vld1q_f64(yp.add(2));
            let mut y2 = vld1q_f64(yp.add(4));
            let mut y3 = vld1q_f64(yp.add(6));
            for (c, t) in [(c0, t0), (c1, t1), (c2, t2), (c3, t3)] {
                let cp = c.add(i);
                y0 = vfmaq_n_f64(y0, vld1q_f64(cp), t);
                y1 = vfmaq_n_f64(y1, vld1q_f64(cp.add(2)), t);
                y2 = vfmaq_n_f64(y2, vld1q_f64(cp.add(4)), t);
                y3 = vfmaq_n_f64(y3, vld1q_f64(cp.add(6)), t);
            }
            vst1q_f64(yp, y0);
            vst1q_f64(yp.add(2), y1);
            vst1q_f64(yp.add(4), y2);
            vst1q_f64(yp.add(6), y3);
            i += 8;
        }
        while i + 2 <= mb {
            let yp = y.add(i);
            let mut v = vld1q_f64(yp);
            v = vfmaq_n_f64(v, vld1q_f64(c0.add(i)), t0);
            v = vfmaq_n_f64(v, vld1q_f64(c1.add(i)), t1);
            v = vfmaq_n_f64(v, vld1q_f64(c2.add(i)), t2);
            v = vfmaq_n_f64(v, vld1q_f64(c3.add(i)), t3);
            vst1q_f64(yp, v);
            i += 2;
        }
        if i < mb {
            let yp = y.add(i);
            let mut s = *yp;
            s = (*c0.add(i)).mul_add(t0, s);
            s = (*c1.add(i)).mul_add(t1, s);
            s = (*c2.add(i)).mul_add(t2, s);
            s = (*c3.add(i)).mul_add(t3, s);
            *yp = s;
        }
        j += 4;
    }
    while j < n {
        let c = a.add(j * lda);
        let t = alpha * *x.add(j);
        let mut i = 0;
        while i + 8 <= mb {
            let (yp, cp) = (y.add(i), c.add(i));
            vst1q_f64(yp, vfmaq_n_f64(vld1q_f64(yp), vld1q_f64(cp), t));
            vst1q_f64(
                yp.add(2),
                vfmaq_n_f64(vld1q_f64(yp.add(2)), vld1q_f64(cp.add(2)), t),
            );
            vst1q_f64(
                yp.add(4),
                vfmaq_n_f64(vld1q_f64(yp.add(4)), vld1q_f64(cp.add(4)), t),
            );
            vst1q_f64(
                yp.add(6),
                vfmaq_n_f64(vld1q_f64(yp.add(6)), vld1q_f64(cp.add(6)), t),
            );
            i += 8;
        }
        while i + 2 <= mb {
            let yp = y.add(i);
            vst1q_f64(yp, vfmaq_n_f64(vld1q_f64(yp), vld1q_f64(c.add(i)), t));
            i += 2;
        }
        if i < mb {
            let yp = y.add(i);
            *yp = (*c.add(i)).mul_add(t, *yp);
        }
        j += 1;
    }
}

/// f32 counterpart of [`notrans_rows_f64`] (16 rows per vector step).
///
/// # Safety
///
/// As for [`notrans_rows_f64`].
#[inline]
unsafe fn notrans_rows_f32(
    mb: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    y: *mut f32,
) {
    let mut j = 0;
    while j + 8 <= n {
        let c0 = a.add(j * lda);
        let cs = [
            c0,
            c0.add(lda),
            c0.add(2 * lda),
            c0.add(3 * lda),
            c0.add(4 * lda),
            c0.add(5 * lda),
            c0.add(6 * lda),
            c0.add(7 * lda),
        ];
        let mut ts = [0.0f32; 8];
        for (k, t) in ts.iter_mut().enumerate() {
            *t = alpha * *x.add(j + k);
        }
        let mut i = 0;
        while i + 16 <= mb {
            let yp = y.add(i);
            let mut y0 = vld1q_f32(yp);
            let mut y1 = vld1q_f32(yp.add(4));
            let mut y2 = vld1q_f32(yp.add(8));
            let mut y3 = vld1q_f32(yp.add(12));
            for (c, &t) in cs.iter().zip(&ts) {
                let cp = c.add(i);
                y0 = vfmaq_n_f32(y0, vld1q_f32(cp), t);
                y1 = vfmaq_n_f32(y1, vld1q_f32(cp.add(4)), t);
                y2 = vfmaq_n_f32(y2, vld1q_f32(cp.add(8)), t);
                y3 = vfmaq_n_f32(y3, vld1q_f32(cp.add(12)), t);
            }
            vst1q_f32(yp, y0);
            vst1q_f32(yp.add(4), y1);
            vst1q_f32(yp.add(8), y2);
            vst1q_f32(yp.add(12), y3);
            i += 16;
        }
        while i + 4 <= mb {
            let yp = y.add(i);
            let mut v = vld1q_f32(yp);
            for (c, &t) in cs.iter().zip(&ts) {
                v = vfmaq_n_f32(v, vld1q_f32(c.add(i)), t);
            }
            vst1q_f32(yp, v);
            i += 4;
        }
        while i < mb {
            let yp = y.add(i);
            let mut acc = *yp;
            for (c, &t) in cs.iter().zip(&ts) {
                acc = (*c.add(i)).mul_add(t, acc);
            }
            *yp = acc;
            i += 1;
        }
        j += 8;
    }
    while j + 4 <= n {
        let c0 = a.add(j * lda);
        let c1 = c0.add(lda);
        let c2 = c1.add(lda);
        let c3 = c2.add(lda);
        let t0 = alpha * *x.add(j);
        let t1 = alpha * *x.add(j + 1);
        let t2 = alpha * *x.add(j + 2);
        let t3 = alpha * *x.add(j + 3);
        let mut i = 0;
        while i + 16 <= mb {
            let yp = y.add(i);
            let mut y0 = vld1q_f32(yp);
            let mut y1 = vld1q_f32(yp.add(4));
            let mut y2 = vld1q_f32(yp.add(8));
            let mut y3 = vld1q_f32(yp.add(12));
            for (c, t) in [(c0, t0), (c1, t1), (c2, t2), (c3, t3)] {
                let cp = c.add(i);
                y0 = vfmaq_n_f32(y0, vld1q_f32(cp), t);
                y1 = vfmaq_n_f32(y1, vld1q_f32(cp.add(4)), t);
                y2 = vfmaq_n_f32(y2, vld1q_f32(cp.add(8)), t);
                y3 = vfmaq_n_f32(y3, vld1q_f32(cp.add(12)), t);
            }
            vst1q_f32(yp, y0);
            vst1q_f32(yp.add(4), y1);
            vst1q_f32(yp.add(8), y2);
            vst1q_f32(yp.add(12), y3);
            i += 16;
        }
        while i + 4 <= mb {
            let yp = y.add(i);
            let mut v = vld1q_f32(yp);
            v = vfmaq_n_f32(v, vld1q_f32(c0.add(i)), t0);
            v = vfmaq_n_f32(v, vld1q_f32(c1.add(i)), t1);
            v = vfmaq_n_f32(v, vld1q_f32(c2.add(i)), t2);
            v = vfmaq_n_f32(v, vld1q_f32(c3.add(i)), t3);
            vst1q_f32(yp, v);
            i += 4;
        }
        while i < mb {
            let yp = y.add(i);
            let mut s = *yp;
            s = (*c0.add(i)).mul_add(t0, s);
            s = (*c1.add(i)).mul_add(t1, s);
            s = (*c2.add(i)).mul_add(t2, s);
            s = (*c3.add(i)).mul_add(t3, s);
            *yp = s;
            i += 1;
        }
        j += 4;
    }
    while j < n {
        let c = a.add(j * lda);
        let t = alpha * *x.add(j);
        let mut i = 0;
        while i + 16 <= mb {
            let (yp, cp) = (y.add(i), c.add(i));
            vst1q_f32(yp, vfmaq_n_f32(vld1q_f32(yp), vld1q_f32(cp), t));
            vst1q_f32(
                yp.add(4),
                vfmaq_n_f32(vld1q_f32(yp.add(4)), vld1q_f32(cp.add(4)), t),
            );
            vst1q_f32(
                yp.add(8),
                vfmaq_n_f32(vld1q_f32(yp.add(8)), vld1q_f32(cp.add(8)), t),
            );
            vst1q_f32(
                yp.add(12),
                vfmaq_n_f32(vld1q_f32(yp.add(12)), vld1q_f32(cp.add(12)), t),
            );
            i += 16;
        }
        while i + 4 <= mb {
            let yp = y.add(i);
            vst1q_f32(yp, vfmaq_n_f32(vld1q_f32(yp), vld1q_f32(c.add(i)), t));
            i += 4;
        }
        while i < mb {
            let yp = y.add(i);
            *yp = (*c.add(i)).mul_add(t, *yp);
            i += 1;
        }
        j += 1;
    }
}

// ---------------------------------------------------------------------------
// Trans kernels
// ---------------------------------------------------------------------------

/// One column's 8-row step: `acc[k] += A[i+2k .. i+2k+2, col]·x[i+2k .. i+2k+2]`.
#[inline(always)]
unsafe fn tcol_step_f64(
    acc: &mut [float64x2_t; 4],
    c: *const f64,
    i: usize,
    xv: &[float64x2_t; 4],
) {
    let cp = c.add(i);
    acc[0] = vfmaq_f64(acc[0], vld1q_f64(cp), xv[0]);
    acc[1] = vfmaq_f64(acc[1], vld1q_f64(cp.add(2)), xv[1]);
    acc[2] = vfmaq_f64(acc[2], vld1q_f64(cp.add(4)), xv[2]);
    acc[3] = vfmaq_f64(acc[3], vld1q_f64(cp.add(6)), xv[3]);
}

/// Folds the last `m − i < 8` rows of one column into `acc` and reduces it.
#[inline(always)]
unsafe fn tcol_finish_f64(
    mut acc: [float64x2_t; 4],
    c: *const f64,
    x: *const f64,
    mut i: usize,
    m: usize,
) -> f64 {
    if m - i >= 4 {
        acc[0] = vfmaq_f64(acc[0], vld1q_f64(c.add(i)), vld1q_f64(x.add(i)));
        acc[1] = vfmaq_f64(acc[1], vld1q_f64(c.add(i + 2)), vld1q_f64(x.add(i + 2)));
        i += 4;
    }
    if m - i >= 2 {
        acc[2] = vfmaq_f64(acc[2], vld1q_f64(c.add(i)), vld1q_f64(x.add(i)));
        i += 2;
    }
    let mut s = vaddvq_f64(vaddq_f64(
        vaddq_f64(acc[0], acc[1]),
        vaddq_f64(acc[2], acc[3]),
    ));
    if i < m {
        s = (*c.add(i)).mul_add(*x.add(i), s);
    }
    s
}

/// Dot products of four adjacent columns (`c0`, `c0 + lda`, …) with `x[..m]`.
///
/// # Safety
///
/// The four columns must be valid for reads of `m` values each, and `x` for
/// reads of `m` values.
#[inline]
unsafe fn dot4_f64(m: usize, c0: *const f64, lda: usize, x: *const f64) -> [f64; 4] {
    let c1 = c0.add(lda);
    let c2 = c1.add(lda);
    let c3 = c2.add(lda);
    let zero = vdupq_n_f64(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = ([zero; 4], [zero; 4], [zero; 4], [zero; 4]);
    let body = m - m % 8;
    let mut i = 0;
    while i < body {
        let xv = [
            vld1q_f64(x.add(i)),
            vld1q_f64(x.add(i + 2)),
            vld1q_f64(x.add(i + 4)),
            vld1q_f64(x.add(i + 6)),
        ];
        tcol_step_f64(&mut a0, c0, i, &xv);
        tcol_step_f64(&mut a1, c1, i, &xv);
        tcol_step_f64(&mut a2, c2, i, &xv);
        tcol_step_f64(&mut a3, c3, i, &xv);
        i += 8;
    }
    [
        tcol_finish_f64(a0, c0, x, i, m),
        tcol_finish_f64(a1, c1, x, i, m),
        tcol_finish_f64(a2, c2, x, i, m),
        tcol_finish_f64(a3, c3, x, i, m),
    ]
}

/// Dot product of one column with `x[..m]`, with exactly the per-column
/// arithmetic of [`dot4_f64`].
///
/// # Safety
///
/// `c` and `x` must be valid for reads of `m` values each.
#[inline]
unsafe fn dot1_f64(m: usize, c: *const f64, x: *const f64) -> f64 {
    let zero = vdupq_n_f64(0.0);
    let mut acc = [zero; 4];
    let body = m - m % 8;
    let mut i = 0;
    while i < body {
        let xv = [
            vld1q_f64(x.add(i)),
            vld1q_f64(x.add(i + 2)),
            vld1q_f64(x.add(i + 4)),
            vld1q_f64(x.add(i + 6)),
        ];
        tcol_step_f64(&mut acc, c, i, &xv);
        i += 8;
    }
    tcol_finish_f64(acc, c, x, i, m)
}

/// One column's 16-row step (f32).
#[inline(always)]
unsafe fn tcol_step_f32(
    acc: &mut [float32x4_t; 4],
    c: *const f32,
    i: usize,
    xv: &[float32x4_t; 4],
) {
    let cp = c.add(i);
    acc[0] = vfmaq_f32(acc[0], vld1q_f32(cp), xv[0]);
    acc[1] = vfmaq_f32(acc[1], vld1q_f32(cp.add(4)), xv[1]);
    acc[2] = vfmaq_f32(acc[2], vld1q_f32(cp.add(8)), xv[2]);
    acc[3] = vfmaq_f32(acc[3], vld1q_f32(cp.add(12)), xv[3]);
}

/// Folds the last `m − i < 16` rows of one column into `acc` and reduces it.
#[inline(always)]
unsafe fn tcol_finish_f32(
    mut acc: [float32x4_t; 4],
    c: *const f32,
    x: *const f32,
    mut i: usize,
    m: usize,
) -> f32 {
    if m - i >= 8 {
        acc[0] = vfmaq_f32(acc[0], vld1q_f32(c.add(i)), vld1q_f32(x.add(i)));
        acc[1] = vfmaq_f32(acc[1], vld1q_f32(c.add(i + 4)), vld1q_f32(x.add(i + 4)));
        i += 8;
    }
    if m - i >= 4 {
        acc[2] = vfmaq_f32(acc[2], vld1q_f32(c.add(i)), vld1q_f32(x.add(i)));
        i += 4;
    }
    let mut s = vaddvq_f32(vaddq_f32(
        vaddq_f32(acc[0], acc[1]),
        vaddq_f32(acc[2], acc[3]),
    ));
    while i < m {
        s = (*c.add(i)).mul_add(*x.add(i), s);
        i += 1;
    }
    s
}

/// f32 counterpart of [`dot4_f64`].
///
/// # Safety
///
/// As for [`dot4_f64`].
#[inline]
unsafe fn dot4_f32(m: usize, c0: *const f32, lda: usize, x: *const f32) -> [f32; 4] {
    let c1 = c0.add(lda);
    let c2 = c1.add(lda);
    let c3 = c2.add(lda);
    let zero = vdupq_n_f32(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = ([zero; 4], [zero; 4], [zero; 4], [zero; 4]);
    let body = m - m % 16;
    let mut i = 0;
    while i < body {
        let xv = [
            vld1q_f32(x.add(i)),
            vld1q_f32(x.add(i + 4)),
            vld1q_f32(x.add(i + 8)),
            vld1q_f32(x.add(i + 12)),
        ];
        tcol_step_f32(&mut a0, c0, i, &xv);
        tcol_step_f32(&mut a1, c1, i, &xv);
        tcol_step_f32(&mut a2, c2, i, &xv);
        tcol_step_f32(&mut a3, c3, i, &xv);
        i += 16;
    }
    [
        tcol_finish_f32(a0, c0, x, i, m),
        tcol_finish_f32(a1, c1, x, i, m),
        tcol_finish_f32(a2, c2, x, i, m),
        tcol_finish_f32(a3, c3, x, i, m),
    ]
}

/// f32 counterpart of [`dot1_f64`].
///
/// # Safety
///
/// As for [`dot1_f64`].
#[inline]
unsafe fn dot1_f32(m: usize, c: *const f32, x: *const f32) -> f32 {
    let zero = vdupq_n_f32(0.0);
    let mut acc = [zero; 4];
    let body = m - m % 16;
    let mut i = 0;
    while i < body {
        let xv = [
            vld1q_f32(x.add(i)),
            vld1q_f32(x.add(i + 4)),
            vld1q_f32(x.add(i + 8)),
            vld1q_f32(x.add(i + 12)),
        ];
        tcol_step_f32(&mut acc, c, i, &xv);
        i += 16;
    }
    tcol_finish_f32(acc, c, x, i, m)
}
