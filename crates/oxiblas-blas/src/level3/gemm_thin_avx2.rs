//! x86-64 AVX2 + FMA kernels of the thin GEMM path (`m == 1` or `n == 1`).
//!
//! They compute the contract of the parent module element by element, with
//! every multiply-add the contract fuses done by one `vfmadd` (the product
//! `t_j = alpha·x[j]` and the final dot-form update stay unfused, as listed
//! below), so their results are bit for bit the aarch64 NEON GEMV kernels'
//! on the same operands, except the sign and payload of NaN results: each
//! instruction set has its own default NaN (an invalid operation such as
//! `0·∞` gives a negative quiet NaN here and a positive one on aarch64),
//! while NaN results sit at the same positions. Like those kernels, they
//! are position-independent: an element's result depends only on its own
//! operands, never on where a part of `y` starts, on alignment, or on the row
//! or column count of the view. Every load is unaligned and inside the
//! operands: `S` is read through its column pointers for the view's rows
//! only, `x` and `y` within their slices; row remainders shorter than a
//! vector are handled one element at a time, never by a wide or masked load.
//!
//! # Axpy form: `y += alpha·S·x`
//!
//! Columns go in groups of eight, then four, then one; `t_j = alpha·x[j]` is
//! formed once per column (a plain product). Each group passes over `y` once:
//! 16 rows (`f64`) or 32 rows (`f32`) are held in four ymm registers, then
//! single ymm steps of 4 or 8 rows, then single rows. In every one of these
//! steps a row receives `y[i] = fma(S[i, j], t_j, y[i])` for the group's
//! columns in ascending order, so over all groups every `y[i]` sees exactly
//! one FMA chain over `j = 0, 1, …` in order.
//!
//! # Dot form: `y[j] += alpha·(S[:, j]·x)`
//!
//! Four columns at a time (each load of `x` feeds four columns), then one;
//! both use the same per-column routine. The contract's `4·L` partial sums
//! (`L` = 2 lanes of `f64`, 4 of `f32` per 128-bit vector) live in two ymm
//! registers per column: `A = [acc0 | acc1]` and `B = [acc2 | acc3]`.
//!
//! - Steps of `4·L` rows: rows `i .. i + 2L` go into `A` and rows
//!   `i + 2L .. i + 4L` into `B`, one 256-bit FMA each (`acc = fma(S, x, acc)`
//!   per lane).
//! - At least `2·L` rows left: one more 256-bit FMA into `A`.
//! - At least `L` rows left: one 128-bit FMA into `acc2` (the low half of `B`).
//! - Reduction: `v = (acc0 + acc1) + (acc2 + acc3)` per lane, then
//!   `v0 + v1` (`f64`) or `(v0 + v1) + (v2 + v3)` (`f32`).
//! - The remaining rows (at most 1 for `f64`, 3 for `f32`) are fused into the
//!   sum one at a time in order.
//! - Finally `y[j] = y[j] + alpha·d_j`: a product and a sum, not fused, as in
//!   the NEON kernels.
//!
//! # Safety model
//!
//! The parent module selects these kernels only for
//! `SimdTier::Avx2Fma`, which `GemmContext` reports only when the CPU has
//! AVX2 and FMA. The safe entry points still check the shapes of their
//! operands and the CPU features themselves, so calling one can never run an
//! unsupported instruction or touch memory outside the operands.

use core::arch::x86_64::{
    __m256, __m256d, _mm_add_pd, _mm_add_ps, _mm_cvtsd_f64, _mm_fmadd_pd, _mm_fmadd_ps,
    _mm_loadu_pd, _mm_loadu_ps, _mm_storeu_ps, _mm_unpackhi_pd, _mm256_castpd256_pd128,
    _mm256_castps256_ps128, _mm256_extractf128_pd, _mm256_extractf128_ps, _mm256_fmadd_pd,
    _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_set1_pd, _mm256_set1_ps,
    _mm256_setzero_pd, _mm256_setzero_ps, _mm256_storeu_pd, _mm256_storeu_ps,
};
use oxiblas_core::simd::dispatch::has_avx2_fma;
use oxiblas_matrix::MatRef;

/// Panics unless the CPU reports AVX2 and FMA (the precondition of every
/// `unsafe` kernel below). `name` identifies the entry point.
#[inline]
fn require_avx2_fma(name: &str) {
    assert!(
        has_avx2_fma(),
        "{name}: the CPU does not report AVX2 and FMA"
    );
}

/// `y += alpha·S·x` (axpy form) for a column-major f64 view.
///
/// # Panics
///
/// Panics unless `s.col_stride() == 1`, `x.len() == s.ncols()` and
/// `y.len() == s.nrows()`, or when the CPU lacks AVX2 or FMA.
pub(super) fn axpy_f64(alpha: f64, s: MatRef<'_, f64>, x: &[f64], y: &mut [f64]) {
    assert!(
        s.col_stride() == 1 && x.len() == s.ncols() && y.len() == s.nrows(),
        "axpy_f64_avx2: operands do not match a column-major view"
    );
    require_avx2_fma("axpy_f64_avx2");
    let (m, n, lda) = (s.nrows(), s.ncols(), s.row_stride());
    // SAFETY: the CPU has AVX2 and FMA (checked above). With
    // `col_stride() == 1`, element `(i, j)` of the view is at
    // `s.as_ptr() + i + j·lda` for `i < m`, `j < n`; the kernel reads exactly
    // those elements, `x[..n]`, and reads and writes `y[..m]`, all in bounds
    // by the assertion above.
    unsafe { axpy_rows_f64(m, n, alpha, s.as_ptr(), lda, x.as_ptr(), y.as_mut_ptr()) };
}

/// `y += alpha·S·x` (axpy form) for a column-major f32 view.
///
/// # Panics
///
/// As for [`axpy_f64`].
pub(super) fn axpy_f32(alpha: f32, s: MatRef<'_, f32>, x: &[f32], y: &mut [f32]) {
    assert!(
        s.col_stride() == 1 && x.len() == s.ncols() && y.len() == s.nrows(),
        "axpy_f32_avx2: operands do not match a column-major view"
    );
    require_avx2_fma("axpy_f32_avx2");
    let (m, n, lda) = (s.nrows(), s.ncols(), s.row_stride());
    // SAFETY: as in `axpy_f64`.
    unsafe { axpy_rows_f32(m, n, alpha, s.as_ptr(), lda, x.as_ptr(), y.as_mut_ptr()) };
}

/// `y += alpha·Sᵀ·x` (dot form) for a column-major f64 view.
///
/// # Panics
///
/// Panics unless `s.col_stride() == 1`, `x.len() == s.nrows()` and
/// `y.len() == s.ncols()`, or when the CPU lacks AVX2 or FMA.
pub(super) fn dot_f64(alpha: f64, s: MatRef<'_, f64>, x: &[f64], y: &mut [f64]) {
    assert!(
        s.col_stride() == 1 && x.len() == s.nrows() && y.len() == s.ncols(),
        "dot_f64_avx2: operands do not match a column-major view"
    );
    require_avx2_fma("dot_f64_avx2");
    let (m, lda) = (s.nrows(), s.row_stride());
    // SAFETY: the CPU has AVX2 and FMA (checked above). Column `j < n` of the
    // view starts at `s.as_ptr() + j·lda` and its rows `..m` are contiguous;
    // the kernel reads those rows of the `y.len() == n` columns and `x[..m]`
    // only (assertion above).
    unsafe { dot_rows_f64(m, alpha, s.as_ptr(), lda, x.as_ptr(), y) };
}

/// `y += alpha·Sᵀ·x` (dot form) for a column-major f32 view.
///
/// # Panics
///
/// As for [`dot_f64`].
pub(super) fn dot_f32(alpha: f32, s: MatRef<'_, f32>, x: &[f32], y: &mut [f32]) {
    assert!(
        s.col_stride() == 1 && x.len() == s.nrows() && y.len() == s.ncols(),
        "dot_f32_avx2: operands do not match a column-major view"
    );
    require_avx2_fma("dot_f32_avx2");
    let (m, lda) = (s.nrows(), s.row_stride());
    // SAFETY: as in `dot_f64`.
    unsafe { dot_rows_f32(m, alpha, s.as_ptr(), lda, x.as_ptr(), y) };
}

// ---------------------------------------------------------------------------
// Axpy form
// ---------------------------------------------------------------------------

/// `y[..m] += Σ_j (alpha·x[j])·S[..m, j]` for `j < n`, `S[i, j] = a[i + j·lda]`.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA; `a` must be valid for reads at
/// `i + j·lda` for `i < m`, `j < n`; `x` for reads of `n` values; `y` for
/// reads and writes of `m` values.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_rows_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    x: *const f64,
    y: *mut f64,
) {
    let mut j = 0;
    while j + 8 <= n {
        axpy_group_f64::<8>(m, alpha, a.wrapping_add(j * lda), lda, x.add(j), y);
        j += 8;
    }
    while j + 4 <= n {
        axpy_group_f64::<4>(m, alpha, a.wrapping_add(j * lda), lda, x.add(j), y);
        j += 4;
    }
    while j < n {
        axpy_group_f64::<1>(m, alpha, a.wrapping_add(j * lda), lda, x.add(j), y);
        j += 1;
    }
}

/// One group of `G` adjacent columns (`c0`, `c0 + lda`, …) applied to
/// `y[..m]`: every row gets the FMAs of the group's columns in order.
///
/// # Safety
///
/// As for [`axpy_rows_f64`] with `n = G`.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn axpy_group_f64<const G: usize>(
    m: usize,
    alpha: f64,
    c0: *const f64,
    lda: usize,
    x: *const f64,
    y: *mut f64,
) {
    let mut cs = [c0; G];
    let mut ts = [0.0f64; G];
    let mut tv = [_mm256_setzero_pd(); G];
    for g in 0..G {
        cs[g] = c0.wrapping_add(g * lda);
        ts[g] = alpha * *x.add(g);
        tv[g] = _mm256_set1_pd(ts[g]);
    }
    let mut i = 0;
    while i + 16 <= m {
        let yp = y.add(i);
        let mut y0 = _mm256_loadu_pd(yp);
        let mut y1 = _mm256_loadu_pd(yp.add(4));
        let mut y2 = _mm256_loadu_pd(yp.add(8));
        let mut y3 = _mm256_loadu_pd(yp.add(12));
        for g in 0..G {
            let cp = cs[g].add(i);
            y0 = _mm256_fmadd_pd(_mm256_loadu_pd(cp), tv[g], y0);
            y1 = _mm256_fmadd_pd(_mm256_loadu_pd(cp.add(4)), tv[g], y1);
            y2 = _mm256_fmadd_pd(_mm256_loadu_pd(cp.add(8)), tv[g], y2);
            y3 = _mm256_fmadd_pd(_mm256_loadu_pd(cp.add(12)), tv[g], y3);
        }
        _mm256_storeu_pd(yp, y0);
        _mm256_storeu_pd(yp.add(4), y1);
        _mm256_storeu_pd(yp.add(8), y2);
        _mm256_storeu_pd(yp.add(12), y3);
        i += 16;
    }
    while i + 4 <= m {
        let yp = y.add(i);
        let mut v = _mm256_loadu_pd(yp);
        for g in 0..G {
            v = _mm256_fmadd_pd(_mm256_loadu_pd(cs[g].add(i)), tv[g], v);
        }
        _mm256_storeu_pd(yp, v);
        i += 4;
    }
    while i < m {
        let yp = y.add(i);
        let mut v = *yp;
        for g in 0..G {
            v = (*cs[g].add(i)).mul_add(ts[g], v);
        }
        *yp = v;
        i += 1;
    }
}

/// f32 counterpart of [`axpy_rows_f64`].
///
/// # Safety
///
/// As for [`axpy_rows_f64`].
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_rows_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    y: *mut f32,
) {
    let mut j = 0;
    while j + 8 <= n {
        axpy_group_f32::<8>(m, alpha, a.wrapping_add(j * lda), lda, x.add(j), y);
        j += 8;
    }
    while j + 4 <= n {
        axpy_group_f32::<4>(m, alpha, a.wrapping_add(j * lda), lda, x.add(j), y);
        j += 4;
    }
    while j < n {
        axpy_group_f32::<1>(m, alpha, a.wrapping_add(j * lda), lda, x.add(j), y);
        j += 1;
    }
}

/// f32 counterpart of [`axpy_group_f64`] (32-row strips, 8-row steps).
///
/// # Safety
///
/// As for [`axpy_group_f64`].
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn axpy_group_f32<const G: usize>(
    m: usize,
    alpha: f32,
    c0: *const f32,
    lda: usize,
    x: *const f32,
    y: *mut f32,
) {
    let mut cs = [c0; G];
    let mut ts = [0.0f32; G];
    let mut tv = [_mm256_setzero_ps(); G];
    for g in 0..G {
        cs[g] = c0.wrapping_add(g * lda);
        ts[g] = alpha * *x.add(g);
        tv[g] = _mm256_set1_ps(ts[g]);
    }
    let mut i = 0;
    while i + 32 <= m {
        let yp = y.add(i);
        let mut y0 = _mm256_loadu_ps(yp);
        let mut y1 = _mm256_loadu_ps(yp.add(8));
        let mut y2 = _mm256_loadu_ps(yp.add(16));
        let mut y3 = _mm256_loadu_ps(yp.add(24));
        for g in 0..G {
            let cp = cs[g].add(i);
            y0 = _mm256_fmadd_ps(_mm256_loadu_ps(cp), tv[g], y0);
            y1 = _mm256_fmadd_ps(_mm256_loadu_ps(cp.add(8)), tv[g], y1);
            y2 = _mm256_fmadd_ps(_mm256_loadu_ps(cp.add(16)), tv[g], y2);
            y3 = _mm256_fmadd_ps(_mm256_loadu_ps(cp.add(24)), tv[g], y3);
        }
        _mm256_storeu_ps(yp, y0);
        _mm256_storeu_ps(yp.add(8), y1);
        _mm256_storeu_ps(yp.add(16), y2);
        _mm256_storeu_ps(yp.add(24), y3);
        i += 32;
    }
    while i + 8 <= m {
        let yp = y.add(i);
        let mut v = _mm256_loadu_ps(yp);
        for g in 0..G {
            v = _mm256_fmadd_ps(_mm256_loadu_ps(cs[g].add(i)), tv[g], v);
        }
        _mm256_storeu_ps(yp, v);
        i += 8;
    }
    while i < m {
        let yp = y.add(i);
        let mut v = *yp;
        for g in 0..G {
            v = (*cs[g].add(i)).mul_add(ts[g], v);
        }
        *yp = v;
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Dot form
// ---------------------------------------------------------------------------

/// `y[j] += alpha·(S[..m, j]·x[..m])` for every `j < y.len()`,
/// `S[i, j] = a[i + j·lda]`: four columns at a time, then one.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA; `a` must be valid for reads at
/// `i + j·lda` for `i < m`, `j < y.len()`; `x` for reads of `m` values.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_rows_f64(
    m: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    x: *const f64,
    y: &mut [f64],
) {
    let mut j = 0;
    for yq in y.chunks_exact_mut(4) {
        let d = dot_cols_f64::<4>(m, a.wrapping_add(j * lda), lda, x);
        for (yv, dv) in yq.iter_mut().zip(d) {
            *yv += alpha * dv;
        }
        j += 4;
    }
    for yv in y.chunks_exact_mut(4).into_remainder() {
        let [d] = dot_cols_f64::<1>(m, a.wrapping_add(j * lda), lda, x);
        *yv += alpha * d;
        j += 1;
    }
}

/// Dot products of `G` adjacent columns (`c0`, `c0 + lda`, …) with `x[..m]`.
/// Every column runs the same sequence, whatever `G` is.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA; the `G` columns must be valid for
/// reads of `m` values each, and `x` for reads of `m` values.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_cols_f64<const G: usize>(
    m: usize,
    c0: *const f64,
    lda: usize,
    x: *const f64,
) -> [f64; G] {
    let mut cs = [c0; G];
    for (g, c) in cs.iter_mut().enumerate() {
        *c = c0.wrapping_add(g * lda);
    }
    // acc01[g] = [acc0 | acc1], acc23[g] = [acc2 | acc3] of column g.
    let mut acc01 = [_mm256_setzero_pd(); G];
    let mut acc23 = [_mm256_setzero_pd(); G];
    let body = m - m % 8;
    let mut i = 0;
    while i < body {
        let xa = _mm256_loadu_pd(x.add(i));
        let xb = _mm256_loadu_pd(x.add(i + 4));
        for g in 0..G {
            let cp = cs[g].add(i);
            acc01[g] = _mm256_fmadd_pd(_mm256_loadu_pd(cp), xa, acc01[g]);
            acc23[g] = _mm256_fmadd_pd(_mm256_loadu_pd(cp.add(4)), xb, acc23[g]);
        }
        i += 8;
    }
    let mut out = [0.0f64; G];
    for g in 0..G {
        out[g] = dot_finish_f64(acc01[g], acc23[g], cs[g], x, i, m);
    }
    out
}

/// Folds the last `m − i < 8` rows of one column into its accumulators and
/// reduces them (module docs).
///
/// # Safety
///
/// The CPU must support AVX2 and FMA; `c` and `x` must be valid for reads of
/// `m` values each.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_finish_f64(
    mut acc01: __m256d,
    acc23: __m256d,
    c: *const f64,
    x: *const f64,
    mut i: usize,
    m: usize,
) -> f64 {
    if m - i >= 4 {
        acc01 = _mm256_fmadd_pd(_mm256_loadu_pd(c.add(i)), _mm256_loadu_pd(x.add(i)), acc01);
        i += 4;
    }
    let mut acc2 = _mm256_castpd256_pd128(acc23);
    let acc3 = _mm256_extractf128_pd(acc23, 1);
    if m - i >= 2 {
        acc2 = _mm_fmadd_pd(_mm_loadu_pd(c.add(i)), _mm_loadu_pd(x.add(i)), acc2);
        i += 2;
    }
    let acc0 = _mm256_castpd256_pd128(acc01);
    let acc1 = _mm256_extractf128_pd(acc01, 1);
    let v = _mm_add_pd(_mm_add_pd(acc0, acc1), _mm_add_pd(acc2, acc3));
    let mut s = _mm_cvtsd_f64(v) + _mm_cvtsd_f64(_mm_unpackhi_pd(v, v));
    if i < m {
        s = (*c.add(i)).mul_add(*x.add(i), s);
    }
    s
}

/// f32 counterpart of [`dot_rows_f64`].
///
/// # Safety
///
/// As for [`dot_rows_f64`].
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_rows_f32(
    m: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    y: &mut [f32],
) {
    let mut j = 0;
    for yq in y.chunks_exact_mut(4) {
        let d = dot_cols_f32::<4>(m, a.wrapping_add(j * lda), lda, x);
        for (yv, dv) in yq.iter_mut().zip(d) {
            *yv += alpha * dv;
        }
        j += 4;
    }
    for yv in y.chunks_exact_mut(4).into_remainder() {
        let [d] = dot_cols_f32::<1>(m, a.wrapping_add(j * lda), lda, x);
        *yv += alpha * d;
        j += 1;
    }
}

/// f32 counterpart of [`dot_cols_f64`] (16-row steps).
///
/// # Safety
///
/// As for [`dot_cols_f64`].
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_cols_f32<const G: usize>(
    m: usize,
    c0: *const f32,
    lda: usize,
    x: *const f32,
) -> [f32; G] {
    let mut cs = [c0; G];
    for (g, c) in cs.iter_mut().enumerate() {
        *c = c0.wrapping_add(g * lda);
    }
    // acc01[g] = [acc0 | acc1], acc23[g] = [acc2 | acc3] of column g.
    let mut acc01 = [_mm256_setzero_ps(); G];
    let mut acc23 = [_mm256_setzero_ps(); G];
    let body = m - m % 16;
    let mut i = 0;
    while i < body {
        let xa = _mm256_loadu_ps(x.add(i));
        let xb = _mm256_loadu_ps(x.add(i + 8));
        for g in 0..G {
            let cp = cs[g].add(i);
            acc01[g] = _mm256_fmadd_ps(_mm256_loadu_ps(cp), xa, acc01[g]);
            acc23[g] = _mm256_fmadd_ps(_mm256_loadu_ps(cp.add(8)), xb, acc23[g]);
        }
        i += 16;
    }
    let mut out = [0.0f32; G];
    for g in 0..G {
        out[g] = dot_finish_f32(acc01[g], acc23[g], cs[g], x, i, m);
    }
    out
}

/// Folds the last `m − i < 16` rows of one column into its accumulators and
/// reduces them (module docs).
///
/// # Safety
///
/// As for [`dot_finish_f64`].
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_finish_f32(
    mut acc01: __m256,
    acc23: __m256,
    c: *const f32,
    x: *const f32,
    mut i: usize,
    m: usize,
) -> f32 {
    if m - i >= 8 {
        acc01 = _mm256_fmadd_ps(_mm256_loadu_ps(c.add(i)), _mm256_loadu_ps(x.add(i)), acc01);
        i += 8;
    }
    let mut acc2 = _mm256_castps256_ps128(acc23);
    let acc3 = _mm256_extractf128_ps(acc23, 1);
    if m - i >= 4 {
        acc2 = _mm_fmadd_ps(_mm_loadu_ps(c.add(i)), _mm_loadu_ps(x.add(i)), acc2);
        i += 4;
    }
    let acc0 = _mm256_castps256_ps128(acc01);
    let acc1 = _mm256_extractf128_ps(acc01, 1);
    let v = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    let mut lanes = [0.0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), v);
    let mut s = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < m {
        s = (*c.add(i)).mul_add(*x.add(i), s);
        i += 1;
    }
    s
}
