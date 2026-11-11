//! Unit tests for the CBLAS complex dense Level-2 wrappers (issue #14):
//! `cblas_{c,z}geru`, `cblas_{c,z}gerc`, `cblas_{c,z}her`, `cblas_{c,z}her2`,
//! `cblas_{c,z}trmv`, `cblas_{c,z}trsv`.
//!
//! Every routine is compared, in both layouts and for every
//! `uplo` / `trans` / `diag` and increment in `{1, 2, -1, -3}`, against a
//! naive dense reference. The unreferenced triangle, the `lda` padding,
//! vector gaps and unit diagonals hold NaN: a read poisons the result and a
//! write is caught by the bitwise checks.

use super::*;
use crate::cblas::strided::tests::{
    DIAGS, Dense, INCS, LAYOUTS, Num, Rng, TRANSES, UPLOS, assert_bits_eq, assert_close,
    assert_gaps_unchanged, assert_solves, check_triangle_update, dense_slot, her_entry, her2_entry,
    hermitian_band, nan, op_mul, poison, store_dense_triangle, strided, tri_band, unstrided,
    widen_all,
};
use core::ptr;

/// Uniform adapter over the two complex precisions. `her` takes `alpha` as
/// `f64`; the test values are exactly representable in `f32`.
trait ComplexApi: Num {
    #[allow(clippy::too_many_arguments)]
    unsafe fn geru(
        layout: CblasLayout,
        m: i32,
        n: i32,
        alpha: Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        a: *mut Self,
        lda: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn gerc(
        layout: CblasLayout,
        m: i32,
        n: i32,
        alpha: Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        a: *mut Self,
        lda: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn her(
        layout: CblasLayout,
        uplo: CblasUplo,
        n: i32,
        alpha: f64,
        x: *const Self,
        incx: i32,
        a: *mut Self,
        lda: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn her2(
        layout: CblasLayout,
        uplo: CblasUplo,
        n: i32,
        alpha: Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        a: *mut Self,
        lda: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn trmv(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        incx: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn trsv(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        incx: i32,
    );
}

impl ComplexApi for Complex32 {
    unsafe fn geru(
        l: CblasLayout,
        m: i32,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_cgeru(l, m, n, &al, x, ix, y, iy, a, lda) }
    }
    unsafe fn gerc(
        l: CblasLayout,
        m: i32,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_cgerc(l, m, n, &al, x, ix, y, iy, a, lda) }
    }
    unsafe fn her(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: f64,
        x: *const Self,
        ix: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_cher(l, u, n, al as f32, x, ix, a, lda) }
    }
    unsafe fn her2(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_cher2(l, u, n, &al, x, ix, y, iy, a, lda) }
    }
    unsafe fn trmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ctrmv(l, u, t, d, n, a, lda, x, ix) }
    }
    unsafe fn trsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ctrsv(l, u, t, d, n, a, lda, x, ix) }
    }
}

impl ComplexApi for Complex64 {
    unsafe fn geru(
        l: CblasLayout,
        m: i32,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_zgeru(l, m, n, &al, x, ix, y, iy, a, lda) }
    }
    unsafe fn gerc(
        l: CblasLayout,
        m: i32,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_zgerc(l, m, n, &al, x, ix, y, iy, a, lda) }
    }
    unsafe fn her(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: f64,
        x: *const Self,
        ix: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_zher(l, u, n, al, x, ix, a, lda) }
    }
    unsafe fn her2(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        a: *mut Self,
        lda: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_zher2(l, u, n, &al, x, ix, y, iy, a, lda) }
    }
    unsafe fn trmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ztrmv(l, u, t, d, n, a, lda, x, ix) }
    }
    unsafe fn trsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ztrsv(l, u, t, d, n, a, lda, x, ix) }
    }
}

fn name<T>() -> &'static str {
    core::any::type_name::<T>()
}

type GerFn<T> = unsafe fn(CblasLayout, i32, i32, T, *const T, i32, *const T, i32, *mut T, i32);

// =============================================================================
// GERU / GERC
// =============================================================================

/// `A + alpha*x*op(y)^T` with `op = conj` for GERC; the `lda` padding (NaN)
/// must stay bitwise untouched.
fn check_ger_reference<T: ComplexApi>(conj_y: bool) {
    let mut rng = Rng::new(if conj_y { 0xc532 } else { 0xc531 });
    let f: GerFn<T> = if conj_y { T::gerc } else { T::geru };
    for &(m, n) in &[(1usize, 1usize), (3, 5), (6, 2)] {
        let a: Dense<T> = (0..m).map(|_| rng.vec(n)).collect();
        for layout in LAYOUTS {
            let (lda, lines) = match layout {
                CblasLayout::ColMajor => (m + 2, n),
                CblasLayout::RowMajor => (n + 2, m),
            };
            let mut before = vec![nan::<T>(); lda * lines];
            for i in 0..m {
                for j in 0..n {
                    before[dense_slot(layout, i, j, lda)] = a[i][j];
                }
            }
            for incx in INCS {
                for incy in INCS {
                    let x = rng.vec::<T>(m);
                    let y = rng.vec::<T>(n);
                    let alpha = rng.num::<T>();
                    let xb = strided(&x, incx, nan());
                    let yb = strided(&y, incy, nan());
                    let mut buf = before.clone();
                    // SAFETY: every buffer is sized for these exact arguments.
                    unsafe {
                        f(
                            layout,
                            m as i32,
                            n as i32,
                            alpha,
                            xb.as_ptr(),
                            incx,
                            yb.as_ptr(),
                            incy,
                            buf.as_mut_ptr(),
                            lda as i32,
                        );
                    }
                    let what = format!(
                        "{} {} {layout:?} m={m} n={n} incx={incx} incy={incy}",
                        name::<T>(),
                        if conj_y { "gerc" } else { "geru" }
                    );
                    let mut touched = vec![false; buf.len()];
                    for i in 0..m {
                        for j in 0..n {
                            let s = dense_slot(layout, i, j, lda);
                            touched[s] = true;
                            let (al, xi) = (alpha.widen(), x[i].widen());
                            let yj = if conj_y {
                                y[j].widen().conj()
                            } else {
                                y[j].widen()
                            };
                            let aij = a[i][j].widen();
                            let want = aij + al * xi * yj;
                            let bound = aij.norm() + al.norm() * xi.norm() * yj.norm();
                            let got = buf[s].widen();
                            assert_close(
                                &[got],
                                &[want],
                                &[bound],
                                T::EPS,
                                &format!("{what} ({i},{j})"),
                            );
                        }
                    }
                    for (s, t) in touched.iter().enumerate() {
                        if !t {
                            assert_eq!(before[s].bits(), buf[s].bits(), "{what}: padding {s}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_cgeru_matches_reference() {
    check_ger_reference::<Complex32>(false);
}

#[test]
fn test_issue_14_zgeru_matches_reference() {
    check_ger_reference::<Complex64>(false);
}

#[test]
fn test_issue_14_cgerc_matches_reference() {
    check_ger_reference::<Complex32>(true);
}

#[test]
fn test_issue_14_zgerc_matches_reference() {
    check_ger_reference::<Complex64>(true);
}

/// `alpha == 0` leaves `A` untouched without reading NaN `x` / `y`; empty
/// shapes, bad `lda` (per layout), zero increments and null pointers are
/// no-ops; `RowMajor` accepts `max(1, n) <= lda < m`.
fn check_ger_quick_returns_and_rejects<T: ComplexApi>() {
    let mut rng = Rng::new(0xc533);
    let (m, n) = (3i32, 5i32);
    let x_nan = [nan::<T>(); 16];
    let ones = [T::one(); 16];
    for f in [T::geru as GerFn<T>, T::gerc] {
        for layout in LAYOUTS {
            let what = format!("{} ger {layout:?}", name::<T>());
            let a0 = poison::<T>(40);
            let mut a = a0.clone();
            // SAFETY: A holds a 3x5 matrix in either layout with lda = 8; x / y
            // are never read for alpha == 0.
            unsafe {
                f(
                    layout,
                    m,
                    n,
                    T::zero(),
                    x_nan.as_ptr(),
                    2,
                    x_nan.as_ptr(),
                    -3,
                    a.as_mut_ptr(),
                    8,
                );
            }
            assert_bits_eq(&a0, &a, &format!("{what} alpha=0"));
            let lda_min = if layout == CblasLayout::ColMajor {
                m
            } else {
                n
            };
            // (m, n, lda, incx, incy)
            let cases = [
                (0, n, 8, 1, 1),
                (m, 0, 8, 1, 1),
                (-1, n, 8, 1, 1),
                (m, -1, 8, 1, 1),
                (m, n, lda_min - 1, 1, 1),
                (m, n, -8, 1, 1),
                (m, n, 8, 0, 1),
                (m, n, 8, 1, 0),
            ];
            for &(mm, nn, lda, incx, incy) in &cases {
                // SAFETY: every case is rejected before any pointer is used.
                unsafe {
                    f(
                        layout,
                        mm,
                        nn,
                        T::one(),
                        ones.as_ptr(),
                        incx,
                        ones.as_ptr(),
                        incy,
                        a.as_mut_ptr(),
                        lda,
                    );
                }
                let case = format!("{what} m={mm} n={nn} lda={lda} incx={incx} incy={incy}");
                assert_bits_eq(&a0, &a, &case);
            }
            // SAFETY: null pointers are rejected before any dereference.
            unsafe {
                f(
                    layout,
                    m,
                    n,
                    T::one(),
                    ptr::null(),
                    1,
                    ones.as_ptr(),
                    1,
                    a.as_mut_ptr(),
                    8,
                );
                f(
                    layout,
                    m,
                    n,
                    T::one(),
                    ones.as_ptr(),
                    1,
                    ptr::null(),
                    1,
                    a.as_mut_ptr(),
                    8,
                );
                f(
                    layout,
                    m,
                    n,
                    T::one(),
                    ones.as_ptr(),
                    1,
                    ones.as_ptr(),
                    1,
                    ptr::null_mut(),
                    8,
                );
            }
            assert_bits_eq(&a0, &a, &format!("{what} null pointers"));
        }
        // RowMajor needs lda >= max(1, n) only: lda = n = 5 >= m is valid, and
        // ColMajor needs lda >= max(1, m) only: lda = m = 3 < n is valid.
        for (layout, lda) in [(CblasLayout::RowMajor, n), (CblasLayout::ColMajor, m)] {
            let x = rng.vec::<T>(m as usize);
            let y = rng.vec::<T>(n as usize);
            let mut a = vec![
                T::zero();
                (lda * if layout == CblasLayout::ColMajor {
                    n
                } else {
                    m
                }) as usize
            ];
            // SAFETY: A holds a 3x5 matrix with the minimal lda of the layout.
            unsafe {
                f(
                    layout,
                    m,
                    n,
                    T::one(),
                    x.as_ptr(),
                    1,
                    y.as_ptr(),
                    1,
                    a.as_mut_ptr(),
                    lda,
                )
            };
            assert!(
                a.iter().any(|v| v.widen().norm() > 0.0),
                "{layout:?} minimal lda rejected"
            );
        }
    }
}

#[test]
fn test_issue_14_cgeru_cgerc_quick_returns_and_rejects() {
    check_ger_quick_returns_and_rejects::<Complex32>();
}

#[test]
fn test_issue_14_zgeru_zgerc_quick_returns_and_rejects() {
    check_ger_quick_returns_and_rejects::<Complex64>();
}

// =============================================================================
// HER / HER2
// =============================================================================

/// Rank updates of the `uplo` triangle; the stored diagonal carries an
/// imaginary part that must be zeroed, and the other triangle and the
/// padding (NaN) must stay bitwise untouched.
fn check_her_her2_reference<T: ComplexApi>() {
    let mut rng = Rng::new(0xc534);
    for n in [1usize, 3, 5] {
        let a = hermitian_band::<T>(&mut rng, n, n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                let lda = n + 1;
                let before = store_dense_triangle(layout, uplo, false, &a, lda, Some(0.625));
                let slot_of = |i: usize, j: usize| dense_slot(layout, i, j, lda);
                for incx in INCS {
                    let alpha = if incx > 0 { -1.25 } else { 0.5 };
                    let x = rng.vec::<T>(n);
                    let xb = strided(&x, incx, nan());
                    let mut buf = before.clone();
                    // SAFETY: every buffer is sized for these exact arguments.
                    unsafe {
                        T::her(
                            layout,
                            uplo,
                            n as i32,
                            alpha,
                            xb.as_ptr(),
                            incx,
                            buf.as_mut_ptr(),
                            lda as i32,
                        );
                    }
                    let what = format!("{} her {layout:?} {uplo:?} n={n} incx={incx}", name::<T>());
                    check_triangle_update(
                        &before,
                        &buf,
                        n,
                        uplo,
                        &slot_of,
                        &|i, j| her_entry(&a, alpha, &x, i, j),
                        &what,
                    );
                    for incy in INCS {
                        let y = rng.vec::<T>(n);
                        let yb = strided(&y, incy, nan());
                        let alpha2 = rng.num::<T>();
                        let mut buf = before.clone();
                        // SAFETY: every buffer is sized for these exact arguments.
                        unsafe {
                            T::her2(
                                layout,
                                uplo,
                                n as i32,
                                alpha2,
                                xb.as_ptr(),
                                incx,
                                yb.as_ptr(),
                                incy,
                                buf.as_mut_ptr(),
                                lda as i32,
                            );
                        }
                        let what = format!(
                            "{} her2 {layout:?} {uplo:?} n={n} incx={incx} incy={incy}",
                            name::<T>()
                        );
                        check_triangle_update(
                            &before,
                            &buf,
                            n,
                            uplo,
                            &slot_of,
                            &|i, j| her2_entry(&a, alpha2, &x, &y, i, j),
                            &what,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_cher_cher2_match_reference_and_zero_diag_imag() {
    check_her_her2_reference::<Complex32>();
}

#[test]
fn test_issue_14_zher_zher2_match_reference_and_zero_diag_imag() {
    check_her_her2_reference::<Complex64>();
}

/// `alpha == 0` leaves `A` — including an imaginary diagonal — bitwise
/// untouched without reading NaN `x` / `y`; invalid arguments and null
/// pointers are no-ops.
fn check_her_her2_quick_returns_and_rejects<T: ComplexApi>() {
    let x_nan = [nan::<T>(); 16];
    let ones = [T::one(); 16];
    for layout in LAYOUTS {
        for uplo in UPLOS {
            let what = format!("{} her/her2 {layout:?} {uplo:?}", name::<T>());
            let a0 = poison::<T>(20);
            let mut a = a0.clone();
            // SAFETY: A holds a 4x4 matrix with lda = 5; x / y are never read.
            unsafe {
                T::her(layout, uplo, 4, 0.0, x_nan.as_ptr(), -3, a.as_mut_ptr(), 5);
                T::her2(
                    layout,
                    uplo,
                    4,
                    T::zero(),
                    x_nan.as_ptr(),
                    2,
                    x_nan.as_ptr(),
                    -1,
                    a.as_mut_ptr(),
                    5,
                );
            }
            assert_bits_eq(&a0, &a, &format!("{what} alpha=0"));
            // (n, lda, incx, incy)
            for &(n, lda, incx, incy) in &[
                (0, 5, 1, 1),
                (-1, 5, 1, 1),
                (4, 3, 1, 1),
                (4, -5, 1, 1),
                (4, 5, 0, 1),
                (4, 5, 1, 0),
            ] {
                // SAFETY: every call below is rejected before any pointer is used
                // (HER has no `y`, so it only runs for its own invalid cases).
                unsafe {
                    if n <= 0 || lda < n.max(1) || incx == 0 {
                        T::her(
                            layout,
                            uplo,
                            n,
                            1.0,
                            ones.as_ptr(),
                            incx,
                            a.as_mut_ptr(),
                            lda,
                        );
                    }
                    T::her2(
                        layout,
                        uplo,
                        n,
                        T::one(),
                        ones.as_ptr(),
                        incx,
                        ones.as_ptr(),
                        incy,
                        a.as_mut_ptr(),
                        lda,
                    );
                }
                assert_bits_eq(
                    &a0,
                    &a,
                    &format!("{what} n={n} lda={lda} incx={incx} incy={incy}"),
                );
            }
            // SAFETY: null pointers are rejected before any dereference.
            unsafe {
                T::her(layout, uplo, 4, 1.0, ptr::null(), 1, a.as_mut_ptr(), 5);
                T::her(layout, uplo, 4, 1.0, ones.as_ptr(), 1, ptr::null_mut(), 5);
                T::her2(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ptr::null(),
                    1,
                    ones.as_ptr(),
                    1,
                    a.as_mut_ptr(),
                    5,
                );
                T::her2(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ones.as_ptr(),
                    1,
                    ptr::null(),
                    1,
                    a.as_mut_ptr(),
                    5,
                );
                T::her2(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ones.as_ptr(),
                    1,
                    ones.as_ptr(),
                    1,
                    ptr::null_mut(),
                    5,
                );
            }
            assert_bits_eq(&a0, &a, &format!("{what} null pointers"));
        }
    }
}

#[test]
fn test_issue_14_cher_cher2_quick_returns_and_rejects() {
    check_her_her2_quick_returns_and_rejects::<Complex32>();
}

#[test]
fn test_issue_14_zher_zher2_quick_returns_and_rejects() {
    check_her_her2_quick_returns_and_rejects::<Complex64>();
}

/// Reference ?HER / ?HER2 skip the column of a zero `x(j)` (HER2: zero
/// `x(j)` and `y(j)`), so an infinite `x(i)` cannot reach it as
/// `x(i)*conj(0)`. Row-major maps that column onto row `j` of the caller's
/// matrix; the wrapper must preserve it in both layouts.
fn check_her_her2_zero_column_skip<T: ComplexApi>() {
    let inf = T::mk(f64::INFINITY, 0.0);
    let x = [T::zero(), inf];
    let y = [T::zero(), T::one()];
    let lda = 2usize;
    // The protected entry: Fortran column 0 below the diagonal. ColMajor Lower
    // stores it as A[1][0]; RowMajor Upper as A[0][1] (both at slot 1).
    for (layout, uplo) in [
        (CblasLayout::ColMajor, CblasUplo::Lower),
        (CblasLayout::RowMajor, CblasUplo::Upper),
    ] {
        let a0 = vec![T::mk(1.0, 0.5), T::mk(2.0, -1.0), nan(), T::mk(3.0, 0.0)];
        let mut a = a0.clone();
        // SAFETY: 2x2 matrix with lda = 2 and unit-stride 2-vectors.
        unsafe {
            T::her(
                layout,
                uplo,
                2,
                1.0,
                x.as_ptr(),
                1,
                a.as_mut_ptr(),
                lda as i32,
            )
        };
        let what = format!("{} her {layout:?} {uplo:?}", name::<T>());
        assert_eq!(
            a[1].bits(),
            a0[1].bits(),
            "{what}: skipped off-diagonal changed"
        );
        assert_eq!(
            a[0].bits(),
            T::mk(1.0, 0.0).bits(),
            "{what}: diagonal imag not zeroed"
        );
        assert_eq!(a[2].bits(), a0[2].bits(), "{what}: other triangle touched");
        let mut a = a0.clone();
        // SAFETY: as above.
        unsafe {
            T::her2(
                layout,
                uplo,
                2,
                T::one(),
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                lda as i32,
            );
        }
        let what = format!("{} her2 {layout:?} {uplo:?}", name::<T>());
        assert_eq!(
            a[1].bits(),
            a0[1].bits(),
            "{what}: skipped off-diagonal changed"
        );
        assert_eq!(
            a[0].bits(),
            T::mk(1.0, 0.0).bits(),
            "{what}: diagonal imag not zeroed"
        );
        assert_eq!(a[2].bits(), a0[2].bits(), "{what}: other triangle touched");
    }
}

#[test]
fn test_issue_14_cher_cher2_skip_zero_columns_like_reference() {
    check_her_her2_zero_column_skip::<Complex32>();
}

#[test]
fn test_issue_14_zher_zher2_skip_zero_columns_like_reference() {
    check_her_her2_zero_column_skip::<Complex64>();
}

// =============================================================================
// TRMV / TRSV
// =============================================================================

fn check_tr_reference<T: ComplexApi>(solve: bool, sizes: &[usize], incs: &[i32]) {
    let mut rng = Rng::new(if solve { 0xc536 } else { 0xc535 });
    for &n in sizes {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for diag in DIAGS {
                        let mut a: Dense<T> = tri_band(&mut rng, n, n, uplo, diag);
                        if n > 16 {
                            // Keep large triangles well conditioned: shrink the
                            // off-diagonal part (the solves stay backward stable
                            // either way; this keeps |x| moderate).
                            let s = T::mk(1.0 / n as f64, 0.0);
                            for (i, row) in a.iter_mut().enumerate() {
                                for (j, v) in row.iter_mut().enumerate() {
                                    if i != j {
                                        *v *= s;
                                    }
                                }
                            }
                        }
                        let lda = n + 1;
                        let unit = diag == CblasDiag::Unit;
                        let buf = store_dense_triangle(layout, uplo, unit, &a, lda, None);
                        for &inc in incs {
                            let x = rng.vec::<T>(n);
                            let x0 = strided(&x, inc, T::mk(55.0, -55.0));
                            let mut xb = x0.clone();
                            // SAFETY: every buffer is sized for these exact arguments.
                            unsafe {
                                let f = if solve { T::trsv } else { T::trmv };
                                f(
                                    layout,
                                    uplo,
                                    trans,
                                    diag,
                                    n as i32,
                                    buf.as_ptr(),
                                    lda as i32,
                                    xb.as_mut_ptr(),
                                    inc,
                                );
                            }
                            let what = format!(
                                "{} {} {layout:?} {uplo:?} {trans:?} {diag:?} n={n} inc={inc}",
                                name::<T>(),
                                if solve { "trsv" } else { "trmv" }
                            );
                            let got = unstrided(&xb, n, inc);
                            if solve {
                                assert_solves(&a, trans, &got, &x, &what);
                            } else {
                                let (want, bound) = op_mul(&a, n, n, trans, &widen_all(&x));
                                assert_close(&widen_all(&got), &want, &bound, T::EPS, &what);
                            }
                            assert_gaps_unchanged(&x0, &xb, n, inc, &what);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_ctrmv_matches_reference() {
    check_tr_reference::<Complex32>(false, &[1, 4, 9], &INCS);
}

#[test]
fn test_issue_14_ztrmv_matches_reference() {
    check_tr_reference::<Complex64>(false, &[1, 4, 9], &INCS);
}

#[test]
fn test_issue_14_ctrsv_solves() {
    check_tr_reference::<Complex32>(true, &[1, 4, 9], &INCS);
}

#[test]
fn test_issue_14_ztrsv_solves() {
    check_tr_reference::<Complex64>(true, &[1, 4, 9], &INCS);
}

/// `n > 128` takes the native blocked TRSV path (64-wide blocks; 133 leaves a
/// partial last block) in every layout / uplo / trans / diag.
#[test]
fn test_issue_14_ctrsv_ztrsv_blocked_path() {
    check_tr_reference::<Complex32>(true, &[133], &[1, -2]);
    check_tr_reference::<Complex64>(true, &[133], &[1, -2]);
}

fn check_tr_rejects<T: ComplexApi>() {
    let ones = [T::one(); 32];
    let x0 = poison::<T>(8);
    for f in [T::trmv as unsafe fn(_, _, _, _, _, _, _, _, _), T::trsv] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    let mut x = x0.clone();
                    // SAFETY: every call is rejected before any pointer is used.
                    unsafe {
                        // (n, lda, incx); the valid baseline is (4, 4, 1).
                        for &(n, lda, incx) in
                            &[(0, 4, 1), (-1, 4, 1), (4, 3, 1), (4, -4, 1), (4, 4, 0)]
                        {
                            f(
                                layout,
                                uplo,
                                trans,
                                CblasDiag::NonUnit,
                                n,
                                ones.as_ptr(),
                                lda,
                                x.as_mut_ptr(),
                                incx,
                            );
                        }
                        f(
                            layout,
                            uplo,
                            trans,
                            CblasDiag::Unit,
                            4,
                            ptr::null(),
                            4,
                            x.as_mut_ptr(),
                            1,
                        );
                        f(
                            layout,
                            uplo,
                            trans,
                            CblasDiag::Unit,
                            4,
                            ones.as_ptr(),
                            4,
                            ptr::null_mut(),
                            1,
                        );
                    }
                    let what = format!("{} tr {layout:?} {uplo:?} {trans:?}", name::<T>());
                    assert_bits_eq(&x0, &x, &what);
                }
            }
        }
    }
}

#[test]
fn test_issue_14_ctrmv_ctrsv_reject_invalid_arguments() {
    check_tr_rejects::<Complex32>();
}

#[test]
fn test_issue_14_ztrmv_ztrsv_reject_invalid_arguments() {
    check_tr_rejects::<Complex64>();
}

/// A hand-checked row-major `ConjTrans` solve (the buffer is written out
/// literally in row-major order).
#[test]
fn test_issue_14_ztrsv_rowmajor_conjtrans_hand_example() {
    let c = Complex64::new;
    // A = [[1, i], [0, 2]] (upper), row-major: [A00, A01, A10(unused), A11].
    let a = [c(1.0, 0.0), c(0.0, 1.0), c(f64::NAN, f64::NAN), c(2.0, 0.0)];
    // A^H = [[1, 0], [-i, 2]]; solve A^H x = b with x = [1+i, 2]:
    // b = [1+i, -i(1+i) + 4] = [1+i, 5-i].
    let mut x = [c(1.0, 1.0), c(5.0, -1.0)];
    // SAFETY: 2x2 matrix with lda = 2 and a unit-stride 2-vector.
    unsafe {
        cblas_ztrsv(
            CblasLayout::RowMajor,
            CblasUplo::Upper,
            CblasTranspose::ConjTrans,
            CblasDiag::NonUnit,
            2,
            a.as_ptr(),
            2,
            x.as_mut_ptr(),
            1,
        );
    }
    assert_eq!(x, [c(1.0, 1.0), c(2.0, 0.0)]);
}
