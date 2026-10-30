//! Unit tests for the CBLAS packed-storage Level-2 wrappers (issue #14):
//! `cblas_{s,d}spmv`, `cblas_{s,d}spr`, `cblas_{s,d}spr2`, `cblas_{c,z}hpmv`,
//! `cblas_{c,z}hpr`, `cblas_{c,z}hpr2`, `cblas_{s,d,c,z}tpmv`,
//! `cblas_{s,d,c,z}tpsv`.
//!
//! Every routine is compared, in both layouts and for every
//! `uplo` / `trans` / `diag` and increment in `{1, 2, -1, -3}`, against a
//! naive dense reference built from the CBLAS packing definition of the
//! layout (column-wise for `ColMajor`, row-wise for `RowMajor`). Vector gaps
//! and unit diagonals hold NaN, so a wrong stride or a read of a unit
//! diagonal poisons the result.

use super::*;
use crate::cblas::strided::tests::{
    DIAGS, Dense, INCS, LAYOUTS, Num, Rng, TRANSES, UPLOS, assert_bits_eq, assert_close,
    assert_gaps_unchanged, assert_solves, check_triangle_update, her_entry, her2_entry,
    hermitian_band, mv_ref, nan, op_mul, packed_slot, poison, store_packed, strided, tri_band,
    unstrided, widen_all, zero,
};
use core::ptr;

/// Uniform adapter over the per-precision packed entry points. For the real
/// precisions `hpmv` / `hpr` / `hpr2` are SPMV / SPR / SPR2 (the same maths
/// for a real symmetric matrix). `hpr` takes `alpha` as `f64`; the test
/// values are exactly representable in `f32`.
trait PackedApi: Num {
    #[allow(clippy::too_many_arguments)]
    unsafe fn hpmv(
        layout: CblasLayout,
        uplo: CblasUplo,
        n: i32,
        alpha: Self,
        ap: *const Self,
        x: *const Self,
        incx: i32,
        beta: Self,
        y: *mut Self,
        incy: i32,
    );
    unsafe fn hpr(
        layout: CblasLayout,
        uplo: CblasUplo,
        n: i32,
        alpha: f64,
        x: *const Self,
        incx: i32,
        ap: *mut Self,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn hpr2(
        layout: CblasLayout,
        uplo: CblasUplo,
        n: i32,
        alpha: Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        ap: *mut Self,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn tpmv(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        incx: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn tpsv(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        incx: i32,
    );
}

impl PackedApi for f32 {
    unsafe fn hpmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        ap: *const Self,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_sspmv(l, u, n, al, ap, x, ix, be, y, iy) }
    }
    unsafe fn hpr(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: f64,
        x: *const Self,
        ix: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_sspr(l, u, n, al as f32, x, ix, ap) }
    }
    unsafe fn hpr2(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_sspr2(l, u, n, al, x, ix, y, iy, ap) }
    }
    unsafe fn tpmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_stpmv(l, u, t, d, n, ap, x, ix) }
    }
    unsafe fn tpsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_stpsv(l, u, t, d, n, ap, x, ix) }
    }
}

impl PackedApi for f64 {
    unsafe fn hpmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        ap: *const Self,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dspmv(l, u, n, al, ap, x, ix, be, y, iy) }
    }
    unsafe fn hpr(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: f64,
        x: *const Self,
        ix: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dspr(l, u, n, al, x, ix, ap) }
    }
    unsafe fn hpr2(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dspr2(l, u, n, al, x, ix, y, iy, ap) }
    }
    unsafe fn tpmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dtpmv(l, u, t, d, n, ap, x, ix) }
    }
    unsafe fn tpsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dtpsv(l, u, t, d, n, ap, x, ix) }
    }
}

impl PackedApi for Complex32 {
    unsafe fn hpmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        ap: *const Self,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` / `be` outlive the call.
        unsafe { cblas_chpmv(l, u, n, &al, ap, x, ix, &be, y, iy) }
    }
    unsafe fn hpr(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: f64,
        x: *const Self,
        ix: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_chpr(l, u, n, al as f32, x, ix, ap) }
    }
    unsafe fn hpr2(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_chpr2(l, u, n, &al, x, ix, y, iy, ap) }
    }
    unsafe fn tpmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ctpmv(l, u, t, d, n, ap, x, ix) }
    }
    unsafe fn tpsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ctpsv(l, u, t, d, n, ap, x, ix) }
    }
}

impl PackedApi for Complex64 {
    unsafe fn hpmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        ap: *const Self,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` / `be` outlive the call.
        unsafe { cblas_zhpmv(l, u, n, &al, ap, x, ix, &be, y, iy) }
    }
    unsafe fn hpr(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: f64,
        x: *const Self,
        ix: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_zhpr(l, u, n, al, x, ix, ap) }
    }
    unsafe fn hpr2(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        al: Self,
        x: *const Self,
        ix: i32,
        y: *const Self,
        iy: i32,
        ap: *mut Self,
    ) {
        // SAFETY: forwards the caller's contract; `al` outlives the call.
        unsafe { cblas_zhpr2(l, u, n, &al, x, ix, y, iy, ap) }
    }
    unsafe fn tpmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ztpmv(l, u, t, d, n, ap, x, ix) }
    }
    unsafe fn tpsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        ap: *const Self,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ztpsv(l, u, t, d, n, ap, x, ix) }
    }
}

fn name<T>() -> &'static str {
    core::any::type_name::<T>()
}

// =============================================================================
// SPMV / HPMV
// =============================================================================

fn check_hpmv_reference<T: PackedApi>() {
    let mut rng = Rng::new(0x9a31);
    for n in [1usize, 2, 5, 8] {
        let a = hermitian_band::<T>(&mut rng, n, n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                // Imaginary garbage on the stored diagonal must be ignored.
                let ap = store_packed(layout, uplo, false, &a, Some(-0.5));
                for incx in INCS {
                    for incy in INCS {
                        let x = rng.vec::<T>(n);
                        let y = rng.vec::<T>(n);
                        let (alpha, beta) = (rng.num::<T>(), rng.num::<T>());
                        let xb = strided(&x, incx, nan());
                        let y0 = strided(&y, incy, T::mk(77.0, -77.0));
                        let mut yb = y0.clone();
                        // SAFETY: every buffer is sized for these exact arguments.
                        unsafe {
                            T::hpmv(
                                layout,
                                uplo,
                                n as i32,
                                alpha,
                                ap.as_ptr(),
                                xb.as_ptr(),
                                incx,
                                beta,
                                yb.as_mut_ptr(),
                                incy,
                            );
                        }
                        let what = format!(
                            "{} hpmv {layout:?} {uplo:?} n={n} incx={incx} incy={incy}",
                            name::<T>()
                        );
                        let (want, bound) =
                            mv_ref(&a, n, n, CblasTranspose::NoTrans, alpha, &x, beta, &y);
                        let got = widen_all(&unstrided(&yb, n, incy));
                        assert_close(&got, &want, &bound, T::EPS, &what);
                        assert_gaps_unchanged(&y0, &yb, n, incy, &what);
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_sspmv_matches_reference() {
    check_hpmv_reference::<f32>();
}

#[test]
fn test_issue_14_dspmv_matches_reference() {
    check_hpmv_reference::<f64>();
}

#[test]
fn test_issue_14_chpmv_matches_reference() {
    check_hpmv_reference::<Complex32>();
}

#[test]
fn test_issue_14_zhpmv_matches_reference() {
    check_hpmv_reference::<Complex64>();
}

fn check_hpmv_quick_returns_and_rejects<T: PackedApi>() {
    let mut rng = Rng::new(0x9a32);
    let n = 4usize;
    let ap_nan = vec![nan::<T>(); n * (n + 1) / 2];
    let x_nan = vec![nan::<T>(); 3 * n];
    let a = hermitian_band::<T>(&mut rng, n, n);
    for layout in LAYOUTS {
        for uplo in UPLOS {
            let what = format!("{} hpmv {layout:?} {uplo:?}", name::<T>());
            for incy in INCS {
                let len = (n - 1) * incy.unsigned_abs() as usize + 1;
                // alpha = 0, beta = 1: y untouched.
                let y0 = poison::<T>(len);
                let mut y = y0.clone();
                // SAFETY: NaN-filled AP / x and y are large enough for any stride.
                unsafe {
                    T::hpmv(
                        layout,
                        uplo,
                        n as i32,
                        T::zero(),
                        ap_nan.as_ptr(),
                        x_nan.as_ptr(),
                        -3,
                        T::one(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                assert_bits_eq(&y0, &y, &format!("{what} alpha=0 beta=1 incy={incy}"));
                // alpha = 0: exactly beta*y; NaN AP and x are never read.
                let yv = rng.vec::<T>(n);
                let beta = rng.num::<T>();
                let mut y = strided(&yv, incy, T::mk(5.0, 5.0));
                // SAFETY: as above.
                unsafe {
                    T::hpmv(
                        layout,
                        uplo,
                        n as i32,
                        T::zero(),
                        ap_nan.as_ptr(),
                        x_nan.as_ptr(),
                        2,
                        beta,
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                let want: Vec<T> = yv.iter().map(|&v| beta * v).collect();
                assert_bits_eq(&want, &unstrided(&y, n, incy), &format!("{what} alpha=0"));
                // alpha = 0, beta = 0: zero fill of a NaN y.
                let mut y = strided(&vec![nan::<T>(); n], incy, nan());
                // SAFETY: as above.
                unsafe {
                    T::hpmv(
                        layout,
                        uplo,
                        n as i32,
                        T::zero(),
                        ap_nan.as_ptr(),
                        x_nan.as_ptr(),
                        1,
                        T::zero(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                let zeros = vec![zero::<T>(); n];
                assert_bits_eq(&zeros, &unstrided(&y, n, incy), &format!("{what} a=b=0"));
            }
            // beta = 0, alpha != 0 overwrites a NaN y.
            let ap = store_packed(layout, uplo, false, &a, Some(3.0));
            let x = rng.vec::<T>(n);
            let alpha = rng.num::<T>();
            let mut y = vec![nan::<T>(); n];
            // SAFETY: packed triangle, x and y match the arguments.
            unsafe {
                T::hpmv(
                    layout,
                    uplo,
                    n as i32,
                    alpha,
                    ap.as_ptr(),
                    x.as_ptr(),
                    1,
                    T::zero(),
                    y.as_mut_ptr(),
                    1,
                );
            }
            let zeros = vec![zero::<T>(); n];
            let (want, bound) = mv_ref(
                &a,
                n,
                n,
                CblasTranspose::NoTrans,
                alpha,
                &x,
                T::zero(),
                &zeros,
            );
            let got = widen_all(&y);
            assert!(got.iter().all(|v| v.is_finite()), "{what} beta=0: {got:?}");
            assert_close(&got, &want, &bound, T::EPS, &format!("{what} beta=0"));

            // n <= 0, zero increments and null pointers: no-op.
            let ones = [T::one(); 16];
            let y0 = poison::<T>(8);
            for &(n, incx, incy) in &[(0, 1, 1), (-2, 1, 1), (4, 0, 1), (4, 1, 0)] {
                let mut y = y0.clone();
                // SAFETY: every case is rejected before any pointer is used.
                unsafe {
                    T::hpmv(
                        layout,
                        uplo,
                        n,
                        T::one(),
                        ones.as_ptr(),
                        ones.as_ptr(),
                        incx,
                        T::one(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                assert_bits_eq(&y0, &y, &format!("{what} n={n} incx={incx} incy={incy}"));
            }
            let mut y = y0.clone();
            // SAFETY: null pointers are rejected before any dereference.
            unsafe {
                T::hpmv(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ptr::null(),
                    ones.as_ptr(),
                    1,
                    T::one(),
                    y.as_mut_ptr(),
                    1,
                );
                T::hpmv(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ones.as_ptr(),
                    ptr::null(),
                    1,
                    T::one(),
                    y.as_mut_ptr(),
                    1,
                );
                T::hpmv(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ones.as_ptr(),
                    ones.as_ptr(),
                    1,
                    T::one(),
                    ptr::null_mut(),
                    1,
                );
            }
            assert_bits_eq(&y0, &y, &format!("{what} null pointers"));
        }
    }
}

#[test]
fn test_issue_14_sspmv_quick_returns_and_rejects() {
    check_hpmv_quick_returns_and_rejects::<f32>();
}

#[test]
fn test_issue_14_dspmv_quick_returns_and_rejects() {
    check_hpmv_quick_returns_and_rejects::<f64>();
}

#[test]
fn test_issue_14_chpmv_quick_returns_and_rejects() {
    check_hpmv_quick_returns_and_rejects::<Complex32>();
}

#[test]
fn test_issue_14_zhpmv_quick_returns_and_rejects() {
    check_hpmv_quick_returns_and_rejects::<Complex64>();
}

// =============================================================================
// SPR / HPR and SPR2 / HPR2
// =============================================================================

/// Rank-1 / rank-2 updates of the packed triangle for every layout / uplo /
/// increment. The stored diagonal carries an imaginary part that the update
/// must zero (complex precisions).
fn check_rank_updates<T: PackedApi>() {
    let mut rng = Rng::new(0x9a33);
    for n in [1usize, 3, 6] {
        let a = hermitian_band::<T>(&mut rng, n, n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                let slot_of = |i: usize, j: usize| packed_slot(layout, uplo, i, j, n);
                for incx in INCS {
                    let alpha = if incx > 0 { 0.75 } else { -1.25 };
                    let before = store_packed(layout, uplo, false, &a, Some(0.625));
                    let mut ap = before.clone();
                    let x = rng.vec::<T>(n);
                    let xb = strided(&x, incx, nan());
                    // SAFETY: every buffer is sized for these exact arguments.
                    unsafe {
                        T::hpr(
                            layout,
                            uplo,
                            n as i32,
                            alpha,
                            xb.as_ptr(),
                            incx,
                            ap.as_mut_ptr(),
                        )
                    };
                    let what = format!("{} hpr {layout:?} {uplo:?} n={n} incx={incx}", name::<T>());
                    check_triangle_update(
                        &before,
                        &ap,
                        n,
                        uplo,
                        &slot_of,
                        &|i, j| her_entry(&a, alpha, &x, i, j),
                        &what,
                    );
                    for incy in INCS {
                        let mut ap = before.clone();
                        let y = rng.vec::<T>(n);
                        let yb = strided(&y, incy, nan());
                        let alpha2 = rng.num::<T>();
                        // SAFETY: every buffer is sized for these exact arguments.
                        unsafe {
                            T::hpr2(
                                layout,
                                uplo,
                                n as i32,
                                alpha2,
                                xb.as_ptr(),
                                incx,
                                yb.as_ptr(),
                                incy,
                                ap.as_mut_ptr(),
                            );
                        }
                        let what = format!(
                            "{} hpr2 {layout:?} {uplo:?} n={n} incx={incx} incy={incy}",
                            name::<T>()
                        );
                        check_triangle_update(
                            &before,
                            &ap,
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
fn test_issue_14_sspr_sspr2_match_reference() {
    check_rank_updates::<f32>();
}

#[test]
fn test_issue_14_dspr_dspr2_match_reference() {
    check_rank_updates::<f64>();
}

#[test]
fn test_issue_14_chpr_chpr2_match_reference_and_zero_diag_imag() {
    check_rank_updates::<Complex32>();
}

#[test]
fn test_issue_14_zhpr_zhpr2_match_reference_and_zero_diag_imag() {
    check_rank_updates::<Complex64>();
}

/// `alpha == 0` leaves `AP` bitwise untouched — including an imaginary
/// diagonal — without reading `x` / `y`; invalid arguments and null pointers
/// are no-ops too.
fn check_rank_update_quick_returns<T: PackedApi>() {
    let n = 4usize;
    let len = n * (n + 1) / 2;
    let x_nan = vec![nan::<T>(); 3 * n];
    let ones = [T::one(); 16];
    for layout in LAYOUTS {
        for uplo in UPLOS {
            let what = format!("{} hpr/hpr2 {layout:?} {uplo:?}", name::<T>());
            let before = poison::<T>(len);
            let mut ap = before.clone();
            // SAFETY: AP holds the n*(n+1)/2 triangle; x / y are never read.
            unsafe {
                T::hpr(
                    layout,
                    uplo,
                    n as i32,
                    0.0,
                    x_nan.as_ptr(),
                    -2,
                    ap.as_mut_ptr(),
                );
                T::hpr2(
                    layout,
                    uplo,
                    n as i32,
                    T::zero(),
                    x_nan.as_ptr(),
                    3,
                    x_nan.as_ptr(),
                    -1,
                    ap.as_mut_ptr(),
                );
            }
            assert_bits_eq(&before, &ap, &format!("{what} alpha=0"));
            for &(n, incx, incy) in &[(0, 1, 1), (-1, 1, 1), (4, 0, 1), (4, 1, 0)] {
                // SAFETY: every call below is rejected before any pointer is used
                // (HPR has no `y`, so it only runs for its own invalid cases).
                unsafe {
                    if n <= 0 || incx == 0 {
                        T::hpr(layout, uplo, n, 1.0, ones.as_ptr(), incx, ap.as_mut_ptr());
                    }
                    T::hpr2(
                        layout,
                        uplo,
                        n,
                        T::one(),
                        ones.as_ptr(),
                        incx,
                        ones.as_ptr(),
                        incy,
                        ap.as_mut_ptr(),
                    );
                }
                assert_bits_eq(
                    &before,
                    &ap,
                    &format!("{what} n={n} incx={incx} incy={incy}"),
                );
            }
            // SAFETY: null pointers are rejected before any dereference.
            unsafe {
                T::hpr(layout, uplo, 4, 1.0, ptr::null(), 1, ap.as_mut_ptr());
                T::hpr(layout, uplo, 4, 1.0, ones.as_ptr(), 1, ptr::null_mut());
                T::hpr2(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ptr::null(),
                    1,
                    ones.as_ptr(),
                    1,
                    ap.as_mut_ptr(),
                );
                T::hpr2(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ones.as_ptr(),
                    1,
                    ptr::null(),
                    1,
                    ap.as_mut_ptr(),
                );
                T::hpr2(
                    layout,
                    uplo,
                    4,
                    T::one(),
                    ones.as_ptr(),
                    1,
                    ones.as_ptr(),
                    1,
                    ptr::null_mut(),
                );
            }
            assert_bits_eq(&before, &ap, &format!("{what} null pointers"));
        }
    }
}

#[test]
fn test_issue_14_sspr_sspr2_quick_returns_and_rejects() {
    check_rank_update_quick_returns::<f32>();
}

#[test]
fn test_issue_14_dspr_dspr2_quick_returns_and_rejects() {
    check_rank_update_quick_returns::<f64>();
}

#[test]
fn test_issue_14_chpr_chpr2_quick_returns_and_rejects() {
    check_rank_update_quick_returns::<Complex32>();
}

#[test]
fn test_issue_14_zhpr_zhpr2_quick_returns_and_rejects() {
    check_rank_update_quick_returns::<Complex64>();
}

/// A null complex `alpha` / `beta` is a no-op (HPMV, HPR2).
#[test]
fn test_issue_14_cz_hpmv_hpr2_null_scalars_are_noops() {
    let one32 = Complex32::new(1.0, 0.0);
    let one64 = Complex64::new(1.0, 0.0);
    let v32 = [one32; 6];
    let v64 = [one64; 6];
    let mut y32 = poison::<Complex32>(6);
    let mut y64 = poison::<Complex64>(6);
    let (y32_0, y64_0) = (y32.clone(), y64.clone());
    for layout in LAYOUTS {
        // SAFETY: the null scalar pointers are rejected before any dereference.
        unsafe {
            cblas_chpmv(
                layout,
                CblasUplo::Upper,
                3,
                ptr::null(),
                v32.as_ptr(),
                v32.as_ptr(),
                1,
                &one32,
                y32.as_mut_ptr(),
                1,
            );
            cblas_zhpmv(
                layout,
                CblasUplo::Lower,
                3,
                &one64,
                v64.as_ptr(),
                v64.as_ptr(),
                1,
                ptr::null(),
                y64.as_mut_ptr(),
                1,
            );
            cblas_chpr2(
                layout,
                CblasUplo::Upper,
                3,
                ptr::null(),
                v32.as_ptr(),
                1,
                v32.as_ptr(),
                1,
                y32.as_mut_ptr(),
            );
            cblas_zhpr2(
                layout,
                CblasUplo::Lower,
                3,
                ptr::null(),
                v64.as_ptr(),
                1,
                v64.as_ptr(),
                1,
                y64.as_mut_ptr(),
            );
        }
    }
    assert_bits_eq(&y32_0, &y32, "c null scalars");
    assert_bits_eq(&y64_0, &y64, "z null scalars");
}

/// An order whose packed byte size overflows (no buffer can be that large) is
/// rejected before any pointer is used.
#[test]
fn test_issue_14_packed_size_overflow_is_a_noop() {
    let huge = i32::MAX;
    let one = Complex64::new(1.0, 0.0);
    let v = [one; 4];
    let y0 = poison::<Complex64>(4);
    let mut y = y0.clone();
    let mut ap = y0.clone();
    for layout in LAYOUTS {
        // SAFETY: `packed_len` rejects the order before any pointer is used.
        unsafe {
            cblas_zhpmv(
                layout,
                CblasUplo::Upper,
                huge,
                &one,
                v.as_ptr(),
                v.as_ptr(),
                1,
                &one,
                y.as_mut_ptr(),
                1,
            );
            cblas_zhpr(
                layout,
                CblasUplo::Lower,
                huge,
                1.0,
                v.as_ptr(),
                1,
                ap.as_mut_ptr(),
            );
            cblas_zhpr2(
                layout,
                CblasUplo::Upper,
                huge,
                &one,
                v.as_ptr(),
                1,
                v.as_ptr(),
                1,
                ap.as_mut_ptr(),
            );
            cblas_ztpmv(
                layout,
                CblasUplo::Upper,
                CblasTranspose::NoTrans,
                CblasDiag::NonUnit,
                huge,
                v.as_ptr(),
                y.as_mut_ptr(),
                1,
            );
            cblas_ztpsv(
                layout,
                CblasUplo::Lower,
                CblasTranspose::ConjTrans,
                CblasDiag::Unit,
                huge,
                v.as_ptr(),
                y.as_mut_ptr(),
                1,
            );
        }
    }
    assert_bits_eq(&y0, &y, "overflowing order: y / x");
    assert_bits_eq(&y0, &ap, "overflowing order: AP");
}

// =============================================================================
// TPMV / TPSV
// =============================================================================

fn check_tp_reference<T: PackedApi>(solve: bool) {
    let mut rng = Rng::new(if solve { 0x9a35 } else { 0x9a34 });
    for n in [1usize, 3, 6] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for diag in DIAGS {
                        let a: Dense<T> = tri_band(&mut rng, n, n, uplo, diag);
                        let ap = store_packed(layout, uplo, diag == CblasDiag::Unit, &a, None);
                        for inc in INCS {
                            let x = rng.vec::<T>(n);
                            let x0 = strided(&x, inc, T::mk(55.0, -55.0));
                            let mut xb = x0.clone();
                            // SAFETY: every buffer is sized for these exact arguments.
                            unsafe {
                                let f = if solve { T::tpsv } else { T::tpmv };
                                f(
                                    layout,
                                    uplo,
                                    trans,
                                    diag,
                                    n as i32,
                                    ap.as_ptr(),
                                    xb.as_mut_ptr(),
                                    inc,
                                );
                            }
                            let what = format!(
                                "{} {} {layout:?} {uplo:?} {trans:?} {diag:?} n={n} inc={inc}",
                                name::<T>(),
                                if solve { "tpsv" } else { "tpmv" }
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
fn test_issue_14_stpmv_matches_reference() {
    check_tp_reference::<f32>(false);
}

#[test]
fn test_issue_14_dtpmv_matches_reference() {
    check_tp_reference::<f64>(false);
}

#[test]
fn test_issue_14_ctpmv_matches_reference() {
    check_tp_reference::<Complex32>(false);
}

#[test]
fn test_issue_14_ztpmv_matches_reference() {
    check_tp_reference::<Complex64>(false);
}

#[test]
fn test_issue_14_stpsv_solves() {
    check_tp_reference::<f32>(true);
}

#[test]
fn test_issue_14_dtpsv_solves() {
    check_tp_reference::<f64>(true);
}

#[test]
fn test_issue_14_ctpsv_solves() {
    check_tp_reference::<Complex32>(true);
}

#[test]
fn test_issue_14_ztpsv_solves() {
    check_tp_reference::<Complex64>(true);
}

fn check_tp_rejects<T: PackedApi>() {
    let ones = [T::one(); 16];
    let x0 = poison::<T>(8);
    for f in [T::tpmv as unsafe fn(_, _, _, _, _, _, _, _), T::tpsv] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    let mut x = x0.clone();
                    // SAFETY: every call is rejected before any pointer is used.
                    unsafe {
                        for &(n, incx) in &[(0, 1), (-3, 1), (4, 0)] {
                            f(
                                layout,
                                uplo,
                                trans,
                                CblasDiag::NonUnit,
                                n,
                                ones.as_ptr(),
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
                            ptr::null_mut(),
                            1,
                        );
                    }
                    let what = format!("{} tp {layout:?} {uplo:?} {trans:?}", name::<T>());
                    assert_bits_eq(&x0, &x, &what);
                }
            }
        }
    }
}

#[test]
fn test_issue_14_stpmv_stpsv_reject_invalid_arguments() {
    check_tp_rejects::<f32>();
}

#[test]
fn test_issue_14_dtpmv_dtpsv_reject_invalid_arguments() {
    check_tp_rejects::<f64>();
}

#[test]
fn test_issue_14_ctpmv_ctpsv_reject_invalid_arguments() {
    check_tp_rejects::<Complex32>();
}

#[test]
fn test_issue_14_ztpmv_ztpsv_reject_invalid_arguments() {
    check_tp_rejects::<Complex64>();
}

/// Hand-checked row-major examples written out literally from the CBLAS
/// row-wise packing definition.
#[test]
fn test_issue_14_rowmajor_packed_hand_examples() {
    let c = Complex64::new;
    // Hermitian A = [[2, 1+i], [1-i, 3]]; row-major Upper packing is
    // [A00, A01, A11] = [2, 1+i, 3].
    let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
    let x = [c(1.0, 0.0), c(0.0, 1.0)];
    let mut y = [c(0.0, 0.0); 2];
    let (one, zero_c) = (c(1.0, 0.0), c(0.0, 0.0));
    // SAFETY: n = 2 packed triangle and unit-stride 2-vectors.
    unsafe {
        cblas_zhpmv(
            CblasLayout::RowMajor,
            CblasUplo::Upper,
            2,
            &one,
            ap.as_ptr(),
            x.as_ptr(),
            1,
            &zero_c,
            y.as_mut_ptr(),
            1,
        );
    }
    // A*x = [2 + (1+i)i, (1-i) + 3i] = [1+i, 1+2i].
    assert_eq!(y, [c(1.0, 1.0), c(1.0, 2.0)]);

    // Lower triangular L = [[1, 0], [2, 3]]; row-major Lower packing is
    // [L00, L10, L11] = [1, 2, 3]. Solve L^T x = [5, 6]: x1 = 2, x0 = 1.
    let lp = [1.0f64, 2.0, 3.0];
    let mut b = [5.0f64, 6.0];
    // SAFETY: n = 2 packed triangle and a unit-stride 2-vector.
    unsafe {
        cblas_dtpsv(
            CblasLayout::RowMajor,
            CblasUplo::Lower,
            CblasTranspose::Trans,
            CblasDiag::NonUnit,
            2,
            lp.as_ptr(),
            b.as_mut_ptr(),
            1,
        );
    }
    assert_eq!(b, [1.0, 2.0]);
}
