//! Unit tests for the CBLAS band-storage Level-2 wrappers (issue #14):
//! `cblas_{s,d,c,z}gbmv`, `cblas_{s,d}sbmv`, `cblas_{c,z}hbmv`,
//! `cblas_{s,d,c,z}tbmv`, `cblas_{s,d,c,z}tbsv`.
//!
//! Every routine is compared, in both layouts and for every
//! `uplo` / `trans` / `diag` and increment in `{1, 2, -1, -3}`, against a
//! naive dense reference built from the CBLAS band-storage definition of the
//! layout. Unused band slots, vector gaps and unit diagonals hold NaN, so any
//! out-of-band read poisons the result.

use super::*;
use crate::cblas::strided::tests::{
    DIAGS, Dense, INCS, LAYOUTS, Num, Rng, TRANSES, UPLOS, assert_bits_eq, assert_close,
    assert_gaps_unchanged, assert_solves, general_band, hermitian_band, mv_ref, nan, op_mul,
    poison, store_band, store_gb, strided, tri_band, unstrided, widen_all, zero,
};
use core::ptr;

/// Uniform adapter over the per-precision band entry points. `hbmv` is SBMV
/// for the real precisions and HBMV for the complex ones (the same maths for
/// a real symmetric matrix).
trait BandApi: Num {
    #[allow(clippy::too_many_arguments)]
    unsafe fn gbmv(
        layout: CblasLayout,
        trans: CblasTranspose,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: Self,
        y: *mut Self,
        incy: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn hbmv(
        layout: CblasLayout,
        uplo: CblasUplo,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        incx: i32,
        beta: Self,
        y: *mut Self,
        incy: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn tbmv(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        incx: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn tbsv(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        incx: i32,
    );
}

impl BandApi for f32 {
    unsafe fn gbmv(
        l: CblasLayout,
        t: CblasTranspose,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_sgbmv(l, t, m, n, kl, ku, al, a, lda, x, ix, be, y, iy) }
    }
    unsafe fn hbmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        k: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ssbmv(l, u, n, k, al, a, lda, x, ix, be, y, iy) }
    }
    unsafe fn tbmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_stbmv(l, u, t, d, n, k, a, lda, x, ix) }
    }
    unsafe fn tbsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_stbsv(l, u, t, d, n, k, a, lda, x, ix) }
    }
}

impl BandApi for f64 {
    unsafe fn gbmv(
        l: CblasLayout,
        t: CblasTranspose,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dgbmv(l, t, m, n, kl, ku, al, a, lda, x, ix, be, y, iy) }
    }
    unsafe fn hbmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        k: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dsbmv(l, u, n, k, al, a, lda, x, ix, be, y, iy) }
    }
    unsafe fn tbmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dtbmv(l, u, t, d, n, k, a, lda, x, ix) }
    }
    unsafe fn tbsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_dtbsv(l, u, t, d, n, k, a, lda, x, ix) }
    }
}

impl BandApi for Complex32 {
    unsafe fn gbmv(
        l: CblasLayout,
        t: CblasTranspose,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` / `be` outlive the call.
        unsafe { cblas_cgbmv(l, t, m, n, kl, ku, &al, a, lda, x, ix, &be, y, iy) }
    }
    unsafe fn hbmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        k: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` / `be` outlive the call.
        unsafe { cblas_chbmv(l, u, n, k, &al, a, lda, x, ix, &be, y, iy) }
    }
    unsafe fn tbmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ctbmv(l, u, t, d, n, k, a, lda, x, ix) }
    }
    unsafe fn tbsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ctbsv(l, u, t, d, n, k, a, lda, x, ix) }
    }
}

impl BandApi for Complex64 {
    unsafe fn gbmv(
        l: CblasLayout,
        t: CblasTranspose,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` / `be` outlive the call.
        unsafe { cblas_zgbmv(l, t, m, n, kl, ku, &al, a, lda, x, ix, &be, y, iy) }
    }
    unsafe fn hbmv(
        l: CblasLayout,
        u: CblasUplo,
        n: i32,
        k: i32,
        al: Self,
        a: *const Self,
        lda: i32,
        x: *const Self,
        ix: i32,
        be: Self,
        y: *mut Self,
        iy: i32,
    ) {
        // SAFETY: forwards the caller's contract; `al` / `be` outlive the call.
        unsafe { cblas_zhbmv(l, u, n, k, &al, a, lda, x, ix, &be, y, iy) }
    }
    unsafe fn tbmv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ztbmv(l, u, t, d, n, k, a, lda, x, ix) }
    }
    unsafe fn tbsv(
        l: CblasLayout,
        u: CblasUplo,
        t: CblasTranspose,
        d: CblasDiag,
        n: i32,
        k: i32,
        a: *const Self,
        lda: i32,
        x: *mut Self,
        ix: i32,
    ) {
        // SAFETY: forwards the caller's contract unchanged.
        unsafe { cblas_ztbsv(l, u, t, d, n, k, a, lda, x, ix) }
    }
}

fn name<T>() -> &'static str {
    core::any::type_name::<T>()
}

/// Vector lengths `(lenx, leny)` of `op(A)` for an `m x n` matrix.
fn op_lens(trans: CblasTranspose, m: usize, n: usize) -> (usize, usize) {
    if trans == CblasTranspose::NoTrans {
        (n, m)
    } else {
        (m, n)
    }
}

// =============================================================================
// GBMV
// =============================================================================

fn check_gbmv_reference<T: BandApi>() {
    let mut rng = Rng::new(0x6b31);
    // Rectangular both ways, kl != ku, diagonal only, and bands wider than the
    // matrix.
    let shapes = [
        (5usize, 4usize, 1usize, 2usize),
        (4, 6, 2, 1),
        (3, 3, 0, 0),
        (2, 5, 3, 4),
        (6, 2, 4, 0),
    ];
    for layout in LAYOUTS {
        for trans in TRANSES {
            for &(m, n, kl, ku) in &shapes {
                let a = general_band::<T>(&mut rng, m, n, kl, ku);
                for incx in INCS {
                    for incy in INCS {
                        // Both the minimal and a padded leading dimension.
                        let lda = kl + ku + 1 + usize::from(incx < 0);
                        let buf = store_gb(layout, &a, n, kl, ku, lda);
                        let (lenx, leny) = op_lens(trans, m, n);
                        let x = rng.vec::<T>(lenx);
                        let y = rng.vec::<T>(leny);
                        let (alpha, beta) = (rng.num::<T>(), rng.num::<T>());
                        let xb = strided(&x, incx, nan());
                        let y0 = strided(&y, incy, T::mk(77.0, -77.0));
                        let mut yb = y0.clone();
                        // SAFETY: every buffer is sized for these exact arguments.
                        unsafe {
                            T::gbmv(
                                layout,
                                trans,
                                m as i32,
                                n as i32,
                                kl as i32,
                                ku as i32,
                                alpha,
                                buf.as_ptr(),
                                lda as i32,
                                xb.as_ptr(),
                                incx,
                                beta,
                                yb.as_mut_ptr(),
                                incy,
                            );
                        }
                        let what = format!(
                            "{} gbmv {layout:?} {trans:?} m={m} n={n} kl={kl} ku={ku} \
                             lda={lda} incx={incx} incy={incy}",
                            name::<T>()
                        );
                        let (want, bound) = mv_ref(&a, m, n, trans, alpha, &x, beta, &y);
                        let got = widen_all(&unstrided(&yb, leny, incy));
                        assert_close(&got, &want, &bound, T::EPS, &what);
                        assert_gaps_unchanged(&y0, &yb, leny, incy, &what);
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_sgbmv_matches_reference() {
    check_gbmv_reference::<f32>();
}

#[test]
fn test_issue_14_dgbmv_matches_reference() {
    check_gbmv_reference::<f64>();
}

#[test]
fn test_issue_14_cgbmv_matches_reference() {
    check_gbmv_reference::<Complex32>();
}

#[test]
fn test_issue_14_zgbmv_matches_reference() {
    check_gbmv_reference::<Complex64>();
}

/// The reference y-update prologue: `alpha == 0 && beta == 1` leaves `y`
/// bitwise unchanged, `alpha == 0` gives exactly `beta*y` without reading NaN
/// `A` / `x`, and `beta == 0` overwrites a NaN `y`.
fn check_gbmv_quick_returns<T: BandApi>() {
    let mut rng = Rng::new(0x6b32);
    let (m, n, kl, ku, lda) = (4usize, 3usize, 1usize, 1usize, 3usize);
    let a_nan = vec![nan::<T>(); lda * m.max(n)];
    let x_nan = vec![nan::<T>(); 2 * m.max(n)];
    let a = general_band::<T>(&mut rng, m, n, kl, ku);
    for layout in LAYOUTS {
        let buf = store_gb(layout, &a, n, kl, ku, lda);
        for trans in TRANSES {
            let (lenx, leny) = op_lens(trans, m, n);
            let what = format!("{} gbmv {layout:?} {trans:?}", name::<T>());
            for incy in INCS {
                // alpha = 0, beta = 1: y untouched (non-finite and -0.0 included).
                let len = (leny - 1) * incy.unsigned_abs() as usize + 1;
                let y0 = poison::<T>(len);
                let mut y = y0.clone();
                // SAFETY: NaN-filled A / x and y are large enough for any stride.
                unsafe {
                    T::gbmv(
                        layout,
                        trans,
                        m as i32,
                        n as i32,
                        kl as i32,
                        ku as i32,
                        T::zero(),
                        a_nan.as_ptr(),
                        lda as i32,
                        x_nan.as_ptr(),
                        2,
                        T::one(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                assert_bits_eq(&y0, &y, &format!("{what} alpha=0 beta=1 incy={incy}"));

                // alpha = 0, beta != 1: exactly beta*y; NaN A and x are never read.
                let yv = rng.vec::<T>(leny);
                let beta = rng.num::<T>();
                let mut y = strided(&yv, incy, T::mk(5.0, 5.0));
                // SAFETY: as above.
                unsafe {
                    T::gbmv(
                        layout,
                        trans,
                        m as i32,
                        n as i32,
                        kl as i32,
                        ku as i32,
                        T::zero(),
                        a_nan.as_ptr(),
                        lda as i32,
                        x_nan.as_ptr(),
                        -1,
                        beta,
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                let want: Vec<T> = yv.iter().map(|&v| beta * v).collect();
                assert_bits_eq(
                    &want,
                    &unstrided(&y, leny, incy),
                    &format!("{what} alpha=0"),
                );

                // alpha = 0, beta = 0: zero fill without reading a NaN y.
                let mut y = strided(&vec![nan::<T>(); leny], incy, nan());
                // SAFETY: as above.
                unsafe {
                    T::gbmv(
                        layout,
                        trans,
                        m as i32,
                        n as i32,
                        kl as i32,
                        ku as i32,
                        T::zero(),
                        a_nan.as_ptr(),
                        lda as i32,
                        x_nan.as_ptr(),
                        1,
                        T::zero(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                let zeros = vec![zero::<T>(); leny];
                assert_bits_eq(&zeros, &unstrided(&y, leny, incy), &format!("{what} a=b=0"));
            }

            // beta = 0, alpha != 0: a NaN y is overwritten by alpha*op(A)*x.
            let x = rng.vec::<T>(lenx);
            let alpha = rng.num::<T>();
            let mut y = vec![nan::<T>(); leny];
            // SAFETY: band storage, x and y match the arguments.
            unsafe {
                T::gbmv(
                    layout,
                    trans,
                    m as i32,
                    n as i32,
                    kl as i32,
                    ku as i32,
                    alpha,
                    buf.as_ptr(),
                    lda as i32,
                    x.as_ptr(),
                    1,
                    T::zero(),
                    y.as_mut_ptr(),
                    1,
                );
            }
            let zeros = vec![zero::<T>(); leny];
            let (want, bound) = mv_ref(&a, m, n, trans, alpha, &x, T::zero(), &zeros);
            let got = widen_all(&y);
            assert!(got.iter().all(|v| v.is_finite()), "{what} beta=0: {got:?}");
            assert_close(&got, &want, &bound, T::EPS, &format!("{what} beta=0"));
        }
    }
}

#[test]
fn test_issue_14_sgbmv_quick_returns() {
    check_gbmv_quick_returns::<f32>();
}

#[test]
fn test_issue_14_dgbmv_quick_returns() {
    check_gbmv_quick_returns::<f64>();
}

#[test]
fn test_issue_14_cgbmv_quick_returns() {
    check_gbmv_quick_returns::<Complex32>();
}

#[test]
fn test_issue_14_zgbmv_quick_returns() {
    check_gbmv_quick_returns::<Complex64>();
}

/// Invalid shapes, band counts, leading dimensions and increments, empty
/// matrices and null pointers are all silent no-ops.
fn check_gbmv_rejects<T: BandApi>() {
    let a = [T::one(); 64];
    let x = [T::one(); 16];
    let y0 = poison::<T>(16);
    // (m, n, kl, ku, lda, incx, incy); the valid baseline is (4, 3, 1, 1, 3, 1, 1).
    let cases: [(i32, i32, i32, i32, i32, i32, i32); 12] = [
        (-1, 3, 1, 1, 3, 1, 1),
        (4, -1, 1, 1, 3, 1, 1),
        (4, 3, -1, 1, 3, 1, 1),
        (4, 3, 1, -1, 3, 1, 1),
        (4, 3, 1, 1, 2, 1, 1),
        (4, 3, 2, 3, 5, 1, 1),
        (4, 3, 1, 1, -3, 1, 1),
        (4, 3, 1, 1, 3, 0, 1),
        (4, 3, 1, 1, 3, 1, 0),
        (0, 3, 1, 1, 3, 1, 1),
        (4, 0, 1, 1, 3, 1, 1),
        (4, 3, i32::MAX, i32::MAX, i32::MAX, 1, 1),
    ];
    for layout in LAYOUTS {
        for trans in TRANSES {
            for &(m, n, kl, ku, lda, incx, incy) in &cases {
                let mut y = y0.clone();
                // SAFETY: every case is rejected before any pointer is used.
                unsafe {
                    T::gbmv(
                        layout,
                        trans,
                        m,
                        n,
                        kl,
                        ku,
                        T::one(),
                        a.as_ptr(),
                        lda,
                        x.as_ptr(),
                        incx,
                        T::one(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                let what = format!(
                    "{} gbmv {layout:?} {trans:?} m={m} n={n} kl={kl} ku={ku} lda={lda} \
                     incx={incx} incy={incy}",
                    name::<T>()
                );
                assert_bits_eq(&y0, &y, &what);
            }
            // Null A / x: no-op; null y: no crash.
            let mut y = y0.clone();
            // SAFETY: null pointers are rejected before any dereference.
            unsafe {
                T::gbmv(
                    layout,
                    trans,
                    4,
                    3,
                    1,
                    1,
                    T::one(),
                    ptr::null(),
                    3,
                    x.as_ptr(),
                    1,
                    T::one(),
                    y.as_mut_ptr(),
                    1,
                );
                T::gbmv(
                    layout,
                    trans,
                    4,
                    3,
                    1,
                    1,
                    T::one(),
                    a.as_ptr(),
                    3,
                    ptr::null(),
                    1,
                    T::one(),
                    y.as_mut_ptr(),
                    1,
                );
                T::gbmv(
                    layout,
                    trans,
                    4,
                    3,
                    1,
                    1,
                    T::one(),
                    a.as_ptr(),
                    3,
                    x.as_ptr(),
                    1,
                    T::one(),
                    ptr::null_mut(),
                    1,
                );
            }
            assert_bits_eq(&y0, &y, "gbmv null pointers");
        }
    }
}

#[test]
fn test_issue_14_sgbmv_rejects_invalid_arguments() {
    check_gbmv_rejects::<f32>();
}

#[test]
fn test_issue_14_dgbmv_rejects_invalid_arguments() {
    check_gbmv_rejects::<f64>();
}

#[test]
fn test_issue_14_cgbmv_rejects_invalid_arguments() {
    check_gbmv_rejects::<Complex32>();
}

#[test]
fn test_issue_14_zgbmv_rejects_invalid_arguments() {
    check_gbmv_rejects::<Complex64>();
}

/// A null complex `alpha` / `beta` is a no-op (the real precisions pass the
/// scalars by value).
#[test]
fn test_issue_14_cz_gbmv_hbmv_null_scalars_are_noops() {
    let one32 = Complex32::new(1.0, 0.0);
    let one64 = Complex64::new(1.0, 0.0);
    let a32 = [one32; 9];
    let a64 = [one64; 9];
    let mut y32 = poison::<Complex32>(3);
    let mut y64 = poison::<Complex64>(3);
    let (y32_0, y64_0) = (y32.clone(), y64.clone());
    for layout in LAYOUTS {
        // SAFETY: the null scalar pointers are rejected before any dereference.
        unsafe {
            cblas_cgbmv(
                layout,
                CblasTranspose::NoTrans,
                3,
                3,
                1,
                1,
                ptr::null(),
                a32.as_ptr(),
                3,
                a32.as_ptr(),
                1,
                &one32,
                y32.as_mut_ptr(),
                1,
            );
            cblas_cgbmv(
                layout,
                CblasTranspose::NoTrans,
                3,
                3,
                1,
                1,
                &one32,
                a32.as_ptr(),
                3,
                a32.as_ptr(),
                1,
                ptr::null(),
                y32.as_mut_ptr(),
                1,
            );
            cblas_zgbmv(
                layout,
                CblasTranspose::ConjTrans,
                3,
                3,
                1,
                1,
                ptr::null(),
                a64.as_ptr(),
                3,
                a64.as_ptr(),
                1,
                &one64,
                y64.as_mut_ptr(),
                1,
            );
            cblas_zgbmv(
                layout,
                CblasTranspose::ConjTrans,
                3,
                3,
                1,
                1,
                &one64,
                a64.as_ptr(),
                3,
                a64.as_ptr(),
                1,
                ptr::null(),
                y64.as_mut_ptr(),
                1,
            );
            cblas_chbmv(
                layout,
                CblasUplo::Upper,
                3,
                1,
                ptr::null(),
                a32.as_ptr(),
                2,
                a32.as_ptr(),
                1,
                &one32,
                y32.as_mut_ptr(),
                1,
            );
            cblas_zhbmv(
                layout,
                CblasUplo::Lower,
                3,
                1,
                &one64,
                a64.as_ptr(),
                2,
                a64.as_ptr(),
                1,
                ptr::null(),
                y64.as_mut_ptr(),
                1,
            );
        }
    }
    assert_bits_eq(&y32_0, &y32, "c null scalars");
    assert_bits_eq(&y64_0, &y64, "z null scalars");
}

// =============================================================================
// SBMV / HBMV
// =============================================================================

fn check_hbmv_reference<T: BandApi>() {
    let mut rng = Rng::new(0x5b31);
    for &(n, k) in &[(1usize, 0usize), (4, 1), (6, 2), (5, 7)] {
        let a = hermitian_band::<T>(&mut rng, n, k);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for incx in INCS {
                    for incy in INCS {
                        let lda = k + 1 + usize::from(incy < 0);
                        // Imaginary garbage on the stored diagonal must be ignored.
                        let buf = store_band(layout, uplo, false, &a, k, lda, Some(0.75));
                        let x = rng.vec::<T>(n);
                        let y = rng.vec::<T>(n);
                        let (alpha, beta) = (rng.num::<T>(), rng.num::<T>());
                        let xb = strided(&x, incx, nan());
                        let y0 = strided(&y, incy, T::mk(77.0, -77.0));
                        let mut yb = y0.clone();
                        // SAFETY: every buffer is sized for these exact arguments.
                        unsafe {
                            T::hbmv(
                                layout,
                                uplo,
                                n as i32,
                                k as i32,
                                alpha,
                                buf.as_ptr(),
                                lda as i32,
                                xb.as_ptr(),
                                incx,
                                beta,
                                yb.as_mut_ptr(),
                                incy,
                            );
                        }
                        let what = format!(
                            "{} hbmv {layout:?} {uplo:?} n={n} k={k} lda={lda} incx={incx} \
                             incy={incy}",
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
fn test_issue_14_ssbmv_matches_reference() {
    check_hbmv_reference::<f32>();
}

#[test]
fn test_issue_14_dsbmv_matches_reference() {
    check_hbmv_reference::<f64>();
}

#[test]
fn test_issue_14_chbmv_matches_reference() {
    check_hbmv_reference::<Complex32>();
}

#[test]
fn test_issue_14_zhbmv_matches_reference() {
    check_hbmv_reference::<Complex64>();
}

fn check_hbmv_quick_returns_and_rejects<T: BandApi>() {
    let mut rng = Rng::new(0x5b32);
    let (n, k, lda) = (4usize, 2usize, 3usize);
    let a_nan = vec![nan::<T>(); lda * n];
    let x_nan = vec![nan::<T>(); 3 * n];
    let a = hermitian_band::<T>(&mut rng, n, k);
    for layout in LAYOUTS {
        for uplo in UPLOS {
            let what = format!("{} hbmv {layout:?} {uplo:?}", name::<T>());
            for incy in INCS {
                let len = (n - 1) * incy.unsigned_abs() as usize + 1;
                // alpha = 0, beta = 1: y untouched.
                let y0 = poison::<T>(len);
                let mut y = y0.clone();
                // SAFETY: NaN-filled A / x and y are large enough for any stride.
                unsafe {
                    T::hbmv(
                        layout,
                        uplo,
                        n as i32,
                        k as i32,
                        T::zero(),
                        a_nan.as_ptr(),
                        lda as i32,
                        x_nan.as_ptr(),
                        -3,
                        T::one(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                assert_bits_eq(&y0, &y, &format!("{what} alpha=0 beta=1"));
                // alpha = 0: exactly beta*y; NaN A and x are never read.
                let yv = rng.vec::<T>(n);
                let beta = rng.num::<T>();
                let mut y = strided(&yv, incy, T::mk(5.0, 5.0));
                // SAFETY: as above.
                unsafe {
                    T::hbmv(
                        layout,
                        uplo,
                        n as i32,
                        k as i32,
                        T::zero(),
                        a_nan.as_ptr(),
                        lda as i32,
                        x_nan.as_ptr(),
                        2,
                        beta,
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                let want: Vec<T> = yv.iter().map(|&v| beta * v).collect();
                assert_bits_eq(&want, &unstrided(&y, n, incy), &format!("{what} alpha=0"));
            }
            // beta = 0 overwrites a NaN y.
            let buf = store_band(layout, uplo, false, &a, k, lda, Some(-2.0));
            let x = rng.vec::<T>(n);
            let alpha = rng.num::<T>();
            let mut y = vec![nan::<T>(); n];
            // SAFETY: band storage, x and y match the arguments.
            unsafe {
                T::hbmv(
                    layout,
                    uplo,
                    n as i32,
                    k as i32,
                    alpha,
                    buf.as_ptr(),
                    lda as i32,
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

            // Invalid arguments, n == 0 and null pointers: no-op.
            let ones = [T::one(); 32];
            let y0 = poison::<T>(8);
            // (n, k, lda, incx, incy); the valid baseline is (4, 2, 3, 1, 1).
            let cases: [(i32, i32, i32, i32, i32); 7] = [
                (-1, 2, 3, 1, 1),
                (0, 2, 3, 1, 1),
                (4, -1, 3, 1, 1),
                (4, 2, 2, 1, 1),
                (4, 2, -3, 1, 1),
                (4, 2, 3, 0, 1),
                (4, 2, 3, 1, 0),
            ];
            for &(n, k, lda, incx, incy) in &cases {
                let mut y = y0.clone();
                // SAFETY: every case is rejected before any pointer is used.
                unsafe {
                    T::hbmv(
                        layout,
                        uplo,
                        n,
                        k,
                        T::one(),
                        ones.as_ptr(),
                        lda,
                        ones.as_ptr(),
                        incx,
                        T::one(),
                        y.as_mut_ptr(),
                        incy,
                    );
                }
                let case = format!("{what} n={n} k={k} lda={lda} incx={incx} incy={incy}");
                assert_bits_eq(&y0, &y, &case);
            }
            let mut y = y0.clone();
            // SAFETY: null pointers are rejected before any dereference.
            unsafe {
                T::hbmv(
                    layout,
                    uplo,
                    4,
                    2,
                    T::one(),
                    ptr::null(),
                    3,
                    ones.as_ptr(),
                    1,
                    T::one(),
                    y.as_mut_ptr(),
                    1,
                );
                T::hbmv(
                    layout,
                    uplo,
                    4,
                    2,
                    T::one(),
                    ones.as_ptr(),
                    3,
                    ptr::null(),
                    1,
                    T::one(),
                    y.as_mut_ptr(),
                    1,
                );
                T::hbmv(
                    layout,
                    uplo,
                    4,
                    2,
                    T::one(),
                    ones.as_ptr(),
                    3,
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
fn test_issue_14_ssbmv_quick_returns_and_rejects() {
    check_hbmv_quick_returns_and_rejects::<f32>();
}

#[test]
fn test_issue_14_dsbmv_quick_returns_and_rejects() {
    check_hbmv_quick_returns_and_rejects::<f64>();
}

#[test]
fn test_issue_14_chbmv_quick_returns_and_rejects() {
    check_hbmv_quick_returns_and_rejects::<Complex32>();
}

#[test]
fn test_issue_14_zhbmv_quick_returns_and_rejects() {
    check_hbmv_quick_returns_and_rejects::<Complex64>();
}

// =============================================================================
// TBMV / TBSV
// =============================================================================

/// `x := op(A)*x` (TBMV) or the solve of `op(A)*x = b` (TBSV) for every
/// layout / uplo / trans / diag / increment. Unit diagonals and unused band
/// slots hold NaN.
fn check_tb_reference<T: BandApi>(solve: bool) {
    let mut rng = Rng::new(if solve { 0x7b32 } else { 0x7b31 });
    for &(n, k) in &[(1usize, 0usize), (4, 1), (7, 3), (3, 5)] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for diag in DIAGS {
                        let a: Dense<T> = tri_band(&mut rng, n, k, uplo, diag);
                        for inc in INCS {
                            let lda = k + 1 + usize::from(inc > 1);
                            let unit = diag == CblasDiag::Unit;
                            let buf = store_band(layout, uplo, unit, &a, k, lda, None);
                            let x = rng.vec::<T>(n);
                            let x0 = strided(&x, inc, T::mk(55.0, -55.0));
                            let mut xb = x0.clone();
                            // SAFETY: every buffer is sized for these exact arguments.
                            unsafe {
                                let f = if solve { T::tbsv } else { T::tbmv };
                                f(
                                    layout,
                                    uplo,
                                    trans,
                                    diag,
                                    n as i32,
                                    k as i32,
                                    buf.as_ptr(),
                                    lda as i32,
                                    xb.as_mut_ptr(),
                                    inc,
                                );
                            }
                            let what = format!(
                                "{} {} {layout:?} {uplo:?} {trans:?} {diag:?} n={n} k={k} \
                                 lda={lda} inc={inc}",
                                name::<T>(),
                                if solve { "tbsv" } else { "tbmv" }
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
fn test_issue_14_stbmv_matches_reference() {
    check_tb_reference::<f32>(false);
}

#[test]
fn test_issue_14_dtbmv_matches_reference() {
    check_tb_reference::<f64>(false);
}

#[test]
fn test_issue_14_ctbmv_matches_reference() {
    check_tb_reference::<Complex32>(false);
}

#[test]
fn test_issue_14_ztbmv_matches_reference() {
    check_tb_reference::<Complex64>(false);
}

#[test]
fn test_issue_14_stbsv_solves() {
    check_tb_reference::<f32>(true);
}

#[test]
fn test_issue_14_dtbsv_solves() {
    check_tb_reference::<f64>(true);
}

#[test]
fn test_issue_14_ctbsv_solves() {
    check_tb_reference::<Complex32>(true);
}

#[test]
fn test_issue_14_ztbsv_solves() {
    check_tb_reference::<Complex64>(true);
}

/// `n == 0`, invalid `n` / `k` / `lda` / `incx` and null pointers are no-ops.
fn check_tb_rejects<T: BandApi>() {
    let ones = [T::one(); 32];
    let x0 = poison::<T>(8);
    // (n, k, lda, incx); the valid baseline is (4, 1, 2, 1).
    let cases: [(i32, i32, i32, i32); 7] = [
        (0, 1, 2, 1),
        (-1, 1, 2, 1),
        (4, -1, 2, 1),
        (4, 1, 1, 1),
        (4, 1, -2, 1),
        (4, 1, 2, 0),
        (4, i32::MAX, i32::MAX, 1),
    ];
    for f in [T::tbmv as unsafe fn(_, _, _, _, _, _, _, _, _, _), T::tbsv] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for &(n, k, lda, incx) in &cases {
                        let mut x = x0.clone();
                        // SAFETY: every case is rejected before any pointer is used.
                        unsafe {
                            f(
                                layout,
                                uplo,
                                trans,
                                CblasDiag::NonUnit,
                                n,
                                k,
                                ones.as_ptr(),
                                lda,
                                x.as_mut_ptr(),
                                incx,
                            );
                        }
                        let what = format!(
                            "{} tb {layout:?} {uplo:?} {trans:?} n={n} k={k} lda={lda} incx={incx}",
                            name::<T>()
                        );
                        assert_bits_eq(&x0, &x, &what);
                    }
                    // Null A: no-op; null x: no crash.
                    let mut x = x0.clone();
                    // SAFETY: null pointers are rejected before any dereference.
                    unsafe {
                        f(
                            layout,
                            uplo,
                            trans,
                            CblasDiag::Unit,
                            4,
                            1,
                            ptr::null(),
                            2,
                            x.as_mut_ptr(),
                            1,
                        );
                        f(
                            layout,
                            uplo,
                            trans,
                            CblasDiag::Unit,
                            4,
                            1,
                            ones.as_ptr(),
                            2,
                            ptr::null_mut(),
                            1,
                        );
                    }
                    assert_bits_eq(&x0, &x, "tb null pointers");
                }
            }
        }
    }
}

#[test]
fn test_issue_14_stbmv_stbsv_reject_invalid_arguments() {
    check_tb_rejects::<f32>();
}

#[test]
fn test_issue_14_dtbmv_dtbsv_reject_invalid_arguments() {
    check_tb_rejects::<f64>();
}

#[test]
fn test_issue_14_ctbmv_ctbsv_reject_invalid_arguments() {
    check_tb_rejects::<Complex32>();
}

#[test]
fn test_issue_14_ztbmv_ztbsv_reject_invalid_arguments() {
    check_tb_rejects::<Complex64>();
}

/// A hand-checked row-major `ConjTrans` example: the band buffer is written
/// out literally from the CBLAS row-major definition.
#[test]
fn test_issue_14_ztbmv_rowmajor_conjtrans_hand_example() {
    let c = Complex64::new;
    // A = [[1+i, 2-i], [0, 3+2i]] (upper, k = 1). Row-major upper band storage
    // (lda = 2): row 0 = [A00, A01], row 1 = [A11, *].
    let a = [
        c(1.0, 1.0),
        c(2.0, -1.0),
        c(3.0, 2.0),
        c(f64::NAN, f64::NAN),
    ];
    let mut x = [c(1.0, 0.0), c(0.0, 1.0)];
    // SAFETY: 2x2 band with lda = 2 and a unit-stride 2-vector.
    unsafe {
        cblas_ztbmv(
            CblasLayout::RowMajor,
            CblasUplo::Upper,
            CblasTranspose::ConjTrans,
            CblasDiag::NonUnit,
            2,
            1,
            a.as_ptr(),
            2,
            x.as_mut_ptr(),
            1,
        );
    }
    // A^H = [[1-i, 0], [2+i, 3-2i]]; A^H * [1, i] = [1-i, (2+i) + (3-2i)i]
    //     = [1-i, 4+4i].
    assert_eq!(x, [c(1.0, -1.0), c(4.0, 4.0)]);
}
