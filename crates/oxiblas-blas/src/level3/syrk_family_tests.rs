//! API-level tests of SYRK, SYR2K and GEMMT (naive and blocked paths) and of
//! their CBLAS wrappers, against a reference computed from the definition.
//!
//! Every case plants NaN wherever the routine must not look: in the triangle
//! of C when `beta == 0`, in A and B when `alpha == 0`, and (with Inf and
//! signed zeros) in the other triangle of C and in the padding of strided
//! storage, both of which must stay bitwise untouched. The data are exact
//! dyadic values (see `tri_gemm_tests.rs`), so the reference check is
//! insensitive to the summation order.

use crate::cblas::*;
use crate::level3::gemmt::gemmt;
use crate::level3::syr2k::syr2k;
use crate::level3::syrk::syrk;
use crate::level3::tri_gemm::tests::{
    Elem, RefSeg, check_tri, in_tri, initial_c, mat, nan_mat, scalar, scalar_pairs, val,
};
use crate::level3::trsm::{Trans, Uplo};
use num_complex::{Complex32, Complex64};
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// The routine under test and its transpose arguments.
#[derive(Clone, Copy, Debug)]
enum Op {
    Syrk(Trans),
    Syr2k(Trans),
    Gemmt(Trans, Trans),
}

/// Stored shape of the left (`a`) and right (`b`) operands for `op` with C of
/// order `n` and inner dimension `k`.
fn shapes(op: Op, n: usize, k: usize) -> ((usize, usize), (usize, usize)) {
    let tall = |t: Trans| if t == Trans::NoTrans { (n, k) } else { (k, n) };
    match op {
        Op::Syrk(t) | Op::Syr2k(t) => (tall(t), tall(t)),
        Op::Gemmt(ta, tb) => {
            let b = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
            (tall(ta), b)
        }
    }
}

/// Reference k-segments of `op` (SYRK / SYR2K treat `ConjTrans` as the plain
/// transpose; GEMMT conjugates for complex types).
fn ref_segments<'a, T: Elem>(op: Op, a: MatRef<'a, T>, b: MatRef<'a, T>) -> Vec<RefSeg<'a, T>> {
    let plain = |t: Trans| {
        if t == Trans::NoTrans {
            Trans::NoTrans
        } else {
            Trans::Trans
        }
    };
    match op {
        Op::Syrk(t) => match plain(t) {
            Trans::NoTrans => vec![(a, Trans::NoTrans, a, Trans::Trans)],
            _ => vec![(a, Trans::Trans, a, Trans::NoTrans)],
        },
        Op::Syr2k(t) => match plain(t) {
            Trans::NoTrans => vec![
                (a, Trans::NoTrans, b, Trans::Trans),
                (b, Trans::NoTrans, a, Trans::Trans),
            ],
            _ => vec![
                (a, Trans::Trans, b, Trans::NoTrans),
                (b, Trans::Trans, a, Trans::NoTrans),
            ],
        },
        Op::Gemmt(ta, tb) => vec![(a, ta, b, tb)],
    }
}

/// Calls the native routine; `true` when it accepted the arguments.
fn call<T: Elem>(
    op: Op,
    uplo: Uplo,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) -> bool {
    match op {
        Op::Syrk(t) => syrk(uplo, t, alpha, a, beta, c).is_ok(),
        Op::Syr2k(t) => syr2k(uplo, t, alpha, a, b, beta, c).is_ok(),
        Op::Gemmt(ta, tb) => gemmt(uplo, ta, tb, alpha, a, b, beta, c).is_ok(),
    }
}

/// One native case with contiguous storage.
fn native_case<T: Elem>(op: Op, uplo: Uplo, n: usize, k: usize, alpha: f64, beta: f64) {
    let ((ar, ac), (br, bc)) = shapes(op, n, k);
    let (a, b) = if alpha == 0.0 {
        (nan_mat::<T>(ar, ac), nan_mat::<T>(br, bc))
    } else {
        (mat::<T>(1, ar, ac), mat::<T>(2, br, bc))
    };
    let (alpha_t, beta_t) = (scalar::<T>(alpha), scalar::<T>(beta));
    let before = initial_c::<T>(uplo, n, beta == 0.0);
    let mut c = before.clone();
    let what = format!(
        "{} {op:?} {uplo:?} n={n} k={k} alpha={alpha} beta={beta}",
        core::any::type_name::<T>()
    );
    assert!(
        call(
            op,
            uplo,
            alpha_t,
            a.as_ref(),
            b.as_ref(),
            beta_t,
            c.as_mut()
        ),
        "{what}: rejected"
    );
    let segs = ref_segments(op, a.as_ref(), b.as_ref());
    check_tri(
        &what,
        uplo,
        alpha_t,
        beta_t,
        &segs,
        &before.as_ref(),
        &c.as_ref(),
    );
}

/// `(n, k)` pairs: naive and blocked paths, `n` off every kernel's MR / NR,
/// `k` below and above the blocked threshold and across KC blocks, and an
/// empty inner dimension (`k = 0`, see [`syrk_family_k_zero_scales_triangle`]).
const SIZES: &[(usize, usize)] = &[
    (1, 0),
    (48, 0),
    (256, 0),
    (1, 1),
    (1, 6),
    (3, 2),
    (6, 7),
    (8, 8),
    (13, 3),
    (17, 1),
    (31, 9),
    (32, 8),
    (33, 7),
    (40, 24),
    (47, 5),
    (50, 50),
    (65, 13),
    (97, 70),
    (130, 33),
];

fn trans_list<T: Elem>(allow_conj: bool) -> Vec<Trans> {
    let mut v = vec![Trans::NoTrans, Trans::Trans];
    if allow_conj {
        v.push(Trans::ConjTrans);
    }
    v
}

fn sweep_syrk<T: Elem>() {
    let pairs = scalar_pairs();
    let mut idx = 0usize;
    for &(n, k) in SIZES {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            // Complex SYRK rejects ConjTrans (tested separately below).
            for t in trans_list::<T>(!T::COMPLEX) {
                let (alpha, beta) = pairs[idx % pairs.len()];
                idx += 1;
                native_case::<T>(Op::Syrk(t), uplo, n, k, alpha, beta);
            }
        }
    }
}

fn sweep_syr2k<T: Elem>() {
    let pairs = scalar_pairs();
    let mut idx = 5usize;
    for &(n, k) in SIZES {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for t in trans_list::<T>(true) {
                let (alpha, beta) = pairs[idx % pairs.len()];
                idx += 1;
                native_case::<T>(Op::Syr2k(t), uplo, n, k, alpha, beta);
            }
        }
    }
}

fn sweep_gemmt<T: Elem>() {
    let pairs = scalar_pairs();
    let mut idx = 11usize;
    for &(n, k) in SIZES {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for ta in trans_list::<T>(true) {
                for tb in trans_list::<T>(true) {
                    let (alpha, beta) = pairs[idx % pairs.len()];
                    idx += 1;
                    native_case::<T>(Op::Gemmt(ta, tb), uplo, n, k, alpha, beta);
                }
            }
        }
    }
}

#[test]
fn syrk_matches_reference_all_types() {
    sweep_syrk::<f64>();
    sweep_syrk::<f32>();
    sweep_syrk::<Complex64>();
    sweep_syrk::<Complex32>();
}

#[test]
fn syr2k_matches_reference_all_types() {
    sweep_syr2k::<f64>();
    sweep_syr2k::<f32>();
    sweep_syr2k::<Complex64>();
    sweep_syr2k::<Complex32>();
}

#[test]
fn gemmt_matches_reference_all_types() {
    sweep_gemmt::<f64>();
    sweep_gemmt::<f32>();
    sweep_gemmt::<Complex64>();
    sweep_gemmt::<Complex32>();
}

/// Every `(alpha, beta)` pair on one naive-path and one blocked-path size.
#[test]
fn syrk_family_full_scalar_cross() {
    for (alpha, beta) in scalar_pairs() {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for &(n, k) in &[(7usize, 5usize), (45, 29)] {
                native_case::<f64>(Op::Syrk(Trans::NoTrans), uplo, n, k, alpha, beta);
                native_case::<f64>(Op::Syr2k(Trans::Trans), uplo, n, k, alpha, beta);
                native_case::<f32>(
                    Op::Gemmt(Trans::Trans, Trans::NoTrans),
                    uplo,
                    n,
                    k,
                    alpha,
                    beta,
                );
                native_case::<Complex64>(Op::Syrk(Trans::Trans), uplo, n, k, alpha, beta);
                native_case::<Complex32>(Op::Syr2k(Trans::NoTrans), uplo, n, k, alpha, beta);
                native_case::<Complex64>(
                    Op::Gemmt(Trans::ConjTrans, Trans::Trans),
                    uplo,
                    n,
                    k,
                    alpha,
                    beta,
                );
            }
        }
    }
}

/// Larger orders (several MC blocks for f64 at n = 300) on the blocked path.
#[test]
fn syrk_family_large_orders() {
    native_case::<f64>(Op::Syrk(Trans::NoTrans), Uplo::Lower, 300, 40, 2.5, -1.0);
    native_case::<f64>(Op::Syrk(Trans::Trans), Uplo::Upper, 257, 19, -1.0, 2.5);
    native_case::<f64>(Op::Syr2k(Trans::NoTrans), Uplo::Upper, 263, 23, 1.0, 1.0);
    native_case::<f32>(Op::Syr2k(Trans::Trans), Uplo::Lower, 290, 11, 2.5, 0.0);
    native_case::<f64>(
        Op::Gemmt(Trans::NoTrans, Trans::NoTrans),
        Uplo::Lower,
        271,
        31,
        -1.0,
        0.0,
    );
    native_case::<Complex64>(Op::Syrk(Trans::NoTrans), Uplo::Upper, 140, 300, 2.5, 1.0);
    // Complex `Trans` with a long k (the complex blocked region).
    native_case::<Complex64>(Op::Syrk(Trans::Trans), Uplo::Lower, 41, 133, -1.0, 2.5);
    native_case::<Complex32>(Op::Syr2k(Trans::Trans), Uplo::Upper, 35, 131, 2.5, 0.0);
    native_case::<Complex32>(
        Op::Gemmt(Trans::Trans, Trans::ConjTrans),
        Uplo::Lower,
        150,
        45,
        1.0,
        -1.0,
    );
}

/// Every routine and transpose combination of [`Op`] that the type accepts.
fn all_ops<T: Elem>() -> Vec<Op> {
    let mut ops = Vec::new();
    for t in trans_list::<T>(!T::COMPLEX) {
        ops.push(Op::Syrk(t));
    }
    for t in trans_list::<T>(true) {
        ops.push(Op::Syr2k(t));
    }
    for ta in trans_list::<T>(true) {
        for tb in trans_list::<T>(true) {
            ops.push(Op::Gemmt(ta, tb));
        }
    }
    ops
}

/// `k = 0` for one type: the product is empty whatever alpha is, so the
/// `uplo` triangle of C must become exactly `beta·C` (`beta = 0` writes zeros
/// without reading the NaN planted there, `beta = 1` leaves C bitwise
/// untouched) and nothing outside it may change.
///
/// Other `beta` are compared by value: like reference `?SYR2K` with
/// `TRANS = 'T'`, the dot-product loops form `beta·c + alpha·0`, which turns a
/// `beta·c` of `-0` into `+0`, while the column-update loops and the blocked
/// kernel only scale.
fn k_zero_sweep<T: Elem>() {
    for &n in &[256usize, 48, 24, 16, 5, 1] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let filled = initial_c::<T>(uplo, n, false);
            let nan_tri = initial_c::<T>(uplo, n, true);
            for op in all_ops::<T>() {
                let ((ar, ac), (br, bc)) = shapes(op, n, 0);
                let (a, b) = (mat::<T>(1, ar, ac), mat::<T>(2, br, bc));
                for (alpha, beta) in scalar_pairs() {
                    let before = if beta == 0.0 { &nan_tri } else { &filled };
                    let (alpha_t, beta_t) = (scalar::<T>(alpha), scalar::<T>(beta));
                    let mut c = before.clone();
                    let what = format!(
                        "k=0 {} {op:?} {uplo:?} n={n} alpha={alpha} beta={beta}",
                        core::any::type_name::<T>()
                    );
                    let accepted = call(
                        op,
                        uplo,
                        alpha_t,
                        a.as_ref(),
                        b.as_ref(),
                        beta_t,
                        c.as_mut(),
                    );
                    assert!(accepted, "{what}: rejected");
                    for j in 0..n {
                        for i in 0..n {
                            let (old, got) = (before[(i, j)], c[(i, j)]);
                            if !in_tri(uplo, i, j) || beta == 1.0 {
                                assert_eq!(got.bits(), old.bits(), "{what}: C[{i},{j}] changed");
                            } else {
                                let want = if beta == 0.0 { T::zero() } else { beta_t * old };
                                assert!(
                                    got == want,
                                    "{what}: C[{i},{j}] = {got:?}, expected {want:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// An empty inner dimension on every path. With `k = 0` and `alpha != 0`
/// (`alpha = 0` always takes the reference loops) the blocked triangle-only
/// kernel is still selected for real SYRK `NoTrans` with `beta = 0` from
/// `n = 256`, real SYRK `Trans` from `n = 24` (`beta = 0`) / `48`, real SYR2K
/// `Trans` with `beta = 0` from `n = 48`, real GEMMT from `n = 16` and
/// complex GEMMT with `beta = 0` from `n = 32`; the other cases take the
/// reference loops. Every `(alpha, beta)` pair over `{0, 1, -1, 2.5}`
/// (complex scalars for the complex types).
#[test]
fn syrk_family_k_zero_scales_triangle() {
    k_zero_sweep::<f64>();
    k_zero_sweep::<f32>();
    k_zero_sweep::<Complex64>();
    k_zero_sweep::<Complex32>();
}

/// Complex SYRK rejects `ConjTrans` and leaves C bitwise untouched.
#[test]
fn complex_syrk_conj_trans_rejected() {
    let a = mat::<Complex64>(1, 40, 40);
    let before = initial_c::<Complex64>(Uplo::Lower, 40, false);
    let mut c = before.clone();
    let one = Complex64::new(1.0, 0.0);
    assert!(
        syrk(
            Uplo::Lower,
            Trans::ConjTrans,
            one,
            a.as_ref(),
            one,
            c.as_mut()
        )
        .is_err()
    );
    for j in 0..40 {
        for i in 0..40 {
            assert_eq!(c[(i, j)].bits(), before[(i, j)].bits());
        }
    }
}

/// Transposed / strided operand views and a submatrix C with NaN padding.
fn views_case<T: Elem>(op: Op, uplo: Uplo, n: usize, k: usize) {
    let ((ar, ac), (br, bc)) = shapes(op, n, k);
    // A stored transposed and viewed through `transpose()` (col_stride != 1).
    let a_store = mat::<T>(3, ac, ar);
    let a = a_store.as_ref().transpose();
    // B as a strided submatrix of a larger matrix.
    let b_store = mat::<T>(4, br + 4, bc + 3);
    let b = b_store.as_ref().submatrix(3, 2, br, bc);
    // C as a submatrix of a NaN-padded matrix.
    let (pr, pc) = (2usize, 3usize);
    let mut big = nan_mat::<T>(n + 5, n + 6);
    let init = initial_c::<T>(uplo, n, false);
    for j in 0..n {
        for i in 0..n {
            big[(pr + i, pc + j)] = init[(i, j)];
        }
    }
    let big_before = big.clone();
    let (alpha, beta) = (scalar::<T>(-1.0), scalar::<T>(2.5));
    let what = format!(
        "views {} {op:?} {uplo:?} n={n} k={k}",
        core::any::type_name::<T>()
    );
    let accepted = {
        let c = big.as_mut().submatrix(pr, pc, n, n);
        call(op, uplo, alpha, a, b, beta, c)
    };
    assert!(accepted, "{what}: rejected");
    let segs = ref_segments(op, a, b);
    check_tri(
        &what,
        uplo,
        alpha,
        beta,
        &segs,
        &big_before.as_ref().submatrix(pr, pc, n, n),
        &big.as_ref().submatrix(pr, pc, n, n),
    );
    for j in 0..n + 6 {
        for i in 0..n + 5 {
            if !((pr..pr + n).contains(&i) && (pc..pc + n).contains(&j)) {
                assert_eq!(
                    big[(i, j)].bits(),
                    big_before[(i, j)].bits(),
                    "{what}: padding C[{i},{j}] changed"
                );
            }
        }
    }
}

fn views_sweep<T: Elem>() {
    for &(n, k) in &[(5usize, 3usize), (37, 21), (70, 9)] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for t in [Trans::NoTrans, Trans::Trans] {
                views_case::<T>(Op::Syrk(t), uplo, n, k);
                views_case::<T>(Op::Syr2k(t), uplo, n, k);
                views_case::<T>(Op::Gemmt(t, Trans::ConjTrans), uplo, n, k);
                views_case::<T>(Op::Gemmt(Trans::NoTrans, t), uplo, n, k);
            }
        }
    }
}

#[test]
fn syrk_family_strided_and_transposed_views() {
    views_sweep::<f64>();
    views_sweep::<f32>();
    views_sweep::<Complex64>();
    views_sweep::<Complex32>();
}

// =============================================================================
// CBLAS wrappers, both layouts
// =============================================================================

/// Per-type access to the CBLAS SYRK / SYR2K / GEMMTR wrappers.
trait CblasRank: Elem {
    #[allow(clippy::too_many_arguments)]
    unsafe fn syrk(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn syr2k(
        layout: CblasLayout,
        uplo: CblasUplo,
        trans: CblasTranspose,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemmtr(
        layout: CblasLayout,
        uplo: CblasUplo,
        ta: CblasTranspose,
        tb: CblasTranspose,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
}

macro_rules! real_cblas {
    ($t:ty, $syrk:ident, $syr2k:ident, $gemmtr:ident) => {
        impl CblasRank for $t {
            unsafe fn syrk(
                layout: CblasLayout,
                uplo: CblasUplo,
                trans: CblasTranspose,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                // SAFETY: forwarded caller contract.
                unsafe { $syrk(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc) }
            }
            unsafe fn syr2k(
                layout: CblasLayout,
                uplo: CblasUplo,
                trans: CblasTranspose,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                // SAFETY: forwarded caller contract.
                unsafe {
                    $syr2k(
                        layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                    )
                }
            }
            unsafe fn gemmtr(
                layout: CblasLayout,
                uplo: CblasUplo,
                ta: CblasTranspose,
                tb: CblasTranspose,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                // SAFETY: forwarded caller contract.
                unsafe {
                    $gemmtr(
                        layout, uplo, ta, tb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                    )
                }
            }
        }
    };
}

macro_rules! complex_cblas {
    ($t:ty, $syrk:ident, $syr2k:ident, $gemmtr:ident) => {
        impl CblasRank for $t {
            unsafe fn syrk(
                layout: CblasLayout,
                uplo: CblasUplo,
                trans: CblasTranspose,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                // SAFETY: forwarded caller contract; the scalars outlive the call.
                unsafe { $syrk(layout, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc) }
            }
            unsafe fn syr2k(
                layout: CblasLayout,
                uplo: CblasUplo,
                trans: CblasTranspose,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                // SAFETY: forwarded caller contract; the scalars outlive the call.
                unsafe {
                    $syr2k(
                        layout, uplo, trans, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
                    )
                }
            }
            unsafe fn gemmtr(
                layout: CblasLayout,
                uplo: CblasUplo,
                ta: CblasTranspose,
                tb: CblasTranspose,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                // SAFETY: forwarded caller contract; the scalars outlive the call.
                unsafe {
                    $gemmtr(
                        layout, uplo, ta, tb, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
                    )
                }
            }
        }
    };
}

real_cblas!(f64, cblas_dsyrk, cblas_dsyr2k, cblas_dgemmtr);
real_cblas!(f32, cblas_ssyrk, cblas_ssyr2k, cblas_sgemmtr);
complex_cblas!(Complex64, cblas_zsyrk, cblas_zsyr2k, cblas_zgemmtr);
complex_cblas!(Complex32, cblas_csyrk, cblas_csyr2k, cblas_cgemmtr);

/// Storage index of logical `(i, j)`.
fn slot(layout: CblasLayout, i: usize, j: usize, ld: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => i + j * ld,
        CblasLayout::RowMajor => i * ld + j,
    }
}

/// An `r × c` matrix in `layout` with leading dimension `ld`; padding is NaN.
fn store<T: Elem>(
    layout: CblasLayout,
    r: usize,
    c: usize,
    ld: usize,
    f: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let len = match layout {
        CblasLayout::ColMajor => ld * c.max(1),
        CblasLayout::RowMajor => ld * r.max(1),
    };
    let mut v = vec![T::nan(); len];
    for j in 0..c {
        for i in 0..r {
            v[slot(layout, i, j, ld)] = f(i, j);
        }
    }
    v
}

/// The logical `r × c` matrix held in `v`.
fn logical<T: Elem>(layout: CblasLayout, v: &[T], r: usize, c: usize, ld: usize) -> Mat<T> {
    let mut m = Mat::<T>::zeros(r, c);
    for j in 0..c {
        for i in 0..r {
            m[(i, j)] = v[slot(layout, i, j, ld)];
        }
    }
    m
}

fn native_trans(t: CblasTranspose) -> Trans {
    match t {
        CblasTranspose::NoTrans => Trans::NoTrans,
        CblasTranspose::Trans => Trans::Trans,
        CblasTranspose::ConjTrans => Trans::ConjTrans,
    }
}

/// One CBLAS case: the logical result must match the reference on the
/// triangle; everything else in C's storage (other triangle, padding) must
/// stay bitwise untouched.
#[allow(clippy::too_many_arguments)]
fn cblas_case<T: CblasRank>(
    layout: CblasLayout,
    uplo: CblasUplo,
    op: Op,
    ct: (CblasTranspose, CblasTranspose),
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
) {
    let ((ar, ac), (br, bc)) = shapes(op, n, k);
    let ld_of = |r: usize, c: usize| {
        2 + match layout {
            CblasLayout::ColMajor => r,
            CblasLayout::RowMajor => c,
        }
    };
    let (lda, ldb, ldc) = (ld_of(ar, ac), ld_of(br, bc), n + 3);
    let a = store::<T>(layout, ar, ac, lda, |i, j| T::from_c64(val(21, i, j)));
    let b = store::<T>(layout, br, bc, ldb, |i, j| T::from_c64(val(22, i, j)));
    let native_uplo = match uplo {
        CblasUplo::Upper => Uplo::Upper,
        CblasUplo::Lower => Uplo::Lower,
    };
    let init = initial_c::<T>(native_uplo, n, beta == 0.0);
    let mut c = store::<T>(layout, n, n, ldc, |i, j| init[(i, j)]);
    let c_before = c.clone();
    let (alpha_t, beta_t) = (scalar::<T>(alpha), scalar::<T>(beta));
    let (n32, k32) = (n as i32, k as i32);
    // SAFETY: A / B / C hold their stored shapes with the given leading
    // dimensions (`store` sizes them), and C does not overlap A or B.
    unsafe {
        match op {
            Op::Syrk(_) => T::syrk(
                layout,
                uplo,
                ct.0,
                n32,
                k32,
                alpha_t,
                a.as_ptr(),
                lda as i32,
                beta_t,
                c.as_mut_ptr(),
                ldc as i32,
            ),
            Op::Syr2k(_) => T::syr2k(
                layout,
                uplo,
                ct.0,
                n32,
                k32,
                alpha_t,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta_t,
                c.as_mut_ptr(),
                ldc as i32,
            ),
            Op::Gemmt(..) => T::gemmtr(
                layout,
                uplo,
                ct.0,
                ct.1,
                n32,
                k32,
                alpha_t,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta_t,
                c.as_mut_ptr(),
                ldc as i32,
            ),
        }
    }
    let what = format!(
        "cblas {} {layout:?} {uplo:?} {op:?} n={n} k={k} alpha={alpha} beta={beta}",
        core::any::type_name::<T>()
    );
    let a_m = logical(layout, &a, ar, ac, lda);
    let b_m = logical(layout, &b, br, bc, ldb);
    let segs = ref_segments(op, a_m.as_ref(), b_m.as_ref());
    let got = logical(layout, &c, n, n, ldc);
    let before = logical(layout, &c_before, n, n, ldc);
    check_tri(
        &what,
        native_uplo,
        alpha_t,
        beta_t,
        &segs,
        &before.as_ref(),
        &got.as_ref(),
    );
    // Padding of C's storage.
    let mut is_elem = vec![false; c.len()];
    for j in 0..n {
        for i in 0..n {
            is_elem[slot(layout, i, j, ldc)] = true;
        }
    }
    for (idx, (x, y)) in c.iter().zip(c_before.iter()).enumerate() {
        if !is_elem[idx] {
            assert_eq!(x.bits(), y.bits(), "{what}: C padding slot {idx} changed");
        }
    }
    // The other triangle in storage order (covered by `check_tri` too, but
    // spelled out against the layout mapping).
    for j in 0..n {
        for i in 0..n {
            if !in_tri(native_uplo, i, j) {
                let s = slot(layout, i, j, ldc);
                assert_eq!(
                    c[s].bits(),
                    c_before[s].bits(),
                    "{what}: C[{i},{j}] changed"
                );
            }
        }
    }
}

fn cblas_sweep<T: CblasRank>() {
    let cblas_t = |t: Trans| match t {
        Trans::NoTrans => CblasTranspose::NoTrans,
        Trans::Trans => CblasTranspose::Trans,
        Trans::ConjTrans => CblasTranspose::ConjTrans,
    };
    let mut idx = 3usize;
    let pairs = scalar_pairs();
    for layout in [CblasLayout::ColMajor, CblasLayout::RowMajor] {
        for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
            for &(n, k) in &[(5usize, 3usize), (45, 37), (48, 0)] {
                for t in [Trans::NoTrans, Trans::Trans] {
                    // The CBLAS layout mapping turns a row-major NoTrans into a
                    // column-major Trans of the same buffer, so the logical
                    // operation is always op(t).
                    let (alpha, beta) = pairs[idx % pairs.len()];
                    idx += 1;
                    let ct = (cblas_t(t), CblasTranspose::NoTrans);
                    cblas_case::<T>(layout, uplo, Op::Syrk(t), ct, n, k, alpha, beta);
                    cblas_case::<T>(layout, uplo, Op::Syr2k(t), ct, n, k, 2.5, -1.0);
                }
                for ta in [Trans::NoTrans, Trans::Trans, Trans::ConjTrans] {
                    for tb in [Trans::NoTrans, Trans::Trans, Trans::ConjTrans] {
                        let (alpha, beta) = pairs[idx % pairs.len()];
                        idx += 1;
                        let op = Op::Gemmt(native_trans(cblas_t(ta)), native_trans(cblas_t(tb)));
                        cblas_case::<T>(
                            layout,
                            uplo,
                            op,
                            (cblas_t(ta), cblas_t(tb)),
                            n,
                            k,
                            alpha,
                            beta,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cblas_syrk_syr2k_gemmtr_both_layouts_real() {
    cblas_sweep::<f64>();
    cblas_sweep::<f32>();
}

#[test]
fn cblas_syrk_syr2k_gemmtr_both_layouts_complex() {
    cblas_sweep::<Complex64>();
    cblas_sweep::<Complex32>();
}
