//! Issue #14 (Level-1 + Level-3 part): every reference-CBLAS entry point that
//! was missing from `oxiblas_blas::cblas` is reachable through the public
//! `oxiblas_blas::cblas::*` path and computes the reference result.
//!
//! Each of the 30 symbols is called at least once (in both storage layouts
//! where the routine takes one) and compared with a naive reference written
//! straight from the mathematical definition. The exhaustive sweeps (every
//! uplo / trans / diag / side, increments, quick returns, NaN propagation)
//! live in the crate's unit tests next to the wrappers.

use num_complex::{Complex32, Complex64};
use oxiblas_blas::cblas::*;

// =============================================================================
// Shared helpers
// =============================================================================

/// Element types the Level-3 checks run on, with a widening to `Complex64`
/// for the reference arithmetic (real types use the real part only).
trait Elem: Copy + std::fmt::Debug {
    /// Relative tolerance for comparisons against the f64 reference.
    const TOL: f64;
    fn from_c64(z: Complex64) -> Self;
    fn to_c64(self) -> Complex64;
}

impl Elem for f32 {
    const TOL: f64 = 1e-4;
    fn from_c64(z: Complex64) -> Self {
        z.re as f32
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl Elem for f64 {
    const TOL: f64 = 1e-12;
    fn from_c64(z: Complex64) -> Self {
        z.re
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Elem for Complex32 {
    const TOL: f64 = 1e-4;
    fn from_c64(z: Complex64) -> Self {
        Complex32::new(z.re as f32, z.im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl Elem for Complex64 {
    const TOL: f64 = 1e-12;
    fn from_c64(z: Complex64) -> Self {
        z
    }
    fn to_c64(self) -> Complex64 {
        self
    }
}

/// Storage slot of logical element `(i, j)`.
fn slot(layout: CblasLayout, i: usize, j: usize, ld: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => i + j * ld,
        CblasLayout::RowMajor => i * ld + j,
    }
}

/// Stores a logical `rows x cols` matrix; unused leading-dimension slots get `pad`.
fn store<T: Elem>(
    layout: CblasLayout,
    rows: usize,
    cols: usize,
    ld: usize,
    pad: T,
    value: impl Fn(usize, usize) -> Complex64,
) -> Vec<T> {
    let len = match layout {
        CblasLayout::ColMajor => ld * cols,
        CblasLayout::RowMajor => ld * rows,
    };
    let mut buf = vec![pad; len.max(1)];
    for j in 0..cols {
        for i in 0..rows {
            buf[slot(layout, i, j, ld)] = T::from_c64(value(i, j));
        }
    }
    buf
}

/// Deterministic, non-symmetric test values (`im` is ignored by real types).
fn val(i: usize, j: usize, seed: usize) -> Complex64 {
    Complex64::new(
        ((7 * i + 3 * j + seed) % 11) as f64 * 0.25 - 1.25,
        ((5 * i + 2 * j + 3 * seed) % 7) as f64 * 0.25 - 0.75,
    )
}

/// Leading dimension of a logical `rows x cols` operand (padded by 2).
fn ld_for(layout: CblasLayout, rows: usize, cols: usize) -> usize {
    2 + match layout {
        CblasLayout::ColMajor => rows,
        CblasLayout::RowMajor => cols,
    }
}

fn assert_close<T: Elem>(what: &str, got: T, want: Complex64) {
    let got = got.to_c64();
    let err = (got - want).norm();
    assert!(
        err <= T::TOL * (1.0 + want.norm()),
        "{what}: got {got}, expected {want}"
    );
}

fn in_triangle(uplo: CblasUplo, i: usize, j: usize) -> bool {
    match uplo {
        CblasUplo::Upper => i <= j,
        CblasUplo::Lower => i >= j,
    }
}

/// `op(S)(i, j)` for a stored matrix `S` given as a closure.
fn op_elem(
    trans: CblasTranspose,
    s: &dyn Fn(usize, usize) -> Complex64,
    i: usize,
    j: usize,
) -> Complex64 {
    match trans {
        CblasTranspose::NoTrans => s(i, j),
        CblasTranspose::Trans => s(j, i),
        CblasTranspose::ConjTrans => s(j, i).conj(),
    }
}

const LAYOUTS: [CblasLayout; 2] = [CblasLayout::ColMajor, CblasLayout::RowMajor];

// =============================================================================
// Level 1
// =============================================================================

#[test]
fn test_issue_14_srotg_drotg_generate_reference_rotation() {
    let (mut a, mut b, mut c, mut s) = (3.0f64, 4.0f64, 0.0f64, 0.0f64);
    // SAFETY: four valid, distinct scalars.
    unsafe { cblas_drotg(&mut a, &mut b, &mut c, &mut s) };
    assert!((a - 5.0).abs() < 1e-15, "r = {a}");
    assert!((c - 0.6).abs() < 1e-15 && (s - 0.8).abs() < 1e-15);
    // |a| <= |b| and c != 0: z = 1 / c.
    assert!((b - 1.0 / 0.6).abs() < 1e-14, "z = {b}");

    let (mut a, mut b, mut c, mut s) = (-4.0f32, 3.0f32, 0.0f32, 0.0f32);
    // SAFETY: four valid, distinct scalars.
    unsafe { cblas_srotg(&mut a, &mut b, &mut c, &mut s) };
    // |a| > |b|: r carries the sign of a, z = s.
    assert!((a + 5.0).abs() < 1e-6 && (c - 0.8).abs() < 1e-6 && (s + 0.6).abs() < 1e-6);
    assert!((b - s).abs() < 1e-7);
}

#[test]
fn test_issue_14_srotmg_drotmg_write_reference_parameters() {
    // Unscaled flag = 1 case: |d1 x1^2| <= |d2 y1^2|.
    let (mut d1, mut d2, mut x1) = (1.0f64, 2.0f64, 1.0f64);
    let mut p = [7.0f64; 5];
    // SAFETY: three valid scalars and a 5-element DPARAM array.
    unsafe { cblas_drotmg(&mut d1, &mut d2, &mut x1, 2.0, p.as_mut_ptr()) };
    // p1 = 1, p2 = 4, q1 = 1, q2 = 8 -> flag 1, h11 = 1/4, h22 = 1/2, u = 9/8.
    assert_eq!(p, [1.0, 0.25, 7.0, 7.0, 0.5]);
    assert_eq!((d1, d2, x1), (16.0 / 9.0, 8.0 / 9.0, 2.25));

    // y1 == 0 -> flag = -2 (identity) and only P[0] is written.
    let (mut d1, mut d2, mut x1) = (1.0f32, 1.0f32, 3.0f32);
    let mut p = [7.0f32; 5];
    // SAFETY: three valid scalars and a 5-element SPARAM array.
    unsafe { cblas_srotmg(&mut d1, &mut d2, &mut x1, 0.0, p.as_mut_ptr()) };
    assert_eq!(p, [-2.0, 7.0, 7.0, 7.0, 7.0]);
    assert_eq!((d1, d2, x1), (1.0, 1.0, 3.0));
}

#[test]
fn test_issue_14_srot_drot_apply_rotation_with_strides() {
    // Unit strides.
    let mut x = [1.0f64, 2.0, 3.0];
    let mut y = [4.0f64, 5.0, 6.0];
    // SAFETY: both vectors hold 3 elements.
    unsafe { cblas_drot(3, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, 0.6, 0.8) };
    for i in 0..3 {
        let (xi, yi) = ([1.0, 2.0, 3.0][i], [4.0, 5.0, 6.0][i]);
        assert!((x[i] - (0.6 * xi + 0.8 * yi)).abs() < 1e-15);
        assert!((y[i] - (0.6 * yi - 0.8 * xi)).abs() < 1e-15);
    }

    // incx = 2, incy = -1: logical y_i lives at offset (n - 1 - i).
    let mut x = [1.0f32, 9.0, 2.0];
    let mut y = [20.0f32, 10.0];
    // SAFETY: x spans (2-1)*2+1 = 3 elements, y spans 2.
    unsafe { cblas_srot(2, x.as_mut_ptr(), 2, y.as_mut_ptr(), -1, 0.0, 1.0) };
    // c = 0, s = 1: x_i' = y_i, y_i' = -x_i with (x0, x1) = (1, 2), (y0, y1) = (10, 20).
    assert_eq!(x, [10.0, 9.0, 20.0]);
    assert_eq!(y, [-2.0, -1.0]);
}

#[test]
fn test_issue_14_srotm_drotm_apply_modified_rotation() {
    // flag = -1: full H with DPARAM order [flag, h11, h21, h12, h22].
    let p = [-1.0f64, 2.0, -1.0, 0.5, 3.0];
    let mut x = [1.0f64, -2.0];
    let mut y = [4.0f64, 0.5];
    // SAFETY: two 2-element vectors and a 5-element DPARAM.
    unsafe { cblas_drotm(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, p.as_ptr()) };
    assert_eq!(x, [4.0, -3.75]);
    assert_eq!(y, [11.0, 3.5]);

    // flag = 0 (unit diagonal) with negative increments; P[1], P[4] unused (NaN).
    let p = [0.0f32, f32::NAN, 3.0, 2.0, f32::NAN];
    let mut x = [1.0f32, 2.0];
    let mut y = [1.0f32, 1.0];
    // SAFETY: two 2-element vectors and a 5-element SPARAM.
    unsafe { cblas_srotm(2, x.as_mut_ptr(), -1, y.as_mut_ptr(), -1, p.as_ptr()) };
    // x' = x + 2 y, y' = 3 x + y (elementwise; both walk backwards together).
    assert_eq!(x, [3.0, 4.0]);
    assert_eq!(y, [4.0, 7.0]);
}

#[test]
fn test_issue_14_crotg_zrotg_follow_reference_conventions() {
    // a = 0, b = 3 + 4i -> c = 0, r = |b| = 5, s = conj(b) / |b| = 0.6 - 0.8i.
    let mut a = Complex64::new(0.0, 0.0);
    let b = Complex64::new(3.0, 4.0);
    let (mut c, mut s) = (7.0f64, Complex64::new(7.0, 7.0));
    // SAFETY: valid scalars.
    unsafe { cblas_zrotg(&mut a, &b, &mut c, &mut s) };
    assert_eq!(c, 0.0);
    assert!((a - Complex64::new(5.0, 0.0)).norm() < 1e-15);
    assert!((s - Complex64::new(0.6, -0.8)).norm() < 1e-15);

    // b = 0 -> c = 1, s = 0, r = a.
    let mut a = Complex32::new(1.5, -2.0);
    let b = Complex32::new(0.0, 0.0);
    let (mut c, mut s) = (7.0f32, Complex32::new(7.0, 7.0));
    // SAFETY: valid scalars.
    unsafe { cblas_crotg(&mut a, &b, &mut c, &mut s) };
    assert_eq!(
        (a, c, s),
        (Complex32::new(1.5, -2.0), 1.0, Complex32::new(0.0, 0.0))
    );
}

#[test]
fn test_issue_14_csrot_zdrot_apply_real_rotation_to_complex_vectors() {
    let x0 = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)];
    let y0 = [Complex64::new(0.5, -1.0), Complex64::new(2.0, 4.0)];
    let (mut x, mut y) = (x0, y0);
    // SAFETY: both vectors hold 2 elements.
    unsafe { cblas_zdrot(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, 0.6, 0.8) };
    for i in 0..2 {
        assert!((x[i] - (x0[i] * 0.6 + y0[i] * 0.8)).norm() < 1e-15);
        assert!((y[i] - (y0[i] * 0.6 - x0[i] * 0.8)).norm() < 1e-15);
    }

    // incx = 0: the reference loop rotates x[0] against every y element in turn.
    let mut x = [Complex32::new(1.0, 1.0)];
    let mut y = [Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)];
    // SAFETY: x spans 1 element (inc 0), y spans 2.
    unsafe { cblas_csrot(2, x.as_mut_ptr(), 0, y.as_mut_ptr(), 1, 0.0, 1.0) };
    // Step 1: x = y0 = 1, y0 = -(1+i). Step 2: x = y1 = i, y1 = -1.
    assert_eq!(x, [Complex32::new(0.0, 1.0)]);
    assert_eq!(y, [Complex32::new(-1.0, -1.0), Complex32::new(-1.0, 0.0)]);
}

#[test]
fn test_issue_14_sdsdot_dsdot_accumulate_in_double() {
    // 2^24 + 1 - 2^24 is exact in f64 but 0 in an f32 accumulation.
    let x = [16_777_216.0f32, 1.0, -16_777_216.0];
    let y = [1.0f32, 1.0, 1.0];
    // SAFETY: both vectors hold 3 elements.
    let d = unsafe { cblas_dsdot(3, x.as_ptr(), 1, y.as_ptr(), 1) };
    assert_eq!(d, 1.0);
    // SAFETY: as above.
    let s = unsafe { cblas_sdsdot(3, 0.5, x.as_ptr(), 1, y.as_ptr(), 1) };
    assert_eq!(s, 1.5);
    // n <= 0 returns alpha / 0; negative increments walk backwards.
    // SAFETY: n = 0 reads nothing.
    unsafe {
        assert_eq!(cblas_sdsdot(0, 2.5, x.as_ptr(), 1, y.as_ptr(), 1), 2.5);
        assert_eq!(cblas_dsdot(0, x.as_ptr(), 1, y.as_ptr(), 1), 0.0);
    }
    let a = [1.0f32, 2.0, 3.0];
    let b = [10.0f32, 20.0];
    // SAFETY: a spans 3 elements (n = 2, inc = -2), b spans 2.
    let d = unsafe { cblas_dsdot(2, a.as_ptr(), -2, b.as_ptr(), 1) };
    // Logical a = (a[2], a[0]) = (3, 1): 3*10 + 1*20 = 50.
    assert_eq!(d, 50.0);
}

#[test]
fn test_issue_14_scabs1_dcabs1() {
    let z = Complex32::new(3.0, -4.0);
    let w = Complex64::new(-1.5, 2.0);
    // SAFETY: valid scalars; null is documented to return 0.
    unsafe {
        assert_eq!(cblas_scabs1(&z), 7.0);
        assert_eq!(cblas_dcabs1(&w), 3.5);
        assert_eq!(cblas_scabs1(std::ptr::null()), 0.0);
        assert_eq!(cblas_dcabs1(std::ptr::null()), 0.0);
    }
}

// =============================================================================
// Level 3 — GEMMTR
// =============================================================================

type GemmtrCall<T> = fn(
    CblasLayout,
    CblasUplo,
    CblasTranspose,
    CblasTranspose,
    usize,
    usize,
    T,
    &[T],
    usize,
    &[T],
    usize,
    T,
    &mut [T],
    usize,
);

fn check_gemmtr<T: Elem>(call: GemmtrCall<T>, transes: &[CblasTranspose]) {
    let (n, k) = (4usize, 3usize);
    let (alpha, beta) = (Complex64::new(1.5, -0.5), Complex64::new(0.5, 0.25));
    for layout in LAYOUTS {
        for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
            for &ta in transes {
                for &tb in transes {
                    let (ar, ac) = if ta == CblasTranspose::NoTrans {
                        (n, k)
                    } else {
                        (k, n)
                    };
                    let (br, bc) = if tb == CblasTranspose::NoTrans {
                        (k, n)
                    } else {
                        (n, k)
                    };
                    let (lda, ldb, ldc) = (
                        ld_for(layout, ar, ac),
                        ld_for(layout, br, bc),
                        ld_for(layout, n, n),
                    );
                    let sa = |i: usize, j: usize| val(i, j, 1);
                    let sb = |i: usize, j: usize| val(i, j, 2);
                    let c0 = |i: usize, j: usize| val(i, j, 3);
                    let a = store::<T>(layout, ar, ac, lda, T::from_c64(c0(9, 9)), sa);
                    let b = store::<T>(layout, br, bc, ldb, T::from_c64(c0(9, 9)), sb);
                    let mut c = store::<T>(layout, n, n, ldc, T::from_c64(c0(9, 9)), c0);
                    call(
                        layout,
                        uplo,
                        ta,
                        tb,
                        n,
                        k,
                        T::from_c64(alpha),
                        &a,
                        lda,
                        &b,
                        ldb,
                        T::from_c64(beta),
                        &mut c,
                        ldc,
                    );
                    // Real types see only the real parts of alpha/beta/values.
                    let re = |z: Complex64| T::from_c64(z).to_c64();
                    for j in 0..n {
                        for i in 0..n {
                            let want = if in_triangle(uplo, i, j) {
                                let sum: Complex64 = (0..k)
                                    .map(|l| {
                                        op_elem(ta, &|r, s| re(sa(r, s)), i, l)
                                            * op_elem(tb, &|r, s| re(sb(r, s)), l, j)
                                    })
                                    .sum();
                                re(alpha) * sum + re(beta) * re(c0(i, j))
                            } else {
                                re(c0(i, j))
                            };
                            assert_close(
                                &format!("gemmtr {layout:?} {uplo:?} {ta:?} {tb:?} C[{i},{j}]"),
                                c[slot(layout, i, j, ldc)],
                                want,
                            );
                        }
                    }
                }
            }
        }
    }
}

const ALL_TRANS: [CblasTranspose; 3] = [
    CblasTranspose::NoTrans,
    CblasTranspose::Trans,
    CblasTranspose::ConjTrans,
];

#[test]
fn test_issue_14_sgemmtr_dgemmtr_both_layouts() {
    check_gemmtr::<f32>(
        |l, u, ta, tb, n, k, al, a, lda, b, ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_sgemmtr(
                    l,
                    u,
                    ta,
                    tb,
                    n as i32,
                    k as i32,
                    al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        &ALL_TRANS,
    );
    check_gemmtr::<f64>(
        |l, u, ta, tb, n, k, al, a, lda, b, ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_dgemmtr(
                    l,
                    u,
                    ta,
                    tb,
                    n as i32,
                    k as i32,
                    al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        &ALL_TRANS,
    );
}

#[test]
fn test_issue_14_cgemmtr_zgemmtr_both_layouts() {
    check_gemmtr::<Complex32>(
        |l, u, ta, tb, n, k, al, a, lda, b, ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_cgemmtr(
                    l,
                    u,
                    ta,
                    tb,
                    n as i32,
                    k as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    &be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        &ALL_TRANS,
    );
    check_gemmtr::<Complex64>(
        |l, u, ta, tb, n, k, al, a, lda, b, ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_zgemmtr(
                    l,
                    u,
                    ta,
                    tb,
                    n as i32,
                    k as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    &be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        &ALL_TRANS,
    );
}

// =============================================================================
// Level 3 — complex SYMM / SYRK / SYR2K
// =============================================================================

type SymmCall<T> = fn(
    CblasLayout,
    CblasSide,
    CblasUplo,
    usize,
    usize,
    T,
    &[T],
    usize,
    &[T],
    usize,
    T,
    &mut [T],
    usize,
);

fn check_symm<T: Elem>(call: SymmCall<T>) {
    let (m, n) = (3usize, 4usize);
    let (alpha, beta) = (Complex64::new(0.5, 1.0), Complex64::new(-0.5, 0.25));
    for layout in LAYOUTS {
        for side in [CblasSide::Left, CblasSide::Right] {
            for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
                let ka = if side == CblasSide::Left { m } else { n };
                let (lda, ldb, ldc) = (
                    ld_for(layout, ka, ka),
                    ld_for(layout, m, n),
                    ld_for(layout, m, n) + 1,
                );
                // Complex SYMMETRIC: A(i, j) = A(j, i), no conjugation.
                let sym = |i: usize, j: usize| val(i.min(j), i.max(j), 4);
                let junk = Complex64::new(100.0, -100.0);
                let a = store::<T>(layout, ka, ka, lda, T::from_c64(junk), |i, j| {
                    if in_triangle(uplo, i, j) {
                        sym(i, j)
                    } else {
                        junk
                    }
                });
                let b = store::<T>(layout, m, n, ldb, T::from_c64(junk), |i, j| val(i, j, 5));
                let mut c = store::<T>(layout, m, n, ldc, T::from_c64(junk), |i, j| val(i, j, 6));
                call(
                    layout,
                    side,
                    uplo,
                    m,
                    n,
                    T::from_c64(alpha),
                    &a,
                    lda,
                    &b,
                    ldb,
                    T::from_c64(beta),
                    &mut c,
                    ldc,
                );
                for j in 0..n {
                    for i in 0..m {
                        let prod: Complex64 = match side {
                            CblasSide::Left => (0..m).map(|l| sym(i, l) * val(l, j, 5)).sum(),
                            CblasSide::Right => (0..n).map(|l| val(i, l, 5) * sym(l, j)).sum(),
                        };
                        assert_close(
                            &format!("symm {layout:?} {side:?} {uplo:?} C[{i},{j}]"),
                            c[slot(layout, i, j, ldc)],
                            alpha * prod + beta * val(i, j, 6),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_csymm_zsymm_both_layouts() {
    check_symm::<Complex32>(|l, sd, u, m, n, al, a, lda, b, ldb, be, c, ldc| {
        // SAFETY: buffers are sized by `store` for the declared shapes.
        unsafe {
            cblas_csymm(
                l,
                sd,
                u,
                m as i32,
                n as i32,
                &al,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                &be,
                c.as_mut_ptr(),
                ldc as i32,
            )
        }
    });
    check_symm::<Complex64>(|l, sd, u, m, n, al, a, lda, b, ldb, be, c, ldc| {
        // SAFETY: buffers are sized by `store` for the declared shapes.
        unsafe {
            cblas_zsymm(
                l,
                sd,
                u,
                m as i32,
                n as i32,
                &al,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                &be,
                c.as_mut_ptr(),
                ldc as i32,
            )
        }
    });
}

/// `(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)`; SYRK
/// wrappers ignore `b`/`ldb`.
type Rank2kCall<T> = fn(
    CblasLayout,
    CblasUplo,
    CblasTranspose,
    usize,
    usize,
    T,
    &[T],
    usize,
    &[T],
    usize,
    T,
    &mut [T],
    usize,
);

fn check_rank_k<T: Elem>(call: Rank2kCall<T>, two_operands: bool) {
    let (n, k) = (4usize, 3usize);
    let (alpha, beta) = (Complex64::new(0.75, -0.5), Complex64::new(0.5, 0.5));
    for layout in LAYOUTS {
        for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
            for trans in [CblasTranspose::NoTrans, CblasTranspose::Trans] {
                let (ar, ac) = if trans == CblasTranspose::NoTrans {
                    (n, k)
                } else {
                    (k, n)
                };
                let (lda, ldc) = (ld_for(layout, ar, ac), ld_for(layout, n, n));
                let ldb = lda + 1;
                let sa = |i: usize, j: usize| val(i, j, 7);
                let sb = |i: usize, j: usize| val(i, j, 8);
                let pad = T::from_c64(Complex64::new(9.0, 9.0));
                let a = store::<T>(layout, ar, ac, lda, pad, sa);
                let b = store::<T>(layout, ar, ac, ldb, pad, sb);
                let mut c = store::<T>(layout, n, n, ldc, pad, |i, j| val(i, j, 9));
                call(
                    layout,
                    uplo,
                    trans,
                    n,
                    k,
                    T::from_c64(alpha),
                    &a,
                    lda,
                    &b,
                    ldb,
                    T::from_c64(beta),
                    &mut c,
                    ldc,
                );
                // op(X) is X (NoTrans, n x k) or X^T (Trans); never conjugated.
                let opx = |s: &dyn Fn(usize, usize) -> Complex64, i: usize, l: usize| {
                    op_elem(trans, s, i, l)
                };
                for j in 0..n {
                    for i in 0..n {
                        let want = if in_triangle(uplo, i, j) {
                            let sum: Complex64 = (0..k)
                                .map(|l| {
                                    if two_operands {
                                        opx(&sa, i, l) * opx(&sb, j, l)
                                            + opx(&sb, i, l) * opx(&sa, j, l)
                                    } else {
                                        opx(&sa, i, l) * opx(&sa, j, l)
                                    }
                                })
                                .sum();
                            alpha * sum + beta * val(i, j, 9)
                        } else {
                            val(i, j, 9)
                        };
                        assert_close(
                            &format!(
                                "rank-k (2 ops: {two_operands}) {layout:?} {uplo:?} {trans:?} C[{i},{j}]"
                            ),
                            c[slot(layout, i, j, ldc)],
                            want,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_csyrk_zsyrk_both_layouts() {
    check_rank_k::<Complex32>(
        |l, u, t, n, k, al, a, lda, _b, _ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_csyrk(
                    l,
                    u,
                    t,
                    n as i32,
                    k as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    &be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        false,
    );
    check_rank_k::<Complex64>(
        |l, u, t, n, k, al, a, lda, _b, _ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_zsyrk(
                    l,
                    u,
                    t,
                    n as i32,
                    k as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    &be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        false,
    );
}

#[test]
fn test_issue_14_csyr2k_zsyr2k_both_layouts() {
    check_rank_k::<Complex32>(
        |l, u, t, n, k, al, a, lda, b, ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_csyr2k(
                    l,
                    u,
                    t,
                    n as i32,
                    k as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    &be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        true,
    );
    check_rank_k::<Complex64>(
        |l, u, t, n, k, al, a, lda, b, ldb, be, c, ldc| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_zsyr2k(
                    l,
                    u,
                    t,
                    n as i32,
                    k as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    &be,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
        },
        true,
    );
}

/// Reference CBLAS 3.12 (`cblas_csyrk.c` / `cblas_csyr2k.c`): ColMajor
/// `ConjTrans` is forwarded as `TRANS = 'C'`, which Fortran CSYRK / CSYR2K
/// reject (`xerbla`, C untouched); RowMajor `ConjTrans` is mapped to `'N'`,
/// exactly like RowMajor `Trans`.
#[test]
fn test_issue_14_csyrk_csyr2k_conj_trans_follows_reference_cblas() {
    let a: Vec<Complex32> = (0..16)
        .map(|p| Complex32::new(0.25 * p as f32 - 1.0, 0.5 - 0.125 * p as f32))
        .collect();
    let b: Vec<Complex32> = a.iter().rev().map(|z| z * 0.5).collect();
    let alpha = Complex32::new(1.0, -0.5);
    let beta = Complex32::new(2.0, 0.0);
    let c0 = vec![Complex32::new(-0.0, 3.0); 16];
    let syrk = |layout: CblasLayout, trans: CblasTranspose| {
        let mut c = c0.clone();
        // SAFETY: 4x4 operands with leading dimension 4.
        unsafe {
            cblas_csyrk(
                layout,
                CblasUplo::Upper,
                trans,
                4,
                4,
                &alpha,
                a.as_ptr(),
                4,
                &beta,
                c.as_mut_ptr(),
                4,
            );
        }
        c
    };
    let syr2k = |layout: CblasLayout, trans: CblasTranspose| {
        let mut c = c0.clone();
        // SAFETY: 4x4 operands with leading dimension 4.
        unsafe {
            cblas_csyr2k(
                layout,
                CblasUplo::Lower,
                trans,
                4,
                4,
                &alpha,
                a.as_ptr(),
                4,
                b.as_ptr(),
                4,
                &beta,
                c.as_mut_ptr(),
                4,
            );
        }
        c
    };
    let bits = |c: &[Complex32]| -> Vec<(u32, u32)> {
        c.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
    };
    // ColMajor: no-op, C bitwise unchanged (including the -0.0 real parts).
    assert_eq!(
        bits(&syrk(CblasLayout::ColMajor, CblasTranspose::ConjTrans)),
        bits(&c0)
    );
    assert_eq!(
        bits(&syr2k(CblasLayout::ColMajor, CblasTranspose::ConjTrans)),
        bits(&c0)
    );
    // RowMajor: identical to Trans (and genuinely updates C).
    let t = syrk(CblasLayout::RowMajor, CblasTranspose::Trans);
    assert_ne!(bits(&t), bits(&c0));
    assert_eq!(
        bits(&syrk(CblasLayout::RowMajor, CblasTranspose::ConjTrans)),
        bits(&t)
    );
    let t2 = syr2k(CblasLayout::RowMajor, CblasTranspose::Trans);
    assert_ne!(bits(&t2), bits(&c0));
    assert_eq!(
        bits(&syr2k(CblasLayout::RowMajor, CblasTranspose::ConjTrans)),
        bits(&t2)
    );
    // n = 0 is a no-op even with null operands (z variant).
    // SAFETY: n = 0 dereferences nothing.
    unsafe {
        cblas_zsyr2k(
            CblasLayout::RowMajor,
            CblasUplo::Lower,
            CblasTranspose::ConjTrans,
            0,
            0,
            std::ptr::null(),
            std::ptr::null(),
            1,
            std::ptr::null(),
            1,
            std::ptr::null(),
            std::ptr::null_mut(),
            1,
        );
    }
}

// =============================================================================
// Level 3 — complex TRMM / TRSM
// =============================================================================

type TriCall<T> = fn(
    CblasLayout,
    CblasSide,
    CblasUplo,
    CblasTranspose,
    CblasDiag,
    usize,
    usize,
    T,
    &[T],
    usize,
    &mut [T],
    usize,
);

/// Stored triangle: well-conditioned values in the referenced triangle, junk
/// elsewhere and on a unit diagonal.
fn tri_stored(uplo: CblasUplo, diag: CblasDiag, i: usize, j: usize) -> Complex64 {
    if i == j {
        match diag {
            CblasDiag::Unit => Complex64::new(100.0, 100.0),
            CblasDiag::NonUnit => Complex64::new(2.0 + 0.5 * i as f64, 0.5 - 0.25 * i as f64),
        }
    } else if in_triangle(uplo, i, j) {
        val(i, j, 10) * 0.25
    } else {
        Complex64::new(-100.0, 100.0)
    }
}

/// Dense triangular `T(i, j)` implied by [`tri_stored`].
fn tri_dense(uplo: CblasUplo, diag: CblasDiag, i: usize, j: usize) -> Complex64 {
    if i == j && diag == CblasDiag::Unit {
        Complex64::new(1.0, 0.0)
    } else if in_triangle(uplo, i, j) {
        tri_stored(uplo, diag, i, j)
    } else {
        Complex64::new(0.0, 0.0)
    }
}

fn check_tri<T: Elem>(trmm: TriCall<T>, trsm: TriCall<T>) {
    let (m, n) = (3usize, 4usize);
    let alpha = Complex64::new(0.5, -1.5);
    for layout in LAYOUTS {
        for side in [CblasSide::Left, CblasSide::Right] {
            for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
                for trans in ALL_TRANS {
                    for diag in [CblasDiag::NonUnit, CblasDiag::Unit] {
                        let ka = if side == CblasSide::Left { m } else { n };
                        let (lda, ldb) = (ld_for(layout, ka, ka), ld_for(layout, m, n));
                        let a = store::<T>(layout, ka, ka, lda, T::from_c64(alpha), |i, j| {
                            tri_stored(uplo, diag, i, j)
                        });
                        let opt = |i: usize, j: usize| {
                            op_elem(trans, &|r, s| tri_dense(uplo, diag, r, s), i, j)
                        };
                        let x0 = |i: usize, j: usize| val(i, j, 11);
                        // y = op(T) x0 (Left) or x0 op(T) (Right).
                        let apply = |i: usize, j: usize| -> Complex64 {
                            match side {
                                CblasSide::Left => (0..m).map(|l| opt(i, l) * x0(l, j)).sum(),
                                CblasSide::Right => (0..n).map(|l| x0(i, l) * opt(l, j)).sum(),
                            }
                        };
                        let what = format!("{layout:?} {side:?} {uplo:?} {trans:?} {diag:?}");
                        // TRMM: B := alpha op(T) B.
                        let mut b = store::<T>(layout, m, n, ldb, T::from_c64(alpha), x0);
                        trmm(
                            layout,
                            side,
                            uplo,
                            trans,
                            diag,
                            m,
                            n,
                            T::from_c64(alpha),
                            &a,
                            lda,
                            &mut b,
                            ldb,
                        );
                        for j in 0..n {
                            for i in 0..m {
                                assert_close(
                                    &format!("trmm {what} B[{i},{j}]"),
                                    b[slot(layout, i, j, ldb)],
                                    alpha * apply(i, j),
                                );
                            }
                        }
                        // TRSM: solving op(T) X = alpha B with B = op(T) x0 gives alpha x0.
                        let mut b = store::<T>(layout, m, n, ldb, T::from_c64(alpha), apply);
                        trsm(
                            layout,
                            side,
                            uplo,
                            trans,
                            diag,
                            m,
                            n,
                            T::from_c64(alpha),
                            &a,
                            lda,
                            &mut b,
                            ldb,
                        );
                        for j in 0..n {
                            for i in 0..m {
                                assert_close(
                                    &format!("trsm {what} X[{i},{j}]"),
                                    b[slot(layout, i, j, ldb)],
                                    alpha * x0(i, j),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_ctrmm_ctrsm_both_layouts() {
    check_tri::<Complex32>(
        |l, sd, u, t, d, m, n, al, a, lda, b, ldb| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_ctrmm(
                    l,
                    sd,
                    u,
                    t,
                    d,
                    m as i32,
                    n as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_mut_ptr(),
                    ldb as i32,
                )
            }
        },
        |l, sd, u, t, d, m, n, al, a, lda, b, ldb| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_ctrsm(
                    l,
                    sd,
                    u,
                    t,
                    d,
                    m as i32,
                    n as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_mut_ptr(),
                    ldb as i32,
                )
            }
        },
    );
}

#[test]
fn test_issue_14_ztrmm_ztrsm_both_layouts() {
    check_tri::<Complex64>(
        |l, sd, u, t, d, m, n, al, a, lda, b, ldb| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_ztrmm(
                    l,
                    sd,
                    u,
                    t,
                    d,
                    m as i32,
                    n as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_mut_ptr(),
                    ldb as i32,
                )
            }
        },
        |l, sd, u, t, d, m, n, al, a, lda, b, ldb| {
            // SAFETY: buffers are sized by `store` for the declared shapes.
            unsafe {
                cblas_ztrsm(
                    l,
                    sd,
                    u,
                    t,
                    d,
                    m as i32,
                    n as i32,
                    &al,
                    a.as_ptr(),
                    lda as i32,
                    b.as_mut_ptr(),
                    ldb as i32,
                )
            }
        },
    );
}
