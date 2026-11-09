//! Unit tests for the CBLAS Level-3 wrappers added for issue #14: `?gemmtr`
//! and the complex `symm` / `syrk` / `syr2k` / `trmm` / `trsm`.
//!
//! Every routine is compared against a dense reference evaluated straight from
//! its mathematical definition (in `Complex64`) for both storage layouts and
//! every `uplo` / `trans` / `diag` / `side`, within the usual
//! `(terms + 4)·8·eps·Σ|products|` bound. Unreferenced triangles, unit
//! diagonals and leading-dimension padding hold NaN: they must never be read
//! (a NaN would propagate into the result) nor written (checked bitwise).
//! The large cases reach the natives' GEMM-backed / blocked code paths.

use super::*;
use core::fmt::Debug;

// =============================================================================
// Scalars and dispatch
// =============================================================================

trait TestScalar: Copy + Debug + PartialEq {
    /// Unit roundoff of the element type.
    const EPS: f64;
    fn from_c64(z: Complex64) -> Self;
    fn to_c64(self) -> Complex64;
    fn bits(self) -> (u64, u64);
}

impl TestScalar for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn from_c64(z: Complex64) -> Self {
        z.re as f32
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
}

impl TestScalar for f64 {
    const EPS: f64 = f64::EPSILON;
    fn from_c64(z: Complex64) -> Self {
        z.re
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl TestScalar for Complex32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn from_c64(z: Complex64) -> Self {
        Complex32::new(z.re as f32, z.im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
}

impl TestScalar for Complex64 {
    const EPS: f64 = f64::EPSILON;
    fn from_c64(z: Complex64) -> Self {
        z
    }
    fn to_c64(self) -> Complex64 {
        self
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

fn nan<T: TestScalar>() -> T {
    T::from_c64(Complex64::new(f64::NAN, f64::NAN))
}

/// The value the routine actually sees once `z` is stored as a `T`.
fn eff<T: TestScalar>(z: Complex64) -> Complex64 {
    T::from_c64(z).to_c64()
}

/// `?gemmtr` for every precision.
trait GemmtrScalar: TestScalar {
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

impl GemmtrScalar for f32 {
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
        cblas_sgemmtr(
            layout, uplo, ta, tb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

impl GemmtrScalar for f64 {
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
        cblas_dgemmtr(
            layout, uplo, ta, tb, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

impl GemmtrScalar for Complex32 {
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
        cblas_cgemmtr(
            layout, uplo, ta, tb, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
        );
    }
}

impl GemmtrScalar for Complex64 {
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
        cblas_zgemmtr(
            layout, uplo, ta, tb, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
        );
    }
}

/// The complex-only routines, dispatched per precision.
trait ComplexL3: GemmtrScalar {
    unsafe fn symm(
        layout: CblasLayout,
        side: CblasSide,
        uplo: CblasUplo,
        m: i32,
        n: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
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
    /// TRMM (`solve == false`) or TRSM (`solve == true`).
    unsafe fn tri(
        solve: bool,
        layout: CblasLayout,
        side: CblasSide,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        m: i32,
        n: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *mut Self,
        ldb: i32,
    );
}

impl ComplexL3 for Complex32 {
    unsafe fn symm(
        layout: CblasLayout,
        side: CblasSide,
        uplo: CblasUplo,
        m: i32,
        n: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    ) {
        cblas_csymm(
            layout, side, uplo, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc,
        );
    }
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
        cblas_csyrk(layout, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc);
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
        cblas_csyr2k(
            layout, uplo, trans, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
        );
    }
    unsafe fn tri(
        solve: bool,
        layout: CblasLayout,
        side: CblasSide,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        m: i32,
        n: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *mut Self,
        ldb: i32,
    ) {
        if solve {
            cblas_ctrsm(
                layout, side, uplo, trans, diag, m, n, &alpha, a, lda, b, ldb,
            );
        } else {
            cblas_ctrmm(
                layout, side, uplo, trans, diag, m, n, &alpha, a, lda, b, ldb,
            );
        }
    }
}

impl ComplexL3 for Complex64 {
    unsafe fn symm(
        layout: CblasLayout,
        side: CblasSide,
        uplo: CblasUplo,
        m: i32,
        n: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    ) {
        cblas_zsymm(
            layout, side, uplo, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc,
        );
    }
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
        cblas_zsyrk(layout, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc);
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
        cblas_zsyr2k(
            layout, uplo, trans, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
        );
    }
    unsafe fn tri(
        solve: bool,
        layout: CblasLayout,
        side: CblasSide,
        uplo: CblasUplo,
        trans: CblasTranspose,
        diag: CblasDiag,
        m: i32,
        n: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *mut Self,
        ldb: i32,
    ) {
        if solve {
            cblas_ztrsm(
                layout, side, uplo, trans, diag, m, n, &alpha, a, lda, b, ldb,
            );
        } else {
            cblas_ztrmm(
                layout, side, uplo, trans, diag, m, n, &alpha, a, lda, b, ldb,
            );
        }
    }
}

// =============================================================================
// Storage and comparison helpers
// =============================================================================

const LAYOUTS: [CblasLayout; 2] = [CblasLayout::ColMajor, CblasLayout::RowMajor];
const UPLOS: [CblasUplo; 2] = [CblasUplo::Upper, CblasUplo::Lower];
const SIDES: [CblasSide; 2] = [CblasSide::Left, CblasSide::Right];
const DIAGS: [CblasDiag; 2] = [CblasDiag::NonUnit, CblasDiag::Unit];
const TRANSES: [CblasTranspose; 3] = [
    CblasTranspose::NoTrans,
    CblasTranspose::Trans,
    CblasTranspose::ConjTrans,
];
/// `(alpha, beta)` pairs: general, `beta == 0` (C must not be read), `beta == 1`.
const SCALARS: [((f64, f64), (f64, f64)); 3] = [
    ((1.5, -0.5), (0.5, 0.25)),
    ((-1.0, 0.0), (0.0, 0.0)),
    ((0.75, 1.0), (1.0, 0.0)),
];

fn cz(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Deterministic, non-symmetric test values (exact in `f32`).
fn val(i: usize, j: usize, seed: usize) -> Complex64 {
    cz(
        ((7 * i + 3 * j + seed) % 11) as f64 * 0.25 - 1.25,
        ((5 * i + 2 * j + 3 * seed) % 7) as f64 * 0.25 - 0.75,
    )
}

fn slot(layout: CblasLayout, i: usize, j: usize, ld: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => i + j * ld,
        CblasLayout::RowMajor => i * ld + j,
    }
}

/// Logical position of storage index `idx` (may lie in the padding).
fn position(layout: CblasLayout, idx: usize, ld: usize) -> (usize, usize) {
    match layout {
        CblasLayout::ColMajor => (idx % ld, idx / ld),
        CblasLayout::RowMajor => (idx / ld, idx % ld),
    }
}

/// Minimal leading dimension of a logical `rows x cols` operand plus `extra`.
fn ld_for(layout: CblasLayout, rows: usize, cols: usize, extra: usize) -> usize {
    extra
        + match layout {
            CblasLayout::ColMajor => rows.max(1),
            CblasLayout::RowMajor => cols.max(1),
        }
}

/// Stores a logical `rows x cols` matrix; every padding slot holds NaN.
fn store<T: TestScalar>(
    layout: CblasLayout,
    rows: usize,
    cols: usize,
    ld: usize,
    value: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let len = match layout {
        CblasLayout::ColMajor => ld * cols,
        CblasLayout::RowMajor => ld * rows,
    };
    let mut buf = vec![nan::<T>(); len.max(1)];
    for j in 0..cols {
        for i in 0..rows {
            buf[slot(layout, i, j, ld)] = value(i, j);
        }
    }
    buf
}

fn in_tri(uplo: CblasUplo, i: usize, j: usize) -> bool {
    match uplo {
        CblasUplo::Upper => i <= j,
        CblasUplo::Lower => i >= j,
    }
}

fn op_at(
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

/// Checks every storage slot: a logical `(i, j)` inside `rows x cols` with
/// `expect(i, j) == Some((want, mag))` must match `want` within
/// `(terms + 4)·8·eps·mag`; every other slot (other triangle, padding) must be
/// bitwise unchanged from `before`.
fn check_slots<T: TestScalar>(
    what: &str,
    layout: CblasLayout,
    (rows, cols, ld): (usize, usize, usize),
    terms: usize,
    before: &[T],
    got: &[T],
    expect: impl Fn(usize, usize) -> Option<(Complex64, f64)>,
) {
    for (idx, (old, new)) in before.iter().zip(got).enumerate() {
        let (i, j) = position(layout, idx, ld);
        let want = if i < rows && j < cols {
            expect(i, j)
        } else {
            None
        };
        match want {
            Some((w, mag)) => {
                let g = new.to_c64();
                let tol = (terms as f64 + 4.0) * 8.0 * T::EPS * mag + 1e-300;
                assert!(
                    (g - w).norm() <= tol,
                    "{what}: ({i},{j}) = {g}, expected {w} (tol {tol:e})"
                );
            }
            None => assert_eq!(
                new.bits(),
                old.bits(),
                "{what}: slot ({i},{j}) must be untouched"
            ),
        }
    }
}

fn bits_of<T: TestScalar>(v: &[T]) -> Vec<(u64, u64)> {
    v.iter().map(|x| x.bits()).collect()
}

// =============================================================================
// GEMMTR
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn gemmtr_case<T: GemmtrScalar>(
    layout: CblasLayout,
    uplo: CblasUplo,
    ta: CblasTranspose,
    tb: CblasTranspose,
    n: usize,
    k: usize,
    alpha: Complex64,
    beta: Complex64,
) {
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
        ld_for(layout, ar, ac, 1),
        ld_for(layout, br, bc, 2),
        ld_for(layout, n, n, 3),
    );
    let sa = |i: usize, j: usize| eff::<T>(val(i, j, 1));
    let sb = |i: usize, j: usize| eff::<T>(val(i, j, 2));
    let c0 = |i: usize, j: usize| eff::<T>(val(i, j, 3));
    let (al, be) = (eff::<T>(alpha), eff::<T>(beta));
    let beta_zero = be == cz(0.0, 0.0);
    let a = store::<T>(layout, ar, ac, lda, |i, j| T::from_c64(sa(i, j)));
    let b = store::<T>(layout, br, bc, ldb, |i, j| T::from_c64(sb(i, j)));
    // beta == 0: the referenced triangle holds NaN and must not be read.
    let mut c = store::<T>(layout, n, n, ldc, |i, j| {
        if in_tri(uplo, i, j) && !beta_zero {
            T::from_c64(c0(i, j))
        } else {
            nan()
        }
    });
    let before = c.clone();
    // SAFETY: buffers sized by `store` for the declared shapes / leading dimensions.
    unsafe {
        T::gemmtr(
            layout,
            uplo,
            ta,
            tb,
            n as i32,
            k as i32,
            T::from_c64(alpha),
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            T::from_c64(beta),
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
    let what = format!("gemmtr {layout:?} {uplo:?} {ta:?} {tb:?} n={n} k={k} a={alpha} b={beta}");
    check_slots(&what, layout, (n, n, ldc), k, &before, &c, |i, j| {
        if !in_tri(uplo, i, j) {
            return None;
        }
        let (mut sum, mut mag) = (cz(0.0, 0.0), 0.0);
        for l in 0..k {
            let p = op_at(ta, &sa, i, l) * op_at(tb, &sb, l, j);
            sum += p;
            mag += p.norm();
        }
        let (base, base_mag) = if beta_zero {
            (cz(0.0, 0.0), 0.0)
        } else {
            (be * c0(i, j), be.norm() * c0(i, j).norm())
        };
        Some((al * sum + base, al.norm() * mag + base_mag))
    });
}

fn sweep_gemmtr<T: GemmtrScalar>() {
    for layout in LAYOUTS {
        for uplo in UPLOS {
            for ta in TRANSES {
                for tb in TRANSES {
                    for (n, k) in [(1, 1), (3, 2), (4, 5)] {
                        for ((ar, ai), (br, bi)) in SCALARS {
                            gemmtr_case::<T>(layout, uplo, ta, tb, n, k, cz(ar, ai), cz(br, bi));
                        }
                    }
                }
            }
            // n >= 32 && k >= 8 reaches the native's GEMM-backed path.
            for (ta, tb) in [
                (CblasTranspose::NoTrans, CblasTranspose::NoTrans),
                (CblasTranspose::Trans, CblasTranspose::ConjTrans),
                (CblasTranspose::ConjTrans, CblasTranspose::NoTrans),
                (CblasTranspose::NoTrans, CblasTranspose::Trans),
            ] {
                gemmtr_case::<T>(layout, uplo, ta, tb, 40, 12, cz(0.5, -1.0), cz(-0.5, 0.5));
                gemmtr_case::<T>(layout, uplo, ta, tb, 37, 9, cz(1.0, 0.0), cz(0.0, 0.0));
            }
        }
    }
}

#[test]
fn test_issue_14_sgemmtr_dgemmtr_match_dense_reference() {
    sweep_gemmtr::<f32>();
    sweep_gemmtr::<f64>();
}

#[test]
fn test_issue_14_cgemmtr_zgemmtr_match_dense_reference() {
    sweep_gemmtr::<Complex32>();
    sweep_gemmtr::<Complex64>();
}

/// Quick returns / special values of `?gemmtr` (the `?gemm`-style quick return
/// and the `k == 0` shortcut are declared deviations, see the module docs).
fn gemmtr_special<T: GemmtrScalar>() {
    let (n, ld) = (4usize, 6usize);
    let nan_buf = vec![nan::<T>(); ld * ld];
    let one = T::from_c64(cz(1.0, 0.0));
    let zero = T::from_c64(cz(0.0, 0.0));
    let half = T::from_c64(cz(0.5, 0.0));
    for layout in LAYOUTS {
        for uplo in UPLOS {
            let c0 = |i: usize, j: usize| eff::<T>(val(i, j, 5));
            let fresh = || {
                store::<T>(layout, n, n, ld, |i, j| {
                    if in_tri(uplo, i, j) {
                        T::from_c64(c0(i, j))
                    } else {
                        nan()
                    }
                })
            };
            let call = |k: i32, alpha: T, beta: T, lda: i32, c: &mut [T]| {
                // SAFETY: A / B are NaN buffers of ld*ld elements (ld >= n, k);
                // C is n x n with leading dimension ld.
                unsafe {
                    T::gemmtr(
                        layout,
                        uplo,
                        CblasTranspose::NoTrans,
                        CblasTranspose::Trans,
                        n as i32,
                        k,
                        alpha,
                        nan_buf.as_ptr(),
                        lda,
                        nan_buf.as_ptr(),
                        ld as i32,
                        beta,
                        c.as_mut_ptr(),
                        ld as i32,
                    );
                }
            };
            let what = format!("gemmtr special {layout:?} {uplo:?}");
            // alpha == 0 (NaN A / B never read): triangle := beta·C.
            for k in [0, 3] {
                let mut c = fresh();
                let before = c.clone();
                call(k, zero, half, ld as i32, &mut c);
                check_slots(&what, layout, (n, n, ld), 1, &before, &c, |i, j| {
                    in_tri(uplo, i, j).then(|| (c0(i, j) * 0.5, c0(i, j).norm()))
                });
            }
            // k == 0 with alpha != 0 behaves the same (A / B never read).
            let mut c = fresh();
            let before = c.clone();
            call(0, one, half, ld as i32, &mut c);
            check_slots(&what, layout, (n, n, ld), 1, &before, &c, |i, j| {
                in_tri(uplo, i, j).then(|| (c0(i, j) * 0.5, c0(i, j).norm()))
            });
            // beta == 0 with alpha == 0 / k == 0: zero-fill without reading C.
            for (k, alpha) in [(3, zero), (0, one)] {
                let mut c = vec![nan::<T>(); ld * ld];
                let before = c.clone();
                call(k, alpha, zero, ld as i32, &mut c);
                check_slots(&what, layout, (n, n, ld), 1, &before, &c, |i, j| {
                    in_tri(uplo, i, j).then(|| (cz(0.0, 0.0), 0.0))
                });
            }
            // (alpha == 0 || k == 0) && beta == 1: C bitwise unchanged, -0.0 and
            // NaN included (no `c + 0` / `1·c` rewrite).
            for (k, alpha) in [(3, zero), (0, one)] {
                let mut c = store::<T>(layout, n, n, ld, |i, j| {
                    if (i + j) % 2 == 0 {
                        T::from_c64(cz(-0.0, -0.0))
                    } else {
                        nan()
                    }
                });
                let before = bits_of(&c);
                call(k, alpha, one, ld as i32, &mut c);
                assert_eq!(bits_of(&c), before, "{what}: k={k} beta=1 must not touch C");
            }
            // Invalid arguments are a no-op: k < 0, lda too small, n < 0, n == 0.
            for (nn, k, lda) in [
                (n as i32, -1, ld as i32),
                (n as i32, 3, 2),
                (-1, 3, ld as i32),
            ] {
                let mut c = fresh();
                let before = bits_of(&c);
                // SAFETY: rejected before any pointer is dereferenced.
                unsafe {
                    T::gemmtr(
                        layout,
                        uplo,
                        CblasTranspose::NoTrans,
                        CblasTranspose::NoTrans,
                        nn,
                        k,
                        one,
                        nan_buf.as_ptr(),
                        lda,
                        nan_buf.as_ptr(),
                        ld as i32,
                        one,
                        c.as_mut_ptr(),
                        ld as i32,
                    );
                }
                assert_eq!(
                    bits_of(&c),
                    before,
                    "{what}: invalid n={nn} k={k} lda={lda}"
                );
            }
            // SAFETY: n == 0 dereferences nothing (null operands).
            unsafe {
                T::gemmtr(
                    layout,
                    uplo,
                    CblasTranspose::NoTrans,
                    CblasTranspose::NoTrans,
                    0,
                    3,
                    one,
                    std::ptr::null(),
                    1,
                    std::ptr::null(),
                    1,
                    one,
                    std::ptr::null_mut(),
                    1,
                );
            }
        }
    }
}

#[test]
fn test_issue_14_gemmtr_quick_returns_and_special_values() {
    gemmtr_special::<f32>();
    gemmtr_special::<f64>();
    gemmtr_special::<Complex32>();
    gemmtr_special::<Complex64>();
}

// =============================================================================
// Complex SYMM
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn symm_case<T: ComplexL3>(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: usize,
    n: usize,
    alpha: Complex64,
    beta: Complex64,
) {
    let ka = if side == CblasSide::Left { m } else { n };
    let (lda, ldb, ldc) = (
        ld_for(layout, ka, ka, 1),
        ld_for(layout, m, n, 2),
        ld_for(layout, m, n, 3),
    );
    // Complex SYMMETRIC: A(i, j) = A(j, i), no conjugation.
    let sym = |i: usize, j: usize| eff::<T>(val(i.min(j), i.max(j), 4));
    let bm = |i: usize, j: usize| eff::<T>(val(i, j, 5));
    let c0 = |i: usize, j: usize| eff::<T>(val(i, j, 6));
    let (al, be) = (eff::<T>(alpha), eff::<T>(beta));
    let beta_zero = be == cz(0.0, 0.0);
    let a = store::<T>(layout, ka, ka, lda, |i, j| {
        if in_tri(uplo, i, j) {
            T::from_c64(sym(i, j))
        } else {
            nan()
        }
    });
    let b = store::<T>(layout, m, n, ldb, |i, j| T::from_c64(bm(i, j)));
    let mut c = store::<T>(layout, m, n, ldc, |i, j| {
        if beta_zero {
            nan()
        } else {
            T::from_c64(c0(i, j))
        }
    });
    let before = c.clone();
    // SAFETY: buffers sized by `store` for the declared shapes / leading dimensions.
    unsafe {
        T::symm(
            layout,
            side,
            uplo,
            m as i32,
            n as i32,
            T::from_c64(alpha),
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            T::from_c64(beta),
            c.as_mut_ptr(),
            ldc as i32,
        );
    }
    let what = format!("symm {layout:?} {side:?} {uplo:?} m={m} n={n} a={alpha} b={beta}");
    check_slots(&what, layout, (m, n, ldc), ka, &before, &c, |i, j| {
        let (mut sum, mut mag) = (cz(0.0, 0.0), 0.0);
        for l in 0..ka {
            let p = match side {
                CblasSide::Left => sym(i, l) * bm(l, j),
                CblasSide::Right => bm(i, l) * sym(l, j),
            };
            sum += p;
            mag += p.norm();
        }
        let (base, base_mag) = if beta_zero {
            (cz(0.0, 0.0), 0.0)
        } else {
            (be * c0(i, j), be.norm() * c0(i, j).norm())
        };
        Some((al * sum + base, al.norm() * mag + base_mag))
    });
}

fn sweep_symm<T: ComplexL3>() {
    for layout in LAYOUTS {
        for side in SIDES {
            for uplo in UPLOS {
                for (m, n) in [(1, 1), (3, 4), (4, 3)] {
                    for ((ar, ai), (br, bi)) in SCALARS {
                        symm_case::<T>(layout, side, uplo, m, n, cz(ar, ai), cz(br, bi));
                    }
                }
                // ka >= 32 reaches the native's GEMM-backed path.
                let (m, n) = if side == CblasSide::Left {
                    (34, 5)
                } else {
                    (5, 34)
                };
                symm_case::<T>(layout, side, uplo, m, n, cz(0.5, 1.0), cz(-0.5, 0.25));
            }
        }
    }
}

#[test]
fn test_issue_14_csymm_zsymm_match_dense_reference() {
    sweep_symm::<Complex32>();
    sweep_symm::<Complex64>();
}

fn symm_special<T: ComplexL3>() {
    let (m, n, ld) = (3usize, 4usize, 6usize);
    let nan_buf = vec![nan::<T>(); ld * ld];
    let one = T::from_c64(cz(1.0, 0.0));
    let zero = T::from_c64(cz(0.0, 0.0));
    let beta = cz(0.5, -0.5);
    for layout in LAYOUTS {
        let c0 = |i: usize, j: usize| eff::<T>(val(i, j, 7));
        let call = |alpha: T, beta: T, c: &mut [T]| {
            // SAFETY: A / B are NaN buffers of ld*ld elements, C is m x n with ld.
            unsafe {
                T::symm(
                    layout,
                    CblasSide::Left,
                    CblasUplo::Upper,
                    m as i32,
                    n as i32,
                    alpha,
                    nan_buf.as_ptr(),
                    ld as i32,
                    nan_buf.as_ptr(),
                    ld as i32,
                    beta,
                    c.as_mut_ptr(),
                    ld as i32,
                );
            }
        };
        let what = format!("symm special {layout:?}");
        // alpha == 0: C := beta·C, A / B (NaN) never read.
        let mut c = store::<T>(layout, m, n, ld, |i, j| T::from_c64(c0(i, j)));
        let before = c.clone();
        call(zero, T::from_c64(beta), &mut c);
        check_slots(&what, layout, (m, n, ld), 1, &before, &c, |i, j| {
            Some((eff::<T>(beta) * c0(i, j), c0(i, j).norm()))
        });
        // alpha == 0, beta == 0: zero-fill without reading C.
        let mut c = vec![nan::<T>(); ld * ld];
        let before = c.clone();
        call(zero, zero, &mut c);
        check_slots(&what, layout, (m, n, ld), 1, &before, &c, |_, _| {
            Some((cz(0.0, 0.0), 0.0))
        });
        // alpha == 0, beta == 1: bitwise unchanged.
        let mut c = store::<T>(layout, m, n, ld, |i, j| {
            if (i + j) % 2 == 0 {
                T::from_c64(cz(-0.0, -0.0))
            } else {
                nan()
            }
        });
        let before = bits_of(&c);
        call(zero, one, &mut c);
        assert_eq!(
            bits_of(&c),
            before,
            "{what}: alpha=0 beta=1 must not touch C"
        );
        // m == 0 / n == 0 / undersized lda: no-op.
        for (mm, nn, lda) in [
            (0, n as i32, ld as i32),
            (m as i32, 0, ld as i32),
            (m as i32, n as i32, 2),
        ] {
            let mut c = store::<T>(layout, m, n, ld, |i, j| T::from_c64(c0(i, j)));
            let before = bits_of(&c);
            // SAFETY: rejected before any operand is read.
            unsafe {
                T::symm(
                    layout,
                    CblasSide::Left,
                    CblasUplo::Upper,
                    mm,
                    nn,
                    one,
                    nan_buf.as_ptr(),
                    lda,
                    nan_buf.as_ptr(),
                    ld as i32,
                    one,
                    c.as_mut_ptr(),
                    ld as i32,
                );
            }
            assert_eq!(bits_of(&c), before, "{what}: m={mm} n={nn} lda={lda}");
        }
    }
}

#[test]
fn test_issue_14_csymm_zsymm_quick_returns_and_special_values() {
    symm_special::<Complex32>();
    symm_special::<Complex64>();
}

// =============================================================================
// Complex SYRK / SYR2K
// =============================================================================

/// One SYRK (`two == false`) or SYR2K (`two == true`) case against the dense
/// symmetric definition (never conjugated). `trans` is the caller's; in
/// row-major layout `ConjTrans` must behave exactly like `Trans`.
#[allow(clippy::too_many_arguments)]
fn rank_case<T: ComplexL3>(
    two: bool,
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: usize,
    k: usize,
    alpha: Complex64,
    beta: Complex64,
) {
    let notrans = trans == CblasTranspose::NoTrans;
    let (ar, ac) = if notrans { (n, k) } else { (k, n) };
    let (lda, ldb, ldc) = (
        ld_for(layout, ar, ac, 1),
        ld_for(layout, ar, ac, 2),
        ld_for(layout, n, n, 3),
    );
    let sa = |i: usize, j: usize| eff::<T>(val(i, j, 8));
    let sb = |i: usize, j: usize| eff::<T>(val(i, j, 9));
    let c0 = |i: usize, j: usize| eff::<T>(val(i, j, 10));
    // op(X)(i, l) of the n x k operand: X(i, l) or X(l, i), never conjugated.
    let opx = |s: &dyn Fn(usize, usize) -> Complex64, i: usize, l: usize| {
        if notrans { s(i, l) } else { s(l, i) }
    };
    let (al, be) = (eff::<T>(alpha), eff::<T>(beta));
    let beta_zero = be == cz(0.0, 0.0);
    let a = store::<T>(layout, ar, ac, lda, |i, j| T::from_c64(sa(i, j)));
    let b = store::<T>(layout, ar, ac, ldb, |i, j| T::from_c64(sb(i, j)));
    let mut c = store::<T>(layout, n, n, ldc, |i, j| {
        if in_tri(uplo, i, j) && !beta_zero {
            T::from_c64(c0(i, j))
        } else {
            nan()
        }
    });
    let before = c.clone();
    // SAFETY: buffers sized by `store` for the declared shapes / leading dimensions.
    unsafe {
        if two {
            T::syr2k(
                layout,
                uplo,
                trans,
                n as i32,
                k as i32,
                T::from_c64(alpha),
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                T::from_c64(beta),
                c.as_mut_ptr(),
                ldc as i32,
            );
        } else {
            T::syrk(
                layout,
                uplo,
                trans,
                n as i32,
                k as i32,
                T::from_c64(alpha),
                a.as_ptr(),
                lda as i32,
                T::from_c64(beta),
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }
    let name = if two { "syr2k" } else { "syrk" };
    let what = format!("{name} {layout:?} {uplo:?} {trans:?} n={n} k={k} a={alpha} b={beta}");
    check_slots(&what, layout, (n, n, ldc), 2 * k, &before, &c, |i, j| {
        if !in_tri(uplo, i, j) {
            return None;
        }
        let (mut sum, mut mag) = (cz(0.0, 0.0), 0.0);
        for l in 0..k {
            let p = if two {
                opx(&sa, i, l) * opx(&sb, j, l) + opx(&sb, i, l) * opx(&sa, j, l)
            } else {
                opx(&sa, i, l) * opx(&sa, j, l)
            };
            sum += p;
            mag += if two {
                (opx(&sa, i, l) * opx(&sb, j, l)).norm() + (opx(&sb, i, l) * opx(&sa, j, l)).norm()
            } else {
                p.norm()
            };
        }
        let (base, base_mag) = if beta_zero {
            (cz(0.0, 0.0), 0.0)
        } else {
            (be * c0(i, j), be.norm() * c0(i, j).norm())
        };
        Some((al * sum + base, al.norm() * mag + base_mag))
    });
}

fn sweep_rank<T: ComplexL3>(two: bool) {
    for layout in LAYOUTS {
        for uplo in UPLOS {
            let transes: &[CblasTranspose] = match layout {
                CblasLayout::ColMajor => &[CblasTranspose::NoTrans, CblasTranspose::Trans],
                CblasLayout::RowMajor => &TRANSES,
            };
            for &trans in transes {
                for (n, k) in [(1, 1), (3, 2), (4, 5)] {
                    for ((ar, ai), (br, bi)) in SCALARS {
                        rank_case::<T>(two, layout, uplo, trans, n, k, cz(ar, ai), cz(br, bi));
                    }
                }
                // n >= 32 && k >= 8 reaches the native's GEMM-backed path.
                rank_case::<T>(
                    two,
                    layout,
                    uplo,
                    trans,
                    36,
                    10,
                    cz(0.5, -0.75),
                    cz(0.25, 0.5),
                );
            }
        }
    }
}

#[test]
fn test_issue_14_csyrk_zsyrk_match_dense_reference() {
    sweep_rank::<Complex32>(false);
    sweep_rank::<Complex64>(false);
}

#[test]
fn test_issue_14_csyr2k_zsyr2k_match_dense_reference() {
    sweep_rank::<Complex32>(true);
    sweep_rank::<Complex64>(true);
}

/// Reference CBLAS 3.12: column-major `ConjTrans` reaches Fortran C/ZSYRK and
/// C/ZSYR2K as `TRANS = 'C'` and is rejected (C bitwise untouched); row-major
/// `ConjTrans` is mapped to `'N'`, i.e. it is exactly row-major `Trans`.
fn rank_conj_trans<T: ComplexL3>(two: bool) {
    let (n, k, ld) = (4usize, 3usize, 5usize);
    let a = store::<T>(CblasLayout::ColMajor, ld, ld, ld, |i, j| {
        T::from_c64(val(i, j, 11))
    });
    let b = store::<T>(CblasLayout::ColMajor, ld, ld, ld, |i, j| {
        T::from_c64(val(i, j, 12))
    });
    let alpha = T::from_c64(cz(0.75, -0.25));
    let beta = T::from_c64(cz(0.5, 0.5));
    let run = |layout: CblasLayout, trans: CblasTranspose| {
        let mut c = store::<T>(layout, n, n, ld, |i, j| {
            if (i + j) % 3 == 0 {
                T::from_c64(cz(-0.0, 1.0))
            } else {
                T::from_c64(val(i, j, 13))
            }
        });
        // SAFETY: A / B / C are ld x ld buffers with ld >= max(n, k).
        unsafe {
            if two {
                T::syr2k(
                    layout,
                    CblasUplo::Lower,
                    trans,
                    n as i32,
                    k as i32,
                    alpha,
                    a.as_ptr(),
                    ld as i32,
                    b.as_ptr(),
                    ld as i32,
                    beta,
                    c.as_mut_ptr(),
                    ld as i32,
                );
            } else {
                T::syrk(
                    layout,
                    CblasUplo::Upper,
                    trans,
                    n as i32,
                    k as i32,
                    alpha,
                    a.as_ptr(),
                    ld as i32,
                    beta,
                    c.as_mut_ptr(),
                    ld as i32,
                );
            }
        }
        c
    };
    let untouched = bits_of(&run(CblasLayout::ColMajor, CblasTranspose::NoTrans));
    let original = {
        let c = store::<T>(CblasLayout::ColMajor, n, n, ld, |i, j| {
            if (i + j) % 3 == 0 {
                T::from_c64(cz(-0.0, 1.0))
            } else {
                T::from_c64(val(i, j, 13))
            }
        });
        bits_of(&c)
    };
    assert_ne!(untouched, original, "sanity: NoTrans must update C");
    assert_eq!(
        bits_of(&run(CblasLayout::ColMajor, CblasTranspose::ConjTrans)),
        original,
        "two={two}: column-major ConjTrans must be a no-op"
    );
    let trans = bits_of(&run(CblasLayout::RowMajor, CblasTranspose::Trans));
    assert_eq!(
        bits_of(&run(CblasLayout::RowMajor, CblasTranspose::ConjTrans)),
        trans,
        "two={two}: row-major ConjTrans must equal row-major Trans"
    );
}

#[test]
fn test_issue_14_csyrk_zsyrk_csyr2k_zsyr2k_conj_trans_follows_reference_cblas() {
    rank_conj_trans::<Complex32>(false);
    rank_conj_trans::<Complex64>(false);
    rank_conj_trans::<Complex32>(true);
    rank_conj_trans::<Complex64>(true);
}

fn rank_special<T: ComplexL3>(two: bool) {
    let (n, ld) = (4usize, 6usize);
    let nan_buf = vec![nan::<T>(); ld * ld];
    let one = T::from_c64(cz(1.0, 0.0));
    let zero = T::from_c64(cz(0.0, 0.0));
    let beta = cz(0.5, 0.25);
    for layout in LAYOUTS {
        for uplo in UPLOS {
            let c0 = |i: usize, j: usize| eff::<T>(val(i, j, 14));
            let fresh = || {
                store::<T>(layout, n, n, ld, |i, j| {
                    if in_tri(uplo, i, j) {
                        T::from_c64(c0(i, j))
                    } else {
                        nan()
                    }
                })
            };
            let call = |k: i32, alpha: T, beta: T, lda: i32, c: &mut [T]| {
                // SAFETY: A / B are NaN buffers of ld*ld elements, C is n x n with ld.
                unsafe {
                    if two {
                        T::syr2k(
                            layout,
                            uplo,
                            CblasTranspose::NoTrans,
                            n as i32,
                            k,
                            alpha,
                            nan_buf.as_ptr(),
                            lda,
                            nan_buf.as_ptr(),
                            ld as i32,
                            beta,
                            c.as_mut_ptr(),
                            ld as i32,
                        );
                    } else {
                        T::syrk(
                            layout,
                            uplo,
                            CblasTranspose::NoTrans,
                            n as i32,
                            k,
                            alpha,
                            nan_buf.as_ptr(),
                            lda,
                            beta,
                            c.as_mut_ptr(),
                            ld as i32,
                        );
                    }
                }
            };
            let what = format!("rank-k special two={two} {layout:?} {uplo:?}");
            // alpha == 0 or k == 0: triangle := beta·C, A / B (NaN) never read.
            for (k, alpha) in [(3, zero), (0, one)] {
                let mut c = fresh();
                let before = c.clone();
                call(k, alpha, T::from_c64(beta), ld as i32, &mut c);
                check_slots(&what, layout, (n, n, ld), 1, &before, &c, |i, j| {
                    in_tri(uplo, i, j).then(|| (eff::<T>(beta) * c0(i, j), c0(i, j).norm()))
                });
                // beta == 0: zero-fill without reading C.
                let mut c = vec![nan::<T>(); ld * ld];
                let before = c.clone();
                call(k, alpha, zero, ld as i32, &mut c);
                check_slots(&what, layout, (n, n, ld), 1, &before, &c, |i, j| {
                    in_tri(uplo, i, j).then(|| (cz(0.0, 0.0), 0.0))
                });
                // beta == 1: C bitwise unchanged (-0.0 and NaN included).
                let mut c = store::<T>(layout, n, n, ld, |i, j| {
                    if (i + j) % 2 == 0 {
                        T::from_c64(cz(-0.0, -0.0))
                    } else {
                        nan()
                    }
                });
                let before = bits_of(&c);
                call(k, alpha, one, ld as i32, &mut c);
                assert_eq!(bits_of(&c), before, "{what}: k={k} beta=1 must not touch C");
            }
            // Invalid: k < 0, lda too small: no-op.
            for (k, lda) in [(-1, ld as i32), (3, 2)] {
                let mut c = fresh();
                let before = bits_of(&c);
                call(k, one, one, lda, &mut c);
                assert_eq!(bits_of(&c), before, "{what}: invalid k={k} lda={lda}");
            }
        }
    }
    // n == 0: no-op with null operands.
    // SAFETY: n == 0 dereferences nothing.
    unsafe {
        T::syrk(
            CblasLayout::ColMajor,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            0,
            3,
            one,
            std::ptr::null(),
            1,
            one,
            std::ptr::null_mut(),
            1,
        );
    }
}

#[test]
fn test_issue_14_csyrk_csyr2k_quick_returns_and_special_values() {
    rank_special::<Complex32>(false);
    rank_special::<Complex64>(false);
    rank_special::<Complex32>(true);
    rank_special::<Complex64>(true);
}

// =============================================================================
// Complex TRMM / TRSM
// =============================================================================

/// One TRMM (`solve == false`: `B := alpha·op(T)·X0` / `alpha·X0·op(T)`) or
/// TRSM (`solve == true`: `B := op(T)·X0` in, `alpha·X0` out) case. The stored
/// triangle is well conditioned; its other triangle and (for `Unit`) its
/// diagonal hold NaN, which must never be read.
#[allow(clippy::too_many_arguments)]
fn tri_case<T: ComplexL3>(
    solve: bool,
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    (m, n): (usize, usize),
    alpha: Complex64,
) {
    let ka = if side == CblasSide::Left { m } else { n };
    let (lda, ldb) = (ld_for(layout, ka, ka, 1), ld_for(layout, m, n, 2));
    let diag_val = |i: usize| cz(2.0 + 0.25 * (i % 5) as f64, 0.5 - 0.25 * (i % 3) as f64);
    let off = |i: usize, j: usize| eff::<T>(val(i, j, 15) * (0.5 / ka as f64));
    let tri = |i: usize, j: usize| -> Complex64 {
        if i == j {
            match diag {
                CblasDiag::Unit => cz(1.0, 0.0),
                CblasDiag::NonUnit => eff::<T>(diag_val(i)),
            }
        } else if in_tri(uplo, i, j) {
            off(i, j)
        } else {
            cz(0.0, 0.0)
        }
    };
    let a = store::<T>(layout, ka, ka, lda, |i, j| {
        if i == j {
            match diag {
                CblasDiag::Unit => nan(),
                CblasDiag::NonUnit => T::from_c64(diag_val(i)),
            }
        } else if in_tri(uplo, i, j) {
            T::from_c64(off(i, j))
        } else {
            nan()
        }
    });
    let x0 = |i: usize, j: usize| eff::<T>(val(i, j, 16));
    let opt = |i: usize, j: usize| op_at(trans, &tri, i, j);
    // (op(T)·X0)(i, j) or (X0·op(T))(i, j), with the sum of |products|.
    let apply = |i: usize, j: usize| -> (Complex64, f64) {
        let (mut sum, mut mag) = (cz(0.0, 0.0), 0.0);
        for l in 0..ka {
            let p = match side {
                CblasSide::Left => opt(i, l) * x0(l, j),
                CblasSide::Right => x0(i, l) * opt(l, j),
            };
            sum += p;
            mag += p.norm();
        }
        (sum, mag)
    };
    let al = eff::<T>(alpha);
    let mut b = store::<T>(layout, m, n, ldb, |i, j| {
        T::from_c64(if solve { apply(i, j).0 } else { x0(i, j) })
    });
    let before = b.clone();
    // SAFETY: buffers sized by `store` for the declared shapes / leading dimensions.
    unsafe {
        T::tri(
            solve,
            layout,
            side,
            uplo,
            trans,
            diag,
            m as i32,
            n as i32,
            T::from_c64(alpha),
            a.as_ptr(),
            lda as i32,
            b.as_mut_ptr(),
            ldb as i32,
        );
    }
    let name = if solve { "trsm" } else { "trmm" };
    let what =
        format!("{name} {layout:?} {side:?} {uplo:?} {trans:?} {diag:?} m={m} n={n} a={alpha}");
    check_slots(&what, layout, (m, n, ldb), ka, &before, &b, |i, j| {
        if solve {
            // The triangle is well conditioned (|diag| >= 1, small
            // off-diagonals), so the solve error stays O(ka·eps·|alpha·x|).
            Some((al * x0(i, j), 8.0 * al.norm() * (1.0 + x0(i, j).norm())))
        } else {
            let (sum, mag) = apply(i, j);
            Some((al * sum, al.norm() * mag))
        }
    });
}

fn sweep_tri<T: ComplexL3>(solve: bool) {
    for layout in LAYOUTS {
        for side in SIDES {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for diag in DIAGS {
                        for shape in [(3, 5), (5, 3), (1, 4), (4, 1)] {
                            for alpha in [cz(0.5, -1.5), cz(1.0, 0.0)] {
                                tri_case::<T>(solve, layout, side, uplo, trans, diag, shape, alpha);
                            }
                        }
                        // Order 130 spans two 128-row diagonal blocks of the
                        // native TRMM/TRSM (in-place 3M-GEMM update between them).
                        let big = 130;
                        let shape = if side == CblasSide::Left {
                            (big, 3)
                        } else {
                            (3, big)
                        };
                        tri_case::<T>(
                            solve,
                            layout,
                            side,
                            uplo,
                            trans,
                            diag,
                            shape,
                            cz(-0.75, 0.5),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_14_ctrmm_ztrmm_match_dense_reference() {
    sweep_tri::<Complex32>(false);
    sweep_tri::<Complex64>(false);
}

#[test]
fn test_issue_14_ctrsm_ztrsm_match_dense_reference() {
    sweep_tri::<Complex32>(true);
    sweep_tri::<Complex64>(true);
}

fn tri_special<T: ComplexL3>() {
    let (m, n, ld) = (3usize, 4usize, 6usize);
    let nan_buf = vec![nan::<T>(); ld * ld];
    let zero = T::from_c64(cz(0.0, 0.0));
    let one = T::from_c64(cz(1.0, 0.0));
    for solve in [false, true] {
        for layout in LAYOUTS {
            for side in SIDES {
                let what = format!("tri special solve={solve} {layout:?} {side:?}");
                // alpha == 0: B := 0 without reading A or B (both NaN).
                let mut b = store::<T>(layout, m, n, ld, |_, _| nan());
                let before = b.clone();
                // SAFETY: A is an ld x ld NaN buffer, B is m x n with ld.
                unsafe {
                    T::tri(
                        solve,
                        layout,
                        side,
                        CblasUplo::Lower,
                        CblasTranspose::ConjTrans,
                        CblasDiag::NonUnit,
                        m as i32,
                        n as i32,
                        zero,
                        nan_buf.as_ptr(),
                        ld as i32,
                        b.as_mut_ptr(),
                        ld as i32,
                    );
                }
                check_slots(&what, layout, (m, n, ld), 1, &before, &b, |_, _| {
                    Some((cz(0.0, 0.0), 0.0))
                });
                // m == 0 / n == 0 / undersized lda: no-op.
                for (mm, nn, lda) in [
                    (0, n as i32, ld as i32),
                    (m as i32, 0, ld as i32),
                    (m as i32, n as i32, 2),
                ] {
                    let mut b = store::<T>(layout, m, n, ld, |i, j| T::from_c64(val(i, j, 17)));
                    let before = bits_of(&b);
                    // SAFETY: rejected before any operand is read.
                    unsafe {
                        T::tri(
                            solve,
                            layout,
                            side,
                            CblasUplo::Upper,
                            CblasTranspose::NoTrans,
                            CblasDiag::Unit,
                            mm,
                            nn,
                            one,
                            nan_buf.as_ptr(),
                            lda,
                            b.as_mut_ptr(),
                            ld as i32,
                        );
                    }
                    assert_eq!(bits_of(&b), before, "{what}: m={mm} n={nn} lda={lda}");
                }
            }
        }
    }
}

#[test]
fn test_issue_14_ctrmm_ctrsm_alpha_zero_and_noops() {
    tri_special::<Complex32>();
    tri_special::<Complex64>();
}
