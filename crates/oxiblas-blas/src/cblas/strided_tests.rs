//! Unit tests for the strided marshaling helpers, plus the test-only
//! scaffolding shared by the `banded`, `packed` and `complex_level2_ext`
//! tests (issue #14):
//!
//! * [`Num`]: one abstraction over the four BLAS precisions, widened to
//!   `Complex64` for the references;
//! * a deterministic generator (no RNG dependency);
//! * CBLAS storage builders written from the storage definitions of each
//!   layout (row-major band rows, row-wise packing, ...) — never from the
//!   wrappers' column-major reinterpretation, so a wrong row-major mapping
//!   cannot cancel out;
//! * naive dense references evaluated in `Complex64` and NaN-safe
//!   assertions.

use super::*;
use num_complex::{Complex32, Complex64};
use oxiblas_core::scalar::Field;

// =============================================================================
// Scalar abstraction and data
// =============================================================================

/// The four BLAS precisions under test. `Field` provides the native
/// arithmetic, so expected values such as `beta*y` can be formed exactly as
/// the wrappers form them (bitwise comparisons).
pub(crate) trait Num: Field + core::fmt::Debug + 'static {
    /// Machine epsilon of the precision.
    const EPS: f64;
    /// Builds a value; real types drop `im`.
    fn mk(re: f64, im: f64) -> Self;
    /// Exact widening to `Complex64`.
    fn widen(self) -> Complex64;
    /// Complex conjugate (identity for real types).
    fn cj(self) -> Self;
    /// Bit patterns of both parts (NaN-safe identity check).
    fn bits(self) -> (u64, u64);
}

impl Num for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn mk(re: f64, _im: f64) -> Self {
        re as f32
    }
    fn widen(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
    fn cj(self) -> Self {
        self
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
}

impl Num for f64 {
    const EPS: f64 = f64::EPSILON;
    fn mk(re: f64, _im: f64) -> Self {
        re
    }
    fn widen(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn cj(self) -> Self {
        self
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl Num for Complex32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn mk(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
    fn widen(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
    fn cj(self) -> Self {
        Complex32::new(self.re, -self.im)
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
}

impl Num for Complex64 {
    const EPS: f64 = f64::EPSILON;
    fn mk(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn widen(self) -> Complex64 {
        self
    }
    fn cj(self) -> Self {
        Complex64::new(self.re, -self.im)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

pub(crate) const LAYOUTS: [CblasLayout; 2] = [CblasLayout::ColMajor, CblasLayout::RowMajor];
pub(crate) const UPLOS: [CblasUplo; 2] = [CblasUplo::Upper, CblasUplo::Lower];
pub(crate) const TRANSES: [CblasTranspose; 3] = [
    CblasTranspose::NoTrans,
    CblasTranspose::Trans,
    CblasTranspose::ConjTrans,
];
pub(crate) const DIAGS: [CblasDiag; 2] = [CblasDiag::NonUnit, CblasDiag::Unit];
/// The increments every Level-2 routine is exercised with.
pub(crate) const INCS: [i32; 4] = [1, 2, -1, -3];

pub(crate) fn nan<T: Num>() -> T {
    T::mk(f64::NAN, f64::NAN)
}

pub(crate) fn zero<T: Num>() -> T {
    T::mk(0.0, 0.0)
}

/// A buffer of distinctive, non-finite and signed-zero values: a routine that
/// must leave it alone is checked bitwise.
pub(crate) fn poison<T: Num>(len: usize) -> Vec<T> {
    let pattern = [
        T::mk(f64::NAN, -0.0),
        T::mk(f64::INFINITY, 1.5),
        T::mk(-0.0, f64::NEG_INFINITY),
        T::mk(2.5, f64::NAN),
    ];
    (0..len).map(|i| pattern[i % pattern.len()]).collect()
}

/// Deterministic SplitMix64 generator.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    pub(crate) fn num<T: Num>(&mut self) -> T {
        let re = self.unit();
        let im = self.unit();
        T::mk(re, im)
    }

    pub(crate) fn vec<T: Num>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.num()).collect()
    }
}

// =============================================================================
// Strided vectors
// =============================================================================

/// Buffer slot of logical element `i` of a length-`n` vector with increment
/// `inc` (reference BLAS: a negative increment starts at the far end).
pub(crate) fn slot(i: usize, n: usize, inc: i32) -> usize {
    let step = inc.unsigned_abs() as usize;
    if inc > 0 {
        i * step
    } else {
        (n - 1 - i) * step
    }
}

/// Lays `v` out with increment `inc`; the gaps hold `fill`.
pub(crate) fn strided<T: Num>(v: &[T], inc: i32, fill: T) -> Vec<T> {
    let n = v.len();
    let len = if n == 0 {
        0
    } else {
        (n - 1) * inc.unsigned_abs() as usize + 1
    };
    let mut buf = vec![fill; len];
    for (i, &e) in v.iter().enumerate() {
        buf[slot(i, n, inc)] = e;
    }
    buf
}

pub(crate) fn unstrided<T: Num>(buf: &[T], n: usize, inc: i32) -> Vec<T> {
    (0..n).map(|i| buf[slot(i, n, inc)]).collect()
}

pub(crate) fn widen_all<T: Num>(v: &[T]) -> Vec<Complex64> {
    v.iter().map(|e| e.widen()).collect()
}

pub(crate) fn assert_bits_eq<T: Num>(before: &[T], after: &[T], what: &str) {
    assert_eq!(before.len(), after.len(), "{what}: length");
    for (i, (p, q)) in before.iter().zip(after).enumerate() {
        assert_eq!(
            p.bits(),
            q.bits(),
            "{what}: slot {i} changed ({p:?} -> {q:?})"
        );
    }
}

/// Asserts that every slot of a strided buffer that the stride does *not*
/// address is bitwise unchanged.
pub(crate) fn assert_gaps_unchanged<T: Num>(
    before: &[T],
    after: &[T],
    n: usize,
    inc: i32,
    what: &str,
) {
    let mut addressed = vec![false; before.len()];
    for i in 0..n {
        addressed[slot(i, n, inc)] = true;
    }
    for (s, used) in addressed.iter().enumerate() {
        if !used {
            assert_eq!(
                before[s].bits(),
                after[s].bits(),
                "{what}: gap slot {s} changed"
            );
        }
    }
}

// =============================================================================
// Dense references (Complex64)
// =============================================================================

/// Logical dense matrix, indexed `[row][col]`.
pub(crate) type Dense<T> = Vec<Vec<T>>;

fn op_entry<T: Num>(a: &Dense<T>, trans: CblasTranspose, i: usize, j: usize) -> Complex64 {
    match trans {
        CblasTranspose::NoTrans => a[i][j].widen(),
        CblasTranspose::Trans => a[j][i].widen(),
        CblasTranspose::ConjTrans => a[j][i].widen().conj(),
    }
}

/// `op(A)*x` (with `A` `rows` x `cols`) and the magnitude bound `|op(A)|*|x|`.
pub(crate) fn op_mul<T: Num>(
    a: &Dense<T>,
    rows: usize,
    cols: usize,
    trans: CblasTranspose,
    x: &[Complex64],
) -> (Vec<Complex64>, Vec<f64>) {
    let (out_len, in_len) = if trans == CblasTranspose::NoTrans {
        (rows, cols)
    } else {
        (cols, rows)
    };
    let mut out = vec![Complex64::new(0.0, 0.0); out_len];
    let mut bound = vec![0.0; out_len];
    for i in 0..out_len {
        for (j, xj) in x.iter().enumerate().take(in_len) {
            let e = op_entry(a, trans, i, j);
            out[i] += e * xj;
            bound[i] += e.norm() * xj.norm();
        }
    }
    (out, bound)
}

/// `alpha*op(A)*x + beta*y` and its magnitude bound.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mv_ref<T: Num>(
    a: &Dense<T>,
    rows: usize,
    cols: usize,
    trans: CblasTranspose,
    alpha: T,
    x: &[T],
    beta: T,
    y: &[T],
) -> (Vec<Complex64>, Vec<f64>) {
    let (ax, ax_bound) = op_mul(a, rows, cols, trans, &widen_all(x));
    let (al, be) = (alpha.widen(), beta.widen());
    let want = ax
        .iter()
        .zip(y)
        .map(|(v, yi)| al * v + be * yi.widen())
        .collect();
    let bound = ax_bound
        .iter()
        .zip(y)
        .map(|(b, yi)| al.norm() * b + be.norm() * yi.widen().norm())
        .collect();
    (want, bound)
}

/// Elementwise `|got - want| <= 64*eps*bound` (generous enough for every
/// summation order on the small problems used here, far below any real
/// error).
pub(crate) fn assert_close(
    got: &[Complex64],
    want: &[Complex64],
    bound: &[f64],
    eps: f64,
    what: &str,
) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for i in 0..want.len() {
        let tol = 64.0 * eps * bound[i] + 1e-30;
        let err = (got[i] - want[i]).norm();
        assert!(
            err <= tol,
            "{what}: element {i}: got {:?}, want {:?}, |err| = {err:e} > tol {tol:e}",
            got[i],
            want[i]
        );
    }
}

/// Backward-stability check of a triangular solve: `op(A)*x == b` up to
/// `O(n*eps)*(|op(A)|*|x| + |b|)`.
pub(crate) fn assert_solves<T: Num>(
    a: &Dense<T>,
    trans: CblasTranspose,
    x: &[T],
    b: &[T],
    what: &str,
) {
    let n = a.len();
    let (ax, bound) = op_mul(a, n, n, trans, &widen_all(x));
    let b = widen_all(b);
    let full_bound: Vec<f64> = bound.iter().zip(&b).map(|(p, q)| p + q.norm()).collect();
    // A solve's backward error grows with `n`; scale the tolerance with it.
    let eps = T::EPS * (n as f64 / 8.0).max(1.0);
    assert_close(&ax, &b, &full_bound, eps, what);
}

// =============================================================================
// CBLAS storage (definitions per layout)
// =============================================================================

/// `A[i][j]` slot of a general band matrix (`kl` sub-, `ku` super-diagonals).
pub(crate) fn gb_slot(
    layout: CblasLayout,
    i: usize,
    j: usize,
    kl: usize,
    ku: usize,
    lda: usize,
) -> usize {
    match layout {
        CblasLayout::ColMajor => (ku + i - j) + j * lda,
        CblasLayout::RowMajor => i * lda + (kl + j - i),
    }
}

/// `A[i][j]` slot of a symmetric / Hermitian / triangular band matrix.
pub(crate) fn sb_slot(
    layout: CblasLayout,
    uplo: CblasUplo,
    i: usize,
    j: usize,
    k: usize,
    lda: usize,
) -> usize {
    match (layout, uplo) {
        (CblasLayout::ColMajor, CblasUplo::Upper) => (k + i - j) + j * lda,
        (CblasLayout::ColMajor, CblasUplo::Lower) => (i - j) + j * lda,
        (CblasLayout::RowMajor, CblasUplo::Upper) => i * lda + (j - i),
        (CblasLayout::RowMajor, CblasUplo::Lower) => i * lda + (k + j - i),
    }
}

/// `A[i][j]` slot of a packed triangle (`i <= j` for Upper, `i >= j` for
/// Lower): column-wise packing for `ColMajor`, row-wise for `RowMajor`.
pub(crate) fn packed_slot(
    layout: CblasLayout,
    uplo: CblasUplo,
    i: usize,
    j: usize,
    n: usize,
) -> usize {
    match (layout, uplo) {
        (CblasLayout::ColMajor, CblasUplo::Upper) => i + j * (j + 1) / 2,
        (CblasLayout::ColMajor, CblasUplo::Lower) => i + j * (2 * n - j - 1) / 2,
        (CblasLayout::RowMajor, CblasUplo::Upper) => j + i * (2 * n - i - 1) / 2,
        (CblasLayout::RowMajor, CblasUplo::Lower) => j + i * (i + 1) / 2,
    }
}

pub(crate) fn dense_slot(layout: CblasLayout, i: usize, j: usize, lda: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => i + j * lda,
        CblasLayout::RowMajor => i * lda + j,
    }
}

/// Whether `(i, j)` lies in the `uplo` triangle within `k` diagonals of it.
pub(crate) fn in_tri_band(uplo: CblasUplo, i: usize, j: usize, k: usize) -> bool {
    match uplo {
        CblasUplo::Upper => i <= j && j <= i + k,
        CblasUplo::Lower => j <= i && i <= j + k,
    }
}

/// Random `m x n` matrix that is zero outside the `(kl, ku)` band.
pub(crate) fn general_band<T: Num>(
    rng: &mut Rng,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
) -> Dense<T> {
    (0..m)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if j + kl >= i && j <= i + ku {
                        rng.num()
                    } else {
                        zero()
                    }
                })
                .collect()
        })
        .collect()
}

/// Random Hermitian (symmetric for real types) matrix of bandwidth `k` with a
/// real diagonal.
pub(crate) fn hermitian_band<T: Num>(rng: &mut Rng, n: usize, k: usize) -> Dense<T> {
    let mut a = vec![vec![zero::<T>(); n]; n];
    for i in 0..n {
        a[i][i] = T::mk(rng.unit(), 0.0);
        for j in (i + 1)..n.min(i + k + 1) {
            let v: T = rng.num();
            a[i][j] = v;
            a[j][i] = v.cj();
        }
    }
    a
}

/// Random triangular matrix of bandwidth `k`; a `NonUnit` diagonal is kept
/// away from zero (well-posed solves), a `Unit` one is logically `1`.
pub(crate) fn tri_band<T: Num>(
    rng: &mut Rng,
    n: usize,
    k: usize,
    uplo: CblasUplo,
    diag: CblasDiag,
) -> Dense<T> {
    let mut a = vec![vec![zero::<T>(); n]; n];
    for i in 0..n {
        for j in 0..n {
            if !in_tri_band(uplo, i, j, k) {
                continue;
            }
            a[i][j] = if i != j {
                rng.num()
            } else if diag == CblasDiag::Unit {
                T::mk(1.0, 0.0)
            } else {
                let v: T = rng.num();
                let v = v.widen();
                T::mk(2.0 + v.re.abs(), v.im)
            };
        }
    }
    a
}

/// General band storage of `a` per the CBLAS definition of `layout`; unused
/// slots hold NaN.
pub(crate) fn store_gb<T: Num>(
    layout: CblasLayout,
    a: &Dense<T>,
    n: usize,
    kl: usize,
    ku: usize,
    lda: usize,
) -> Vec<T> {
    let m = a.len();
    let lines = if layout == CblasLayout::ColMajor {
        n
    } else {
        m
    };
    let mut buf = vec![nan::<T>(); lda * lines];
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if j + kl >= i && j <= i + ku {
                buf[gb_slot(layout, i, j, kl, ku, lda)] = v;
            }
        }
    }
    buf
}

/// Band storage of the `uplo` triangle of a symmetric / Hermitian /
/// triangular band matrix; unused slots and a unit diagonal hold NaN, and
/// `diag_imag` replaces the imaginary part of the stored diagonal (the
/// Hermitian routines must ignore it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn store_band<T: Num>(
    layout: CblasLayout,
    uplo: CblasUplo,
    unit: bool,
    a: &Dense<T>,
    k: usize,
    lda: usize,
    diag_imag: Option<f64>,
) -> Vec<T> {
    let n = a.len();
    let mut buf = vec![nan::<T>(); lda * n];
    for i in 0..n {
        for j in 0..n {
            if !in_tri_band(uplo, i, j, k) || (unit && i == j) {
                continue;
            }
            let v = match diag_imag {
                Some(im) if i == j => T::mk(a[i][j].widen().re, im),
                _ => a[i][j],
            };
            buf[sb_slot(layout, uplo, i, j, k, lda)] = v;
        }
    }
    buf
}

/// Packed storage of the `uplo` triangle; a unit diagonal holds NaN and
/// `diag_imag` replaces the imaginary part of the stored diagonal.
pub(crate) fn store_packed<T: Num>(
    layout: CblasLayout,
    uplo: CblasUplo,
    unit: bool,
    a: &Dense<T>,
    diag_imag: Option<f64>,
) -> Vec<T> {
    let n = a.len();
    let mut buf = vec![nan::<T>(); n * (n + 1) / 2];
    for i in 0..n {
        for j in 0..n {
            if !in_tri_band(uplo, i, j, n) || (unit && i == j) {
                continue;
            }
            let v = match diag_imag {
                Some(im) if i == j => T::mk(a[i][j].widen().re, im),
                _ => a[i][j],
            };
            buf[packed_slot(layout, uplo, i, j, n)] = v;
        }
    }
    buf
}

/// Dense storage (leading dimension `lda`) of the `uplo` triangle; the other
/// triangle, the padding and a unit diagonal hold NaN, and `diag_imag`
/// replaces the imaginary part of the stored diagonal.
pub(crate) fn store_dense_triangle<T: Num>(
    layout: CblasLayout,
    uplo: CblasUplo,
    unit: bool,
    a: &Dense<T>,
    lda: usize,
    diag_imag: Option<f64>,
) -> Vec<T> {
    let n = a.len();
    let mut buf = vec![nan::<T>(); lda * n];
    for i in 0..n {
        for j in 0..n {
            if !in_tri_band(uplo, i, j, n) || (unit && i == j) {
                continue;
            }
            let v = match diag_imag {
                Some(im) if i == j => T::mk(a[i][j].widen().re, im),
                _ => a[i][j],
            };
            buf[dense_slot(layout, i, j, lda)] = v;
        }
    }
    buf
}

// =============================================================================
// Hermitian (symmetric) rank updates
// =============================================================================

/// Reference `A + alpha*x*x^H` entry (real `alpha`; diagonal real part only).
pub(crate) fn her_entry<T: Num>(
    a: &Dense<T>,
    alpha: f64,
    x: &[T],
    i: usize,
    j: usize,
) -> (Complex64, f64) {
    let (xi, xj) = (x[i].widen(), x[j].widen());
    let aij = a[i][j].widen();
    let base = if i == j {
        Complex64::new(aij.re, 0.0)
    } else {
        aij
    };
    (
        base + alpha * xi * xj.conj(),
        aij.norm() + alpha.abs() * xi.norm() * xj.norm(),
    )
}

/// Reference `A + alpha*x*y^H + conj(alpha)*y*x^H` entry.
pub(crate) fn her2_entry<T: Num>(
    a: &Dense<T>,
    alpha: T,
    x: &[T],
    y: &[T],
    i: usize,
    j: usize,
) -> (Complex64, f64) {
    let al = alpha.widen();
    let (xi, xj, yi, yj) = (x[i].widen(), x[j].widen(), y[i].widen(), y[j].widen());
    let aij = a[i][j].widen();
    let base = if i == j {
        Complex64::new(aij.re, 0.0)
    } else {
        aij
    };
    let upd = al * xi * yj.conj() + al.conj() * yi * xj.conj();
    let bound = aij.norm() + al.norm() * (xi.norm() * yj.norm() + yi.norm() * xj.norm());
    (base + upd, bound)
}

/// Checks a rank update of the stored `uplo` triangle: entry `(i, j)` must
/// match `upd(i, j)`, a diagonal imaginary part must be exactly zero, and
/// every slot outside the triangle must be bitwise untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_triangle_update<T: Num>(
    before: &[T],
    after: &[T],
    n: usize,
    uplo: CblasUplo,
    slot_of: &dyn Fn(usize, usize) -> usize,
    upd: &dyn Fn(usize, usize) -> (Complex64, f64),
    what: &str,
) {
    let mut touched = vec![false; before.len()];
    for i in 0..n {
        for j in 0..n {
            if !in_tri_band(uplo, i, j, n) {
                continue;
            }
            let s = slot_of(i, j);
            touched[s] = true;
            let (want, bound) = upd(i, j);
            let got = after[s].widen();
            assert_close(
                &[got],
                &[want],
                &[bound],
                T::EPS,
                &format!("{what} ({i},{j})"),
            );
            if i == j {
                assert_eq!(
                    got.im, 0.0,
                    "{what}: diagonal ({i},{i}) imaginary part not zeroed"
                );
            }
        }
    }
    for (s, t) in touched.iter().enumerate() {
        if !t {
            assert_eq!(
                before[s].bits(),
                after[s].bits(),
                "{what}: untouched slot {s} changed"
            );
        }
    }
}

// =============================================================================
// Tests of the helpers themselves
// =============================================================================

#[test]
fn test_issue_14_vec_offset_follows_reference_increments() {
    // Positive increments walk forward from the pointer.
    assert_eq!(vec_offset(0, 4, 2), 0);
    assert_eq!(vec_offset(3, 4, 2), 6);
    // Negative increments start at the far end and walk back to offset 0.
    assert_eq!(vec_offset(0, 4, -3), 9);
    assert_eq!(vec_offset(3, 4, -3), 0);
    for inc in INCS {
        for i in 0..5 {
            assert_eq!(vec_offset(i, 5, inc as isize), slot(i, 5, inc) as isize);
        }
    }
}

#[test]
fn test_issue_14_gather_scatter_roundtrip_with_conjugation() {
    let mut rng = Rng::new(1);
    let v: Vec<Complex64> = rng.vec(5);
    for inc in INCS {
        let buf = strided(&v, inc, nan());
        // SAFETY: `buf` holds the 5-element strided walk of `inc`.
        let got = unsafe { gather(buf.as_ptr(), 5, inc as isize, false) };
        assert_bits_eq(&v, &got, "gather");
        // SAFETY: as above.
        let got_c = unsafe { gather(buf.as_ptr(), 5, inc as isize, true) };
        let want_c: Vec<Complex64> = v.iter().map(|z| z.conj()).collect();
        assert_bits_eq(&want_c, &got_c, "gather conj");

        let mut out = strided(&[zero::<Complex64>(); 5], inc, T_FILL);
        let before = out.clone();
        // SAFETY: `out` holds the 5-element strided walk of `inc`.
        unsafe { scatter(out.as_mut_ptr(), &got_c, inc as isize, true) };
        assert_bits_eq(&v, &unstrided(&out, 5, inc), "scatter conj");
        assert_gaps_unchanged(&before, &out, 5, inc, "scatter gaps");
    }
}

/// Sentinel used for strided-buffer gaps that must stay untouched.
const T_FILL: Complex64 = Complex64::new(123.0, -456.0);

#[test]
fn test_issue_14_y_update_prologue_quick_returns() {
    let one = Complex64::new(1.0, 0.0);
    let zero_c = Complex64::new(0.0, 0.0);
    for inc in INCS {
        // alpha == 0 && beta == 1: y untouched, even non-finite / -0.0.
        let mut y = strided(&poison::<Complex64>(4), inc, T_FILL);
        let before = y.clone();
        // SAFETY: `y` holds the 4-element strided walk of `inc`.
        let more = unsafe { y_update_prologue(zero_c, one, y.as_mut_ptr(), 4, inc as isize) };
        assert!(!more);
        assert_bits_eq(&before, &y, "alpha=0 beta=1");

        // beta == 0 zero-fills NaN y without reading it; alpha != 0 continues.
        let mut y = strided(&[nan::<Complex64>(); 4], inc, T_FILL);
        let before = y.clone();
        let alpha = Complex64::new(0.5, -1.0);
        // SAFETY: as above.
        let more = unsafe { y_update_prologue(alpha, zero_c, y.as_mut_ptr(), 4, inc as isize) };
        assert!(more);
        assert_bits_eq(&[zero_c; 4], &unstrided(&y, 4, inc), "beta=0");
        assert_gaps_unchanged(&before, &y, 4, inc, "beta=0 gaps");

        // alpha == 0, beta != 1: y := beta*y, then the call is complete.
        let vals = vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)];
        let mut y = strided(&vals, inc, T_FILL);
        let beta = Complex64::new(2.0, -1.0);
        // SAFETY: `y` holds the 2-element strided walk of `inc`.
        let more = unsafe { y_update_prologue(zero_c, beta, y.as_mut_ptr(), 2, inc as isize) };
        assert!(!more);
        let want: Vec<Complex64> = vals.iter().map(|v| beta * v).collect();
        assert_bits_eq(&want, &unstrided(&y, 2, inc), "alpha=0 beta*y");

        // beta == 1 with alpha != 0: no rescaling pass (Inf * (1+0i) = NaN).
        let mut y = vec![Complex64::new(f64::INFINITY, 0.0)];
        // SAFETY: one element, unit increment.
        let more = unsafe { y_update_prologue(alpha, one, y.as_mut_ptr(), 1, 1) };
        assert!(more);
        assert_bits_eq(
            &[Complex64::new(f64::INFINITY, 0.0)],
            &y,
            "beta=1 no rescale",
        );
    }
}

#[test]
fn test_issue_14_layout_views_follow_reference_cblas() {
    use CblasLayout::{ColMajor, RowMajor};
    use CblasTranspose::{ConjTrans, NoTrans, Trans};
    use CblasUplo::{Lower, Upper};
    assert_eq!(flip_uplo(Upper), Lower);
    assert_eq!(flip_uplo(Lower), Upper);
    assert_eq!(hermitian_view(ColMajor, Upper), (Upper, false));
    assert_eq!(hermitian_view(RowMajor, Upper), (Lower, true));
    assert_eq!(transpose_view(ColMajor, ConjTrans), (ConjTrans, false));
    assert_eq!(transpose_view(RowMajor, NoTrans), (Trans, false));
    assert_eq!(transpose_view(RowMajor, Trans), (NoTrans, false));
    assert_eq!(transpose_view(RowMajor, ConjTrans), (NoTrans, true));
    assert_eq!(
        triangular_view(ColMajor, Lower, Trans),
        (Lower, Trans, false)
    );
    assert_eq!(
        triangular_view(RowMajor, Lower, NoTrans),
        (Upper, Trans, false)
    );
    assert_eq!(
        triangular_view(RowMajor, Upper, ConjTrans),
        (Lower, NoTrans, true)
    );
}

#[test]
fn test_issue_14_packed_len_is_overflow_checked() {
    assert_eq!(packed_len::<f64>(0), Some(0));
    assert_eq!(packed_len::<f64>(1), Some(1));
    assert_eq!(packed_len::<Complex64>(4), Some(10));
    // Expected values computed exactly in u128: `None` exactly when the byte
    // size of n*(n+1)/2 elements exceeds isize::MAX (which covers both the
    // element-count and the byte-count overflow on any pointer width).
    let max_bytes = isize::MAX as u128;
    for n in [5usize, 1 << 15, 1_200_000_000, 1 << 31, usize::MAX] {
        let len = n as u128 * (n as u128 + 1) / 2;
        for (size, got) in [
            (1u128, packed_len::<u8>(n)),
            (16u128, packed_len::<Complex64>(n)),
        ] {
            let want = len
                .checked_mul(size)
                .filter(|&bytes| bytes <= max_bytes)
                .map(|_| len as usize);
            assert_eq!(got, want, "n = {n}, element size = {size}");
        }
    }
}

#[test]
fn test_issue_14_storage_builders_match_cblas_definitions() {
    // Spot-check the row-major definitions on a 3x3 example:
    // row-major Upper packing is A00 A01 A02 A11 A12 A22.
    let order: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    for (s, &(i, j)) in order.iter().enumerate() {
        assert_eq!(
            packed_slot(CblasLayout::RowMajor, CblasUplo::Upper, i, j, 3),
            s
        );
        // Column-major Lower packing has the same order with (i, j) swapped.
        assert_eq!(
            packed_slot(CblasLayout::ColMajor, CblasUplo::Lower, j, i, 3),
            s
        );
    }
    // Row-major Lower packing is A00 A10 A11 A20 A21 A22.
    let order: Vec<(usize, usize)> = vec![(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)];
    for (s, &(i, j)) in order.iter().enumerate() {
        assert_eq!(
            packed_slot(CblasLayout::RowMajor, CblasUplo::Lower, i, j, 3),
            s
        );
    }
    // Row-major general band: row i holds A[i][i-kl ..= i+ku] left-aligned
    // at offset kl + j - i (kl = 1, ku = 2, lda = 4).
    assert_eq!(gb_slot(CblasLayout::RowMajor, 2, 1, 1, 2, 4), 8);
    assert_eq!(gb_slot(CblasLayout::RowMajor, 2, 4, 1, 2, 4), 11);
    // Row-major symmetric band (k = 2, lda = 3): Upper row i starts at the
    // diagonal, Lower row i ends at it.
    assert_eq!(
        sb_slot(CblasLayout::RowMajor, CblasUplo::Upper, 1, 1, 2, 3),
        3
    );
    assert_eq!(
        sb_slot(CblasLayout::RowMajor, CblasUplo::Lower, 1, 1, 2, 3),
        5
    );
    assert_eq!(dense_slot(CblasLayout::RowMajor, 1, 2, 5), 7);
    let mut rng = Rng::new(9);
    let a = general_band::<f64>(&mut rng, 3, 4, 0, 1);
    assert_eq!(a[2][0].bits(), zero::<f64>().bits());
    assert!(
        store_gb(CblasLayout::ColMajor, &a, 4, 0, 1, 2)
            .iter()
            .any(|v| v.is_nan())
    );
}
