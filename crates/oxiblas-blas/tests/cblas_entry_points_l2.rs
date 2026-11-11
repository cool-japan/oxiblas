//! Issue #14 (Level-2 part): the 48 reference-CBLAS Level-2 entry points that
//! were missing from `oxiblas_blas::cblas`.
//!
//! * banded: `cblas_{s,d,c,z}gbmv`, `cblas_{s,d}sbmv`, `cblas_{c,z}hbmv`,
//!   `cblas_{s,d,c,z}tbmv`, `cblas_{s,d,c,z}tbsv`;
//! * packed: `cblas_{s,d}spmv`, `cblas_{s,d}spr`, `cblas_{s,d}spr2`,
//!   `cblas_{c,z}hpmv`, `cblas_{c,z}hpr`, `cblas_{c,z}hpr2`,
//!   `cblas_{s,d,c,z}tpmv`, `cblas_{s,d,c,z}tpsv`;
//! * complex dense: `cblas_{c,z}geru`, `cblas_{c,z}gerc`, `cblas_{c,z}her`,
//!   `cblas_{c,z}her2`, `cblas_{c,z}trmv`, `cblas_{c,z}trsv`.
//!
//! Every symbol is called through the public `oxiblas_blas::cblas::*` path in
//! both layouts and compared with a naive dense reference evaluated in
//! `Complex64`. The operand buffers are built from the CBLAS storage
//! definitions of each layout (row-major band rows, row-wise packing, ...),
//! never from the wrapper's internal column-major transformation, so a wrong
//! row-major mapping cannot cancel out. Unused band slots, the unreferenced
//! triangle and unit diagonals hold NaN: reading them would poison the result.
//!
//! The `const _` table below coerces each symbol to the exact prototype of
//! netlib `cblas.h` (`CBLAS_INT` = `i32`, complex `void *` = `*const/*mut
//! Complex*`), so parameter-order or pointer-vs-value drift is a compile error.

use num_complex::{Complex32, Complex64};
use oxiblas_blas::cblas::*;

// =============================================================================
// Signature locks (netlib cblas.h)
// =============================================================================

type GbmvReal<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasTranspose,
    i32,
    i32,
    i32,
    i32,
    T,
    *const T,
    i32,
    *const T,
    i32,
    T,
    *mut T,
    i32,
);
type GbmvCplx<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasTranspose,
    i32,
    i32,
    i32,
    i32,
    *const T,
    *const T,
    i32,
    *const T,
    i32,
    *const T,
    *mut T,
    i32,
);
type SbmvReal<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    i32,
    i32,
    T,
    *const T,
    i32,
    *const T,
    i32,
    T,
    *mut T,
    i32,
);
type HbmvCplx<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    i32,
    i32,
    *const T,
    *const T,
    i32,
    *const T,
    i32,
    *const T,
    *mut T,
    i32,
);
type TbFn<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    CblasTranspose,
    CblasDiag,
    i32,
    i32,
    *const T,
    i32,
    *mut T,
    i32,
);
type SpmvReal<T> =
    unsafe extern "C" fn(CblasLayout, CblasUplo, i32, T, *const T, *const T, i32, T, *mut T, i32);
type HpmvCplx<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    i32,
    *const T,
    *const T,
    *const T,
    i32,
    *const T,
    *mut T,
    i32,
);
type SprReal<T> = unsafe extern "C" fn(CblasLayout, CblasUplo, i32, T, *const T, i32, *mut T);
type HprCplx<R, T> = unsafe extern "C" fn(CblasLayout, CblasUplo, i32, R, *const T, i32, *mut T);
type Spr2Real<T> =
    unsafe extern "C" fn(CblasLayout, CblasUplo, i32, T, *const T, i32, *const T, i32, *mut T);
type Hpr2Cplx<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    i32,
    *const T,
    *const T,
    i32,
    *const T,
    i32,
    *mut T,
);
type TpFn<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    CblasTranspose,
    CblasDiag,
    i32,
    *const T,
    *mut T,
    i32,
);
type GerCplx<T> = unsafe extern "C" fn(
    CblasLayout,
    i32,
    i32,
    *const T,
    *const T,
    i32,
    *const T,
    i32,
    *mut T,
    i32,
);
type HerCplx<R, T> =
    unsafe extern "C" fn(CblasLayout, CblasUplo, i32, R, *const T, i32, *mut T, i32);
type Her2Cplx<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    i32,
    *const T,
    *const T,
    i32,
    *const T,
    i32,
    *mut T,
    i32,
);
type TrFn<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    CblasTranspose,
    CblasDiag,
    i32,
    *const T,
    i32,
    *mut T,
    i32,
);

const _: GbmvReal<f32> = cblas_sgbmv;
const _: GbmvReal<f64> = cblas_dgbmv;
const _: GbmvCplx<Complex32> = cblas_cgbmv;
const _: GbmvCplx<Complex64> = cblas_zgbmv;
const _: SbmvReal<f32> = cblas_ssbmv;
const _: SbmvReal<f64> = cblas_dsbmv;
const _: HbmvCplx<Complex32> = cblas_chbmv;
const _: HbmvCplx<Complex64> = cblas_zhbmv;
const _: TbFn<f32> = cblas_stbmv;
const _: TbFn<f64> = cblas_dtbmv;
const _: TbFn<Complex32> = cblas_ctbmv;
const _: TbFn<Complex64> = cblas_ztbmv;
const _: TbFn<f32> = cblas_stbsv;
const _: TbFn<f64> = cblas_dtbsv;
const _: TbFn<Complex32> = cblas_ctbsv;
const _: TbFn<Complex64> = cblas_ztbsv;
const _: SpmvReal<f32> = cblas_sspmv;
const _: SpmvReal<f64> = cblas_dspmv;
const _: SprReal<f32> = cblas_sspr;
const _: SprReal<f64> = cblas_dspr;
const _: Spr2Real<f32> = cblas_sspr2;
const _: Spr2Real<f64> = cblas_dspr2;
const _: HpmvCplx<Complex32> = cblas_chpmv;
const _: HpmvCplx<Complex64> = cblas_zhpmv;
const _: HprCplx<f32, Complex32> = cblas_chpr;
const _: HprCplx<f64, Complex64> = cblas_zhpr;
const _: Hpr2Cplx<Complex32> = cblas_chpr2;
const _: Hpr2Cplx<Complex64> = cblas_zhpr2;
const _: TpFn<f32> = cblas_stpmv;
const _: TpFn<f64> = cblas_dtpmv;
const _: TpFn<Complex32> = cblas_ctpmv;
const _: TpFn<Complex64> = cblas_ztpmv;
const _: TpFn<f32> = cblas_stpsv;
const _: TpFn<f64> = cblas_dtpsv;
const _: TpFn<Complex32> = cblas_ctpsv;
const _: TpFn<Complex64> = cblas_ztpsv;
const _: GerCplx<Complex32> = cblas_cgeru;
const _: GerCplx<Complex64> = cblas_zgeru;
const _: GerCplx<Complex32> = cblas_cgerc;
const _: GerCplx<Complex64> = cblas_zgerc;
const _: HerCplx<f32, Complex32> = cblas_cher;
const _: HerCplx<f64, Complex64> = cblas_zher;
const _: Her2Cplx<Complex32> = cblas_cher2;
const _: Her2Cplx<Complex64> = cblas_zher2;
const _: TrFn<Complex32> = cblas_ctrmv;
const _: TrFn<Complex64> = cblas_ztrmv;
const _: TrFn<Complex32> = cblas_ctrsv;
const _: TrFn<Complex64> = cblas_ztrsv;

// =============================================================================
// Scalars, data, strided vectors
// =============================================================================

/// The four BLAS precisions, widened to `Complex64` for the references.
trait Num: Copy + std::fmt::Debug {
    const EPS: f64;
    /// Builds a value; real types drop `im`.
    fn mk(re: f64, im: f64) -> Self;
    fn widen(self) -> Complex64;
    fn conj(self) -> Self;
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
    fn conj(self) -> Self {
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
    fn conj(self) -> Self {
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
    fn conj(self) -> Self {
        Complex32::conj(&self)
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
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

const LAYOUTS: [CblasLayout; 2] = [CblasLayout::ColMajor, CblasLayout::RowMajor];
const UPLOS: [CblasUplo; 2] = [CblasUplo::Upper, CblasUplo::Lower];
const TRANSES: [CblasTranspose; 3] = [
    CblasTranspose::NoTrans,
    CblasTranspose::Trans,
    CblasTranspose::ConjTrans,
];
const DIAGS: [CblasDiag; 2] = [CblasDiag::NonUnit, CblasDiag::Unit];
/// Increment pairs: unit stride, and a mixed negative / positive stride.
const INC_PAIRS: [(i32, i32); 2] = [(1, 1), (-2, 3)];

/// Deterministic, well-scaled test value number `k` (entries in `[-1, 1)`).
fn val<T: Num>(k: usize) -> T {
    let re = ((k * 37 + 11) % 23) as f64 / 11.5 - 1.0;
    let im = ((k * 53 + 5) % 19) as f64 / 9.5 - 1.0;
    T::mk(re, im)
}

/// A diagonal entry bounded away from zero (keeps the solves well posed).
fn diag_val<T: Num>(k: usize) -> T {
    let v = val::<T>(k).widen();
    T::mk(3.0 + v.re.abs(), v.im)
}

fn nan<T: Num>() -> T {
    T::mk(f64::NAN, f64::NAN)
}

fn zero<T: Num>() -> T {
    T::mk(0.0, 0.0)
}

fn widen_all<T: Num>(v: &[T]) -> Vec<Complex64> {
    v.iter().map(|e| e.widen()).collect()
}

/// Buffer slot of logical element `i` of a length-`n` vector with increment
/// `inc` (reference BLAS: a negative increment starts at the far end).
fn slot(i: usize, n: usize, inc: i32) -> usize {
    let step = inc.unsigned_abs() as usize;
    if inc > 0 {
        i * step
    } else {
        (n - 1 - i) * step
    }
}

fn strided<T: Num>(v: &[T], inc: i32, fill: T) -> Vec<T> {
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

fn unstrided<T: Num>(buf: &[T], n: usize, inc: i32) -> Vec<T> {
    (0..n).map(|i| buf[slot(i, n, inc)]).collect()
}

fn values<T: Num>(n: usize, seed: usize) -> Vec<T> {
    (0..n).map(|i| val(seed + i)).collect()
}

// =============================================================================
// References and assertions
// =============================================================================

/// Logical dense matrix, indexed `[row][col]`.
type Dense<T> = Vec<Vec<T>>;

fn op_entry<T: Num>(a: &Dense<T>, trans: CblasTranspose, i: usize, j: usize) -> Complex64 {
    match trans {
        CblasTranspose::NoTrans => a[i][j].widen(),
        CblasTranspose::Trans => a[j][i].widen(),
        CblasTranspose::ConjTrans => a[j][i].widen().conj(),
    }
}

/// `op(A)*x` (with `A` `rows` x `cols`) and the magnitude bound `|op(A)|*|x|`.
fn op_mul<T: Num>(
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
fn mv_ref<T: Num>(
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

fn assert_close(got: &[Complex64], want: &[Complex64], bound: &[f64], eps: f64, what: &str) {
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
fn assert_solves<T: Num>(a: &Dense<T>, trans: CblasTranspose, x: &[T], b: &[T], what: &str) {
    let n = a.len();
    let (ax, bound) = op_mul(a, n, n, trans, &widen_all(x));
    let b = widen_all(b);
    let full_bound: Vec<f64> = bound.iter().zip(&b).map(|(p, q)| p + q.norm()).collect();
    assert_close(&ax, &b, &full_bound, T::EPS, what);
}

// =============================================================================
// Storage builders (CBLAS definitions for each layout)
// =============================================================================

/// `A[i][j]` slot of a general band matrix (`kl` sub-, `ku` super-diagonals).
fn gb_slot(layout: CblasLayout, i: usize, j: usize, kl: usize, ku: usize, lda: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => (ku + i - j) + j * lda,
        CblasLayout::RowMajor => i * lda + (kl + j - i),
    }
}

/// `A[i][j]` slot of a symmetric / Hermitian / triangular band matrix.
fn sb_slot(
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

/// `A[i][j]` slot of a packed triangle (`i <= j` for Upper, `i >= j` for Lower).
fn packed_slot(layout: CblasLayout, uplo: CblasUplo, i: usize, j: usize, n: usize) -> usize {
    match (layout, uplo) {
        (CblasLayout::ColMajor, CblasUplo::Upper) => i + j * (j + 1) / 2,
        (CblasLayout::ColMajor, CblasUplo::Lower) => i + j * (2 * n - j - 1) / 2,
        (CblasLayout::RowMajor, CblasUplo::Upper) => j + i * (2 * n - i - 1) / 2,
        (CblasLayout::RowMajor, CblasUplo::Lower) => j + i * (i + 1) / 2,
    }
}

fn dense_slot(layout: CblasLayout, i: usize, j: usize, lda: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => i + j * lda,
        CblasLayout::RowMajor => i * lda + j,
    }
}

/// Whether `(i, j)` lies in the `uplo` triangle within `k` diagonals of it.
fn in_tri_band(uplo: CblasUplo, i: usize, j: usize, k: usize) -> bool {
    match uplo {
        CblasUplo::Upper => i <= j && j <= i + k,
        CblasUplo::Lower => j <= i && i <= j + k,
    }
}

/// Hermitian (symmetric for real types) matrix with bandwidth `k` and a real
/// diagonal.
fn hermitian_band<T: Num>(n: usize, k: usize, seed: usize) -> Dense<T> {
    // Strictly-upper entries are generated, the lower ones mirrored.
    let upper = |i: usize, j: usize| -> T { val(seed + 7 * i + 3 * j + 1) };
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        T::mk(val::<T>(seed + i).widen().re, 0.0)
                    } else if i < j && j <= i + k {
                        upper(i, j)
                    } else if j < i && i <= j + k {
                        upper(j, i).conj()
                    } else {
                        zero()
                    }
                })
                .collect()
        })
        .collect()
}

/// Triangular matrix with bandwidth `k`; a unit diagonal is stored logically
/// as `1`.
fn tri_band<T: Num>(n: usize, k: usize, uplo: CblasUplo, diag: CblasDiag, seed: usize) -> Dense<T> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if !in_tri_band(uplo, i, j, k) {
                        zero()
                    } else if i == j {
                        if diag == CblasDiag::Unit {
                            T::mk(1.0, 0.0)
                        } else {
                            diag_val(seed + i)
                        }
                    } else {
                        val(seed + 7 * i + 3 * j + 2)
                    }
                })
                .collect()
        })
        .collect()
}

/// Band storage of a triangular / Hermitian band matrix; unused slots and a
/// unit diagonal hold NaN. `diag_imag` overwrites the imaginary part of the
/// stored diagonal (the Hermitian routines must ignore it).
#[allow(clippy::too_many_arguments)]
fn store_band<T: Num>(
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

/// Packed storage of the `uplo` triangle; a unit diagonal holds NaN.
fn store_packed<T: Num>(
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

// =============================================================================
// Family checkers (uniform safe adapters over each symbol)
// =============================================================================

type GbmvCall<T> =
    fn(CblasLayout, CblasTranspose, i32, i32, i32, i32, T, &[T], i32, &[T], i32, T, &mut [T], i32);
type HbmvCall<T> = fn(CblasLayout, CblasUplo, i32, i32, T, &[T], i32, &[T], i32, T, &mut [T], i32);
type TbCall<T> =
    fn(CblasLayout, CblasUplo, CblasTranspose, CblasDiag, i32, i32, &[T], i32, &mut [T], i32);
type HpmvCall<T> = fn(CblasLayout, CblasUplo, i32, T, &[T], &[T], i32, T, &mut [T], i32);
type HprCall<T> = fn(CblasLayout, CblasUplo, i32, f64, &[T], i32, &mut [T]);
type Hpr2Call<T> = fn(CblasLayout, CblasUplo, i32, T, &[T], i32, &[T], i32, &mut [T]);
type TpCall<T> = fn(CblasLayout, CblasUplo, CblasTranspose, CblasDiag, i32, &[T], &mut [T], i32);
type GerCall<T> = fn(CblasLayout, i32, i32, T, &[T], i32, &[T], i32, &mut [T], i32);
type HerCall<T> = fn(CblasLayout, CblasUplo, i32, f64, &[T], i32, &mut [T], i32);
type Her2Call<T> = fn(CblasLayout, CblasUplo, i32, T, &[T], i32, &[T], i32, &mut [T], i32);
type TrCall<T> =
    fn(CblasLayout, CblasUplo, CblasTranspose, CblasDiag, i32, &[T], i32, &mut [T], i32);

fn check_gbmv<T: Num>(name: &str, call: GbmvCall<T>) {
    // Rectangular both ways, kl != ku, and a band wider than the matrix.
    for &(m, n, kl, ku) in &[(4usize, 3usize, 1usize, 2usize), (3, 5, 2, 0), (2, 3, 3, 4)] {
        let a: Dense<T> = (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if j + kl >= i && j <= i + ku {
                            val(i * 7 + j * 3 + 1)
                        } else {
                            zero()
                        }
                    })
                    .collect()
            })
            .collect();
        let lda = kl + ku + 2;
        for layout in LAYOUTS {
            let mut buf = vec![
                nan::<T>();
                lda * if layout == CblasLayout::ColMajor {
                    n
                } else {
                    m
                }
            ];
            for i in 0..m {
                for j in 0..n {
                    if j + kl >= i && j <= i + ku {
                        buf[gb_slot(layout, i, j, kl, ku, lda)] = a[i][j];
                    }
                }
            }
            for trans in TRANSES {
                for (incx, incy) in INC_PAIRS {
                    let (lenx, leny) = if trans == CblasTranspose::NoTrans {
                        (n, m)
                    } else {
                        (m, n)
                    };
                    let x = values::<T>(lenx, 100);
                    let y = values::<T>(leny, 200);
                    let (alpha, beta) = (val::<T>(300), val::<T>(301));
                    let xb = strided(&x, incx, nan());
                    let mut yb = strided(&y, incy, T::mk(9.0, 9.0));
                    call(
                        layout, trans, m as i32, n as i32, kl as i32, ku as i32, alpha, &buf,
                        lda as i32, &xb, incx, beta, &mut yb, incy,
                    );
                    let (want, bound) = mv_ref(&a, m, n, trans, alpha, &x, beta, &y);
                    let what = format!(
                        "{name} {layout:?} {trans:?} m={m} n={n} kl={kl} ku={ku} incx={incx} incy={incy}"
                    );
                    assert_close(
                        &widen_all(&unstrided(&yb, leny, incy)),
                        &want,
                        &bound,
                        T::EPS,
                        &what,
                    );
                }
            }
        }
    }
}

fn check_hbmv<T: Num>(name: &str, call: HbmvCall<T>) {
    for &(n, k) in &[(4usize, 1usize), (5, 2), (3, 4)] {
        let a = hermitian_band::<T>(n, k, 10 * n + k);
        let lda = k + 2;
        for layout in LAYOUTS {
            for uplo in UPLOS {
                let buf = store_band(layout, uplo, false, &a, k, lda, Some(0.625));
                for (incx, incy) in INC_PAIRS {
                    let x = values::<T>(n, 40);
                    let y = values::<T>(n, 60);
                    let (alpha, beta) = (val::<T>(80), val::<T>(81));
                    let xb = strided(&x, incx, nan());
                    let mut yb = strided(&y, incy, T::mk(9.0, 9.0));
                    call(
                        layout, uplo, n as i32, k as i32, alpha, &buf, lda as i32, &xb, incx, beta,
                        &mut yb, incy,
                    );
                    let (want, bound) =
                        mv_ref(&a, n, n, CblasTranspose::NoTrans, alpha, &x, beta, &y);
                    let what =
                        format!("{name} {layout:?} {uplo:?} n={n} k={k} incx={incx} incy={incy}");
                    assert_close(
                        &widen_all(&unstrided(&yb, n, incy)),
                        &want,
                        &bound,
                        T::EPS,
                        &what,
                    );
                }
            }
        }
    }
}

fn check_tb<T: Num>(name: &str, call: TbCall<T>, solve: bool) {
    for &(n, k) in &[(4usize, 1usize), (5, 2), (3, 4)] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for diag in DIAGS {
                        let a = tri_band::<T>(n, k, uplo, diag, 13 * n + k);
                        let lda = k + 2;
                        let unit = diag == CblasDiag::Unit;
                        let buf = store_band(layout, uplo, unit, &a, k, lda, None);
                        for inc in [1, -2] {
                            let x = values::<T>(n, 90);
                            let mut xb = strided(&x, inc, T::mk(7.0, 7.0));
                            call(
                                layout, uplo, trans, diag, n as i32, k as i32, &buf, lda as i32,
                                &mut xb, inc,
                            );
                            let got = unstrided(&xb, n, inc);
                            let what = format!(
                                "{name} {layout:?} {uplo:?} {trans:?} {diag:?} n={n} k={k} inc={inc}"
                            );
                            if solve {
                                assert_solves(&a, trans, &got, &x, &what);
                            } else {
                                let (want, bound) = op_mul(&a, n, n, trans, &widen_all(&x));
                                assert_close(&widen_all(&got), &want, &bound, T::EPS, &what);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn check_hpmv<T: Num>(name: &str, call: HpmvCall<T>) {
    for n in [1usize, 4, 5] {
        let a = hermitian_band::<T>(n, n, 20 + n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                let ap = store_packed(layout, uplo, false, &a, Some(0.625));
                for (incx, incy) in INC_PAIRS {
                    let x = values::<T>(n, 140);
                    let y = values::<T>(n, 160);
                    let (alpha, beta) = (val::<T>(180), val::<T>(181));
                    let xb = strided(&x, incx, nan());
                    let mut yb = strided(&y, incy, T::mk(9.0, 9.0));
                    call(
                        layout, uplo, n as i32, alpha, &ap, &xb, incx, beta, &mut yb, incy,
                    );
                    let (want, bound) =
                        mv_ref(&a, n, n, CblasTranspose::NoTrans, alpha, &x, beta, &y);
                    let what = format!("{name} {layout:?} {uplo:?} n={n} incx={incx} incy={incy}");
                    assert_close(
                        &widen_all(&unstrided(&yb, n, incy)),
                        &want,
                        &bound,
                        T::EPS,
                        &what,
                    );
                }
            }
        }
    }
}

/// Checks a Hermitian (symmetric) rank update of the stored triangle: entry
/// `(i, j)` must equal `upd(i, j)`, the diagonal imaginary part must be exactly
/// zero, and every slot outside the triangle must be bitwise untouched.
#[allow(clippy::too_many_arguments)]
fn check_triangle_update<T: Num>(
    before: &[T],
    after: &[T],
    n: usize,
    uplo: CblasUplo,
    slot_of: &dyn Fn(usize, usize) -> usize,
    upd: &dyn Fn(usize, usize) -> (Complex64, f64),
    eps: f64,
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
            assert_close(&[got], &[want], &[bound], eps, &format!("{what} ({i},{j})"));
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

/// Reference `A + alpha*x*x^H` entry (real `alpha`; diagonal real part only).
fn her_entry<T: Num>(a: &Dense<T>, alpha: f64, x: &[T], i: usize, j: usize) -> (Complex64, f64) {
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
fn her2_entry<T: Num>(
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

fn check_hpr<T: Num>(name: &str, call: HprCall<T>) {
    let alpha = 0.75;
    for n in [1usize, 4, 5] {
        let a = hermitian_band::<T>(n, n, 30 + n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for (incx, _) in INC_PAIRS {
                    let before = store_packed(layout, uplo, false, &a, Some(0.625));
                    let mut ap = before.clone();
                    let x = values::<T>(n, 240);
                    let xb = strided(&x, incx, nan());
                    call(layout, uplo, n as i32, alpha, &xb, incx, &mut ap);
                    let what = format!("{name} {layout:?} {uplo:?} n={n} incx={incx}");
                    check_triangle_update(
                        &before,
                        &ap,
                        n,
                        uplo,
                        &|i, j| packed_slot(layout, uplo, i, j, n),
                        &|i, j| her_entry(&a, alpha, &x, i, j),
                        T::EPS,
                        &what,
                    );
                }
            }
        }
    }
}

fn check_hpr2<T: Num>(name: &str, call: Hpr2Call<T>) {
    for n in [1usize, 4, 5] {
        let a = hermitian_band::<T>(n, n, 40 + n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for (incx, incy) in INC_PAIRS {
                    let before = store_packed(layout, uplo, false, &a, Some(0.625));
                    let mut ap = before.clone();
                    let x = values::<T>(n, 340);
                    let y = values::<T>(n, 360);
                    let alpha = val::<T>(380);
                    let xb = strided(&x, incx, nan());
                    let yb = strided(&y, incy, nan());
                    call(layout, uplo, n as i32, alpha, &xb, incx, &yb, incy, &mut ap);
                    let what = format!("{name} {layout:?} {uplo:?} n={n} incx={incx} incy={incy}");
                    check_triangle_update(
                        &before,
                        &ap,
                        n,
                        uplo,
                        &|i, j| packed_slot(layout, uplo, i, j, n),
                        &|i, j| her2_entry(&a, alpha, &x, &y, i, j),
                        T::EPS,
                        &what,
                    );
                }
            }
        }
    }
}

fn check_tp<T: Num>(name: &str, call: TpCall<T>, solve: bool) {
    for n in [1usize, 4, 5] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for diag in DIAGS {
                        let a = tri_band::<T>(n, n, uplo, diag, 17 * n);
                        let unit = diag == CblasDiag::Unit;
                        let ap = store_packed(layout, uplo, unit, &a, None);
                        for inc in [1, -2] {
                            let x = values::<T>(n, 190);
                            let mut xb = strided(&x, inc, T::mk(7.0, 7.0));
                            call(layout, uplo, trans, diag, n as i32, &ap, &mut xb, inc);
                            let got = unstrided(&xb, n, inc);
                            let what = format!(
                                "{name} {layout:?} {uplo:?} {trans:?} {diag:?} n={n} inc={inc}"
                            );
                            if solve {
                                assert_solves(&a, trans, &got, &x, &what);
                            } else {
                                let (want, bound) = op_mul(&a, n, n, trans, &widen_all(&x));
                                assert_close(&widen_all(&got), &want, &bound, T::EPS, &what);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn check_ger<T: Num>(name: &str, call: GerCall<T>, conj_y: bool) {
    for &(m, n) in &[(3usize, 4usize), (4, 2)] {
        let a: Dense<T> = (0..m)
            .map(|i| (0..n).map(|j| val(i * 5 + j * 11 + 3)).collect())
            .collect();
        for layout in LAYOUTS {
            let lda = if layout == CblasLayout::ColMajor {
                m + 1
            } else {
                n + 1
            };
            let rows_stored = if layout == CblasLayout::ColMajor {
                n
            } else {
                m
            };
            let mut before = vec![nan::<T>(); lda * rows_stored];
            for i in 0..m {
                for j in 0..n {
                    before[dense_slot(layout, i, j, lda)] = a[i][j];
                }
            }
            for (incx, incy) in INC_PAIRS {
                let mut buf = before.clone();
                let x = values::<T>(m, 400);
                let y = values::<T>(n, 420);
                let alpha = val::<T>(440);
                let xb = strided(&x, incx, nan());
                let yb = strided(&y, incy, nan());
                call(
                    layout, m as i32, n as i32, alpha, &xb, incx, &yb, incy, &mut buf, lda as i32,
                );
                let what = format!("{name} {layout:?} m={m} n={n} incx={incx} incy={incy}");
                let mut touched = vec![false; buf.len()];
                for i in 0..m {
                    for j in 0..n {
                        let s = dense_slot(layout, i, j, lda);
                        touched[s] = true;
                        let yj = if conj_y {
                            y[j].widen().conj()
                        } else {
                            y[j].widen()
                        };
                        let (al, xi) = (alpha.widen(), x[i].widen());
                        let want = a[i][j].widen() + al * xi * yj;
                        let bound = a[i][j].widen().norm() + al.norm() * xi.norm() * yj.norm();
                        assert_close(&[buf[s].widen()], &[want], &[bound], T::EPS, &what);
                    }
                }
                for (s, t) in touched.iter().enumerate() {
                    if !t {
                        assert_eq!(
                            before[s].bits(),
                            buf[s].bits(),
                            "{what}: padding slot {s} changed"
                        );
                    }
                }
            }
        }
    }
}

/// Dense storage of the `uplo` triangle with leading dimension `n + 1`; the
/// other triangle and the padding hold NaN.
fn store_dense_triangle<T: Num>(
    layout: CblasLayout,
    uplo: CblasUplo,
    unit: bool,
    a: &Dense<T>,
    diag_imag: Option<f64>,
) -> (Vec<T>, usize) {
    let n = a.len();
    let lda = n + 1;
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
    (buf, lda)
}

fn check_her<T: Num>(name: &str, call: HerCall<T>) {
    let alpha = -1.25;
    for n in [1usize, 3, 4] {
        let a = hermitian_band::<T>(n, n, 50 + n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for (incx, _) in INC_PAIRS {
                    let (before, lda) = store_dense_triangle(layout, uplo, false, &a, Some(0.625));
                    let mut buf = before.clone();
                    let x = values::<T>(n, 500);
                    let xb = strided(&x, incx, nan());
                    call(
                        layout, uplo, n as i32, alpha, &xb, incx, &mut buf, lda as i32,
                    );
                    let what = format!("{name} {layout:?} {uplo:?} n={n} incx={incx}");
                    check_triangle_update(
                        &before,
                        &buf,
                        n,
                        uplo,
                        &|i, j| dense_slot(layout, i, j, lda),
                        &|i, j| her_entry(&a, alpha, &x, i, j),
                        T::EPS,
                        &what,
                    );
                }
            }
        }
    }
}

fn check_her2<T: Num>(name: &str, call: Her2Call<T>) {
    for n in [1usize, 3, 4] {
        let a = hermitian_band::<T>(n, n, 60 + n);
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for (incx, incy) in INC_PAIRS {
                    let (before, lda) = store_dense_triangle(layout, uplo, false, &a, Some(0.625));
                    let mut buf = before.clone();
                    let x = values::<T>(n, 600);
                    let y = values::<T>(n, 620);
                    let alpha = val::<T>(640);
                    let xb = strided(&x, incx, nan());
                    let yb = strided(&y, incy, nan());
                    call(
                        layout, uplo, n as i32, alpha, &xb, incx, &yb, incy, &mut buf, lda as i32,
                    );
                    let what = format!("{name} {layout:?} {uplo:?} n={n} incx={incx} incy={incy}");
                    check_triangle_update(
                        &before,
                        &buf,
                        n,
                        uplo,
                        &|i, j| dense_slot(layout, i, j, lda),
                        &|i, j| her2_entry(&a, alpha, &x, &y, i, j),
                        T::EPS,
                        &what,
                    );
                }
            }
        }
    }
}

fn check_tr<T: Num>(name: &str, call: TrCall<T>, solve: bool) {
    for n in [1usize, 3, 5] {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in TRANSES {
                    for diag in DIAGS {
                        let a = tri_band::<T>(n, n, uplo, diag, 19 * n);
                        let unit = diag == CblasDiag::Unit;
                        let (buf, lda) = store_dense_triangle(layout, uplo, unit, &a, None);
                        for inc in [1, -2] {
                            let x = values::<T>(n, 700);
                            let mut xb = strided(&x, inc, T::mk(7.0, 7.0));
                            call(
                                layout, uplo, trans, diag, n as i32, &buf, lda as i32, &mut xb, inc,
                            );
                            let got = unstrided(&xb, n, inc);
                            let what = format!(
                                "{name} {layout:?} {uplo:?} {trans:?} {diag:?} n={n} inc={inc}"
                            );
                            if solve {
                                assert_solves(&a, trans, &got, &x, &what);
                            } else {
                                let (want, bound) = op_mul(&a, n, n, trans, &widen_all(&x));
                                assert_close(&widen_all(&got), &want, &bound, T::EPS, &what);
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// banded.rs — gbmv / sbmv / hbmv / tbmv / tbsv
// =============================================================================

#[test]
fn test_issue_14_sgbmv_both_layouts() {
    check_gbmv::<f32>(
        "cblas_sgbmv",
        |l, t, m, n, kl, ku, al, a, lda, x, ix, be, y, iy| {
            // SAFETY: `check_gbmv` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_sgbmv(
                    l,
                    t,
                    m,
                    n,
                    kl,
                    ku,
                    al,
                    a.as_ptr(),
                    lda,
                    x.as_ptr(),
                    ix,
                    be,
                    y.as_mut_ptr(),
                    iy,
                )
            }
        },
    );
}

#[test]
fn test_issue_14_dgbmv_both_layouts() {
    check_gbmv::<f64>(
        "cblas_dgbmv",
        |l, t, m, n, kl, ku, al, a, lda, x, ix, be, y, iy| {
            // SAFETY: `check_gbmv` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_dgbmv(
                    l,
                    t,
                    m,
                    n,
                    kl,
                    ku,
                    al,
                    a.as_ptr(),
                    lda,
                    x.as_ptr(),
                    ix,
                    be,
                    y.as_mut_ptr(),
                    iy,
                )
            }
        },
    );
}

#[test]
fn test_issue_14_cgbmv_both_layouts() {
    check_gbmv::<Complex32>(
        "cblas_cgbmv",
        |l, t, m, n, kl, ku, al, a, lda, x, ix, be, y, iy| {
            // SAFETY: `check_gbmv` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_cgbmv(
                    l,
                    t,
                    m,
                    n,
                    kl,
                    ku,
                    &al,
                    a.as_ptr(),
                    lda,
                    x.as_ptr(),
                    ix,
                    &be,
                    y.as_mut_ptr(),
                    iy,
                )
            }
        },
    );
}

#[test]
fn test_issue_14_zgbmv_both_layouts() {
    check_gbmv::<Complex64>(
        "cblas_zgbmv",
        |l, t, m, n, kl, ku, al, a, lda, x, ix, be, y, iy| {
            // SAFETY: `check_gbmv` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_zgbmv(
                    l,
                    t,
                    m,
                    n,
                    kl,
                    ku,
                    &al,
                    a.as_ptr(),
                    lda,
                    x.as_ptr(),
                    ix,
                    &be,
                    y.as_mut_ptr(),
                    iy,
                )
            }
        },
    );
}

#[test]
fn test_issue_14_ssbmv_both_layouts() {
    check_hbmv::<f32>("cblas_ssbmv", |l, u, n, k, al, a, lda, x, ix, be, y, iy| {
        // SAFETY: `check_hbmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_ssbmv(
                l,
                u,
                n,
                k,
                al,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                ix,
                be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_dsbmv_both_layouts() {
    check_hbmv::<f64>("cblas_dsbmv", |l, u, n, k, al, a, lda, x, ix, be, y, iy| {
        // SAFETY: `check_hbmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_dsbmv(
                l,
                u,
                n,
                k,
                al,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                ix,
                be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_chbmv_both_layouts() {
    check_hbmv::<Complex32>("cblas_chbmv", |l, u, n, k, al, a, lda, x, ix, be, y, iy| {
        // SAFETY: `check_hbmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_chbmv(
                l,
                u,
                n,
                k,
                &al,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                ix,
                &be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_zhbmv_both_layouts() {
    check_hbmv::<Complex64>("cblas_zhbmv", |l, u, n, k, al, a, lda, x, ix, be, y, iy| {
        // SAFETY: `check_hbmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_zhbmv(
                l,
                u,
                n,
                k,
                &al,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                ix,
                &be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_stbmv_both_layouts() {
    check_tb::<f32>(
        "cblas_stbmv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_stbmv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_dtbmv_both_layouts() {
    check_tb::<f64>(
        "cblas_dtbmv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_dtbmv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_ctbmv_both_layouts() {
    check_tb::<Complex32>(
        "cblas_ctbmv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_ctbmv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_ztbmv_both_layouts() {
    check_tb::<Complex64>(
        "cblas_ztbmv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_ztbmv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_stbsv_both_layouts() {
    check_tb::<f32>(
        "cblas_stbsv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_stbsv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        true,
    );
}

#[test]
fn test_issue_14_dtbsv_both_layouts() {
    check_tb::<f64>(
        "cblas_dtbsv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_dtbsv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        true,
    );
}

#[test]
fn test_issue_14_ctbsv_both_layouts() {
    check_tb::<Complex32>(
        "cblas_ctbsv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_ctbsv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        true,
    );
}

#[test]
fn test_issue_14_ztbsv_both_layouts() {
    check_tb::<Complex64>(
        "cblas_ztbsv",
        |l, u, t, d, n, k, a, lda, x, ix| {
            // SAFETY: `check_tb` sizes every buffer for exactly these arguments.
            unsafe { cblas_ztbsv(l, u, t, d, n, k, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        true,
    );
}

// =============================================================================
// packed.rs — spmv / spr / spr2 / hpmv / hpr / hpr2 / tpmv / tpsv
// =============================================================================

#[test]
fn test_issue_14_sspmv_both_layouts() {
    check_hpmv::<f32>("cblas_sspmv", |l, u, n, al, ap, x, ix, be, y, iy| {
        // SAFETY: `check_hpmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_sspmv(
                l,
                u,
                n,
                al,
                ap.as_ptr(),
                x.as_ptr(),
                ix,
                be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_dspmv_both_layouts() {
    check_hpmv::<f64>("cblas_dspmv", |l, u, n, al, ap, x, ix, be, y, iy| {
        // SAFETY: `check_hpmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_dspmv(
                l,
                u,
                n,
                al,
                ap.as_ptr(),
                x.as_ptr(),
                ix,
                be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_chpmv_both_layouts() {
    check_hpmv::<Complex32>("cblas_chpmv", |l, u, n, al, ap, x, ix, be, y, iy| {
        // SAFETY: `check_hpmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_chpmv(
                l,
                u,
                n,
                &al,
                ap.as_ptr(),
                x.as_ptr(),
                ix,
                &be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_zhpmv_both_layouts() {
    check_hpmv::<Complex64>("cblas_zhpmv", |l, u, n, al, ap, x, ix, be, y, iy| {
        // SAFETY: `check_hpmv` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_zhpmv(
                l,
                u,
                n,
                &al,
                ap.as_ptr(),
                x.as_ptr(),
                ix,
                &be,
                y.as_mut_ptr(),
                iy,
            )
        }
    });
}

#[test]
fn test_issue_14_sspr_both_layouts() {
    check_hpr::<f32>("cblas_sspr", |l, u, n, al, x, ix, ap| {
        // SAFETY: `check_hpr` sizes every buffer for exactly these arguments.
        unsafe { cblas_sspr(l, u, n, al as f32, x.as_ptr(), ix, ap.as_mut_ptr()) }
    });
}

#[test]
fn test_issue_14_dspr_both_layouts() {
    check_hpr::<f64>("cblas_dspr", |l, u, n, al, x, ix, ap| {
        // SAFETY: `check_hpr` sizes every buffer for exactly these arguments.
        unsafe { cblas_dspr(l, u, n, al, x.as_ptr(), ix, ap.as_mut_ptr()) }
    });
}

#[test]
fn test_issue_14_chpr_both_layouts() {
    check_hpr::<Complex32>("cblas_chpr", |l, u, n, al, x, ix, ap| {
        // SAFETY: `check_hpr` sizes every buffer for exactly these arguments.
        unsafe { cblas_chpr(l, u, n, al as f32, x.as_ptr(), ix, ap.as_mut_ptr()) }
    });
}

#[test]
fn test_issue_14_zhpr_both_layouts() {
    check_hpr::<Complex64>("cblas_zhpr", |l, u, n, al, x, ix, ap| {
        // SAFETY: `check_hpr` sizes every buffer for exactly these arguments.
        unsafe { cblas_zhpr(l, u, n, al, x.as_ptr(), ix, ap.as_mut_ptr()) }
    });
}

#[test]
fn test_issue_14_sspr2_both_layouts() {
    check_hpr2::<f32>("cblas_sspr2", |l, u, n, al, x, ix, y, iy, ap| {
        // SAFETY: `check_hpr2` sizes every buffer for exactly these arguments.
        unsafe { cblas_sspr2(l, u, n, al, x.as_ptr(), ix, y.as_ptr(), iy, ap.as_mut_ptr()) }
    });
}

#[test]
fn test_issue_14_dspr2_both_layouts() {
    check_hpr2::<f64>("cblas_dspr2", |l, u, n, al, x, ix, y, iy, ap| {
        // SAFETY: `check_hpr2` sizes every buffer for exactly these arguments.
        unsafe { cblas_dspr2(l, u, n, al, x.as_ptr(), ix, y.as_ptr(), iy, ap.as_mut_ptr()) }
    });
}

#[test]
fn test_issue_14_chpr2_both_layouts() {
    check_hpr2::<Complex32>("cblas_chpr2", |l, u, n, al, x, ix, y, iy, ap| {
        // SAFETY: `check_hpr2` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_chpr2(
                l,
                u,
                n,
                &al,
                x.as_ptr(),
                ix,
                y.as_ptr(),
                iy,
                ap.as_mut_ptr(),
            )
        }
    });
}

#[test]
fn test_issue_14_zhpr2_both_layouts() {
    check_hpr2::<Complex64>("cblas_zhpr2", |l, u, n, al, x, ix, y, iy, ap| {
        // SAFETY: `check_hpr2` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_zhpr2(
                l,
                u,
                n,
                &al,
                x.as_ptr(),
                ix,
                y.as_ptr(),
                iy,
                ap.as_mut_ptr(),
            )
        }
    });
}

#[test]
fn test_issue_14_stpmv_both_layouts() {
    check_tp::<f32>(
        "cblas_stpmv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_stpmv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_dtpmv_both_layouts() {
    check_tp::<f64>(
        "cblas_dtpmv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_dtpmv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_ctpmv_both_layouts() {
    check_tp::<Complex32>(
        "cblas_ctpmv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_ctpmv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_ztpmv_both_layouts() {
    check_tp::<Complex64>(
        "cblas_ztpmv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_ztpmv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_stpsv_both_layouts() {
    check_tp::<f32>(
        "cblas_stpsv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_stpsv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        true,
    );
}

#[test]
fn test_issue_14_dtpsv_both_layouts() {
    check_tp::<f64>(
        "cblas_dtpsv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_dtpsv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        true,
    );
}

#[test]
fn test_issue_14_ctpsv_both_layouts() {
    check_tp::<Complex32>(
        "cblas_ctpsv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_ctpsv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        true,
    );
}

#[test]
fn test_issue_14_ztpsv_both_layouts() {
    check_tp::<Complex64>(
        "cblas_ztpsv",
        |l, u, t, d, n, ap, x, ix| {
            // SAFETY: `check_tp` sizes every buffer for exactly these arguments.
            unsafe { cblas_ztpsv(l, u, t, d, n, ap.as_ptr(), x.as_mut_ptr(), ix) }
        },
        true,
    );
}

// =============================================================================
// complex_level2_ext.rs — geru / gerc / her / her2 / trmv / trsv
// =============================================================================

#[test]
fn test_issue_14_cgeru_both_layouts() {
    check_ger::<Complex32>(
        "cblas_cgeru",
        |l, m, n, al, x, ix, y, iy, a, lda| {
            // SAFETY: `check_ger` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_cgeru(
                    l,
                    m,
                    n,
                    &al,
                    x.as_ptr(),
                    ix,
                    y.as_ptr(),
                    iy,
                    a.as_mut_ptr(),
                    lda,
                )
            }
        },
        false,
    );
}

#[test]
fn test_issue_14_zgeru_both_layouts() {
    check_ger::<Complex64>(
        "cblas_zgeru",
        |l, m, n, al, x, ix, y, iy, a, lda| {
            // SAFETY: `check_ger` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_zgeru(
                    l,
                    m,
                    n,
                    &al,
                    x.as_ptr(),
                    ix,
                    y.as_ptr(),
                    iy,
                    a.as_mut_ptr(),
                    lda,
                )
            }
        },
        false,
    );
}

#[test]
fn test_issue_14_cgerc_both_layouts() {
    check_ger::<Complex32>(
        "cblas_cgerc",
        |l, m, n, al, x, ix, y, iy, a, lda| {
            // SAFETY: `check_ger` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_cgerc(
                    l,
                    m,
                    n,
                    &al,
                    x.as_ptr(),
                    ix,
                    y.as_ptr(),
                    iy,
                    a.as_mut_ptr(),
                    lda,
                )
            }
        },
        true,
    );
}

#[test]
fn test_issue_14_zgerc_both_layouts() {
    check_ger::<Complex64>(
        "cblas_zgerc",
        |l, m, n, al, x, ix, y, iy, a, lda| {
            // SAFETY: `check_ger` sizes every buffer for exactly these arguments.
            unsafe {
                cblas_zgerc(
                    l,
                    m,
                    n,
                    &al,
                    x.as_ptr(),
                    ix,
                    y.as_ptr(),
                    iy,
                    a.as_mut_ptr(),
                    lda,
                )
            }
        },
        true,
    );
}

#[test]
fn test_issue_14_cher_both_layouts() {
    check_her::<Complex32>("cblas_cher", |l, u, n, al, x, ix, a, lda| {
        // SAFETY: `check_her` sizes every buffer for exactly these arguments.
        unsafe { cblas_cher(l, u, n, al as f32, x.as_ptr(), ix, a.as_mut_ptr(), lda) }
    });
}

#[test]
fn test_issue_14_zher_both_layouts() {
    check_her::<Complex64>("cblas_zher", |l, u, n, al, x, ix, a, lda| {
        // SAFETY: `check_her` sizes every buffer for exactly these arguments.
        unsafe { cblas_zher(l, u, n, al, x.as_ptr(), ix, a.as_mut_ptr(), lda) }
    });
}

#[test]
fn test_issue_14_cher2_both_layouts() {
    check_her2::<Complex32>("cblas_cher2", |l, u, n, al, x, ix, y, iy, a, lda| {
        // SAFETY: `check_her2` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_cher2(
                l,
                u,
                n,
                &al,
                x.as_ptr(),
                ix,
                y.as_ptr(),
                iy,
                a.as_mut_ptr(),
                lda,
            )
        }
    });
}

#[test]
fn test_issue_14_zher2_both_layouts() {
    check_her2::<Complex64>("cblas_zher2", |l, u, n, al, x, ix, y, iy, a, lda| {
        // SAFETY: `check_her2` sizes every buffer for exactly these arguments.
        unsafe {
            cblas_zher2(
                l,
                u,
                n,
                &al,
                x.as_ptr(),
                ix,
                y.as_ptr(),
                iy,
                a.as_mut_ptr(),
                lda,
            )
        }
    });
}

#[test]
fn test_issue_14_ctrmv_both_layouts() {
    check_tr::<Complex32>(
        "cblas_ctrmv",
        |l, u, t, d, n, a, lda, x, ix| {
            // SAFETY: `check_tr` sizes every buffer for exactly these arguments.
            unsafe { cblas_ctrmv(l, u, t, d, n, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_ztrmv_both_layouts() {
    check_tr::<Complex64>(
        "cblas_ztrmv",
        |l, u, t, d, n, a, lda, x, ix| {
            // SAFETY: `check_tr` sizes every buffer for exactly these arguments.
            unsafe { cblas_ztrmv(l, u, t, d, n, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        false,
    );
}

#[test]
fn test_issue_14_ctrsv_both_layouts() {
    check_tr::<Complex32>(
        "cblas_ctrsv",
        |l, u, t, d, n, a, lda, x, ix| {
            // SAFETY: `check_tr` sizes every buffer for exactly these arguments.
            unsafe { cblas_ctrsv(l, u, t, d, n, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        true,
    );
}

#[test]
fn test_issue_14_ztrsv_both_layouts() {
    check_tr::<Complex64>(
        "cblas_ztrsv",
        |l, u, t, d, n, a, lda, x, ix| {
            // SAFETY: `check_tr` sizes every buffer for exactly these arguments.
            unsafe { cblas_ztrsv(l, u, t, d, n, a.as_ptr(), lda, x.as_mut_ptr(), ix) }
        },
        true,
    );
}
