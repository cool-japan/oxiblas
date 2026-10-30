//! Unit tests for the CBLAS triangular / symmetric Level-3 wrappers
//! (`cblas_{d,s}{trsm,trmm,syrk,syr2k,symm}`).

use super::*;
use crate::cblas::{cblas_daxpy, cblas_ddot, cblas_dgemm, cblas_dnrm2, cblas_zdotc_sub};
use num_complex::Complex64;

#[test]
fn test_cblas_ddot() {
    let x = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0f64, 3.0, 4.0, 5.0, 6.0];

    // SAFETY: `x` and `y` are local arrays of 5 `f64` each, matching `n = 5` with
    // unit increments.
    unsafe {
        let result = cblas_ddot(5, x.as_ptr(), 1, y.as_ptr(), 1);
        // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        assert!((result - 70.0).abs() < 1e-10);
    }
}

#[test]
fn test_cblas_dnrm2() {
    let x = [3.0f64, 4.0];

    // SAFETY: `x` is a local array of 2 `f64`, matching `n = 2` with `incx = 1`.
    unsafe {
        let result = cblas_dnrm2(2, x.as_ptr(), 1);
        // sqrt(9 + 16) = sqrt(25) = 5
        assert!((result - 5.0).abs() < 1e-10);
    }
}

#[test]
fn test_cblas_daxpy() {
    let x = [1.0f64, 2.0, 3.0];
    let mut y = [4.0f64, 5.0, 6.0];

    // SAFETY: `x` and `y` are local arrays of 3 `f64` each (`n = 3`, unit
    // increments); `y` is exclusively borrowed and does not alias `x`.
    unsafe {
        cblas_daxpy(3, 2.0, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        // y = 2*x + y = [2+4, 4+5, 6+6] = [6, 9, 12]
        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 9.0).abs() < 1e-10);
        assert!((y[2] - 12.0).abs() < 1e-10);
    }
}

#[test]
fn test_cblas_dgemm_colmajor() {
    // A = [[1, 2], [3, 4]] in column-major = [1, 3, 2, 4]
    // B = [[5, 6], [7, 8]] in column-major = [5, 7, 6, 8]
    // C = A * B = [[19, 22], [43, 50]] in column-major = [19, 43, 22, 50]
    let a = [1.0f64, 3.0, 2.0, 4.0];
    let b = [5.0f64, 7.0, 6.0, 8.0];
    let mut c = [0.0f64; 4];

    // SAFETY: `a`, `b` and `c` are local 2x2 column-major buffers with
    // `lda = ldb = ldc = 2`; `c` is exclusively borrowed and aliases neither input.
    unsafe {
        cblas_dgemm(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );

        assert!((c[0] - 19.0).abs() < 1e-10);
        assert!((c[1] - 43.0).abs() < 1e-10);
        assert!((c[2] - 22.0).abs() < 1e-10);
        assert!((c[3] - 50.0).abs() < 1e-10);
    }
}

#[test]
fn test_cblas_dgemm_with_beta() {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [1.0f64; 4];
    let mut c = [1.0f64; 4];

    // SAFETY: `a`, `b` and `c` are local 2x2 column-major buffers with
    // `lda = ldb = ldc = 2`; `c` is exclusively borrowed and aliases neither input.
    unsafe {
        cblas_dgemm(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            2.0, // beta = 2
            c.as_mut_ptr(),
            2,
        );

        // C = 1.0 * A * B + 2.0 * C_initial
        // A * B = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]
        // C = [[4, 4], [6, 6]] + [[2, 2], [2, 2]] = [[6, 6], [8, 8]]
        assert!((c[0] - 6.0).abs() < 1e-10);
        assert!((c[1] - 8.0).abs() < 1e-10);
    }
}

#[test]
fn test_cblas_zdotc() {
    let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let y = [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
    let mut result = Complex64::new(0.0, 0.0);

    // SAFETY: `x` and `y` are local arrays of 2 `Complex64` (`n = 2`, unit
    // increments) and `result` is a valid, exclusively borrowed output.
    unsafe {
        cblas_zdotc_sub(2, x.as_ptr(), 1, y.as_ptr(), 1, &mut result);
        // conj(x[0]) * y[0] + conj(x[1]) * y[1]
        // = (1-2i)(5+6i) + (3-4i)(7+8i)
        // = (5+6i-10i+12) + (21+24i-28i+32)
        // = (17-4i) + (53-4i)
        // = 70 - 8i
        assert!((result.re - 70.0).abs() < 1e-10);
        assert!((result.im - (-8.0)).abs() < 1e-10);
    }
}

#[test]
fn test_cblas_dtrsm() {
    // Lower triangular matrix A (column-major):
    // [2  0]   stored as [2, 3, 0, 4]
    // [3  4]
    // B = [6, 11] (column vector stored as 2x1)
    // Solve A * X = B => X = A^{-1} * B
    // X[0] = 6/2 = 3
    // X[1] = (11 - 3*3)/4 = 2/4 = 0.5
    let a = [2.0f64, 3.0, 0.0, 4.0];
    let mut b = [6.0f64, 11.0];

    // SAFETY: `a` is a local 2x2 column-major buffer (`lda = 2`) and `b` a local
    // 2x1 buffer (`ldb = 2`) that is exclusively borrowed and does not alias `a`.
    unsafe {
        cblas_dtrsm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Lower,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            2,
            1,
            1.0,
            a.as_ptr(),
            2,
            b.as_mut_ptr(),
            2,
        );

        assert!((b[0] - 3.0).abs() < 1e-10);
        assert!((b[1] - 0.5).abs() < 1e-10);
    }
}

#[test]
fn test_cblas_dsyrk() {
    // A = [[1, 2], [3, 4]] in column-major = [1, 3, 2, 4]
    // C = A * A^T (NoTrans case)
    // C[0,0] = 1*1 + 2*2 = 5
    // C[1,0] = C[0,1] = 1*3 + 2*4 = 11
    // C[1,1] = 3*3 + 4*4 = 25
    let a = [1.0f64, 3.0, 2.0, 4.0];
    let mut c = [0.0f64; 4];

    // SAFETY: `a` is a local 2x2 column-major buffer (`lda = 2`) and `c` a local
    // 2x2 output (`ldc = 2`) that is exclusively borrowed and does not alias `a`.
    unsafe {
        cblas_dsyrk(
            CblasLayout::ColMajor,
            CblasUplo::Lower,
            CblasTranspose::NoTrans,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );

        // Lower triangle only
        assert!((c[0] - 5.0).abs() < 1e-10); // C[0,0]
        assert!((c[1] - 11.0).abs() < 1e-10); // C[1,0]
        assert!((c[3] - 25.0).abs() < 1e-10); // C[1,1]
    }
}

#[test]
fn test_cblas_dsymm() {
    // Symmetric matrix A (stored in lower):
    // [1  2]
    // [2  3]
    // B = [[1, 0], [0, 1]] (identity)
    // C = A * B = A
    let a = [1.0f64, 2.0, 0.0, 3.0]; // Lower stored
    let b = [1.0f64, 0.0, 0.0, 1.0];
    let mut c = [0.0f64; 4];

    // SAFETY: `a`, `b` and `c` are local 2x2 column-major buffers with
    // `lda = ldb = ldc = 2`; `c` is exclusively borrowed and aliases neither input.
    unsafe {
        cblas_dsymm(
            CblasLayout::ColMajor,
            CblasSide::Left,
            CblasUplo::Lower,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );

        // Result should be symmetric matrix A
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 2.0).abs() < 1e-10);
        assert!((c[2] - 2.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }
}

// =========================================================================
// Issue #7: row-major TRMM / TRSM with m != n.
//
// The RowMajor wrappers flip `side`, so the order of the triangular operand
// must come from the CALLER's side (`m` for Left, `n` for Right). Deriving
// it from the flipped side while keeping the caller's unswapped `m` / `n`
// made the internal dimension check fail whenever m != n, and the dropped
// `Err` turned the call into a silent no-op (TRSM also left B scaled by
// alpha). SYMM already used the right rule; its test below is audit evidence.
// =========================================================================

/// Entry-point shape shared by `cblas_{d,s}trmm` and `cblas_{d,s}trsm`.
type TriFn<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasSide,
    CblasUplo,
    CblasTranspose,
    CblasDiag,
    i32,
    i32,
    T,
    *const T,
    i32,
    *mut T,
    i32,
);

/// Entry-point shape of `cblas_{d,s}symm`.
type SymmFn<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasSide,
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

/// Padding value for every storage slot outside a logical matrix.
const PAD: f64 = 777.0;

/// Real-scalar bridge so one harness drives both the `f32` and `f64` entry points.
trait TestReal: Copy {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl TestReal for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl TestReal for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

/// Storage slot of logical element `(i, j)` for a layout / leading dimension.
fn slot(layout: CblasLayout, i: usize, j: usize, ld: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => i + j * ld,
        CblasLayout::RowMajor => i * ld + j,
    }
}

/// Stores a logical `rows x cols` matrix into a buffer padded with [`PAD`].
fn store_matrix<T: TestReal>(
    layout: CblasLayout,
    rows: usize,
    cols: usize,
    ld: usize,
    value: impl Fn(usize, usize) -> f64,
) -> Vec<T> {
    let len = match layout {
        CblasLayout::ColMajor => ld * cols,
        CblasLayout::RowMajor => ld * rows,
    };
    let mut buf = vec![T::from_f64(PAD); len];
    for j in 0..cols {
        for i in 0..rows {
            buf[slot(layout, i, j, ld)] = T::from_f64(value(i, j));
        }
    }
    buf
}

/// Asserts that every storage slot outside the logical `rows x cols` region
/// still holds [`PAD`] (the call must not write into leading-dimension padding).
fn assert_padding_untouched<T: TestReal>(
    what: &str,
    layout: CblasLayout,
    buf: &[T],
    rows: usize,
    cols: usize,
    ld: usize,
) {
    for (idx, v) in buf.iter().enumerate() {
        let (r, c) = match layout {
            CblasLayout::ColMajor => (idx % ld, idx / ld),
            CblasLayout::RowMajor => (idx / ld, idx % ld),
        };
        if r >= rows || c >= cols {
            assert_eq!(
                v.to_f64(),
                PAD,
                "{what}: padding slot ({r},{c}) was written"
            );
        }
    }
}

/// Whether logical element `(i, j)` lies inside the referenced triangle.
fn in_triangle(uplo: CblasUplo, i: usize, j: usize) -> bool {
    match uplo {
        CblasUplo::Upper => i <= j,
        CblasUplo::Lower => i >= j,
    }
}

/// Stored triangular operand: well-conditioned values inside the referenced
/// triangle, finite junk (`100.0`) in the other triangle and, for `Unit`, on
/// the stored diagonal. None of the junk may influence a result.
fn stored_tri(uplo: CblasUplo, diag: CblasDiag, i: usize, j: usize) -> f64 {
    if i == j {
        match diag {
            CblasDiag::Unit => 100.0,
            CblasDiag::NonUnit => 2.0 + 0.5 * i as f64,
        }
    } else if in_triangle(uplo, i, j) {
        ((3 * i + 5 * j) % 7) as f64 * 0.125 - 0.25
    } else {
        100.0
    }
}

/// Dense `op(T)[i][j]` for the triangular matrix `T` implied by
/// [`stored_tri`] (zeros outside the triangle, implicit ones on a unit diagonal).
fn dense_op_tri(
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    i: usize,
    j: usize,
) -> f64 {
    let (r, c) = match trans {
        CblasTranspose::NoTrans => (i, j),
        CblasTranspose::Trans | CblasTranspose::ConjTrans => (j, i),
    };
    if r == c {
        match diag {
            CblasDiag::Unit => 1.0,
            CblasDiag::NonUnit => stored_tri(uplo, diag, r, c),
        }
    } else if in_triangle(uplo, r, c) {
        stored_tri(uplo, diag, r, c)
    } else {
        0.0
    }
}

/// One TRMM / TRSM call configuration.
#[derive(Clone, Copy, Debug)]
struct TriCase {
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    trans: CblasTranspose,
    diag: CblasDiag,
    m: usize,
    n: usize,
    alpha: f64,
}

impl TriCase {
    /// Dense `op(T) * B` (Left) or `B * op(T)` (Right) for a logical `m x n`
    /// matrix `b`; the result is row-major `m x n`.
    fn apply(&self, b: &dyn Fn(usize, usize) -> f64) -> Vec<f64> {
        let (uplo, trans, diag) = (self.uplo, self.trans, self.diag);
        let mut out = vec![0.0; self.m * self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                out[i * self.n + j] = match self.side {
                    CblasSide::Left => (0..self.m)
                        .map(|l| dense_op_tri(uplo, trans, diag, i, l) * b(l, j))
                        .sum::<f64>(),
                    CblasSide::Right => (0..self.n)
                        .map(|l| b(i, l) * dense_op_tri(uplo, trans, diag, l, j))
                        .sum::<f64>(),
                };
            }
        }
        out
    }
}

/// Runs one case through `func` and compares B with a dense reference.
///
/// TRMM: `B := X0`, expected `alpha * op(T) * X0` (or `alpha * X0 * op(T)`).
/// TRSM: `B := op(T) * X0`, expected `alpha * X0` (the call solves
/// `op(T) X = alpha B`).
fn check_tri_case<T: TestReal>(func: TriFn<T>, solve: bool, case: TriCase, tol: f64) {
    let TriCase {
        layout,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
    } = case;
    let k = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };
    let lda = k + 2;
    let ldb = 3 + match layout {
        CblasLayout::ColMajor => m,
        CblasLayout::RowMajor => n,
    };
    let x0 = |i: usize, j: usize| 0.5 * ((7 * i + 3 * j) % 11) as f64 - 2.0;
    let applied = case.apply(&x0);
    let plain: Vec<f64> = (0..m * n).map(|p| x0(p / n, p % n)).collect();
    let (b_in, expected): (Vec<f64>, Vec<f64>) = if solve {
        (applied, plain.iter().map(|v| alpha * v).collect())
    } else {
        (plain, applied.iter().map(|v| alpha * v).collect())
    };

    let a_buf = store_matrix::<T>(layout, k, k, lda, |i, j| stored_tri(uplo, diag, i, j));
    let mut b_buf = store_matrix::<T>(layout, m, n, ldb, |i, j| b_in[i * n + j]);
    // SAFETY: `a_buf` / `b_buf` hold `ld * cols` (col-major) or `ld * rows`
    // (row-major) elements with `ld` above the minimum the call validates, so
    // every strided offset the call reads or writes is in bounds.
    unsafe {
        func(
            layout,
            side,
            uplo,
            trans,
            diag,
            m as i32,
            n as i32,
            T::from_f64(alpha),
            a_buf.as_ptr(),
            lda as i32,
            b_buf.as_mut_ptr(),
            ldb as i32,
        );
    }
    for j in 0..n {
        for i in 0..m {
            let got = b_buf[slot(layout, i, j, ldb)].to_f64();
            let want = expected[i * n + j];
            assert!(
                (got - want).abs() <= tol * (1.0 + want.abs()),
                "{case:?} solve={solve}: B[{i},{j}] = {got}, expected {want}"
            );
        }
    }
    assert_padding_untouched(&format!("{case:?}"), layout, &b_buf, m, n, ldb);
}

/// Sweeps layout x side x uplo x trans x diag x shape x alpha for TRMM and TRSM.
fn sweep_tri<T: TestReal>(trmm: TriFn<T>, trsm: TriFn<T>, tol: f64) {
    let transes = [
        CblasTranspose::NoTrans,
        CblasTranspose::Trans,
        CblasTranspose::ConjTrans,
    ];
    for layout in [CblasLayout::RowMajor, CblasLayout::ColMajor] {
        for side in [CblasSide::Left, CblasSide::Right] {
            for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
                for trans in transes {
                    for diag in [CblasDiag::NonUnit, CblasDiag::Unit] {
                        for (m, n) in [(3, 5), (5, 3), (1, 4), (4, 1)] {
                            for alpha in [2.0, 0.5] {
                                let case = TriCase {
                                    layout,
                                    side,
                                    uplo,
                                    trans,
                                    diag,
                                    m,
                                    n,
                                    alpha,
                                };
                                check_tri_case(trmm, false, case, tol);
                                check_tri_case(trsm, true, case, tol);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_7_row_major_trmm_trsm_non_square_sweep_f64() {
    sweep_tri::<f64>(cblas_dtrmm, cblas_dtrsm, 1e-10);
}

#[test]
fn test_issue_7_row_major_trmm_trsm_non_square_sweep_f32() {
    sweep_tri::<f32>(cblas_strmm, cblas_strsm, 1e-4);
}

/// Reporter's repro shape: RowMajor Left/Upper/NoTrans/NonUnit, m=2, n=3,
/// alpha=1, A=[[2,3],[5,4]] (the 5 is not referenced), lda=2, ldb=3.
fn run_reporter_case<T: TestReal>(func: TriFn<T>, b_in: [f64; 6]) -> [f64; 6] {
    let a = [2.0, 3.0, 5.0, 4.0].map(T::from_f64);
    let mut b = b_in.map(T::from_f64);
    // SAFETY: A is 2x2 (lda = 2) and B is 2x3 row-major (ldb = 3); both buffers are exact.
    unsafe {
        func(
            CblasLayout::RowMajor,
            CblasSide::Left,
            CblasUplo::Upper,
            CblasTranspose::NoTrans,
            CblasDiag::NonUnit,
            2,
            3,
            T::from_f64(1.0),
            a.as_ptr(),
            2,
            b.as_mut_ptr(),
            3,
        );
    }
    b.map(T::to_f64)
}

/// With T = [[2,3],[0,4]] and B = [[1,2,3],[4,5,6]], T*B = [[14,19,24],[16,20,24]].
const REPORTER_B: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const REPORTER_TB: [f64; 6] = [14.0, 19.0, 24.0, 16.0, 20.0, 24.0];

#[test]
fn test_issue_7_row_major_dtrmm_non_square_reporter_case() {
    assert_eq!(
        run_reporter_case::<f64>(cblas_dtrmm, REPORTER_B),
        REPORTER_TB
    );
}

#[test]
fn test_issue_7_row_major_strmm_non_square_reporter_case() {
    assert_eq!(
        run_reporter_case::<f32>(cblas_strmm, REPORTER_B),
        REPORTER_TB
    );
}

/// The inverse of the reporter's TRMM repro: solving T X = T*B must recover B.
#[test]
fn test_issue_7_row_major_dtrsm_non_square_reporter_case() {
    assert_eq!(
        run_reporter_case::<f64>(cblas_dtrsm, REPORTER_TB),
        REPORTER_B
    );
}

#[test]
fn test_issue_7_row_major_strsm_non_square_reporter_case() {
    assert_eq!(
        run_reporter_case::<f32>(cblas_strsm, REPORTER_TB),
        REPORTER_B
    );
}

/// Audit evidence: SYMM already derives the order of A from the flipped side
/// together with the swapped (bm, bn); check RowMajor m != n against a dense
/// reference for every side / uplo, with beta != 0 and junk in the unreferenced triangle.
fn check_symm<T: TestReal>(func: SymmFn<T>, tol: f64) {
    let (alpha, beta) = (2.0, 0.5);
    for layout in [CblasLayout::RowMajor, CblasLayout::ColMajor] {
        for side in [CblasSide::Left, CblasSide::Right] {
            for uplo in [CblasUplo::Upper, CblasUplo::Lower] {
                for (m, n) in [(3usize, 5usize), (5, 3)] {
                    let k = match side {
                        CblasSide::Left => m,
                        CblasSide::Right => n,
                    };
                    let lda = k + 2;
                    let ldb = 3 + match layout {
                        CblasLayout::ColMajor => m,
                        CblasLayout::RowMajor => n,
                    };
                    let ldc = ldb + 1;
                    let sym = |i: usize, j: usize| {
                        ((3 * i.min(j) + 5 * i.max(j)) % 7) as f64 * 0.125
                            + if i == j { 1.0 } else { 0.0 }
                    };
                    let bmat = |i: usize, j: usize| 0.25 * ((i + 2 * j) % 5) as f64 - 0.5;
                    let c0 = |i: usize, j: usize| 0.125 * ((2 * i + j) % 7) as f64;
                    let a_buf = store_matrix::<T>(layout, k, k, lda, |i, j| {
                        if in_triangle(uplo, i, j) {
                            sym(i, j)
                        } else {
                            100.0
                        }
                    });
                    let b_buf = store_matrix::<T>(layout, m, n, ldb, bmat);
                    let mut c_buf = store_matrix::<T>(layout, m, n, ldc, c0);
                    // SAFETY: buffers hold `ld * cols` / `ld * rows` elements with
                    // `ld` above the validated minimum, so all strided accesses are in bounds.
                    unsafe {
                        func(
                            layout,
                            side,
                            uplo,
                            m as i32,
                            n as i32,
                            T::from_f64(alpha),
                            a_buf.as_ptr(),
                            lda as i32,
                            b_buf.as_ptr(),
                            ldb as i32,
                            T::from_f64(beta),
                            c_buf.as_mut_ptr(),
                            ldc as i32,
                        );
                    }
                    for j in 0..n {
                        for i in 0..m {
                            let prod = match side {
                                CblasSide::Left => {
                                    (0..m).map(|l| sym(i, l) * bmat(l, j)).sum::<f64>()
                                }
                                CblasSide::Right => {
                                    (0..n).map(|l| bmat(i, l) * sym(l, j)).sum::<f64>()
                                }
                            };
                            let want = alpha * prod + beta * c0(i, j);
                            let got = c_buf[slot(layout, i, j, ldc)].to_f64();
                            assert!(
                                (got - want).abs() <= tol * (1.0 + want.abs()),
                                "symm {layout:?}/{side:?}/{uplo:?} m={m} n={n}: \
                                 C[{i},{j}] = {got}, expected {want}"
                            );
                        }
                    }
                    assert_padding_untouched("symm C", layout, &c_buf, m, n, ldc);
                }
            }
        }
    }
}

#[test]
fn test_issue_7_row_major_dsymm_non_square_audit() {
    check_symm::<f64>(cblas_dsymm, 1e-12);
}

#[test]
fn test_issue_7_row_major_ssymm_non_square_audit() {
    check_symm::<f32>(cblas_ssymm, 1e-4);
}

// =============================================================================
// Issue #9 class audit: the Level-3 wrappers must follow the reference alpha == 0
// and beta == 0 rules. Reference BLAS never reads A / B when alpha == 0 and never
// reads C when beta == 0, so NaN / Inf in those operands must not leak into the
// result.
// =============================================================================

/// Entry-point shape of `cblas_{d,s}syrk`.
type SyrkFn<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    CblasTranspose,
    i32,
    i32,
    T,
    *const T,
    i32,
    T,
    *mut T,
    i32,
);

/// Entry-point shape of `cblas_{d,s}syr2k`.
type Syr2kFn<T> = unsafe extern "C" fn(
    CblasLayout,
    CblasUplo,
    CblasTranspose,
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

const LAYOUTS: [CblasLayout; 2] = [CblasLayout::RowMajor, CblasLayout::ColMajor];
const SIDES: [CblasSide; 2] = [CblasSide::Left, CblasSide::Right];
const UPLOS: [CblasUplo; 2] = [CblasUplo::Upper, CblasUplo::Lower];

/// Alternating NaN / Inf so both kinds of non-finite poison are exercised.
fn poison(i: usize, j: usize) -> f64 {
    if (i + j) % 2 == 0 {
        f64::NAN
    } else {
        f64::INFINITY
    }
}

/// Equality that treats NaN == NaN (for "this element must come back as stored").
fn same_or_both_nan(got: f64, want: f64) -> bool {
    (got.is_nan() && want.is_nan()) || got == want
}

/// TRSM and TRMM with alpha == 0 set B := 0 without reading A or B (reference
/// ?TRSM / ?TRMM); the old TRSM wrapper prescaled `B *= alpha` (NaN * 0 = NaN)
/// and then solved with the poisoned A.
fn check_tri_alpha_zero<T: TestReal>(func: TriFn<T>) {
    for layout in LAYOUTS {
        for side in SIDES {
            for uplo in UPLOS {
                for trans in [CblasTranspose::NoTrans, CblasTranspose::Trans] {
                    for diag in [CblasDiag::NonUnit, CblasDiag::Unit] {
                        for (m, n) in [(3usize, 5usize), (5, 3)] {
                            let k = match side {
                                CblasSide::Left => m,
                                CblasSide::Right => n,
                            };
                            let lda = k + 2;
                            let ldb = 3 + match layout {
                                CblasLayout::ColMajor => m,
                                CblasLayout::RowMajor => n,
                            };
                            let a = store_matrix::<T>(layout, k, k, lda, |_, _| f64::NAN);
                            let mut b = store_matrix::<T>(layout, m, n, ldb, poison);
                            // SAFETY: `a` / `b` hold `ld * cols` (col-major) or `ld * rows`
                            // (row-major) elements with `ld` above the validated minimum.
                            unsafe {
                                func(
                                    layout,
                                    side,
                                    uplo,
                                    trans,
                                    diag,
                                    m as i32,
                                    n as i32,
                                    T::from_f64(0.0),
                                    a.as_ptr(),
                                    lda as i32,
                                    b.as_mut_ptr(),
                                    ldb as i32,
                                );
                            }
                            for j in 0..n {
                                for i in 0..m {
                                    let got = b[slot(layout, i, j, ldb)].to_f64();
                                    assert!(
                                        got == 0.0,
                                        "{layout:?}/{side:?}/{uplo:?}/{trans:?}/{diag:?} {m}x{n}: \
                                         B[{i},{j}] = {got}, expected 0"
                                    );
                                }
                            }
                            assert_padding_untouched("B", layout, &b, m, n, ldb);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_9_cblas_dtrsm_alpha_zero_zero_fills_b() {
    check_tri_alpha_zero::<f64>(cblas_dtrsm);
}

#[test]
fn test_issue_9_cblas_strsm_alpha_zero_zero_fills_b() {
    check_tri_alpha_zero::<f32>(cblas_strsm);
}

#[test]
fn test_issue_9_cblas_dtrmm_alpha_zero_zero_fills_b() {
    check_tri_alpha_zero::<f64>(cblas_dtrmm);
}

#[test]
fn test_issue_9_cblas_strmm_alpha_zero_zero_fills_b() {
    check_tri_alpha_zero::<f32>(cblas_strmm);
}

/// One alpha / beta scenario. `poison_ab` replaces every stored element of the A / B
/// operands by non-finite values (only legal when alpha == 0); `poison_c` does the same
/// for C (only legal when beta == 0).
#[derive(Clone, Copy, Debug)]
struct Scalars {
    alpha: f64,
    beta: f64,
    poison_ab: bool,
    poison_c: bool,
}

const SCALAR_CASES: [Scalars; 5] = [
    // C := 0 without reading anything.
    Scalars {
        alpha: 0.0,
        beta: 0.0,
        poison_ab: true,
        poison_c: true,
    },
    // C := 2 C, A and B never read.
    Scalars {
        alpha: 0.0,
        beta: 2.0,
        poison_ab: true,
        poison_c: false,
    },
    // quick return: C untouched, A and B never read.
    Scalars {
        alpha: 0.0,
        beta: 1.0,
        poison_ab: true,
        poison_c: false,
    },
    // C := 2 A B with C never read.
    Scalars {
        alpha: 2.0,
        beta: 0.0,
        poison_ab: false,
        poison_c: true,
    },
    // plain update.
    Scalars {
        alpha: 2.0,
        beta: 0.5,
        poison_ab: false,
        poison_c: false,
    },
];

/// SYMM alpha / beta scenarios. The (40, 38) shape pushes the order of A past the
/// GEMM threshold of the native `symm`, so both internal paths are covered.
fn check_symm_scalars<T: TestReal>(func: SymmFn<T>, tol: f64) {
    let sym = |i: usize, j: usize| {
        ((3 * i.min(j) + 5 * i.max(j)) % 7) as f64 * 0.125 + if i == j { 1.0 } else { 0.0 }
    };
    let bmat = |i: usize, j: usize| 0.25 * ((i + 2 * j) % 5) as f64 - 0.5;
    let c0 = |i: usize, j: usize| 0.125 * ((2 * i + j) % 7) as f64;
    for sc in SCALAR_CASES {
        for layout in LAYOUTS {
            for side in SIDES {
                for uplo in UPLOS {
                    for (m, n) in [(3usize, 5usize), (5, 3), (40, 38)] {
                        let k = match side {
                            CblasSide::Left => m,
                            CblasSide::Right => n,
                        };
                        let lda = k + 2;
                        let ldb = 3 + match layout {
                            CblasLayout::ColMajor => m,
                            CblasLayout::RowMajor => n,
                        };
                        let ldc = ldb + 1;
                        let a = store_matrix::<T>(layout, k, k, lda, |i, j| {
                            if sc.poison_ab {
                                f64::NAN
                            } else if in_triangle(uplo, i, j) {
                                sym(i, j)
                            } else {
                                100.0
                            }
                        });
                        let b = store_matrix::<T>(layout, m, n, ldb, |i, j| {
                            if sc.poison_ab {
                                poison(i, j)
                            } else {
                                bmat(i, j)
                            }
                        });
                        let mut c = store_matrix::<T>(layout, m, n, ldc, |i, j| {
                            if sc.poison_c { poison(i, j) } else { c0(i, j) }
                        });
                        // SAFETY: buffers hold `ld * cols` / `ld * rows` elements with `ld`
                        // above the validated minimum, so all strided accesses are in bounds.
                        unsafe {
                            func(
                                layout,
                                side,
                                uplo,
                                m as i32,
                                n as i32,
                                T::from_f64(sc.alpha),
                                a.as_ptr(),
                                lda as i32,
                                b.as_ptr(),
                                ldb as i32,
                                T::from_f64(sc.beta),
                                c.as_mut_ptr(),
                                ldc as i32,
                            );
                        }
                        for j in 0..n {
                            for i in 0..m {
                                let prod = if sc.alpha == 0.0 {
                                    0.0
                                } else {
                                    match side {
                                        CblasSide::Left => {
                                            (0..m).map(|l| sym(i, l) * bmat(l, j)).sum::<f64>()
                                        }
                                        CblasSide::Right => {
                                            (0..n).map(|l| bmat(i, l) * sym(l, j)).sum::<f64>()
                                        }
                                    }
                                };
                                let c_term = if sc.beta == 0.0 {
                                    0.0
                                } else {
                                    sc.beta * c0(i, j)
                                };
                                let want = sc.alpha * prod + c_term;
                                let got = c[slot(layout, i, j, ldc)].to_f64();
                                assert!(
                                    (got - want).abs() <= tol * (1.0 + want.abs()),
                                    "symm {layout:?}/{side:?}/{uplo:?} {m}x{n} {sc:?}: \
                                     C[{i},{j}] = {got}, expected {want}"
                                );
                            }
                        }
                        assert_padding_untouched("symm C", layout, &c, m, n, ldc);
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_9_cblas_dsymm_alpha_beta_zero_semantics() {
    check_symm_scalars::<f64>(cblas_dsymm, 1e-12);
}

#[test]
fn test_issue_9_cblas_ssymm_alpha_beta_zero_semantics() {
    check_symm_scalars::<f32>(cblas_ssymm, 1e-4);
}

/// SYRK / SYR2K alpha / beta / k scenarios. (40, 24) crosses the native GEMM
/// threshold (n >= 32 and k >= 8); (5, 0) is the k == 0 case (C := beta * C).
fn check_rank_k_scalars<T: TestReal>(syrk: SyrkFn<T>, syr2k: Syr2kFn<T>, tol: f64) {
    let a_f = |i: usize, j: usize| 0.5 * ((2 * i + 3 * j) % 7) as f64 - 1.0;
    let b_f = |i: usize, j: usize| 0.25 * ((i + 5 * j) % 6) as f64 - 0.5;
    let c_f = |i: usize, j: usize| 0.125 * ((3 * i + j) % 5) as f64;
    for sc in SCALAR_CASES {
        for layout in LAYOUTS {
            for uplo in UPLOS {
                for trans in [CblasTranspose::NoTrans, CblasTranspose::Trans] {
                    for (n, k) in [(4usize, 3usize), (40, 24), (5, 0)] {
                        for two_k in [false, true] {
                            let notrans = trans == CblasTranspose::NoTrans;
                            let (ar, ac) = if notrans { (n, k) } else { (k, n) };
                            let lda = 2 + match layout {
                                CblasLayout::ColMajor => ar,
                                CblasLayout::RowMajor => ac,
                            };
                            let ldc = n + 3;
                            let opa =
                                |i: usize, l: usize| if notrans { a_f(i, l) } else { a_f(l, i) };
                            let opb =
                                |i: usize, l: usize| if notrans { b_f(i, l) } else { b_f(l, i) };
                            let a = store_matrix::<T>(layout, ar, ac, lda, |i, j| {
                                if sc.poison_ab {
                                    poison(i, j)
                                } else {
                                    a_f(i, j)
                                }
                            });
                            let b = store_matrix::<T>(layout, ar, ac, lda, |i, j| {
                                if sc.poison_ab {
                                    poison(i + 1, j)
                                } else {
                                    b_f(i, j)
                                }
                            });
                            let mut c = store_matrix::<T>(layout, n, n, ldc, |i, j| {
                                if sc.poison_c { poison(i, j) } else { c_f(i, j) }
                            });
                            let (alpha, beta) = (T::from_f64(sc.alpha), T::from_f64(sc.beta));
                            // SAFETY: A / B hold `ld * cols` (col-major) or `ld * rows`
                            // (row-major) elements and C is n x n with ldc >= n, so every
                            // validated leading dimension is respected.
                            unsafe {
                                if two_k {
                                    syr2k(
                                        layout,
                                        uplo,
                                        trans,
                                        n as i32,
                                        k as i32,
                                        alpha,
                                        a.as_ptr(),
                                        lda as i32,
                                        b.as_ptr(),
                                        lda as i32,
                                        beta,
                                        c.as_mut_ptr(),
                                        ldc as i32,
                                    );
                                } else {
                                    syrk(
                                        layout,
                                        uplo,
                                        trans,
                                        n as i32,
                                        k as i32,
                                        alpha,
                                        a.as_ptr(),
                                        lda as i32,
                                        beta,
                                        c.as_mut_ptr(),
                                        ldc as i32,
                                    );
                                }
                            }
                            for j in 0..n {
                                for i in 0..n {
                                    let got = c[slot(layout, i, j, ldc)].to_f64();
                                    if in_triangle(uplo, i, j) {
                                        let acc: f64 = if sc.alpha == 0.0 {
                                            0.0
                                        } else {
                                            (0..k)
                                                .map(|l| {
                                                    if two_k {
                                                        opa(i, l) * opb(j, l)
                                                            + opb(i, l) * opa(j, l)
                                                    } else {
                                                        opa(i, l) * opa(j, l)
                                                    }
                                                })
                                                .sum()
                                        };
                                        let c_term = if sc.beta == 0.0 {
                                            0.0
                                        } else {
                                            sc.beta * c_f(i, j)
                                        };
                                        let want = sc.alpha * acc + c_term;
                                        assert!(
                                            (got - want).abs() <= tol * (1.0 + want.abs()),
                                            "{} {layout:?}/{uplo:?}/{trans:?} n={n} k={k} {sc:?}: \
                                             C[{i},{j}] = {got}, expected {want}",
                                            if two_k { "syr2k" } else { "syrk" }
                                        );
                                    } else {
                                        // The other triangle is never referenced.
                                        let before =
                                            if sc.poison_c { poison(i, j) } else { c_f(i, j) };
                                        assert!(
                                            same_or_both_nan(got, before),
                                            "{} {layout:?}/{uplo:?}/{trans:?} n={n} k={k} {sc:?}: \
                                             unreferenced C[{i},{j}] changed to {got}",
                                            if two_k { "syr2k" } else { "syrk" }
                                        );
                                    }
                                }
                            }
                            assert_padding_untouched("rank-k C", layout, &c, n, n, ldc);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_9_cblas_dsyrk_dsyr2k_alpha_beta_zero_semantics() {
    check_rank_k_scalars::<f64>(cblas_dsyrk, cblas_dsyr2k, 1e-12);
}

#[test]
fn test_issue_9_cblas_ssyrk_ssyr2k_alpha_beta_zero_semantics() {
    check_rank_k_scalars::<f32>(cblas_ssyrk, cblas_ssyr2k, 1e-4);
}
