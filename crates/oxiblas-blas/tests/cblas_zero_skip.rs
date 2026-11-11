//! Issue #14 / #13: reference BLAS skips the column update of the rank-1 and
//! rank-2 routines when its scaling element is exactly zero — `y(j)` for
//! GER / GERU / GERC, `x(j)` for SYR / SPR / HER / HPR, `x(j)` and `y(j)` for
//! SYR2 / SPR2 / HER2 / HPR2 — so an Inf/NaN in the other operand never
//! reaches that column as `Inf * 0 = NaN`. HER / HPR / HER2 / HPR2 still drop
//! the imaginary part of the skipped column's diagonal.
//!
//! The entry points are called through the public `oxiblas_blas::cblas::*`
//! path in both layouts. A row-major call runs the column-major kernel on the
//! buffer read as `A^T`, as reference CBLAS does, so the skipped "column" `c`
//! is column `c` of the caller's matrix in `ColMajor` and row `c` in
//! `RowMajor`. The operand buffers are built from the CBLAS storage
//! definitions of each layout.
//!
//! * The packed updates agree bit for bit with their dense twins
//!   (`cblas_?hpr` / `?her`, `cblas_?hpr2` / `?her2`, `cblas_?spr` / `?syr`,
//!   `cblas_?spr2` / `?syr2`) on this NaN-free data; with NaN input the NaN
//!   sign/payload of `?hpr` / `?hpr2` may differ from `?her` / `?her2`.
//! * The skipped column (row) is bitwise untouched apart from the imaginary
//!   part of its diagonal, while the Inf does reach the matrix elsewhere.
//! * GER / GERU / GERC: in `ColMajor` a zero `y(j)` protects column `j`; in
//!   `RowMajor` (operands swapped) a zero `x(i)` protects row `i`. A zero in
//!   the other vector protects nothing, exactly as in the reference.

use num_complex::{Complex32, Complex64};
use oxiblas_blas::cblas::*;

const LAYOUTS: [CblasLayout; 2] = [CblasLayout::ColMajor, CblasLayout::RowMajor];
const UPLOS: [CblasUplo; 2] = [CblasUplo::Upper, CblasUplo::Lower];
const INF: f64 = f64::INFINITY;
/// Order of the symmetric / Hermitian test matrices.
const N: usize = 3;

/// The four BLAS precisions.
trait Elem: Copy + core::fmt::Debug {
    /// Builds a value; real types drop `im`.
    fn mk(re: f64, im: f64) -> Self;
    /// Bit patterns of both parts (NaN-safe identity check).
    fn bits(self) -> (u64, u64);
    fn finite(self) -> bool;
    /// Both parts compare equal to zero (`-0.0` included), like Fortran's
    /// `.EQ.ZERO`.
    fn is_zero(self) -> bool;
    fn re(self) -> f64;
}

impl Elem for f32 {
    fn mk(re: f64, _im: f64) -> Self {
        re as f32
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn re(self) -> f64 {
        f64::from(self)
    }
}

impl Elem for f64 {
    fn mk(re: f64, _im: f64) -> Self {
        re
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn re(self) -> f64 {
        self
    }
}

impl Elem for Complex32 {
    fn mk(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    fn re(self) -> f64 {
        f64::from(self.re)
    }
}

impl Elem for Complex64 {
    fn mk(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    fn re(self) -> f64 {
        self.re
    }
}

fn nan<T: Elem>() -> T {
    T::mk(f64::NAN, f64::NAN)
}

fn in_triangle(uplo: CblasUplo, i: usize, j: usize) -> bool {
    match uplo {
        CblasUplo::Upper => i <= j,
        CblasUplo::Lower => i >= j,
    }
}

/// Slot of `A[i][j]` in dense storage with leading dimension `lda`.
fn dense_slot(layout: CblasLayout, i: usize, j: usize, lda: usize) -> usize {
    match layout {
        CblasLayout::ColMajor => i + j * lda,
        CblasLayout::RowMajor => i * lda + j,
    }
}

/// Slot of `A[i][j]` in the packed `uplo` triangle: column-wise packing for
/// `ColMajor`, row-wise for `RowMajor`.
fn packed_slot(layout: CblasLayout, uplo: CblasUplo, i: usize, j: usize, n: usize) -> usize {
    match (layout, uplo) {
        (CblasLayout::ColMajor, CblasUplo::Upper) => i + j * (j + 1) / 2,
        (CblasLayout::ColMajor, CblasUplo::Lower) => i + j * (2 * n - j - 1) / 2,
        (CblasLayout::RowMajor, CblasUplo::Upper) => j + i * (2 * n - i - 1) / 2,
        (CblasLayout::RowMajor, CblasUplo::Lower) => j + i * (i + 1) / 2,
    }
}

/// Stored entries that reference BLAS leaves alone when the scaling element
/// at index `z` is zero: column `z` of the `uplo` triangle in `ColMajor`, row
/// `z` in `RowMajor` (the column of the transposed buffer the kernel updates).
fn skipped_entries(layout: CblasLayout, uplo: CblasUplo, z: usize) -> Vec<(usize, usize)> {
    (0..N)
        .map(|k| match layout {
            CblasLayout::ColMajor => (k, z),
            CblasLayout::RowMajor => (z, k),
        })
        .filter(|&(i, j)| in_triangle(uplo, i, j))
        .collect()
}

/// `N x N` Hermitian (real: symmetric) test matrix whose diagonal carries an
/// imaginary part the update must drop.
fn hermitian<T: Elem>() -> Vec<Vec<T>> {
    let entry = |i: usize, j: usize| {
        if i == j {
            return T::mk(1.0 + i as f64, 0.625);
        }
        // The strictly lower entry (hi, lo); the upper one is its conjugate.
        let (hi, lo) = (i.max(j), i.min(j));
        let (re, im) = (
            0.25 * (hi + 1) as f64 - 0.125 * lo as f64,
            0.5 - 0.25 * hi as f64,
        );
        if i > j { T::mk(re, im) } else { T::mk(re, -im) }
    };
    (0..N)
        .map(|i| (0..N).map(|j| entry(i, j)).collect())
        .collect()
}

type DenseCall<'a, T> = &'a dyn Fn(CblasLayout, CblasUplo, &[T], &[T], &mut [T], i32);
type PackedCall<'a, T> = &'a dyn Fn(CblasLayout, CblasUplo, &[T], &[T], &mut [T]);

/// Runs a rank update through its dense and its packed entry point on the
/// same matrix, for both layouts and triangles, with an exact zero (`x(z)`;
/// for the rank-2 forms also `y(z)`) facing an infinite `x(1)`. Checks that
/// every stored entry agrees bit for bit, that the skipped column (row) is
/// untouched apart from its diagonal's imaginary part, that the unreferenced
/// slots are untouched, and that the Inf does reach `A[1][1]`.
fn check_packed_matches_dense<T: Elem>(
    name: &str,
    rank2: bool,
    dense: DenseCall<'_, T>,
    packed: PackedCall<'_, T>,
) {
    let lda = N + 1;
    let h = hermitian::<T>();
    // The first case skips index 0, the second index 2; between them every
    // layout / triangle has a skipped off-diagonal entry facing the Inf.
    let cases = [
        (
            [T::mk(0.0, 0.0), T::mk(INF, 0.0), T::mk(1.0, 0.5)],
            [T::mk(-0.0, 0.0), T::mk(0.5, -1.0), T::mk(2.0, 1.0)],
        ),
        (
            [T::mk(1.0, 0.5), T::mk(INF, 0.0), T::mk(-0.0, 0.0)],
            [T::mk(2.0, 1.0), T::mk(0.5, -1.0), T::mk(0.0, -0.0)],
        ),
    ];
    for layout in LAYOUTS {
        for uplo in UPLOS {
            for (x, y) in &cases {
                let what = format!("{name} {layout:?} {uplo:?} x={x:?}");
                let mut a0 = vec![nan::<T>(); lda * N];
                let mut ap0 = vec![nan::<T>(); N * (N + 1) / 2];
                let mut referenced = vec![false; a0.len()];
                for i in 0..N {
                    for j in 0..N {
                        if in_triangle(uplo, i, j) {
                            let d = dense_slot(layout, i, j, lda);
                            a0[d] = h[i][j];
                            referenced[d] = true;
                            ap0[packed_slot(layout, uplo, i, j, N)] = h[i][j];
                        }
                    }
                }
                let (mut a, mut ap) = (a0.clone(), ap0.clone());
                dense(layout, uplo, x, y, &mut a, lda as i32);
                packed(layout, uplo, x, y, &mut ap);
                for i in 0..N {
                    for j in 0..N {
                        if in_triangle(uplo, i, j) {
                            let (p, d) = (
                                ap[packed_slot(layout, uplo, i, j, N)],
                                a[dense_slot(layout, i, j, lda)],
                            );
                            assert_eq!(p.bits(), d.bits(), "{what}: ({i},{j}) {p:?} vs {d:?}");
                        }
                    }
                }
                for (s, &used) in referenced.iter().enumerate() {
                    if !used {
                        assert_eq!(a[s].bits(), a0[s].bits(), "{what}: dense slot {s} changed");
                    }
                }
                for z in 0..N {
                    if !x[z].is_zero() || (rank2 && !y[z].is_zero()) {
                        continue;
                    }
                    for (i, j) in skipped_entries(layout, uplo, z) {
                        let s = packed_slot(layout, uplo, i, j, N);
                        let want = if i == j {
                            T::mk(ap0[s].re(), 0.0)
                        } else {
                            ap0[s]
                        };
                        assert_eq!(
                            ap[s].bits(),
                            want.bits(),
                            "{what}: skipped ({i},{j}) {:?} -> {:?}",
                            ap0[s],
                            ap[s]
                        );
                    }
                }
                let s = packed_slot(layout, uplo, 1, 1, N);
                assert!(!ap[s].finite(), "{what}: the Inf did not reach A[1][1]");
            }
        }
    }
}

#[test]
fn test_issue_14_cblas_zhpr_matches_zher_bitwise() {
    check_packed_matches_dense::<Complex64>(
        "zher/zhpr",
        false,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, _, a, lda| unsafe {
            cblas_zher(l, u, N as i32, 1.5, x.as_ptr(), 1, a.as_mut_ptr(), lda)
        },
        // SAFETY: packed N x N triangle and a unit-stride N-vector.
        &|l, u, x, _, ap| unsafe {
            cblas_zhpr(l, u, N as i32, 1.5, x.as_ptr(), 1, ap.as_mut_ptr())
        },
    );
}

#[test]
fn test_issue_14_cblas_chpr_matches_cher_bitwise() {
    check_packed_matches_dense::<Complex32>(
        "cher/chpr",
        false,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, _, a, lda| unsafe {
            cblas_cher(l, u, N as i32, 1.5, x.as_ptr(), 1, a.as_mut_ptr(), lda)
        },
        // SAFETY: packed N x N triangle and a unit-stride N-vector.
        &|l, u, x, _, ap| unsafe {
            cblas_chpr(l, u, N as i32, 1.5, x.as_ptr(), 1, ap.as_mut_ptr())
        },
    );
}

#[test]
fn test_issue_14_cblas_zhpr2_matches_zher2_bitwise() {
    let alpha = Complex64::new(0.75, -0.5);
    check_packed_matches_dense::<Complex64>(
        "zher2/zhpr2",
        true,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, y, a, lda| unsafe {
            cblas_zher2(
                l,
                u,
                N as i32,
                &alpha,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                lda,
            )
        },
        // SAFETY: packed N x N triangle and unit-stride N-vectors.
        &|l, u, x, y, ap| unsafe {
            cblas_zhpr2(
                l,
                u,
                N as i32,
                &alpha,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                ap.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_issue_14_cblas_chpr2_matches_cher2_bitwise() {
    let alpha = Complex32::new(0.75, -0.5);
    check_packed_matches_dense::<Complex32>(
        "cher2/chpr2",
        true,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, y, a, lda| unsafe {
            cblas_cher2(
                l,
                u,
                N as i32,
                &alpha,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                lda,
            )
        },
        // SAFETY: packed N x N triangle and unit-stride N-vectors.
        &|l, u, x, y, ap| unsafe {
            cblas_chpr2(
                l,
                u,
                N as i32,
                &alpha,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                ap.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_issue_14_cblas_dspr_matches_dsyr_and_skips_zero_columns() {
    check_packed_matches_dense::<f64>(
        "dsyr/dspr",
        false,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, _, a, lda| unsafe {
            cblas_dsyr(l, u, N as i32, 1.5, x.as_ptr(), 1, a.as_mut_ptr(), lda)
        },
        // SAFETY: packed N x N triangle and a unit-stride N-vector.
        &|l, u, x, _, ap| unsafe {
            cblas_dspr(l, u, N as i32, 1.5, x.as_ptr(), 1, ap.as_mut_ptr())
        },
    );
}

#[test]
fn test_issue_14_cblas_sspr_matches_ssyr_and_skips_zero_columns() {
    check_packed_matches_dense::<f32>(
        "ssyr/sspr",
        false,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, _, a, lda| unsafe {
            cblas_ssyr(l, u, N as i32, 1.5, x.as_ptr(), 1, a.as_mut_ptr(), lda)
        },
        // SAFETY: packed N x N triangle and a unit-stride N-vector.
        &|l, u, x, _, ap| unsafe {
            cblas_sspr(l, u, N as i32, 1.5, x.as_ptr(), 1, ap.as_mut_ptr())
        },
    );
}

#[test]
fn test_issue_14_cblas_dspr2_matches_dsyr2_and_skips_zero_columns() {
    check_packed_matches_dense::<f64>(
        "dsyr2/dspr2",
        true,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, y, a, lda| unsafe {
            cblas_dsyr2(
                l,
                u,
                N as i32,
                1.5,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                lda,
            )
        },
        // SAFETY: packed N x N triangle and unit-stride N-vectors.
        &|l, u, x, y, ap| unsafe {
            cblas_dspr2(
                l,
                u,
                N as i32,
                1.5,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                ap.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_issue_14_cblas_sspr2_matches_ssyr2_and_skips_zero_columns() {
    check_packed_matches_dense::<f32>(
        "ssyr2/sspr2",
        true,
        // SAFETY: N x N matrix with lda >= N and unit-stride N-vectors.
        &|l, u, x, y, a, lda| unsafe {
            cblas_ssyr2(
                l,
                u,
                N as i32,
                1.5,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                lda,
            )
        },
        // SAFETY: packed N x N triangle and unit-stride N-vectors.
        &|l, u, x, y, ap| unsafe {
            cblas_sspr2(
                l,
                u,
                N as i32,
                1.5,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                ap.as_mut_ptr(),
            )
        },
    );
}

// =============================================================================
// GER / GERU / GERC
// =============================================================================

type GerCall<'a, T> = &'a dyn Fn(CblasLayout, &[T], &[T], &mut [T], i32);

/// `2 x 3` rank-1 update. `ColMajor` skips the column of a zero `y(j)`,
/// `RowMajor` (reference CBLAS swaps the operands) the row of a zero `x(i)`;
/// a zero in the other vector protects nothing.
fn check_ger<T: Elem>(name: &str, call: GerCall<'_, T>) {
    let (m, n) = (2usize, 3usize);
    // x(0) infinite, y(1) = 0 and y(2) = -0.
    let inf_x = (
        vec![T::mk(INF, 0.0), T::mk(1.0, -0.5)],
        vec![T::mk(1.0, 0.25), T::mk(0.0, 0.0), T::mk(-0.0, 0.0)],
    );
    // x(0) = 0, y(0) infinite.
    let inf_y = (
        vec![T::mk(0.0, -0.0), T::mk(1.0, -0.5)],
        vec![T::mk(INF, 0.0), T::mk(2.0, 0.5), T::mk(3.0, -1.0)],
    );
    for layout in LAYOUTS {
        let (lda, lines) = match layout {
            CblasLayout::ColMajor => (m + 1, n),
            CblasLayout::RowMajor => (n + 1, m),
        };
        let mut a0 = vec![nan::<T>(); lda * lines];
        let mut referenced = vec![false; a0.len()];
        for i in 0..m {
            for j in 0..n {
                let s = dense_slot(layout, i, j, lda);
                a0[s] = T::mk(1.0 + i as f64, 0.5 * j as f64 - 0.25);
                referenced[s] = true;
            }
        }
        for (case, (x, y)) in [("inf x", &inf_x), ("inf y", &inf_y)] {
            let what = format!("{name} {layout:?} {case}");
            let mut a = a0.clone();
            call(layout, x, y, &mut a, lda as i32);
            let at = |i: usize, j: usize| a[dense_slot(layout, i, j, lda)];
            let at0 = |i: usize, j: usize| a0[dense_slot(layout, i, j, lda)];
            match (layout, case) {
                (CblasLayout::ColMajor, "inf x") => {
                    // y(1) == 0 and y(2) == -0: columns 1 and 2 are skipped.
                    for j in 1..n {
                        for i in 0..m {
                            assert_eq!(at(i, j).bits(), at0(i, j).bits(), "{what}: ({i},{j})");
                        }
                    }
                    assert!(!at(0, 0).finite(), "{what}: column 0 missed the Inf");
                }
                (CblasLayout::ColMajor, _) => {
                    // x(0) == 0 protects nothing: A00 = a + 0 * (alpha * Inf).
                    assert!(!at(0, 0).finite(), "{what}: A00 should be NaN");
                    assert!(!at(1, 0).finite(), "{what}: A10 missed the Inf");
                }
                (CblasLayout::RowMajor, "inf y") => {
                    // x(0) == 0: row 0 is skipped.
                    for j in 0..n {
                        assert_eq!(at(0, j).bits(), at0(0, j).bits(), "{what}: (0,{j})");
                    }
                    assert!(!at(1, 0).finite(), "{what}: row 1 missed the Inf");
                }
                (CblasLayout::RowMajor, _) => {
                    // y(1) == 0 protects nothing: A01 = a + 0 * (alpha * Inf).
                    assert!(!at(0, 1).finite(), "{what}: A01 should be NaN");
                    assert!(!at(0, 0).finite(), "{what}: A00 missed the Inf");
                }
            }
            for (s, &used) in referenced.iter().enumerate() {
                if !used {
                    assert_eq!(a[s].bits(), a0[s].bits(), "{what}: padding slot {s}");
                }
            }
        }
    }
}

#[test]
fn test_issue_14_cblas_dger_sger_skip_zero_columns() {
    // SAFETY: 2 x 3 matrix with a valid lda for the layout; unit-stride vectors.
    check_ger::<f64>("dger", &|l, x, y, a, lda| unsafe {
        cblas_dger(
            l,
            2,
            3,
            1.5,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a.as_mut_ptr(),
            lda,
        )
    });
    // SAFETY: as above.
    check_ger::<f32>("sger", &|l, x, y, a, lda| unsafe {
        cblas_sger(
            l,
            2,
            3,
            1.5,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a.as_mut_ptr(),
            lda,
        )
    });
}

#[test]
fn test_issue_14_cblas_zgeru_zgerc_skip_zero_columns() {
    let alpha = Complex64::new(0.75, -0.5);
    // SAFETY: 2 x 3 matrix with a valid lda for the layout; unit-stride vectors.
    check_ger::<Complex64>("zgeru", &|l, x, y, a, lda| unsafe {
        cblas_zgeru(
            l,
            2,
            3,
            &alpha,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a.as_mut_ptr(),
            lda,
        )
    });
    // SAFETY: as above.
    check_ger::<Complex64>("zgerc", &|l, x, y, a, lda| unsafe {
        cblas_zgerc(
            l,
            2,
            3,
            &alpha,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a.as_mut_ptr(),
            lda,
        )
    });
}

#[test]
fn test_issue_14_cblas_cgeru_cgerc_skip_zero_columns() {
    let alpha = Complex32::new(0.75, -0.5);
    // SAFETY: 2 x 3 matrix with a valid lda for the layout; unit-stride vectors.
    check_ger::<Complex32>("cgeru", &|l, x, y, a, lda| unsafe {
        cblas_cgeru(
            l,
            2,
            3,
            &alpha,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a.as_mut_ptr(),
            lda,
        )
    });
    // SAFETY: as above.
    check_ger::<Complex32>("cgerc", &|l, x, y, a, lda| unsafe {
        cblas_cgerc(
            l,
            2,
            3,
            &alpha,
            x.as_ptr(),
            1,
            y.as_ptr(),
            1,
            a.as_mut_ptr(),
            lda,
        )
    });
}
