//! Tests for the elimination loops and the blocked drivers of [`Lu`].
//!
//! The `oracle` module keeps the previous row-order loops (`for i` outside
//! `for j`) verbatim, together with copies of the previous blocked, recursive
//! and parallel drivers. Those copied `L11`, `U12` and `L21` out of the factor,
//! formed `L21 · U12` in a zeroed temporary and subtracted it afterwards.
//!
//! - The unblocked paths (`compute`, `unblocked_factor`, `factor_panel`) must
//!   reproduce the oracle bit for bit, along with its pivots, swap count and
//!   errors: the loop interchange gives every element exactly the same
//!   floating-point operations on the same operands, only in a different
//!   order.
//! - The blocked, recursive and parallel drivers compute in a different
//!   order, so the last bits of their factors can differ from the oracle's:
//!   1. a panel wider than 16 columns is factored recursively (LAPACK
//!      `?getrf2`), so most of its updates are GEMMs, which accumulate each
//!      entry's products before subtracting them, instead of one rank-1
//!      update per column;
//!   2. `compute_recursive` splits the whole matrix that way (the oracle's
//!      recursion was a right-looking blocked factorization with panels of at
//!      most 64 columns);
//!   3. the trailing update is one in-place `gemm(−1, L21, U12, 1, A22)`,
//!      which adds the product of each K block of the GEMM's partition to
//!      `A22` in turn: once a panel is wider than one K block it rounds
//!      `(a − p₁) − p₂` where the oracle rounded `a − (p₁ + p₂)`.
//!
//!   These drivers are checked for the oracle's pivots and swap count and a
//!   backward error `max |PA − LU| / max |A|` of at most `c · n · eps`.
//! - The parallel drivers must equal the sequential ones bit for bit: TRSM
//!   and GEMM give bitwise the same result for any `Par`.

use super::*;
use crate::lu::tolerance::lu_pivot_tolerances;
use num_complex::{Complex32, Complex64};
use oxiblas_blas::level3::gemm::gemm;
use oxiblas_blas::level3::trsm::trsm_in_place;
use oxiblas_matrix::MatMut;

/// Factors, pivots and swap count, as produced by the oracle.
type Factors<T> = (Mat<T>, Vec<usize>, usize);

/// Signature shared by `gemm` and the parallel `gemm_with_par` wrapper.
type GemmFn<T> = fn(T, MatRef<'_, T>, MatRef<'_, T>, T, MatMut<'_, T>);

/// The previous row-order implementation, kept as the reference.
mod oracle {
    use super::*;

    /// Per-column singularity thresholds, as production computes them (the
    /// oracle checks the loop order, not the tolerance rule).
    fn singularity_tol<T: Field>(a: MatRef<'_, T>) -> Vec<T::Real> {
        lu_pivot_tolerances(a)
    }

    fn copy_square<T: Field + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Mat<T> {
        let n = a.nrows();
        let mut lu = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                lu[(i, j)] = a[(i, j)];
            }
        }
        lu
    }

    /// The previous `Lu::compute`.
    pub(super) fn compute<T: Field + bytemuck::Zeroable>(
        a: MatRef<'_, T>,
    ) -> Result<Factors<T>, LuError> {
        let n = a.nrows();
        let tol = singularity_tol(a);
        let mut lu = copy_square(a);
        let mut pivot = vec![0usize; n];
        let mut num_swaps = 0;

        for k in 0..n {
            let mut pivot_row = k;
            let mut pivot_val = Scalar::abs(lu[(k, k)]);
            for i in (k + 1)..n {
                let val = Scalar::abs(lu[(i, k)]);
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
            if pivot_val <= tol[k] {
                return Err(LuError::Singular { index: k });
            }
            pivot[k] = pivot_row;
            if pivot_row != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                num_swaps += 1;
            }

            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;
                for j in (k + 1)..n {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }

        Ok((lu, pivot, num_swaps))
    }

    /// The previous `Lu::factor_panel`.
    pub(super) fn factor_panel<T: Field>(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        jb: usize,
        jb_size: usize,
        tol: &[T::Real],
    ) -> Result<(), LuError> {
        for k in jb..(jb + jb_size) {
            let mut pivot_row = k;
            let mut pivot_val = Scalar::abs(lu[(k, k)]);
            for i in (k + 1)..n {
                let val = Scalar::abs(lu[(i, k)]);
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
            if pivot_val <= tol[k] {
                return Err(LuError::Singular { index: k });
            }
            pivot[k] = pivot_row;
            if pivot_row != k {
                for j in 0..(jb + jb_size) {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                *num_swaps += 1;
            }

            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;
                for j in (k + 1)..(jb + jb_size) {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }
        Ok(())
    }

    /// The previous `Lu::unblocked_factor`.
    fn unblocked_factor<T: Field>(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        start: usize,
        tol: &[T::Real],
    ) -> Result<(), LuError> {
        for k in start..n {
            let mut pivot_row = k;
            let mut pivot_val = Scalar::abs(lu[(k, k)]);
            for i in (k + 1)..n {
                let val = Scalar::abs(lu[(i, k)]);
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
            if pivot_val <= tol[k] {
                return Err(LuError::Singular { index: k });
            }
            pivot[k] = pivot_row;
            if pivot_row != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                *num_swaps += 1;
            }

            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;
                for j in (k + 1)..n {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }
        Ok(())
    }

    /// Row interchanges, TRSM and GEMM update after panel `jb..jb + width`,
    /// exactly as the blocked, recursive and parallel drivers apply them.
    fn update_trailing<T: Field + GemmKernel + bytemuck::Zeroable>(
        lu: &mut Mat<T>,
        pivot: &[usize],
        n: usize,
        jb: usize,
        width: usize,
        gemm_fn: GemmFn<T>,
    ) {
        let trailing = n - jb - width;
        if trailing == 0 {
            return;
        }
        for k in jb..jb + width {
            let pk = pivot[k];
            if pk != k {
                for j in (jb + width)..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pk, j)];
                    lu[(pk, j)] = tmp;
                }
            }
        }

        let mut l11: Mat<T> = Mat::zeros(width, width);
        for i in 0..width {
            l11[(i, i)] = T::one();
            for j in 0..i {
                l11[(i, j)] = lu[(jb + i, jb + j)];
            }
        }
        let mut u12: Mat<T> = Mat::zeros(width, trailing);
        for j in 0..trailing {
            for i in 0..width {
                u12[(i, j)] = lu[(jb + i, jb + width + j)];
            }
        }
        let _ = trsm_in_place(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::Unit,
            l11.as_ref(),
            u12.as_mut(),
        );
        for j in 0..trailing {
            for i in 0..width {
                lu[(jb + i, jb + width + j)] = u12[(i, j)];
            }
        }

        let mut l21: Mat<T> = Mat::zeros(trailing, width);
        for j in 0..width {
            for i in 0..trailing {
                l21[(i, j)] = lu[(jb + width + i, jb + j)];
            }
        }
        let mut update: Mat<T> = Mat::zeros(trailing, trailing);
        gemm_fn(
            T::one(),
            l21.as_ref(),
            u12.as_ref(),
            T::zero(),
            update.as_mut(),
        );
        for j in 0..trailing {
            for i in 0..trailing {
                lu[(jb + width + i, jb + width + j)] =
                    lu[(jb + width + i, jb + width + j)] - update[(i, j)];
            }
        }
    }

    /// The previous `Lu::compute_with_block_size` (and, with a parallel
    /// `gemm_fn`, `Lu::compute_blocked_par_with_block_size`).
    pub(super) fn compute_with_block_size<T: Field + GemmKernel + bytemuck::Zeroable>(
        a: MatRef<'_, T>,
        nb: usize,
        gemm_fn: GemmFn<T>,
    ) -> Result<Factors<T>, LuError> {
        let n = a.nrows();
        let tol = singularity_tol(a);
        let mut lu = copy_square(a);
        let mut pivot = vec![0usize; n];
        let mut num_swaps = 0;

        if n >= nb {
            let mut jb = 0;
            while jb < n {
                let jb_size = nb.min(n - jb);
                factor_panel(&mut lu, &mut pivot, &mut num_swaps, n, jb, jb_size, &tol)?;
                update_trailing(&mut lu, &pivot, n, jb, jb_size, gemm_fn);
                jb += jb_size;
            }
        } else {
            unblocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0, &tol)?;
        }

        Ok((lu, pivot, num_swaps))
    }

    /// The previous recursion threshold of `Lu::compute_recursive`.
    pub(super) const RECURSIVE_THRESHOLD: usize = 64;

    fn recursive_factor<T: Field + GemmKernel + bytemuck::Zeroable>(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        col_start: usize,
        width: usize,
        tol: &[T::Real],
    ) -> Result<(), LuError> {
        if width == 0 {
            return Ok(());
        }
        if width <= RECURSIVE_THRESHOLD {
            factor_panel(lu, pivot, num_swaps, n, col_start, width, tol)?;
            update_trailing(lu, pivot, n, col_start, width, gemm::<T>);
            return Ok(());
        }
        let n1 = width / 2;
        let n2 = width - n1;
        recursive_factor(lu, pivot, num_swaps, n, col_start, n1, tol)?;
        recursive_factor(lu, pivot, num_swaps, n, col_start + n1, n2, tol)
    }

    /// The previous `Lu::compute_recursive`.
    pub(super) fn compute_recursive<T: Field + GemmKernel + bytemuck::Zeroable>(
        a: MatRef<'_, T>,
    ) -> Result<Factors<T>, LuError> {
        let n = a.nrows();
        let tol = singularity_tol(a);
        let mut lu = copy_square(a);
        let mut pivot = vec![0usize; n];
        let mut num_swaps = 0;
        recursive_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0, n, &tol)?;
        Ok((lu, pivot, num_swaps))
    }
}

/// Scalar types under test: random sampling, an exact bit pattern, a
/// magnitude and the machine epsilon of the precision.
trait BitScalar: Field + GemmKernel + bytemuck::Zeroable {
    /// Machine epsilon of the underlying real type.
    const EPS: f64;
    fn sample(rng: &mut XorShift64) -> Self;
    fn bits(self) -> (u64, u64);
    fn mag(self) -> f64;
}

impl BitScalar for f64 {
    const EPS: f64 = f64::EPSILON;
    fn sample(rng: &mut XorShift64) -> Self {
        rng.next_signed()
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
    fn mag(self) -> f64 {
        self.abs()
    }
}

impl BitScalar for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn sample(rng: &mut XorShift64) -> Self {
        rng.next_signed() as f32
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
    fn mag(self) -> f64 {
        f64::from(self.abs())
    }
}

impl BitScalar for Complex64 {
    const EPS: f64 = f64::EPSILON;
    fn sample(rng: &mut XorShift64) -> Self {
        Complex64::new(rng.next_signed(), rng.next_signed())
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
    fn mag(self) -> f64 {
        self.norm()
    }
}

impl BitScalar for Complex32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn sample(rng: &mut XorShift64) -> Self {
        Complex32::new(rng.next_signed() as f32, rng.next_signed() as f32)
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
    fn mag(self) -> f64 {
        f64::from(self.norm())
    }
}

/// Deterministic xorshift64 generator (no external dependency).
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Any non-zero state works; mix the seed so small seeds differ a lot.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform sample in `[-1, 1)`.
    fn next_signed(&mut self) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * unit - 1.0
    }
}

fn random_matrix<T: BitScalar>(n: usize, seed: u64) -> Mat<T> {
    let mut rng = XorShift64::new(seed);
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = T::sample(&mut rng);
        }
    }
    a
}

/// Random matrix with column `col` and row `row` set exactly to zero, so the
/// elimination meets an exactly zero pivot and both loops must report it.
fn singular_matrix<T: BitScalar>(n: usize, seed: u64, col: usize, row: usize) -> Mat<T> {
    let mut a = random_matrix::<T>(n, seed);
    for i in 0..n {
        a[(i, col)] = T::zero();
    }
    for j in 0..n {
        a[(row, j)] = T::zero();
    }
    a
}

fn assert_same_bits<T: BitScalar>(label: &str, got: &Mat<T>, want: &Mat<T>) {
    let n = want.nrows();
    assert_eq!(got.nrows(), n, "{label}: sizes differ");
    for j in 0..n {
        for i in 0..n {
            assert_eq!(
                got[(i, j)].bits(),
                want[(i, j)].bits(),
                "{label}: LU({i}, {j}) differs: {} vs {}",
                got[(i, j)],
                want[(i, j)]
            );
        }
    }
}

fn assert_same<T: BitScalar>(
    label: &str,
    got: Result<Lu<T>, LuError>,
    want: Result<Factors<T>, LuError>,
) {
    match (got, want) {
        (Ok(lu), Ok((factors, pivot, num_swaps))) => {
            assert_eq!(lu.pivot, pivot, "{label}: pivots differ");
            assert_eq!(lu.num_swaps, num_swaps, "{label}: swap counts differ");
            assert_same_bits(label, &lu.lu, &factors);
        }
        (Err(got_err), Err(want_err)) => {
            assert_eq!(got_err, want_err, "{label}: errors differ");
        }
        (got, want) => panic!(
            "{label}: outcome differs: got {:?}, oracle {:?}",
            got.map(|_| ()),
            want.map(|_| ())
        ),
    }
}

/// Compares the unblocked paths of `Lu` (whose arithmetic is unchanged)
/// against the oracle, bit for bit.
fn check_unblocked_variants<T: BitScalar>(label: &str, a: &Mat<T>) {
    let a = a.as_ref();
    let n = a.nrows();
    assert_same(
        &format!("{label} compute"),
        Lu::compute(a),
        oracle::compute(a),
    );
    // nb > n takes the unblocked path of the blocked driver.
    assert_same(
        &format!("{label} compute_with_block_size(nb={})", n + 1),
        Lu::compute_with_block_size(a, n + 1),
        oracle::compute_with_block_size(a, n + 1, gemm::<T>),
    );
    // A recursion base case is one unblocked panel and no trailing update.
    if n <= PANEL_BASE.min(oracle::RECURSIVE_THRESHOLD) {
        assert_same(
            &format!("{label} compute_recursive"),
            Lu::compute_recursive(a),
            oracle::compute_recursive(a),
        );
    }
}

const SIZES: [usize; 22] = [
    0, 1, 2, 3, 5, 8, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 129, 200, 255, 256, 257, 300,
];

#[test]
fn test_lu_interchange_bit_identical_f64() {
    for (idx, &n) in SIZES.iter().enumerate() {
        let a = random_matrix::<f64>(n, 1000 + idx as u64);
        check_unblocked_variants(&format!("f64 n={n}"), &a);
    }
}

#[test]
fn test_lu_interchange_bit_identical_f32() {
    for (idx, &n) in SIZES.iter().enumerate().filter(|&(_, &n)| n <= 200) {
        let a = random_matrix::<f32>(n, 2000 + idx as u64);
        check_unblocked_variants(&format!("f32 n={n}"), &a);
    }
}

#[test]
fn test_lu_interchange_bit_identical_complex() {
    for (idx, &n) in SIZES.iter().enumerate().filter(|&(_, &n)| n <= 100) {
        let a64 = random_matrix::<Complex64>(n, 3000 + idx as u64);
        check_unblocked_variants(&format!("c64 n={n}"), &a64);
        let a32 = random_matrix::<Complex32>(n, 4000 + idx as u64);
        check_unblocked_variants(&format!("c32 n={n}"), &a32);
    }
}

#[test]
fn test_lu_interchange_bit_identical_singular() {
    // An exactly zero column and row: the error index must match, and for the
    // rows reached before the failure the work must be identical.
    for &(n, col, row) in &[
        (5usize, 2usize, 4usize),
        (40, 7, 39),
        (130, 70, 129),
        (130, 3, 5),
    ] {
        let a = singular_matrix::<f64>(n, 5000 + n as u64, col, row);
        check_unblocked_variants(&format!("singular n={n} col={col}"), &a);
    }
}

/// Runs the production `factor_panel` and the oracle's on copies of `a` for
/// the panel `jb..jb + w` and compares the results bit for bit: with the row
/// interchanges applied to whole rows (`swap_from = 0`, as the oracle does),
/// and with them deferred for the columns left of the panel (`swap_from =
/// jb`) and applied afterwards by `apply_row_swaps`, as the blocked drivers
/// do.
fn check_factor_panel<T: BitScalar>(label: &str, a: &Mat<T>, jb: usize, w: usize) {
    let n = a.nrows();
    let tol = lu_pivot_tolerances(a.as_ref());
    let mut want = a.clone();
    let mut want_piv = vec![0usize; n];
    let mut want_swaps = 0usize;
    let want_res = oracle::factor_panel(&mut want, &mut want_piv, &mut want_swaps, n, jb, w, &tol);
    for swap_from in [0, jb] {
        let mut got = a.clone();
        let mut got_piv = vec![0usize; n];
        let mut got_swaps = 0usize;
        let got_res = Lu::<T>::factor_panel(
            &mut got,
            &mut got_piv,
            &mut got_swaps,
            n,
            jb,
            w,
            swap_from,
            &tol,
        );
        let label = format!("{label} factor_panel(jb={jb}, w={w}, swap_from={swap_from})");
        assert_eq!(got_res, want_res, "{label}: outcome differs");
        if got_res.is_err() {
            continue;
        }
        apply_row_swaps(&mut got, &got_piv, jb, jb + w, 0, swap_from);
        assert_eq!(got_piv, want_piv, "{label}: pivots differ");
        assert_eq!(got_swaps, want_swaps, "{label}: swap counts differ");
        assert_same_bits(&label, &got, &want);
    }
}

#[test]
fn test_lu_factor_panel_bit_identical() {
    // Panels at the start, in the middle (with columns on both sides, which
    // the row interchanges also move) and at the end of the matrix.
    for &(n, jb, w) in &[
        (1usize, 0usize, 1usize),
        (17, 0, 17),
        (17, 5, 7),
        (64, 32, 32),
        (100, 0, 1),
        (100, 37, 23),
        (130, 64, 64),
        (130, 129, 1),
    ] {
        let seed = 7000 + n as u64 + jb as u64;
        check_factor_panel(&format!("f64 n={n}"), &random_matrix::<f64>(n, seed), jb, w);
        check_factor_panel(
            &format!("c64 n={n}"),
            &random_matrix::<Complex64>(n, seed + 100),
            jb,
            w,
        );
    }
    // A zero pivot inside the panel.
    let a = singular_matrix::<f64>(90, 7200, 50, 89);
    check_factor_panel("singular n=90", &a, 40, 20);
}

/// `max |PA − LU| / max |A|` for the factors `lu` and pivots of `a`, using
/// the triangular structure of `L` and `U` (about `n³ / 3` multiply-adds).
fn backward_error<T: BitScalar>(a: &Mat<T>, lu: &Mat<T>, pivot: &[usize]) -> f64 {
    let n = a.nrows();
    let mut pa = a.clone();
    for (k, &pk) in pivot.iter().enumerate() {
        for j in 0..n {
            let tmp = pa[(k, j)];
            pa[(k, j)] = pa[(pk, j)];
            pa[(pk, j)] = tmp;
        }
    }
    let f = lu.as_ref();
    let (mut worst, mut a_max) = (0.0f64, 0.0f64);
    let mut r = vec![T::zero(); n];
    for j in 0..n {
        for (i, ri) in r.iter_mut().enumerate() {
            *ri = pa[(i, j)];
            a_max = a_max.max(ri.mag());
        }
        // Column j of L·U: the sum over k <= j of L[:, k] · U[k, j], L[k, k] = 1.
        for k in 0..=j {
            let u_kj = f[(k, j)];
            r[k] = r[k] - u_kj;
            let l_col = f.col_as_slice(k).expect("contiguous column");
            for (ri, &l_ik) in r[k + 1..].iter_mut().zip(&l_col[k + 1..]) {
                *ri = *ri - l_ik * u_kj;
            }
        }
        for ri in &r {
            worst = worst.max(ri.mag());
        }
    }
    if a_max == 0.0 { worst } else { worst / a_max }
}

/// Bound `c` of the backward-error check `max |PA − LU| / max |A| <= c · n ·
/// eps`; the oracle's factors of the test matrices stay well below it.
const BACKWARD_ERROR_C: f64 = 2.0;

/// Checks a blocked driver against the oracle's driver with the same panel
/// width: same outcome (or error), same pivots and swap count, and a small
/// backward error.
fn assert_close<T: BitScalar>(
    label: &str,
    a: &Mat<T>,
    got: Result<Lu<T>, LuError>,
    want: Result<Factors<T>, LuError>,
) {
    match (got, want) {
        (Ok(lu), Ok((factors, pivot, num_swaps))) => {
            assert_eq!(lu.pivot, pivot, "{label}: pivots differ");
            assert_eq!(lu.num_swaps, num_swaps, "{label}: swap counts differ");
            let n = a.nrows();
            let bound = BACKWARD_ERROR_C * n.max(1) as f64 * T::EPS;
            let err = backward_error(a, &lu.lu, &lu.pivot);
            assert!(
                err <= bound,
                "{label}: backward error {err:e} exceeds {bound:e} (oracle: {:e})",
                backward_error(a, &factors, &pivot)
            );
        }
        (Err(got_err), Err(want_err)) => {
            assert_eq!(got_err, want_err, "{label}: errors differ");
        }
        (got, want) => panic!(
            "{label}: outcome differs: got {:?}, oracle {:?}",
            got.map(|_| ()),
            want.map(|_| ())
        ),
    }
}

/// Block sizes for a matrix of order `n`: the degenerate widths 1, 2 and 3
/// only while they stay cheap, then the default and wider panels (129 makes
/// `L11` span two diagonal blocks of the TRSM; 301 > n is unblocked).
fn block_sizes(n: usize) -> &'static [usize] {
    if n <= 100 {
        &[1, 2, 3, 8, 32, 64, 100, 301]
    } else {
        &[8, 32, 64, 129, 301]
    }
}

/// Compares the blocked and recursive drivers against the oracle's.
fn check_blocked_variants<T: BitScalar>(label: &str, a: &Mat<T>, block_sizes: &[usize]) {
    let ar = a.as_ref();
    for &nb in block_sizes {
        assert_close(
            &format!("{label} compute_with_block_size(nb={nb})"),
            a,
            Lu::compute_with_block_size(ar, nb),
            oracle::compute_with_block_size(ar, nb, gemm::<T>),
        );
    }
    let nb = crate::workspace::optimal_block_size_lu(a.nrows(), a.ncols());
    assert_close(
        &format!("{label} compute_blocked(nb={nb})"),
        a,
        Lu::compute_blocked(ar),
        oracle::compute_with_block_size(ar, nb, gemm::<T>),
    );
    assert_close(
        &format!("{label} compute_recursive"),
        a,
        Lu::compute_recursive(ar),
        oracle::compute_recursive(ar),
    );
}

#[test]
fn test_lu_blocked_backward_error_f64() {
    for (idx, &n) in SIZES.iter().enumerate() {
        let a = random_matrix::<f64>(n, 1000 + idx as u64);
        check_blocked_variants(&format!("f64 n={n}"), &a, block_sizes(n));
    }
}

#[test]
fn test_lu_blocked_backward_error_f32_complex() {
    for (idx, &n) in SIZES.iter().enumerate().filter(|&(_, &n)| n <= 200) {
        let a = random_matrix::<f32>(n, 2000 + idx as u64);
        check_blocked_variants(&format!("f32 n={n}"), &a, &[1, 7, 32, 64]);
    }
    for (idx, &n) in SIZES.iter().enumerate().filter(|&(_, &n)| n <= 100) {
        let a64 = random_matrix::<Complex64>(n, 3000 + idx as u64);
        check_blocked_variants(&format!("c64 n={n}"), &a64, &[1, 8, 32]);
        let a32 = random_matrix::<Complex32>(n, 4000 + idx as u64);
        check_blocked_variants(&format!("c32 n={n}"), &a32, &[1, 8, 32]);
    }
}

#[test]
fn test_lu_blocked_singular_matches_oracle() {
    // An exactly zero column and row: every driver reports the oracle's index.
    for &(n, col, row) in &[
        (5usize, 2usize, 4usize),
        (40, 7, 39),
        (130, 70, 129),
        (130, 3, 5),
    ] {
        let a = singular_matrix::<f64>(n, 5000 + n as u64, col, row);
        check_blocked_variants(&format!("singular n={n} col={col}"), &a, &[1, 8, 32, 64]);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_lu_parallel_matches_sequential_bitwise() {
    // Large enough for the parallel TRSM and GEMM paths (work gates of about
    // 128³ multiply-adds), with L11 inside one diagonal block of the TRSM and
    // across two (nb = 129).
    for (idx, &n) in [65usize, 300, 401].iter().enumerate() {
        let a = random_matrix::<f64>(n, 6000 + idx as u64);
        for &nb in &[8usize, 32, 64, 129] {
            let label = format!("par n={n} nb={nb}");
            let seq = Lu::compute_with_block_size(a.as_ref(), nb).expect("non-singular");
            let par =
                Lu::compute_blocked_par_with_block_size(a.as_ref(), nb).expect("non-singular");
            assert_eq!(par.pivot, seq.pivot, "{label}: pivots differ");
            assert_eq!(par.num_swaps, seq.num_swaps, "{label}: swap counts differ");
            assert_same_bits(&label, &par.lu, &seq.lu);
        }
        let seq = Lu::compute_blocked(a.as_ref()).expect("non-singular");
        let par = Lu::compute_blocked_par(a.as_ref()).expect("non-singular");
        assert_eq!(par.pivot, seq.pivot, "par n={n}: pivots differ");
        assert_same_bits(&format!("par n={n} compute_blocked_par"), &par.lu, &seq.lu);
    }
    // Errors are reported identically.
    let a = singular_matrix::<f64>(130, 5130, 70, 129);
    assert_eq!(
        Lu::compute_blocked_par_with_block_size(a.as_ref(), 32).map(|_| ()),
        Lu::compute_with_block_size(a.as_ref(), 32).map(|_| ())
    );
}
