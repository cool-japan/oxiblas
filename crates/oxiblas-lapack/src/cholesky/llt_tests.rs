//! Tests for the diagonal-block kernel and the blocked drivers of
//! [`Cholesky`].
//!
//! - `unblocked_factor_block` streams down columns instead of striding along
//!   rows, with the same arithmetic per entry: it, and `compute`, which now
//!   runs it on the whole matrix, must reproduce the previous row-oriented
//!   loops (kept in `oracle`) bit for bit, errors included.
//! - The blocked, recursive and parallel drivers work in place: TRSM on views
//!   and the lower triangle of the trailing update as GEMMs on its blocks
//!   (`alpha = −1`, `beta = 1`). The previous drivers formed the whole
//!   `L21 · L21ᵀ` in a temporary and subtracted its lower triangle, so the
//!   summation order of the update differs and the last bits can change. The
//!   drivers are checked for a backward error `max |A − LLᵀ| / max |A|` of at
//!   most `c · n · eps`, an exactly zero strict upper triangle and the
//!   failure index of non positive definite matrices.
//! - The parallel drivers must equal the sequential ones bit for bit: TRSM
//!   and GEMM give bitwise the same result for any `Par`.

use super::*;
use crate::lu::tolerance::llt_pivot_tolerances;

/// The previous row-oriented loops, kept as the reference.
mod oracle {
    use super::*;

    /// The previous `Cholesky::compute` (Cholesky-Banachiewicz, row by row).
    pub(super) fn compute<T: Field + Real + bytemuck::Zeroable>(
        a: MatRef<'_, T>,
    ) -> Result<Mat<T>, CholeskyError> {
        let n = a.nrows();
        let mut l = Mat::zeros(n, n);
        let col_tol = llt_pivot_tolerances(a);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = T::zero();
                if j == i {
                    for k in 0..j {
                        sum = sum + l[(j, k)] * l[(j, k)];
                    }
                    let diag = a[(i, i)] - sum;
                    if diag <= col_tol[i] {
                        return Err(CholeskyError::NotPositiveDefinite { index: i });
                    }
                    l[(i, j)] = Real::sqrt(diag);
                } else {
                    for k in 0..j {
                        sum = sum + l[(i, k)] * l[(j, k)];
                    }
                    l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
                }
            }
        }
        Ok(l)
    }

    /// The previous `Cholesky::unblocked_factor_block`.
    pub(super) fn unblocked_factor_block<T: Field + Real>(
        l: &mut Mat<T>,
        jb: usize,
        jb_size: usize,
        col_tol: &[T],
    ) -> Result<(), CholeskyError> {
        for i in 0..jb_size {
            let gi = jb + i;
            let mut sum = T::zero();
            for k in jb..gi {
                sum = sum + l[(gi, k)] * l[(gi, k)];
            }
            let diag = l[(gi, gi)] - sum;
            if diag <= col_tol[gi] {
                return Err(CholeskyError::NotPositiveDefinite { index: gi });
            }
            l[(gi, gi)] = Real::sqrt(diag);
            for j in (i + 1)..jb_size {
                let gj = jb + j;
                let mut sum = T::zero();
                for k in jb..gi {
                    sum = sum + l[(gj, k)] * l[(gi, k)];
                }
                l[(gj, gi)] = (l[(gj, gi)] - sum) / l[(gi, gi)];
            }
        }
        Ok(())
    }
}

/// Real scalar types under test.
trait TestReal: Field + Real + GemmKernel + bytemuck::Zeroable {
    /// Machine epsilon.
    const EPS: f64;
    fn of_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
}

impl TestReal for f64 {
    const EPS: f64 = f64::EPSILON;
    fn of_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl TestReal for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn of_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// Deterministic xorshift64 generator (no external dependency).
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Uniform sample in `[-1, 1)`.
    fn next_signed(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        2.0 * ((x >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }
}

/// Symmetric matrix with entries in `[-1, 1)` off the diagonal and `n` on it
/// (strictly diagonally dominant, so positive definite and well conditioned).
/// Both triangles are stored.
fn random_spd<T: TestReal>(n: usize, seed: u64) -> Mat<T> {
    let mut rng = XorShift64::new(seed);
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        a[(j, j)] = T::of_f64(n as f64);
        for i in (j + 1)..n {
            let v = T::of_f64(rng.next_signed());
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    a
}

/// The lower triangle of `a` in an otherwise zero matrix, as the drivers
/// start from.
fn lower_copy<T: TestReal>(a: &Mat<T>) -> Mat<T> {
    let n = a.nrows();
    let mut l = Mat::zeros(n, n);
    for j in 0..n {
        for i in j..n {
            l[(i, j)] = a[(i, j)];
        }
    }
    l
}

fn assert_same_bits<T: TestReal>(label: &str, got: &Mat<T>, want: &Mat<T>) {
    let n = want.nrows();
    assert_eq!(got.nrows(), n, "{label}: sizes differ");
    for j in 0..n {
        for i in 0..n {
            assert_eq!(
                got[(i, j)].bits(),
                want[(i, j)].bits(),
                "{label}: L({i}, {j}) differs: {} vs {}",
                got[(i, j)],
                want[(i, j)]
            );
        }
    }
}

/// Runs the production diagonal-block kernel and the oracle's on the lower
/// triangle of `a` for the block `jb..jb + w`, bit for bit.
fn check_block_kernel<T: TestReal>(label: &str, a: &Mat<T>, jb: usize, w: usize) {
    let tol = llt_pivot_tolerances(a.as_ref());
    let mut got = lower_copy(a);
    let mut want = lower_copy(a);
    let got_res = Cholesky::<T>::unblocked_factor_block(&mut got, jb, w, &tol);
    let want_res = oracle::unblocked_factor_block(&mut want, jb, w, &tol);
    let label = format!("{label} block(jb={jb}, w={w})");
    assert_eq!(got_res, want_res, "{label}: outcome differs");
    assert_same_bits(&label, &got, &want);
}

#[test]
fn test_llt_block_kernel_bit_identical() {
    for &(n, jb, w) in &[
        (1usize, 0usize, 1usize),
        (2, 0, 2),
        (17, 0, 17),
        (17, 5, 7),
        (64, 32, 32),
        (100, 37, 63),
        (130, 2, 128),
    ] {
        let seed = 100 + n as u64 + jb as u64;
        check_block_kernel(&format!("f64 n={n}"), &random_spd::<f64>(n, seed), jb, w);
        check_block_kernel(&format!("f32 n={n}"), &random_spd::<f32>(n, seed), jb, w);
    }
    // A diagonal entry that turns the block indefinite: same failure index.
    let mut a = random_spd::<f64>(40, 7);
    a[(25, 25)] = -1.0;
    check_block_kernel("indefinite n=40", &a, 10, 30);
}

fn assert_compute_same<T: TestReal>(label: &str, a: &Mat<T>) {
    let want = oracle::compute(a.as_ref());
    // `compute`, and the unblocked path of the blocked driver (nb > n).
    let variants = [
        ("compute", Cholesky::compute(a.as_ref())),
        (
            "compute_with_block_size(n + 1)",
            Cholesky::compute_with_block_size(a.as_ref(), a.nrows() + 1),
        ),
    ];
    for (name, got) in variants {
        let label = format!("{label} {name}");
        match (got, &want) {
            (Ok(got), Ok(want)) => assert_same_bits(&label, &got.l, want),
            (Err(got), Err(want)) => assert_eq!(got, *want, "{label}: errors differ"),
            (got, want) => panic!(
                "{label}: outcome differs: got {:?}, oracle {:?}",
                got.map(|_| ()),
                want.as_ref().map(|_| ())
            ),
        }
    }
}

#[test]
fn test_llt_compute_bit_identical() {
    for &n in &[
        0usize, 1, 2, 3, 5, 8, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 129, 200,
    ] {
        let seed = 300 + n as u64;
        assert_compute_same(&format!("f64 n={n}"), &random_spd::<f64>(n, seed));
        assert_compute_same(&format!("f32 n={n}"), &random_spd::<f32>(n, seed));
    }
    // Not positive definite at an interior index, and NaN / infinite entries
    // that propagate (the bits of the propagated values must match too).
    let mut indefinite = random_spd::<f64>(50, 9);
    indefinite[(31, 31)] = -2.0;
    assert_compute_same("indefinite n=50", &indefinite);
    let mut nan = random_spd::<f64>(20, 10);
    nan[(13, 4)] = f64::NAN;
    nan[(4, 13)] = f64::NAN;
    assert_compute_same("nan n=20", &nan);
    let mut inf = random_spd::<f64>(20, 11);
    inf[(7, 7)] = f64::INFINITY;
    assert_compute_same("inf n=20", &inf);
}

/// `max |A − L·Lᵀ| / max |A|` over the lower triangle (A is symmetric),
/// using the triangular structure of `L` (about `n³ / 6` multiply-adds).
fn backward_error<T: TestReal>(a: &Mat<T>, l: &Mat<T>) -> f64 {
    let n = a.nrows();
    let f = l.as_ref();
    let (mut worst, mut a_max) = (0.0f64, 0.0f64);
    let mut r = vec![T::zero(); n];
    for j in 0..n {
        for i in j..n {
            r[i] = a[(i, j)];
            a_max = a_max.max(Scalar::abs(r[i]).to_f64());
        }
        // Rows j.. of column j of L·Lᵀ: the sum over k <= j of L[:, k]·L[j, k].
        for k in 0..=j {
            let l_jk = f[(j, k)];
            let l_col = f.col_as_slice(k).expect("contiguous column");
            for (ri, &l_ik) in r[j..].iter_mut().zip(&l_col[j..]) {
                *ri = *ri - l_ik * l_jk;
            }
        }
        for &ri in &r[j..] {
            worst = worst.max(Scalar::abs(ri).to_f64());
        }
    }
    if a_max == 0.0 { worst } else { worst / a_max }
}

/// Bound `c` of the backward-error check `max |A − LLᵀ| / max |A| <= c · n ·
/// eps`.
const BACKWARD_ERROR_C: f64 = 2.0;

fn check_factor<T: TestReal>(label: &str, a: &Mat<T>, got: Result<Cholesky<T>, CholeskyError>) {
    let chol = got.unwrap_or_else(|e| panic!("{label}: {e}"));
    let l = chol.l_factor();
    let n = a.nrows();
    for j in 0..n {
        for i in 0..j {
            assert_eq!(
                l[(i, j)].bits(),
                T::zero().bits(),
                "{label}: L({i}, {j}) above the diagonal is {}",
                l[(i, j)]
            );
        }
    }
    let bound = BACKWARD_ERROR_C * n.max(1) as f64 * T::EPS;
    let err = backward_error(a, &l);
    assert!(
        err <= bound,
        "{label}: backward error {err:e} exceeds {bound:e}"
    );
}

type CholVariant<T> = (
    &'static str,
    fn(&Mat<T>) -> Result<Cholesky<T>, CholeskyError>,
);

/// Every driver whose update order differs from the unblocked algorithm.
fn blocked_variants<T: TestReal + Send + Sync>() -> Vec<CholVariant<T>> {
    let base: [CholVariant<T>; 2] = [
        ("compute_blocked", |a| Cholesky::compute_blocked(a.as_ref())),
        ("compute_recursive", |a| {
            Cholesky::compute_recursive(a.as_ref())
        }),
    ];
    #[cfg(feature = "parallel")]
    let par: [CholVariant<T>; 1] = [("compute_blocked_par", |a| {
        Cholesky::compute_blocked_par(a.as_ref())
    })];
    #[cfg(not(feature = "parallel"))]
    let par: [CholVariant<T>; 0] = [];
    base.into_iter().chain(par).collect()
}

#[test]
fn test_llt_blocked_backward_error() {
    for &n in &[
        1usize, 2, 3, 5, 8, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 129, 200, 257, 300,
    ] {
        let a = random_spd::<f64>(n, 500 + n as u64);
        let block_sizes: &[usize] = if n <= 100 {
            &[1, 3, 8, 32, 64, 100, 301]
        } else {
            &[8, 32, 64, 129, 301]
        };
        for &nb in block_sizes {
            check_factor(
                &format!("f64 n={n} nb={nb}"),
                &a,
                Cholesky::compute_with_block_size(a.as_ref(), nb),
            );
        }
        for (name, factor) in blocked_variants::<f64>() {
            check_factor(&format!("f64 n={n} {name}"), &a, factor(&a));
        }
    }
    for &n in &[7usize, 33, 64, 65, 150] {
        let a = random_spd::<f32>(n, 600 + n as u64);
        for &nb in &[1usize, 8, 32] {
            check_factor(
                &format!("f32 n={n} nb={nb}"),
                &a,
                Cholesky::compute_with_block_size(a.as_ref(), nb),
            );
        }
        for (name, factor) in blocked_variants::<f32>() {
            check_factor(&format!("f32 n={n} {name}"), &a, factor(&a));
        }
    }
}

#[test]
fn test_llt_blocked_not_positive_definite_index() {
    // A negative diagonal entry makes the radicand of its column negative
    // whatever the summation order, and every earlier column is positive
    // definite: all drivers report that index (inside the first block, on a
    // block boundary and in a later block).
    for &(n, k) in &[(40usize, 3usize), (130, 64), (130, 100), (200, 199)] {
        let mut a = random_spd::<f64>(n, 700 + k as u64);
        a[(k, k)] = -1.0;
        let want = Err(CholeskyError::NotPositiveDefinite { index: k });
        assert_eq!(
            Cholesky::compute(a.as_ref()).map(|_| ()),
            want,
            "compute n={n}"
        );
        for &nb in &[1usize, 8, 32, 64] {
            assert_eq!(
                Cholesky::compute_with_block_size(a.as_ref(), nb).map(|_| ()),
                want,
                "n={n} nb={nb}"
            );
        }
        for (name, factor) in blocked_variants::<f64>() {
            assert_eq!(factor(&a).map(|_| ()), want, "n={n} {name}");
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_llt_parallel_matches_sequential_bitwise() {
    // Large enough for the parallel TRSM and GEMM paths, with L11 inside one
    // diagonal block of the TRSM and across two (nb = 129).
    for &n in &[65usize, 300, 401] {
        let a = random_spd::<f64>(n, 800 + n as u64);
        for &nb in &[8usize, 32, 64, 129] {
            let seq = Cholesky::compute_with_block_size(a.as_ref(), nb).expect("SPD");
            let par = Cholesky::compute_blocked_par_with_block_size(a.as_ref(), nb).expect("SPD");
            assert_same_bits(&format!("par n={n} nb={nb}"), &par.l, &seq.l);
        }
        let seq = Cholesky::compute_blocked(a.as_ref()).expect("SPD");
        let par = Cholesky::compute_blocked_par(a.as_ref()).expect("SPD");
        assert_same_bits(&format!("par n={n} compute_blocked_par"), &par.l, &seq.l);
    }
}
