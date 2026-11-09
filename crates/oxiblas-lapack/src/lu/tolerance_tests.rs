//! Tests for the per-column (scale-invariant) singularity tolerance of every
//! LU variant, and for the shared tolerance helpers.

use super::*;
use crate::lu::{BandLu, BandLuError, Lu, LuError, LuFullPiv, LuFullPivError, LuRook, LuRookError};
use crate::lu::{band_to_dense, dense_to_band};
use oxiblas_matrix::Mat;

/// 2^-64 and 2^64: exact powers of two, far from overflow and underflow for
/// the O(1)..O(n) entries used below.
const TWO_POW_M64: f64 = 1.0 / 18_446_744_073_709_551_616.0;
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
const POW2_SCALES: [f64; 2] = [TWO_POW_M64, TWO_POW_64];

/// Deterministic xorshift64 generator (no external dependency).
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
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

/// Random matrix with entries in `[-1, 1)`, plus `shift` on the diagonal.
fn random_matrix(n: usize, seed: u64, shift: f64) -> Mat<f64> {
    let mut rng = XorShift64::new(seed);
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = rng.next_signed();
        }
        a[(j, j)] += shift;
    }
    a
}

fn scaled(a: &Mat<f64>, s: f64) -> Mat<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut b = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            b[(i, j)] = a[(i, j)] * s;
        }
    }
    b
}

/// `A · diag(c)`: column `j` multiplied by `c[j]`.
fn column_scaled(a: &Mat<f64>, c: &[f64]) -> Mat<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut b = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            b[(i, j)] = a[(i, j)] * c[j];
        }
    }
    b
}

/// One power of two in `2^-60 ..= 2^60` per column, drawn from `seed`.
fn column_factors(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    (0..n)
        .map(|_| 2.0f64.powi((rng.next_u64() % 121) as i32 - 60))
        .collect()
}

/// A random 6x6 matrix whose column 3 is exactly zero.
fn zero_column_matrix() -> Mat<f64> {
    let mut a = random_matrix(6, 77, 0.0);
    for i in 0..6 {
        a[(i, 3)] = 0.0;
    }
    a
}

/// A random 6x6 matrix whose row 2 is exactly zero.
fn zero_row_matrix() -> Mat<f64> {
    let mut a = random_matrix(6, 78, 0.0);
    for j in 0..6 {
        a[(2, j)] = 0.0;
    }
    a
}

/// Rank 2: the third row is the sum of the first two.
fn rank_deficient_3x3() -> Mat<f64> {
    Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[5.0, 7.0, 9.0]])
}

/// Diagonal matrix with the given entries.
fn diag(entries: &[f64]) -> Mat<f64> {
    let n = entries.len();
    let mut a = Mat::zeros(n, n);
    for (j, &d) in entries.iter().enumerate() {
        a[(j, j)] = d;
    }
    a
}

/// FEM stiffness matrix with a penalty term: the 1-D Laplacian
/// `tridiag(-1, 2, -1)` of order `n` plus `1e20 · e_i e_iᵀ`, which imposes a
/// boundary condition at node `i` by penalty.
fn fem_penalty(n: usize, i: usize) -> Mat<f64> {
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        a[(j, j)] = 2.0;
        if j + 1 < n {
            a[(j + 1, j)] = -1.0;
            a[(j, j + 1)] = -1.0;
        }
    }
    a[(i, i)] += 1.0e20;
    a
}

/// Matrices that factor exactly but whose columns differ wildly in scale.
/// The previous global reference `eps · n · max |a_ij|` reported every one of
/// them singular; a per-column reference (like LAPACK, which only rejects
/// exactly zero pivots) accepts them.
fn badly_scaled_matrices() -> Vec<(&'static str, Mat<f64>)> {
    vec![
        ("diag(1e6, 1e-10)", diag(&[1.0e6, 1.0e-10])),
        ("diag(1e20, 1)", diag(&[1.0e20, 1.0])),
        ("FEM penalty", fem_penalty(6, 2)),
    ]
}

/// Row-major copy of a square matrix (the layout `dense_to_band` reads).
fn row_major(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = a[(i, j)];
        }
    }
    v
}

/// Checks that `scaled_lu` (from `A · diag(c)`) equals `base` (from `A`):
/// identical pivots and outcome, L bit-identical and column `j` of U
/// multiplied by exactly `c[j]`.
fn assert_lu_column_scaled(
    label: &str,
    base: &Result<Lu<f64>, LuError>,
    scaled_lu: &Result<Lu<f64>, LuError>,
    c: &[f64],
) {
    match (base, scaled_lu) {
        (Ok(base), Ok(other)) => {
            assert_eq!(base.pivot(), other.pivot(), "{label}: pivots differ");
            assert_combined_scaled(label, base.lu_matrix(), other.lu_matrix(), c);
        }
        (Err(e1), Err(e2)) => assert_eq!(e1, e2, "{label}: errors differ"),
        (b, o) => panic!(
            "{label}: outcome depends on scale: A -> {:?}, scaled A -> {:?}",
            b.as_ref().map(|_| ()),
            o.as_ref().map(|_| ())
        ),
    }
}

/// Combined LU storage: strictly lower part (L) bit-equal, upper part (U)
/// equal to the base value times the column factor `c[j]`, bit for bit.
fn assert_combined_scaled(
    label: &str,
    base: oxiblas_matrix::MatRef<'_, f64>,
    other: oxiblas_matrix::MatRef<'_, f64>,
    c: &[f64],
) {
    let n = base.nrows();
    assert_eq!(other.nrows(), n, "{label}: sizes differ");
    for j in 0..n {
        for i in 0..n {
            let expected = if i <= j {
                base[(i, j)] * c[j]
            } else {
                base[(i, j)]
            };
            assert_eq!(
                other[(i, j)].to_bits(),
                expected.to_bits(),
                "{label}: entry ({i}, {j}) is {} but expected {}",
                other[(i, j)],
                expected
            );
        }
    }
}

type LuVariant = (&'static str, fn(&Mat<f64>) -> Result<Lu<f64>, LuError>);

fn lu_variants() -> Vec<LuVariant> {
    let base: [LuVariant; 7] = [
        ("compute", |a| Lu::compute(a.as_ref())),
        ("compute_with_block_size(1)", |a| {
            Lu::compute_with_block_size(a.as_ref(), 1)
        }),
        ("compute_with_block_size(8)", |a| {
            Lu::compute_with_block_size(a.as_ref(), 8)
        }),
        ("compute_with_block_size(32)", |a| {
            Lu::compute_with_block_size(a.as_ref(), 32)
        }),
        ("compute_blocked", |a| Lu::compute_blocked(a.as_ref())),
        ("compute_recursive", |a| Lu::compute_recursive(a.as_ref())),
        ("compute_auto", |a| Lu::compute_auto(a.as_ref())),
    ];
    #[cfg(feature = "parallel")]
    let par: [LuVariant; 2] = [
        ("compute_blocked_par", |a| {
            Lu::compute_blocked_par(a.as_ref())
        }),
        ("compute_blocked_par_with_block_size(8)", |a| {
            Lu::compute_blocked_par_with_block_size(a.as_ref(), 8)
        }),
    ];
    #[cfg(not(feature = "parallel"))]
    let par: [LuVariant; 0] = [];
    base.into_iter().chain(par).collect()
}

/// Test matrices: well-conditioned and plain random ones of several sizes
/// (the larger ones exercise the blocked and recursive paths), exactly
/// singular ones, a numerically rank-deficient one and badly scaled ones.
fn lu_test_matrices() -> Vec<(String, Mat<f64>)> {
    let mut mats = Vec::new();
    for (idx, &n) in [1usize, 2, 5, 33, 70, 150].iter().enumerate() {
        mats.push((
            format!("random n={n}"),
            random_matrix(n, 10 + idx as u64, 0.0),
        ));
        mats.push((
            format!("dominant n={n}"),
            random_matrix(n, 20 + idx as u64, n as f64),
        ));
    }
    mats.push(("zero column".to_string(), zero_column_matrix()));
    mats.push(("zero row".to_string(), zero_row_matrix()));
    mats.push(("rank-deficient 3x3".to_string(), rank_deficient_3x3()));
    for (name, a) in badly_scaled_matrices() {
        mats.push((name.to_string(), a));
    }
    mats
}

#[test]
fn test_lu_tolerance_helpers_skip_non_finite() {
    assert_eq!(finite_max([1.0f64, -3.0, 2.0]), 2.0);
    assert_eq!(finite_max([f64::NAN, 1.5, f64::INFINITY]), 1.5);
    assert_eq!(finite_max([f64::NEG_INFINITY, -2.0]), 0.0);
    assert_eq!(finite_max(core::iter::empty::<f64>()), 0.0);
    // eps * n is formed first, so a scale near the largest finite value does
    // not overflow to infinity.
    let tol = pivot_tolerance(2, f64::MAX);
    assert!(tol.is_finite() && tol > 0.0, "tol = {tol}");
    assert_eq!(pivot_tolerance(4, 1.0f64), 4.0 * f64::EPSILON);
    assert_eq!(pivot_tolerance(3, 0.0f32), 0.0);
    assert_eq!(
        into_pivot_tolerances(4, vec![1.0f64, 0.0, 0.5]),
        vec![4.0 * f64::EPSILON, 0.0, 2.0 * f64::EPSILON]
    );

    let a: Mat<f64> = Mat::from_rows(&[
        &[1.0, f64::NAN, 0.5],
        &[-7.0, f64::NEG_INFINITY, 2.0],
        &[3.0, -4.0, f64::INFINITY],
    ]);
    // One reference per column; NaN and ±∞ never count.
    assert_eq!(column_max_abs_finite(a.as_ref()), vec![7.0, 4.0, 2.0]);
    // The symmetric column of the lower triangle (1; -7, -inf; 3, -4, inf)
    // also counts row j: column 0 is (1, -7, 3), column 1 is (-7, -inf, -4)
    // and column 2 is (3, -4, inf).
    assert_eq!(
        symmetric_column_max_abs_finite(a.as_ref()),
        vec![7.0, 7.0, 4.0]
    );
    // Diagonal references: real part, floored at zero, non-finite skipped.
    assert_eq!(diagonal_finite(a.as_ref()), vec![1.0, 0.0, 0.0]);
    let negative: Mat<f64> = Mat::from_rows(&[&[-2.0, 0.0], &[0.0, 3.0]]);
    assert_eq!(diagonal_finite(negative.as_ref()), vec![0.0, 3.0]);
    assert_eq!(
        lu_pivot_tolerances(negative.as_ref()),
        vec![4.0 * f64::EPSILON, 6.0 * f64::EPSILON]
    );
    assert_eq!(
        llt_pivot_tolerances(negative.as_ref()),
        vec![0.0, 6.0 * f64::EPSILON]
    );

    // A transposed view has non-contiguous columns: the indexing fallback
    // must agree with the slice path.
    let at = a.as_ref().transpose();
    assert!(at.col_as_slice(0).is_none());
    assert_eq!(column_max_abs_finite(at), vec![1.0, 7.0, 4.0]);
    // Lower triangle of the transpose: (1; NaN, -inf; 0.5, 2, inf).
    assert_eq!(symmetric_column_max_abs_finite(at), vec![1.0, 2.0, 2.0]);
}

#[test]
fn test_lu_tolerance_partial_power_of_two_scaling() {
    for (variant, factor) in lu_variants() {
        for (name, a) in lu_test_matrices() {
            let base = factor(&a);
            for s in POW2_SCALES {
                let label = format!("Lu::{variant} on {name} scaled by {s:e}");
                let c = vec![s; a.ncols()];
                assert_lu_column_scaled(&label, &base, &factor(&scaled(&a, s)), &c);
            }
        }
    }
}

#[test]
fn test_lu_tolerance_partial_column_scaling() {
    // Partial pivoting compares the entries of one column only, so scaling
    // the columns by powers of two leaves the pivots and the multipliers
    // unchanged and scales column j of U, and its reference, by c_j.
    for (variant, factor) in lu_variants() {
        for (name, a) in lu_test_matrices() {
            let base = factor(&a);
            for seed in [1u64, 2] {
                let c = column_factors(a.ncols(), seed);
                let label = format!("Lu::{variant} on {name} with column scaling #{seed}");
                assert_lu_column_scaled(&label, &base, &factor(&column_scaled(&a, &c)), &c);
            }
        }
    }
}

#[test]
fn test_lu_tolerance_partial_power_of_two_scaling_f32() {
    let a64 = random_matrix(40, 99, 0.0);
    let mut a: Mat<f32> = Mat::zeros(40, 40);
    for j in 0..40 {
        for i in 0..40 {
            a[(i, j)] = a64[(i, j)] as f32;
        }
    }
    for s in [2.0f32.powi(-32), 2.0f32.powi(32)] {
        let mut b: Mat<f32> = Mat::zeros(40, 40);
        for j in 0..40 {
            for i in 0..40 {
                b[(i, j)] = a[(i, j)] * s;
            }
        }
        for nb in [1usize, 8, 64] {
            let base = Lu::compute_with_block_size(a.as_ref(), nb).expect("random f32 matrix");
            let other = Lu::compute_with_block_size(b.as_ref(), nb)
                .unwrap_or_else(|e| panic!("f32 nb={nb} scaled by {s:e}: {e}"));
            assert_eq!(base.pivot(), other.pivot(), "f32 nb={nb} s={s:e}: pivots");
            let (bm, om) = (base.lu_matrix(), other.lu_matrix());
            for j in 0..40 {
                for i in 0..40 {
                    let expected = if i <= j { bm[(i, j)] * s } else { bm[(i, j)] };
                    assert_eq!(
                        om[(i, j)].to_bits(),
                        expected.to_bits(),
                        "f32 nb={nb} s={s:e}: entry ({i}, {j})"
                    );
                }
            }
        }
    }
}

fn assert_pivoted_scaled<E: core::fmt::Debug + PartialEq>(
    label: &str,
    base: Result<(Vec<usize>, Vec<usize>, Mat<f64>), E>,
    other: Result<(Vec<usize>, Vec<usize>, Mat<f64>), E>,
    s: f64,
) {
    match (base, other) {
        (Ok((rows, cols, lu)), Ok((rows_s, cols_s, lu_s))) => {
            assert_eq!(rows, rows_s, "{label}: row pivots differ");
            assert_eq!(cols, cols_s, "{label}: column pivots differ");
            let c = vec![s; lu.ncols()];
            assert_combined_scaled(label, lu.as_ref(), lu_s.as_ref(), &c);
        }
        (Err(e1), Err(e2)) => assert_eq!(e1, e2, "{label}: errors differ"),
        (b, o) => panic!(
            "{label}: outcome depends on scale: A -> {:?}, A*s -> {:?}",
            b.map(|_| ()),
            o.map(|_| ())
        ),
    }
}

fn full_piv_parts(a: &Mat<f64>) -> Result<(Vec<usize>, Vec<usize>, Mat<f64>), LuFullPivError> {
    LuFullPiv::compute(a.as_ref()).map(|f| {
        let mut lu = Mat::zeros(a.nrows(), a.ncols());
        lu.copy_from(&f.lu_matrix());
        (f.row_pivot().to_vec(), f.col_pivot().to_vec(), lu)
    })
}

fn rook_parts(a: &Mat<f64>) -> Result<(Vec<usize>, Vec<usize>, Mat<f64>), LuRookError> {
    LuRook::compute(a.as_ref()).map(|f| {
        let mut lu = Mat::zeros(a.nrows(), a.ncols());
        lu.copy_from(&f.lu_matrix());
        (f.row_pivot().to_vec(), f.col_pivot().to_vec(), lu)
    })
}

#[test]
fn test_lu_tolerance_full_and_rook_power_of_two_scaling() {
    // Rook and full pivoting compare entries of different columns, so only a
    // global scaling (not a column scaling) leaves their pivots unchanged.
    for (name, a) in lu_test_matrices() {
        let full = full_piv_parts(&a);
        let rook = rook_parts(&a);
        let full_rank = LuFullPiv::compute(a.as_ref()).map(|f| f.rank());
        for s in POW2_SCALES {
            let b = scaled(&a, s);
            assert_pivoted_scaled(
                &format!("LuFullPiv on {name} scaled by {s:e}"),
                full.clone(),
                full_piv_parts(&b),
                s,
            );
            assert_pivoted_scaled(
                &format!("LuRook on {name} scaled by {s:e}"),
                rook.clone(),
                rook_parts(&b),
                s,
            );
            assert_eq!(
                full_rank,
                LuFullPiv::compute(b.as_ref()).map(|f| f.rank()),
                "LuFullPiv rank on {name} scaled by {s:e}"
            );
        }
    }
}

#[test]
fn test_lu_tolerance_full_and_rook_follow_column_permutation() {
    // Every column permutation of diag(1e20, 1, 1e-20) factors: each pivot is
    // compared with the reference of the original column it comes from,
    // which the column interchanges of rook and full pivoting move around.
    // Testing the pivot of step k against the reference of input column k
    // would reject, for instance, the last pivot 1e-20 of the permutation
    // [2, 1, 0] against the reference 1e20 of input column 2.
    let d = [1.0e20, 1.0, 1.0e-20];
    for perm in [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut a = Mat::zeros(3, 3);
        for (col, &src) in perm.iter().enumerate() {
            a[(src, col)] = d[src];
        }
        let full = LuFullPiv::compute(a.as_ref())
            .unwrap_or_else(|e| panic!("LuFullPiv on permutation {perm:?}: {e}"));
        let rook = LuRook::compute(a.as_ref())
            .unwrap_or_else(|e| panic!("LuRook on permutation {perm:?}: {e}"));
        for det in [full.determinant(), rook.determinant()] {
            assert!(
                f64::abs(f64::abs(det) - 1.0) <= 4.0 * f64::EPSILON,
                "determinant {det} of permutation {perm:?}"
            );
        }
        // The rank keeps its global threshold `n · eps · max |a_ij|`: next to
        // 1e20, the entries 1 and 1e-20 are numerically zero.
        assert_eq!(full.rank(), 1, "LuFullPiv rank on permutation {perm:?}");
    }

    // Rook pivoting never interchanges the columns of a permuted diagonal
    // matrix. Here both strategies take 1e20 from input column 1 first and
    // swap it to position 0, which leaves the pivot 1e-20 of input column 0
    // (reference 1e-20) at position 1. Comparing it with the reference 1e20
    // of input column 1 would report a well-conditioned matrix singular:
    // scaling its columns gives [[0, 1], [1, 1e-20]].
    let swapped: Mat<f64> = Mat::from_rows(&[&[0.0, 1.0e20], &[1.0e-20, 1.0]]);
    for s in [1.0, TWO_POW_M64, TWO_POW_64] {
        let b = scaled(&swapped, s);
        let full = LuFullPiv::compute(b.as_ref())
            .unwrap_or_else(|e| panic!("LuFullPiv on the swapped fixture scaled by {s:e}: {e}"));
        assert_eq!(full.col_pivot(), &[1, 1]);
        let rook = LuRook::compute(b.as_ref())
            .unwrap_or_else(|e| panic!("LuRook on the swapped fixture scaled by {s:e}: {e}"));
        assert_eq!(rook.col_pivot(), &[1, 1]);
    }
}

/// Row-major dense band matrix with `kl` sub- and `ku` super-diagonals.
fn random_band_dense(n: usize, kl: usize, ku: usize, seed: u64, shift: f64) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in i.saturating_sub(kl)..=(i + ku).min(n - 1) {
            a[i * n + j] = rng.next_signed();
        }
        a[i * n + i] += shift;
    }
    a
}

/// Band test cases `(name, n, kl, ku, row-major dense)`.
fn band_cases() -> Vec<(String, usize, usize, usize, Vec<f64>)> {
    let mut cases: Vec<(String, usize, usize, usize, Vec<f64>)> = Vec::new();
    for &(n, kl, ku) in &[(1usize, 0usize, 0usize), (8, 1, 1), (30, 2, 3), (40, 4, 1)] {
        cases.push((
            format!("random n={n} kl={kl} ku={ku}"),
            n,
            kl,
            ku,
            random_band_dense(n, kl, ku, 31 + n as u64, 0.0),
        ));
    }
    // Exactly singular: column 2 of a tridiagonal matrix set to zero.
    let mut singular = random_band_dense(6, 1, 1, 5, 0.0);
    for i in 0..6 {
        singular[i * 6 + 2] = 0.0;
    }
    cases.push(("zero column".to_string(), 6, 1, 1, singular));
    for (name, a) in badly_scaled_matrices() {
        let n = a.nrows();
        let k = 1.min(n - 1);
        cases.push((name.to_string(), n, k, k, row_major(&a)));
    }
    cases
}

/// Checks a band LU of `A · diag(c)` against the band LU of `A`: identical
/// pivots and outcome, the multipliers of L bit-identical and every entry of
/// U in column `j` (band rows `0..=kl+ku`, fill-in included) scaled by `c[j]`.
fn assert_band_column_scaled(
    label: &str,
    base: &Result<BandLu<f64>, BandLuError>,
    other: Result<BandLu<f64>, BandLuError>,
    (kl, ku): (usize, usize),
    c: &[f64],
) {
    match (base, other) {
        (Ok(base), Ok(other)) => {
            assert_eq!(base.pivot(), other.pivot(), "{label}: pivots differ");
            let ldab = 2 * kl + ku + 1;
            for (idx, (&x, &y)) in base.ab().iter().zip(other.ab()).enumerate() {
                let expected = if idx % ldab <= kl + ku {
                    x * c[idx / ldab]
                } else {
                    x
                };
                assert_eq!(y.to_bits(), expected.to_bits(), "{label}: ab[{idx}]");
            }
        }
        (Err(e1), Err(e2)) => assert_eq!(*e1, e2, "{label}: errors differ"),
        (b, o) => panic!(
            "{label}: outcome depends on scale: A -> {:?}, scaled A -> {:?}",
            b.as_ref().map(|_| ()),
            o.map(|_| ())
        ),
    }
}

#[test]
fn test_lu_tolerance_band_scaling() {
    for (name, n, kl, ku, dense) in band_cases() {
        let ab = dense_to_band(&dense, n, kl, ku);
        let ldab = 2 * kl + ku + 1;
        let base = BandLu::compute(n, kl, ku, &ab);
        let mut scalings: Vec<(String, Vec<f64>)> = POW2_SCALES
            .iter()
            .map(|&s| (format!("scaled by {s:e}"), vec![s; n]))
            .collect();
        for seed in [3u64, 4] {
            scalings.push((format!("column scaling #{seed}"), column_factors(n, seed)));
        }
        for (how, c) in scalings {
            let ab_s: Vec<f64> = ab
                .iter()
                .enumerate()
                .map(|(idx, &x)| x * c[idx / ldab])
                .collect();
            assert_band_column_scaled(
                &format!("BandLu on {name} {how}"),
                &base,
                BandLu::compute(n, kl, ku, &ab_s),
                (kl, ku),
                &c,
            );
        }
    }
}

/// Relative residual `max |A x - b| / (max |A| · max |x|)` of a solve.
fn relative_residual(a: &Mat<f64>, x: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let n = a.nrows();
    let mut worst: f64 = 0.0;
    for i in 0..n {
        let mut ax = 0.0;
        for j in 0..n {
            ax += a[(i, j)] * x[(j, 0)];
        }
        worst = worst.max((ax - b[(i, 0)]).abs());
    }
    let a_max = finite_max(column_max_abs_finite(a.as_ref()));
    let x_max = finite_max(column_max_abs_finite(x.as_ref()));
    worst / (a_max * x_max)
}

#[test]
fn test_lu_tolerance_tiny_scale_accepted() {
    // A well-conditioned matrix scaled to 1e-30 is perfectly factorable; the
    // old absolute threshold eps * n rejected its very first pivot.
    for &n in &[2usize, 5, 40, 100] {
        let a = scaled(&random_matrix(n, 40 + n as u64, n as f64), 1.0e-30);
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            b[(i, 0)] = (0..n).map(|j| a[(i, j)]).sum::<f64>();
        }
        for (variant, factor) in lu_variants() {
            let lu = factor(&a).unwrap_or_else(|e| panic!("Lu::{variant} n={n} at 1e-30: {e}"));
            let x = lu.solve(b.as_ref()).expect("dimensions match");
            let res = relative_residual(&a, &x, &b);
            assert!(res < 1e-13, "Lu::{variant} n={n}: residual {res}");
        }
        assert!(
            LuFullPiv::compute(a.as_ref()).is_ok(),
            "LuFullPiv n={n} at 1e-30"
        );
        assert!(LuRook::compute(a.as_ref()).is_ok(), "LuRook n={n} at 1e-30");

        let kl = 1.min(n - 1);
        let ab = dense_to_band(&row_major(&a), n, kl, kl);
        assert!(
            BandLu::compute(n, kl, kl, &ab).is_ok(),
            "BandLu n={n} at 1e-30"
        );
    }
}

#[test]
fn test_lu_tolerance_singular_still_fails() {
    let rank_one: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
    let zero_col = zero_column_matrix();
    let zero_row = zero_row_matrix();
    let rank_two = rank_deficient_3x3();
    let zero: Mat<f64> = Mat::zeros(3, 3);
    for s in [1.0, TWO_POW_M64, TWO_POW_64, 1.0e-30, 1.0e30] {
        for (name, a, index) in [
            ("rank one", &rank_one, 1usize),
            ("zero column", &zero_col, 3),
            ("zero row", &zero_row, 5),
            ("rank-deficient 3x3", &rank_two, 2),
            ("zero", &zero, 0),
        ] {
            let b = scaled(a, s);
            for (variant, factor) in lu_variants() {
                assert_eq!(
                    factor(&b).map(|_| ()),
                    Err(LuError::Singular { index }),
                    "Lu::{variant} on {name} scaled by {s:e}"
                );
            }
            assert!(
                matches!(
                    LuFullPiv::compute(b.as_ref()),
                    Err(LuFullPivError::Singular { .. })
                ),
                "LuFullPiv on {name} scaled by {s:e}"
            );
            assert!(
                matches!(
                    LuRook::compute(b.as_ref()),
                    Err(LuRookError::Singular { .. })
                ),
                "LuRook on {name} scaled by {s:e}"
            );
        }
        // The rank-2 matrix leaves a rounding residue of about 1e-15 as the
        // last pivot: far below the reference of the column it belongs to.
        let b = scaled(&rank_two, s);
        assert_eq!(
            LuFullPiv::compute(b.as_ref()).map(|_| ()),
            Err(LuFullPivError::Singular { index: 2 }),
            "LuFullPiv on rank-deficient 3x3 scaled by {s:e}"
        );
        assert_eq!(
            LuRook::compute(b.as_ref()).map(|_| ()),
            Err(LuRookError::Singular { index: 2 }),
            "LuRook on rank-deficient 3x3 scaled by {s:e}"
        );

        let ab = dense_to_band(&[1.0 * s, 2.0 * s, 2.0 * s, 4.0 * s], 2, 1, 1);
        assert_eq!(
            BandLu::compute(2, 1, 1, &ab).map(|_| ()),
            Err(BandLuError::Singular { index: 1 }),
            "BandLu rank one scaled by {s:e}"
        );
    }
}

#[test]
fn test_lu_tolerance_badly_scaled_accepted() {
    // diag(1e6, 1e-10), diag(1e20, 1) and a penalised FEM matrix factor
    // exactly; every pivot is large next to the scale of its own column.
    // The global reference eps * n * max |a_ij| rejected them (for
    // diag(1e20, 1): 1 <= eps * 2 * 1e20 = 4.4e4), as 0.2.2 and LAPACK do not.
    for (name, a) in badly_scaled_matrices() {
        let n = a.nrows();
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            b[(i, 0)] = (0..n).map(|j| a[(i, j)]).sum::<f64>();
        }
        for (variant, factor) in lu_variants() {
            let lu = factor(&a).unwrap_or_else(|e| panic!("Lu::{variant} on {name}: {e}"));
            let x = lu.solve(b.as_ref()).expect("dimensions match");
            let res = relative_residual(&a, &x, &b);
            assert!(res < 1e-13, "Lu::{variant} on {name}: residual {res}");
        }
        assert!(
            LuFullPiv::compute(a.as_ref()).is_ok(),
            "LuFullPiv on {name}"
        );
        assert!(LuRook::compute(a.as_ref()).is_ok(), "LuRook on {name}");
        let k = 1.min(n - 1);
        let ab = dense_to_band(&row_major(&a), n, k, k);
        assert!(BandLu::compute(n, k, k, &ab).is_ok(), "BandLu on {name}");
    }
    // A pivot that is tiny relative to its own column is still rejected: in
    // [[1, 1], [1, 1 + eps]] the second pivot is the cancellation residue
    // eps, below eps * 2 * (1 + eps). The second column of diag(1, 1e-16), by
    // contrast, is small but not dependent, so its pivot is accepted.
    let near: Mat<f64> = Mat::from_rows(&[&[1.0, 1.0], &[1.0, 1.0 + f64::EPSILON]]);
    for (variant, factor) in lu_variants() {
        assert_eq!(
            factor(&near).map(|_| ()),
            Err(LuError::Singular { index: 1 }),
            "Lu::{variant} on [[1, 1], [1, 1 + eps]]"
        );
    }
    let fine: Mat<f64> = diag(&[1.0, 1.0e-16]);
    for (variant, factor) in lu_variants() {
        assert!(factor(&fine).is_ok(), "Lu::{variant} on diag(1, 1e-16)");
    }
}

#[test]
fn test_lu_tolerance_explicit_tolerance_unchanged() {
    // An explicit tolerance is absolute and used exactly as given, for every
    // column.
    let a = scaled(&random_matrix(4, 3, 4.0), 1.0e-30);
    assert!(matches!(
        LuFullPiv::compute_with_tol(a.as_ref(), Some(1.0e-20)),
        Err(LuFullPivError::Singular { index: 0 })
    ));
    assert!(matches!(
        LuRook::compute_with_tol(a.as_ref(), Some(1.0e-20)),
        Err(LuRookError::Singular { index: 0 })
    ));
    assert!(LuFullPiv::compute_with_tol(a.as_ref(), Some(0.0)).is_ok());
    assert!(LuRook::compute_with_tol(a.as_ref(), Some(0.0)).is_ok());
    // An absolute tolerance above the small pivot 1e-10 rejects what the
    // per-column default accepts; one below it accepts.
    let c = diag(&[1.0e6, 1.0e-10]);
    assert!(LuFullPiv::compute(c.as_ref()).is_ok());
    assert!(LuRook::compute(c.as_ref()).is_ok());
    assert_eq!(
        LuFullPiv::compute_with_tol(c.as_ref(), Some(1.0e-9)).map(|_| ()),
        Err(LuFullPivError::Singular { index: 1 })
    );
    assert_eq!(
        LuRook::compute_with_tol(c.as_ref(), Some(1.0e-9)).map(|_| ()),
        Err(LuRookError::Singular { index: 1 })
    );
    assert!(LuFullPiv::compute_with_tol(c.as_ref(), Some(1.0e-12)).is_ok());
    assert!(LuRook::compute_with_tol(c.as_ref(), Some(1.0e-12)).is_ok());
}

#[test]
fn test_lu_tolerance_non_finite_entries_propagate() {
    // NaN and infinite entries do not enter the references: they still
    // propagate through the factors instead of being reported as singular.
    let nan: Mat<f64> = Mat::from_rows(&[&[1.0, f64::NAN], &[3.0, 4.0]]);
    let inf: Mat<f64> = Mat::from_rows(&[&[1.0, f64::INFINITY], &[3.0, 4.0]]);
    // A column holding nothing but non-finite values has the reference 0.
    let all_inf: Mat<f64> = Mat::from_rows(&[&[1.0, f64::INFINITY], &[3.0, f64::NAN]]);
    for (variant, factor) in lu_variants() {
        let lu = factor(&nan).unwrap_or_else(|e| panic!("Lu::{variant} NaN: {e}"));
        assert!(lu.determinant().is_nan(), "Lu::{variant} NaN determinant");
        let lu = factor(&inf).unwrap_or_else(|e| panic!("Lu::{variant} inf: {e}"));
        assert!(
            !lu.determinant().is_finite(),
            "Lu::{variant} inf determinant"
        );
        let lu = factor(&all_inf).unwrap_or_else(|e| panic!("Lu::{variant} inf/NaN: {e}"));
        assert!(
            lu.determinant().is_nan(),
            "Lu::{variant} inf/NaN determinant"
        );
    }
    assert!(LuFullPiv::compute(inf.as_ref()).is_ok());
    assert!(LuRook::compute(inf.as_ref()).is_ok());
    let band = dense_to_band(&[1.0, f64::INFINITY, 3.0, 4.0], 2, 1, 1);
    assert!(BandLu::compute(2, 1, 1, &band).is_ok());
    // Round trip of the helpers used above.
    assert_eq!(
        band_to_dense(&dense_to_band(&[1.0, 2.0, 3.0, 4.0], 2, 1, 1), 2, 1, 1)[3],
        4.0
    );
}
