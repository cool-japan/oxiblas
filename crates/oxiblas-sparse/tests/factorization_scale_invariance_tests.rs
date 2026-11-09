//! Scale-invariance regression tests for the sparse direct factorizations
//! `SparseLU` / `SparseCholesky` and the incomplete LU factorizations
//! `ILU0` / `ILUT` / `ILUTP`.
//!
//! Every pivot, radicand and drop test of these factorizations compares a
//! quantity with a reference of the same units taken from the input, so for a
//! power of two `s` (absent overflow and underflow):
//!
//! - building on `s*A` succeeds exactly when building on `A` does, and fails
//!   with the same error and index otherwise;
//! - the permutations and the sparsity patterns of the factors do not change;
//! - the LU factors are `L(s*A) = L(A)` and `U(s*A) = s*U(A)`, and the
//!   Cholesky factor is `L(s*A) = sqrt(s)*L(A)` (`s = 4^k`), bit for bit;
//! - so a solve with the factors of `s*A` and the right-hand side `s*b`
//!   returns the bit-identical solution, and a preconditioner built from
//!   `s*A` is exactly `1/s` times the one built from `A` (GMRES then takes the
//!   same iterations and returns the bit-identical solution).
//!
//! An absolute threshold such as `|pivot| <= eps` breaks all of this: it
//! rejects every pivot of a matrix scaled by `2^-64`, and accepts numerically
//! singular pivots of one scaled by `2^64`.

use oxiblas_sparse::linalg::cholesky::{SparseCholError, SparseCholesky};
use oxiblas_sparse::linalg::iterative::pgmres;
use oxiblas_sparse::linalg::lu::{ILU0, ILUT, ILUTP, SparseLU, SparseLUError};
use oxiblas_sparse::ops::spmv;
use oxiblas_sparse::{CscMatrix, CsrMatrix};

const TOL: f64 = 1e-10;
const MAX_ITER: usize = 1000;
const RESTART: usize = 20;
/// Bound on the true relative residual of a converged GMRES run.
const TRUE_RESIDUAL_BOUND: f64 = 1e-8;
/// Bound on the relative residual of a solve with a complete factorization.
const DIRECT_RESIDUAL_BOUND: f64 = 1e-12;

const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

/// `[2^-64, 2^64]`, both exactly representable.
fn scale_factors() -> [f64; 2] {
    [1.0 / TWO_POW_64, TWO_POW_64]
}

/// `1` and the scale factors.
fn all_scales() -> [f64; 3] {
    [1.0, 1.0 / TWO_POW_64, TWO_POW_64]
}

/// `(s, sqrt(s))` for `s = 2^-64, 2^64, 2^-128, 2^128`: powers of four, so
/// the Cholesky factor scales exactly. `2^-128` is included because the old
/// absolute test `|l_jj| <= eps` rejected every pivot of `2^-128 * A`
/// (`l_jj ~ 2^-63`).
fn cholesky_scales() -> [(f64, f64); 4] {
    let two_pow_32 = 4_294_967_296.0_f64;
    [
        (1.0 / TWO_POW_64, 1.0 / two_pow_32),
        (TWO_POW_64, two_pow_32),
        (1.0 / (TWO_POW_64 * TWO_POW_64), 1.0 / TWO_POW_64),
        (TWO_POW_64 * TWO_POW_64, TWO_POW_64),
    ]
}

// =============================================================================
// Problem generators and helpers
// =============================================================================

/// 5-point operator on a `grid x grid` mesh with a row-dependent diagonal and
/// the given `[south, west, east, north]` couplings.
fn five_point(grid: usize, diag: impl Fn(usize) -> f64, couplings: [f64; 4]) -> CsrMatrix<f64> {
    let [south, west, east, north] = couplings;
    let n = grid * grid;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];
    for i in 0..n {
        let row = i / grid;
        let col = i % grid;
        if row > 0 {
            values.push(south);
            col_indices.push(i - grid);
        }
        if col > 0 {
            values.push(west);
            col_indices.push(i - 1);
        }
        values.push(diag(i));
        col_indices.push(i);
        if col + 1 < grid {
            values.push(east);
            col_indices.push(i + 1);
        }
        if row + 1 < grid {
            values.push(north);
            col_indices.push(i + grid);
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// SPD 5-point operator whose diagonal varies by row.
fn spd(grid: usize) -> CsrMatrix<f64> {
    five_point(
        grid,
        |i| 4.0 + 0.25 * (i % 7) as f64,
        [-1.0, -1.0, -1.0, -1.0],
    )
}

/// Non-symmetric, diagonally dominant 5-point operator.
fn nonsymmetric(grid: usize) -> CsrMatrix<f64> {
    five_point(
        grid,
        |i| 4.0 + 0.25 * (i % 5) as f64,
        [-1.1, -1.25, -0.75, -0.9],
    )
}

/// Non-symmetric 5-point operator whose couplings are larger than the
/// diagonal, so that row pivoting (`SparseLU`) and column pivoting (`ILUTP`)
/// both swap.
fn weak_diagonal(grid: usize) -> CsrMatrix<f64> {
    five_point(
        grid,
        |i| 0.5 + 0.25 * (i % 3) as f64,
        [-1.0, 1.25, -0.75, 1.5],
    )
}

/// Tridiagonal matrix: ILU(0) has no fill-in to drop, so it is the exact LU
/// factorization.
fn tridiagonal(n: usize) -> CsrMatrix<f64> {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];
    for i in 0..n {
        if i > 0 {
            values.push(-1.0 - 0.125 * (i % 3) as f64);
            col_indices.push(i - 1);
        }
        values.push(3.0 + 0.5 * (i % 4) as f64);
        col_indices.push(i);
        if i + 1 < n {
            values.push(-0.75);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// A dense `n x n` matrix in CSR form (row-major `entries`, zeros dropped).
fn dense_csr(n: usize, entries: &[f64]) -> CsrMatrix<f64> {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];
    for i in 0..n {
        for j in 0..n {
            let v = entries[i * n + j];
            if v != 0.0 {
                values.push(v);
                col_indices.push(j);
            }
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// Right-hand side with entries `1 + k/8` (exact in binary).
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + (i % 7) as f64 / 8.0).collect()
}

/// Deterministic probe vectors with entries of mixed sign and magnitude.
fn probes(n: usize) -> Vec<Vec<f64>> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut random = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };
    vec![rhs(n), (0..n).map(|_| random()).collect()]
}

fn scaled_csr(a: &CsrMatrix<f64>, s: f64) -> CsrMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|&v| v * s).collect();
    CsrMatrix::new(
        a.nrows(),
        a.ncols(),
        a.row_ptrs().to_vec(),
        a.col_indices().to_vec(),
        values,
    )
    .unwrap()
}

fn scaled_csc(a: &CscMatrix<f64>, s: f64) -> CscMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|&v| v * s).collect();
    CscMatrix::new(
        a.nrows(),
        a.ncols(),
        a.col_ptrs().to_vec(),
        a.row_indices().to_vec(),
        values,
    )
    .unwrap()
}

fn scaled_vec(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn scaled_bits(values: &[f64], s: f64) -> Vec<u64> {
    values.iter().map(|v| (v * s).to_bits()).collect()
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|e| e * e).sum::<f64>().sqrt()
}

fn relative_residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    spmv(1.0, a, x, 0.0, &mut ax);
    let r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    norm2(&r) / norm2(b)
}

type Dense = Vec<Vec<f64>>;

fn csr_dense(a: &CsrMatrix<f64>) -> Dense {
    let mut d = vec![vec![0.0; a.ncols()]; a.nrows()];
    for (i, row) in d.iter_mut().enumerate() {
        for idx in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
            row[a.col_indices()[idx]] = a.values()[idx];
        }
    }
    d
}

fn csc_dense(a: &CscMatrix<f64>) -> Dense {
    let mut d = vec![vec![0.0; a.ncols()]; a.nrows()];
    for (row, col, &value) in a.iter() {
        d[row][col] = value;
    }
    d
}

fn matmul(a: &Dense, b: &Dense) -> Dense {
    a.iter()
        .map(|row| {
            (0..b[0].len())
                .map(|j| row.iter().zip(b).map(|(x, b_row)| x * b_row[j]).sum())
                .collect()
        })
        .collect()
}

fn max_abs(a: &Dense) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// `max |a_ij - b_ij| / max |b_ij|`.
fn relative_difference(a: &Dense, b: &Dense) -> f64 {
    let mut worst = 0.0_f64;
    for (ra, rb) in a.iter().zip(b) {
        for (x, y) in ra.iter().zip(rb) {
            worst = worst.max((x - y).abs());
        }
    }
    worst / max_abs(b)
}

// =============================================================================
// SparseCholesky
// =============================================================================

type CholFactor = fn(&CscMatrix<f64>) -> Result<SparseCholesky<f64>, SparseCholError>;

fn cholesky_variants() -> [(&'static str, CholFactor); 2] {
    [
        ("SparseCholesky::new", SparseCholesky::new),
        (
            "SparseCholesky::with_ordering",
            SparseCholesky::with_ordering,
        ),
    ]
}

/// `L(s*A)` is exactly `sqrt(s) * L(A)` with the same permutation and
/// pattern, and solving `(s*A) x = s*b` gives the bit-identical `x`.
#[test]
fn test_sparse_cholesky_factor_exactly_scaled() {
    let a = spd(6).to_csc();
    let a_csr = spd(6);
    let b = rhs(a.nrows());
    for (name, factor) in cholesky_variants() {
        let base = factor(&a).unwrap_or_else(|e| panic!("{name}: factorization of A failed: {e}"));
        let x = base.solve(&b);
        let res = relative_residual(&a_csr, &x, &b);
        assert!(res <= DIRECT_RESIDUAL_BOUND, "{name}: residual {res:e}");
        for (s, sqrt_s) in cholesky_scales() {
            let f = factor(&scaled_csc(&a, s)).unwrap_or_else(|e| {
                panic!("{name}: factorization of A scaled by {s:e} failed ({e})")
            });
            assert_eq!(
                f.perm(),
                base.perm(),
                "{name}: permutation changed at {s:e}"
            );
            assert_eq!(
                f.l().col_ptrs(),
                base.l().col_ptrs(),
                "{name}: pattern of L changed at {s:e}"
            );
            assert_eq!(
                f.l().row_indices(),
                base.l().row_indices(),
                "{name}: pattern of L changed at {s:e}"
            );
            assert!(
                bits(f.l().values()) == scaled_bits(base.l().values(), sqrt_s),
                "{name}: L of A scaled by {s:e} is not exactly {sqrt_s:e} * L(A)"
            );
            assert!(
                bits(&f.solve(&scaled_vec(&b, s))) == bits(&x),
                "{name}: solution is not bit-identical at {s:e}"
            );
        }
    }
}

/// The complete factorization keeps every computed entry (only exact zeros
/// are dropped): `L L^T = P A P^T` to rounding.
#[test]
fn test_sparse_cholesky_is_complete() {
    let a = spd(6).to_csc();
    let dense = csc_dense(&a);
    for (name, factor) in cholesky_variants() {
        let f = factor(&a).unwrap();
        let l = csc_dense(f.l());
        let lt: Dense = (0..l.len())
            .map(|i| (0..l.len()).map(|j| l[j][i]).collect())
            .collect();
        let llt = matmul(&l, &lt);
        let perm = f.perm();
        let pap: Dense = (0..perm.len())
            .map(|i| (0..perm.len()).map(|j| dense[perm[i]][perm[j]]).collect())
            .collect();
        let diff = relative_difference(&llt, &pap);
        assert!(diff <= 1e-14, "{name}: |L L^T - P A P^T| / |A| = {diff:e}");
    }
}

/// A radicand at or below `eps * n * a_jj` is not positive definite, at every
/// scale (the rule of the dense Cholesky factorization). The second radicand
/// of `[[1, 1], [1, 1 + eps]]` is `eps`; it used to be accepted at `s = 1`
/// (`sqrt(eps) > eps`) and rejected at `s = 2^128`'s reciprocal only through
/// the absolute `|l_jj| <= eps` test.
#[test]
fn test_sparse_cholesky_relative_radicand_rule() {
    let near = dense_csr(2, &[1.0, 1.0, 1.0, 1.0 + f64::EPSILON]).to_csc();
    let regular = dense_csr(2, &[1.0, 1.0, 1.0, 1.0 + 2.0_f64.powi(-40)]).to_csc();
    let singular = dense_csr(2, &[1.0, 1.0, 1.0, 1.0]).to_csc();
    let indefinite = dense_csr(2, &[1.0, 2.0, 2.0, 1.0]).to_csc();
    let mut scales = vec![1.0];
    scales.extend(cholesky_scales().iter().map(|&(s, _)| s));
    for s in scales {
        for (name, factor) in cholesky_variants() {
            for (label, m) in [
                ("near-singular", &near),
                ("singular", &singular),
                ("indefinite", &indefinite),
            ] {
                assert!(
                    matches!(
                        factor(&scaled_csc(m, s)),
                        Err(SparseCholError::NotPositiveDefinite { index: 1 })
                    ),
                    "{name}: {label} matrix at {s:e}"
                );
            }
            assert!(
                factor(&scaled_csc(&regular, s)).is_ok(),
                "{name}: regular matrix at {s:e}"
            );
        }
    }
}

/// A diagonal entry that is not stored is reported as `ZeroDiagonal` (as by
/// `IC0` / `ICT`); a stored zero diagonal fails the radicand test.
#[test]
fn test_sparse_cholesky_missing_diagonal() {
    // [[1, 1], [1, .]]: a_11 is not stored.
    let missing = CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 0], vec![1.0, 1.0, 1.0])
        .unwrap()
        .to_csc();
    let stored_zero = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 0.0],
    )
    .unwrap()
    .to_csc();
    for s in all_scales() {
        assert_eq!(
            SparseCholesky::new(&scaled_csc(&missing, s)).unwrap_err(),
            SparseCholError::ZeroDiagonal { index: 1 },
            "missing diagonal at {s:e}"
        );
        assert_eq!(
            SparseCholesky::new(&scaled_csc(&stored_zero, s)).unwrap_err(),
            SparseCholError::NotPositiveDefinite { index: 1 },
            "stored zero diagonal at {s:e}"
        );
    }
}

// =============================================================================
// SparseLU
// =============================================================================

/// `P A = L U` to rounding; for `s*A`: same permutation, `L` bit-identical,
/// `U` exactly `s` times, the same patterns, and a bit-identical solution of
/// `(s*A) x = s*b`.
fn check_sparse_lu_exactly_scaled(a_csr: &CsrMatrix<f64>) {
    let a = a_csr.to_csc();
    let b = rhs(a.nrows());
    let base = SparseLU::new(&a).unwrap_or_else(|e| panic!("factorization of A failed: {e}"));
    let dense = csc_dense(&a);
    let pa: Dense = base.perm().iter().map(|&p| dense[p].clone()).collect();
    let diff = relative_difference(&matmul(&csc_dense(base.l()), &csc_dense(base.u())), &pa);
    assert!(diff <= 1e-14, "|L U - P A| / |A| = {diff:e}");
    let x = base.solve(&b);
    let res = relative_residual(a_csr, &x, &b);
    assert!(res <= DIRECT_RESIDUAL_BOUND, "residual {res:e}");
    for s in scale_factors() {
        let f = SparseLU::new(&scaled_csc(&a, s))
            .unwrap_or_else(|e| panic!("factorization of A scaled by {s:e} failed ({e})"));
        assert_eq!(f.perm(), base.perm(), "permutation changed at {s:e}");
        assert_eq!(f.l().col_ptrs(), base.l().col_ptrs(), "L pattern at {s:e}");
        assert_eq!(
            f.l().row_indices(),
            base.l().row_indices(),
            "L pattern at {s:e}"
        );
        assert!(
            bits(f.l().values()) == bits(base.l().values()),
            "L is not bit-identical at {s:e}"
        );
        assert_eq!(f.u().col_ptrs(), base.u().col_ptrs(), "U pattern at {s:e}");
        assert_eq!(
            f.u().row_indices(),
            base.u().row_indices(),
            "U pattern at {s:e}"
        );
        assert!(
            bits(f.u().values()) == scaled_bits(base.u().values(), s),
            "U of A scaled by {s:e} is not exactly {s:e} * U(A)"
        );
        assert!(
            bits(&f.solve(&scaled_vec(&b, s))) == bits(&x),
            "solution is not bit-identical at {s:e}"
        );
    }
}

#[test]
fn test_sparse_lu_nonsymmetric_exactly_scaled() {
    check_sparse_lu_exactly_scaled(&nonsymmetric(6));
}

#[test]
fn test_sparse_lu_with_row_swaps_exactly_scaled() {
    let a = weak_diagonal(6);
    let lu = SparseLU::new(&a.to_csc()).unwrap();
    assert!(
        lu.perm().iter().enumerate().any(|(i, &p)| i != p),
        "the test matrix must make partial pivoting swap rows"
    );
    check_sparse_lu_exactly_scaled(&a);
}

/// Exactly and numerically singular matrices are reported at every scale
/// with the same index; `[[1, 1], [1, 1 + 2^-20]]` factors at every scale.
#[test]
fn test_sparse_lu_singular_at_every_scale() {
    let cases: [(&str, CsrMatrix<f64>, Option<usize>); 4] = [
        ("rank one", dense_csr(2, &[1.0, 2.0, 2.0, 4.0]), Some(1)),
        (
            "near singular",
            dense_csr(2, &[1.0, 1.0, 1.0, 1.0 + f64::EPSILON]),
            Some(1),
        ),
        ("zero column", dense_csr(2, &[1.0, 0.0, 1.0, 0.0]), Some(1)),
        (
            "regular",
            dense_csr(2, &[1.0, 1.0, 1.0, 1.0 + 2.0_f64.powi(-20)]),
            None,
        ),
    ];
    for (label, a, singular_at) in &cases {
        for s in all_scales() {
            let result = SparseLU::new(&scaled_csr(a, s).to_csc());
            match singular_at {
                Some(row) => assert_eq!(
                    result.unwrap_err(),
                    SparseLUError::Singular { row: *row },
                    "{label} at {s:e}"
                ),
                None => assert!(result.is_ok(), "{label} at {s:e}"),
            }
        }
    }
}

// =============================================================================
// Incomplete LU factorizations
// =============================================================================

/// Rows of a CSR matrix: `(row_ptrs, col_indices, values)`.
type Rows = (Vec<usize>, Vec<usize>, Vec<f64>);

/// The factors of an incomplete LU factorization in a uniform form.
struct IluFactors {
    /// Strictly lower part of `L` (its unit diagonal is implicit).
    l: Rows,
    /// `U`, diagonal included.
    u: Rows,
    /// Column permutation (`A Q = L U` with `Q e_i = e_perm[i]`).
    perm: Vec<usize>,
    /// The preconditioner `M^{-1}`.
    apply: Apply,
}

/// Applies `M^{-1}` to a vector.
type Apply = Box<dyn Fn(&[f64]) -> Vec<f64>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ilu {
    Ilu0,
    /// ILUT with active dropping and fill limit.
    IlutDropping,
    /// ILUT without dropping or fill limit (the complete LU factorization).
    IlutComplete,
    /// ILUTP with dropping, fill limit and threshold column pivoting.
    IlutpDropping,
    /// ILUTP without dropping, with full column pivoting (the complete LU
    /// factorization of `A Q`).
    IlutpComplete,
}

fn rows(m: &CsrMatrix<f64>) -> Rows {
    (
        m.row_ptrs().to_vec(),
        m.col_indices().to_vec(),
        m.values().to_vec(),
    )
}

/// Splits the combined `L + U - I` of ILU(0) into its strictly lower part and
/// its upper part.
fn split_lu(lu: &CsrMatrix<f64>) -> (Rows, Rows) {
    let mut l: Rows = (vec![0], vec![], vec![]);
    let mut u: Rows = (vec![0], vec![], vec![]);
    for i in 0..lu.nrows() {
        for idx in lu.row_ptrs()[i]..lu.row_ptrs()[i + 1] {
            let (j, v) = (lu.col_indices()[idx], lu.values()[idx]);
            let part = if j < i { &mut l } else { &mut u };
            part.1.push(j);
            part.2.push(v);
        }
        l.0.push(l.1.len());
        u.0.push(u.1.len());
    }
    (l, u)
}

fn build_ilu(kind: Ilu, a: &CsrMatrix<f64>) -> Result<IluFactors, SparseLUError> {
    let n = a.nrows();
    let identity: Vec<usize> = (0..n).collect();
    Ok(match kind {
        Ilu::Ilu0 => {
            let f = ILU0::new(a)?;
            let (l, u) = split_lu(f.lu());
            IluFactors {
                l,
                u,
                perm: identity,
                apply: Box::new(move |r: &[f64]| f.apply(r)),
            }
        }
        Ilu::IlutDropping | Ilu::IlutComplete => {
            let f = if kind == Ilu::IlutDropping {
                ILUT::new(a, 0.05, 3)?
            } else {
                ILUT::new(a, 0.0, n)?
            };
            let (l, u) = (rows(f.l()), rows(f.u()));
            IluFactors {
                l,
                u,
                perm: identity,
                apply: Box::new(move |r: &[f64]| f.apply(r)),
            }
        }
        Ilu::IlutpDropping | Ilu::IlutpComplete => {
            let f = if kind == Ilu::IlutpDropping {
                ILUTP::new(a, 0.05, 3, 0.5)?
            } else {
                ILUTP::new(a, 0.0, n, 1.0)?
            };
            let (l, u, perm) = (rows(f.l()), rows(f.u()), f.perm().to_vec());
            IluFactors {
                l,
                u,
                perm,
                apply: Box::new(move |r: &[f64]| f.apply(r)),
            }
        }
    })
}

/// Building on `s*A` succeeds because building on `A` does; the permutation
/// and the patterns are unchanged, `L` is bit-identical, `U` is exactly `s`
/// times, and `M(s*A)^{-1} r = M(A)^{-1} r / s` bit for bit.
fn check_ilu_exactly_scaled(kind: Ilu, a: &CsrMatrix<f64>) {
    let base = build_ilu(kind, a).unwrap_or_else(|e| panic!("{kind:?}: building on A failed: {e}"));
    let vectors = probes(a.nrows());
    let reference: Vec<Vec<f64>> = vectors.iter().map(|r| (base.apply)(r)).collect();
    for z in &reference {
        assert!(
            z.iter().all(|v| v.is_finite()),
            "{kind:?}: non-finite output"
        );
    }
    for s in scale_factors() {
        let f = build_ilu(kind, &scaled_csr(a, s)).unwrap_or_else(|e| {
            panic!("{kind:?}: building on A scaled by {s:e} failed ({e}) although building on A succeeded")
        });
        assert_eq!(f.perm, base.perm, "{kind:?}: permutation changed at {s:e}");
        assert_eq!(f.l.0, base.l.0, "{kind:?}: pattern of L changed at {s:e}");
        assert_eq!(f.l.1, base.l.1, "{kind:?}: pattern of L changed at {s:e}");
        assert!(
            bits(&f.l.2) == bits(&base.l.2),
            "{kind:?}: L is not bit-identical at {s:e}"
        );
        assert_eq!(f.u.0, base.u.0, "{kind:?}: pattern of U changed at {s:e}");
        assert_eq!(f.u.1, base.u.1, "{kind:?}: pattern of U changed at {s:e}");
        assert!(
            bits(&f.u.2) == scaled_bits(&base.u.2, s),
            "{kind:?}: U of A scaled by {s:e} is not exactly {s:e} * U(A)"
        );
        for (probe, (r, z)) in vectors.iter().zip(&reference).enumerate() {
            assert!(
                bits(&(f.apply)(r)) == scaled_bits(z, 1.0 / s),
                "{kind:?}: preconditioner of A scaled by {s:e} is not exactly 1/s times that of A (probe {probe})"
            );
        }
    }
}

/// Right-preconditioned GMRES with `kind` rebuilt from the scaled matrix
/// converges at `s = 1` and has exactly the same outcome at `s = 2^-64, 2^64`:
/// iteration count, bit-identical solution, exactly scaled residual history.
fn check_pgmres_scale_invariant(kind: Ilu, a: &CsrMatrix<f64>) {
    let b = rhs(a.nrows());
    let x0 = vec![0.0; a.nrows()];
    let m = build_ilu(kind, a).unwrap_or_else(|e| panic!("{kind:?}: building on A failed: {e}"));
    let reference = pgmres(a, &b, &x0, |r: &[f64]| (m.apply)(r), RESTART, TOL, MAX_ITER)
        .unwrap_or_else(|e| panic!("{kind:?}: unscaled solve failed: {e}"));
    assert!(
        reference.converged,
        "{kind:?}: unscaled solve did not converge in {} iterations",
        reference.iterations
    );
    let res = relative_residual(a, &reference.x, &b);
    assert!(
        res <= TRUE_RESIDUAL_BOUND,
        "{kind:?}: reported convergence with true relative residual {res:e}"
    );
    for s in scale_factors() {
        let a_s = scaled_csr(a, s);
        let m_s = build_ilu(kind, &a_s)
            .unwrap_or_else(|e| panic!("{kind:?}: building on A scaled by {s:e} failed ({e})"));
        let sol = pgmres(
            &a_s,
            &scaled_vec(&b, s),
            &x0,
            |r: &[f64]| (m_s.apply)(r),
            RESTART,
            TOL,
            MAX_ITER,
        )
        .unwrap_or_else(|e| panic!("{kind:?}: solve at {s:e} failed: {e}"));
        assert_eq!(
            sol.iterations, reference.iterations,
            "{kind:?}: iterations at {s:e}"
        );
        assert_eq!(sol.converged, reference.converged);
        assert!(
            bits(&sol.x) == bits(&reference.x),
            "{kind:?}: solution is not bit-identical at {s:e}"
        );
        assert_eq!(
            sol.residual_norm.to_bits(),
            (reference.residual_norm * s).to_bits(),
            "{kind:?}: residual norm is not exactly scaled at {s:e}"
        );
        assert!(
            bits(&sol.residual_history) == scaled_bits(&reference.residual_history, s),
            "{kind:?}: residual history is not exactly scaled at {s:e}"
        );
    }
}

/// The matrix each incomplete factorization is checked on.
fn ilu_test_matrix(kind: Ilu) -> CsrMatrix<f64> {
    match kind {
        Ilu::Ilu0 | Ilu::IlutDropping | Ilu::IlutComplete => nonsymmetric(8),
        Ilu::IlutpDropping | Ilu::IlutpComplete => weak_diagonal(6),
    }
}

macro_rules! ilu_scale_tests {
    ($($exact:ident, $solve:ident => $kind:expr),* $(,)?) => {
        $(
            #[test]
            fn $exact() {
                check_ilu_exactly_scaled($kind, &ilu_test_matrix($kind));
            }

            #[test]
            fn $solve() {
                check_pgmres_scale_invariant($kind, &ilu_test_matrix($kind));
            }
        )*
    };
}

ilu_scale_tests! {
    test_ilu0_exactly_scaled, test_pgmres_ilu0_scale_invariant => Ilu::Ilu0,
    test_ilut_dropping_exactly_scaled, test_pgmres_ilut_dropping_scale_invariant => Ilu::IlutDropping,
    test_ilut_complete_exactly_scaled, test_pgmres_ilut_complete_scale_invariant => Ilu::IlutComplete,
    test_ilutp_dropping_exactly_scaled, test_pgmres_ilutp_dropping_scale_invariant => Ilu::IlutpDropping,
    test_ilutp_complete_exactly_scaled, test_pgmres_ilutp_complete_scale_invariant => Ilu::IlutpComplete,
}

/// Dense `L U` of the uniform factors (unit diagonal added to `L`), in the
/// permuted column numbering.
fn lu_product(f: &IluFactors, n: usize) -> Dense {
    let mut l = vec![vec![0.0; n]; n];
    let mut u = vec![vec![0.0; n]; n];
    for i in 0..n {
        l[i][i] = 1.0;
        for idx in f.l.0[i]..f.l.0[i + 1] {
            l[i][f.l.1[idx]] = f.l.2[idx];
        }
        for idx in f.u.0[i]..f.u.0[i + 1] {
            u[i][f.u.1[idx]] = f.u.2[idx];
        }
    }
    matmul(&l, &u)
}

/// `A Q` (`(A Q)[:, i] = A[:, perm[i]]`).
fn permuted_columns(a: &CsrMatrix<f64>, perm: &[usize]) -> Dense {
    let dense = csr_dense(a);
    dense
        .iter()
        .map(|row| perm.iter().map(|&p| row[p]).collect())
        .collect()
}

/// Without dropping, ILUT is the complete LU factorization (`L U = A`) and
/// ILUTP the complete LU factorization of `A Q`, at every scale. ILUTP used to
/// leave the earlier rows of `U` in the old column numbering after a column
/// swap, so its factors did not multiply back to `A Q`.
#[test]
fn test_complete_ilut_and_ilutp_are_exact_lu() {
    for kind in [Ilu::IlutComplete, Ilu::IlutpComplete] {
        let a = ilu_test_matrix(kind);
        let n = a.nrows();
        for s in all_scales() {
            let a_s = scaled_csr(&a, s);
            let f = build_ilu(kind, &a_s)
                .unwrap_or_else(|e| panic!("{kind:?}: building at {s:e} failed: {e}"));
            let diff = relative_difference(&lu_product(&f, n), &permuted_columns(&a_s, &f.perm));
            assert!(
                diff <= 1e-13,
                "{kind:?} at {s:e}: |L U - A Q| / |A| = {diff:e}"
            );
        }
        if kind == Ilu::IlutpComplete {
            let f = build_ilu(kind, &a).unwrap();
            assert!(
                f.perm.iter().enumerate().any(|(i, &p)| i != p),
                "the test matrix must make ILUTP swap columns"
            );
        }
    }
}

/// ILU(0) of a tridiagonal matrix drops nothing, so applying it solves
/// `A x = b` exactly (to rounding). `apply` used to divide the multipliers of
/// `L` by the pivots a second time.
#[test]
fn test_ilu0_apply_is_the_lu_solve() {
    let a = tridiagonal(12);
    let ilu = ILU0::new(&a).unwrap();
    for x in probes(a.nrows()) {
        let mut b = vec![0.0; x.len()];
        spmv(1.0, &a, &x, 0.0, &mut b);
        let y = ilu.apply(&b);
        let err = y
            .iter()
            .zip(&x)
            .fold(0.0_f64, |m, (yi, xi)| m.max((yi - xi).abs()));
        assert!(err <= 1e-14, "ILU(0) solve error {err:e}");
    }
}

/// `pivot_tol = 0` means no pivoting: the permutation stays the identity.
/// It used to swap in the last nonzero entry of every row.
#[test]
fn test_ilutp_zero_pivot_tolerance_never_permutes() {
    for a in [nonsymmetric(5), weak_diagonal(5)] {
        let n = a.nrows();
        let f = ILUTP::new(&a, 0.0, n, 0.0).unwrap();
        let identity: Vec<usize> = (0..n).collect();
        assert_eq!(f.perm(), identity.as_slice());
        let g = ILUT::new(&a, 0.0, n).unwrap();
        assert_eq!(
            rows(f.l()),
            rows(g.l()),
            "ILUTP(pivot_tol = 0) differs from ILUT"
        );
        assert_eq!(
            rows(f.u()),
            rows(g.u()),
            "ILUTP(pivot_tol = 0) differs from ILUT"
        );
    }
}

/// With a large drop tolerance every off-diagonal entry is dropped at every
/// scale: `L` is empty and `U` is the diagonal of `A`. The multipliers of `L`
/// used to be compared with `tau * ||row_i(A)||` directly, so at `2^-64` they
/// were all kept.
#[test]
fn test_ilut_large_tau_drops_every_off_diagonal() {
    let a = nonsymmetric(6);
    let n = a.nrows();
    let diag: Vec<f64> = (0..n)
        .map(|i| {
            (a.row_ptrs()[i]..a.row_ptrs()[i + 1])
                .find(|&idx| a.col_indices()[idx] == i)
                .map(|idx| a.values()[idx])
                .unwrap()
        })
        .collect();
    for s in all_scales() {
        let a_s = scaled_csr(&a, s);
        let f = ILUT::new(&a_s, 1.0, n).unwrap();
        assert_eq!(f.l().nnz(), 0, "ILUT L at {s:e}");
        assert_eq!(f.u().nnz(), n, "ILUT U at {s:e}");
        assert!(
            bits(f.u().values()) == scaled_bits(&diag, s),
            "ILUT diagonal at {s:e}"
        );
        let g = ILUTP::new(&a_s, 1.0, n, 0.0).unwrap();
        assert_eq!(g.l().nnz(), 0, "ILUTP L at {s:e}");
        assert_eq!(g.u().nnz(), n, "ILUTP U at {s:e}");
    }
}

/// Exactly and numerically singular inputs are reported by every incomplete
/// factorization at every scale, with the same index.
#[test]
fn test_incomplete_lu_singular_at_every_scale() {
    let cases: [(&str, CsrMatrix<f64>, Option<usize>); 5] = [
        ("rank one", dense_csr(2, &[1.0, 2.0, 2.0, 4.0]), Some(1)),
        ("all ones", dense_csr(2, &[1.0, 1.0, 1.0, 1.0]), Some(1)),
        (
            "near singular",
            dense_csr(2, &[1.0, 1.0, 1.0, 1.0 + f64::EPSILON]),
            Some(1),
        ),
        ("zero column", dense_csr(2, &[1.0, 0.0, 1.0, 0.0]), Some(1)),
        (
            "regular",
            dense_csr(2, &[1.0, 1.0, 1.0, 1.0 + 2.0_f64.powi(-20)]),
            None,
        ),
    ];
    for (label, a, singular_at) in &cases {
        for s in all_scales() {
            let a_s = scaled_csr(a, s);
            let results: [(&str, Result<(), SparseLUError>); 4] = [
                ("ILU0", ILU0::new(&a_s).map(|_| ())),
                ("ILUT", ILUT::new(&a_s, 0.0, 2).map(|_| ())),
                ("ILUTP(0)", ILUTP::new(&a_s, 0.0, 2, 0.0).map(|_| ())),
                ("ILUTP(1)", ILUTP::new(&a_s, 0.0, 2, 1.0).map(|_| ())),
            ];
            for (name, result) in results {
                match singular_at {
                    Some(row) => assert_eq!(
                        result,
                        Err(SparseLUError::Singular { row: *row }),
                        "{name}: {label} at {s:e}"
                    ),
                    None => assert_eq!(result, Ok(()), "{name}: {label} at {s:e}"),
                }
            }
        }
    }
}

/// A missing diagonal entry is a zero pivot for ILU(0), also in the last row
/// (which no later row uses as a pivot).
#[test]
fn test_ilu0_missing_diagonal_reported() {
    // [[1, 1], [1, .]]: a_11 is not stored.
    let a = CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 0], vec![1.0, 1.0, 1.0]).unwrap();
    assert_eq!(
        ILU0::new(&a).err(),
        Some(SparseLUError::Singular { row: 1 })
    );
}
