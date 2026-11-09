//! Regression tests for the pivot and radicand thresholds of the sparse
//! factorizations, and for the row norms of the threshold-dropping ones.
//!
//! A pivot (LU) or radicand (LLᵀ) is negligible when it is at or below
//! `eps · m · s`, where `s` is the reference of the original matrix (largest
//! finite magnitude of the column, or the diagonal entry) and `m` is the
//! number of terms summed to form it: the original entry plus every product
//! actually subtracted from it during the elimination. The thresholds used
//! the global order `n` instead, which rejects well-conditioned pivots of
//! large matrices with short rows: an identity of order `10^5` with one
//! `2 x 2` block `[[1, 1], [1, 1 + 1e-12]]` was "not positive definite".
//!
//! `ICT`, `ILUT` and `ILUTP` drop relative to the 2-norm of a row of `A`,
//! which was an unscaled sum of squares: it overflowed near `2^520 · A`
//! (dropping every entry) and underflowed near `2^-600 · A` (dropping none).

use oxiblas_sparse::CsrMatrix;
use oxiblas_sparse::linalg::cholesky::{IC0, ICT, SparseCholError, SparseCholesky};
use oxiblas_sparse::linalg::lu::{ILU0, ILUT, ILUTP, SparseLU, SparseLUError};

/// Identity of order `n` whose rows and columns `k..k + m` hold the dense
/// `m x m` block `block` (row-major).
fn identity_with_block(n: usize, k: usize, block: &[f64]) -> CsrMatrix<f64> {
    let m = (block.len() as f64).sqrt() as usize;
    assert_eq!(m * m, block.len());
    let mut row_ptrs = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        if (k..k + m).contains(&i) {
            let bi = i - k;
            for bj in 0..m {
                let v = block[bi * m + bj];
                if v != 0.0 || bi == bj {
                    col_indices.push(k + bj);
                    values.push(v);
                }
            }
        } else {
            col_indices.push(i);
            values.push(1.0);
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// `[[1, 0, .., 1], [0, 1, .., 1], .., [1, 1, .., d]]` of order `k + 1`: the
/// last pivot (or radicand) is `d − k`, formed from `k + 1` terms.
fn arrow(k: usize, d: f64) -> Vec<f64> {
    let m = k + 1;
    let mut block = vec![0.0; m * m];
    for i in 0..k {
        block[i * m + i] = 1.0;
        block[i * m + k] = 1.0;
        block[k * m + i] = 1.0;
    }
    block[k * m + k] = d;
    block
}

fn scaled(a: &CsrMatrix<f64>, s: f64) -> CsrMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|v| v * s).collect();
    CsrMatrix::new(
        a.nrows(),
        a.ncols(),
        a.row_ptrs().to_vec(),
        a.col_indices().to_vec(),
        values,
    )
    .unwrap()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn scaled_bits(values: &[f64], s: f64) -> Vec<u64> {
    values.iter().map(|v| (v * s).to_bits()).collect()
}

const LARGE_N: usize = 100_000;
const BLOCK_AT: usize = LARGE_N / 2;
/// `[[1, 1], [1, 1 + 1e-12]]`: its second radicand / pivot is about
/// `1e-12`, far above `eps · 2`, but below `eps · 10^5 ≈ 2.2e-11`.
const NEAR_BLOCK: [f64; 4] = [1.0, 1.0, 1.0, 1.0 + 1e-12];
const SINGULAR_BLOCK: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
/// `2^64` and its reciprocal.
const SCALES: [f64; 2] = [5.421_010_862_427_522e-20, 18_446_744_073_709_551_616.0];

// =============================================================================
// Incomplete and complete Cholesky
// =============================================================================

#[test]
fn test_incomplete_cholesky_large_identity_with_block_factors() {
    let a = identity_with_block(LARGE_N, BLOCK_AT, &NEAR_BLOCK);
    let ic0 = IC0::new(&a).expect("IC0 must accept the well-conditioned 2 x 2 block");
    let ict = ICT::new(&a, 0.0, 4).expect("ICT must accept the well-conditioned 2 x 2 block");
    for s in SCALES {
        let a_s = scaled(&a, s);
        let sqrt_s = s.sqrt();
        let ic0_s = IC0::new(&a_s).unwrap_or_else(|e| panic!("IC0 at {s:e}: {e}"));
        let ict_s = ICT::new(&a_s, 0.0, 4).unwrap_or_else(|e| panic!("ICT at {s:e}: {e}"));
        assert_eq!(ic0_s.l().col_indices(), ic0.l().col_indices());
        assert!(bits(ic0_s.l().values()) == scaled_bits(ic0.l().values(), sqrt_s));
        assert_eq!(ict_s.l().col_indices(), ict.l().col_indices());
        assert!(bits(ict_s.l().values()) == scaled_bits(ict.l().values(), sqrt_s));
    }
}

#[test]
fn test_incomplete_cholesky_large_identity_singular_block_still_fails() {
    let singular = identity_with_block(LARGE_N, BLOCK_AT, &SINGULAR_BLOCK);
    let near = identity_with_block(LARGE_N, BLOCK_AT, &[1.0, 1.0, 1.0, 1.0 + f64::EPSILON]);
    let expected = SparseCholError::NotPositiveDefinite {
        index: BLOCK_AT + 1,
    };
    for s in [1.0, SCALES[0], SCALES[1]] {
        for (label, a) in [("singular", &singular), ("eps-singular", &near)] {
            let a_s = scaled(a, s);
            assert_eq!(
                IC0::new(&a_s).unwrap_err(),
                expected,
                "IC0 {label} at {s:e}"
            );
            assert_eq!(
                ICT::new(&a_s, 0.0, 4).unwrap_err(),
                expected,
                "ICT {label} at {s:e}"
            );
        }
    }
}

#[test]
fn test_sparse_cholesky_identity_with_block_uses_local_term_count() {
    // SparseCholesky is left-looking over every previous column, so a
    // moderate order keeps the test fast: with n = 8192 the global rule
    // rejected the radicand 2^-40 (eps · n = 2^-39).
    let n = 8192;
    let k = n / 2;
    let near = identity_with_block(n, k, &[1.0, 1.0, 1.0, 1.0 + 2.0_f64.powi(-40)]);
    let singular = identity_with_block(n, k, &SINGULAR_BLOCK);
    let chol = SparseCholesky::new(&near.to_csc()).expect("radicand 2^-40 >> eps · 2");
    for s in SCALES {
        let chol_s = SparseCholesky::new(&scaled(&near, s).to_csc())
            .unwrap_or_else(|e| panic!("SparseCholesky at {s:e}: {e}"));
        assert!(bits(chol_s.l().values()) == scaled_bits(chol.l().values(), s.sqrt()));
        assert_eq!(
            SparseCholesky::new(&scaled(&singular, s).to_csc()).unwrap_err(),
            SparseCholError::NotPositiveDefinite { index: k + 1 },
            "singular block at {s:e}"
        );
    }
}

#[test]
fn test_cholesky_radicand_threshold_counts_terms() {
    // Inside an identity of order 64, the arrow block with k = 3 has the last
    // radicand d − 3, formed from 4 terms: the threshold is eps · 4 · d, not
    // eps · 64 · d. d = 3 + 12 eps gives the radicand 12 eps exactly, at the
    // threshold (rejected); d = 3 + 14 eps gives 14 eps (accepted).
    let eps = f64::EPSILON;
    let (n, k) = (64, 20);
    let at_threshold = identity_with_block(n, k, &arrow(3, 3.0 + 12.0 * eps));
    let above = identity_with_block(n, k, &arrow(3, 3.0 + 14.0 * eps));
    for s in [1.0, SCALES[0], SCALES[1]] {
        let rejected = SparseCholError::NotPositiveDefinite { index: k + 3 };
        let a_s = scaled(&at_threshold, s);
        assert_eq!(IC0::new(&a_s).unwrap_err(), rejected, "IC0 at {s:e}");
        assert_eq!(
            ICT::new(&a_s, 0.0, n).unwrap_err(),
            rejected,
            "ICT at {s:e}"
        );
        assert_eq!(
            SparseCholesky::new(&a_s.to_csc()).unwrap_err(),
            rejected,
            "SparseCholesky at {s:e}"
        );
        let b_s = scaled(&above, s);
        assert!(IC0::new(&b_s).is_ok(), "IC0 at {s:e}");
        assert!(ICT::new(&b_s, 0.0, n).is_ok(), "ICT at {s:e}");
        assert!(
            SparseCholesky::new(&b_s.to_csc()).is_ok(),
            "SparseCholesky at {s:e}"
        );
    }
}

// =============================================================================
// Incomplete and complete LU
// =============================================================================

fn incomplete_lu_results(a: &CsrMatrix<f64>) -> [(&'static str, Result<(), SparseLUError>); 4] {
    [
        ("ILU0", ILU0::new(a).map(|_| ())),
        ("ILUT", ILUT::new(a, 0.0, 4).map(|_| ())),
        ("ILUTP(0)", ILUTP::new(a, 0.0, 4, 0.0).map(|_| ())),
        ("ILUTP(1)", ILUTP::new(a, 0.0, 4, 1.0).map(|_| ())),
    ]
}

#[test]
fn test_incomplete_lu_large_identity_with_block_factors() {
    let a = identity_with_block(LARGE_N, BLOCK_AT, &NEAR_BLOCK);
    for (name, result) in incomplete_lu_results(&a) {
        assert_eq!(result, Ok(()), "{name} must accept the 2 x 2 block");
    }
    let ilu0 = ILU0::new(&a).unwrap();
    let ilut = ILUT::new(&a, 0.0, 4).unwrap();
    for s in SCALES {
        let a_s = scaled(&a, s);
        for (name, result) in incomplete_lu_results(&a_s) {
            assert_eq!(result, Ok(()), "{name} at {s:e}");
        }
        let ilu0_s = ILU0::new(&a_s).unwrap();
        // L(s·A) = L(A) and U(s·A) = s·U(A): the combined L + U − I stores the
        // multiplier 1 below the diagonal and s-scaled values elsewhere.
        for (i, (vs, v)) in ilu0_s
            .lu()
            .values()
            .iter()
            .zip(ilu0.lu().values())
            .enumerate()
        {
            let row = ilu0.lu().row_ptrs().partition_point(|&p| p <= i) - 1;
            let col = ilu0.lu().col_indices()[i];
            let expected = if col < row { *v } else { v * s };
            assert_eq!(vs.to_bits(), expected.to_bits(), "ILU0 entry {i} at {s:e}");
        }
        let ilut_s = ILUT::new(&a_s, 0.0, 4).unwrap();
        assert!(bits(ilut_s.l().values()) == bits(ilut.l().values()));
        assert!(bits(ilut_s.u().values()) == scaled_bits(ilut.u().values(), s));
    }
}

#[test]
fn test_incomplete_lu_large_identity_singular_block_still_fails() {
    let singular = identity_with_block(LARGE_N, BLOCK_AT, &SINGULAR_BLOCK);
    let near = identity_with_block(LARGE_N, BLOCK_AT, &[1.0, 1.0, 1.0, 1.0 + f64::EPSILON]);
    for s in [1.0, SCALES[0], SCALES[1]] {
        for (label, a) in [("singular", &singular), ("eps-singular", &near)] {
            for (name, result) in incomplete_lu_results(&scaled(a, s)) {
                assert_eq!(
                    result,
                    Err(SparseLUError::Singular { row: BLOCK_AT + 1 }),
                    "{name} {label} at {s:e}"
                );
            }
        }
    }
}

#[test]
fn test_sparse_lu_identity_with_block_uses_local_term_count() {
    // SparseLU works on dense n x n arrays, so the order stays small: with
    // n = 256 the global rule rejected the pivot 2^-44 (eps · n = 2^-44).
    let n = 256;
    let k = 100;
    let near = identity_with_block(n, k, &[1.0, 1.0, 1.0, 1.0 + 2.0_f64.powi(-44)]);
    let singular = identity_with_block(n, k, &SINGULAR_BLOCK);
    let lu = SparseLU::new(&near.to_csc()).expect("pivot 2^-44 >> eps · 2");
    for s in SCALES {
        let lu_s = SparseLU::new(&scaled(&near, s).to_csc())
            .unwrap_or_else(|e| panic!("SparseLU at {s:e}: {e}"));
        assert_eq!(lu_s.perm(), lu.perm());
        assert!(bits(lu_s.l().values()) == bits(lu.l().values()));
        assert!(bits(lu_s.u().values()) == scaled_bits(lu.u().values(), s));
        assert_eq!(
            SparseLU::new(&scaled(&singular, s).to_csc()).unwrap_err(),
            SparseLUError::Singular { row: k + 1 },
            "singular block at {s:e}"
        );
    }
}

#[test]
fn test_lu_pivot_threshold_counts_terms() {
    // The arrow block (k = 3) inside an identity of order 64: the last pivot
    // d − 3 is formed from 4 terms and its column reference is d, so the
    // threshold is eps · 4 · d (see the Cholesky counterpart above).
    let eps = f64::EPSILON;
    let (n, k) = (64, 20);
    let at_threshold = identity_with_block(n, k, &arrow(3, 3.0 + 12.0 * eps));
    let above = identity_with_block(n, k, &arrow(3, 3.0 + 14.0 * eps));
    for s in [1.0, SCALES[0], SCALES[1]] {
        let rejected = Err(SparseLUError::Singular { row: k + 3 });
        let a_s = scaled(&at_threshold, s);
        for (name, result) in incomplete_lu_results(&a_s) {
            assert_eq!(result, rejected, "{name} at {s:e}");
        }
        assert_eq!(
            SparseLU::new(&a_s.to_csc()).map(|_| ()),
            rejected,
            "SparseLU at {s:e}"
        );
        let b_s = scaled(&above, s);
        for (name, result) in incomplete_lu_results(&b_s) {
            assert_eq!(result, Ok(()), "{name} at {s:e}");
        }
        assert!(SparseLU::new(&b_s.to_csc()).is_ok(), "SparseLU at {s:e}");
    }
}

// =============================================================================
// Row norms of the threshold-dropping factorizations at extreme scales
// =============================================================================

/// SPD 5-point operator on a `grid x grid` mesh with a row-dependent diagonal.
fn variable_spd(grid: usize) -> CsrMatrix<f64> {
    let n = grid * grid;
    let mut row_ptrs = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        let (row, col) = (i / grid, i % grid);
        let mut push = |j: usize, v: f64| {
            col_indices.push(j);
            values.push(v);
        };
        if row > 0 {
            push(i - grid, -1.0);
        }
        if col > 0 {
            push(i - 1, -1.0);
        }
        push(i, 4.0 + 0.25 * (i % 7) as f64);
        if col + 1 < grid {
            push(i + 1, -1.0);
        }
        if row + 1 < grid {
            push(i + grid, -1.0);
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// `(s, sqrt(s))` for `s = 2^520` and `s = 2^-600`: the sum of squares of a
/// row overflows at the first and underflows to zero at the second.
fn extreme_scales() -> [(f64, f64); 2] {
    [
        (2.0_f64.powi(520), 2.0_f64.powi(260)),
        (2.0_f64.powi(-600), 2.0_f64.powi(-300)),
    ]
}

#[test]
fn test_ict_drop_rule_at_extreme_scales() {
    let a = variable_spd(10);
    let n = a.nrows();
    let tau = 0.05;
    let ict = ICT::new(&a, tau, n).unwrap();
    // The drop rule matters at scale 1: it keeps some fill-in, but not all.
    let complete = ICT::new(&a, 0.0, n).unwrap();
    assert!(ict.nnz() < complete.nnz(), "tau = {tau} drops nothing");
    assert!(
        ict.nnz() > IC0::new(&a).unwrap().nnz(),
        "tau = {tau} drops all fill-in"
    );
    for (s, sqrt_s) in extreme_scales() {
        let ict_s =
            ICT::new(&scaled(&a, s), tau, n).unwrap_or_else(|e| panic!("ICT at {s:e}: {e}"));
        assert_eq!(
            ict_s.l().row_ptrs(),
            ict.l().row_ptrs(),
            "ICT pattern at {s:e}"
        );
        assert_eq!(
            ict_s.l().col_indices(),
            ict.l().col_indices(),
            "ICT pattern at {s:e}"
        );
        assert!(
            bits(ict_s.l().values()) == scaled_bits(ict.l().values(), sqrt_s),
            "ICT factor at {s:e} is not exactly {sqrt_s:e} * L(A)"
        );
    }
}

#[test]
fn test_ilut_and_ilutp_drop_rule_at_extreme_scales() {
    let a = variable_spd(10);
    let n = a.nrows();
    let tau = 0.05;
    let complete = ILUT::new(&a, 0.0, n).unwrap();
    let ilut = ILUT::new(&a, tau, n).unwrap();
    assert!(ilut.nnz() < complete.nnz(), "tau = {tau} drops nothing");
    assert!(ilut.nnz() > a.nnz(), "tau = {tau} drops all fill-in");
    let ilutp = ILUTP::new(&a, tau, n, 0.5).unwrap();
    for (s, _) in extreme_scales() {
        let a_s = scaled(&a, s);
        let ilut_s = ILUT::new(&a_s, tau, n).unwrap_or_else(|e| panic!("ILUT at {s:e}: {e}"));
        assert_eq!(
            ilut_s.l().col_indices(),
            ilut.l().col_indices(),
            "ILUT L pattern at {s:e}"
        );
        assert_eq!(
            ilut_s.u().col_indices(),
            ilut.u().col_indices(),
            "ILUT U pattern at {s:e}"
        );
        assert!(
            bits(ilut_s.l().values()) == bits(ilut.l().values()),
            "ILUT L at {s:e}"
        );
        assert!(
            bits(ilut_s.u().values()) == scaled_bits(ilut.u().values(), s),
            "ILUT U at {s:e}"
        );
        let ilutp_s =
            ILUTP::new(&a_s, tau, n, 0.5).unwrap_or_else(|e| panic!("ILUTP at {s:e}: {e}"));
        assert_eq!(ilutp_s.perm(), ilutp.perm(), "ILUTP permutation at {s:e}");
        assert_eq!(
            ilutp_s.l().col_indices(),
            ilutp.l().col_indices(),
            "ILUTP L at {s:e}"
        );
        assert_eq!(
            ilutp_s.u().col_indices(),
            ilutp.u().col_indices(),
            "ILUTP U at {s:e}"
        );
        assert!(
            bits(ilutp_s.l().values()) == bits(ilutp.l().values()),
            "ILUTP L at {s:e}"
        );
        assert!(
            bits(ilutp_s.u().values()) == scaled_bits(ilutp.u().values(), s),
            "ILUTP U at {s:e}"
        );
    }
}

// =============================================================================
// Storage order of the rows
// =============================================================================

/// `a` with the entries of every row stored in reverse column order (CSR
/// does not require sorted rows).
fn reversed_rows(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut col_indices = Vec::with_capacity(a.nnz());
    let mut values = Vec::with_capacity(a.nnz());
    for i in 0..a.nrows() {
        for idx in (a.row_ptrs()[i]..a.row_ptrs()[i + 1]).rev() {
            col_indices.push(a.col_indices()[idx]);
            values.push(a.values()[idx]);
        }
    }
    CsrMatrix::new(
        a.nrows(),
        a.ncols(),
        a.row_ptrs().to_vec(),
        col_indices,
        values,
    )
    .unwrap()
}

/// Entries of a CSR matrix keyed by position, for comparing matrices whose
/// rows are stored in different orders.
fn entries_by_position(m: &CsrMatrix<f64>) -> std::collections::BTreeMap<(usize, usize), u64> {
    let mut entries = std::collections::BTreeMap::new();
    for i in 0..m.nrows() {
        for idx in m.row_ptrs()[i]..m.row_ptrs()[i + 1] {
            entries.insert((i, m.col_indices()[idx]), m.values()[idx].to_bits());
        }
    }
    entries
}

/// Every incomplete factorization must give the same factors whether or not
/// the rows of `A` are sorted. ILU(0) used to eliminate a row's entries in
/// storage order and to stop at the first column `j >= i`, so a row stored
/// as `[i, i - 1]` was not eliminated at all.
#[test]
fn test_incomplete_factorizations_ignore_row_storage_order() {
    let sorted = {
        let a = variable_spd(8);
        // A nonsymmetric 5-point operator with the same pattern.
        let values: Vec<f64> = a
            .values()
            .iter()
            .enumerate()
            .map(|(k, v)| {
                if *v < 0.0 {
                    v * (1.0 + (k % 3) as f64 / 8.0)
                } else {
                    *v
                }
            })
            .collect();
        CsrMatrix::new(
            a.nrows(),
            a.ncols(),
            a.row_ptrs().to_vec(),
            a.col_indices().to_vec(),
            values,
        )
        .unwrap()
    };
    let reversed = reversed_rows(&sorted);
    let n = sorted.nrows();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64 / 4.0).collect();

    let ilu0_sorted = ILU0::new(&sorted).unwrap();
    let ilu0_reversed = ILU0::new(&reversed).unwrap();
    assert!(
        entries_by_position(ilu0_sorted.lu()) == entries_by_position(ilu0_reversed.lu()),
        "ILU0 factors depend on the storage order of the rows"
    );
    // `apply` sums each row in storage order, so it agrees to rounding.
    for (s, r) in ilu0_sorted.apply(&b).iter().zip(ilu0_reversed.apply(&b)) {
        assert!((s - r).abs() <= 1e-14 * s.abs(), "ILU0 apply: {s} vs {r}");
    }

    for tau in [0.0, 0.05] {
        let s = ILUT::new(&sorted, tau, n).unwrap();
        let r = ILUT::new(&reversed, tau, n).unwrap();
        assert!(
            bits(s.l().values()) == bits(r.l().values()),
            "ILUT L, tau {tau}"
        );
        assert!(
            bits(s.u().values()) == bits(r.u().values()),
            "ILUT U, tau {tau}"
        );
        let s = ILUTP::new(&sorted, tau, n, 0.5).unwrap();
        let r = ILUTP::new(&reversed, tau, n, 0.5).unwrap();
        assert_eq!(s.perm(), r.perm(), "ILUTP permutation, tau {tau}");
        assert!(
            bits(s.u().values()) == bits(r.u().values()),
            "ILUTP U, tau {tau}"
        );
    }

    let spd = variable_spd(8);
    let spd_reversed = reversed_rows(&spd);
    let s = IC0::new(&spd).unwrap();
    let r = IC0::new(&spd_reversed).unwrap();
    assert!(bits(s.l().values()) == bits(r.l().values()), "IC0");
    let s = ICT::new(&spd, 0.05, n).unwrap();
    let r = ICT::new(&spd_reversed, 0.05, n).unwrap();
    assert!(bits(s.l().values()) == bits(r.l().values()), "ICT");
}
