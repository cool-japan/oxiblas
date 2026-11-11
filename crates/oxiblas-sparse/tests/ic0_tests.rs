//! Regression tests for the incomplete Cholesky factorization `IC0`.
//!
//! IC(0) keeps the sparsity pattern of the lower triangle of `A` and is
//! defined by `(L Lᵀ)_ij = a_ij` for every `(i, j)` in that pattern. The
//! elimination of row `i` therefore has to apply `Σ_{k<j} l_ik·l_jk` to every
//! retained `l_ij`. The factorization used to walk row `j` of the stored lower
//! factor looking for entries `k > j`, which it never holds, so that sum was
//! never applied. Tridiagonal and 5-point patterns hide this (no two retained
//! entries of a row are coupled), so the tests below use 9-point stencils,
//! dense blocks and random patterns, and compare against a dense reference.

// The dense reference and the property checks index rows and columns
// directly, as the formulas they implement do.
#![allow(clippy::needless_range_loop)]

use oxiblas_sparse::CsrMatrix;
use oxiblas_sparse::linalg::cholesky::IC0;

/// Builds a CSR matrix from `(row, col, value)` triplets; `reverse_rows`
/// stores the columns of every row in decreasing order (CSR does not require
/// sorted rows).
fn csr_from_triplets(
    n: usize,
    triplets: &[(usize, usize, f64)],
    reverse_rows: bool,
) -> CsrMatrix<f64> {
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(i, j, v) in triplets {
        rows[i].push((j, v));
    }
    let mut row_ptrs = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for row in &mut rows {
        row.sort_by_key(|&(j, _)| j);
        if reverse_rows {
            row.reverse();
        }
        for &(j, v) in row.iter() {
            col_indices.push(j);
            values.push(v);
        }
        row_ptrs.push(col_indices.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// SPD 9-point operator on a `grid x grid` mesh: off-diagonal `-1` for each
/// of the 8 neighbours and a strictly dominant, row-dependent diagonal.
fn nine_point(grid: usize, reverse_rows: bool) -> CsrMatrix<f64> {
    let n = grid * grid;
    let mut triplets = Vec::new();
    for r in 0..grid {
        for c in 0..grid {
            let i = r * grid + c;
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    let (rr, cc) = (r as i64 + dr, c as i64 + dc);
                    if rr < 0 || cc < 0 || rr >= grid as i64 || cc >= grid as i64 {
                        continue;
                    }
                    let j = rr as usize * grid + cc as usize;
                    let value = if i == j {
                        8.5 + 0.25 * (i % 3) as f64
                    } else {
                        -1.0
                    };
                    triplets.push((i, j, value));
                }
            }
        }
    }
    csr_from_triplets(n, &triplets, reverse_rows)
}

/// SplitMix64, for reproducible pseudo-random patterns.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9_007_199_254_740_992.0
    }
}

/// Random symmetric M-matrix (negative off-diagonals, diagonal = row sum of
/// magnitudes + 1): strictly diagonally dominant, so IC(0) exists
/// (Meijerink and van der Vorst).
fn random_m_matrix(n: usize, density: f64, seed: u64) -> CsrMatrix<f64> {
    let mut rng = SplitMix(seed);
    let mut dense = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..i {
            if rng.unit() < density {
                let v = -(0.125 + rng.unit());
                dense[i][j] = v;
                dense[j][i] = v;
            }
        }
    }
    let mut triplets = Vec::new();
    for i in 0..n {
        let off: f64 = dense[i].iter().map(|v| v.abs()).sum();
        dense[i][i] = off + 1.0;
        for j in 0..n {
            if dense[i][j] != 0.0 {
                triplets.push((i, j, dense[i][j]));
            }
        }
    }
    csr_from_triplets(n, &triplets, false)
}

/// Dense copy of a CSR matrix.
fn to_dense(a: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let n = a.nrows();
    let mut dense = vec![vec![0.0; a.ncols()]; n];
    for (i, row) in dense.iter_mut().enumerate() {
        for idx in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
            row[a.col_indices()[idx]] = a.values()[idx];
        }
    }
    dense
}

/// Pattern of the lower triangle of `A` (stored entries).
fn lower_pattern(a: &CsrMatrix<f64>) -> Vec<Vec<bool>> {
    let n = a.nrows();
    let mut pattern = vec![vec![false; n]; n];
    for (i, row) in pattern.iter_mut().enumerate() {
        for idx in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
            let j = a.col_indices()[idx];
            if j <= i {
                row[j] = true;
            }
        }
    }
    pattern
}

/// Dense reference IC(0): `l_ij = (a_ij − Σ_{k<j} l_ik·l_jk) / l_jj` and
/// `l_ii = sqrt(a_ii − Σ_{k<i} l_ik²)` over the lower pattern of `A`, summing
/// in increasing `k`.
fn dense_ic0(a: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let n = a.nrows();
    let dense = to_dense(a);
    let pattern = lower_pattern(a);
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            if !pattern[i][j] {
                continue;
            }
            let mut s = dense[i][j];
            for k in 0..j {
                if pattern[i][k] && pattern[j][k] {
                    s -= l[i][k] * l[j][k];
                }
            }
            if j < i {
                l[i][j] = s / l[j][j];
            } else {
                l[i][i] = s.sqrt();
            }
        }
    }
    l
}

/// Largest `|(L Lᵀ)_ij − a_ij|` over the lower pattern of `A`, and the
/// largest allowed rounding error `64·eps·(|a_ij| + Σ_k |l_ik·l_jk|)` at the
/// same positions (the check fails if any position exceeds its own bound).
fn ic0_property_violation(a: &CsrMatrix<f64>, l: &[Vec<f64>]) -> (f64, bool) {
    let n = a.nrows();
    let dense = to_dense(a);
    let pattern = lower_pattern(a);
    let mut worst = 0.0_f64;
    let mut within_bounds = true;
    for i in 0..n {
        for j in 0..=i {
            if !pattern[i][j] {
                continue;
            }
            let (mut product, mut magnitude) = (0.0_f64, dense[i][j].abs());
            for k in 0..=j {
                product += l[i][k] * l[j][k];
                magnitude += (l[i][k] * l[j][k]).abs();
            }
            let error = (product - dense[i][j]).abs();
            worst = worst.max(error);
            if error > 64.0 * f64::EPSILON * magnitude {
                within_bounds = false;
            }
        }
    }
    (worst, within_bounds)
}

/// The IC(0) factor of `a` as a dense lower triangle, checking that it is
/// stored with exactly the pattern of the lower triangle of `a`.
fn ic0_dense_factor(a: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let ic0 = IC0::new(a).unwrap();
    let l = ic0.l();
    let pattern = lower_pattern(a);
    let n = a.nrows();
    let mut stored = vec![vec![false; n]; n];
    for (i, row) in stored.iter_mut().enumerate() {
        for idx in l.row_ptrs()[i]..l.row_ptrs()[i + 1] {
            let j = l.col_indices()[idx];
            assert!(!row[j], "duplicate entry ({i}, {j}) in L");
            row[j] = true;
        }
    }
    assert!(stored == pattern, "L does not have the pattern of tril(A)");
    to_dense(l)
}

#[test]
fn test_ic0_dense_3x3_example_exact() {
    // [[4,1,1],[1,4,1],[1,1,4]] has a dense lower triangle, so IC(0) is the
    // complete Cholesky factorization. The missing l_20·l_10 term used to
    // give l_21 = 1/l_11 = 0.5164 instead of 0.75/l_11 = 0.3873, and
    // (L Lᵀ − A)_21 = 0.25.
    let a = csr_from_triplets(
        3,
        &[
            (0, 0, 4.0),
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 0, 1.0),
            (1, 1, 4.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
        ],
        false,
    );
    let l = ic0_dense_factor(&a);
    let l11 = 3.75_f64.sqrt();
    let l21 = (1.0 - 0.5 * 0.5) / l11;
    let l22 = ((4.0 - 0.25) - l21 * l21).sqrt();
    let expected = [[2.0, 0.0, 0.0], [0.5, l11, 0.0], [0.5, l21, l22]];
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(
                l[i][j].to_bits(),
                expected[i][j].to_bits(),
                "l[{i}][{j}] = {} (expected {})",
                l[i][j],
                expected[i][j]
            );
        }
    }
    assert!((l21 - 0.387_298_334_620_741_7).abs() <= 1e-15);
    let (worst, within) = ic0_property_violation(&a, &l);
    assert!(within, "max |(L L^T - A)_ij| = {worst:e}");
}

#[test]
fn test_ic0_nine_point_satisfies_ic0_property() {
    // A correct IC(0) reproduces A on its pattern to rounding (the broken
    // elimination left max |L L^T - A| = 0.15 on the 4 x 4 grid).
    for grid in [4, 10] {
        let a = nine_point(grid, false);
        let l = ic0_dense_factor(&a);
        let (worst, within) = ic0_property_violation(&a, &l);
        assert!(
            within,
            "9-point {grid}x{grid}: max |(L L^T - A)_ij| on the pattern = {worst:e}"
        );
        assert!(worst <= 1e-14, "9-point {grid}x{grid}: {worst:e}");
    }
}

#[test]
fn test_ic0_matches_dense_reference() {
    // The row-wise elimination sums Σ_{k<j} l_ik·l_jk in increasing k, as the
    // dense reference does, so the factors agree bit for bit.
    let mut matrices = vec![nine_point(4, false), nine_point(10, false)];
    for (n, density, seed) in [(12, 0.5, 1), (30, 0.2, 2), (60, 0.08, 3), (25, 1.0, 4)] {
        matrices.push(random_m_matrix(n, density, seed));
    }
    for (index, a) in matrices.iter().enumerate() {
        let l = ic0_dense_factor(a);
        let reference = dense_ic0(a);
        for (i, (row, ref_row)) in l.iter().zip(&reference).enumerate() {
            for (j, (v, r)) in row.iter().zip(ref_row).enumerate() {
                assert_eq!(
                    v.to_bits(),
                    r.to_bits(),
                    "matrix {index}: l[{i}][{j}] = {v:e}, reference {r:e}"
                );
            }
        }
    }
}

#[test]
fn test_ic0_random_patterns_satisfy_ic0_property() {
    for seed in 10..30u64 {
        let n = 8 + (seed as usize * 7) % 40;
        let density = 0.05 + 0.04 * (seed % 10) as f64;
        let a = random_m_matrix(n, density, seed);
        let l = ic0_dense_factor(&a);
        let (worst, within) = ic0_property_violation(&a, &l);
        assert!(
            within,
            "random pattern (n = {n}, density = {density}, seed = {seed}): \
             max |(L L^T - A)_ij| on the pattern = {worst:e}"
        );
    }
}

#[test]
fn test_ic0_unsorted_rows_give_the_same_factor() {
    // CSR rows need not be sorted; the factor must not depend on the storage
    // order of the entries within a row.
    for grid in [4, 6] {
        let sorted = nine_point(grid, false);
        let reversed = nine_point(grid, true);
        let l_sorted = ic0_dense_factor(&sorted);
        let l_reversed = ic0_dense_factor(&reversed);
        for (i, (row_s, row_r)) in l_sorted.iter().zip(&l_reversed).enumerate() {
            for (j, (s, r)) in row_s.iter().zip(row_r).enumerate() {
                assert_eq!(s.to_bits(), r.to_bits(), "grid {grid}: l[{i}][{j}]");
            }
        }
    }
}

#[test]
fn test_ic0_nine_point_factor_exactly_scaled() {
    // L(s·A) = 2^k · L(A) for s = 4^k, also on a pattern where the elimination
    // couples the retained entries of a row.
    let a = nine_point(10, false);
    let l = IC0::new(&a).unwrap();
    for (s, sqrt_s) in [
        (2.0_f64.powi(-128), 2.0_f64.powi(-64)),
        (2.0_f64.powi(128), 2.0_f64.powi(64)),
    ] {
        let values: Vec<f64> = a.values().iter().map(|v| v * s).collect();
        let a_s = CsrMatrix::new(
            a.nrows(),
            a.ncols(),
            a.row_ptrs().to_vec(),
            a.col_indices().to_vec(),
            values,
        )
        .unwrap();
        let l_s = IC0::new(&a_s).unwrap();
        assert_eq!(l_s.l().row_ptrs(), l.l().row_ptrs());
        assert_eq!(l_s.l().col_indices(), l.l().col_indices());
        for (vs, v) in l_s.l().values().iter().zip(l.l().values()) {
            assert_eq!(vs.to_bits(), (v * sqrt_s).to_bits(), "scale {s:e}");
        }
    }
}

#[test]
fn test_ic0_apply_solves_with_the_factor() {
    // apply(b) solves (L Lᵀ) x = b with the stored factor.
    let a = nine_point(6, false);
    let ic0 = IC0::new(&a).unwrap();
    let l = to_dense(ic0.l());
    let n = a.nrows();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64 / 4.0).collect();
    let x = ic0.apply(&b);
    // y = Lᵀ x, then L y must reproduce b.
    let y: Vec<f64> = (0..n)
        .map(|i| (i..n).map(|k| l[k][i] * x[k]).sum())
        .collect();
    for i in 0..n {
        let li_y: f64 = (0..=i).map(|k| l[i][k] * y[k]).sum();
        assert!(
            (li_y - b[i]).abs() <= 1e-12 * b[i].abs(),
            "row {i}: {li_y} vs {}",
            b[i]
        );
    }
}
