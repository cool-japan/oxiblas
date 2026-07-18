//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::csr::CsrMatrix;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Convenience function for randomized sparse SVD.
pub fn randomized_sparse_svd<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    a: &CsrMatrix<T>,
    k: usize,
    power_iterations: usize,
) -> Result<RandomizedSparseSvdResult<T>, SVDError> {
    let config = RandomizedSparseSvdConfig {
        num_singular_values: k,
        power_iterations,
        ..Default::default()
    };
    let rsvd = RandomizedSparseSvd::new(config);
    rsvd.compute(a)
}
/// Transpose a dense matrix stored as row vectors.
pub(super) fn transpose_dense<T: Clone>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let _rows = matrix.len();
    let cols = matrix[0].len();
    let mut result: Vec<Vec<T>> = vec![vec![]; cols];
    for row in matrix {
        for (j, val) in row.iter().enumerate() {
            result[j].push(val.clone());
        }
    }
    result
}
/// Orthonormalize dense vectors using modified Gram-Schmidt.
#[allow(dead_code)]
fn orthonormalize_dense<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    vectors: &[Vec<T>],
) -> Vec<Vec<T>> {
    let mut result: Vec<Vec<T>> = Vec::new();
    for v in vectors {
        let mut u = v.clone();
        for q in &result {
            let dot: T = u
                .iter()
                .zip(q.iter())
                .map(|(ui, qi)| ui.clone() * qi.clone())
                .fold(T::zero(), |acc, x| acc + x);
            for (ui, qi) in u.iter_mut().zip(q.iter()) {
                *ui = ui.clone() - dot.clone() * qi.clone();
            }
        }
        let norm: T = Real::sqrt(
            u.iter()
                .map(|x| x.clone() * x.clone())
                .fold(T::zero(), |acc, x| acc + x),
        );
        if norm > T::from_f64(1e-14).unwrap_or_else(T::zero) {
            u = u.iter().map(|x| x.clone() / norm.clone()).collect();
            result.push(u);
        }
    }
    result
}
/// Orthonormalize a set of column vectors (each of length `dim`) using modified
/// Gram-Schmidt with reorthogonalization and a rank-revealing tolerance.
///
/// Returns `(basis, coeffs)` where `basis` holds the `p'` accepted orthonormal
/// vectors (the *numerical* rank of the input set) and `coeffs` is a `p' × ncols`
/// matrix such that `column_j = Σ_i coeffs[i][j] · basis[i]`.
///
/// Linearly dependent columns produce a residual whose norm falls below the
/// relative tolerance, so they contribute **no** new basis vector: the numerical
/// rank `p'` can therefore be strictly smaller than `ncols`. This is precisely
/// what prevents rank-deficient residual blocks from being counted as though they
/// added `ncols` new orthogonal directions.
pub(super) fn orthonormal_basis_with_coeffs<
    T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive,
>(
    columns: &[Vec<T>],
    dim: usize,
) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    let ncols = columns.len();
    if ncols == 0 || dim == 0 {
        return (Vec::new(), Vec::new());
    }
    // The largest input column norm sets the scale for the relative rank test.
    let mut max_norm = T::zero();
    for column in columns.iter() {
        let mut sum_sq = T::zero();
        for value in column.iter().take(dim) {
            sum_sq = sum_sq + value.clone() * value.clone();
        }
        let norm = Real::sqrt(sum_sq);
        if norm > max_norm {
            max_norm = norm;
        }
    }
    let rel_tol = T::from_f64(1e-9).unwrap_or_else(T::zero);
    let threshold = rel_tol * max_norm;
    let mut basis: Vec<Vec<T>> = Vec::new();
    let mut coeff_columns: Vec<Vec<T>> = Vec::with_capacity(ncols);
    for column in columns.iter() {
        let mut residual = column.clone();
        let mut coeff = vec![T::zero(); basis.len()];
        // Two Gram-Schmidt passes (reorthogonalization) for numerical stability.
        for _pass in 0..2 {
            for (bi, basis_vec) in basis.iter().enumerate() {
                let mut dot = T::zero();
                for i in 0..dim {
                    dot = dot + basis_vec[i].clone() * residual[i].clone();
                }
                for i in 0..dim {
                    residual[i] = residual[i].clone() - dot.clone() * basis_vec[i].clone();
                }
                coeff[bi] = coeff[bi].clone() + dot;
            }
        }
        let mut sum_sq = T::zero();
        for value in residual.iter().take(dim) {
            sum_sq = sum_sq + value.clone() * value.clone();
        }
        let norm = Real::sqrt(sum_sq);
        if norm > threshold {
            let inv = T::one() / norm.clone();
            for value in residual.iter_mut().take(dim) {
                *value = value.clone() * inv.clone();
            }
            basis.push(residual);
            coeff.push(norm);
        }
        coeff_columns.push(coeff);
    }
    let rank = basis.len();
    let mut coeffs = vec![vec![T::zero(); ncols]; rank];
    for (j, coeff) in coeff_columns.iter().enumerate() {
        for (i, value) in coeff.iter().enumerate() {
            if i < rank {
                coeffs[i][j] = value.clone();
            }
        }
    }
    (basis, coeffs)
}
/// Full singular value decomposition of a small dense matrix via **one-sided
/// Jacobi rotations**.
///
/// One-sided Jacobi orthogonalizes the columns of the matrix through a sequence
/// of plane rotations. Unlike power iteration with deflation — which reuses a
/// single deterministic start vector and therefore loses the second copy of a
/// repeated singular value once the first has been deflated out of that
/// subspace — Jacobi resolves repeated and tightly clustered singular values
/// *exactly*: when two singular values are equal the relevant columns are
/// already mutually orthogonal (`A Aᵀ` is a scalar on that subspace), so the
/// rotation that would separate them is the identity and both magnitudes are
/// retained.
///
/// Returns a [`RandomizedSparseSvdResult`] whose `u` is `rows × r`, `v` is
/// `cols × r` and `singular_values` has length `r = min(rows, cols)`, sorted in
/// descending order. Both `u` and `v` store singular vectors as **columns**,
/// i.e. `u[i][j]` is the `i`-th component of the `j`-th left singular vector and
/// `v[i][j]` the `i`-th component of the `j`-th right singular vector.
pub(super) fn dense_svd_full<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    matrix: &[Vec<T>],
) -> Result<RandomizedSparseSvdResult<T>, SVDError> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(SVDError::InvalidConfig("Empty matrix".to_string()));
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    if rows >= cols {
        let (u, s, v) = one_sided_jacobi_svd(matrix, rows, cols);
        Ok(RandomizedSparseSvdResult {
            singular_values: s,
            u: Some(u),
            v: Some(v),
        })
    } else {
        // One-sided Jacobi needs at least as many rows as columns; run it on the
        // transpose and swap the roles of the left and right singular vectors.
        let transposed = transpose_dense(matrix);
        let (u_t, s, v_t) = one_sided_jacobi_svd(&transposed, cols, rows);
        Ok(RandomizedSparseSvdResult {
            singular_values: s,
            u: Some(v_t),
            v: Some(u_t),
        })
    }
}

/// One-sided Jacobi SVD kernel for a matrix with `m >= n` (`m` rows, `n` cols).
///
/// Returns `(u, s, v)` with `u` of shape `m × n`, `s` of length `n` and `v` of
/// shape `n × n`, with singular values (and their matching singular vectors)
/// sorted in descending order. The columns of `u` are the normalized, mutually
/// orthogonal columns of the rotated matrix; `v` accumulates the applied Jacobi
/// rotations so that `input = u · diag(s) · vᵀ`.
fn one_sided_jacobi_svd<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    input: &[Vec<T>],
    m: usize,
    n: usize,
) -> (Vec<Vec<T>>, Vec<T>, Vec<Vec<T>>) {
    // Working copy whose columns are progressively orthogonalized.
    let mut a: Vec<Vec<T>> = input.to_vec();
    // Accumulated right rotations, initialized to the identity.
    let mut v = vec![vec![T::zero(); n]; n];
    for (i, v_row) in v.iter_mut().enumerate() {
        v_row[i] = T::one();
    }
    let eps = T::from_f64(1e-15).unwrap_or_else(T::zero);
    let two = T::one() + T::one();
    let max_sweeps = 60;
    for _sweep in 0..max_sweeps {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = T::zero();
                let mut beta = T::zero();
                let mut gamma = T::zero();
                for row in a.iter() {
                    alpha = alpha + row[p].clone() * row[p].clone();
                    beta = beta + row[q].clone() * row[q].clone();
                    gamma = gamma + row[p].clone() * row[q].clone();
                }
                let denom = Real::sqrt(alpha.clone() * beta.clone());
                if denom <= eps {
                    continue;
                }
                // Columns already (numerically) orthogonal: nothing to do.
                if Scalar::abs(gamma.clone()) <= eps.clone() * denom {
                    continue;
                }
                converged = false;
                // Jacobi rotation that diagonalizes the 2×2 Gram submatrix
                // [[alpha, gamma], [gamma, beta]].
                let zeta = (beta.clone() - alpha.clone()) / (two.clone() * gamma.clone());
                let abs_zeta = Scalar::abs(zeta.clone());
                let root = Real::sqrt(T::one() + zeta.clone() * zeta.clone());
                let magnitude = T::one() / (abs_zeta + root);
                let t = if zeta < T::zero() {
                    T::zero() - magnitude
                } else {
                    magnitude
                };
                let c = T::one() / Real::sqrt(T::one() + t.clone() * t.clone());
                let s = c.clone() * t;
                for row in a.iter_mut() {
                    let aip = row[p].clone();
                    let aiq = row[q].clone();
                    row[p] = c.clone() * aip.clone() - s.clone() * aiq.clone();
                    row[q] = s.clone() * aip + c.clone() * aiq;
                }
                for v_row in v.iter_mut() {
                    let vip = v_row[p].clone();
                    let viq = v_row[q].clone();
                    v_row[p] = c.clone() * vip.clone() - s.clone() * viq.clone();
                    v_row[q] = s.clone() * vip + c.clone() * viq;
                }
            }
        }
        if converged {
            break;
        }
    }
    // Column norms are the singular values; normalized columns form U.
    let mut norms = vec![T::zero(); n];
    for (j, norm_slot) in norms.iter_mut().enumerate() {
        let mut sum_sq = T::zero();
        for row in a.iter() {
            sum_sq = sum_sq + row[j].clone() * row[j].clone();
        }
        *norm_slot = Real::sqrt(sum_sq);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        norms[y]
            .partial_cmp(&norms[x])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let tol = T::from_f64(1e-14).unwrap_or_else(T::zero);
    let mut singular_values = Vec::with_capacity(n);
    let mut u = vec![vec![T::zero(); n]; m];
    let mut v_sorted = vec![vec![T::zero(); n]; n];
    for (new_j, &old_j) in order.iter().enumerate() {
        let sigma = norms[old_j].clone();
        singular_values.push(sigma.clone());
        if sigma > tol {
            let inv = T::one() / sigma;
            for i in 0..m {
                u[i][new_j] = a[i][old_j].clone() * inv.clone();
            }
        }
        for i in 0..n {
            v_sorted[i][new_j] = v[i][old_j].clone();
        }
    }
    (u, singular_values, v_sorted)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }
    #[test]
    fn test_truncated_svd_diagonal() {
        let values = vec![3.0, 2.0, 1.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let config = TruncatedSVDConfig {
            num_singular_values: 2,
            max_iterations: 100,
            tolerance: 1e-10,
            compute_vectors: true,
            krylov_dimension: 3,
            full_reorthogonalization: true,
        };
        let svd = TruncatedSVD::new(config);
        let result = svd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 2);
        assert!(approx_eq(result.singular_values[0], 3.0, 1e-6));
        assert!(approx_eq(result.singular_values[1], 2.0, 1e-6));
    }
    #[test]
    fn test_truncated_svd_simple_matrix() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4, 6];
        let a = CsrMatrix::new(3, 2, row_ptrs, col_indices, values).unwrap();
        let config = TruncatedSVDConfig {
            num_singular_values: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            compute_vectors: true,
            krylov_dimension: 2,
            full_reorthogonalization: true,
        };
        let svd = TruncatedSVD::new(config);
        let result = svd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 1);
        assert!(result.singular_values[0] > 9.0);
        assert!(result.singular_values[0] < 10.0);
    }
    #[test]
    fn test_truncated_svd_rectangular_tall() {
        let values = vec![2.0, 3.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2, 2, 2];
        let a = CsrMatrix::new(4, 2, row_ptrs, col_indices, values).unwrap();
        let config = TruncatedSVDConfig {
            num_singular_values: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            compute_vectors: true,
            krylov_dimension: 2,
            full_reorthogonalization: true,
        };
        let svd = TruncatedSVD::new(config);
        let result = svd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 1);
        assert!(approx_eq(result.singular_values[0], 3.0, 1e-6));
    }
    #[test]
    fn test_truncated_svd_rectangular_wide() {
        let values = vec![2.0, 3.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2];
        let a = CsrMatrix::new(2, 4, row_ptrs, col_indices, values).unwrap();
        let config = TruncatedSVDConfig {
            num_singular_values: 1,
            max_iterations: 100,
            tolerance: 1e-10,
            compute_vectors: true,
            krylov_dimension: 2,
            full_reorthogonalization: true,
        };
        let svd = TruncatedSVD::new(config);
        let result = svd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 1);
        assert!(approx_eq(result.singular_values[0], 3.0, 1e-6));
    }
    #[test]
    fn test_randomized_svd_diagonal() {
        let values = vec![5.0, 4.0, 3.0, 2.0];
        let col_indices = vec![0, 1, 2, 3];
        let row_ptrs = vec![0, 1, 2, 3, 4];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let config = RandomizedSparseSvdConfig {
            num_singular_values: 2,
            oversampling: 2,
            power_iterations: 2,
            seed: Some(42),
            compute_vectors: true,
        };
        let rsvd = RandomizedSparseSvd::new(config);
        let result = rsvd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 2);
        assert!(result.singular_values[0] > 4.5);
        assert!(result.singular_values[1] > 3.5);
    }
    #[test]
    fn test_randomized_svd_simple_matrix() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let col_indices = vec![0, 1, 1, 2, 2, 3, 3];
        let row_ptrs = vec![0, 2, 4, 6, 7];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let config = RandomizedSparseSvdConfig {
            num_singular_values: 2,
            oversampling: 2,
            power_iterations: 3,
            seed: Some(12345),
            compute_vectors: true,
        };
        let rsvd = RandomizedSparseSvd::new(config);
        let result = rsvd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 2);
        assert!(result.singular_values[0] > 0.0);
        assert!(result.singular_values[0] >= result.singular_values[1]);
    }
    #[test]
    fn test_randomized_svd_convenience_function() {
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let col_indices = vec![0, 1, 2, 3, 4];
        let row_ptrs = vec![0, 1, 2, 3, 4, 5];
        let a = CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap();
        let config = RandomizedSparseSvdConfig {
            num_singular_values: 2,
            oversampling: 2,
            power_iterations: 2,
            seed: Some(42),
            compute_vectors: true,
        };
        let rsvd = RandomizedSparseSvd::new(config);
        let result = rsvd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 2);
        assert!(result.singular_values[0] > 4.0);
    }
    #[test]
    fn test_randomized_svd_vectors_orthonormal() {
        let values = vec![4.0, 3.0, 2.0, 1.0];
        let col_indices = vec![0, 1, 2, 3];
        let row_ptrs = vec![0, 1, 2, 3, 4];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let config = RandomizedSparseSvdConfig {
            num_singular_values: 2,
            oversampling: 2,
            power_iterations: 2,
            seed: Some(99),
            compute_vectors: true,
        };
        let rsvd = RandomizedSparseSvd::new(config);
        let result = rsvd.compute(&a).unwrap();
        let u = result.u.as_ref().unwrap();
        for i in 0..u.len() {
            for j in 0..u.len() {
                let mut dot = 0.0;
                for k in 0..u[i].len() {
                    dot += u[i][k] * u[j][k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 0.1),
                    "U^T*U[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }
    #[test]
    fn test_randomized_svd_tall_matrix() {
        let values = vec![5.0, 3.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2, 2, 2, 2];
        let a = CsrMatrix::new(5, 2, row_ptrs, col_indices, values).unwrap();
        let config = RandomizedSparseSvdConfig {
            num_singular_values: 1,
            oversampling: 1,
            power_iterations: 2,
            seed: Some(42),
            compute_vectors: true,
        };
        let rsvd = RandomizedSparseSvd::new(config);
        let result = rsvd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 1);
        assert!(result.singular_values[0] > 4.0);
    }
    #[test]
    fn test_randomized_svd_wide_matrix() {
        let values = vec![5.0, 3.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2];
        let a = CsrMatrix::new(2, 5, row_ptrs, col_indices, values).unwrap();
        let config = RandomizedSparseSvdConfig {
            num_singular_values: 1,
            oversampling: 1,
            power_iterations: 2,
            seed: Some(42),
            compute_vectors: true,
        };
        let rsvd = RandomizedSparseSvd::new(config);
        let result = rsvd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 1);
        assert!(result.singular_values[0] > 4.0);
    }
    #[test]
    fn test_randomized_svd_no_vectors() {
        let values = vec![3.0, 2.0, 1.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let config = RandomizedSparseSvdConfig {
            num_singular_values: 2,
            oversampling: 1,
            power_iterations: 1,
            seed: Some(42),
            compute_vectors: false,
        };
        let rsvd = RandomizedSparseSvd::new(config);
        let result = rsvd.compute(&a).unwrap();
        assert_eq!(result.singular_values.len(), 2);
        assert!(result.u.is_none());
        assert!(result.v.is_none());
    }
    #[test]
    fn test_incremental_svd_initialization() {
        let values = vec![3.0, 2.0, 1.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 3,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        let result = isvd.initialize(&a, 2);
        assert!(result.is_ok());
        assert_eq!(isvd.rank(), 2);
        assert_eq!(isvd.dimensions(), (3, 3));
        let (u, s, vt) = isvd.get_svd();
        assert_eq!(u.len(), 3);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.len(), 2);
    }
    #[test]
    fn test_incremental_svd_add_rows() {
        let values = vec![2.0, 1.0, 1.0, 2.0];
        let col_indices = vec![0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 5,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 2).unwrap();
        let new_rows = vec![vec![1.0, 1.0]];
        let result = isvd.add_rows(&new_rows);
        assert!(result.is_ok());
        assert_eq!(isvd.dimensions(), (3, 2));
        let (u, s, _vt) = isvd.get_svd();
        assert_eq!(u.len(), 3);
        assert!(!s.is_empty());
    }
    #[test]
    fn test_incremental_svd_add_columns() {
        let values = vec![2.0, 1.0, 1.0, 2.0];
        let col_indices = vec![0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 5,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 2).unwrap();
        let new_cols = vec![vec![1.0, 1.0]];
        let result = isvd.add_columns(&new_cols);
        assert!(result.is_ok());
        assert_eq!(isvd.dimensions(), (2, 3));
        let (_u, s, vt) = isvd.get_svd();
        assert_eq!(vt.len(), s.len());
        if !vt.is_empty() {
            assert_eq!(vt[0].len(), 3);
        }
    }
    #[test]
    fn test_incremental_svd_orthogonality() {
        let values = vec![2.0, 1.0, 1.0, 2.0];
        let col_indices = vec![0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 2,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 2).unwrap();
        let (u, _s, vt) = isvd.get_svd();
        let k = u[0].len();
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for row in u.iter() {
                    dot += row[i] * row[j];
                }
                if i == j {
                    assert!(
                        approx_eq(dot, 1.0, 1e-6),
                        "U diagonal {},{} = {}",
                        i,
                        j,
                        dot
                    );
                } else {
                    assert!(
                        approx_eq(dot, 0.0, 1e-6),
                        "U off-diag {},{} = {}",
                        i,
                        j,
                        dot
                    );
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..vt[i].len() {
                    dot += vt[i][l] * vt[j][l];
                }
                if i == j {
                    assert!(
                        approx_eq(dot, 1.0, 1e-6),
                        "V diagonal {},{} = {}",
                        i,
                        j,
                        dot
                    );
                } else {
                    assert!(
                        approx_eq(dot, 0.0, 1e-6),
                        "V off-diag {},{} = {}",
                        i,
                        j,
                        dot
                    );
                }
            }
        }
    }
    #[test]
    fn test_incremental_svd_rank_preservation() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 5,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 2).unwrap();
        let initial_rank = isvd.rank();
        let new_rows = vec![vec![1.0, 1.0]];
        isvd.add_rows(&new_rows).unwrap();
        let new_rank = isvd.rank();
        assert!(
            new_rank <= initial_rank + 1,
            "Rank grew too much: {} -> {}",
            initial_rank,
            new_rank
        );
    }
    #[test]
    fn test_incremental_svd_error_empty_matrix() {
        let values: Vec<f64> = vec![];
        let col_indices: Vec<usize> = vec![];
        let row_ptrs = vec![0];
        let a = CsrMatrix::new(0, 0, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 5,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        let result = isvd.initialize(&a, 1);
        assert!(result.is_err());
    }
    #[test]
    fn test_incremental_svd_error_wrong_dimensions() {
        let values = vec![1.0, 1.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2];
        let a = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 5,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 1).unwrap();
        let wrong_row = vec![vec![1.0, 2.0, 3.0]];
        let result = isvd.add_rows(&wrong_row);
        assert!(result.is_err());
        let wrong_col = vec![vec![1.0]];
        let result2 = isvd.add_columns(&wrong_col);
        assert!(result2.is_err());
    }
    #[test]
    fn test_incremental_svd_multiple_updates() {
        let values = vec![2.0, 1.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2];
        let a = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 10,
            tolerance: 1e-10,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 2).unwrap();
        let new_row1 = vec![vec![1.0, 0.0]];
        isvd.add_rows(&new_row1).unwrap();
        assert_eq!(isvd.dimensions(), (3, 2));
        let new_row2 = vec![vec![0.0, 1.0]];
        isvd.add_rows(&new_row2).unwrap();
        assert_eq!(isvd.dimensions(), (4, 2));
        let new_col1 = vec![vec![1.0, 1.0, 1.0, 1.0]];
        isvd.add_columns(&new_col1).unwrap();
        assert_eq!(isvd.dimensions(), (4, 3));
        let (_u, s, _vt) = isvd.get_svd();
        assert!(!s.is_empty());
        for &sigma in s {
            assert!(sigma >= 0.0);
        }
    }

    /// Reconstruct a dense matrix from the `dense_svd_full` factors (columns are
    /// singular vectors): `A[i][j] = Σ_l u[i][l] · s[l] · v[j][l]`.
    fn reconstruct_from_columns(u: &[Vec<f64>], s: &[f64], v: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = u.len();
        let n = v.len();
        let r = s.len();
        let mut out = vec![vec![0.0; n]; m];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, out_val) in out_row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for l in 0..r {
                    sum += u[i][l] * s[l] * v[j][l];
                }
                *out_val = sum;
            }
        }
        out
    }

    /// Reconstruct from the incremental-SVD factors `(U, Σ, Vᵀ)`.
    fn reconstruct_from_incremental(u: &[Vec<f64>], s: &[f64], vt: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = u.len();
        let n = if vt.is_empty() { 0 } else { vt[0].len() };
        let r = s.len();
        let mut out = vec![vec![0.0; n]; m];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, out_val) in out_row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for l in 0..r {
                    sum += u[i][l] * s[l] * vt[l][j];
                }
                *out_val = sum;
            }
        }
        out
    }

    fn frobenius_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        let mut sum = 0.0;
        for (row_a, row_b) in a.iter().zip(b.iter()) {
            for (va, vb) in row_a.iter().zip(row_b.iter()) {
                let d = va - vb;
                sum += d * d;
            }
        }
        sum.sqrt()
    }

    #[test]
    fn test_dense_svd_repeated_singular_values() {
        // diag(4, 3, 1, 1): the repeated unit singular value must be resolved.
        // The old power-iteration-with-deflation helper returned [4, 3, 1, ~0].
        let matrix = vec![
            vec![4.0, 0.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let result = dense_svd_full(&matrix).unwrap();
        assert_eq!(result.singular_values.len(), 4);
        let mut sorted = result.singular_values.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 4.0).abs() < 1e-10, "s0 = {}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1e-10, "s1 = {}", sorted[1]);
        assert!((sorted[2] - 1.0).abs() < 1e-10, "s2 = {}", sorted[2]);
        assert!(
            (sorted[3] - 1.0).abs() < 1e-10,
            "repeated singular value not resolved: s3 = {}",
            sorted[3]
        );
        let u = result.u.clone().unwrap();
        let v = result.v.clone().unwrap();
        let recon = reconstruct_from_columns(&u, &result.singular_values, &v);
        let err = frobenius_diff(&matrix, &recon);
        assert!(err < 1e-10, "reconstruction error {}", err);
    }

    #[test]
    fn test_dense_svd_general_reconstruction() {
        // Square, tall, wide and rank-deficient cases all reconstruct exactly.
        let matrices: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![
                vec![2.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 5.0],
            ],
        ];
        for matrix in &matrices {
            let result = dense_svd_full(matrix).unwrap();
            let u = result.u.clone().unwrap();
            let v = result.v.clone().unwrap();
            let recon = reconstruct_from_columns(&u, &result.singular_values, &v);
            let err = frobenius_diff(matrix, &recon);
            assert!(err < 1e-9, "reconstruction error {} for {:?}", err, matrix);
            for w in result.singular_values.windows(2) {
                assert!(w[0] >= w[1] - 1e-12, "singular values not descending");
            }
            for &sv in &result.singular_values {
                assert!(sv >= -1e-12, "negative singular value {}", sv);
            }
        }
    }

    #[test]
    fn test_incremental_svd_sequential_single_row_adds() {
        // Start from a 2×4 matrix with singular values 4 and 3, then append the
        // unit rows e2 and e3. The second update's coupling matrix is
        // diag(4, 3, 1, 1) with a repeated singular value 1. With the old dense
        // SVD helper the reconstruction error came out ~1.0.
        let a = CsrMatrix::new(2, 4, vec![0, 1, 2], vec![0, 1], vec![4.0, 3.0]).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 8,
            tolerance: 1e-12,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 2).unwrap();
        isvd.add_rows(&[vec![0.0, 0.0, 1.0, 0.0]]).unwrap();
        isvd.add_rows(&[vec![0.0, 0.0, 0.0, 1.0]]).unwrap();
        assert_eq!(isvd.dimensions(), (4, 4));
        let expected = vec![
            vec![4.0, 0.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let (u, s, vt) = isvd.get_svd();
        let recon = reconstruct_from_incremental(u, s, vt);
        let err = frobenius_diff(&expected, &recon);
        assert!(err < 1e-6, "reconstruction error {} (expected near zero)", err);
        let mut sorted = s.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(sorted.len(), 4);
        assert!((sorted[0] - 4.0).abs() < 1e-4, "s0 = {}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1e-4, "s1 = {}", sorted[1]);
        assert!((sorted[2] - 1.0).abs() < 1e-4, "s2 = {}", sorted[2]);
        assert!(
            (sorted[3] - 1.0).abs() < 1e-4,
            "repeated singular value not resolved: s3 = {}",
            sorted[3]
        );
    }

    #[test]
    fn test_incremental_svd_add_columns_reconstruction() {
        // Adding a column that introduces a new left-singular direction must
        // update U as well as V so the factorization still reconstructs A'.
        let a = CsrMatrix::new(3, 2, vec![0, 1, 2, 2], vec![0, 1], vec![3.0, 2.0]).unwrap();
        let config = IncrementalSVDConfig {
            max_rank: 8,
            tolerance: 1e-12,
            reorthogonalize: true,
        };
        let mut isvd = IncrementalSVD::new(config);
        isvd.initialize(&a, 2).unwrap();
        isvd.add_columns(&[vec![0.0, 0.0, 1.0]]).unwrap();
        assert_eq!(isvd.dimensions(), (3, 3));
        let expected = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (u, s, vt) = isvd.get_svd();
        let recon = reconstruct_from_incremental(u, s, vt);
        let err = frobenius_diff(&expected, &recon);
        assert!(err < 1e-6, "reconstruction error {} (expected near zero)", err);
    }
}
