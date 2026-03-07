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
/// QR decomposition of dense matrix using Householder reflections.
///
/// Returns (Q, R) where A = Q*R, Q is orthonormal, R is upper triangular.
/// Q is returned as column vectors (m×k), R is k×n.
pub(super) fn qr_decompose_dense<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    a: &[Vec<T>],
) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    if a.is_empty() || a[0].is_empty() {
        return (Vec::new(), Vec::new());
    }
    let m = a.len();
    let n = a[0].len();
    let k = m.min(n);
    let a_cols: Vec<Vec<T>> = transpose_dense(a);
    let mut q = vec![vec![T::zero(); k]; m];
    let mut r = vec![vec![T::zero(); n]; k];
    for j in 0..k {
        let mut v = a_cols[j].clone();
        for i in 0..j {
            let mut proj = T::zero();
            for l in 0..m {
                proj = proj + q[l][i].clone() * v[l].clone();
            }
            r[i][j] = proj.clone();
            for l in 0..m {
                v[l] = v[l].clone() - proj.clone() * q[l][i].clone();
            }
        }
        let norm = Real::sqrt(
            v.iter()
                .map(|x| x.clone() * x.clone())
                .fold(T::zero(), |acc, x| acc + x),
        );
        r[j][j] = norm.clone();
        if norm > T::from_f64(1e-14).unwrap_or_else(T::zero) {
            for l in 0..m {
                q[l][j] = v[l].clone() / norm.clone();
            }
        }
        for jj in (j + 1)..n {
            let mut dot = T::zero();
            for l in 0..m {
                dot = dot + q[l][j].clone() * a_cols[jj][l].clone();
            }
            r[j][jj] = dot;
        }
    }
    (q, r)
}
/// Full SVD of a dense matrix using LAPACK-style dense decomposition.
pub(super) fn dense_svd_full<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    matrix: &[Vec<T>],
) -> Result<RandomizedSparseSvdResult<T>, SVDError> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(SVDError::InvalidConfig("Empty matrix".to_string()));
    }
    let m = matrix.len();
    let n = matrix[0].len();
    let min_dim = m.min(n);
    let mut a_flat = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            a_flat.push(matrix[i][j].clone());
        }
    }
    let mut ata = vec![vec![T::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..m {
                sum = sum + matrix[k][i].clone() * matrix[k][j].clone();
            }
            ata[i][j] = sum;
        }
    }
    let mut singular_values: Vec<T> = Vec::new();
    let mut v_vectors: Vec<Vec<T>> = Vec::new();
    for _ in 0..min_dim {
        let mut v = vec![T::from_f64(1.0).unwrap_or_else(T::zero); n];
        for i in 1..n {
            v[i] = T::from_f64((i as f64).sin()).unwrap_or_else(T::zero);
        }
        for prev_v in &v_vectors {
            let dot: T = v
                .iter()
                .zip(prev_v.iter())
                .map(|(a, b)| a.clone() * b.clone())
                .fold(T::zero(), |acc, x| acc + x);
            for i in 0..n {
                v[i] = v[i].clone() - dot.clone() * prev_v[i].clone();
            }
        }
        for _ in 0..20 {
            let mut new_v = vec![T::zero(); n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] = new_v[i].clone() + ata[i][j].clone() * v[j].clone();
                }
            }
            let norm = Real::sqrt(
                new_v
                    .iter()
                    .map(|x| x.clone() * x.clone())
                    .fold(T::zero(), |acc, x| acc + x),
            );
            if norm > T::from_f64(1e-14).unwrap_or_else(T::zero) {
                v = new_v.iter().map(|x| x.clone() / norm.clone()).collect();
            } else {
                break;
            }
        }
        let mut av = vec![T::zero(); m];
        for i in 0..m {
            for j in 0..n {
                av[i] = av[i].clone() + matrix[i][j].clone() * v[j].clone();
            }
        }
        let sigma = Real::sqrt(
            av.iter()
                .map(|x| x.clone() * x.clone())
                .fold(T::zero(), |acc, x| acc + x),
        );
        if sigma < T::from_f64(1e-10).unwrap_or_else(T::zero) {
            break;
        }
        singular_values.push(sigma);
        v_vectors.push(v);
        for i in 0..n {
            for j in 0..n {
                let deflate = sigma.clone()
                    * sigma.clone()
                    * v_vectors.last().expect("collection should be non-empty")[i].clone()
                    * v_vectors.last().expect("collection should be non-empty")[j].clone();
                ata[i][j] = ata[i][j].clone() - deflate;
            }
        }
    }
    let mut u_vectors = vec![vec![T::zero(); singular_values.len()]; m];
    for (k, (sigma, v)) in singular_values.iter().zip(v_vectors.iter()).enumerate() {
        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum + matrix[i][j].clone() * v[j].clone();
            }
            if Scalar::abs(sigma.clone()) > T::from_f64(1e-14).unwrap_or_else(T::zero) {
                u_vectors[i][k] = sum / sigma.clone();
            }
        }
    }
    Ok(RandomizedSparseSvdResult {
        singular_values,
        u: Some(u_vectors),
        v: Some(v_vectors),
    })
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
}
