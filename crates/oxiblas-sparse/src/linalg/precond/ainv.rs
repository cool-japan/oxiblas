//! AINV (Approximate Inverse) preconditioner.

use super::types::PreconditionerError;
use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Configuration for AINV (Approximate Inverse) preconditioner.
pub struct AINVConfig {
    /// Drop tolerance for small entries (default: 0.1).
    pub drop_tolerance: f64,
    /// Maximum number of non-zeros per column of Z and W (default: 20).
    pub max_nnz_per_col: usize,
    /// Use modified Gram-Schmidt process (default: true).
    pub modified_gs: bool,
}

impl Default for AINVConfig {
    fn default() -> Self {
        Self {
            drop_tolerance: 0.1,
            max_nnz_per_col: 20,
            modified_gs: true,
        }
    }
}

/// Approximate Inverse (AINV) preconditioner.
///
/// AINV computes a factored sparse approximate inverse M = Z * D^{-1} * W^T ≈ A^{-1}
/// using a stabilized incomplete biconjugation algorithm (Benzi-Tuma).
///
/// For symmetric positive definite matrices, Z = W, giving M = Z * D^{-1} * Z^T.
///
/// # Algorithm
///
/// The algorithm computes sparse approximations to the inverse factors by
/// performing an A-biorthogonalization of the columns of the identity matrix:
///
/// z_i^T * A * w_j = 0 for i ≠ j
///
/// This produces Z and W such that Z^T * A * W = D (diagonal).
///
/// # Advantages
///
/// - Good for ill-conditioned matrices
/// - No forward/backward substitution during application
/// - Parallelizable
///
/// # Example
///
/// ```
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::precond::{AINV, AINVConfig};
///
/// // Diagonally dominant tridiagonal matrix [[4,1,0],[1,4,1],[0,1,4]].
/// let matrix = CsrMatrix::new(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0],
/// )
/// .unwrap();
///
/// let config = AINVConfig::default();
/// let ainv = AINV::new(&matrix, config).unwrap();
/// let r = vec![1.0, 1.0, 1.0];
/// let mut z = vec![0.0; 3];
/// ainv.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct AINV<T: Scalar> {
    /// Z factor (lower triangular sparse, stored in CSC for efficiency)
    z_values: Vec<T>,
    z_row_indices: Vec<usize>,
    z_col_ptrs: Vec<usize>,
    /// W factor (lower triangular sparse, stored in CSC for efficiency)
    w_values: Vec<T>,
    w_row_indices: Vec<usize>,
    w_col_ptrs: Vec<usize>,
    /// Diagonal D^{-1}
    d_inv: Vec<T>,
    /// Matrix dimension
    n: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd + Real> AINV<T> {
    /// Create a new AINV preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to precondition (CSR format).
    /// * `config` - Configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or if a zero pivot is encountered.
    pub fn new(a: &CsrMatrix<T>, config: AINVConfig) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        let n = a.nrows();

        if n == 0 {
            return Ok(Self {
                z_values: Vec::new(),
                z_row_indices: Vec::new(),
                z_col_ptrs: vec![0],
                w_values: Vec::new(),
                w_row_indices: Vec::new(),
                w_col_ptrs: vec![0],
                d_inv: Vec::new(),
                n: 0,
            });
        }

        // Convert A to CSC for column access
        let a_csc = a.to_csc();

        // Initialize Z and W as identity matrices (unit lower triangular)
        // We'll build them column by column using sparse storage
        let mut z_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut w_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut d_inv = vec![T::zero(); n];

        // Working vectors for A*z and A^T*w
        let mut az = vec![T::zero(); n];
        let mut atw = vec![T::zero(); n];

        // The stabilized biconjugation algorithm
        // z_j and w_j start as e_j (unit vectors)
        // We orthogonalize against previous vectors

        for j in 0..n {
            // Initialize z_j = e_j and w_j = e_j
            let mut z_j = vec![T::zero(); n];
            let mut w_j = vec![T::zero(); n];
            z_j[j] = T::one();
            w_j[j] = T::one();

            if config.modified_gs {
                // Modified Gram-Schmidt biconjugation (Benzi-Tuma).
                //
                // A-biorthogonalize z_j against the previously computed z_k and
                // w_j against the previously computed w_k so that the resulting
                // factors satisfy W^T A Z = D (diagonal). Both A-inner-products
                // must be evaluated as *full* dot products over all n rows: a
                // partial sum over the sparse structure of a single factor is
                // NOT the A-inner-product and yields the wrong coefficient.
                for k in 0..j {
                    // az = A * z_k and atw = A^T * w_k (dense, over all n rows).
                    Self::sparse_mv_csc(&a_csc, &z_cols[k], &mut az, n);
                    Self::sparse_mv_at(a, &w_cols[k], &mut atw);

                    // A-inner-products as full dot products over all n rows:
                    //   w_k^T A z_j = (A^T w_k) . z_j = dot(atw, z_j)
                    //   w_j^T A z_k = w_j . (A z_k)   = dot(w_j, az)
                    let mut wkt_a_zj = T::zero();
                    let mut wjt_a_zk = T::zero();
                    for i in 0..n {
                        wkt_a_zj = wkt_a_zj + atw[i].clone() * z_j[i].clone();
                        wjt_a_zk = wjt_a_zk + w_j[i].clone() * az[i].clone();
                    }

                    // Skip orthogonalization against a numerically negligible pivot.
                    let tiny = T::from_f64(1e-14).unwrap_or(T::zero());
                    if Scalar::abs(d_inv[k].clone()) > tiny.clone() {
                        // d_k = 1 / d_inv[k]
                        let dk = T::one() / d_inv[k].clone();
                        if Scalar::abs(dk.clone()) > tiny {
                            // z_j <- z_j - (w_k^T A z_j / d_k) * z_k
                            let alpha = wkt_a_zj / dk.clone();
                            for (row, val) in &z_cols[k] {
                                z_j[*row] = z_j[*row].clone() - alpha.clone() * val.clone();
                            }
                            // w_j <- w_j - (w_j^T A z_k / d_k) * w_k
                            let beta = wjt_a_zk / dk;
                            for (row, val) in &w_cols[k] {
                                w_j[*row] = w_j[*row].clone() - beta.clone() * val.clone();
                            }
                        }
                    }
                }
            }

            // Compute d_j = w_j^T * A * z_j
            // First compute A * z_j
            Self::compute_mv(a, &z_j, &mut az);

            let mut d_j = T::zero();
            for i in 0..n {
                d_j = d_j + w_j[i].clone() * az[i].clone();
            }

            // Store d_j^{-1}
            let drop_tol = T::from_f64(config.drop_tolerance).unwrap_or(T::zero());
            if Scalar::abs(d_j.clone()) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                // Use regularization for zero pivot
                d_inv[j] = T::from_f64(1e10).unwrap_or(T::one());
            } else {
                d_inv[j] = T::one() / d_j;
            }

            // Apply dropping to z_j and w_j
            // Keep entries larger than drop_tolerance * ||z_j||
            let z_norm_sq: T = z_j
                .iter()
                .map(|v| v.clone() * v.clone())
                .fold(T::zero(), |a, b| a + b);
            let z_norm = Real::sqrt(z_norm_sq);

            let w_norm_sq: T = w_j
                .iter()
                .map(|v| v.clone() * v.clone())
                .fold(T::zero(), |a, b| a + b);
            let w_norm = Real::sqrt(w_norm_sq);

            let z_threshold = drop_tol.clone() * z_norm;
            let w_threshold = drop_tol.clone() * w_norm;

            // Store sparse z_j. The AINV Z-factor is unit upper triangular, so
            // z_j has support in rows i <= j. Keep the unit diagonal plus the
            // off-diagonal entries (i < j) above the drop threshold.
            let mut z_entries: Vec<(usize, T)> = Vec::new();
            for i in 0..j {
                if Scalar::abs(z_j[i].clone()) >= z_threshold {
                    z_entries.push((i, z_j[i].clone()));
                }
            }

            // Cap the off-diagonal fill, keeping the entries largest in magnitude,
            // while always reserving room for the diagonal entry.
            let z_max_off = config.max_nnz_per_col.saturating_sub(1);
            if z_entries.len() > z_max_off {
                z_entries.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                z_entries.truncate(z_max_off);
            }

            // Always retain the (unit) diagonal entry, then restore row order.
            z_entries.push((j, z_j[j].clone()));
            z_entries.sort_by_key(|(i, _)| *i);
            z_cols[j] = z_entries;

            // Store sparse w_j. The AINV W-factor is likewise unit upper
            // triangular (support in rows i <= j).
            let mut w_entries: Vec<(usize, T)> = Vec::new();
            for i in 0..j {
                if Scalar::abs(w_j[i].clone()) >= w_threshold {
                    w_entries.push((i, w_j[i].clone()));
                }
            }

            let w_max_off = config.max_nnz_per_col.saturating_sub(1);
            if w_entries.len() > w_max_off {
                w_entries.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                w_entries.truncate(w_max_off);
            }

            w_entries.push((j, w_j[j].clone()));
            w_entries.sort_by_key(|(i, _)| *i);
            w_cols[j] = w_entries;
        }

        // Convert column storage to CSC format
        let mut z_values = Vec::new();
        let mut z_row_indices = Vec::new();
        let mut z_col_ptrs = vec![0];

        for col in &z_cols {
            for (row, val) in col {
                z_row_indices.push(*row);
                z_values.push(val.clone());
            }
            z_col_ptrs.push(z_values.len());
        }

        let mut w_values = Vec::new();
        let mut w_row_indices = Vec::new();
        let mut w_col_ptrs = vec![0];

        for col in &w_cols {
            for (row, val) in col {
                w_row_indices.push(*row);
                w_values.push(val.clone());
            }
            w_col_ptrs.push(w_values.len());
        }

        Ok(Self {
            z_values,
            z_row_indices,
            z_col_ptrs,
            w_values,
            w_row_indices,
            w_col_ptrs,
            d_inv,
            n,
        })
    }

    /// Compute y = A * x for dense vectors
    fn compute_mv(a: &CsrMatrix<T>, x: &[T], y: &mut [T]) {
        for val in y.iter_mut() {
            *val = T::zero();
        }
        for i in 0..a.nrows() {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            let mut sum = T::zero();
            for idx in start..end {
                let j = a.col_indices()[idx];
                sum = sum + a.values()[idx].clone() * x[j].clone();
            }
            y[i] = sum;
        }
    }

    /// Compute y = A * z where z is stored as sparse column entries
    fn sparse_mv_csc(a_csc: &CscMatrix<T>, z_col: &[(usize, T)], y: &mut [T], _n: usize) {
        for val in y.iter_mut() {
            *val = T::zero();
        }
        for (row, z_val) in z_col {
            // Column 'row' of A multiplied by z_val
            let start = a_csc.col_ptrs()[*row];
            let end = a_csc.col_ptrs()[*row + 1];
            for idx in start..end {
                let i = a_csc.row_indices()[idx];
                y[i] = y[i].clone() + a_csc.values()[idx].clone() * z_val.clone();
            }
        }
    }

    /// Compute y = A^T * w where w is stored as sparse column entries.
    ///
    /// Using the CSR (row) storage of A, scattering row `i` of A weighted by
    /// `w[i]` accumulates `y[col] += A[i, col] * w[i]`, i.e. `y = A^T w`.
    /// (Scattering *columns* of A, as a CSC traversal would, computes `A w`
    /// instead, which is only correct for symmetric A.)
    fn sparse_mv_at(a: &CsrMatrix<T>, w_col: &[(usize, T)], y: &mut [T]) {
        for val in y.iter_mut() {
            *val = T::zero();
        }
        for (i, w_val) in w_col {
            let start = a.row_ptrs()[*i];
            let end = a.row_ptrs()[*i + 1];
            for idx in start..end {
                let col = a.col_indices()[idx];
                y[col] = y[col].clone() + a.values()[idx].clone() * w_val.clone();
            }
        }
    }

    /// Apply the preconditioner: z = M * r = Z * D^{-1} * W^T * r.
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        assert_eq!(r.len(), self.n, "r length must match matrix size");
        assert_eq!(z.len(), self.n, "z length must match matrix size");

        if self.n == 0 {
            return;
        }

        // Step 1: y = W^T * r
        // W is stored in CSC (column format), so W^T * r = sum over columns
        let mut y = vec![T::zero(); self.n];
        for j in 0..self.n {
            let start = self.w_col_ptrs[j];
            let end = self.w_col_ptrs[j + 1];
            let mut sum = T::zero();
            for idx in start..end {
                let i = self.w_row_indices[idx];
                sum = sum + self.w_values[idx].clone() * r[i].clone();
            }
            y[j] = sum;
        }

        // Step 2: y = D^{-1} * y
        for j in 0..self.n {
            y[j] = y[j].clone() * self.d_inv[j].clone();
        }

        // Step 3: z = Z * y
        for val in z.iter_mut() {
            *val = T::zero();
        }
        for j in 0..self.n {
            let start = self.z_col_ptrs[j];
            let end = self.z_col_ptrs[j + 1];
            for idx in start..end {
                let i = self.z_row_indices[idx];
                z[i] = z[i].clone() + self.z_values[idx].clone() * y[j].clone();
            }
        }
    }

    /// Returns the total number of non-zeros in Z and W.
    pub fn nnz(&self) -> usize {
        self.z_values.len() + self.w_values.len()
    }

    /// Returns the number of non-zeros in Z.
    pub fn z_nnz(&self) -> usize {
        self.z_values.len()
    }

    /// Returns the number of non-zeros in W.
    pub fn w_nnz(&self) -> usize {
        self.w_values.len()
    }

    /// Returns the dimension of the preconditioner.
    pub fn dim(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::iterative::{cg, pcg};

    /// Densify a CSC-stored factor (columns of (row, value)) into an n x n matrix.
    fn densify(vals: &[f64], rows: &[usize], col_ptrs: &[usize], n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0f64; n]; n];
        for c in 0..n {
            for idx in col_ptrs[c]..col_ptrs[c + 1] {
                m[rows[idx]][c] = vals[idx];
            }
        }
        m
    }

    /// Densify a CSR matrix into an n x n matrix.
    fn dense_from_csr(a: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
        let n = a.nrows();
        let mut m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for idx in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
                m[i][a.col_indices()[idx]] = a.values()[idx];
            }
        }
        m
    }

    fn matmul(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
        let mut c = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for k in 0..n {
                let aik = a[i][k];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i][j] += aik * b[k][j];
                }
            }
        }
        c
    }

    /// Build the 1D Laplacian tridiagonal SPD matrix diag = `diag`, off = -1.
    fn laplacian(n: usize, diag: f64) -> CsrMatrix<f64> {
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];
        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(diag);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).expect("valid CSR")
    }

    #[test]
    fn test_ainv_biorthogonality_spd() {
        // 4x4 SPD tridiagonal matrix. With no dropping the factored inverse is
        // exact, so W^T A Z must be (numerically) diagonal and A * M ~= I.
        let n = 4;
        let a = laplacian(n, 4.0);

        let config = AINVConfig {
            drop_tolerance: 0.0,
            max_nnz_per_col: n,
            modified_gs: true,
        };
        let ainv = AINV::new(&a, config).expect("AINV builds");

        let z = densify(&ainv.z_values, &ainv.z_row_indices, &ainv.z_col_ptrs, n);
        let w = densify(&ainv.w_values, &ainv.w_row_indices, &ainv.w_col_ptrs, n);
        let a_dense = dense_from_csr(&a);

        // W^T A Z must be diagonal.
        let az = matmul(&a_dense, &z, n);
        // wtaz[i][j] = sum_p W[p][i] * (A Z)[p][j]
        let mut wtaz = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..n {
                    s += w[p][i] * az[p][j];
                }
                wtaz[i][j] = s;
            }
        }
        for i in 0..n {
            assert!(
                wtaz[i][i].abs() > 1e-6,
                "diagonal of W^T A Z must be nonzero, got {} at {}",
                wtaz[i][i],
                i
            );
            for j in 0..n {
                if i != j {
                    assert!(
                        wtaz[i][j].abs() < 1e-8,
                        "W^T A Z must be diagonal; off-diagonal ({},{}) = {}",
                        i,
                        j,
                        wtaz[i][j]
                    );
                }
            }
        }

        // A * M ~= I, where column c of M is M applied to e_c.
        let mut m_mat = vec![vec![0.0f64; n]; n];
        for c in 0..n {
            let mut e = vec![0.0f64; n];
            e[c] = 1.0;
            let mut col = vec![0.0f64; n];
            ainv.apply(&e, &mut col);
            for (r, val) in col.iter().enumerate() {
                m_mat[r][c] = *val;
            }
        }
        let am = matmul(&a_dense, &m_mat, n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (am[i][j] - expected).abs() < 1e-8,
                    "A*M must be identity; ({},{}) = {}",
                    i,
                    j,
                    am[i][j]
                );
            }
        }
    }

    #[test]
    fn test_ainv_pcg_reduces_iterations() {
        // 1D Laplacian (diag 2, off -1): reasonably ill-conditioned SPD system.
        let n = 30;
        let a = laplacian(n, 2.0);

        let b: Vec<f64> = vec![1.0; n];
        let x0: Vec<f64> = vec![0.0; n];
        let tol = 1e-8_f64;
        let max_iter = 500;

        // Unpreconditioned CG baseline.
        let cg_res = cg(&a, &b, &x0, tol, max_iter).expect("cg runs");
        assert!(cg_res.converged, "baseline CG should converge");

        // AINV-preconditioned CG. With no dropping the factored inverse is
        // exact, so PCG converges in a single step.
        let config = AINVConfig {
            drop_tolerance: 0.0,
            max_nnz_per_col: n,
            modified_gs: true,
        };
        let ainv = AINV::new(&a, config).expect("AINV builds");
        let precond = |r: &[f64]| {
            let mut z = vec![0.0f64; n];
            ainv.apply(r, &mut z);
            z
        };
        let pcg_res = pcg(&a, &b, &x0, precond, tol, max_iter).expect("pcg runs");

        assert!(pcg_res.converged, "AINV-preconditioned CG should converge");
        assert!(
            pcg_res.iterations < cg_res.iterations,
            "AINV should reduce PCG iterations: pcg={} vs cg={}",
            pcg_res.iterations,
            cg_res.iterations
        );
        assert!(
            pcg_res.iterations <= 2,
            "exact AINV should make PCG converge almost immediately, got {}",
            pcg_res.iterations
        );
    }
}
