//! SPAI (Sparse Approximate Inverse) preconditioner.

use super::types::PreconditionerError;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Configuration for SPAI (Sparse Approximate Inverse) preconditioner.
pub struct SPAIConfig {
    /// Target residual tolerance (default: 0.4).
    pub tolerance: f64,
    /// Maximum number of non-zeros per column of M (default: 10).
    pub max_nnz_per_col: usize,
    /// Use A's sparsity pattern for M (default: true).
    pub use_a_pattern: bool,
    /// Maximum improvement iterations (default: 5).
    pub max_iterations: usize,
}

impl Default for SPAIConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.4,
            max_nnz_per_col: 10,
            use_a_pattern: true,
            max_iterations: 5,
        }
    }
}

/// Sparse Approximate Inverse (SPAI) preconditioner.
///
/// SPAI computes a sparse approximation M ≈ A^{-1} by solving independent
/// least-squares problems for each column:
///
/// min_j ||A * m_j - e_j||_2
///
/// where m_j is the j-th column of M and e_j is the j-th unit vector.
///
/// # Advantages
///
/// - Highly parallelizable (columns computed independently)
/// - Good for ill-conditioned matrices
/// - No need for triangular solves during application
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::{SPAI, SPAIConfig};
///
/// let config = SPAIConfig::default();
/// let spai = SPAI::new(&matrix, config)?;
/// let mut z = vec![0.0; n];
/// spai.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct SPAI<T: Scalar> {
    /// The approximate inverse matrix M (stored in CSR).
    m_values: Vec<T>,
    m_col_indices: Vec<usize>,
    m_row_ptrs: Vec<usize>,
    n: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> SPAI<T> {
    /// Create a new SPAI preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to precondition (CSR format).
    /// * `config` - Configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square.
    pub fn new(a: &CsrMatrix<T>, config: SPAIConfig) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        let n = a.nrows();

        if n == 0 {
            return Ok(Self {
                m_values: Vec::new(),
                m_col_indices: Vec::new(),
                m_row_ptrs: vec![0],
                n: 0,
            });
        }

        // Build initial sparsity pattern for M
        let m_pattern: Vec<Vec<usize>> = if config.use_a_pattern {
            Self::get_a_pattern(a)
        } else {
            // Use diagonal pattern as initial guess
            (0..n).map(|i| vec![i]).collect()
        };

        // Compute M column by column
        let mut m_columns: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);

        for j in 0..n {
            let col_result = Self::compute_column(a, j, &m_pattern[j], &config);
            m_columns.push(col_result);
        }

        // Convert column-wise storage to CSR (which is row-wise)
        // M in CSR means we need rows of M
        // Each m_columns[j] gives column j of M, i.e., M[:, j]
        // For CSR, we need row i of M, i.e., M[i, :]

        // First, collect entries by row
        let mut row_entries: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        for (j, col) in m_columns.iter().enumerate() {
            for &(i, ref val) in col {
                row_entries[i].push((j, val.clone()));
            }
        }

        // Sort each row by column index
        for row in &mut row_entries {
            row.sort_by_key(|(j, _)| *j);
        }

        // Build CSR arrays
        let mut m_values = Vec::new();
        let mut m_col_indices = Vec::new();
        let mut m_row_ptrs = vec![0];

        for row in row_entries {
            for (j, val) in row {
                m_col_indices.push(j);
                m_values.push(val);
            }
            m_row_ptrs.push(m_values.len());
        }

        Ok(Self {
            m_values,
            m_col_indices,
            m_row_ptrs,
            n,
        })
    }

    /// Get sparsity pattern from matrix A.
    fn get_a_pattern(a: &CsrMatrix<T>) -> Vec<Vec<usize>> {
        let n = a.nrows();
        let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); n];

        // For column j, pattern includes all rows i where A[i,j] != 0
        // We need to transpose A's pattern
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            for idx in start..end {
                let j = a.col_indices()[idx];
                pattern[j].push(i);
            }
        }

        // Sort and remove duplicates
        for pat in &mut pattern {
            pat.sort_unstable();
            pat.dedup();
        }

        pattern
    }

    /// Compute column j of M.
    fn compute_column(
        a: &CsrMatrix<T>,
        j: usize,
        pattern: &[usize],
        _config: &SPAIConfig,
    ) -> Vec<(usize, T)> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let n = a.nrows();

        // Find the set of rows of A that affect the residual
        // When computing A * m_j, for each non-zero m[k,j], we get contributions
        // to rows where A[i,k] != 0

        // Determine which rows of A we need (I set)
        let mut i_set: Vec<usize> = Vec::new();
        for &k in pattern {
            let start = a.row_ptrs()[k];
            let end = a.row_ptrs()[k + 1];

            for idx in start..end {
                i_set.push(a.col_indices()[idx]);
            }
        }
        i_set.sort_unstable();
        i_set.dedup();

        // Also include row j for the e_j component
        if !i_set.contains(&j) {
            i_set.push(j);
            i_set.sort_unstable();
        }

        let n_i = i_set.len();
        let n_k = pattern.len();

        if n_i == 0 || n_k == 0 {
            return vec![(j, T::one())]; // Fallback to diagonal
        }

        // Build the small least-squares system
        // A_hat[i, k] = A[i_set[i], pattern[k]]
        // We want to minimize ||A_hat * m - e_j_hat||_2

        // Map indices
        let mut i_to_local: Vec<usize> = vec![usize::MAX; n];
        for (local, &global) in i_set.iter().enumerate() {
            i_to_local[global] = local;
        }

        let mut k_to_local: Vec<usize> = vec![usize::MAX; n];
        for (local, &global) in pattern.iter().enumerate() {
            k_to_local[global] = local;
        }

        // Build A_hat as dense matrix (n_i x n_k)
        let mut a_hat = vec![T::zero(); n_i * n_k];

        for (local_k, &k) in pattern.iter().enumerate() {
            let start = a.row_ptrs()[k];
            let end = a.row_ptrs()[k + 1];

            for idx in start..end {
                let i_global = a.col_indices()[idx];
                let local_i = i_to_local[i_global];
                if local_i != usize::MAX {
                    // A_hat is stored column-major for least squares
                    a_hat[local_i + local_k * n_i] = a.values()[idx].clone();
                }
            }
        }

        // Build e_j_hat (the portion of e_j corresponding to i_set)
        let mut e_hat = vec![T::zero(); n_i];
        let j_local = i_to_local[j];
        if j_local != usize::MAX {
            e_hat[j_local] = T::one();
        }

        // Solve least squares using normal equations: A^T A m = A^T e
        // For small systems, this is efficient enough

        // Compute A^T A (n_k x n_k)
        let mut ata = vec![T::zero(); n_k * n_k];
        for k1 in 0..n_k {
            for k2 in 0..n_k {
                let mut sum = T::zero();
                for i in 0..n_i {
                    sum = sum + a_hat[i + k1 * n_i].clone() * a_hat[i + k2 * n_i].clone();
                }
                ata[k1 + k2 * n_k] = sum;
            }
        }

        // Compute A^T e (n_k x 1)
        let mut ate = vec![T::zero(); n_k];
        for k in 0..n_k {
            let mut sum = T::zero();
            for i in 0..n_i {
                sum = sum + a_hat[i + k * n_i].clone() * e_hat[i].clone();
            }
            ate[k] = sum;
        }

        // Solve A^T A m = A^T e using Cholesky or LU
        // For robustness, add small regularization
        let reg = T::from_f64(1e-12).unwrap_or(T::zero());
        for k in 0..n_k {
            ata[k + k * n_k] = ata[k + k * n_k].clone() + reg.clone();
        }

        // Simple Gaussian elimination for small system
        let m_local = Self::solve_small_system(&ata, &ate, n_k);

        // Build result
        let mut result = Vec::new();
        for (local_k, &k) in pattern.iter().enumerate() {
            let val = m_local[local_k].clone();
            if Scalar::abs(val.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                result.push((k, val));
            }
        }

        // Ensure at least diagonal entry
        if result.is_empty() {
            result.push((j, T::one()));
        }

        result
    }

    /// Solve a small dense linear system using Gaussian elimination.
    fn solve_small_system(a: &[T], b: &[T], n: usize) -> Vec<T> {
        if n == 0 {
            return Vec::new();
        }

        // Copy to working arrays
        let mut aug = vec![T::zero(); n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a[i + j * n].clone();
            }
            aug[i * (n + 1) + n] = b[i].clone();
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = Scalar::abs(aug[k * (n + 1) + k].clone());
            let mut max_row = k;

            for i in (k + 1)..n {
                let val = Scalar::abs(aug[i * (n + 1) + k].clone());
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                for j in 0..(n + 1) {
                    let tmp = aug[k * (n + 1) + j].clone();
                    aug[k * (n + 1) + j] = aug[max_row * (n + 1) + j].clone();
                    aug[max_row * (n + 1) + j] = tmp;
                }
            }

            // Check for zero pivot
            let pivot = aug[k * (n + 1) + k].clone();
            if Scalar::abs(pivot.clone()) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                continue;
            }

            // Eliminate below
            for i in (k + 1)..n {
                let factor = aug[i * (n + 1) + k].clone() / pivot.clone();
                for j in k..(n + 1) {
                    let temp = aug[k * (n + 1) + j].clone() * factor.clone();
                    aug[i * (n + 1) + j] = aug[i * (n + 1) + j].clone() - temp;
                }
            }
        }

        // Back substitution
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n].clone();
            for j in (i + 1)..n {
                sum = sum - aug[i * (n + 1) + j].clone() * x[j].clone();
            }

            let diag = aug[i * (n + 1) + i].clone();
            if Scalar::abs(diag.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                x[i] = sum / diag;
            }
        }

        x
    }

    /// Apply the preconditioner: z = M * r.
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        assert_eq!(r.len(), self.n, "r length must match matrix size");
        assert_eq!(z.len(), self.n, "z length must match matrix size");

        // z = M * r (sparse matrix-vector product)
        for i in 0..self.n {
            let start = self.m_row_ptrs[i];
            let end = self.m_row_ptrs[i + 1];

            let mut sum = T::zero();
            for idx in start..end {
                let j = self.m_col_indices[idx];
                sum = sum + self.m_values[idx].clone() * r[j].clone();
            }
            z[i] = sum;
        }
    }

    /// Returns the number of non-zeros in M.
    pub fn nnz(&self) -> usize {
        self.m_values.len()
    }

    /// Returns the dimension of M.
    pub fn dim(&self) -> usize {
        self.n
    }
}
