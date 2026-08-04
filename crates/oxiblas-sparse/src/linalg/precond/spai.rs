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
/// ```
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::precond::{SPAI, SPAIConfig};
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
/// let config = SPAIConfig::default();
/// let spai = SPAI::new(&matrix, config).unwrap();
/// let r = vec![1.0, 1.0, 1.0];
/// let mut z = vec![0.0; 3];
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

/// Result of solving the reduced least-squares problem for a single column of M.
///
/// For a given column sparsity pattern `J` (`j_set`), the reduced problem is
/// `min ||A(I, J) m_hat - e_j(I)||_2`, where `I` is the set of rows touched by
/// the columns in `J`. The struct bundles the least-squares solution together
/// with the row set and the resulting residual so that the pattern-augmentation
/// loop can both accept the solution and decide how to grow `J`.
struct ReducedSolve<T> {
    /// Least-squares coefficients, aligned with the (sorted) `j_set`.
    m_hat: Vec<T>,
    /// Rows of A that participate in this column's reduced problem (sorted).
    i_set: Vec<usize>,
    /// Residual `A(I, J) m_hat - e_j(I)`, aligned with `i_set`.
    residual: Vec<T>,
    /// Squared 2-norm of `residual` (avoids a `sqrt` on the generic scalar).
    resnorm_sq: T,
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

        // Precompute the squared 2-norm of each column of A. This is needed by the
        // Grote-Huckle candidate-selection step of the pattern-augmentation loop,
        // where the residual reduction of a candidate index c is
        //   rho_c^2 = ||r||^2 - (r^T A_{.,c})^2 / ||A_{.,c}||^2 .
        // Computing it once here keeps the per-column work cheap.
        let mut col_norm_sq = vec![T::zero(); n];
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for idx in start..end {
                let c = a.col_indices()[idx];
                let v = a.values()[idx].clone();
                col_norm_sq[c] = col_norm_sq[c].clone() + v.clone() * v;
            }
        }

        // Compute M column by column
        let mut m_columns: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);

        for j in 0..n {
            let col_result = Self::compute_column(a, j, &m_pattern[j], &config, &col_norm_sq);
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

    /// Compute column `orig_j` of the approximate inverse M.
    ///
    /// This implements the Grote-Huckle SPAI algorithm with adaptive
    /// sparsity: starting from the initial pattern, the column's least-squares
    /// approximate-inverse problem is solved, and the pattern is greedily
    /// augmented one index at a time until the residual falls below
    /// `config.tolerance`, or the pattern reaches `config.max_nnz_per_col`
    /// entries, or `config.max_iterations` augmentation rounds have been
    /// performed. All three tuning parameters therefore genuinely influence
    /// both the sparsity and the accuracy of the resulting column.
    fn compute_column(
        a: &CsrMatrix<T>,
        orig_j: usize,
        pattern: &[usize],
        config: &SPAIConfig,
        col_norm_sq: &[T],
    ) -> Vec<(usize, T)> {
        // Current column sparsity pattern J (indices where m_j may be non-zero),
        // kept sorted and duplicate-free for binary-search membership tests.
        let mut j_set: Vec<usize> = pattern.to_vec();
        j_set.sort_unstable();
        j_set.dedup();
        if j_set.is_empty() {
            // Fall back to the minimal sparse pattern (the diagonal position).
            j_set.push(orig_j);
        }

        // Stopping thresholds derived from the caller-supplied configuration.
        // Compare against the squared residual norm to avoid a generic sqrt.
        let tol = T::from_f64(config.tolerance).unwrap_or(T::zero());
        let tol_sq = tol.clone() * tol;
        // At least one non-zero must be permitted per column.
        let max_nnz = config.max_nnz_per_col.max(1);
        let max_iter = config.max_iterations;
        let drop_tol = T::from_f64(1e-14).unwrap_or(T::zero());

        // Solve the least-squares problem on the initial pattern.
        let mut solved = Self::solve_reduced(a, orig_j, &j_set);
        let mut augment_count: usize = 0;

        loop {
            // Stopping criterion 1: residual tolerance reached.
            if solved.resnorm_sq <= tol_sq {
                break;
            }
            // Stopping criterion 2: per-column sparsity budget reached.
            if j_set.len() >= max_nnz {
                break;
            }
            // Stopping criterion 3: augmentation-iteration budget reached.
            if augment_count >= max_iter {
                break;
            }

            // Pattern augmentation: pick the single most profitable new index.
            let candidate = Self::best_candidate(
                a,
                &j_set,
                &solved.i_set,
                &solved.residual,
                col_norm_sq,
                &drop_tol,
            );

            match candidate {
                Some(new_idx) => match j_set.binary_search(&new_idx) {
                    // Should not already be present, but guard against looping.
                    Ok(_) => break,
                    Err(pos) => {
                        j_set.insert(pos, new_idx);
                        augment_count += 1;
                        solved = Self::solve_reduced(a, orig_j, &j_set);
                    }
                },
                // No remaining index can reduce the residual further.
                None => break,
            }
        }

        // Assemble the sparse column from the final solution, dropping numerical
        // zeros so that the reported sparsity reflects the true support.
        let mut result = Vec::new();
        for (local_k, &k) in j_set.iter().enumerate() {
            let val = solved.m_hat[local_k].clone();
            if Scalar::abs(val.clone()) > drop_tol {
                result.push((k, val));
            }
        }

        // Ensure at least the diagonal entry so M is never structurally empty.
        if result.is_empty() {
            result.push((orig_j, T::one()));
        }

        result
    }

    /// Solve the reduced least-squares problem for column `orig_j` restricted to
    /// the column pattern `j_set`.
    ///
    /// Builds `A(I, J)` where `I` is the set of rows touched by the columns in
    /// `J`, solves `min ||A(I, J) m_hat - e_j(I)||_2` via the (regularized)
    /// normal equations, and returns the solution together with the residual so
    /// the caller can evaluate the stopping criterion and grow the pattern.
    fn solve_reduced(a: &CsrMatrix<T>, orig_j: usize, j_set: &[usize]) -> ReducedSolve<T> {
        let n = a.nrows();
        let n_k = j_set.len();

        // I set: rows i with A[i, k] != 0 for some k in J. Row orig_j is always
        // included so the e_j component contributes to the residual even when the
        // current pattern does not yet touch it.
        let mut i_set: Vec<usize> = Vec::new();
        for &k in j_set {
            let start = a.row_ptrs()[k];
            let end = a.row_ptrs()[k + 1];
            for idx in start..end {
                i_set.push(a.col_indices()[idx]);
            }
        }
        i_set.push(orig_j);
        i_set.sort_unstable();
        i_set.dedup();

        let n_i = i_set.len();

        // Global-row -> local-row map for the reduced system.
        let mut i_to_local: Vec<usize> = vec![usize::MAX; n];
        for (local, &global) in i_set.iter().enumerate() {
            i_to_local[global] = local;
        }

        // Build A_hat (n_i x n_k), stored column-major for the least-squares math.
        let mut a_hat = vec![T::zero(); n_i * n_k];
        for (local_k, &k) in j_set.iter().enumerate() {
            let start = a.row_ptrs()[k];
            let end = a.row_ptrs()[k + 1];
            for idx in start..end {
                let i_global = a.col_indices()[idx];
                let local_i = i_to_local[i_global];
                if local_i != usize::MAX {
                    a_hat[local_i + local_k * n_i] = a.values()[idx].clone();
                }
            }
        }

        // Reduced right-hand side e_j restricted to I.
        let mut e_hat = vec![T::zero(); n_i];
        let j_local = i_to_local[orig_j];
        if j_local != usize::MAX {
            e_hat[j_local] = T::one();
        }

        // Normal equations A^T A m = A^T e (A^T A is n_k x n_k, column-major).
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

        let mut ate = vec![T::zero(); n_k];
        for k in 0..n_k {
            let mut sum = T::zero();
            for i in 0..n_i {
                sum = sum + a_hat[i + k * n_i].clone() * e_hat[i].clone();
            }
            ate[k] = sum;
        }

        // Small Tikhonov regularization for numerical robustness of the solve.
        let reg = T::from_f64(1e-12).unwrap_or(T::zero());
        for k in 0..n_k {
            ata[k + k * n_k] = ata[k + k * n_k].clone() + reg.clone();
        }

        let m_hat = Self::solve_small_system(&ata, &ate, n_k);

        // Residual r = A(I, J) m_hat - e_j(I). Because m_hat is supported on J,
        // the full residual A m_j - e_j is non-zero only on rows in I, so this
        // restricted residual is exact.
        let mut residual = vec![T::zero(); n_i];
        let mut resnorm_sq = T::zero();
        for i in 0..n_i {
            let mut s = T::zero();
            for local_k in 0..n_k {
                s = s + a_hat[i + local_k * n_i].clone() * m_hat[local_k].clone();
            }
            let r_i = s - e_hat[i].clone();
            resnorm_sq = resnorm_sq + r_i.clone() * r_i.clone();
            residual[i] = r_i;
        }

        ReducedSolve {
            m_hat,
            i_set,
            residual,
            resnorm_sq,
        }
    }

    /// Select the most profitable index to add to the column pattern.
    ///
    /// Implements the Grote-Huckle candidate ranking: only indices that appear
    /// in a residual-non-zero row and are not already in `j_set` are considered,
    /// and each is scored by its one-dimensional residual reduction
    /// `(r^T A_{.,c})^2 / ||A_{.,c}||^2`. The index with the largest reduction is
    /// returned (ties broken by smallest index for determinism). Returns `None`
    /// when no candidate can further reduce the residual.
    fn best_candidate(
        a: &CsrMatrix<T>,
        j_set: &[usize],
        i_set: &[usize],
        residual: &[T],
        col_norm_sq: &[T],
        drop_tol: &T,
    ) -> Option<usize> {
        use std::collections::HashMap;

        // Accumulate r^T A_{.,c} for each candidate column c. The residual is
        // zero outside the rows in i_set, so summing over residual-non-zero rows
        // gives the exact inner product.
        let mut dots: HashMap<usize, T> = HashMap::new();
        for (local_i, &row) in i_set.iter().enumerate() {
            let r_i = residual[local_i].clone();
            if Scalar::abs(r_i.clone()) <= *drop_tol {
                continue;
            }
            let start = a.row_ptrs()[row];
            let end = a.row_ptrs()[row + 1];
            for idx in start..end {
                let c = a.col_indices()[idx];
                if j_set.binary_search(&c).is_ok() {
                    continue;
                }
                let contrib = r_i.clone() * a.values()[idx].clone();
                dots.entry(c)
                    .and_modify(|acc| *acc = acc.clone() + contrib.clone())
                    .or_insert(contrib);
            }
        }

        // Rank candidates deterministically (ascending index) and keep the one
        // with the strictly-largest residual reduction.
        let mut candidates: Vec<usize> = dots.keys().copied().collect();
        candidates.sort_unstable();

        let mut best: Option<(usize, T)> = None;
        for c in candidates {
            let denom = col_norm_sq[c].clone();
            if Scalar::abs(denom.clone()) <= *drop_tol {
                // Empty/near-zero column cannot reduce the residual.
                continue;
            }
            let dot = dots[&c].clone();
            let score = dot.clone() * dot / denom;
            let take = match &best {
                Some((_, best_score)) => score > *best_score,
                None => true,
            };
            if take {
                best = Some((c, score));
            }
        }

        best.map(|(c, _)| c)
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
