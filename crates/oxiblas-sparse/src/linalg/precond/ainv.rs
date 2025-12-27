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
/// ```ignore
/// use oxiblas_sparse::linalg::precond::{AINV, AINVConfig};
///
/// let config = AINVConfig::default();
/// let ainv = AINV::new(&matrix, config)?;
/// let mut z = vec![0.0; n];
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
                // Modified Gram-Schmidt: orthogonalize against columns 0..j
                for k in 0..j {
                    // Compute A * z_k (we need this for biorthogonalization)
                    Self::sparse_mv_csc(&a_csc, &z_cols[k], &mut az, n);
                    Self::sparse_mv_csc_t(&a_csc, &w_cols[k], &mut atw, n);

                    // Compute z_j^T * A * z_k
                    let mut zjt_a_zk = T::zero();
                    for (row, _val) in &z_cols[k] {
                        zjt_a_zk = zjt_a_zk + z_j[*row].clone() * az[*row].clone();
                    }
                    for idx in a_csc.col_ptrs()[k]..a_csc.col_ptrs()[k + 1] {
                        let row = a_csc.row_indices()[idx];
                        zjt_a_zk = zjt_a_zk + z_j[row].clone() * a_csc.values()[idx].clone();
                    }

                    // Compute w_k^T * A * z_j
                    let mut wkt_a_zj = T::zero();
                    for (row, val) in &w_cols[k] {
                        wkt_a_zj = wkt_a_zj + val.clone() * az[*row].clone();
                    }

                    // Update: z_j = z_j - (z_j^T * A * z_k) / d_k * z_k
                    if Scalar::abs(d_inv[k].clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                        let dk = T::one() / d_inv[k].clone();
                        if Scalar::abs(dk.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                            let alpha = zjt_a_zk / dk.clone();
                            for (row, val) in &z_cols[k] {
                                z_j[*row] = z_j[*row].clone() - alpha.clone() * val.clone();
                            }
                        }
                    }

                    // Similarly update w_j
                    let mut wjt_a_wk = T::zero();
                    for (row, _val) in &w_cols[k] {
                        wjt_a_wk = wjt_a_wk + w_j[*row].clone() * atw[*row].clone();
                    }
                    for idx in a_csc.col_ptrs()[k]..a_csc.col_ptrs()[k + 1] {
                        let row = a_csc.row_indices()[idx];
                        wjt_a_wk = wjt_a_wk + w_j[row].clone() * a_csc.values()[idx].clone();
                    }

                    if Scalar::abs(d_inv[k].clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                        let dk = T::one() / d_inv[k].clone();
                        if Scalar::abs(dk.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                            let beta = wjt_a_wk / dk.clone();
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

            // Store sparse z_j (only lower triangular part, i >= j)
            let mut z_entries: Vec<(usize, T)> = Vec::new();
            for i in j..n {
                if Scalar::abs(z_j[i].clone()) >= z_threshold {
                    z_entries.push((i, z_j[i].clone()));
                }
            }

            // Ensure at least diagonal entry
            if z_entries.is_empty() || z_entries[0].0 != j {
                z_entries.insert(0, (j, T::one()));
            }

            // Limit number of entries
            if z_entries.len() > config.max_nnz_per_col {
                // Keep the largest entries
                z_entries.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                z_entries.truncate(config.max_nnz_per_col);
                z_entries.sort_by_key(|(i, _)| *i);
            }
            z_cols[j] = z_entries;

            // Store sparse w_j (only lower triangular part, i >= j)
            let mut w_entries: Vec<(usize, T)> = Vec::new();
            for i in j..n {
                if Scalar::abs(w_j[i].clone()) >= w_threshold {
                    w_entries.push((i, w_j[i].clone()));
                }
            }

            if w_entries.is_empty() || w_entries[0].0 != j {
                w_entries.insert(0, (j, T::one()));
            }

            if w_entries.len() > config.max_nnz_per_col {
                w_entries.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                w_entries.truncate(config.max_nnz_per_col);
                w_entries.sort_by_key(|(i, _)| *i);
            }
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

    /// Compute y = A^T * w where w is stored as sparse column entries
    fn sparse_mv_csc_t(a_csc: &CscMatrix<T>, w_col: &[(usize, T)], y: &mut [T], _n: usize) {
        for val in y.iter_mut() {
            *val = T::zero();
        }
        for (row, w_val) in w_col {
            // Row 'row' of A^T (= column 'row' of A) multiplied by w_val
            let start = a_csc.col_ptrs()[*row];
            let end = a_csc.col_ptrs()[*row + 1];
            for idx in start..end {
                let j = a_csc.row_indices()[idx];
                y[j] = y[j].clone() + a_csc.values()[idx].clone() * w_val.clone();
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
