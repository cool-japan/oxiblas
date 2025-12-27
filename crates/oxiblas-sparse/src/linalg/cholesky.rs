//! Sparse Cholesky decomposition.
//!
//! For symmetric positive definite matrices A, computes L such that A = L * L^T.
//! Uses a left-looking algorithm with fill-in prediction.

use crate::csc::CscMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error type for sparse Cholesky decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseCholError {
    /// Matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Matrix is not positive definite.
    NotPositiveDefinite {
        /// Row/column where failure occurred.
        index: usize,
    },
    /// Matrix has zero diagonal.
    ZeroDiagonal {
        /// Index of zero diagonal.
        index: usize,
    },
}

impl core::fmt::Display for SparseCholError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::NotPositiveDefinite { index } => {
                write!(f, "Matrix is not positive definite at index {index}")
            }
            Self::ZeroDiagonal { index } => {
                write!(f, "Zero diagonal element at index {index}")
            }
        }
    }
}

impl std::error::Error for SparseCholError {}

/// Sparse Cholesky factorization.
///
/// Stores the lower triangular factor L such that A = L * L^T.
#[derive(Debug, Clone)]
pub struct SparseCholesky<T: Scalar> {
    /// Lower triangular factor in CSC format.
    l: CscMatrix<T>,
    /// Permutation for fill-reducing ordering (identity if none).
    perm: Vec<usize>,
    /// Inverse permutation.
    perm_inv: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SparseCholesky<T> {
    /// Computes the Cholesky factorization of a symmetric positive definite matrix.
    ///
    /// The input matrix should be symmetric; only the lower triangle is used.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive definite matrix in CSC format
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or not positive definite.
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SparseCholError> {
        if a.nrows() != a.ncols() {
            return Err(SparseCholError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();

        // Use identity permutation (no fill-reducing ordering for now)
        let perm: Vec<usize> = (0..n).collect();
        let perm_inv: Vec<usize> = (0..n).collect();

        // Perform numeric factorization
        let l = Self::factorize_numeric(a, &perm, &perm_inv)?;

        Ok(Self { l, perm, perm_inv })
    }

    /// Computes Cholesky with fill-reducing ordering.
    ///
    /// Uses approximate minimum degree ordering.
    pub fn with_ordering(a: &CscMatrix<T>) -> Result<Self, SparseCholError> {
        if a.nrows() != a.ncols() {
            return Err(SparseCholError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();

        // Compute fill-reducing ordering using approximate minimum degree
        let perm = super::ordering::approximate_minimum_degree(a);
        let mut perm_inv = vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        // Permute matrix: P * A * P^T
        let ap = permute_symmetric(a, &perm, &perm_inv);

        // Perform numeric factorization
        let l = Self::factorize_numeric(
            &ap,
            &(0..n).collect::<Vec<_>>(),
            &(0..n).collect::<Vec<_>>(),
        )?;

        Ok(Self { l, perm, perm_inv })
    }

    /// Performs the numeric Cholesky factorization (left-looking algorithm).
    fn factorize_numeric(
        a: &CscMatrix<T>,
        _perm: &[usize],
        _perm_inv: &[usize],
    ) -> Result<CscMatrix<T>, SparseCholError> {
        let n = a.nrows();

        // Build L column by column using left-looking algorithm
        let mut l_col_ptrs: Vec<usize> = vec![0usize; n + 1];
        let mut l_row_indices: Vec<usize> = Vec::new();
        let mut l_values: Vec<T> = Vec::new();

        // Workspace
        let mut x: Vec<T> = vec![T::zero(); n]; // Dense accumulator for current column
        let mut pattern: Vec<bool> = vec![false; n]; // Sparsity pattern
        let mut stack: Vec<usize> = Vec::with_capacity(n); // Work stack for pattern

        for j in 0..n {
            // Clear workspace
            for idx in &stack {
                x[*idx] = T::zero();
                pattern[*idx] = false;
            }
            stack.clear();

            // Load column j of A into workspace
            let a_start = a.col_ptrs()[j];
            let a_end = a.col_ptrs()[j + 1];

            for i in a_start..a_end {
                let row = a.row_indices()[i];
                if row >= j {
                    // Only use lower triangle
                    x[row] = a.values()[i].clone();
                    if !pattern[row] {
                        pattern[row] = true;
                        stack.push(row);
                    }
                }
            }

            // Subtract contributions from previous columns of L
            // For each column k < j where L[j,k] != 0
            for k in 0..j {
                let l_start = l_col_ptrs[k];
                let l_end = l_col_ptrs[k + 1];

                // Find L[j,k]
                let mut ljk = T::zero();
                let mut found = false;

                for idx in l_start..l_end {
                    if l_row_indices[idx] == j {
                        ljk = l_values[idx].clone();
                        found = true;
                        break;
                    }
                }

                if !found {
                    continue;
                }

                // Update x[j..] -= ljk * L[j.., k]
                for idx in l_start..l_end {
                    let row = l_row_indices[idx];
                    if row >= j {
                        let lv: T = l_values[idx].clone();
                        let prod: T = ljk.clone() * lv;
                        let current: T = x[row].clone();
                        x[row] = current - prod;
                        if !pattern[row] {
                            pattern[row] = true;
                            stack.push(row);
                        }
                    }
                }
            }

            // Compute L[j,j] = sqrt(x[j])
            let diag = x[j].clone();

            if !(diag > T::zero()) {
                return Err(SparseCholError::NotPositiveDefinite { index: j });
            }

            let ljj = Real::sqrt(diag);

            if Scalar::abs(ljj.clone()) <= <T as Scalar>::epsilon() {
                return Err(SparseCholError::ZeroDiagonal { index: j });
            }

            // Scale x[j+1..] by 1/ljj
            let ljj_inv = T::one() / ljj.clone();

            // Store column j of L
            // Sort the pattern indices
            stack.sort_unstable();

            for &row in &stack {
                if row == j {
                    l_row_indices.push(j);
                    l_values.push(ljj.clone());
                } else if row > j {
                    let val = x[row].clone() * ljj_inv.clone();
                    if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                        l_row_indices.push(row);
                        l_values.push(val);
                    }
                }
            }

            l_col_ptrs[j + 1] = l_values.len();
        }

        Ok(unsafe { CscMatrix::new_unchecked(n, n, l_col_ptrs, l_row_indices, l_values) })
    }

    /// Returns the lower triangular factor L.
    pub fn l(&self) -> &CscMatrix<T> {
        &self.l
    }

    /// Returns the permutation used.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the inverse permutation.
    pub fn perm_inv(&self) -> &[usize] {
        &self.perm_inv
    }

    /// Solves A * x = b.
    ///
    /// Uses forward and backward substitution with L.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.l.nrows();
        assert_eq!(b.len(), n, "RHS vector length must match matrix size");

        // Apply permutation: b_perm = P * b
        let mut b_perm = vec![T::zero(); n];
        for i in 0..n {
            b_perm[i] = b[self.perm[i]].clone();
        }

        // Solve L * y = b_perm
        let y = super::triangular::solve_lower_csc(&self.l, &b_perm);

        // Solve L^T * x_perm = y
        let x_perm = super::triangular::solve_lower_transpose_csc(&self.l, &y);

        // Apply inverse permutation: x = P^T * x_perm
        let mut x = vec![T::zero(); n];
        for i in 0..n {
            x[self.perm[i]] = x_perm[i].clone();
        }

        x
    }

    /// Solves A * X = B for multiple right-hand sides.
    pub fn solve_multi(&self, b: &oxiblas_matrix::MatRef<'_, T>) -> oxiblas_matrix::Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let n = self.l.nrows();
        let nrhs = b.ncols();
        assert_eq!(b.nrows(), n, "RHS matrix rows must match matrix size");

        let mut x = oxiblas_matrix::Mat::zeros(n, nrhs);

        for j in 0..nrhs {
            // Extract column j of B
            let mut bj = vec![T::zero(); n];
            for i in 0..n {
                bj[i] = b[(i, j)].clone();
            }

            // Solve
            let xj = self.solve(&bj);

            // Store in X
            for i in 0..n {
                x[(i, j)] = xj[i].clone();
            }
        }

        x
    }

    /// Computes the log determinant of A.
    pub fn log_determinant(&self) -> T {
        let mut log_det = T::zero();
        let n = self.l.nrows();

        for j in 0..n {
            // Diagonal of L is at the start of each column
            let start = self.l.col_ptrs()[j];
            let diag = self.l.values()[start].clone();
            log_det = log_det + Real::ln(diag);
        }

        // det(A) = det(L)^2, so log(det(A)) = 2 * log(det(L))
        log_det + log_det
    }
}

/// Permutes a symmetric matrix: returns P * A * P^T
fn permute_symmetric<T: Scalar + Clone + Field>(
    a: &CscMatrix<T>,
    perm: &[usize],
    perm_inv: &[usize],
) -> CscMatrix<T> {
    let n = a.nrows();

    // Build permuted matrix
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for new_j in 0..n {
        let old_j = perm[new_j];

        // Collect entries for this column
        let mut entries: Vec<(usize, T)> = Vec::new();

        let start = a.col_ptrs()[old_j];
        let end = a.col_ptrs()[old_j + 1];

        for idx in start..end {
            let old_i = a.row_indices()[idx];
            let new_i = perm_inv[old_i];

            if new_i >= new_j {
                entries.push((new_i, a.values()[idx].clone()));
            }
        }

        // Also need to pick up entries from the upper triangle that map to lower
        for old_k in 0..n {
            if old_k == old_j {
                continue;
            }

            let new_k = perm_inv[old_k];
            if new_k < new_j {
                continue;
            }

            let k_start = a.col_ptrs()[old_k];
            let k_end = a.col_ptrs()[old_k + 1];

            for idx in k_start..k_end {
                if a.row_indices()[idx] == old_j {
                    entries.push((new_k, a.values()[idx].clone()));
                    break;
                }
            }
        }

        // Sort by row index
        entries.sort_by_key(|(row, _)| *row);

        // Remove duplicates, keeping the one from lower triangle
        entries.dedup_by_key(|(row, _)| *row);

        for (row, val) in entries {
            row_indices.push(row);
            values.push(val);
        }

        col_ptrs[new_j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

use crate::csr::CsrMatrix;

/// Incomplete Cholesky factorization with zero fill-in (IC(0)).
///
/// Computes L such that L * L^T ≈ A with the same sparsity pattern as the
/// lower triangle of A. Used as a preconditioner for CG on SPD systems.
///
/// The matrix must be symmetric positive definite. Only the lower triangle
/// is used.
#[derive(Debug, Clone)]
pub struct IC0<T: Scalar> {
    /// Lower triangular factor in CSR format.
    l: CsrMatrix<T>,
    /// Size of the matrix.
    n: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> IC0<T> {
    /// Computes the IC(0) factorization.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive definite matrix in CSR format (only lower triangle used)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ic0 = IC0::new(&a)?;
    /// let x = ic0.apply(&b);  // Preconditioner application
    /// ```
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, SparseCholError> {
        if a.nrows() != a.ncols() {
            return Err(SparseCholError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            let l = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            return Ok(Self { l, n });
        }

        // Extract lower triangular part and copy structure
        let mut row_ptrs = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // First pass: extract lower triangle structure
        for i in 0..n {
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];

            for idx in row_start..row_end {
                let j = a.col_indices()[idx];
                if j <= i {
                    col_indices.push(j);
                    values.push(a.values()[idx].clone());
                }
            }
            row_ptrs[i + 1] = col_indices.len();
        }

        // Build column index lookup for each row
        let mut col_to_idx: Vec<std::collections::HashMap<usize, usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            let mut map = std::collections::HashMap::new();
            for idx in start..end {
                map.insert(col_indices[idx], idx);
            }
            col_to_idx.push(map);
        }

        // IC(0) factorization (row-wise)
        for i in 0..n {
            let row_start = row_ptrs[i];
            let row_end = row_ptrs[i + 1];

            // Find diagonal position
            let diag_idx = match col_to_idx[i].get(&i) {
                Some(&idx) => idx,
                None => return Err(SparseCholError::ZeroDiagonal { index: i }),
            };

            // For each j < i where L[i,j] != 0
            for idx in row_start..row_end {
                let j = col_indices[idx];
                if j >= i {
                    break;
                }

                // Find L[j,j] (diagonal of row j)
                let ljj_idx = match col_to_idx[j].get(&j) {
                    Some(&idx) => idx,
                    None => return Err(SparseCholError::ZeroDiagonal { index: j }),
                };

                let ljj = values[ljj_idx].clone();
                if Scalar::abs(ljj.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseCholError::ZeroDiagonal { index: j });
                }

                // L[i,j] = L[i,j] / L[j,j]
                values[idx] = values[idx].clone() / ljj;
                let lij = values[idx].clone();

                // For each k > j where L[i,k] != 0 and L[j,k] != 0
                let j_start = row_ptrs[j];
                let j_end = row_ptrs[j + 1];

                for j_idx in j_start..j_end {
                    let k = col_indices[j_idx];
                    if k <= j {
                        continue;
                    }
                    if k > i {
                        break;
                    }

                    // Check if L[i,k] exists
                    if let Some(&i_k_idx) = col_to_idx[i].get(&k) {
                        // L[i,k] = L[i,k] - L[i,j] * L[j,k]
                        values[i_k_idx] =
                            values[i_k_idx].clone() - lij.clone() * values[j_idx].clone();
                    }
                    // If L[i,k] doesn't exist, we drop the fill-in (IC(0))
                }

                // Update diagonal: L[i,i] = L[i,i] - L[i,j]^2
                let lij_sq = lij.clone() * lij;
                values[diag_idx] = values[diag_idx].clone() - lij_sq;
            }

            // Compute L[i,i] = sqrt(L[i,i])
            let diag = values[diag_idx].clone();
            if !(diag > T::zero()) {
                return Err(SparseCholError::NotPositiveDefinite { index: i });
            }

            values[diag_idx] = Real::sqrt(diag);
        }

        let l = unsafe { CsrMatrix::new_unchecked(n, n, row_ptrs, col_indices, values) };

        Ok(Self { l, n })
    }

    /// Returns the lower triangular factor L.
    pub fn l(&self) -> &CsrMatrix<T> {
        &self.l
    }

    /// Applies the preconditioner: solves (L * L^T) * x = b.
    ///
    /// Uses forward substitution for L and backward substitution for L^T.
    pub fn apply(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        // Forward substitution: L * y = b
        let mut y = b.to_vec();
        for i in 0..n {
            let row_start = self.l.row_ptrs()[i];
            let row_end = self.l.row_ptrs()[i + 1];

            // Find diagonal
            let mut diag = T::one();
            for idx in row_start..row_end {
                let j = self.l.col_indices()[idx];
                if j == i {
                    diag = self.l.values()[idx].clone();
                } else if j < i {
                    y[i] = y[i].clone() - self.l.values()[idx].clone() * y[j].clone();
                }
            }

            y[i] = y[i].clone() / diag;
        }

        // Backward substitution: L^T * x = y
        // L is stored row-wise, so L^T is accessed column-wise
        let mut x = y;
        for i in (0..n).rev() {
            let row_start = self.l.row_ptrs()[i];
            let row_end = self.l.row_ptrs()[i + 1];

            // Find diagonal L[i,i]
            let mut diag = T::one();
            for idx in row_start..row_end {
                if self.l.col_indices()[idx] == i {
                    diag = self.l.values()[idx].clone();
                    break;
                }
            }

            x[i] = x[i].clone() / diag;

            // Update x[j] for j < i: L^T contribution
            for idx in row_start..row_end {
                let j = self.l.col_indices()[idx];
                if j < i {
                    x[j] = x[j].clone() - self.l.values()[idx].clone() * x[i].clone();
                }
            }
        }

        x
    }

    /// Returns the number of nonzeros in L.
    pub fn nnz(&self) -> usize {
        self.l.nnz()
    }
}

/// Incomplete Cholesky factorization with threshold (ICT).
///
/// Similar to IC(0) but allows fill-in based on a drop tolerance.
/// Elements smaller than τ * ||row_i(A)|| are dropped.
#[derive(Debug, Clone)]
pub struct ICT<T: Scalar> {
    /// Lower triangular factor in CSR format.
    l: CsrMatrix<T>,
    /// Size of the matrix.
    n: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> ICT<T> {
    /// Computes the ICT factorization.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive definite matrix in CSR format
    /// * `tau` - Drop tolerance (elements with |l_ij| < tau * ||row_i|| are dropped)
    /// * `p` - Maximum number of fill-in elements per row
    pub fn new(a: &CsrMatrix<T>, tau: T, p: usize) -> Result<Self, SparseCholError> {
        if a.nrows() != a.ncols() {
            return Err(SparseCholError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            let l = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            return Ok(Self { l, n });
        }

        // Storage for L (row-wise)
        let mut l_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];

        // Working vector
        let mut w = vec![T::zero(); n];
        let mut w_indices: Vec<usize> = Vec::with_capacity(n);

        for i in 0..n {
            // Clear working vector
            for &j in &w_indices {
                w[j] = T::zero();
            }
            w_indices.clear();

            // Copy row i of A (lower triangle only) into w
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];
            let mut row_norm = T::zero();

            for idx in row_start..row_end {
                let j = a.col_indices()[idx];
                if j <= i {
                    let val = a.values()[idx].clone();
                    row_norm = row_norm + Scalar::abs(val.clone()) * Scalar::abs(val.clone());
                    w[j] = val;
                    w_indices.push(j);
                }
            }

            row_norm = Real::sqrt(row_norm);
            let drop_tol = tau.clone() * row_norm;

            // Sort indices for processing
            w_indices.sort_unstable();

            // Process columns j < i
            let mut k = 0;
            while k < w_indices.len() {
                let j = w_indices[k];
                if j >= i {
                    break;
                }

                // Get L[j,j] from previous rows
                let ljj = if let Some((_, val)) = l_rows[j].iter().find(|(col, _)| *col == j) {
                    val.clone()
                } else {
                    k += 1;
                    continue;
                };

                if Scalar::abs(ljj.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseCholError::ZeroDiagonal { index: j });
                }

                // w[j] = w[j] / L[j,j]
                w[j] = w[j].clone() / ljj;
                let w_j = w[j].clone();

                // Apply drop tolerance
                if Scalar::abs(w_j.clone()) <= drop_tol {
                    w[j] = T::zero();
                    k += 1;
                    continue;
                }

                // Update w[k] for k > j: w[k] = w[k] - w[j] * L[j,k]
                for &(col, ref l_val) in &l_rows[j] {
                    if col <= j {
                        continue;
                    }
                    if col > i {
                        break;
                    }

                    let new_val = w[col].clone() - w_j.clone() * l_val.clone();
                    if Scalar::abs(w[col].clone()) <= <T as Scalar>::epsilon()
                        && Scalar::abs(new_val.clone()) > <T as Scalar>::epsilon()
                    {
                        w_indices.push(col);
                        w_indices.sort_unstable();
                    }
                    w[col] = new_val;
                }

                // Update diagonal: w[i] = w[i] - w[j]^2
                let w_j_sq = w_j.clone() * w_j.clone();
                w[i] = w[i].clone() - w_j_sq;

                k += 1;
            }

            // Extract L row (j <= i) with dropping
            let mut l_row: Vec<(usize, T)> = Vec::new();
            for &j in &w_indices {
                if j > i {
                    break;
                }
                let val = w[j].clone();
                if j == i {
                    // Diagonal: compute sqrt
                    if !(val > T::zero()) {
                        return Err(SparseCholError::NotPositiveDefinite { index: i });
                    }
                    l_row.push((j, Real::sqrt(val)));
                } else if Scalar::abs(val.clone()) > drop_tol {
                    l_row.push((j, val));
                }
            }

            // Keep only p largest elements (excluding diagonal)
            if l_row.len() > p + 1 {
                // Separate diagonal
                let diag_val = l_row
                    .iter()
                    .find(|(col, _)| *col == i)
                    .map(|(_, v)| v.clone());
                l_row.retain(|(col, _)| *col != i);

                // Keep p largest by magnitude
                l_row.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                l_row.truncate(p);

                // Re-add diagonal
                if let Some(d) = diag_val {
                    l_row.push((i, d));
                }
                l_row.sort_by_key(|(col, _)| *col);
            }

            l_rows[i] = l_row;
        }

        // Convert to CSR format
        let l = Self::rows_to_csr(n, n, &l_rows);

        Ok(Self { l, n })
    }

    /// Converts row-wise storage to CSR.
    fn rows_to_csr(nrows: usize, ncols: usize, rows: &[Vec<(usize, T)>]) -> CsrMatrix<T> {
        let mut row_ptrs = vec![0usize; nrows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for (i, row) in rows.iter().enumerate() {
            for (col, val) in row {
                col_indices.push(*col);
                values.push(val.clone());
            }
            row_ptrs[i + 1] = col_indices.len();
        }

        unsafe { CsrMatrix::new_unchecked(nrows, ncols, row_ptrs, col_indices, values) }
    }

    /// Returns the lower triangular factor L.
    pub fn l(&self) -> &CsrMatrix<T> {
        &self.l
    }

    /// Applies the preconditioner: solves (L * L^T) * x = b.
    pub fn apply(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        // Forward substitution: L * y = b
        let mut y = b.to_vec();
        for i in 0..n {
            let row_start = self.l.row_ptrs()[i];
            let row_end = self.l.row_ptrs()[i + 1];

            let mut diag = T::one();
            for idx in row_start..row_end {
                let j = self.l.col_indices()[idx];
                if j == i {
                    diag = self.l.values()[idx].clone();
                } else if j < i {
                    y[i] = y[i].clone() - self.l.values()[idx].clone() * y[j].clone();
                }
            }

            y[i] = y[i].clone() / diag;
        }

        // Backward substitution: L^T * x = y
        let mut x = y;
        for i in (0..n).rev() {
            let row_start = self.l.row_ptrs()[i];
            let row_end = self.l.row_ptrs()[i + 1];

            let mut diag = T::one();
            for idx in row_start..row_end {
                if self.l.col_indices()[idx] == i {
                    diag = self.l.values()[idx].clone();
                    break;
                }
            }

            x[i] = x[i].clone() / diag;

            for idx in row_start..row_end {
                let j = self.l.col_indices()[idx];
                if j < i {
                    x[j] = x[j].clone() - self.l.values()[idx].clone() * x[i].clone();
                }
            }
        }

        x
    }

    /// Returns the number of nonzeros in L.
    pub fn nnz(&self) -> usize {
        self.l.nnz()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spd_matrix() -> CscMatrix<f64> {
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        // Symmetric positive definite
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];

        CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap()
    }

    #[test]
    fn test_sparse_cholesky() {
        let a = make_spd_matrix();
        let chol = SparseCholesky::new(&a).unwrap();

        let l = chol.l();
        assert_eq!(l.nrows(), 3);
        assert_eq!(l.ncols(), 3);

        // Verify L is lower triangular
        for (row, col, _) in l.iter() {
            assert!(row >= col, "L should be lower triangular");
        }
    }

    #[test]
    fn test_sparse_cholesky_solve() {
        let a = make_spd_matrix();
        let chol = SparseCholesky::new(&a).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let x = chol.solve(&b);

        // Verify A * x ≈ b by computing residual
        let mut ax = [0.0; 3];
        for col in 0..3 {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];

            for i in start..end {
                let row = a.row_indices()[i];
                ax[row] += a.values()[i] * x[col];
            }
        }

        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-10, "Solution verification failed");
        }
    }

    #[test]
    fn test_sparse_cholesky_identity() {
        let a = CscMatrix::<f64>::eye(5);
        let chol = SparseCholesky::new(&a).unwrap();

        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = chol.solve(&b);

        for i in 0..5 {
            assert!((x[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_cholesky_not_spd() {
        // Negative definite matrix
        let values = vec![-4.0, 1.0, 1.0, -4.0, 1.0, 1.0, -4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];

        let a = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();
        let result = SparseCholesky::new(&a);

        assert!(matches!(
            result,
            Err(SparseCholError::NotPositiveDefinite { .. })
        ));
    }

    fn make_spd_csr_matrix() -> CsrMatrix<f64> {
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        // Symmetric positive definite
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }

    fn make_larger_spd_csr_matrix() -> CsrMatrix<f64> {
        // 5x5 diagonally dominant SPD matrix (symmetric)
        let n = 5;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];

        for i in 0..n {
            if i > 0 {
                values.push(1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
    }

    #[test]
    fn test_ic0_basic() {
        let a = make_spd_csr_matrix();
        let ic0 = IC0::new(&a).unwrap();

        // Check L has correct dimensions
        assert_eq!(ic0.l().nrows(), 3);
        assert_eq!(ic0.l().ncols(), 3);
        assert!(ic0.nnz() > 0);
    }

    #[test]
    fn test_ic0_apply() {
        let a = make_spd_csr_matrix();
        let ic0 = IC0::new(&a).unwrap();

        let b = vec![5.0, 6.0, 5.0];
        let x = ic0.apply(&b);

        // Should produce finite results
        assert_eq!(x.len(), 3);
        assert!(x.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_ic0_preconditioner_quality() {
        let a = make_larger_spd_csr_matrix();
        let ic0 = IC0::new(&a).unwrap();

        // Use IC0 to approximately solve A*x = b
        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let x = ic0.apply(&b);

        // Compute residual r = b - A*x
        let mut ax = [0.0; 5];
        for i in 0..5 {
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];
            for idx in row_start..row_end {
                ax[i] += a.values()[idx] * x[a.col_indices()[idx]];
            }
        }

        let residual: f64 = (0..5).map(|i| (b[i] - ax[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

        // IC0 should give reasonable approximation
        assert!(
            residual / b_norm < 2.0,
            "IC0 relative residual too large: {}",
            residual / b_norm
        );
    }

    #[test]
    fn test_ict_basic() {
        let a = make_spd_csr_matrix();
        let ict = ICT::new(&a, 1e-6, 10).unwrap();

        assert_eq!(ict.l().nrows(), 3);
        assert_eq!(ict.l().ncols(), 3);
        assert!(ict.nnz() > 0);
    }

    #[test]
    fn test_ict_apply() {
        let a = make_larger_spd_csr_matrix();
        let ict = ICT::new(&a, 1e-10, 5).unwrap();

        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let x = ict.apply(&b);

        assert_eq!(x.len(), 5);
        assert!(x.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_ict_dropping() {
        // Create a matrix with many small off-diagonal elements
        let n = 5;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    values.push(10.0);
                } else {
                    values.push(0.01);
                }
                col_indices.push(j);
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        // With large tau, small elements should be dropped
        let ict_drop = ICT::new(&a, 0.1, 2).unwrap();
        // With small tau, more elements kept
        let ict_keep = ICT::new(&a, 1e-10, 10).unwrap();

        // Dropping version should have fewer nonzeros
        assert!(ict_drop.nnz() <= ict_keep.nnz());
    }

    #[test]
    fn test_ic0_vs_ict() {
        let a = make_larger_spd_csr_matrix();

        let ic0 = IC0::new(&a).unwrap();
        let ict = ICT::new(&a, 1e-10, 10).unwrap();

        // Both should produce valid approximations
        let b = vec![1.0; 5];

        let x_ic0 = ic0.apply(&b);
        let x_ict = ict.apply(&b);

        // Both should give finite results
        assert!(x_ic0.iter().all(|&v| v.is_finite()));
        assert!(x_ict.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_ic0_identity() {
        // Identity matrix
        let values = vec![1.0f64; 3];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];

        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let ic0 = IC0::new(&a).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let x = ic0.apply(&b);

        // Should return b for identity
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-10);
        }
    }
}
