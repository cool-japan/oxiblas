//! Sparse LU decomposition.
//!
//! Provides:
//! - Full sparse LU factorization with partial pivoting
//! - ILU(0) incomplete factorization for preconditioning

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error type for sparse LU decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseLUError {
    /// Matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Matrix is singular (zero pivot).
    Singular {
        /// Row where singularity was detected.
        row: usize,
    },
    /// Numerical instability detected.
    NumericalInstability {
        /// Row where instability occurred.
        row: usize,
    },
}

impl core::fmt::Display for SparseLUError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::Singular { row } => {
                write!(f, "Matrix is singular at row {row}")
            }
            Self::NumericalInstability { row } => {
                write!(f, "Numerical instability at row {row}")
            }
        }
    }
}

impl std::error::Error for SparseLUError {}

/// Sparse LU factorization with partial pivoting.
///
/// Computes P * A = L * U where:
/// - P is a permutation matrix (row pivoting)
/// - L is lower triangular with unit diagonal
/// - U is upper triangular
#[derive(Debug, Clone)]
pub struct SparseLU<T: Scalar> {
    /// Lower triangular factor (unit diagonal, stored in CSC).
    l: CscMatrix<T>,
    /// Upper triangular factor (stored in CSC).
    u: CscMatrix<T>,
    /// Row permutation.
    perm: Vec<usize>,
    /// Inverse row permutation.
    #[allow(dead_code)]
    perm_inv: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SparseLU<T> {
    /// Computes the LU factorization of a sparse matrix.
    ///
    /// Uses partial pivoting for stability.
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();

        // Initialize permutation as identity
        let mut perm: Vec<usize> = (0..n).collect();

        // Work with dense columns for simplicity (can be optimized for truly sparse case)
        let mut l_data = vec![vec![T::zero(); n]; n];
        let mut u_data = vec![vec![T::zero(); n]; n];

        // Copy A into working storage
        let mut work = vec![vec![T::zero(); n]; n];
        for col in 0..n {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                work[a.row_indices()[idx]][col] = a.values()[idx].clone();
            }
        }

        // LU factorization with partial pivoting
        for k in 0..n {
            // Find pivot (maximum absolute value in column k, rows k..n)
            let mut max_val = T::zero();
            let mut max_row = k;

            for i in k..n {
                let perm_i = perm[i];
                let val = Scalar::abs(work[perm_i][k].clone());
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val <= <T as Scalar>::epsilon() {
                return Err(SparseLUError::Singular { row: k });
            }

            // Swap rows k and max_row in permutation
            perm.swap(k, max_row);

            let pivot_row = perm[k];
            let pivot = work[pivot_row][k].clone();

            // Store U[k, k]
            u_data[k][k] = pivot.clone();

            // Compute L[i, k] = A[perm[i], k] / U[k, k] for i > k
            // and update A[perm[i], j] for j > k
            for i in (k + 1)..n {
                let row_i = perm[i];
                let lik = work[row_i][k].clone() / pivot.clone();

                l_data[i][k] = lik.clone();

                // Update row i
                for j in (k + 1)..n {
                    work[row_i][j] =
                        work[row_i][j].clone() - lik.clone() * work[pivot_row][j].clone();
                }
            }

            // Store U[k, j] for j > k
            for j in (k + 1)..n {
                u_data[k][j] = work[pivot_row][j].clone();
            }
        }

        // Set L diagonal to 1
        for i in 0..n {
            l_data[i][i] = T::one();
        }

        // Convert to CSC format
        let l = dense_to_csc_lower(&l_data);
        let u = dense_to_csc_upper(&u_data);

        // Compute inverse permutation
        let mut perm_inv = vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        Ok(Self {
            l,
            u,
            perm,
            perm_inv,
        })
    }

    /// Returns the lower triangular factor L.
    pub fn l(&self) -> &CscMatrix<T> {
        &self.l
    }

    /// Returns the upper triangular factor U.
    pub fn u(&self) -> &CscMatrix<T> {
        &self.u
    }

    /// Returns the row permutation.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Solves A * x = b.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.l.nrows();
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        // Apply permutation: b_perm[i] = b[perm[i]]
        let mut b_perm = vec![T::zero(); n];
        for i in 0..n {
            b_perm[i] = b[self.perm[i]].clone();
        }

        // Solve L * y = b_perm
        let y = super::triangular::solve_lower_csc(&self.l, &b_perm);

        // Solve U * x = y
        super::triangular::solve_upper_csc(&self.u, &y)
    }

    /// Computes the determinant.
    pub fn determinant(&self) -> T {
        let n = self.u.nrows();
        let mut det = T::one();

        // Determinant is product of U diagonal
        for j in 0..n {
            let start = self.u.col_ptrs()[j];
            let end = self.u.col_ptrs()[j + 1];

            // Find diagonal element U[j,j]
            // In upper triangular CSC, entries are stored in ascending row order,
            // so diagonal (row j) is typically the last entry in column j
            let mut diag = T::zero();
            for idx in start..end {
                if self.u.row_indices()[idx] == j {
                    diag = self.u.values()[idx].clone();
                    break;
                }
            }
            det = det * diag;
        }

        // Account for permutation sign
        let sign = permutation_sign(&self.perm);
        if sign < 0 {
            det = T::zero() - det;
        }

        det
    }
}

/// Incomplete LU factorization with zero fill-in (ILU(0)).
///
/// Computes L and U such that L * U ≈ A with the same sparsity pattern as A.
/// Used as a preconditioner for iterative solvers.
#[derive(Debug, Clone)]
pub struct ILU0<T: Scalar> {
    /// Combined L + U - I stored in CSR format (same pattern as input).
    lu: CsrMatrix<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> ILU0<T> {
    /// Computes the ILU(0) factorization.
    ///
    /// The input matrix should be in CSR format.
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();

        // Copy input structure and values
        let row_ptrs = a.row_ptrs().to_vec();
        let col_indices = a.col_indices().to_vec();
        let mut values = a.values().to_vec();

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

        // ILU(0) factorization
        for i in 0..n {
            let row_start = row_ptrs[i];
            let row_end = row_ptrs[i + 1];

            // For each j < i where A[i,j] != 0
            for idx in row_start..row_end {
                let j = col_indices[idx];
                if j >= i {
                    break;
                }

                // Find A[j,j] (diagonal of row j)
                let diag_idx = match col_to_idx[j].get(&j) {
                    Some(&idx) => idx,
                    None => return Err(SparseLUError::Singular { row: j }),
                };

                let ajj = values[diag_idx].clone();
                if Scalar::abs(ajj.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: j });
                }

                // A[i,j] = A[i,j] / A[j,j]
                values[idx] = values[idx].clone() / ajj;

                let aij = values[idx].clone();

                // For each k > j where A[i,k] != 0 and A[j,k] != 0
                let j_start = row_ptrs[j];
                let j_end = row_ptrs[j + 1];

                for j_idx in j_start..j_end {
                    let k = col_indices[j_idx];
                    if k <= j {
                        continue;
                    }

                    // Check if A[i,k] exists
                    if let Some(&i_k_idx) = col_to_idx[i].get(&k) {
                        // A[i,k] = A[i,k] - A[i,j] * A[j,k]
                        values[i_k_idx] =
                            values[i_k_idx].clone() - aij.clone() * values[j_idx].clone();
                    }
                    // If A[i,k] doesn't exist, we drop the fill-in (ILU(0))
                }
            }
        }

        let lu = unsafe { CsrMatrix::new_unchecked(n, n, row_ptrs, col_indices, values) };

        Ok(Self { lu })
    }

    /// Returns the combined L + U - I matrix.
    pub fn lu(&self) -> &CsrMatrix<T> {
        &self.lu
    }

    /// Applies the preconditioner: solves (LU) * x = b.
    pub fn apply(&self, b: &[T]) -> Vec<T> {
        super::triangular::solve_ilu0(&self.lu, b)
    }
}

/// Incomplete LU factorization with threshold (ILUT).
///
/// Computes L and U such that L * U ≈ A with dropping based on:
/// 1. Absolute threshold: elements smaller than τ * ||row_i(A)|| are dropped
/// 2. Fill limit: at most `p` elements kept in each row of L and U
///
/// This produces a more accurate approximation than ILU(0) at the cost
/// of more fill-in and computation.
#[derive(Debug, Clone)]
pub struct ILUT<T: Scalar> {
    /// Lower triangular factor (unit diagonal implicit).
    l: CsrMatrix<T>,
    /// Upper triangular factor.
    u: CsrMatrix<T>,
    /// Size of the matrix.
    n: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> ILUT<T> {
    /// Computes the ILUT factorization.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix in CSR format
    /// * `tau` - Drop tolerance (elements with |a_ij| < tau * ||row_i|| are dropped)
    /// * `p` - Maximum number of fill-in elements per row (in addition to original nnz)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ilut = ILUT::new(&a, 1e-3, 10)?;
    /// let x = ilut.apply(&b);
    /// ```
    pub fn new(a: &CsrMatrix<T>, tau: T, p: usize) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            let l = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            let u = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            return Ok(Self { l, u, n });
        }

        // Storage for L and U (row-wise)
        let mut l_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut u_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];

        // Working vector for current row
        let mut w = vec![T::zero(); n];
        let mut w_indices: Vec<usize> = Vec::with_capacity(n);

        for i in 0..n {
            // Clear working vector
            for &j in &w_indices {
                w[j] = T::zero();
            }
            w_indices.clear();

            // Copy row i of A into w
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];
            let mut row_norm = T::zero();

            for idx in row_start..row_end {
                let j = a.col_indices()[idx];
                let val = a.values()[idx].clone();
                row_norm = row_norm + Scalar::abs(val.clone()) * Scalar::abs(val.clone());
                w[j] = val;
                w_indices.push(j);
            }

            row_norm = Real::sqrt(row_norm);
            let drop_tol = tau.clone() * row_norm;

            // Sort indices for processing
            w_indices.sort_unstable();

            // Process columns j < i where w[j] != 0
            let mut k = 0;
            while k < w_indices.len() {
                let j = w_indices[k];
                if j >= i {
                    break;
                }

                // Get u_jj from U
                let u_jj = if let Some((_, val)) = u_rows[j].iter().find(|(col, _)| *col == j) {
                    val.clone()
                } else {
                    k += 1;
                    continue;
                };

                if Scalar::abs(u_jj.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: j });
                }

                // w[j] = w[j] / u_jj
                w[j] = w[j].clone() / u_jj;
                let w_j = w[j].clone();

                // Apply drop tolerance to L element
                if Scalar::abs(w_j.clone()) <= drop_tol {
                    w[j] = T::zero();
                    k += 1;
                    continue;
                }

                // Update w[k] for k > j: w[k] = w[k] - w[j] * u[j,k]
                for &(col, ref u_val) in &u_rows[j] {
                    if col <= j {
                        continue;
                    }
                    let new_val = w[col].clone() - w_j.clone() * u_val.clone();
                    if Scalar::abs(w[col].clone()) <= <T as Scalar>::epsilon()
                        && Scalar::abs(new_val.clone()) > <T as Scalar>::epsilon()
                    {
                        // New fill-in
                        w_indices.push(col);
                        w_indices.sort_unstable();
                    }
                    w[col] = new_val;
                }

                k += 1;
            }

            // Extract L row (j < i) with dropping
            let mut l_row: Vec<(usize, T)> = Vec::new();
            for &j in &w_indices {
                if j >= i {
                    break;
                }
                let val = w[j].clone();
                if Scalar::abs(val.clone()) > drop_tol {
                    l_row.push((j, val));
                }
            }

            // Keep only p largest elements in L row (by magnitude)
            if l_row.len() > p {
                l_row.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                l_row.truncate(p);
                l_row.sort_by_key(|(col, _)| *col);
            }
            l_rows[i] = l_row;

            // Extract U row (j >= i) with dropping
            let mut u_row: Vec<(usize, T)> = Vec::new();
            for &j in &w_indices {
                if j < i {
                    continue;
                }
                let val = w[j].clone();
                if j == i || Scalar::abs(val.clone()) > drop_tol {
                    u_row.push((j, val));
                }
            }

            // Keep diagonal + p largest elements in U row (by magnitude)
            if u_row.len() > p + 1 {
                // Separate diagonal
                let diag_val = u_row
                    .iter()
                    .find(|(col, _)| *col == i)
                    .map(|(_, v)| v.clone());
                u_row.retain(|(col, _)| *col != i);

                // Keep p largest
                u_row.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                u_row.truncate(p);

                // Re-add diagonal
                if let Some(d) = diag_val {
                    u_row.push((i, d));
                }
                u_row.sort_by_key(|(col, _)| *col);
            }

            // Check for zero pivot
            let diag = u_row.iter().find(|(col, _)| *col == i);
            if diag.is_none()
                || Scalar::abs(diag.expect("diag checked above").1.clone())
                    <= <T as Scalar>::epsilon()
            {
                return Err(SparseLUError::Singular { row: i });
            }

            u_rows[i] = u_row;
        }

        // Convert to CSR format
        let l = Self::rows_to_csr(n, n, &l_rows);
        let u = Self::rows_to_csr(n, n, &u_rows);

        Ok(Self { l, u, n })
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

    /// Returns the lower triangular factor L (unit diagonal implicit).
    pub fn l(&self) -> &CsrMatrix<T> {
        &self.l
    }

    /// Returns the upper triangular factor U.
    pub fn u(&self) -> &CsrMatrix<T> {
        &self.u
    }

    /// Applies the preconditioner: solves (LU) * x = b.
    ///
    /// Uses forward substitution for L and backward substitution for U.
    pub fn apply(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        // Forward substitution: L * y = b (L has unit diagonal)
        let mut y = b.to_vec();
        for i in 0..n {
            let row_start = self.l.row_ptrs()[i];
            let row_end = self.l.row_ptrs()[i + 1];

            for idx in row_start..row_end {
                let j = self.l.col_indices()[idx];
                let l_ij = self.l.values()[idx].clone();
                y[i] = y[i].clone() - l_ij * y[j].clone();
            }
        }

        // Backward substitution: U * x = y
        let mut x = y;
        for i in (0..n).rev() {
            let row_start = self.u.row_ptrs()[i];
            let row_end = self.u.row_ptrs()[i + 1];

            let mut diag = T::one();
            for idx in row_start..row_end {
                let j = self.u.col_indices()[idx];
                if j == i {
                    diag = self.u.values()[idx].clone();
                } else if j > i {
                    x[i] = x[i].clone() - self.u.values()[idx].clone() * x[j].clone();
                }
            }

            x[i] = x[i].clone() / diag;
        }

        x
    }

    /// Returns the approximate number of nonzeros in L + U.
    pub fn nnz(&self) -> usize {
        self.l.nnz() + self.u.nnz()
    }
}

/// Incomplete LU factorization with threshold and pivoting (ILUTP).
///
/// Similar to ILUT but includes column pivoting for improved stability.
#[derive(Debug, Clone)]
pub struct ILUTP<T: Scalar> {
    /// Lower triangular factor (unit diagonal implicit).
    l: CsrMatrix<T>,
    /// Upper triangular factor.
    u: CsrMatrix<T>,
    /// Column permutation.
    perm: Vec<usize>,
    #[allow(dead_code)]
    /// Inverse column permutation.
    perm_inv: Vec<usize>,
    /// Size of the matrix.
    n: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> ILUTP<T> {
    /// Computes the ILUTP factorization.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix in CSR format
    /// * `tau` - Drop tolerance
    /// * `p` - Maximum fill-in per row
    /// * `pivot_tol` - Pivoting tolerance (0.0 for no pivoting, 1.0 for full pivoting)
    pub fn new(a: &CsrMatrix<T>, tau: T, p: usize, pivot_tol: T) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            let l = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            let u = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            return Ok(Self {
                l,
                u,
                perm: vec![],
                perm_inv: vec![],
                n,
            });
        }

        // Initialize permutation as identity
        let mut perm: Vec<usize> = (0..n).collect();
        let mut perm_inv: Vec<usize> = (0..n).collect();

        // Storage for L and U (row-wise)
        let mut l_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut u_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];

        // Working vector for current row
        let mut w = vec![T::zero(); n];
        let mut w_indices: Vec<usize> = Vec::with_capacity(n);

        for i in 0..n {
            // Clear working vector
            for &j in &w_indices {
                w[j] = T::zero();
            }
            w_indices.clear();

            // Copy row i of A into w (with column permutation)
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];
            let mut row_norm = T::zero();

            for idx in row_start..row_end {
                let orig_col = a.col_indices()[idx];
                let j = perm_inv[orig_col];
                let val = a.values()[idx].clone();
                row_norm = row_norm + Scalar::abs(val.clone()) * Scalar::abs(val.clone());
                w[j] = val;
                w_indices.push(j);
            }

            row_norm = Real::sqrt(row_norm);
            let drop_tol = tau.clone() * row_norm;

            w_indices.sort_unstable();
            w_indices.dedup();

            // Process columns j < i
            let mut k = 0;
            while k < w_indices.len() {
                let j = w_indices[k];
                if j >= i {
                    break;
                }

                let u_jj = if let Some((_, val)) = u_rows[j].iter().find(|(col, _)| *col == j) {
                    val.clone()
                } else {
                    k += 1;
                    continue;
                };

                if Scalar::abs(u_jj.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: j });
                }

                w[j] = w[j].clone() / u_jj;
                let w_j = w[j].clone();

                if Scalar::abs(w_j.clone()) <= drop_tol {
                    w[j] = T::zero();
                    k += 1;
                    continue;
                }

                for &(col, ref u_val) in &u_rows[j] {
                    if col <= j {
                        continue;
                    }
                    let new_val = w[col].clone() - w_j.clone() * u_val.clone();
                    if Scalar::abs(w[col].clone()) <= <T as Scalar>::epsilon()
                        && Scalar::abs(new_val.clone()) > <T as Scalar>::epsilon()
                    {
                        w_indices.push(col);
                        w_indices.sort_unstable();
                        w_indices.dedup();
                    }
                    w[col] = new_val;
                }

                k += 1;
            }

            // Column pivoting: find maximum magnitude element in U part
            let mut max_col = i;
            let mut max_val = if Scalar::abs(w[i].clone()) > <T as Scalar>::epsilon() {
                Scalar::abs(w[i].clone())
            } else {
                T::zero()
            };

            for &j in &w_indices {
                if j <= i {
                    continue;
                }
                let val = Scalar::abs(w[j].clone());
                if val > max_val * pivot_tol {
                    max_val = val;
                    max_col = j;
                }
            }

            // Apply column pivoting if needed
            if max_col != i && Scalar::abs(max_val.clone()) > <T as Scalar>::epsilon() {
                // Swap columns i and max_col in permutation
                let orig_i = perm[i];
                let orig_max = perm[max_col];
                perm.swap(i, max_col);
                perm_inv[orig_i] = max_col;
                perm_inv[orig_max] = i;

                // Swap in working vector
                let tmp = w[i].clone();
                w[i] = w[max_col].clone();
                w[max_col] = tmp;
            }

            // Extract L row (j < i) with dropping
            let mut l_row: Vec<(usize, T)> = Vec::new();
            for &j in &w_indices {
                if j >= i {
                    break;
                }
                let val = w[j].clone();
                if Scalar::abs(val.clone()) > drop_tol {
                    l_row.push((j, val));
                }
            }

            if l_row.len() > p {
                l_row.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                l_row.truncate(p);
                l_row.sort_by_key(|(col, _)| *col);
            }
            l_rows[i] = l_row;

            // Extract U row (j >= i) with dropping
            let mut u_row: Vec<(usize, T)> = Vec::new();
            for &j in &w_indices {
                if j < i {
                    continue;
                }
                let val = w[j].clone();
                if j == i || Scalar::abs(val.clone()) > drop_tol {
                    u_row.push((j, val));
                }
            }

            // Ensure diagonal is present (may be zero from dropping)
            if !u_row.iter().any(|(col, _)| *col == i) {
                u_row.push((i, w[i].clone()));
                u_row.sort_by_key(|(col, _)| *col);
            }

            if u_row.len() > p + 1 {
                let diag_val = u_row
                    .iter()
                    .find(|(col, _)| *col == i)
                    .map(|(_, v)| v.clone());
                u_row.retain(|(col, _)| *col != i);
                u_row.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                u_row.truncate(p);
                if let Some(d) = diag_val {
                    u_row.push((i, d));
                }
                u_row.sort_by_key(|(col, _)| *col);
            }

            let diag = u_row.iter().find(|(col, _)| *col == i);
            if diag.is_none()
                || Scalar::abs(diag.expect("diag checked above").1.clone())
                    <= <T as Scalar>::epsilon()
            {
                return Err(SparseLUError::Singular { row: i });
            }

            u_rows[i] = u_row;
        }

        let l = ILUT::rows_to_csr(n, n, &l_rows);
        let u = ILUT::rows_to_csr(n, n, &u_rows);

        Ok(Self {
            l,
            u,
            perm,
            perm_inv,
            n,
        })
    }

    /// Returns the lower triangular factor L.
    pub fn l(&self) -> &CsrMatrix<T> {
        &self.l
    }

    /// Returns the upper triangular factor U.
    pub fn u(&self) -> &CsrMatrix<T> {
        &self.u
    }

    /// Returns the column permutation.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Applies the preconditioner: solves (LU) * P * x = b.
    pub fn apply(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        // Forward substitution: L * y = b
        let mut y = b.to_vec();
        for i in 0..n {
            let row_start = self.l.row_ptrs()[i];
            let row_end = self.l.row_ptrs()[i + 1];

            for idx in row_start..row_end {
                let j = self.l.col_indices()[idx];
                let l_ij = self.l.values()[idx].clone();
                y[i] = y[i].clone() - l_ij * y[j].clone();
            }
        }

        // Backward substitution: U * z = y
        let mut z = y;
        for i in (0..n).rev() {
            let row_start = self.u.row_ptrs()[i];
            let row_end = self.u.row_ptrs()[i + 1];

            let mut diag = T::one();
            for idx in row_start..row_end {
                let j = self.u.col_indices()[idx];
                if j == i {
                    diag = self.u.values()[idx].clone();
                } else if j > i {
                    z[i] = z[i].clone() - self.u.values()[idx].clone() * z[j].clone();
                }
            }

            z[i] = z[i].clone() / diag;
        }

        // Apply column permutation: x[perm[i]] = z[i]
        let mut x = vec![T::zero(); n];
        for i in 0..n {
            x[self.perm[i]] = z[i].clone();
        }

        x
    }

    /// Returns the approximate number of nonzeros.
    pub fn nnz(&self) -> usize {
        self.l.nnz() + self.u.nnz()
    }
}

/// Converts a dense lower triangular matrix to CSC format.
fn dense_to_csc_lower<T: Scalar + Clone + Field>(data: &[Vec<T>]) -> CscMatrix<T> {
    let n = data.len();
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        for i in j..n {
            let val = data[i][j].clone();
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() || i == j {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

/// Converts a dense upper triangular matrix to CSC format.
fn dense_to_csc_upper<T: Scalar + Clone + Field>(data: &[Vec<T>]) -> CscMatrix<T> {
    let n = data.len();
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        for i in 0..=j {
            let val = data[i][j].clone();
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() || i == j {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

/// Computes the sign of a permutation (+1 for even, -1 for odd).
fn permutation_sign(perm: &[usize]) -> i32 {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut sign = 1i32;

    for i in 0..n {
        if visited[i] {
            continue;
        }

        let mut cycle_len = 0;
        let mut j = i;

        while !visited[j] {
            visited[j] = true;
            j = perm[j];
            cycle_len += 1;
        }

        if cycle_len % 2 == 0 {
            sign = -sign;
        }
    }

    sign
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_matrix() -> CscMatrix<f64> {
        // A = [2 1 0]
        //     [1 3 1]
        //     [0 1 2]
        let values = vec![2.0f64, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];

        CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap()
    }

    #[test]
    fn test_sparse_lu() {
        let a = make_test_matrix();
        let lu = SparseLU::new(&a).unwrap();

        assert_eq!(lu.l().nrows(), 3);
        assert_eq!(lu.u().nrows(), 3);
    }

    #[test]
    fn test_sparse_lu_solve() {
        let a = make_test_matrix();
        let lu = SparseLU::new(&a).unwrap();

        let b = vec![3.0, 5.0, 3.0];
        let x = lu.solve(&b);

        // Verify A * x ≈ b
        let mut ax = [0.0; 3];
        for col in 0..3 {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                ax[a.row_indices()[idx]] += a.values()[idx] * x[col];
            }
        }

        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-10, "LU solve failed at index {i}");
        }
    }

    #[test]
    fn test_sparse_lu_determinant() {
        let a = make_test_matrix();
        let lu = SparseLU::new(&a).unwrap();

        let det = lu.determinant();

        // Expected det for tridiagonal [2,1,0; 1,3,1; 0,1,2]
        // det = 2*(3*2 - 1*1) - 1*(1*2 - 0*1) = 2*5 - 2 = 8
        assert!(
            (det - 8.0).abs() < 1e-10,
            "Determinant is {det}, expected 8.0"
        );
    }

    #[test]
    fn test_ilu0() {
        // Create a simple test matrix in CSR
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let ilu = ILU0::new(&a).unwrap();

        let b = vec![5.0, 6.0, 5.0];
        let x = ilu.apply(&b);

        // x should be approximately the solution to A*x = b
        assert_eq!(x.len(), 3);
    }

    #[test]
    fn test_permutation_sign() {
        // Identity: even
        assert_eq!(permutation_sign(&[0, 1, 2]), 1);

        // Single swap: odd
        assert_eq!(permutation_sign(&[1, 0, 2]), -1);

        // Two swaps: even
        assert_eq!(permutation_sign(&[1, 2, 0]), 1);

        // Cycle of 3: even (2 transpositions)
        assert_eq!(permutation_sign(&[2, 0, 1]), 1);
    }

    fn make_csr_test_matrix() -> CsrMatrix<f64> {
        // Same matrix as make_test_matrix but in CSR format
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }

    fn make_larger_csr_matrix() -> CsrMatrix<f64> {
        // 5x5 diagonally dominant SPD matrix
        let n = 5;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
    }

    #[test]
    fn test_ilut_basic() {
        let a = make_csr_test_matrix();
        let ilut = ILUT::new(&a, 1e-6, 10).unwrap();

        // Check that L and U have the right dimensions
        assert_eq!(ilut.l().nrows(), 3);
        assert_eq!(ilut.u().nrows(), 3);

        // Apply preconditioner
        let b = vec![5.0, 6.0, 5.0];
        let x = ilut.apply(&b);
        assert_eq!(x.len(), 3);

        // Check nnz
        assert!(ilut.nnz() > 0);
    }

    #[test]
    fn test_ilut_solve_quality() {
        let a = make_larger_csr_matrix();
        let ilut = ILUT::new(&a, 1e-10, 5).unwrap();

        // Solve A*x = b using ILUT as exact solver (with tight tolerance)
        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let x = ilut.apply(&b);

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

        // ILUT with tight tolerance should give reasonable approximation
        assert!(
            residual / b_norm < 1.0,
            "ILUT relative residual too large: {}",
            residual / b_norm
        );
    }

    #[test]
    fn test_ilut_dropping() {
        // Create a matrix with many small off-diagonal elements
        let n = 5;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    values.push(10.0);
                    col_indices.push(j);
                } else {
                    values.push(0.01); // Small value
                    col_indices.push(j);
                }
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        // With large tau, small elements should be dropped
        let ilut_drop = ILUT::new(&a, 0.1, 2).unwrap();
        // With small tau, more elements kept
        let ilut_keep = ILUT::new(&a, 1e-10, 10).unwrap();

        // Dropping version should have fewer nonzeros
        assert!(ilut_drop.nnz() <= ilut_keep.nnz());
    }

    #[test]
    fn test_ilutp_basic() {
        let a = make_csr_test_matrix();
        let ilutp = ILUTP::new(&a, 1e-6, 10, 0.5).unwrap();

        // Check dimensions
        assert_eq!(ilutp.l().nrows(), 3);
        assert_eq!(ilutp.u().nrows(), 3);
        assert_eq!(ilutp.perm().len(), 3);

        // Apply preconditioner
        let b = vec![5.0, 6.0, 5.0];
        let x = ilutp.apply(&b);
        assert_eq!(x.len(), 3);
    }

    #[test]
    fn test_ilutp_pivoting() {
        // Create a matrix that benefits from pivoting
        // First column has very small diagonal
        let values = vec![0.001, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        // ILUTP should handle this better than ILUT due to pivoting
        let _ilutp = ILUTP::new(&a, 1e-10, 10, 1.0);
        // Note: Pivoting helps with small diagonal, but success not guaranteed
    }

    #[test]
    fn test_ilut_vs_ilu0() {
        let a = make_larger_csr_matrix();

        let ilu0 = ILU0::new(&a).unwrap();
        let ilut = ILUT::new(&a, 1e-10, 10).unwrap();

        // Both should produce valid approximations
        let b = vec![1.0; 5];

        let x_ilu0 = ilu0.apply(&b);
        let x_ilut = ilut.apply(&b);

        // Both should give finite results
        assert!(x_ilu0.iter().all(|&v| v.is_finite()));
        assert!(x_ilut.iter().all(|&v| v.is_finite()));
    }
}
