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
///
/// # Positive definiteness
///
/// Column `j` fails with [`SparseCholError::ZeroDiagonal`] when its diagonal
/// entry `a_jj` is not stored (as for [`IC0`] and [`ICT`]), and with
/// [`SparseCholError::NotPositiveDefinite`] when its radicand
/// `a_jj − Σ_k l_jk²` is at or below `eps · m_j · a_jj` (a NaN, ±∞ or negative
/// `a_jj` counting as zero; a NaN radicand is reported as well), where
/// `m_j = 1 + #{k < j : l_jk ≠ 0}` is the number of terms of the radicand —
/// the rule of the crate's dense Cholesky factorization with the order `n`
/// replaced by the length of the sum it accounts for (see
/// `linalg::precond::tolerance`). Radicand and threshold scale alike and
/// `m_j` does not change, so `s·A` factors exactly when `A` does, and for
/// `s = 4^k` the factor is exactly `L(s·A) = 2^k · L(A)` with the same
/// pattern. Indices refer to the permuted matrix when a fill-reducing ordering
/// is used.
///
/// The factorization is complete: every computed entry of `L` is stored,
/// except exact zeros.
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
    /// - [`SparseCholError::NotSquare`] if `a` is not square;
    /// - [`SparseCholError::ZeroDiagonal`] if a diagonal entry is not stored;
    /// - [`SparseCholError::NotPositiveDefinite`] if a radicand fails the
    ///   relative test described on [`SparseCholesky`].
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
    ///
    /// # Errors
    ///
    /// As for [`Self::new`], with indices referring to the permuted matrix.
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
            let mut a_jj: Option<T> = None;

            for i in a_start..a_end {
                let row = a.row_indices()[i];
                if row >= j {
                    // Only use lower triangle
                    let value = a.values()[i].clone();
                    if row == j {
                        a_jj = Some(value.clone());
                    }
                    x[row] = value;
                    if !pattern[row] {
                        pattern[row] = true;
                        stack.push(row);
                    }
                }
            }

            // The original diagonal entry is the reference of the
            // positive-definiteness threshold; a diagonal that is not stored
            // is reported as such.
            let a_jj = match a_jj {
                Some(a_jj) => a_jj,
                None => return Err(SparseCholError::ZeroDiagonal { index: j }),
            };
            // Terms of the radicand: a_jj and every l_jk² subtracted below.
            let mut radicand_terms = 1usize;

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
                // Stored off-diagonal entries are nonzero, so l_jk² is
                // subtracted from the radicand.
                radicand_terms += 1;

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

            // Compute L[j,j] = sqrt(x[j]). The radicand must exceed
            // eps · m_j · a_jj, m_j its number of terms (a NaN radicand fails
            // the comparison too).
            let diag = x[j].clone();
            let radicand_tol = llt_radicand_tolerance(radicand_terms, a_jj);

            if !(diag > radicand_tol) {
                return Err(SparseCholError::NotPositiveDefinite { index: j });
            }

            let ljj = Real::sqrt(diag);

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
                    // A complete factorization keeps every computed entry;
                    // only exact zeros (cancellation) are not stored.
                    let val = x[row].clone() * ljj_inv.clone();
                    if val != T::zero() {
                        l_row_indices.push(row);
                        l_values.push(val);
                    }
                }
            }

            l_col_ptrs[j + 1] = l_values.len();
        }

        // SAFETY: `l_col_ptrs` has length n + 1 and `l_col_ptrs[j + 1]` is set
        // to the running entry count after every column j in 0..n (an error
        // returns before the matrix is built), so it is monotonic and ends at
        // `l_values.len()`; `l_row_indices` and `l_values` are pushed in pairs;
        // and the row indices of column j are the diagonal j and the rows of
        // the `stack` pattern (rows >= j of column j of A and of earlier
        // columns of L, each pushed once thanks to the `pattern` flags),
        // sorted, so they are distinct and in j..n.
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

    // SAFETY: `col_ptrs` has length n + 1 and `col_ptrs[new_j + 1]` is set to
    // the running entry count after every column `new_j` in 0..n, so it is
    // monotonic and ends at `values.len()`; `row_indices` and `values` are
    // pushed in pairs; and every row index is a `perm_inv` entry, i.e. a
    // position in 0..n (`perm_inv` has length n and is filled with the
    // indices of `perm`), sorted and deduplicated within its column.
    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

use super::precond::tolerance::{llt_radicand_tolerance, norm2};
use crate::csr::CsrMatrix;

/// Incomplete Cholesky factorization with zero fill-in (IC(0)).
///
/// Computes L such that L * L^T ≈ A with the same sparsity pattern as the
/// lower triangle of A. Used as a preconditioner for CG on SPD systems.
///
/// The matrix must be symmetric positive definite. Only the lower triangle
/// is used.
///
/// # Elimination
///
/// IC(0) is defined by `(L Lᵀ)_ij = a_ij` for every `(i, j)` in the lower
/// pattern of `A`. Row `i` is eliminated with the earlier rows in increasing
/// column order, restricted to that pattern: once `l_ij` is known, the
/// products `l_ij · l_kj` of the finished rows `k` with `j < k < i` are
/// subtracted from the entries `(i, k)` that are in the pattern (fill-in
/// elsewhere is discarded), and `l_ij²` from the diagonal. So
/// `l_ij = (a_ij − Σ_{k<j} l_ik·l_jk) / l_jj` and
/// `l_ii = sqrt(a_ii − Σ_{k<i} l_ik²)`, each sum taken over the pattern in
/// increasing `k`. The rows of `A` need not be sorted; the factor stores
/// every row of the pattern sorted by column, explicit zeros included.
///
/// # Positive definiteness
///
/// Row `i` fails with [`SparseCholError::NotPositiveDefinite`] when its
/// radicand `a_ii − Σ_j l_ij²` is at or below `eps · m_i · a_ii`, where
/// `a_ii` is the original diagonal entry (a NaN, ±∞ or negative `a_ii` counts
/// as zero) and `m_i = 1 + #{j < i : l_ij ≠ 0}` is the number of terms of the
/// radicand — the rule of the crate's dense Cholesky factorization with the
/// order `n` replaced by the length of the sum it accounts for (see
/// `linalg::precond::tolerance`). A NaN radicand is reported as well. Radicand
/// and threshold scale alike and `m_i` does not change, so `s·A` factors
/// whenever `A` does, and for `s = 4^k` the factor is exactly
/// `L(s·A) = 2^k · L(A)`.
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
    /// ```
    /// use oxiblas_sparse::csr::CsrMatrix;
    /// use oxiblas_sparse::linalg::cholesky::IC0;
    ///
    /// // Symmetric positive definite tridiagonal matrix [[4,1,0],[1,4,1],[0,1,4]].
    /// let a = CsrMatrix::new(
    ///     3,
    ///     3,
    ///     vec![0, 2, 5, 7],
    ///     vec![0, 1, 0, 1, 2, 1, 2],
    ///     vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0],
    /// )
    /// .unwrap();
    ///
    /// let ic0 = IC0::new(&a).unwrap();
    /// let b = vec![1.0, 1.0, 1.0];
    /// let x = ic0.apply(&b); // Preconditioner application
    /// assert_eq!(x.len(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// - [`SparseCholError::NotSquare`] if `a` is not square;
    /// - [`SparseCholError::ZeroDiagonal`] if a diagonal entry is not stored;
    /// - [`SparseCholError::NotPositiveDefinite`] if a radicand fails the
    ///   relative test described on [`IC0`].
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, SparseCholError> {
        if a.nrows() != a.ncols() {
            return Err(SparseCholError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            // SAFETY: the empty 0x0 CSR matrix: `row_ptrs = [0]` has length
            // nrows + 1 = 1, is monotonic and ends at nnz = 0, and the empty
            // `col_indices` and `values` have equal lengths and no indices.
            let l = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            return Ok(Self { l, n });
        }

        // Lower triangle of every row of A, sorted by column (CSR rows need
        // not be sorted). The factor keeps exactly this pattern.
        let mut row_ptrs = vec![0usize; n + 1];
        let mut col_indices: Vec<usize> = Vec::new();
        let mut values: Vec<T> = Vec::new();
        let mut row_entries: Vec<(usize, T)> = Vec::new();
        for i in 0..n {
            row_entries.clear();
            for idx in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
                let j = a.col_indices()[idx];
                if j <= i {
                    row_entries.push((j, a.values()[idx].clone()));
                }
            }
            row_entries.sort_unstable_by_key(|&(j, _)| j);
            for (j, value) in row_entries.drain(..) {
                col_indices.push(j);
                values.push(value);
            }
            row_ptrs[i + 1] = col_indices.len();
        }

        // The strictly lower part of the finished rows of L by column:
        // `l_cols[j]` lists `(k, l_kj)` for the finished rows `k > j` with
        // `l_kj ≠ 0`, in increasing `k` (rows are finished in order). The
        // elimination of row `i` reads column `j` of L for every pivot `j`.
        let mut l_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut l_diag: Vec<T> = vec![T::zero(); n];
        // Position in `values` of every column of the row being eliminated
        // (`usize::MAX` for the columns outside its pattern).
        let mut position = vec![usize::MAX; n];

        for i in 0..n {
            let row_start = row_ptrs[i];
            let row_end = row_ptrs[i + 1];
            for idx in row_start..row_end {
                position[col_indices[idx]] = idx;
            }

            // The row is sorted with columns <= i, so a stored diagonal is
            // its last entry.
            let diag_idx = position[i];
            if diag_idx == usize::MAX {
                return Err(SparseCholError::ZeroDiagonal { index: i });
            }
            let a_ii = values[diag_idx].clone();
            // Terms of the radicand: a_ii and every l_ij² subtracted below.
            let mut radicand_terms = 1usize;

            // Pivots j < i in increasing order. When pivot j is reached,
            // values[idx] holds a_ij − Σ_{k<j} l_ik·l_jk: every earlier pivot
            // k has already subtracted its product.
            for idx in row_start..diag_idx {
                let j = col_indices[idx];
                // l_jj > 0: row j passed the radicand test.
                let l_ij = values[idx].clone() / l_diag[j].clone();
                values[idx] = l_ij.clone();
                if l_ij == T::zero() {
                    continue;
                }

                // (i, k) -= l_ij · l_kj for the finished rows j < k < i whose
                // column k is in the pattern of row i (other fill-in is
                // discarded).
                for &(k, ref l_kj) in &l_cols[j] {
                    let target = position[k];
                    if target != usize::MAX {
                        values[target] = values[target].clone() - l_ij.clone() * l_kj.clone();
                    }
                }

                // Diagonal: a_ii − Σ l_ij²
                values[diag_idx] = values[diag_idx].clone() - l_ij.clone() * l_ij;
                radicand_terms += 1;
            }

            // The radicand must exceed eps · m_i · a_ii, m_i its number of
            // terms (a NaN radicand fails the comparison too).
            let radicand = values[diag_idx].clone();
            if !(radicand > llt_radicand_tolerance(radicand_terms, a_ii)) {
                return Err(SparseCholError::NotPositiveDefinite { index: i });
            }
            let l_ii = Real::sqrt(radicand);
            values[diag_idx] = l_ii.clone();
            l_diag[i] = l_ii;

            for idx in row_start..diag_idx {
                let value = values[idx].clone();
                if value != T::zero() {
                    l_cols[col_indices[idx]].push((i, value));
                }
            }
            for idx in row_start..row_end {
                position[col_indices[idx]] = usize::MAX;
            }
        }

        // SAFETY: `row_ptrs` has length n + 1 and holds running counts, so it
        // is monotonic and ends at `values.len()`; `col_indices` and `values`
        // were pushed in pairs (equal lengths); every column index is a
        // column `j <= i < n` of row `i` of the valid CSR matrix `a`, and the
        // columns of a row are distinct because `a` has no duplicate entries.
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
/// Similar to IC(0) but allows fill-in: row `i` of `L` is computed by
/// eliminating with the previous rows (right-looking within the row, so
/// fill-in appears wherever the elimination creates it), dropping small
/// entries as it goes, and keeping at most `p` off-diagonal entries.
///
/// # Dropping
///
/// When the elimination of row `i` reaches column `j`, the working entry
/// `w_j` (the eliminated value of `(L Lᵀ)_ij` before the division by `l_jj`,
/// i.e. `l_ij · l_jj`) is tested *unscaled*: `l_ij = w_j / l_jj` is dropped
/// when `|w_j| <= τ · ‖row_i(A)‖`, the 2-norm of the lower triangle of row
/// `i` of `A` (diagonal included). A dropped entry is not used to update the
/// rest of the row. This is a scale-consistent variant of Saad's ILUT
/// dropping rule (which tests the entry after the division against the same
/// relative tolerance), not Saad's rule verbatim: both sides of the test
/// scale like row `i` of `A`, so the drop decisions do not depend on the
/// scale of `A`.
///
/// `‖row_i(A)‖` is computed with an overflow-safe scaled 2-norm (see
/// `linalg::precond::tolerance`), so the rule also holds for entries close to
/// the overflow or underflow thresholds (for example `2^520 · A` or
/// `2^-600 · A`, where a plain sum of squares overflows to `∞` and drops
/// everything, or underflows to `0` and drops nothing). Exact scale
/// invariance still requires that no entry of `A` or of the factor overflows
/// or underflows.
///
/// # Positive definiteness
///
/// As for [`IC0`], row `i` fails with
/// [`SparseCholError::NotPositiveDefinite`] when its radicand
/// `a_ii − Σ_j l_ij²` (over the entries kept by the elimination) is at or
/// below `eps · m_i · a_ii`, `m_i = 1 + #{j : l_ij kept by the elimination}`
/// being the number of terms of the radicand (a NaN, ±∞ or negative original
/// `a_ii` counting as zero; a NaN radicand is reported as well), and with
/// [`SparseCholError::ZeroDiagonal`] when `a_ii` is not stored. The entries
/// removed afterwards by the fill cap `p` were subtracted, so they count. So
/// `s·A` factors whenever `A` does, and for `s = 4^k` the factor is exactly
/// `L(s·A) = 2^k · L(A)`.
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
    /// * `a` - Symmetric positive definite matrix in CSR format (only the
    ///   lower triangle is used)
    /// * `tau` - Drop tolerance: `l_ij` is dropped when its unscaled working
    ///   value `|w_j| = |l_ij| · l_jj` is at most `tau · ‖row_i(A)‖` (see
    ///   [`ICT`])
    /// * `p` - Maximum number of off-diagonal entries kept per row of `L`
    ///   (the `p` largest in magnitude; the diagonal is always kept)
    ///
    /// # Errors
    ///
    /// - [`SparseCholError::NotSquare`] if `a` is not square;
    /// - [`SparseCholError::ZeroDiagonal`] if a diagonal entry is not stored;
    /// - [`SparseCholError::NotPositiveDefinite`] if a radicand fails the
    ///   relative test described on [`ICT`].
    pub fn new(a: &CsrMatrix<T>, tau: T, p: usize) -> Result<Self, SparseCholError> {
        if a.nrows() != a.ncols() {
            return Err(SparseCholError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            // SAFETY: the empty 0x0 CSR matrix: `row_ptrs = [0]` has length
            // nrows + 1 = 1, is monotonic and ends at nnz = 0, and the empty
            // `col_indices` and `values` have equal lengths and no indices.
            let l = unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) };
            return Ok(Self { l, n });
        }

        // Storage for L (row-wise), its diagonal, and its strictly lower
        // columns: `l_cols[j]` lists `(k, l_kj)` for the finished rows
        // `k > j`, in increasing `k` (rows are finished in order). The
        // elimination of row `i` reads column `j` of L for every pivot `j`.
        let mut l_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut l_cols: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut l_diag: Vec<T> = vec![T::zero(); n];

        // Working row, its (sorted) pattern and a membership marker
        let mut w = vec![T::zero(); n];
        let mut in_pattern = vec![false; n];
        let mut w_indices: Vec<usize> = Vec::with_capacity(n);

        for i in 0..n {
            // Clear working vector
            for &j in &w_indices {
                w[j] = T::zero();
                in_pattern[j] = false;
            }
            w_indices.clear();

            // Copy row i of A (lower triangle only) into w
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];
            let mut a_ii: Option<T> = None;

            for idx in row_start..row_end {
                let j = a.col_indices()[idx];
                if j <= i {
                    let val = a.values()[idx].clone();
                    if j == i {
                        a_ii = Some(val.clone());
                    }
                    w[j] = val;
                    if !in_pattern[j] {
                        in_pattern[j] = true;
                        w_indices.push(j);
                    }
                }
            }

            let a_ii = match a_ii {
                Some(a_ii) => a_ii,
                None => return Err(SparseCholError::ZeroDiagonal { index: i }),
            };
            // Terms of the radicand: a_ii and every l_ij² subtracted below.
            let mut radicand_terms = 1usize;

            // Overflow-safe 2-norm of the lower triangle of row i of A.
            let row_norm = norm2(
                (row_start..row_end)
                    .filter(|&idx| a.col_indices()[idx] <= i)
                    .map(|idx| a.values()[idx].clone()),
            );
            let drop_tol = tau.clone() * row_norm;

            // Sort indices for processing
            w_indices.sort_unstable();

            // Eliminate the columns j < i in increasing order; fill-in enters
            // the pattern at positions between j and i, so the scan sees it.
            let mut k = 0;
            while k < w_indices.len() {
                let j = w_indices[k];
                if j >= i {
                    break;
                }
                k += 1;

                // Drop l_ij = w_j / l_jj when its contribution l_ij·l_jj = w_j
                // to row i is at most drop_tol (scale-invariant).
                if Scalar::abs(w[j].clone()) <= drop_tol {
                    w[j] = T::zero();
                    continue;
                }

                // l_ij = w_j / l_jj (l_jj > 0: row j passed the radicand test)
                let l_ij = w[j].clone() / l_diag[j].clone();
                w[j] = l_ij.clone();

                // w_k -= l_ij · l_kj for the finished rows j < k < i
                for &(row, ref l_kj) in &l_cols[j] {
                    w[row] = w[row].clone() - l_ij.clone() * l_kj.clone();
                    if !in_pattern[row] {
                        // Fill-in: row > j, so it belongs after position k - 1.
                        in_pattern[row] = true;
                        let offset = match w_indices[k..].binary_search(&row) {
                            Ok(pos) | Err(pos) => pos,
                        };
                        w_indices.insert(k + offset, row);
                    }
                }

                // Update diagonal: w[i] = w[i] - l_ij^2 (l_ij ≠ 0: it passed
                // the drop test, |w_j| > drop_tol >= 0)
                w[i] = w[i].clone() - l_ij.clone() * l_ij;
                radicand_terms += 1;
            }

            // Diagonal: the radicand must exceed eps · m_i · a_ii, m_i its
            // number of terms (a NaN radicand fails the comparison too).
            let radicand = w[i].clone();
            if !(radicand > llt_radicand_tolerance(radicand_terms, a_ii)) {
                return Err(SparseCholError::NotPositiveDefinite { index: i });
            }
            let l_ii = Real::sqrt(radicand);

            // Off-diagonal entries kept by the elimination (dropped ones are
            // exactly zero).
            let mut l_row: Vec<(usize, T)> = Vec::new();
            for &j in &w_indices {
                if j >= i {
                    break;
                }
                let val = w[j].clone();
                if val != T::zero() {
                    l_row.push((j, val));
                }
            }

            // Keep only the p largest off-diagonal entries
            if l_row.len() > p {
                l_row.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                l_row.truncate(p);
                l_row.sort_by_key(|(col, _)| *col);
            }

            for (col, val) in &l_row {
                l_cols[*col].push((i, val.clone()));
            }
            l_row.push((i, l_ii.clone()));
            l_diag[i] = l_ii;
            l_rows[i] = l_row;
        }

        // Convert to CSR format
        let l = Self::rows_to_csr(n, n, &l_rows);

        Ok(Self { l, n })
    }

    /// Converts row-wise storage to CSR.
    fn rows_to_csr(nrows: usize, ncols: usize, rows: &[Vec<(usize, T)>]) -> CsrMatrix<T> {
        let mut row_ptrs = Vec::with_capacity(nrows + 1);
        row_ptrs.push(0usize);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Exactly `nrows` rows (a missing row is empty), so every pointer is
        // the running count after its row.
        for i in 0..nrows {
            if let Some(row) = rows.get(i) {
                for (col, val) in row {
                    col_indices.push(*col);
                    values.push(val.clone());
                }
            }
            row_ptrs.push(col_indices.len());
        }

        // SAFETY: `row_ptrs` holds 0 followed by the running entry count after
        // each of the `nrows` rows, so it has length nrows + 1, is monotonic
        // and ends at `values.len()`; `col_indices` and `values` are pushed in
        // pairs, so their lengths are equal; and the only caller, `new`, passes
        // rows whose columns are distinct and below `ncols = n`: the kept
        // off-diagonal entries `j < i` (distinct pattern positions of the
        // working row) followed by the diagonal `i`.
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
