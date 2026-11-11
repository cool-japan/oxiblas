//! Sparse LU decomposition.
//!
//! Provides:
//! - [`SparseLU`]: complete LU factorization with partial (row) pivoting
//! - [`ILU0`]: incomplete LU factorization with zero fill-in
//! - [`ILUT`]: incomplete LU factorization with threshold dropping and a fill
//!   limit (Saad's ILUT)
//! - [`ILUTP`]: ILUT with threshold column pivoting
//!
//! # Pivot and drop tolerances
//!
//! Every test is relative to the input, so a factorization of `s·A`, `s` a
//! power of two, succeeds exactly when the factorization of `A` does, with the
//! same permutation and sparsity pattern and exactly scaled factors
//! (`L(s·A) = L(A)`, `U(s·A) = s·U(A)`), as long as nothing overflows or
//! underflows:
//!
//! - a pivot of column `j` is treated as zero, and reported as
//!   [`SparseLUError::Singular`], when its magnitude is at or below
//!   `eps · m · s_j`, where `s_j = max_i |a_ij|` over the finite entries of
//!   the original column `j` (for [`ILUTP`], the original column the pivot
//!   comes from) and `m` is the number of terms of the pivot: the original
//!   entry plus every product `l_ik · u_kj` actually subtracted from it during
//!   the elimination. This is the rule of the crate's dense LU factorization
//!   with the order `n` replaced by the length of the sum it accounts for
//!   (see `linalg::precond::tolerance`); a global `n` rejects well-conditioned
//!   pivots of large matrices with short rows. A NaN pivot is not reported;
//!   it propagates into the factors;
//! - [`SparseLU`] and [`ILU0`] drop nothing but exact zeros (ILU(0) keeps the
//!   pattern of `A`); [`ILUT`] and [`ILUTP`] drop by a scale-consistent
//!   variant of Saad's rule, relative to the overflow-safe 2-norm of the row
//!   of `A` being factorized.

use super::precond::tolerance::{lu_column_scales, norm2, pivot_tolerance};
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
    /// The matrix is singular to working precision: a pivot is missing, zero,
    /// or at or below `eps · m` times the largest finite magnitude of its
    /// original column, `m` being the number of terms of the pivot (see the
    /// documentation of each factorization).
    Singular {
        /// Elimination step (row of `U`) where the zero pivot was found.
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

/// Per-column pivot references `max_i |a_ij|` of a square CSR matrix (finite
/// entries only).
fn csr_column_scales<T: Scalar<Real = T> + Clone + Field + Real>(a: &CsrMatrix<T>) -> Vec<T> {
    lu_column_scales(
        a.nrows(),
        a.col_indices()
            .iter()
            .copied()
            .zip(a.values().iter().cloned()),
    )
}

/// Overflow-safe 2-norm of row `i` of `a` (all stored entries, in storage
/// order), the reference of the ILUT drop rule.
fn csr_row_norm<T: Scalar<Real = T> + Clone + Field + Real>(a: &CsrMatrix<T>, i: usize) -> T {
    norm2(
        a.values()[a.row_ptrs()[i]..a.row_ptrs()[i + 1]]
            .iter()
            .cloned(),
    )
}

/// Sorts `row` by decreasing magnitude, keeps the `p` largest entries, and
/// sorts them back by column (no-op when the row holds at most `p` entries).
fn keep_largest<T: Scalar<Real = T> + Clone + Field + Real>(row: &mut Vec<(usize, T)>, p: usize) {
    if row.len() > p {
        row.sort_by(|a, b| {
            Scalar::abs(b.1.clone())
                .partial_cmp(&Scalar::abs(a.1.clone()))
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        row.truncate(p);
        row.sort_by_key(|(col, _)| *col);
    }
}

/// Sparse LU factorization with partial pivoting.
///
/// Computes P * A = L * U where:
/// - P is a permutation matrix (row pivoting)
/// - L is lower triangular with unit diagonal
/// - U is upper triangular
///
/// # Singularity
///
/// At step `k` the pivot is the entry of largest magnitude among the rows not
/// yet pivoted in column `k` (a NaN entry is never chosen). The matrix is
/// reported as [`SparseLUError::Singular`] `{ row: k }` when that magnitude is
/// at or below `eps · m_k · s_k`, where `s_k = max_i |a_ik|` is taken over the
/// finite entries of the original column `k` and
/// `m_k = 1 + #{j < k : l_kj ≠ 0 and u_jk ≠ 0}` is the number of terms of the
/// chosen pivot `a − Σ_j l_kj · u_jk` — the rule of the crate's dense LU
/// factorization with the order `n` replaced by the length of the sum it
/// accounts for (rows are swapped, columns never move). Pivot and threshold
/// scale alike and `m_k` does not change: for `s` a power of two, `s·A`
/// factors exactly when `A` does, with the same permutation, `L(s·A) = L(A)`
/// and `U(s·A) = s·U(A)`; the same holds for column scalings `A·diag(c)` by
/// powers of two, which leave `L` unchanged and scale column `j` of `U` by
/// `c_j`.
///
/// The factorization is complete: every computed entry of `L` and `U` is
/// stored, except exact zeros.
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
    ///
    /// # Errors
    ///
    /// - [`SparseLUError::NotSquare`] if `a` is not square;
    /// - [`SparseLUError::Singular`] if a pivot fails the relative test
    ///   described on [`SparseLU`].
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();

        // Per-column pivot references max_i |a_ij| of the original matrix
        // (finite entries only).
        let col_scale = lu_column_scales(
            n,
            (0..n).flat_map(|j| {
                (a.col_ptrs()[j]..a.col_ptrs()[j + 1]).map(move |idx| (j, a.values()[idx].clone()))
            }),
        );

        // Initialize permutation as identity
        let mut perm: Vec<usize> = (0..n).collect();

        // Work with dense columns for simplicity (can be optimized for truly sparse case).
        // `l_data[i]` holds the multipliers of the row at position `i`.
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

            // Terms of the pivot: the original entry and every product
            // l_kj · u_jk with two nonzero factors that the elimination
            // subtracted from it (`l_data[max_row]` holds the multipliers of
            // the pivot row until the swap below).
            let pivot_terms = 1
                + (0..k)
                    .filter(|&j| l_data[max_row][j] != T::zero() && u_data[j][k] != T::zero())
                    .count();

            // Singular when the pivot is at or below eps · m_k times the
            // reference of the original column k (scale-invariant).
            if max_val <= pivot_tolerance(pivot_terms, col_scale[k].clone()) {
                return Err(SparseLUError::Singular { row: k });
            }

            // Swap rows k and max_row in the permutation. The multipliers
            // already computed for these two positions (columns 0..k) move
            // with their rows.
            perm.swap(k, max_row);
            l_data.swap(k, max_row);

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
///
/// # Zero pivots
///
/// Row `i` fails with [`SparseLUError::Singular`] `{ row: i }` when its
/// diagonal entry is not stored, or when its pivot `u_ii` (after the
/// elimination of row `i`) has magnitude at or below `eps · m_i · s_i`, where
/// `s_i = max_k |a_ki|` over the finite entries of column `i` of `A` and
/// `m_i = 1 + #{j < i : l_ij ≠ 0 and u_ji ≠ 0}` (over the pattern) is the
/// number of terms of the pivot — the per-column rule of the crate's dense LU
/// with the order `n` replaced by the length of the sum it accounts for.
/// Every pivot is tested, including those of rows that no later row
/// eliminates with. A NaN pivot is not reported; it propagates. Only exact
/// zeros are dropped: the factors keep the pattern of `A`. For `s` a power of
/// two, `s·A` factors exactly when `A` does, with `L(s·A) = L(A)` and
/// `U(s·A) = s·U(A)`, so the preconditioner of `s·A` is exactly `1/s` times
/// that of `A`.
///
/// Row `i` is eliminated with its stored entries `j < i` in increasing column
/// order, whatever their storage order in the row (CSR rows need not be
/// sorted); the factor keeps the storage order of `A`.
#[derive(Debug, Clone)]
pub struct ILU0<T: Scalar> {
    /// Combined L + U - I stored in CSR format (same pattern as input).
    lu: CsrMatrix<T>,
    /// Position of the diagonal entry of every row in `lu`.
    diag_pos: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> ILU0<T> {
    /// Computes the ILU(0) factorization.
    ///
    /// The input matrix should be in CSR format.
    ///
    /// # Errors
    ///
    /// - [`SparseLUError::NotSquare`] if `a` is not square;
    /// - [`SparseLUError::Singular`] if a diagonal entry is missing or a pivot
    ///   fails the relative test described on [`ILU0`].
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();

        // Per-column pivot references of the original matrix.
        let col_scale = csr_column_scales(a);

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
        let mut diag_pos: Vec<usize> = Vec::with_capacity(n);
        // The entries j < i of the row being eliminated, as (j, position),
        // sorted by column.
        let mut lower: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            let row_start = row_ptrs[i];
            let row_end = row_ptrs[i + 1];
            lower.clear();
            lower.extend(
                (row_start..row_end)
                    .map(|idx| (col_indices[idx], idx))
                    .filter(|&(j, _)| j < i),
            );
            lower.sort_unstable_by_key(|&(j, _)| j);

            // A missing diagonal entry is a zero pivot.
            let diag_idx = match col_to_idx[i].get(&i) {
                Some(&idx) => idx,
                None => return Err(SparseLUError::Singular { row: i }),
            };
            // Terms of the pivot: a_ii and every product l_ij · u_ji with two
            // nonzero factors subtracted from it below.
            let mut pivot_terms = 1usize;

            // For each j < i where A[i,j] != 0, in increasing j
            for &(j, idx) in &lower {
                // U[j,j] passed the pivot test when row j was finished.
                let ujj = values[diag_pos[j]].clone();

                // A[i,j] = A[i,j] / A[j,j]
                values[idx] = values[idx].clone() / ujj;

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
                        if i_k_idx == diag_idx && aij != T::zero() && values[j_idx] != T::zero() {
                            pivot_terms += 1;
                        }
                        // A[i,k] = A[i,k] - A[i,j] * A[j,k]
                        values[i_k_idx] =
                            values[i_k_idx].clone() - aij.clone() * values[j_idx].clone();
                    }
                    // If A[i,k] doesn't exist, we drop the fill-in (ILU(0))
                }
            }

            // Pivot test against eps · m_i times the reference of the
            // original column i (a NaN pivot fails the comparison and
            // propagates).
            if Scalar::abs(values[diag_idx].clone())
                <= pivot_tolerance(pivot_terms, col_scale[i].clone())
            {
                return Err(SparseLUError::Singular { row: i });
            }
            diag_pos.push(diag_idx);
        }

        // SAFETY: `row_ptrs`, `col_indices` and `values` are unmodified-length
        // copies of the arrays of the valid CSR matrix `a` (the elimination
        // only overwrote entries of `values`), so `row_ptrs` has length
        // nrows + 1 = n + 1 and is monotonic, `col_indices` and `values` have
        // equal lengths, and every column index is below `a.ncols() = n` (`a`
        // is square, checked above).
        let lu = unsafe { CsrMatrix::new_unchecked(n, n, row_ptrs, col_indices, values) };

        Ok(Self { lu, diag_pos })
    }

    /// Returns the combined L + U - I matrix.
    pub fn lu(&self) -> &CsrMatrix<T> {
        &self.lu
    }

    /// Applies the preconditioner: solves (LU) * x = b.
    ///
    /// `L` is the unit lower triangle whose multipliers are stored below the
    /// diagonal of [`Self::lu`], and `U` the upper triangle, diagonal
    /// included.
    pub fn apply(&self, b: &[T]) -> Vec<T> {
        let n = self.lu.nrows();
        assert_eq!(b.len(), n, "RHS length must match matrix size");
        let row_ptrs = self.lu.row_ptrs();
        let col_indices = self.lu.col_indices();
        let values = self.lu.values();

        // Forward substitution: L * y = b (unit diagonal)
        let mut x = b.to_vec();
        for i in 0..n {
            let mut sum = x[i].clone();
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[idx];
                if j < i {
                    sum = sum - values[idx].clone() * x[j].clone();
                }
            }
            x[i] = sum;
        }

        // Backward substitution: U * x = y
        for i in (0..n).rev() {
            let mut sum = x[i].clone();
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[idx];
                if j > i {
                    sum = sum - values[idx].clone() * x[j].clone();
                }
            }
            x[i] = sum / values[self.diag_pos[i]].clone();
        }

        x
    }
}

/// Incomplete LU factorization with threshold (ILUT).
///
/// Saad's ILUT: row `i` of `L` and `U` is computed by eliminating row `i` of
/// `A` with the previous rows of `U` in increasing column order (fill-in
/// enters wherever the elimination creates it), dropping small entries as it
/// goes, and keeping at most `p` entries in each row of `L` and `p`
/// off-diagonal entries in each row of `U` (the largest in magnitude; the
/// diagonal is always kept).
///
/// # Dropping
///
/// With `τ_i = τ · ‖row_i(A)‖₂` (the 2-norm of row `i` of `A`):
///
/// - a multiplier `l_ij` is dropped when its contribution `|l_ij · u_jj|` to
///   row `i` — the unscaled entry `w_j` it eliminates from the working row,
///   tested before the division by `u_jj` — is at most `τ_i`; a dropped
///   multiplier is not used to update the row. This is a scale-consistent
///   variant of Saad's rule (which tests `l_ij` itself), stated for the
///   contribution so that both sides scale like row `i` of `A`;
/// - an entry `u_ij`, `j > i`, is dropped when `|u_ij| <= τ_i`.
///
/// Both tests are invariant under scaling `A` (and under scaling the rows of
/// `A`), so the pattern of the factors does not depend on the scale of `A`.
/// `‖row_i(A)‖₂` is computed with an overflow-safe scaled 2-norm (see
/// `linalg::precond::tolerance`), so this holds also for entries close to the
/// overflow or underflow thresholds (for example `2^520 · A` or
/// `2^-600 · A`), as long as no entry of `A` or of the factors overflows or
/// underflows.
///
/// # Zero pivots
///
/// Row `i` fails with [`SparseLUError::Singular`] `{ row: i }` when its
/// diagonal is not in the pattern of the eliminated row, or when the pivot
/// `u_ii` has magnitude at or below `eps · m_i · s_i`, where
/// `s_i = max_k |a_ki|` over the finite entries of column `i` of `A` and
/// `m_i` is the number of terms of the pivot: one for the entry of row `i` of
/// `A`, plus one per product `l_ij · u_ji` subtracted from it by the
/// elimination (the rule of the crate's dense LU, with the order `n` replaced
/// by the length of the sum it accounts for). A NaN pivot is not reported.
/// For `s` a power of two, `s·A` factors exactly when `A` does, with the same
/// patterns, `L(s·A) = L(A)` and `U(s·A) = s·U(A)`.
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
    /// * `tau` - Drop tolerance, relative to the 2-norm of each row of `a`
    ///   (see [`ILUT`]); `0` drops only exact zeros
    /// * `p` - Maximum number of entries kept in each row of `L`, and of
    ///   off-diagonal entries in each row of `U`
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_sparse::csr::CsrMatrix;
    /// use oxiblas_sparse::linalg::ILUT;
    ///
    /// // Diagonally dominant tridiagonal matrix [[4,1,0],[1,4,1],[0,1,4]].
    /// let a = CsrMatrix::new(
    ///     3,
    ///     3,
    ///     vec![0, 2, 5, 7],
    ///     vec![0, 1, 0, 1, 2, 1, 2],
    ///     vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0],
    /// )
    /// .unwrap();
    ///
    /// let ilut = ILUT::new(&a, 1e-3, 10).unwrap();
    /// let x = ilut.apply(&[1.0, 1.0, 1.0]);
    /// assert_eq!(x.len(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// - [`SparseLUError::NotSquare`] if `a` is not square;
    /// - [`SparseLUError::Singular`] if a pivot is missing or fails the
    ///   relative test described on [`ILUT`].
    pub fn new(a: &CsrMatrix<T>, tau: T, p: usize) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            return Ok(Self {
                l: empty_csr(),
                u: empty_csr(),
                n,
            });
        }

        // Per-column pivot references of the original matrix.
        let col_scale = csr_column_scales(a);

        // Rows of L (multipliers) and the off-diagonal entries of the rows of
        // U, sorted by column, and the diagonal of U.
        let mut l_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut u_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut u_diag: Vec<T> = vec![T::zero(); n];

        // Working row, its sorted pattern and a membership marker
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

            // Copy row i of A into w
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];

            for idx in row_start..row_end {
                let j = a.col_indices()[idx];
                let val = a.values()[idx].clone();
                w[j] = val;
                if !in_pattern[j] {
                    in_pattern[j] = true;
                    w_indices.push(j);
                }
            }

            let drop_tol = tau.clone() * csr_row_norm(a, i);
            // Terms of the pivot: the entry of A and every product
            // l_ij · u_ji subtracted from it below.
            let mut pivot_terms = 1usize;

            // Sort indices for processing
            w_indices.sort_unstable();

            // Eliminate the columns j < i in increasing order; fill-in enters
            // the pattern after position k - 1, so the scan sees it.
            let mut k = 0;
            while k < w_indices.len() {
                let j = w_indices[k];
                if j >= i {
                    break;
                }
                k += 1;

                // Drop l_ij = w_j / u_jj when its contribution l_ij·u_jj = w_j
                // to row i is at most drop_tol (scale-invariant).
                if Scalar::abs(w[j].clone()) <= drop_tol {
                    w[j] = T::zero();
                    continue;
                }

                // l_ij = w_j / u_jj (u_jj passed the pivot test of row j)
                let l_ij = w[j].clone() / u_diag[j].clone();
                w[j] = l_ij.clone();

                // w_c -= l_ij · u_jc for the off-diagonal entries of row j of U
                // (l_ij ≠ 0 passed the drop test and stored u_jc ≠ 0 passed
                // theirs, so a product hitting the diagonal is a pivot term)
                for &(col, ref u_val) in &u_rows[j] {
                    if col == i {
                        pivot_terms += 1;
                    }
                    w[col] = w[col].clone() - l_ij.clone() * u_val.clone();
                    if !in_pattern[col] {
                        // Fill-in: col > j, so it belongs after position k - 1.
                        in_pattern[col] = true;
                        let offset = match w_indices[k..].binary_search(&col) {
                            Ok(pos) | Err(pos) => pos,
                        };
                        w_indices.insert(k + offset, col);
                    }
                }
            }

            // L part: the multipliers kept by the elimination (dropped ones
            // are exactly zero), at most p of them.
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
            keep_largest(&mut l_row, p);

            // U part: off-diagonal entries above the drop tolerance, at most
            // p of them (the diagonal is kept separately).
            let mut u_row: Vec<(usize, T)> = Vec::new();
            for &j in &w_indices {
                if j <= i {
                    continue;
                }
                let val = w[j].clone();
                if !(Scalar::abs(val.clone()) <= drop_tol) {
                    u_row.push((j, val));
                }
            }
            keep_largest(&mut u_row, p);

            // Pivot test against eps · m_i times the reference of the
            // original column i.
            if !in_pattern[i]
                || Scalar::abs(w[i].clone()) <= pivot_tolerance(pivot_terms, col_scale[i].clone())
            {
                return Err(SparseLUError::Singular { row: i });
            }

            l_rows[i] = l_row;
            u_rows[i] = u_row;
            u_diag[i] = w[i].clone();
        }

        // Convert to CSR format (the diagonal leads each row of U).
        let u_full: Vec<Vec<(usize, T)>> = u_rows
            .into_iter()
            .zip(u_diag)
            .enumerate()
            .map(|(i, (row, diag))| {
                let mut full = Vec::with_capacity(row.len() + 1);
                full.push((i, diag));
                full.extend(row);
                full
            })
            .collect();
        let l = Self::rows_to_csr(n, n, &l_rows);
        let u = Self::rows_to_csr(n, n, &u_full);

        Ok(Self { l, u, n })
    }

    /// Converts row-wise storage to CSR.
    ///
    /// Every row must hold distinct column indices below `ncols`.
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
        // pairs, so their lengths are equal; and the callers (`ILUT::new` and
        // `ILUTP::new`, with `nrows = ncols = n`) pass rows whose columns are
        // distinct and below n: positions of the working row (distinct by its
        // pattern flags) plus the diagonal, renumbered by ILUTP through the
        // permutation `perm_inv` of 0..n.
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
/// [`ILUT`] with threshold column pivoting (Saad's ILUTP): it computes
/// `A·Q ≈ L·U`, where `Q` permutes the columns (`A·Q·e_i = A·e_perm[i]`).
///
/// # Pivoting
///
/// After the elimination of row `i`, let `d` be the magnitude of the entry in
/// the pivot position (zero if it is not in the pattern). Among the later
/// positions of the row, the entry of largest magnitude `t` with
/// `t · pivot_tol > d` (the first one on ties) becomes the pivot, and the two
/// columns are swapped. So `pivot_tol = 0` never pivots, `pivot_tol = 1`
/// always moves the largest entry of the row onto the diagonal, and values in
/// between pivot only when the diagonal is smaller than `pivot_tol` times the
/// candidate. The comparison is between entries of one row, so the pivoting
/// decisions do not depend on the scale of `A`.
///
/// The rows of `L` and `U` are kept in the original column numbering while
/// the factorization runs, so a swap never leaves an earlier row of `U` in a
/// stale numbering; they are renumbered with the final permutation at the
/// end.
///
/// # Dropping and zero pivots
///
/// Dropping follows [`ILUT`] (relative to the overflow-safe 2-norm of each
/// row of `A`). Row `i` fails with [`SparseLUError::Singular`] `{ row: i }`
/// when the pivot position is not in the pattern, or when the pivot has
/// magnitude at or below `eps · m · s_c`, where `s_c` is the largest finite
/// magnitude of the original column `c = perm[i]` the pivot comes from and
/// `m` is the number of terms of the pivot (one for the entry of row `i` of
/// `A` in column `c`, plus one per product subtracted from it by the
/// elimination; see [`ILUT`]). A NaN pivot is not reported. For `s` a power
/// of two, `s·A` factors exactly when `A` does, with the same permutation and
/// patterns, `L(s·A) = L(A)` and `U(s·A) = s·U(A)`.
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
    /// * `tau` - Drop tolerance, relative to the 2-norm of each row of `a`
    ///   (see [`ILUT`])
    /// * `p` - Maximum number of entries kept in each row of `L`, and of
    ///   off-diagonal entries in each row of `U`
    /// * `pivot_tol` - Pivoting tolerance (0.0 for no pivoting, 1.0 for full
    ///   pivoting; see [`ILUTP`])
    ///
    /// # Errors
    ///
    /// - [`SparseLUError::NotSquare`] if `a` is not square;
    /// - [`SparseLUError::Singular`] if a pivot is missing or fails the
    ///   relative test described on [`ILUTP`].
    pub fn new(a: &CsrMatrix<T>, tau: T, p: usize, pivot_tol: T) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            return Ok(Self {
                l: empty_csr(),
                u: empty_csr(),
                perm: vec![],
                perm_inv: vec![],
                n,
            });
        }

        // Pivot references of the original columns.
        let col_scale = csr_column_scales(a);

        // Position -> original column, and its inverse.
        let mut perm: Vec<usize> = (0..n).collect();
        let mut perm_inv: Vec<usize> = (0..n).collect();

        // Rows of L (multipliers) and the off-diagonal entries of the rows of
        // U, in original column numbering, and the pivots (row k's pivot sits
        // in original column perm[k], which no later swap moves).
        let mut l_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut u_rows: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut u_diag: Vec<T> = vec![T::zero(); n];

        // Working row (by original column), its pattern, a membership marker,
        // and the sorted positions (< i) of its entries left to eliminate.
        let mut w = vec![T::zero(); n];
        let mut in_pattern = vec![false; n];
        let mut pattern: Vec<usize> = Vec::with_capacity(n);
        let mut lower: Vec<usize> = Vec::with_capacity(n);
        // Products subtracted from each entry of the working row (by
        // original column): the pivot column is only known after the
        // elimination, and its count gives the number of terms of the pivot.
        let mut updates = vec![0usize; n];

        for i in 0..n {
            // Clear working vector
            for &c in &pattern {
                w[c] = T::zero();
                in_pattern[c] = false;
                updates[c] = 0;
            }
            pattern.clear();
            lower.clear();

            // Copy row i of A into w
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];

            for idx in row_start..row_end {
                let c = a.col_indices()[idx];
                let val = a.values()[idx].clone();
                w[c] = val;
                if !in_pattern[c] {
                    in_pattern[c] = true;
                    pattern.push(c);
                    if perm_inv[c] < i {
                        lower.push(perm_inv[c]);
                    }
                }
            }

            let drop_tol = tau.clone() * csr_row_norm(a, i);

            lower.sort_unstable();

            // Eliminate with the rows of U in increasing position order.
            // Row q of U holds original columns at positions > q only, so
            // fill-in in the L part belongs after position k - 1.
            let mut k = 0;
            while k < lower.len() {
                let q = lower[k];
                k += 1;
                let c = perm[q];

                // Drop l_iq when its contribution |l_iq · u_qq| = |w_c| to row
                // i is at most drop_tol (scale-invariant).
                if Scalar::abs(w[c].clone()) <= drop_tol {
                    w[c] = T::zero();
                    continue;
                }

                // l_iq = w_c / u_qq (u_qq passed the pivot test of row q)
                let l_iq = w[c].clone() / u_diag[q].clone();
                w[c] = l_iq.clone();

                // (l_iq ≠ 0 passed the drop test and stored u_qc ≠ 0 passed
                // theirs, so every product is a term of the entry it updates)
                for &(col, ref u_val) in &u_rows[q] {
                    w[col] = w[col].clone() - l_iq.clone() * u_val.clone();
                    updates[col] += 1;
                    if !in_pattern[col] {
                        in_pattern[col] = true;
                        pattern.push(col);
                        let pos = perm_inv[col];
                        if pos < i {
                            let offset = match lower[k..].binary_search(&pos) {
                                Ok(o) | Err(o) => o,
                            };
                            lower.insert(k + offset, pos);
                        }
                    }
                }
            }

            // L part: the multipliers kept by the elimination (dropped ones
            // are exactly zero), at most p of them.
            let mut l_row: Vec<(usize, T)> = Vec::new();
            for &q in &lower {
                let c = perm[q];
                let val = w[c].clone();
                if val != T::zero() {
                    l_row.push((q, val));
                }
            }
            keep_largest(&mut l_row, p);
            let l_row: Vec<(usize, T)> = l_row.into_iter().map(|(q, v)| (perm[q], v)).collect();

            // Column pivoting among the later positions of the row (the
            // pattern is in insertion order, so scan the candidates sorted by
            // position, which makes ties go to the first position).
            let mut upper: Vec<usize> = pattern
                .iter()
                .map(|&c| perm_inv[c])
                .filter(|&pos| pos > i)
                .collect();
            upper.sort_unstable();
            let pivot_col = perm[i];
            let d = if in_pattern[pivot_col] {
                Scalar::abs(w[pivot_col].clone())
            } else {
                T::zero()
            };
            let mut best: Option<usize> = None;
            let mut best_val = d.clone();
            for &pos in &upper {
                let t = Scalar::abs(w[perm[pos]].clone());
                if t > best_val && t.clone() * pivot_tol.clone() > d {
                    best_val = t;
                    best = Some(pos);
                }
            }
            if let Some(pos) = best {
                let (ci, cp) = (perm[i], perm[pos]);
                perm.swap(i, pos);
                perm_inv[ci] = pos;
                perm_inv[cp] = i;
            }
            let pivot_col = perm[i];

            // U part: off-diagonal entries above the drop tolerance, at most
            // p of them (scanned by position for deterministic ties).
            let mut u_row: Vec<(usize, T)> = Vec::new();
            let mut positions: Vec<usize> = pattern
                .iter()
                .map(|&c| perm_inv[c])
                .filter(|&pos| pos > i)
                .collect();
            positions.sort_unstable();
            for pos in positions {
                let val = w[perm[pos]].clone();
                if !(Scalar::abs(val.clone()) <= drop_tol) {
                    u_row.push((pos, val));
                }
            }
            keep_largest(&mut u_row, p);
            let u_row: Vec<(usize, T)> = u_row.into_iter().map(|(pos, v)| (perm[pos], v)).collect();

            // Pivot test against eps · m times the reference of the original
            // column the pivot comes from, m = 1 + the products subtracted
            // from the pivot entry.
            let pivot_terms = 1 + updates[pivot_col];
            if !in_pattern[pivot_col]
                || Scalar::abs(w[pivot_col].clone())
                    <= pivot_tolerance(pivot_terms, col_scale[pivot_col].clone())
            {
                return Err(SparseLUError::Singular { row: i });
            }

            l_rows[i] = l_row;
            u_rows[i] = u_row;
            u_diag[i] = w[pivot_col].clone();
        }

        // Renumber the columns with the final permutation: positions below
        // and at i are final after step i, so L stays strictly lower and U
        // upper.
        let renumber = |row: Vec<(usize, T)>| -> Vec<(usize, T)> {
            let mut row: Vec<(usize, T)> = row.into_iter().map(|(c, v)| (perm_inv[c], v)).collect();
            row.sort_by_key(|(col, _)| *col);
            row
        };
        let l_rows: Vec<Vec<(usize, T)>> = l_rows.into_iter().map(renumber).collect();
        let u_full: Vec<Vec<(usize, T)>> = u_rows
            .into_iter()
            .zip(u_diag)
            .enumerate()
            .map(|(i, (row, diag))| {
                let mut full = Vec::with_capacity(row.len() + 1);
                full.push((i, diag));
                full.extend(renumber(row));
                full
            })
            .collect();

        let l = ILUT::rows_to_csr(n, n, &l_rows);
        let u = ILUT::rows_to_csr(n, n, &u_full);

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

/// The empty `0 x 0` CSR matrix.
fn empty_csr<T: Scalar + Clone + Field>() -> CsrMatrix<T> {
    // SAFETY: `row_ptrs = [0]` has length nrows + 1 = 1, is monotonic and
    // ends at nnz = 0, and the empty `col_indices` and `values` have equal
    // lengths and no indices.
    unsafe { CsrMatrix::new_unchecked(0, 0, vec![0], vec![], vec![]) }
}

/// Converts a dense lower triangular matrix to CSC format, keeping every
/// entry on or below the diagonal except exact zeros off the diagonal.
fn dense_to_csc_lower<T: Scalar + Clone + Field>(data: &[Vec<T>]) -> CscMatrix<T> {
    let n = data.len();
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        for i in j..n {
            let val = data[i][j].clone();
            if i == j || val != T::zero() {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    // SAFETY: `col_ptrs` has length n + 1 and `col_ptrs[j + 1]` is the running
    // entry count after every column j in 0..n, so it is monotonic and ends at
    // `values.len()`; `row_indices` and `values` are pushed in pairs; and the
    // row indices of column j come from the loop `i in j..n`, so they are
    // distinct, increasing and below n.
    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

/// Converts a dense upper triangular matrix to CSC format, keeping every
/// entry on or above the diagonal except exact zeros off the diagonal.
fn dense_to_csc_upper<T: Scalar + Clone + Field>(data: &[Vec<T>]) -> CscMatrix<T> {
    let n = data.len();
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        for i in 0..=j {
            let val = data[i][j].clone();
            if i == j || val != T::zero() {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    // SAFETY: `col_ptrs` has length n + 1 and `col_ptrs[j + 1]` is the running
    // entry count after every column j in 0..n, so it is monotonic and ends at
    // `values.len()`; `row_indices` and `values` are pushed in pairs; and the
    // row indices of column j come from the loop `i in 0..=j` with j < n, so
    // they are distinct, increasing and below n.
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
