//! Sparse QR decomposition for overdetermined and underdetermined sparse systems.
//!
//! Uses left-looking column-by-column Givens rotations with COLAMD column ordering.
//! Factorization: A*P = Q*R, where
//! - P is the COLAMD column permutation
//! - Q is orthogonal (stored implicitly as a sequence of Givens rotations)
//! - R is upper triangular (stored as [`CscMatrix`])
//!
//! # Usage
//!
//! ```rust,ignore
//! use oxiblas_sparse::linalg::{SparseQr, SparseQrConfig};
//!
//! let qr = SparseQr::compute(&a).unwrap();
//! let x = qr.solve_least_squares(&b).unwrap();
//! ```

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use crate::linalg::ordering::colamd;
use oxiblas_core::scalar::Scalar;

/// Error type for sparse QR decomposition.
#[derive(Debug, Clone)]
pub enum SparseQrError {
    /// Matrix is singular (zero diagonal in R).
    SingularMatrix,
    /// RHS vector has incompatible length.
    IncompatibleDimensions {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension provided.
        got: usize,
    },
    /// Matrix data is structurally invalid.
    InvalidMatrix(String),
    /// Numerical failure during factorization.
    NumericalFailure(String),
    /// Matrix is rank deficient.
    RankDeficient {
        /// Detected rank.
        rank: usize,
        /// Number of columns.
        n: usize,
    },
}

impl std::fmt::Display for SparseQrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingularMatrix => write!(f, "Matrix is singular: zero diagonal in R"),
            Self::IncompatibleDimensions { expected, got } => {
                write!(f, "Incompatible dimensions: expected {expected}, got {got}")
            }
            Self::InvalidMatrix(msg) => write!(f, "Invalid matrix: {msg}"),
            Self::NumericalFailure(msg) => write!(f, "Numerical failure: {msg}"),
            Self::RankDeficient { rank, n } => {
                write!(f, "Matrix is rank deficient: rank {rank} of {n}")
            }
        }
    }
}

impl std::error::Error for SparseQrError {}

/// Configuration for sparse QR factorization.
#[derive(Debug, Clone)]
pub struct SparseQrConfig {
    /// Whether to use COLAMD column ordering (default: `true`).
    pub column_ordering: bool,
    /// Threshold for numerical zero during factorization (default: `1e-14`).
    pub drop_tol: f64,
    /// Minimum acceptable diagonal pivot in R (default: `1e-12`).
    /// Entries smaller than this trigger rank-deficiency detection.
    pub min_diagonal: f64,
}

impl Default for SparseQrConfig {
    fn default() -> Self {
        Self {
            column_ordering: true,
            drop_tol: 1e-14,
            min_diagonal: 1e-12,
        }
    }
}

/// A single Givens rotation that zeros out entry at row `row_j` using row `row_i`.
///
/// The rotation satisfies: `G * [a; b] = [r; 0]`, stored as (c, s) with
/// `G = [[c, s], [-s, c]]`.
#[derive(Debug, Clone)]
struct GivensRot {
    /// The pivot row (upper row in the rotation).
    row_i: usize,
    /// The row being zeroed (lower row in the rotation).
    row_j: usize,
    /// Cosine component.
    c: f64,
    /// Sine component.
    s: f64,
}

/// Sparse QR factorization result: `A * P = Q * R`
///
/// - `P` is the COLAMD column permutation (`col_perm`)
/// - `Q` is orthogonal, stored implicitly as a sequence of Givens rotations
/// - `R` is upper triangular, stored in CSC format (`r_factor`)
///
/// Supports solving overdetermined (`m > n`) least squares problems and
/// square (`m == n`) full-rank exact solves.
#[derive(Debug)]
pub struct SparseQr<T: Scalar> {
    /// Upper triangular R factor in CSC format (n × n).
    pub r_factor: CscMatrix<T>,
    /// Column permutation: `col_perm[i]` is the original column index for
    /// the i-th column in the permuted system.
    pub col_perm: Vec<usize>,
    /// Estimated rank of the matrix.
    pub rank: usize,
    /// Diagonal entries of R (length n), used for rank and condition estimation.
    pub r_diag: Vec<T>,
    config: SparseQrConfig,
    m: usize,
    n: usize,
    /// Stored Givens rotations in application order (for computing Q^T * b).
    givens: Vec<GivensRot>,
}

impl SparseQr<f64> {
    /// Compute the sparse QR factorization of matrix `a` (m × n) with default config.
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::RankDeficient`] if the matrix has rank < n.
    /// Returns [`SparseQrError::InvalidMatrix`] if the matrix is empty.
    pub fn compute(a: &CsrMatrix<f64>) -> Result<Self, SparseQrError> {
        Self::compute_with_config(a, SparseQrConfig::default())
    }

    /// Compute the sparse QR factorization of matrix `a` (m × n) with custom config.
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::RankDeficient`] if the matrix has rank < n.
    /// Returns [`SparseQrError::InvalidMatrix`] if the matrix is empty.
    pub fn compute_with_config(
        a: &CsrMatrix<f64>,
        config: SparseQrConfig,
    ) -> Result<Self, SparseQrError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SparseQrError::InvalidMatrix(
                "Matrix must have at least one row and one column".to_string(),
            ));
        }

        // Convert CSR to CSC for COLAMD (which expects a CscMatrix).
        let a_csc = csr_to_csc_f64(a);

        // Compute COLAMD column ordering if requested.
        let col_perm: Vec<usize> = if config.column_ordering {
            colamd(&a_csc)
        } else {
            (0..n).collect()
        };

        // Build inverse permutation: inv_perm[col_perm[i]] = i (permuted index).
        let mut inv_perm = vec![0usize; n];
        for (i, &p) in col_perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        // Build dense working matrix with permuted columns.
        // work[i][j] = A[i, col_perm[j]].
        let mut work = vec![vec![0.0f64; n]; m];
        for i in 0..m {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for idx in start..end {
                let orig_col = a.col_indices()[idx];
                let perm_col = inv_perm[orig_col];
                work[i][perm_col] = a.values()[idx];
            }
        }

        // Left-looking Givens QR: for each column j, zero out all entries below the
        // diagonal in that column using Givens rotations.
        let mut givens: Vec<GivensRot> = Vec::new();

        for j in 0..n {
            for i in (j + 1)..m {
                let a_ij = work[i][j];
                if a_ij.abs() <= config.drop_tol {
                    work[i][j] = 0.0;
                    continue;
                }

                let a_jj = work[j][j];
                // Givens rotation: G * [a_jj; a_ij] = [r; 0]
                let (c, s, r) = givens_rotation_f64(a_jj, a_ij);

                // Apply rotation to all columns k >= j in rows j and i.
                for k in (j + 1)..n {
                    let rjk = work[j][k];
                    let rik = work[i][k];
                    work[j][k] = c * rjk + s * rik;
                    work[i][k] = -s * rjk + c * rik;
                }
                // Update diagonal explicitly (column j is done).
                work[j][j] = r;
                // Enforce structural zero.
                work[i][j] = 0.0;

                givens.push(GivensRot {
                    row_i: j,
                    row_j: i,
                    c,
                    s,
                });
            }
        }

        // Collect diagonal entries and detect rank deficiency.
        let mut rank = 0usize;
        let mut r_diag = Vec::with_capacity(n);
        for j in 0..n {
            let d = work[j][j];
            r_diag.push(d);
            if d.abs() > config.min_diagonal {
                rank += 1;
            }
        }

        if rank < n {
            return Err(SparseQrError::RankDeficient { rank, n });
        }

        // Extract upper triangular n × n block as CSC.
        let r_factor = extract_upper_triangular_csc(&work, n, config.drop_tol);

        Ok(Self {
            r_factor,
            col_perm,
            rank,
            r_diag,
            config,
            m,
            n,
            givens,
        })
    }

    /// Solve the least squares problem: minimize `||A*x - b||` for overdetermined systems.
    ///
    /// Procedure:
    /// 1. Apply all stored Givens rotations to `b`: `b' = G_k * ... * G_1 * b` (= Q^T * b)
    /// 2. Back-substitute with R: `x_perm = R^{-1} * b'[0..n]`
    /// 3. Apply inverse column permutation: `x[col_perm[i]] = x_perm[i]`
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::IncompatibleDimensions`] if `b.len() != m`.
    /// Returns [`SparseQrError::SingularMatrix`] if a zero diagonal is encountered.
    pub fn solve_least_squares(&self, b: &[f64]) -> Result<Vec<f64>, SparseQrError> {
        if b.len() != self.m {
            return Err(SparseQrError::IncompatibleDimensions {
                expected: self.m,
                got: b.len(),
            });
        }

        // Step 1: Apply Givens rotations to b (computes Q^T * b).
        let mut rhs = b.to_vec();
        for g in &self.givens {
            let bi = rhs[g.row_i];
            let bj = rhs[g.row_j];
            rhs[g.row_i] = g.c * bi + g.s * bj;
            rhs[g.row_j] = -g.s * bi + g.c * bj;
        }

        // Step 2: Back-substitute with upper triangular R (CSC format, n × n).
        let mut x_perm = rhs[..self.n].to_vec();

        for j in (0..self.n).rev() {
            let col_start = self.r_factor.col_ptrs()[j];
            let col_end = self.r_factor.col_ptrs()[j + 1];

            // Find diagonal R[j, j].
            let mut rjj = 0.0f64;
            for idx in col_start..col_end {
                if self.r_factor.row_indices()[idx] == j {
                    rjj = self.r_factor.values()[idx];
                    break;
                }
            }

            if rjj.abs() < self.config.min_diagonal {
                return Err(SparseQrError::SingularMatrix);
            }

            x_perm[j] /= rjj;

            // Update x_perm for rows above j.
            for idx in col_start..col_end {
                let row = self.r_factor.row_indices()[idx];
                if row < j {
                    x_perm[row] -= self.r_factor.values()[idx] * x_perm[j];
                }
            }
        }

        // Step 3: Apply inverse column permutation.
        // col_perm[i] = original column index for permuted column i.
        let mut x = vec![0.0f64; self.n];
        for i in 0..self.n {
            x[self.col_perm[i]] = x_perm[i];
        }

        Ok(x)
    }

    /// Solve exactly: `A * x = b` when `m == n` and full rank.
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::IncompatibleDimensions`] if `m != n`.
    /// Returns errors from [`Self::solve_least_squares`].
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SparseQrError> {
        if self.m != self.n {
            return Err(SparseQrError::IncompatibleDimensions {
                expected: self.n,
                got: self.m,
            });
        }
        self.solve_least_squares(b)
    }

    /// Return the estimated rank of the matrix.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Estimate the condition number as `max(|R_diag|) / min(|R_diag|)`.
    ///
    /// Returns `f64::INFINITY` if any diagonal entry is zero or the diag is empty.
    pub fn condition_number_estimate(&self) -> f64 {
        if self.r_diag.is_empty() {
            return f64::INFINITY;
        }
        let max_d = self.r_diag.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
        let min_d = self
            .r_diag
            .iter()
            .map(|d| d.abs())
            .fold(f64::INFINITY, f64::min);
        if min_d == 0.0 {
            f64::INFINITY
        } else {
            max_d / min_d
        }
    }
}

/// Compute a Givens rotation `(c, s, r)` such that `[c s; -s c] * [a; b] = [r; 0]`.
///
/// Uses the standard numerically stable algorithm. The sign of `r` is chosen
/// to be non-negative (conventional for QR factorizations).
fn givens_rotation_f64(a: f64, b: f64) -> (f64, f64, f64) {
    let eps = f64::EPSILON * 4.0;
    if b.abs() <= eps {
        return (1.0, 0.0, a);
    }
    if a.abs() <= eps {
        let sign_b = if b >= 0.0 { 1.0 } else { -1.0 };
        return (0.0, sign_b, b.abs());
    }

    let (c, s, r) = if b.abs() > a.abs() {
        let t = a / b;
        let s = 1.0 / (1.0 + t * t).sqrt();
        let c = s * t;
        let r = b / s;
        (c, s, r)
    } else {
        let t = b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;
        let r = a / c;
        (c, s, r)
    };

    // Ensure r >= 0 by convention.
    if r < 0.0 { (-c, -s, -r) } else { (c, s, r) }
}

/// Convert a `CsrMatrix<f64>` to `CscMatrix<f64>` using a two-pass algorithm.
fn csr_to_csc_f64(a: &CsrMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();

    // First pass: count non-zeros per column.
    let mut col_counts = vec![0usize; n];
    for i in 0..m {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        for idx in start..end {
            col_counts[a.col_indices()[idx]] += 1;
        }
    }

    // Build col_ptrs from counts.
    let mut col_ptrs = vec![0usize; n + 1];
    for j in 0..n {
        col_ptrs[j + 1] = col_ptrs[j] + col_counts[j];
    }
    let nnz = col_ptrs[n];

    // Second pass: fill row_indices and values.
    let mut row_indices = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    let mut fill = vec![0usize; n];

    for i in 0..m {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        for idx in start..end {
            let j = a.col_indices()[idx];
            let pos = col_ptrs[j] + fill[j];
            row_indices[pos] = i;
            values[pos] = a.values()[idx];
            fill[j] += 1;
        }
    }

    // Safety: col_ptrs, row_indices, values are consistent by construction.
    unsafe { CscMatrix::new_unchecked(m, n, col_ptrs, row_indices, values) }
}

/// Extract the upper triangular n × n block of `work` as a `CscMatrix<f64>`.
///
/// Only entries with `|v| > drop_tol` are stored.
fn extract_upper_triangular_csc(work: &[Vec<f64>], n: usize, drop_tol: f64) -> CscMatrix<f64> {
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        // Only rows 0..=j are in the upper triangle.
        for i in 0..=j {
            if i >= work.len() {
                break;
            }
            let v = work[i][j];
            if v.abs() > drop_tol {
                row_indices.push(i);
                values.push(v);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    // Safety: col_ptrs, row_indices, values are consistent by construction.
    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `CsrMatrix<f64>` from (row, col, val) triplets.
    /// Duplicate entries are not supported; each (row, col) must be unique.
    fn build_csr(m: usize, n: usize, triplets: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
        let mut row_counts = vec![0usize; m];
        for &(r, _, _) in triplets {
            row_counts[r] += 1;
        }
        let mut row_ptrs = vec![0usize; m + 1];
        for i in 0..m {
            row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
        }
        let nnz = triplets.len();
        let mut col_indices = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut fill = vec![0usize; m];
        for &(r, c, v) in triplets {
            let pos = row_ptrs[r] + fill[r];
            col_indices[pos] = c;
            values[pos] = v;
            fill[r] += 1;
        }
        CsrMatrix::new(m, n, row_ptrs, col_indices, values).expect("valid CSR matrix")
    }

    /// Compute A*x via CSR SpMV.
    fn spmv_csr(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
        let m = a.nrows();
        let mut y = vec![0.0f64; m];
        for i in 0..m {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for idx in start..end {
                y[i] += a.values()[idx] * x[a.col_indices()[idx]];
            }
        }
        y
    }

    /// L2 norm of a vector.
    fn norm2(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn test_sparse_qr_3x3_identity() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let a = build_csr(3, 3, &triplets);
        let qr = SparseQr::compute(&a).expect("QR of identity should succeed");

        let b = vec![3.0, 5.0, 7.0];
        let x = qr.solve(&b).expect("solve of identity should succeed");

        assert_eq!(x.len(), 3);
        for i in 0..3 {
            assert!(
                (x[i] - b[i]).abs() < 1e-10,
                "identity solve: x[{i}]={} should equal b[{i}]={}",
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_qr_tridiagonal() {
        // 5x5 tridiagonal: main diag = 4, off-diags = -1 (1D discrete Laplacian).
        let mut triplets = Vec::new();
        for i in 0..5usize {
            triplets.push((i, i, 4.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i < 4 {
                triplets.push((i, i + 1, -1.0));
            }
        }
        let a = build_csr(5, 5, &triplets);
        let qr = SparseQr::compute(&a).expect("QR of tridiagonal should succeed");

        let b = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let x = qr.solve(&b).expect("tridiagonal solve should succeed");

        let ax = spmv_csr(&a, &x);
        let residual: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(
            residual < 1e-9,
            "tridiagonal residual {residual} exceeds tolerance"
        );
    }

    #[test]
    fn test_sparse_qr_overdetermined() {
        // 10x5 overdetermined system with b exactly in the column space of A.
        // A[i][j] = 1 / (1 + |i - j|) for i in 0..10, j in 0..5 (entries > 1e-3).
        let mut triplets = Vec::new();
        for i in 0..10usize {
            for j in 0..5usize {
                let v = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                if v > 1e-3 {
                    triplets.push((i, j, v));
                }
            }
        }
        let a = build_csr(10, 5, &triplets);

        let x_true = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let b = spmv_csr(&a, &x_true);

        let qr = SparseQr::compute(&a).expect("overdetermined QR should succeed");
        let x = qr
            .solve_least_squares(&b)
            .expect("overdetermined solve should succeed");

        let ax = spmv_csr(&a, &x);
        let residual = norm2(&(0..10).map(|i| ax[i] - b[i]).collect::<Vec<_>>());
        assert!(
            residual < 1e-8,
            "overdetermined residual {residual} exceeds tolerance"
        );
    }

    #[test]
    fn test_sparse_qr_rank() {
        // Rank-deficient 3x3 matrix: column 2 = column 0 + column 1.
        // A = [1  0  1]
        //     [0  1  1]
        //     [1  1  2]
        let triplets = vec![
            (0, 0, 1.0),
            (0, 2, 1.0),
            (1, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ];
        let a = build_csr(3, 3, &triplets);

        let config = SparseQrConfig {
            column_ordering: true,
            drop_tol: 1e-14,
            min_diagonal: 1e-10,
        };
        let result = SparseQr::compute_with_config(&a, config);
        assert!(
            matches!(result, Err(SparseQrError::RankDeficient { .. })),
            "expected RankDeficient error, got: {result:?}"
        );
        if let Err(SparseQrError::RankDeficient { rank, n }) = result {
            assert!(rank < n, "rank {rank} should be less than n {n}");
            assert_eq!(n, 3);
        }
    }

    #[test]
    fn test_sparse_qr_condition_number() {
        // Well-conditioned 3x3: diagonally dominant.
        let well = build_csr(
            3,
            3,
            &[
                (0, 0, 10.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 10.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 10.0),
            ],
        );
        let qr_well = SparseQr::compute(&well).expect("well-conditioned QR should succeed");
        let cond_well = qr_well.condition_number_estimate();

        // Ill-conditioned 3x3: rows nearly linearly dependent.
        let ill = build_csr(
            3,
            3,
            &[
                (0, 0, 1.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 0, 1.0),
                (1, 1, 1.0 + 1e-7),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
                (2, 2, 1.0 + 2e-7),
            ],
        );
        let qr_ill = SparseQr::compute(&ill).expect("ill-conditioned QR should succeed");
        let cond_ill = qr_ill.condition_number_estimate();

        assert!(
            cond_well >= 1.0,
            "condition number must be >= 1, got {cond_well}"
        );
        assert!(
            cond_ill > cond_well,
            "ill-conditioned matrix should have larger condition number: \
             cond_ill={cond_ill}, cond_well={cond_well}"
        );
    }

    #[test]
    fn test_sparse_qr_exact_solve() {
        // A = [2 1 0; 1 3 1; 0 1 2], b = [5; 10; 5].
        // Verify that QR solve gives A*x == b to floating-point precision.
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ];
        let a = build_csr(3, 3, &triplets);
        let b = vec![5.0, 10.0, 5.0];

        let qr = SparseQr::compute(&a).expect("exact solve QR should succeed");
        let x = qr.solve(&b).expect("exact solve should succeed");

        let ax = spmv_csr(&a, &x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-9,
                "exact solve: A*x[{i}]={} should equal b[{i}]={}",
                ax[i],
                b[i]
            );
        }
    }
}
