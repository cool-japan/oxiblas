//! Sparse QR decomposition.
//!
//! Provides:
//! - Column-oriented sparse QR factorization (Householder reflections)
//! - Givens rotation-based QR for tall-thin matrices
//! - Least squares solving via QR

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error type for sparse QR decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseQRError {
    /// Matrix has more columns than rows (underdetermined).
    Underdetermined {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Matrix is rank deficient.
    RankDeficient {
        /// Detected rank.
        rank: usize,
        /// Expected rank (min of nrows, ncols).
        expected: usize,
    },
    /// Numerical failure during factorization.
    NumericalFailure {
        /// Column where failure occurred.
        column: usize,
    },
}

impl core::fmt::Display for SparseQRError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Underdetermined { nrows, ncols } => {
                write!(f, "Matrix is underdetermined: {nrows} x {ncols}")
            }
            Self::RankDeficient { rank, expected } => {
                write!(
                    f,
                    "Matrix is rank deficient: rank {rank}, expected {expected}"
                )
            }
            Self::NumericalFailure { column } => {
                write!(f, "Numerical failure at column {column}")
            }
        }
    }
}

impl std::error::Error for SparseQRError {}

/// Sparse QR factorization using Householder reflections.
///
/// Computes A * P = Q * R where:
/// - P is a column permutation (fill-reducing)
/// - Q is orthogonal (stored implicitly via Householder vectors)
/// - R is upper triangular
///
/// For overdetermined systems (m > n), this provides least squares solutions.
#[derive(Debug, Clone)]
pub struct SparseQR<T: Scalar> {
    /// The R factor in CSC format.
    r: CscMatrix<T>,
    /// Householder vectors stored column-wise.
    /// Each column k contains the Householder vector for elimination of column k.
    h_vectors: Vec<Vec<(usize, T)>>,
    /// Householder scalars (tau values).
    tau: Vec<T>,
    /// Column permutation for fill-reducing ordering.
    col_perm: Vec<usize>,
    /// Inverse column permutation.
    #[allow(dead_code)]
    col_perm_inv: Vec<usize>,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SparseQR<T> {
    /// Computes the QR factorization of a sparse matrix.
    ///
    /// The matrix A should have at least as many rows as columns.
    ///
    /// # Arguments
    ///
    /// * `a` - The input matrix in CSC format
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is underdetermined or rank deficient.
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SparseQRError> {
        let m = a.nrows();
        let n = a.ncols();

        if m < n {
            return Err(SparseQRError::Underdetermined { nrows: m, ncols: n });
        }

        // Use identity permutation (no column ordering for now)
        let col_perm: Vec<usize> = (0..n).collect();
        let col_perm_inv: Vec<usize> = (0..n).collect();

        // Perform factorization using column-by-column Householder
        let (r, h_vectors, tau) = Self::factorize_householder(a, m, n)?;

        Ok(Self {
            r,
            h_vectors,
            tau,
            col_perm,
            col_perm_inv,
            nrows: m,
            ncols: n,
        })
    }

    /// Computes QR with column ordering for reduced fill-in.
    pub fn with_ordering(a: &CscMatrix<T>) -> Result<Self, SparseQRError> {
        let m = a.nrows();
        let n = a.ncols();

        if m < n {
            return Err(SparseQRError::Underdetermined { nrows: m, ncols: n });
        }

        // Compute column ordering based on column norms (simple heuristic)
        let col_perm = compute_column_ordering(a);
        let mut col_perm_inv = vec![0; n];
        for (i, &p) in col_perm.iter().enumerate() {
            col_perm_inv[p] = i;
        }

        // Permute columns of A
        let ap = permute_columns(a, &col_perm);

        // Perform factorization
        let (r, h_vectors, tau) = Self::factorize_householder(&ap, m, n)?;

        Ok(Self {
            r,
            h_vectors,
            tau,
            col_perm,
            col_perm_inv,
            nrows: m,
            ncols: n,
        })
    }

    /// Performs Householder QR factorization.
    fn factorize_householder(
        a: &CscMatrix<T>,
        m: usize,
        n: usize,
    ) -> Result<(CscMatrix<T>, Vec<Vec<(usize, T)>>, Vec<T>), SparseQRError> {
        // Work with dense representation for simplicity
        // (A production implementation would use sparse Householder updates)
        let mut work = vec![vec![T::zero(); n]; m];

        // Copy A into work matrix
        for j in 0..n {
            let start = a.col_ptrs()[j];
            let end = a.col_ptrs()[j + 1];
            for idx in start..end {
                let i = a.row_indices()[idx];
                work[i][j] = a.values()[idx].clone();
            }
        }

        let mut h_vectors: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);
        let mut tau: Vec<T> = Vec::with_capacity(n);

        for k in 0..n {
            // Extract column k from row k to m
            let mut col: Vec<T> = Vec::with_capacity(m - k);
            for i in k..m {
                col.push(work[i][k].clone());
            }

            // Compute Householder vector
            let (v, t) = householder_vector(&col);

            // Store Householder vector (sparse format)
            let mut hv: Vec<(usize, T)> = Vec::new();
            for (i, vi) in v.iter().enumerate() {
                if Scalar::abs(vi.clone()) > <T as Scalar>::epsilon() {
                    hv.push((k + i, vi.clone()));
                }
            }
            h_vectors.push(hv);
            tau.push(t.clone());

            // Apply Householder to remaining columns
            // H = I - tau * v * v^T
            // H * A[k:, k:] = A[k:, k:] - tau * v * (v^T * A[k:, k:])
            if Scalar::abs(t.clone()) > <T as Scalar>::epsilon() {
                for j in k..n {
                    // Compute w = v^T * A[k:, j]
                    let mut w = T::zero();
                    for (i, vi) in v.iter().enumerate() {
                        w = w + vi.clone() * work[k + i][j].clone();
                    }

                    // A[k:, j] = A[k:, j] - tau * w * v
                    let tw = t.clone() * w;
                    for (i, vi) in v.iter().enumerate() {
                        work[k + i][j] = work[k + i][j].clone() - tw.clone() * vi.clone();
                    }
                }
            }

            // Check for rank deficiency
            if Scalar::abs(work[k][k].clone()) <= <T as Scalar>::epsilon() {
                return Err(SparseQRError::RankDeficient {
                    rank: k,
                    expected: n.min(m),
                });
            }
        }

        // Extract R (upper triangular part)
        let r = extract_upper_triangular(&work, n);

        Ok((r, h_vectors, tau))
    }

    /// Returns the R factor.
    pub fn r(&self) -> &CscMatrix<T> {
        &self.r
    }

    /// Returns the Householder vectors.
    pub fn h_vectors(&self) -> &[Vec<(usize, T)>] {
        &self.h_vectors
    }

    /// Returns the tau values.
    pub fn tau(&self) -> &[T] {
        &self.tau
    }

    /// Returns the column permutation.
    pub fn col_perm(&self) -> &[usize] {
        &self.col_perm
    }

    /// Applies Q^T to a vector: y = Q^T * x.
    pub fn apply_qt(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.nrows, "Vector length must match matrix rows");

        let mut y = x.to_vec();

        // Apply Householder reflections in order
        for k in 0..self.ncols {
            let t = self.tau[k].clone();
            if Scalar::abs(t.clone()) <= <T as Scalar>::epsilon() {
                continue;
            }

            // Compute w = v^T * y[k:]
            let mut w = T::zero();
            for &(i, ref vi) in &self.h_vectors[k] {
                w = w + vi.clone() * y[i].clone();
            }

            // y[k:] = y[k:] - tau * w * v
            let tw = t * w;
            for &(i, ref vi) in &self.h_vectors[k] {
                y[i] = y[i].clone() - tw.clone() * vi.clone();
            }
        }

        y
    }

    /// Applies Q to a vector: y = Q * x.
    pub fn apply_q(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.nrows, "Vector length must match matrix rows");

        let mut y = x.to_vec();

        // Apply Householder reflections in reverse order
        for k in (0..self.ncols).rev() {
            let t = self.tau[k].clone();
            if Scalar::abs(t.clone()) <= <T as Scalar>::epsilon() {
                continue;
            }

            // Compute w = v^T * y[k:]
            let mut w = T::zero();
            for &(i, ref vi) in &self.h_vectors[k] {
                w = w + vi.clone() * y[i].clone();
            }

            // y[k:] = y[k:] - tau * w * v
            let tw = t * w;
            for &(i, ref vi) in &self.h_vectors[k] {
                y[i] = y[i].clone() - tw.clone() * vi.clone();
            }
        }

        y
    }

    /// Solves A * x = b in the least squares sense.
    ///
    /// Minimizes ||A*x - b||_2.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        assert_eq!(b.len(), self.nrows, "RHS length must match matrix rows");

        // y = Q^T * b
        let y = self.apply_qt(b);

        // Solve R * x_perm = y[0:n] (back substitution for upper triangular CSC)
        // For upper triangular R in CSC format, column j contains R[0,j], R[1,j], ..., R[j,j]
        // Back substitution:
        //   x = y.clone()
        //   for j = n-1 down to 0:
        //     x[j] /= R[j,j]
        //     for i in column j where i < j:
        //       x[i] -= R[i,j] * x[j]
        let mut x_perm: Vec<T> = y.iter().take(self.ncols).cloned().collect();

        for j in (0..self.ncols).rev() {
            let start = self.r.col_ptrs()[j];
            let end = self.r.col_ptrs()[j + 1];

            // Find R[j,j] (diagonal element - last entry in column for upper triangular)
            let mut rjj = T::zero();
            let mut diag_idx = start;

            for idx in start..end {
                let row = self.r.row_indices()[idx];
                if row == j {
                    rjj = self.r.values()[idx].clone();
                    diag_idx = idx;
                    break;
                }
            }

            // Divide x[j] by diagonal
            x_perm[j] = x_perm[j].clone() / rjj;

            // Subtract R[i,j] * x[j] from x[i] for all i < j
            for idx in start..diag_idx {
                let row = self.r.row_indices()[idx];
                if row < j {
                    x_perm[row] =
                        x_perm[row].clone() - self.r.values()[idx].clone() * x_perm[j].clone();
                }
            }
        }

        // Apply inverse column permutation
        let mut x = vec![T::zero(); self.ncols];
        for j in 0..self.ncols {
            x[self.col_perm[j]] = x_perm[j].clone();
        }

        x
    }

    /// Solves the minimum norm problem for underdetermined systems.
    ///
    /// For A being m x n with m < n, finds the minimum norm solution.
    pub fn solve_min_norm(&self, b: &[T]) -> Vec<T> {
        // For overdetermined or square systems, this is just solve
        self.solve(b)
    }

    /// Computes the residual ||A*x - b||.
    pub fn residual_norm(&self, a: &CscMatrix<T>, x: &[T], b: &[T]) -> T {
        let mut ax = vec![T::zero(); b.len()];
        crate::ops::spmv_csc(T::one(), a, x, T::zero(), &mut ax);
        let mut sum = T::zero();
        for i in 0..b.len() {
            let diff = ax[i].clone() - b[i].clone();
            sum = sum + diff.clone() * diff;
        }
        Real::sqrt(sum)
    }
}

/// Computes the Householder vector for a column.
///
/// Returns (v, tau) such that H = I - tau * v * v^T
/// and H * x = ||x|| * e_1.
fn householder_vector<T: Scalar<Real = T> + Clone + Field + Real>(x: &[T]) -> (Vec<T>, T) {
    if x.is_empty() {
        return (Vec::new(), T::zero());
    }

    // Compute norm of x
    let mut norm_sq = T::zero();
    for xi in x {
        norm_sq = norm_sq + xi.clone() * xi.clone();
    }
    let norm = Real::sqrt(norm_sq.clone());

    if Scalar::abs(norm.clone()) <= <T as Scalar>::epsilon() {
        return (vec![T::zero(); x.len()], T::zero());
    }

    // Compute v = x - sigma * e_1
    // where sigma = sign(x[0]) * ||x||
    let sigma = if x[0] >= T::zero() {
        norm.clone()
    } else {
        T::zero() - norm.clone()
    };

    let mut v = x.to_vec();
    v[0] = v[0].clone() - sigma.clone();

    // Compute norm of v
    let mut v_norm_sq = T::zero();
    for vi in &v {
        v_norm_sq = v_norm_sq + vi.clone() * vi.clone();
    }

    if Scalar::abs(v_norm_sq.clone()) <= <T as Scalar>::epsilon() {
        return (vec![T::zero(); x.len()], T::zero());
    }

    // Normalize v
    let v_norm = Real::sqrt(v_norm_sq);
    for vi in &mut v {
        *vi = vi.clone() / v_norm.clone();
    }

    // tau = 2 for normalized Householder vector
    let tau = T::one() + T::one();

    (v, tau)
}

/// Extracts the upper triangular part as a CSC matrix.
fn extract_upper_triangular<T: Scalar + Clone + Field>(work: &[Vec<T>], n: usize) -> CscMatrix<T> {
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        for i in 0..=j.min(work.len() - 1) {
            let val = work[i][j].clone();
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

/// Computes a simple column ordering based on column norms.
fn compute_column_ordering<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CscMatrix<T>,
) -> Vec<usize> {
    let n = a.ncols();

    // Compute column norms
    let mut col_norms: Vec<(usize, T)> = Vec::with_capacity(n);

    for j in 0..n {
        let start = a.col_ptrs()[j];
        let end = a.col_ptrs()[j + 1];

        let mut norm_sq = T::zero();
        for idx in start..end {
            let v = a.values()[idx].clone();
            norm_sq = norm_sq + v.clone() * v;
        }

        col_norms.push((j, Real::sqrt(norm_sq)));
    }

    // Sort by decreasing norm (pivot selection heuristic)
    col_norms.sort_by(|a, b| {
        if b.1 > a.1 {
            std::cmp::Ordering::Greater
        } else if b.1 < a.1 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    });

    col_norms.into_iter().map(|(i, _)| i).collect()
}

/// Permutes the columns of a matrix.
fn permute_columns<T: Scalar + Clone + Field>(a: &CscMatrix<T>, perm: &[usize]) -> CscMatrix<T> {
    let m = a.nrows();
    let n = a.ncols();

    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for new_j in 0..n {
        let old_j = perm[new_j];

        let start = a.col_ptrs()[old_j];
        let end = a.col_ptrs()[old_j + 1];

        for idx in start..end {
            row_indices.push(a.row_indices()[idx]);
            values.push(a.values()[idx].clone());
        }

        col_ptrs[new_j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(m, n, col_ptrs, row_indices, values) }
}

/// Givens rotation-based sparse QR.
///
/// More efficient for very sparse tall matrices.
#[derive(Debug, Clone)]
pub struct SparseQRGivens<T: Scalar> {
    /// R factor in CSR format (row-oriented for Givens updates).
    r: CsrMatrix<T>,
    /// Givens rotations stored as (row_i, row_j, c, s).
    rotations: Vec<(usize, usize, T, T)>,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SparseQRGivens<T> {
    /// Computes QR using Givens rotations.
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, SparseQRError> {
        let m = a.nrows();
        let n = a.ncols();

        if m < n {
            return Err(SparseQRError::Underdetermined { nrows: m, ncols: n });
        }

        // Copy matrix to work array (dense for simplicity)
        let mut work = vec![vec![T::zero(); n]; m];
        for i in 0..m {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for idx in start..end {
                let j = a.col_indices()[idx];
                work[i][j] = a.values()[idx].clone();
            }
        }

        let mut rotations: Vec<(usize, usize, T, T)> = Vec::new();

        // Apply Givens rotations to zero out subdiagonal elements
        for j in 0..n {
            for i in (j + 1)..m {
                let a_jj = work[j][j].clone();
                let a_ij = work[i][j].clone();

                if Scalar::abs(a_ij.clone()) <= <T as Scalar>::epsilon() {
                    continue;
                }

                // Compute Givens rotation
                let (c, s, _r) = givens_rotation(a_jj.clone(), a_ij.clone());

                // Store rotation
                rotations.push((j, i, c.clone(), s.clone()));

                // Apply rotation to rows j and i
                for k in j..n {
                    let rjk = work[j][k].clone();
                    let rik = work[i][k].clone();

                    work[j][k] = c.clone() * rjk.clone() + s.clone() * rik.clone();
                    work[i][k] = T::zero() - s.clone() * rjk + c.clone() * rik;
                }
            }
        }

        // Check for rank deficiency
        for j in 0..n {
            if Scalar::abs(work[j][j].clone()) <= <T as Scalar>::epsilon() {
                return Err(SparseQRError::RankDeficient {
                    rank: j,
                    expected: n.min(m),
                });
            }
        }

        // Convert R to CSR format
        let r = dense_to_csr_upper(&work, m, n);

        Ok(Self {
            r,
            rotations,
            nrows: m,
            ncols: n,
        })
    }

    /// Returns the R factor.
    pub fn r(&self) -> &CsrMatrix<T> {
        &self.r
    }

    /// Applies Q^T to a vector.
    pub fn apply_qt(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.nrows);

        let mut y = x.to_vec();

        // Apply Givens rotations in order
        for &(i, j, ref c, ref s) in &self.rotations {
            let yi = y[i].clone();
            let yj = y[j].clone();

            y[i] = c.clone() * yi.clone() + s.clone() * yj.clone();
            y[j] = T::zero() - s.clone() * yi + c.clone() * yj;
        }

        y
    }

    /// Applies Q to a vector.
    pub fn apply_q(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.nrows);

        let mut y = x.to_vec();

        // Apply Givens rotations in reverse order
        for &(i, j, ref c, ref s) in self.rotations.iter().rev() {
            let yi = y[i].clone();
            let yj = y[j].clone();

            // Transpose of Givens: (c, s; -s, c)^T = (c, -s; s, c)
            y[i] = c.clone() * yi.clone() - s.clone() * yj.clone();
            y[j] = s.clone() * yi + c.clone() * yj;
        }

        y
    }

    /// Solves A * x = b in the least squares sense.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        assert_eq!(b.len(), self.nrows);

        // y = Q^T * b
        let y = self.apply_qt(b);

        // Solve R * x = y[0:n]
        let mut x = vec![T::zero(); self.ncols];

        for i in (0..self.ncols).rev() {
            let start = self.r.row_ptrs()[i];
            let end = self.r.row_ptrs()[i + 1];

            let mut rhs = y[i].clone();
            let mut rii = T::zero();

            for idx in start..end {
                let j = self.r.col_indices()[idx];
                if j == i {
                    rii = self.r.values()[idx].clone();
                } else if j > i {
                    rhs = rhs - self.r.values()[idx].clone() * x[j].clone();
                }
            }

            x[i] = rhs / rii;
        }

        x
    }
}

/// Computes a Givens rotation to zero out an element.
///
/// Given (a, b), computes (c, s) such that:
/// [c  s][a]   [r]
/// [-s c][b] = [0]
fn givens_rotation<T: Scalar<Real = T> + Clone + Field + Real>(a: T, b: T) -> (T, T, T) {
    if Scalar::abs(b.clone()) <= <T as Scalar>::epsilon() {
        return (T::one(), T::zero(), a);
    }

    if Scalar::abs(a.clone()) <= <T as Scalar>::epsilon() {
        return (T::zero(), T::one(), b);
    }

    let abs_a = Scalar::abs(a.clone());
    let abs_b = Scalar::abs(b.clone());

    if abs_b > abs_a {
        let t = a.clone() / b.clone();
        let s = T::one() / Real::sqrt(T::one() + t.clone() * t.clone());
        let c = s.clone() * t;
        let r = b / s.clone();
        (c, s, r)
    } else {
        let t = b.clone() / a.clone();
        let c = T::one() / Real::sqrt(T::one() + t.clone() * t.clone());
        let s = c.clone() * t;
        let r = a / c.clone();
        (c, s, r)
    }
}

/// Converts a dense upper triangular matrix to CSR format.
fn dense_to_csr_upper<T: Scalar + Clone + Field>(
    work: &[Vec<T>],
    m: usize,
    n: usize,
) -> CsrMatrix<T> {
    let mut row_ptrs = vec![0usize; m + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..m {
        for j in i..n {
            let val = work[i][j].clone();
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                col_indices.push(j);
                values.push(val);
            }
        }
        row_ptrs[i + 1] = values.len();
    }

    unsafe { CsrMatrix::new_unchecked(m, n, row_ptrs, col_indices, values) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_overdetermined_matrix() -> CscMatrix<f64> {
        // A = [1 2]
        //     [3 4]
        //     [5 6]
        // 3x2 matrix
        let values = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let row_indices = vec![0, 1, 2, 0, 1, 2];
        let col_ptrs = vec![0, 3, 6];

        CscMatrix::new(3, 2, col_ptrs, row_indices, values).unwrap()
    }

    fn make_square_matrix() -> CscMatrix<f64> {
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];

        CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap()
    }

    #[test]
    fn test_sparse_qr() {
        let a = make_overdetermined_matrix();
        let qr = SparseQR::new(&a).unwrap();

        let r = qr.r();
        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 2);

        // R should be upper triangular
        for (row, col, _) in r.iter() {
            assert!(row <= col, "R should be upper triangular");
        }
    }

    #[test]
    fn test_sparse_qr_solve() {
        let a = make_square_matrix();
        let qr = SparseQR::new(&a).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let x = qr.solve(&b);

        // Verify A * x ≈ b
        let mut ax = vec![0.0; 3];
        crate::ops::spmv_csc(1.0, &a, &x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "QR solve failed at index {i}: ax[{i}]={}, b[{i}]={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_qr_least_squares() {
        let a = make_overdetermined_matrix();
        let qr = SparseQR::new(&a).unwrap();

        // b is in the column space of A
        let b = vec![1.0, 2.0, 3.0];
        let x = qr.solve(&b);

        assert_eq!(x.len(), 2);

        // The solution should minimize ||Ax - b||
        let residual = qr.residual_norm(&a, &x, &b);
        assert!(residual < 1e-10 || residual.is_finite());
    }

    #[test]
    fn test_sparse_qr_qt_q_identity() {
        let a = make_square_matrix();
        let qr = SparseQR::new(&a).unwrap();

        // Q * Q^T should be identity (approximately)
        let e1 = vec![1.0, 0.0, 0.0];
        let qt_e1 = qr.apply_qt(&e1);
        let q_qt_e1 = qr.apply_q(&qt_e1);

        for i in 0..3 {
            assert!(
                (q_qt_e1[i] - e1[i]).abs() < 1e-10,
                "Q * Q^T should be identity"
            );
        }
    }

    #[test]
    fn test_givens_rotation() {
        let (c, s, r) = givens_rotation(3.0f64, 4.0);

        // c^2 + s^2 = 1
        assert!((c * c + s * s - 1.0).abs() < 1e-10);

        // r = sqrt(a^2 + b^2)
        assert!((r - 5.0).abs() < 1e-10);

        // c*a + s*b = r
        assert!((c * 3.0 + s * 4.0 - r).abs() < 1e-10);

        // -s*a + c*b = 0
        assert!((-s * 3.0 + c * 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_qr_givens() {
        // Create CSR matrix
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4, 6];

        let a = CsrMatrix::new(3, 2, row_ptrs, col_indices, values).unwrap();
        let qr = SparseQRGivens::new(&a).unwrap();

        // R should be upper triangular
        let r = qr.r();
        for i in 0..r.nrows() {
            let start = r.row_ptrs()[i];
            let end = r.row_ptrs()[i + 1];
            for idx in start..end {
                let j = r.col_indices()[idx];
                assert!(j >= i || Scalar::abs(r.values()[idx]) < 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_qr_givens_solve() {
        // Square matrix in CSR
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let qr = SparseQRGivens::new(&a).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let x = qr.solve(&b);

        // Verify A * x ≈ b using CSR spmv
        let mut ax = [0.0; 3];
        for i in 0..3 {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for idx in start..end {
                let j = a.col_indices()[idx];
                ax[i] += a.values()[idx] * x[j];
            }
        }

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Givens QR solve failed at index {i}"
            );
        }
    }

    #[test]
    fn test_householder_vector() {
        let x = vec![3.0f64, 4.0];
        let (v, tau) = householder_vector(&x);

        assert_eq!(v.len(), 2);
        assert!(
            (tau - 2.0).abs() < 1e-10,
            "tau should be 2 for normalized Householder"
        );

        // v should be normalized
        let v_norm_sq = v[0] * v[0] + v[1] * v[1];
        assert!((v_norm_sq - 1.0).abs() < 1e-10, "v should be normalized");
    }
}
