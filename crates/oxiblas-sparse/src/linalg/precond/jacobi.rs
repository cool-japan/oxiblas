//! Jacobi and Block Jacobi preconditioners.

use super::types::PreconditionerError;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Jacobi (diagonal) preconditioner.
///
/// The Jacobi preconditioner is M = diag(A), the diagonal of A.
/// To apply the preconditioner, we solve M z = r, which is simply z_i = r_i / a_ii.
///
/// This is the simplest preconditioner and works well for diagonally dominant matrices.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::Jacobi;
///
/// let jacobi = Jacobi::new(&matrix)?;
/// let mut z = vec![0.0; n];
/// jacobi.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct Jacobi<T: Scalar> {
    /// Inverse of diagonal elements (1 / a_ii)
    diag_inv: Vec<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> Jacobi<T> {
    /// Create a new Jacobi preconditioner from matrix A.
    ///
    /// Extracts the diagonal and computes its inverse.
    ///
    /// # Errors
    ///
    /// Returns error if any diagonal element is zero or matrix is not square.
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        let n = a.nrows();
        let mut diag_inv = vec![T::zero(); n];

        // Extract diagonal elements
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            let mut found = false;
            for k in start..end {
                if a.col_indices()[k] == i {
                    let diag_val = a.values()[k].clone();
                    if Scalar::abs(diag_val.clone()) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                        return Err(PreconditionerError::ZeroDiagonal(i));
                    }
                    diag_inv[i] = T::one() / diag_val;
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(PreconditionerError::ZeroDiagonal(i));
            }
        }

        Ok(Self { diag_inv })
    }

    /// Apply the preconditioner: solve M z = r for z.
    ///
    /// For Jacobi, this is simply z_i = r_i / a_ii.
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        assert_eq!(
            r.len(),
            self.diag_inv.len(),
            "r length must match matrix size"
        );
        assert_eq!(
            z.len(),
            self.diag_inv.len(),
            "z length must match matrix size"
        );

        for i in 0..r.len() {
            z[i] = r[i].clone() * self.diag_inv[i].clone();
        }
    }

    /// Get the number of rows/columns in the preconditioner.
    pub fn size(&self) -> usize {
        self.diag_inv.len()
    }
}

/// Block Jacobi preconditioner.
///
/// Divides the matrix into diagonal blocks and inverts each block.
/// More powerful than point Jacobi but requires more storage and computation.
///
/// The matrix is partitioned as:
/// ```text
/// A = | A11  0   0  |
///     |  0  A22  0  |
///     |  0   0  A33 |
/// ```
///
/// Each block A_ii is inverted and used as a preconditioner.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::BlockJacobi;
///
/// let block_sizes = vec![10, 10, 10]; // Three blocks of size 10
/// let block_jacobi = BlockJacobi::new(&matrix, &block_sizes)?;
/// let mut z = vec![0.0; n];
/// block_jacobi.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct BlockJacobi<T: Scalar> {
    /// Block starting indices
    block_starts: Vec<usize>,
    /// Inverse of each diagonal block (stored densely)
    block_invs: Vec<Vec<T>>,
    /// Size of each block
    block_sizes: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> BlockJacobi<T> {
    /// Create a new Block Jacobi preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to precondition
    /// * `block_sizes` - Size of each diagonal block. Sum must equal matrix size.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Matrix is not square
    /// - Block sizes don't sum to matrix size
    /// - Any block is singular
    pub fn new(a: &CsrMatrix<T>, block_sizes: &[usize]) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        let n = a.nrows();
        let total_size: usize = block_sizes.iter().sum();
        if total_size != n {
            return Err(PreconditionerError::InvalidMatrix(format!(
                "Block sizes sum to {} but matrix size is {}",
                total_size, n
            )));
        }

        let num_blocks = block_sizes.len();
        let mut block_starts = Vec::with_capacity(num_blocks);
        let mut current = 0;
        for &size in block_sizes {
            block_starts.push(current);
            current += size;
        }

        let mut block_invs = Vec::with_capacity(num_blocks);

        // Extract and invert each diagonal block
        for (block_idx, &block_size) in block_sizes.iter().enumerate() {
            let start_row = block_starts[block_idx];
            let end_row = start_row + block_size;

            // Extract block as dense matrix
            let mut block = vec![T::zero(); block_size * block_size];
            for i in start_row..end_row {
                let row_start = a.row_ptrs()[i];
                let row_end = a.row_ptrs()[i + 1];

                for k in row_start..row_end {
                    let j = a.col_indices()[k];
                    if j >= start_row && j < end_row {
                        let local_i = i - start_row;
                        let local_j = j - start_row;
                        block[local_i * block_size + local_j] = a.values()[k].clone();
                    }
                }
            }

            // Invert the block using Gauss-Jordan elimination
            let block_inv = Self::invert_dense(&block, block_size)?;
            block_invs.push(block_inv);
        }

        Ok(Self {
            block_starts,
            block_invs,
            block_sizes: block_sizes.to_vec(),
        })
    }

    /// Invert a dense matrix using Gauss-Jordan elimination.
    fn invert_dense(matrix: &[T], n: usize) -> Result<Vec<T>, PreconditionerError> {
        // Create augmented matrix [A | I]
        let mut aug = vec![T::zero(); n * (2 * n)];

        // Copy A to left side
        for i in 0..n {
            for j in 0..n {
                aug[i * (2 * n) + j] = matrix[i * n + j].clone();
            }
            // Set identity on right side
            aug[i * (2 * n) + n + i] = T::one();
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            let mut max_val = Scalar::abs(aug[i * (2 * n) + i].clone());
            for k in (i + 1)..n {
                let val = Scalar::abs(aug[k * (2 * n) + i].clone());
                if val > max_val {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if max_val < T::from_f64(1e-14).unwrap_or(T::zero()) {
                return Err(PreconditionerError::SingularBlock(i));
            }

            // Swap rows if needed
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = aug[i * (2 * n) + j].clone();
                    aug[i * (2 * n) + j] = aug[pivot_row * (2 * n) + j].clone();
                    aug[pivot_row * (2 * n) + j] = temp;
                }
            }

            // Scale pivot row
            let pivot = aug[i * (2 * n) + i].clone();
            for j in 0..(2 * n) {
                aug[i * (2 * n) + j] = aug[i * (2 * n) + j].clone() / pivot.clone();
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug[k * (2 * n) + i].clone();
                    for j in 0..(2 * n) {
                        aug[k * (2 * n) + j] = aug[k * (2 * n) + j].clone()
                            - factor.clone() * aug[i * (2 * n) + j].clone();
                    }
                }
            }
        }

        // Extract inverse from right side
        let mut inv = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                inv[i * n + j] = aug[i * (2 * n) + n + j].clone();
            }
        }

        Ok(inv)
    }

    /// Apply the preconditioner: solve M z = r for z.
    ///
    /// For Block Jacobi, solves each block independently:
    /// z_i = A_ii^{-1} r_i
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        let n: usize = self.block_sizes.iter().sum();
        assert_eq!(r.len(), n, "r length must match matrix size");
        assert_eq!(z.len(), n, "z length must match matrix size");

        for (block_idx, &block_size) in self.block_sizes.iter().enumerate() {
            let start = self.block_starts[block_idx];
            let block_inv = &self.block_invs[block_idx];

            // z[start:start+block_size] = block_inv * r[start:start+block_size]
            for i in 0..block_size {
                let mut sum = T::zero();
                for j in 0..block_size {
                    sum = sum + block_inv[i * block_size + j].clone() * r[start + j].clone();
                }
                z[start + i] = sum;
            }
        }
    }

    /// Get the number of rows/columns in the preconditioner.
    pub fn size(&self) -> usize {
        self.block_sizes.iter().sum()
    }

    /// Get the number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_sizes.len()
    }
}
