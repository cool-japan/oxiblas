//! Sparse triangular solvers.
//!
//! Provides forward and backward substitution for sparse triangular matrices.

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Solves L * x = b for lower triangular L in CSC format.
///
/// Uses forward substitution.
pub fn solve_lower_csc<T: Scalar + Clone + Field>(l: &CscMatrix<T>, b: &[T]) -> Vec<T> {
    let n = l.nrows();
    assert_eq!(l.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    let mut x = b.to_vec();

    for j in 0..n {
        let col_start = l.col_ptrs()[j];
        let col_end = l.col_ptrs()[j + 1];

        if col_start == col_end {
            continue;
        }

        // First entry should be diagonal (L[j,j])
        let diag_idx = col_start;
        let diag = l.values()[diag_idx].clone();

        // x[j] = x[j] / L[j,j]
        x[j] = x[j].clone() / diag;

        let xj = x[j].clone();

        // Update x[i] for i > j
        for idx in (col_start + 1)..col_end {
            let i = l.row_indices()[idx];
            x[i] = x[i].clone() - l.values()[idx].clone() * xj.clone();
        }
    }

    x
}

/// Solves L^T * x = b for lower triangular L in CSC format.
///
/// Uses backward substitution on the transpose.
pub fn solve_lower_transpose_csc<T: Scalar + Clone + Field>(l: &CscMatrix<T>, b: &[T]) -> Vec<T> {
    let n = l.nrows();
    assert_eq!(l.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    let mut x = b.to_vec();

    for j in (0..n).rev() {
        let col_start = l.col_ptrs()[j];
        let col_end = l.col_ptrs()[j + 1];

        if col_start == col_end {
            continue;
        }

        // Accumulate contributions from entries below diagonal
        for idx in (col_start + 1)..col_end {
            let i = l.row_indices()[idx];
            x[j] = x[j].clone() - l.values()[idx].clone() * x[i].clone();
        }

        // Divide by diagonal (first entry in column)
        let diag = l.values()[col_start].clone();
        x[j] = x[j].clone() / diag;
    }

    x
}

/// Solves U * x = b for upper triangular U in CSC format.
///
/// Uses backward substitution.
pub fn solve_upper_csc<T: Scalar + Clone + Field>(u: &CscMatrix<T>, b: &[T]) -> Vec<T> {
    let n = u.nrows();
    assert_eq!(u.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    let mut x = b.to_vec();

    for j in (0..n).rev() {
        let col_start = u.col_ptrs()[j];
        let col_end = u.col_ptrs()[j + 1];

        if col_start == col_end {
            continue;
        }

        // Find diagonal (last entry with row == j)
        let mut diag = T::one();
        let mut diag_idx = col_end;

        for idx in col_start..col_end {
            if u.row_indices()[idx] == j {
                diag = u.values()[idx].clone();
                diag_idx = idx;
                break;
            }
        }

        // x[j] = x[j] / U[j,j]
        x[j] = x[j].clone() / diag;

        let xj = x[j].clone();

        // Update x[i] for i < j
        for idx in col_start..diag_idx {
            let i = u.row_indices()[idx];
            if i < j {
                x[i] = x[i].clone() - u.values()[idx].clone() * xj.clone();
            }
        }
    }

    x
}

/// Solves U^T * x = b for upper triangular U in CSC format.
///
/// Uses forward substitution on the transpose.
pub fn solve_upper_transpose_csc<T: Scalar + Clone + Field>(u: &CscMatrix<T>, b: &[T]) -> Vec<T> {
    let n = u.nrows();
    assert_eq!(u.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    let mut x = b.to_vec();

    for j in 0..n {
        let col_start = u.col_ptrs()[j];
        let col_end = u.col_ptrs()[j + 1];

        if col_start == col_end {
            continue;
        }

        // Find diagonal and accumulate contributions from entries above
        let mut diag = T::one();

        for idx in col_start..col_end {
            let i = u.row_indices()[idx];
            if i == j {
                diag = u.values()[idx].clone();
            } else if i < j {
                x[j] = x[j].clone() - u.values()[idx].clone() * x[i].clone();
            }
        }

        x[j] = x[j].clone() / diag;
    }

    x
}

/// Solves L * x = b for lower triangular L in CSR format.
pub fn solve_lower<T: Scalar + Clone + Field>(l: &CsrMatrix<T>, b: &[T]) -> Vec<T> {
    let n = l.nrows();
    assert_eq!(l.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    let mut x = b.to_vec();

    for i in 0..n {
        let row_start = l.row_ptrs()[i];
        let row_end = l.row_ptrs()[i + 1];

        // Accumulate L[i, 0..i] * x[0..i]
        let mut sum = T::zero();
        let mut diag = T::one();

        for idx in row_start..row_end {
            let j = l.col_indices()[idx];
            if j < i {
                sum = sum + l.values()[idx].clone() * x[j].clone();
            } else if j == i {
                diag = l.values()[idx].clone();
            }
        }

        x[i] = (x[i].clone() - sum) / diag;
    }

    x
}

/// Solves U * x = b for upper triangular U in CSR format.
pub fn solve_upper<T: Scalar + Clone + Field>(u: &CsrMatrix<T>, b: &[T]) -> Vec<T> {
    let n = u.nrows();
    assert_eq!(u.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    let mut x = b.to_vec();

    for i in (0..n).rev() {
        let row_start = u.row_ptrs()[i];
        let row_end = u.row_ptrs()[i + 1];

        // Accumulate U[i, i+1..n] * x[i+1..n]
        let mut sum = T::zero();
        let mut diag = T::one();

        for idx in row_start..row_end {
            let j = u.col_indices()[idx];
            if j > i {
                sum = sum + u.values()[idx].clone() * x[j].clone();
            } else if j == i {
                diag = u.values()[idx].clone();
            }
        }

        x[i] = (x[i].clone() - sum) / diag;
    }

    x
}

/// Solves L * L^T * x = b (for Cholesky factor L) in CSC format.
pub fn solve_cholesky<T: Scalar + Clone + Field>(l: &CscMatrix<T>, b: &[T]) -> Vec<T> {
    // Forward solve: L * y = b
    let y = solve_lower_csc(l, b);

    // Backward solve: L^T * x = y
    solve_lower_transpose_csc(l, &y)
}

/// Solves (D + L) * D^{-1} * (D + U) * x = b for ILU(0).
///
/// Where L is strict lower triangular, U is strict upper triangular, and D is diagonal.
/// The matrix is stored as L + D + U in CSR format.
pub fn solve_ilu0<T: Scalar + Clone + Field>(lu: &CsrMatrix<T>, b: &[T]) -> Vec<T> {
    let n = lu.nrows();
    assert_eq!(lu.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    // Extract diagonal
    let mut diag = vec![T::one(); n];
    for i in 0..n {
        let row_start = lu.row_ptrs()[i];
        let row_end = lu.row_ptrs()[i + 1];

        for idx in row_start..row_end {
            if lu.col_indices()[idx] == i {
                diag[i] = lu.values()[idx].clone();
                break;
            }
        }
    }

    // Forward solve: (D + L) * y = b
    let mut y = b.to_vec();
    for i in 0..n {
        let row_start = lu.row_ptrs()[i];
        let row_end = lu.row_ptrs()[i + 1];

        let mut sum = T::zero();
        for idx in row_start..row_end {
            let j = lu.col_indices()[idx];
            if j < i {
                sum = sum + lu.values()[idx].clone() * y[j].clone();
            }
        }

        y[i] = (y[i].clone() - sum) / diag[i].clone();
    }

    // Backward solve: (D + U) * x = D * y
    let mut x = vec![T::zero(); n];
    for i in 0..n {
        x[i] = diag[i].clone() * y[i].clone();
    }

    for i in (0..n).rev() {
        let row_start = lu.row_ptrs()[i];
        let row_end = lu.row_ptrs()[i + 1];

        let mut sum = T::zero();
        for idx in row_start..row_end {
            let j = lu.col_indices()[idx];
            if j > i {
                sum = sum + lu.values()[idx].clone() * x[j].clone();
            }
        }

        x[i] = (x[i].clone() - sum) / diag[i].clone();
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_lower_csc() {
        // L = [2 0 0]
        //     [1 3 0]
        //     [2 1 4]
        let values = vec![2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 2, 1, 2, 2];
        let col_ptrs = vec![0, 3, 5, 6];

        let l = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();
        let b = vec![2.0, 5.0, 14.0];

        let x = solve_lower_csc(&l, &b);

        // Verify L * x = b
        // x = [1, 4/3, 11/4] approximately
        assert!((x[0] - 1.0).abs() < 1e-10);

        // Manual verification
        let mut lx = [0.0; 3];
        for col in 0..3 {
            let start = l.col_ptrs()[col];
            let end = l.col_ptrs()[col + 1];
            for idx in start..end {
                lx[l.row_indices()[idx]] += l.values()[idx] * x[col];
            }
        }

        for i in 0..3 {
            assert!((lx[i] - b[i]).abs() < 1e-10, "L*x != b at index {i}");
        }
    }

    #[test]
    fn test_solve_lower_csr() {
        // L = [2 0 0]
        //     [1 3 0]
        //     [2 1 4]
        let values = vec![2.0f64, 1.0, 3.0, 2.0, 1.0, 4.0];
        let col_indices = vec![0, 0, 1, 0, 1, 2];
        let row_ptrs = vec![0, 1, 3, 6];

        let l = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let b = vec![2.0, 5.0, 14.0];

        let x = solve_lower(&l, &b);

        // Verify by computing L * x
        let mut lx = [0.0; 3];
        for row in 0..3 {
            let start = l.row_ptrs()[row];
            let end = l.row_ptrs()[row + 1];
            for idx in start..end {
                lx[row] += l.values()[idx] * x[l.col_indices()[idx]];
            }
        }

        for i in 0..3 {
            assert!((lx[i] - b[i]).abs() < 1e-10, "L*x != b at index {i}");
        }
    }

    #[test]
    fn test_solve_upper_csr() {
        // U = [2 1 2]
        //     [0 3 1]
        //     [0 0 4]
        let values = vec![2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 2, 1, 2, 2];
        let row_ptrs = vec![0, 3, 5, 6];

        let u = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let b = vec![7.0, 7.0, 4.0];

        let x = solve_upper(&u, &b);

        // Verify by computing U * x
        let mut ux = [0.0; 3];
        for row in 0..3 {
            let start = u.row_ptrs()[row];
            let end = u.row_ptrs()[row + 1];
            for idx in start..end {
                ux[row] += u.values()[idx] * x[u.col_indices()[idx]];
            }
        }

        for i in 0..3 {
            assert!((ux[i] - b[i]).abs() < 1e-10, "U*x != b at index {i}");
        }
    }

    #[test]
    fn test_solve_identity() {
        let l = CscMatrix::<f64>::eye(5);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let x = solve_lower_csc(&l, &b);

        for i in 0..5 {
            assert!((x[i] - b[i]).abs() < 1e-10);
        }
    }
}
