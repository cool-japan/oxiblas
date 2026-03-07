//! Utility functions for eigenvalue computations.
//!
//! This module provides helper functions for vector and matrix operations
//! used by the eigenvalue solvers.

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

// =============================================================================
// Vector Operations
// =============================================================================

/// Compute Givens rotation parameters.
///
/// Returns (c, s, r) such that:
/// [c  s][a] = [r]
/// [-s c][b]   [0]
pub fn givens_rotation<T: Scalar<Real = T> + Clone + Field + Real>(a: T, b: T) -> (T, T, T) {
    if Scalar::abs(b.clone()) <= <T as Scalar>::epsilon() {
        return (T::one(), T::zero(), a);
    }

    if Scalar::abs(a.clone()) <= <T as Scalar>::epsilon() {
        return (
            T::zero(),
            if b >= T::zero() {
                T::one()
            } else {
                T::zero() - T::one()
            },
            Scalar::abs(b),
        );
    }

    let r = Real::sqrt(a.clone() * a.clone() + b.clone() * b.clone());
    let c = a / r.clone();
    let s = b / r.clone();

    (c, s, r)
}

/// Computes the dot product of two vectors.
pub fn dot<T: Scalar + Clone + Field>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i].clone() * b[i].clone();
    }
    sum
}

/// Computes the 2-norm of a vector.
pub fn norm<T: Scalar<Real = T> + Clone + Field + Real>(v: &[T]) -> T {
    Real::sqrt(dot(v, v))
}

// =============================================================================
// Matrix Operations
// =============================================================================

/// Compute C = A - sigma * B (sparse matrix subtraction).
pub fn subtract_scaled_matrices<T: Scalar + Clone>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    sigma: T,
) -> CsrMatrix<T> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n);
    assert_eq!(b.nrows(), n);
    assert_eq!(b.ncols(), n);

    // Use symbolic addition to find sparsity pattern
    let mut row_ptrs = vec![0usize; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        let b_start = b.row_ptrs()[i];
        let b_end = b.row_ptrs()[i + 1];

        // Merge sorted column indices
        let mut a_ptr = a_start;
        let mut b_ptr = b_start;

        while a_ptr < a_end || b_ptr < b_end {
            let a_col = if a_ptr < a_end {
                Some(a.col_indices()[a_ptr])
            } else {
                None
            };
            let b_col = if b_ptr < b_end {
                Some(b.col_indices()[b_ptr])
            } else {
                None
            };

            match (a_col, b_col) {
                (Some(ac), Some(bc)) if ac < bc => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (Some(ac), Some(bc)) if ac > bc => {
                    col_indices.push(bc);
                    values.push(T::zero() - sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (Some(ac), Some(_bc)) => {
                    // ac == bc
                    let val = a.values()[a_ptr].clone() - sigma.clone() * b.values()[b_ptr].clone();
                    if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                        col_indices.push(ac);
                        values.push(val);
                    }
                    a_ptr += 1;
                    b_ptr += 1;
                }
                (Some(ac), None) => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (None, Some(bc)) => {
                    col_indices.push(bc);
                    values.push(T::zero() - sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (None, None) => break,
            }
        }

        row_ptrs[i + 1] = col_indices.len();
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values)
        .expect("CSR matrix construction with valid parameters")
}

/// Compute C = A + sigma * B (sparse matrix addition).
pub fn add_scaled_matrices<T: Scalar + Clone>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    sigma: T,
) -> CsrMatrix<T> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n);
    assert_eq!(b.nrows(), n);
    assert_eq!(b.ncols(), n);

    let mut row_ptrs = vec![0usize; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        let b_start = b.row_ptrs()[i];
        let b_end = b.row_ptrs()[i + 1];

        let mut a_ptr = a_start;
        let mut b_ptr = b_start;

        while a_ptr < a_end || b_ptr < b_end {
            let a_col = if a_ptr < a_end {
                Some(a.col_indices()[a_ptr])
            } else {
                None
            };
            let b_col = if b_ptr < b_end {
                Some(b.col_indices()[b_ptr])
            } else {
                None
            };

            match (a_col, b_col) {
                (Some(ac), Some(bc)) if ac < bc => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (Some(ac), Some(bc)) if ac > bc => {
                    col_indices.push(bc);
                    values.push(sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (Some(ac), Some(_bc)) => {
                    let val = a.values()[a_ptr].clone() + sigma.clone() * b.values()[b_ptr].clone();
                    if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                        col_indices.push(ac);
                        values.push(val);
                    }
                    a_ptr += 1;
                    b_ptr += 1;
                }
                (Some(ac), None) => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (None, Some(bc)) => {
                    col_indices.push(bc);
                    values.push(sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (None, None) => break,
            }
        }

        row_ptrs[i + 1] = col_indices.len();
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values)
        .expect("CSR matrix construction with valid parameters")
}

/// Convert CSR matrix to CSC format.
pub fn csr_to_csc<T: Scalar + Clone>(csr: &CsrMatrix<T>) -> CscMatrix<T> {
    let nrows = csr.nrows();
    let ncols = csr.ncols();
    let nnz = csr.nnz();

    if nnz == 0 {
        return CscMatrix::new(nrows, ncols, vec![0; ncols + 1], vec![], vec![])
            .expect("CSC matrix construction with valid parameters");
    }

    // Count entries per column
    let mut col_counts = vec![0usize; ncols];
    for &col in csr.col_indices() {
        col_counts[col] += 1;
    }

    // Build column pointers
    let mut col_ptrs = vec![0usize; ncols + 1];
    for j in 0..ncols {
        col_ptrs[j + 1] = col_ptrs[j] + col_counts[j];
    }

    // Fill row indices and values
    let mut row_indices = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut col_pos = col_ptrs[..ncols].to_vec();

    for i in 0..nrows {
        let row_start = csr.row_ptrs()[i];
        let row_end = csr.row_ptrs()[i + 1];
        for k in row_start..row_end {
            let j = csr.col_indices()[k];
            let pos = col_pos[j];
            row_indices[pos] = i;
            values[pos] = csr.values()[k].clone();
            col_pos[j] += 1;
        }
    }

    CscMatrix::new(nrows, ncols, col_ptrs, row_indices, values)
        .expect("CSC matrix construction with valid parameters")
}
