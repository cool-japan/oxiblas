//! Tests for sparse eigenvalue solvers.
//!
//! Split by algorithm family (2026-08 hygiene pass) to keep every module
//! under the workspace's 2000-line file limit; `make_symmetric_matrix` and
//! `make_larger_symmetric_matrix` below are shared fixtures used by several
//! of the submodules, so they stay here rather than moving into any one of
//! them.

use crate::csr::CsrMatrix;

fn make_symmetric_matrix() -> CsrMatrix<f64> {
    // A = [4 1 0]
    //     [1 4 1]
    //     [0 1 4]
    let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
    let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
    let row_ptrs = vec![0, 2, 5, 7];

    CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
}

fn make_larger_symmetric_matrix(n: usize) -> CsrMatrix<f64> {
    // Tridiagonal: A[i,i] = 2, A[i,i+1] = A[i+1,i] = -1
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        if i > 0 {
            values.push(-1.0);
            col_indices.push(i - 1);
        }
        values.push(2.0);
        col_indices.push(i);
        if i < n - 1 {
            values.push(-1.0);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

mod block;
mod generalized;
mod interval;
mod iram_hessenberg;
mod iram_restart;
mod lanczos_arnoldi;
mod polynomial_filtered;
mod shift_invert;
