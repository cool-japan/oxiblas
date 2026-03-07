//! Sparse matrix integration with ndarray types.
//!
//! This module provides conversion functions between ndarray dense matrices
//! and OxiBLAS sparse matrix formats (CSR, CSC), plus sparse linear algebra
//! operations that accept and return ndarray types.
//!
//! # Features
//!
//! - **Conversions**: Dense (Array2) to/from CSR and CSC sparse formats
//! - **SpMV**: Sparse matrix-vector multiplication returning Array1
//! - **Sparse Solve**: Solve sparse linear systems using CG, returning Array1
//!
//! # Example
//!
//! ```ignore
//! use ndarray::array;
//! use oxiblas_ndarray::sparse::{array2_to_csr, csr_to_array2, spmv_ndarray};
//!
//! let dense = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
//! let csr = array2_to_csr(&dense);
//! assert_eq!(csr.nnz(), 5);
//!
//! let x = array![1.0, 1.0, 1.0];
//! let y = spmv_ndarray(&csr, &x);
//! ```

use ndarray::{Array1, Array2};
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_sparse::csc::CscMatrix;
use oxiblas_sparse::csr::CsrMatrix;

// =============================================================================
// Dense to Sparse Conversions
// =============================================================================

/// Converts a dense Array2 to CSR (Compressed Sparse Row) format.
///
/// Non-zero elements are identified by comparing their absolute value
/// against the scalar epsilon threshold.
///
/// # Arguments
/// * `arr` - Dense 2D array
///
/// # Returns
/// CSR matrix containing only the non-zero elements
///
/// # Example
/// ```ignore
/// use ndarray::array;
/// use oxiblas_ndarray::sparse::array2_to_csr;
///
/// let a = array![[1.0, 0.0], [0.0, 2.0]];
/// let csr = array2_to_csr(&a);
/// assert_eq!(csr.nnz(), 2);
/// ```
pub fn array2_to_csr<T: Scalar + Clone + Field>(arr: &Array2<T>) -> CsrMatrix<T> {
    let (nrows, ncols) = arr.dim();
    let eps = <T as Scalar>::epsilon();

    let mut row_ptrs = Vec::with_capacity(nrows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0);

    for i in 0..nrows {
        for j in 0..ncols {
            let val = arr[[i, j]];
            if Scalar::abs(val) > eps {
                col_indices.push(j);
                values.push(val);
            }
        }
        row_ptrs.push(values.len());
    }

    // Safety: we construct valid CSR arrays by design:
    // - row_ptrs has length nrows + 1
    // - row_ptrs is monotonically increasing
    // - all col_indices are < ncols (from the loop bounds)
    // - values.len() == col_indices.len()
    unsafe { CsrMatrix::new_unchecked(nrows, ncols, row_ptrs, col_indices, values) }
}

/// Converts a CSR matrix back to a dense Array2.
///
/// # Arguments
/// * `csr` - Sparse CSR matrix
///
/// # Returns
/// Dense 2D array with all elements (including zeros)
pub fn csr_to_array2<T: Scalar + Clone + Field>(csr: &CsrMatrix<T>) -> Array2<T> {
    let (nrows, ncols) = csr.shape();
    let mut result = Array2::zeros((nrows, ncols));

    for i in 0..nrows {
        for (col, val) in csr.row_iter(i) {
            result[[i, col]] = *val;
        }
    }

    result
}

/// Converts a dense Array2 to CSC (Compressed Sparse Column) format.
///
/// Non-zero elements are identified by comparing their absolute value
/// against the scalar epsilon threshold.
///
/// # Arguments
/// * `arr` - Dense 2D array
///
/// # Returns
/// CSC matrix containing only the non-zero elements
pub fn array2_to_csc<T: Scalar + Clone + Field>(arr: &Array2<T>) -> CscMatrix<T> {
    let (nrows, ncols) = arr.dim();
    let eps = <T as Scalar>::epsilon();

    let mut col_ptrs = Vec::with_capacity(ncols + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    col_ptrs.push(0);

    for j in 0..ncols {
        for i in 0..nrows {
            let val = arr[[i, j]];
            if Scalar::abs(val) > eps {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs.push(values.len());
    }

    // Safety: we construct valid CSC arrays by design
    unsafe { CscMatrix::new_unchecked(nrows, ncols, col_ptrs, row_indices, values) }
}

/// Converts a CSC matrix back to a dense Array2.
///
/// # Arguments
/// * `csc` - Sparse CSC matrix
///
/// # Returns
/// Dense 2D array with all elements (including zeros)
pub fn csc_to_array2<T: Scalar + Clone + Field>(csc: &CscMatrix<T>) -> Array2<T> {
    let (nrows, ncols) = csc.shape();
    let mut result = Array2::zeros((nrows, ncols));

    for j in 0..ncols {
        for (row, val) in csc.col_iter(j) {
            result[[row, j]] = *val;
        }
    }

    result
}

// =============================================================================
// Sparse Matrix-Vector Multiplication
// =============================================================================

/// Sparse matrix-vector multiplication: y = A * x
///
/// Computes the product of a CSR sparse matrix with a dense vector,
/// returning a new dense Array1.
///
/// # Arguments
/// * `a` - Sparse CSR matrix (m x n)
/// * `x` - Dense input vector (length n)
///
/// # Returns
/// Dense output vector (length m)
///
/// # Panics
/// Panics if the vector length does not match the number of matrix columns.
pub fn spmv_ndarray<T: Scalar + Clone + Field>(a: &CsrMatrix<T>, x: &Array1<T>) -> Array1<T> {
    assert_eq!(
        x.len(),
        a.ncols(),
        "Vector length {} must match matrix columns {}",
        x.len(),
        a.ncols()
    );

    let x_vec: Vec<T> = x.iter().cloned().collect();
    let mut y_vec = vec![T::zero(); a.nrows()];

    oxiblas_sparse::ops::spmv(T::one(), a, &x_vec, T::zero(), &mut y_vec);

    Array1::from_vec(y_vec)
}

/// Sparse matrix-vector multiplication with scaling: y = alpha * A * x + beta * y
///
/// General form of SpMV that supports scaling factors.
///
/// # Arguments
/// * `alpha` - Scalar multiplier for A * x
/// * `a` - Sparse CSR matrix (m x n)
/// * `x` - Dense input vector (length n)
/// * `beta` - Scalar multiplier for existing y
/// * `y` - Dense output vector (length m), modified in place
///
/// # Panics
/// Panics if dimensions do not match.
pub fn spmv_full_ndarray<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CsrMatrix<T>,
    x: &Array1<T>,
    beta: T,
    y: &mut Array1<T>,
) {
    assert_eq!(x.len(), a.ncols(), "x length must match matrix columns");
    assert_eq!(y.len(), a.nrows(), "y length must match matrix rows");

    let x_vec: Vec<T> = x.iter().cloned().collect();

    if let Some(y_slice) = y.as_slice_mut() {
        oxiblas_sparse::ops::spmv(alpha, a, &x_vec, beta, y_slice);
    } else {
        let mut y_vec: Vec<T> = y.iter().cloned().collect();
        oxiblas_sparse::ops::spmv(alpha, a, &x_vec, beta, &mut y_vec);
        for (yi, val) in y.iter_mut().zip(y_vec.into_iter()) {
            *yi = val;
        }
    }
}

// =============================================================================
// Sparse Linear Solve
// =============================================================================

/// Error type for sparse ndarray operations.
#[derive(Debug, Clone)]
pub enum SparseNdarrayError {
    /// The matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Dimension mismatch between matrix and vector.
    DimensionMismatch {
        /// Matrix dimension.
        matrix_dim: usize,
        /// Vector length.
        vector_len: usize,
    },
    /// The iterative solver did not converge.
    NotConverged {
        /// Number of iterations performed.
        iterations: usize,
        /// Final residual norm.
        residual_norm: f64,
    },
    /// Solver encountered an error.
    SolverError(String),
}

impl core::fmt::Display for SparseNdarrayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix must be square: got {nrows}x{ncols}")
            }
            Self::DimensionMismatch {
                matrix_dim,
                vector_len,
            } => {
                write!(
                    f,
                    "Dimension mismatch: matrix dim={matrix_dim}, vector len={vector_len}"
                )
            }
            Self::NotConverged {
                iterations,
                residual_norm,
            } => {
                write!(
                    f,
                    "CG did not converge after {iterations} iterations (residual={residual_norm})"
                )
            }
            Self::SolverError(msg) => write!(f, "Solver error: {msg}"),
        }
    }
}

impl std::error::Error for SparseNdarrayError {}

/// Solve a sparse linear system A * x = b using Conjugate Gradient.
///
/// This function solves the system using the CG iterative method, which
/// requires A to be symmetric positive definite (SPD). For non-SPD matrices,
/// consider using other solvers.
///
/// # Arguments
/// * `a` - Sparse CSR matrix (n x n), must be SPD
/// * `b` - Right-hand side vector (length n)
///
/// # Returns
/// Solution vector x, or an error if the solver fails
///
/// # Errors
/// Returns `SparseNdarrayError` if:
/// - Matrix is not square
/// - Dimensions don't match
/// - CG solver does not converge
pub fn sparse_solve_ndarray(
    a: &CsrMatrix<f64>,
    b: &Array1<f64>,
) -> Result<Array1<f64>, SparseNdarrayError> {
    let (nrows, ncols) = a.shape();

    if nrows != ncols {
        return Err(SparseNdarrayError::NotSquare { nrows, ncols });
    }

    if b.len() != nrows {
        return Err(SparseNdarrayError::DimensionMismatch {
            matrix_dim: nrows,
            vector_len: b.len(),
        });
    }

    let b_vec: Vec<f64> = b.iter().copied().collect();
    let x0 = vec![0.0f64; nrows];

    let tol = 1e-10;
    let max_iter = nrows * 2 + 100;

    match oxiblas_sparse::linalg::cg(a, &b_vec, &x0, tol, max_iter) {
        Ok(result) => {
            if result.converged {
                Ok(Array1::from_vec(result.x))
            } else {
                Err(SparseNdarrayError::NotConverged {
                    iterations: result.iterations,
                    residual_norm: result.residual_norm,
                })
            }
        }
        Err(e) => Err(SparseNdarrayError::SolverError(e.to_string())),
    }
}

/// Solve a sparse linear system with custom tolerance and max iterations.
///
/// # Arguments
/// * `a` - Sparse CSR matrix (n x n), must be SPD
/// * `b` - Right-hand side vector (length n)
/// * `tol` - Convergence tolerance (relative to norm of b)
/// * `max_iter` - Maximum number of CG iterations
///
/// # Returns
/// Solution vector x, or an error if the solver fails
///
/// # Errors
/// Returns `SparseNdarrayError` if:
/// - Matrix is not square
/// - Dimensions don't match
/// - CG solver does not converge within max_iter
pub fn sparse_solve_ndarray_with_options(
    a: &CsrMatrix<f64>,
    b: &Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Array1<f64>, SparseNdarrayError> {
    let (nrows, ncols) = a.shape();

    if nrows != ncols {
        return Err(SparseNdarrayError::NotSquare { nrows, ncols });
    }

    if b.len() != nrows {
        return Err(SparseNdarrayError::DimensionMismatch {
            matrix_dim: nrows,
            vector_len: b.len(),
        });
    }

    let b_vec: Vec<f64> = b.iter().copied().collect();
    let x0 = vec![0.0f64; nrows];

    match oxiblas_sparse::linalg::cg(a, &b_vec, &x0, tol, max_iter) {
        Ok(result) => {
            if result.converged {
                Ok(Array1::from_vec(result.x))
            } else {
                Err(SparseNdarrayError::NotConverged {
                    iterations: result.iterations,
                    residual_norm: result.residual_norm,
                })
            }
        }
        Err(e) => Err(SparseNdarrayError::SolverError(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // =========================================================================
    // Conversion Tests
    // =========================================================================

    #[test]
    fn test_array2_to_csr_basic() {
        let a = array![[1.0f64, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let csr = array2_to_csr(&a);

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 5);

        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(0, 2), Some(&2.0));
        assert_eq!(csr.get(1, 1), Some(&3.0));
        assert_eq!(csr.get(2, 0), Some(&4.0));
        assert_eq!(csr.get(2, 2), Some(&5.0));

        // Zero elements
        assert_eq!(csr.get(0, 1), None);
        assert_eq!(csr.get(1, 0), None);
    }

    #[test]
    fn test_csr_to_array2_basic() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];
        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values)
            .expect("Failed to create CSR matrix");

        let arr = csr_to_array2(&csr);
        assert_eq!(arr.dim(), (3, 3));
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((arr[[0, 1]]).abs() < 1e-15);
        assert!((arr[[0, 2]] - 2.0).abs() < 1e-15);
        assert!((arr[[1, 1]] - 3.0).abs() < 1e-15);
        assert!((arr[[2, 0]] - 4.0).abs() < 1e-15);
        assert!((arr[[2, 2]] - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_roundtrip_csr() {
        let original = array![
            [1.0f64, 0.0, 3.0, 0.0],
            [0.0, 5.0, 0.0, 7.0],
            [9.0, 0.0, 11.0, 0.0]
        ];

        let csr = array2_to_csr(&original);
        let recovered = csr_to_array2(&csr);

        assert_eq!(original.dim(), recovered.dim());
        for i in 0..3 {
            for j in 0..4 {
                assert!(
                    (original[[i, j]] - recovered[[i, j]]).abs() < 1e-15,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_array2_to_csc_basic() {
        let a = array![[1.0f64, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let csc = array2_to_csc(&a);

        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 3);
        assert_eq!(csc.nnz(), 5);

        assert_eq!(csc.get(0, 0), Some(&1.0));
        assert_eq!(csc.get(0, 2), Some(&2.0));
        assert_eq!(csc.get(1, 1), Some(&3.0));
        assert_eq!(csc.get(2, 0), Some(&4.0));
        assert_eq!(csc.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_csc_to_array2_basic() {
        let values = vec![1.0f64, 4.0, 3.0, 2.0, 5.0];
        let row_indices = vec![0, 2, 1, 0, 2];
        let col_ptrs = vec![0, 2, 3, 5];
        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values)
            .expect("Failed to create CSC matrix");

        let arr = csc_to_array2(&csc);
        assert_eq!(arr.dim(), (3, 3));
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((arr[[2, 0]] - 4.0).abs() < 1e-15);
        assert!((arr[[1, 1]] - 3.0).abs() < 1e-15);
        assert!((arr[[0, 2]] - 2.0).abs() < 1e-15);
        assert!((arr[[2, 2]] - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_roundtrip_csc() {
        let original = array![
            [0.0f64, 2.0, 0.0],
            [4.0, 0.0, 6.0],
            [0.0, 8.0, 0.0],
            [10.0, 0.0, 12.0]
        ];

        let csc = array2_to_csc(&original);
        let recovered = csc_to_array2(&csc);

        assert_eq!(original.dim(), recovered.dim());
        for i in 0..4 {
            for j in 0..3 {
                assert!(
                    (original[[i, j]] - recovered[[i, j]]).abs() < 1e-15,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_empty_matrix_csr() {
        let a: Array2<f64> = Array2::zeros((3, 4));
        let csr = array2_to_csr(&a);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.shape(), (3, 4));

        let recovered = csr_to_array2(&csr);
        for i in 0..3 {
            for j in 0..4 {
                assert!(recovered[[i, j]].abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_dense_matrix_csr() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let csr = array2_to_csr(&a);
        assert_eq!(csr.nnz(), 4);
    }

    #[test]
    fn test_identity_csr() {
        let n = 5;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 1.0;
        }

        let csr = array2_to_csr(&a);
        assert_eq!(csr.nnz(), n);

        for i in 0..n {
            assert_eq!(csr.get(i, i), Some(&1.0));
        }
    }

    #[test]
    fn test_f32_conversions() {
        let a = array![[1.0f32, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let csr = array2_to_csr(&a);
        assert_eq!(csr.nnz(), 3);

        let recovered = csr_to_array2(&csr);
        assert!((recovered[[0, 0]] - 1.0f32).abs() < 1e-6);
        assert!((recovered[[0, 2]] - 2.0f32).abs() < 1e-6);
        assert!((recovered[[1, 1]] - 3.0f32).abs() < 1e-6);
    }

    // =========================================================================
    // SpMV Tests
    // =========================================================================

    #[test]
    fn test_spmv_ndarray_basic() {
        let a = array![[1.0f64, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let csr = array2_to_csr(&a);
        let x = array![1.0f64, 1.0, 1.0];

        let y = spmv_ndarray(&csr, &x);

        // y[0] = 1*1 + 0*1 + 2*1 = 3
        // y[1] = 0*1 + 3*1 + 0*1 = 3
        // y[2] = 4*1 + 0*1 + 5*1 = 9
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
        assert!((y[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_spmv_ndarray_identity() {
        let n = 10;
        let csr: CsrMatrix<f64> = CsrMatrix::eye(n);
        let x = Array1::from_shape_fn(n, |i| (i + 1) as f64);

        let y = spmv_ndarray(&csr, &x);

        for i in 0..n {
            assert!((y[i] - x[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_spmv_full_ndarray() {
        let a = array![[2.0f64, 0.0], [0.0, 3.0]];
        let csr = array2_to_csr(&a);
        let x = array![1.0f64, 2.0];
        let mut y = array![10.0f64, 20.0];

        // y = 2.0 * A * x + 0.5 * y
        // y[0] = 2.0 * (2*1 + 0*2) + 0.5 * 10 = 4 + 5 = 9
        // y[1] = 2.0 * (0*1 + 3*2) + 0.5 * 20 = 12 + 10 = 22
        spmv_full_ndarray(2.0, &csr, &x, 0.5, &mut y);

        assert!((y[0] - 9.0).abs() < 1e-10);
        assert!((y[1] - 22.0).abs() < 1e-10);
    }

    // =========================================================================
    // Sparse Solve Tests
    // =========================================================================

    #[test]
    fn test_sparse_solve_identity() {
        let n = 5;
        let csr: CsrMatrix<f64> = CsrMatrix::eye(n);
        let b = Array1::from_shape_fn(n, |i| (i + 1) as f64);

        let x = sparse_solve_ndarray(&csr, &b).expect("Solve should succeed for identity");

        for i in 0..n {
            assert!(
                (x[i] - b[i]).abs() < 1e-8,
                "Mismatch at {}: got {}, expected {}",
                i,
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_solve_spd() {
        // SPD tridiagonal: [4 -1 0; -1 4 -1; 0 -1 4]
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values)
            .expect("Failed to create CSR matrix");

        let b = array![3.0f64, 2.0, 3.0];
        let x = sparse_solve_ndarray(&csr, &b).expect("Solve should succeed for SPD matrix");

        // Verify A * x = b
        let residual = spmv_ndarray(&csr, &x);
        for i in 0..3 {
            assert!(
                (residual[i] - b[i]).abs() < 1e-8,
                "Residual mismatch at {}: got {}, expected {}",
                i,
                residual[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_solve_larger_spd() {
        let n = 20;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0f64);
                col_indices.push(i - 1);
            }
            values.push(4.0f64);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0f64);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        let csr = CsrMatrix::new(n, n, row_ptrs, col_indices, values)
            .expect("Failed to create CSR matrix");

        let b = Array1::from_shape_fn(n, |i| (i + 1) as f64);
        let x = sparse_solve_ndarray(&csr, &b).expect("Solve should succeed for larger SPD");

        let residual = spmv_ndarray(&csr, &x);
        for i in 0..n {
            assert!(
                (residual[i] - b[i]).abs() < 1e-6,
                "Residual mismatch at {}: got {}, expected {}",
                i,
                residual[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_solve_not_square() {
        let csr: CsrMatrix<f64> = CsrMatrix::zeros(3, 4);
        let b = array![1.0f64, 2.0, 3.0];
        let result = sparse_solve_ndarray(&csr, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_solve_dimension_mismatch() {
        let csr: CsrMatrix<f64> = CsrMatrix::eye(3);
        let b = array![1.0f64, 2.0]; // Wrong length
        let result = sparse_solve_ndarray(&csr, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_solve_with_options() {
        let csr: CsrMatrix<f64> = CsrMatrix::eye(3);
        let b = array![1.0f64, 2.0, 3.0];

        let x = sparse_solve_ndarray_with_options(&csr, &b, 1e-12, 100)
            .expect("Solve with options should succeed");

        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-10);
        }
    }
}
